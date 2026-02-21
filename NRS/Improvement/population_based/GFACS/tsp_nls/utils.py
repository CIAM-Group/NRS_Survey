import os
import pickle

import torch
from torch_geometric.data import Data

import re

# ! 新增
# 1) 解析 TSPLIB：返回 name / dim / coords / edge_weight_type
def parse_tsplib_file(path):
    """
    仅支持 EUC_2D / CEIL_2D（与 ICAM 保持一致）。
    返回: name(str), dim(int), coords(torch.FloatTensor[n,2]), edge_type(str)
    """
    name, dim, edge_type = None, None, None
    coords_list = []
    started = False
    with open(path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # header
            if line.startswith("NAME"):
                # 允许 NAME: xxx 或 NAME xxx
                parts = line.replace(":", " ").split()
                name = parts[-1]
            elif line.startswith("DIMENSION"):
                parts = line.replace(":", " ").split()
                dim = int(parts[-1])
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                parts = line.replace(":", " ").split()
                t = parts[-1]
                if t not in ["EUC_2D", "CEIL_2D"]:
                    # 与 ICAM 保持一致：遇到不支持的类型直接丢弃
                    return None, None, None, None
                edge_type = t
            elif line.startswith("NODE_COORD_SECTION"):
                started = True
                continue
            elif line.startswith("EOF"):
                break
            # coords
            if started:
                parts = line.split()
                # TSPLIB 常见是：index x y
                if len(parts) >= 3:
                    x = float(parts[-2])
                    y = float(parts[-1])
                    coords_list.append([x, y])

    if name is None or dim is None or edge_type is None:
        return None, None, None, None
    assert len(coords_list) == dim, f"{path} 坐标行数与 DIMENSION 不一致"

    coords = torch.tensor(coords_list, dtype=torch.float32)
    return name, dim, coords, edge_type

# ! 新增
# 2) 生成 “TSPLIB 整数口径” 的距离矩阵（EUC_2D=四舍五入，CEIL_2D=向上取整）
def tsplib_integer_distance(coords: torch.Tensor, edge_type: str) -> torch.Tensor:
    """
    coords: [n,2] float
    return: [n,n] float(整数值)，对角线置 1e9 以屏蔽自环
    """
    n = coords.size(0)
    # 欧式距离（浮点）
    diff = coords[:, None, :] - coords[None, :, :]
    dist = torch.linalg.norm(diff, dim=2)  # [n,n] float

    if edge_type == "EUC_2D":
        # 四舍五入到最近整数（TSPLIB规范）
        dist = torch.floor(dist + 0.5)
    elif edge_type == "CEIL_2D":
        # 向上取整
        dist = torch.ceil(dist)
    else:
        raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE: {edge_type}")

    # 避免自环
    dist[torch.arange(n), torch.arange(n)] = 1e9
    return dist

# ! 新增
# 3) KNN 构图（建议基于原始欧式距离做 KNN，搜索/计分用整数矩阵）
def build_pyg_from_coords(coords: torch.Tensor, k_sparse: int, start_node=None):
    """
    返回 PyG Data（x, edge_index, edge_attr） + 欧式浮点距离（供 KNN）
    """
    n = coords.size(0)

    # ! 强制放到 CPU 做全矩阵，避免 CUDA OOM
    # coords_cpu = coords.detach().to('cpu')
    # 欧式浮点距离用于 KNN
    diff = coords[:, None, :] - coords[None, :, :]
    # diff = coords_cpu[:, None, :] - coords_cpu[None, :, :]  # [n,n,2] on CPU
    euc = torch.linalg.norm(diff, dim=2)
    # 屏蔽自环
    euc[torch.arange(n), torch.arange(n)] = 1e9

    k = min(k_sparse, n - 1) if n > 1 else 0
    topk_values, topk_indices = torch.topk(euc, k=k, dim=1, largest=False)

    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(n, device=coords.device), repeats=k),
        topk_indices.reshape(-1)
    ], dim=0)
    edge_attr = topk_values.reshape(-1, 1)

    # 节点特征：保持你现有逻辑
    if start_node is None:
        node_feature = coords  # [n,2]
    else:
        node_feature = torch.zeros((n, 1), device=coords.device, dtype=coords.dtype)
        node_feature[start_node, 0] = 1.0
    # if start_node is None:
    #     node_feature = coords_cpu  # [n,2]（CPU）
    # else:
    #     node_feature = torch.zeros((n,1), dtype=coords_cpu.dtype)
    #     node_feature[start_node, 0] = 1.0

    pyg_data = Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data, euc  # 返回欧式浮点距离(如果你还有其他用途)

# ! 新增
# 4) 主加载函数：像 ICAM 一样从目录读取，不再依赖 n_nodes → 区间映射
def load_tsplib_from_dir(
    root_dir: str,
    k_sparse_factor: int,
    device: str,
    start_node: int = None,
    dim_ranges=None,
):
    """
    root_dir: TSPLIB 根目录（会递归查 .tsp）
    k_sparse_factor: k = n // k_sparse_factor
    device: 'cpu' 或 'cuda:0'
    start_node: 同你现有逻辑
    dim_ranges: 可选过滤，如 [[100,300], [300,700]]；None 表示不过滤

    返回:
      test_list: [ (pyg_data, distances_int), ... ]
      name_list: [ name, ... ]
    """
    test_list = []
    name_list = []

    for root, _, files in os.walk(root_dir):
        # ! 筛出所有 .tsp 文件，用字典序排序，保证顺序稳定
        files = sorted([f for f in files if f.lower().endswith(".tsp")])
        
        for fn in files:
            if not fn.lower().endswith(".tsp"):
                continue
            full = os.path.join(root, fn)
            name, dim, coords, edge_type = parse_tsplib_file(full)
            if name is None:
                # 不支持的类型/解析失败
                continue

            # 维度过滤（按需）
            if dim_ranges is not None:
                ok = any((lo <= dim < hi) for (lo, hi) in dim_ranges)
                if not ok:
                    continue

            coords = coords.to(device)
            # 稀疏构图
            k_sparse = max(1, dim // k_sparse_factor) if dim > 1 else 0
            pyg_data, _ = build_pyg_from_coords(coords, k_sparse=k_sparse, start_node=start_node)
            # 整数口径距离矩阵（用于 ACO/计分），放 CPU（跟你 ACO 一样）
            dist_int = tsplib_integer_distance(coords, edge_type).to('cpu')

            test_list.append((pyg_data, dist_int))
            name_list.append(name)

    if len(test_list) == 0:
        raise FileNotFoundError(f"在目录 {root_dir} 下没有找到可解析的 .tsp 文件，或都不在筛选维度范围内。")

    return test_list, name_list

# ! 新增：不计算举例矩阵的读取
# ========== 新增：一次性读取所有实例的“轻量句柄” ==========
def load_tsplib_handles(root_dir: str, dim_ranges=None):
    """
    返回轻量实例句柄列表：[{name, dim, coords, edge_type}, ...]
    仅解析 .tsp（coords 是 [n,2] float32，内存很小），不构图、不建距离矩阵。
    """
    handles = []
    for root, _, files in os.walk(root_dir):
        files = sorted([f for f in files if f.lower().endswith(".tsp")])
        for fn in files:
            full = os.path.join(root, fn)
            name, dim, coords, edge_type = parse_tsplib_file(full)
            if name is None:
                continue
            if dim_ranges is not None:
                ok = any((lo <= dim < hi) for (lo, hi) in dim_ranges)
                if not ok:
                    continue
            handles.append({
                "name": name,
                "dim": dim,
                "coords": coords,          # CPU float32 [n,2]
                "edge_type": edge_type,    # "EUC_2D" / "CEIL_2D"
            })

    # ! 新增：按规模从小到大排序
    handles.sort(key=lambda x: x["dim"])
    return handles


# ! 新增：单实例准备（保持“全矩阵”）
def prepare_instance_fullmatrix(handle, k_sparse_factor, device, start_node=None):
    """
    输入: 单个实例句柄（name/dim/coords/edge_type）
    输出: (pyg_data, dist_int)
      - pyg_data: 用欧式距离做 KNN 的稀疏图（和你原版一致）
      - dist_int: TSPLIB 口径的整数距离矩阵（全矩阵，和原版一致）
    """
    name     = handle["name"]
    dim      = handle["dim"]
    coords   = handle["coords"].to(device)       # [n,2] float32
    edge_typ = handle["edge_type"]

    # KNN 构图（和你原 build_pyg_from_coords 一致）
    k_sparse = max(1, dim // k_sparse_factor) if dim > 1 else 0
    pyg_data, _ = build_pyg_from_coords(coords, k_sparse=k_sparse, start_node=start_node)
    # # 这一步可能抛 SkipInstanceError（OOM/过大）
    # pyg_data_cpu, euc_cpu = build_pyg_from_coords(coords, k_sparse=k_sparse, start_node=start_node)


    # 全矩阵（保持“全矩阵”做法）
    # dist_int = tsplib_integer_distance(coords, edge_typ).to('cpu')
    # 根据 edge_type 生成整数口径的整张距离矩阵（仍然在 CPU 上）
    dist_int = tsplib_integer_distance(coords, edge_typ)

    # # 如果后续模型需要把 pyg_data 放到某个 device，这里再搬过去
    # pyg_data = pyg_data_cpu.to(device) if device != 'cpu' else pyg_data_cpu

    return pyg_data, dist_int




def gen_distance_matrix(tsp_coordinates):
    '''
    Args:
        tsp_coordinates: torch tensor [n_nodes, 2] for node coordinates
    Returns:
        distance_matrix: torch tensor [n_nodes, n_nodes] for EUC distances
    '''
    n_nodes = len(tsp_coordinates)
    distances = torch.norm(tsp_coordinates[:, None] - tsp_coordinates, dim=2, p=2)
    distances[torch.arange(n_nodes), torch.arange(n_nodes)] = 1e9 # note here
    return distances


def gen_pyg_data(tsp_coordinates, k_sparse, start_node=None):
    '''
    Args:
        tsp_coordinates: torch tensor [n_nodes, 2] for node coordinates
        k_sparse: int, number of edges to keep for each node
        start_node: int, index of the start node, if None, use random start node
    Returns:
        pyg_data: pyg Data instance
        distances: distance matrix
    '''
    n_nodes = len(tsp_coordinates)
    distances = gen_distance_matrix(tsp_coordinates)
    topk_values, topk_indices = torch.topk(distances, 
                                           k=k_sparse, 
                                           dim=1, largest=False)
    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device),
                                repeats=k_sparse),
        torch.flatten(topk_indices)
        ])
    edge_attr = topk_values.reshape(-1, 1)

    if start_node is None:
        node_feature = tsp_coordinates
    else:
        node_feature = torch.zeros((n_nodes,1), device=tsp_coordinates.device, dtype=tsp_coordinates.dtype)
        node_feature[start_node, 0] = 1.0
    pyg_data = Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data, distances


def gen_pyg_data_tsplib(tsp_coordinates, k_sparse, start_node=None):
    '''
    Args:
        tsp_coordinates: torch tensor [n_nodes, 2] for node coordinates
        k_sparse: int, number of edges to keep for each node
        start_node: int, index of the start node, if None, use random start node
    Returns:
        pyg_data: pyg Data instance
        distances: distance matrix
    '''
    n_nodes = len(tsp_coordinates)
    distances = gen_distance_matrix(tsp_coordinates)
    topk_values, topk_indices = torch.topk(distances, k=k_sparse, dim=1, largest=False)
    edge_index = torch.stack(
        [
            torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device), repeats=k_sparse),
            torch.flatten(topk_indices)
        ]
    )
    edge_attr = topk_values.reshape(-1, 1)

    if start_node is None:
        node_feature = tsp_coordinates
    else:
        node_feature = torch.zeros((n_nodes,1), device=tsp_coordinates.device, dtype=tsp_coordinates.dtype)
        node_feature[start_node, 0] = 1.0
    pyg_data = Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data, distances


def load_val_dataset(n_nodes, k_sparse, device, start_node=None):
    if not os.path.isfile(f"../data/tsp/valDataset-{n_nodes}.pt"):
        val_tensor = torch.rand((50, n_nodes, 2))
        torch.save(val_tensor, f"../data/tsp/valDataset-{n_nodes}.pt")
    else:
        val_tensor = torch.load(f"../data/tsp/valDataset-{n_nodes}.pt")

    val_list = []
    for instance in val_tensor:
        instance = instance.to(device)
        data, distances = gen_pyg_data(instance, k_sparse=k_sparse, start_node=start_node)
        val_list.append((data, distances))
    return val_list


def use_saved_problems_tsp_txt(filename,k_sparse,device, start_node=None, start=0):
    nodes_coords = []
    solution = []
    from tqdm import tqdm
    for line in tqdm(open(filename, "r").readlines(), ascii=True):
        line = line.split(" ")
        num_nodes = int(line.index('output') // 2)
        nodes_coords.append(
            [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]
        )
        tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]]
        solution.append(tour_nodes)

    problems = torch.tensor(nodes_coords,device=device)  # shape: (batch, problem, 2)
    solution = torch.tensor(solution,device=device)  # shape: (batch, problem)
    gathering_index = solution.unsqueeze(2).expand(-1, -1, 2)
    # shape: (batch, problem, 2)
    ordered_seq = problems.gather(dim=1, index=gathering_index)
    rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
    segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(2).sqrt()
    # shape: (batch, problem)
    travel_distances = segment_lengths.sum(1)
    # shape: (batch,)
    optimal_score = travel_distances
    
    test_list = []
    for instance in problems:
        instance = instance.to(device)
        data, distances = gen_pyg_data(instance, k_sparse=k_sparse, start_node=start_node)
        test_list.append((data, distances))

    return test_list,optimal_score

def load_test_dataset(n_nodes, k_sparse, device, start_node=None, filename=None):
    filename = filename or f"../data/tsp/testDataset-{n_nodes}.pt"
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"File {filename} not found, please download the test dataset from the original repository."
        )
    test_tensor = torch.load(filename)

    test_list = []
    for instance in test_tensor:
        instance = instance.to(device)
        data, distances = gen_pyg_data(instance, k_sparse=k_sparse, start_node=start_node)
        test_list.append((data, distances))
    return test_list


def load_tsplib_dataset(n_nodes, k_sparse_factor, device, start_node=None, filename=None):
    scale_map = {200: ("100", "299"), 500: ("300", "699"), 1000: ("700", "1499")}
    # filename = filename or f"../data/tsp/tsplib/tsplib_{'_'.join(scale_map[n_nodes])}.pkl"
    filename = filename or f"./data/tsp/tsplib/tsplib_{'_'.join(scale_map[n_nodes])}.pkl"
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"File {filename} not found, please download the test dataset from the original repository."
        )

    with open(filename, "rb") as f:
        tsplib_list = pickle.load(f)

    test_list = []
    scale_list = []
    name_list = []
    for instance, scale, name in tsplib_list:
        instance = instance.to(device)
        data, distances = gen_pyg_data_tsplib(instance, k_sparse=instance.shape[0] // k_sparse_factor, start_node=start_node)
        test_list.append((data, distances))
        scale_list.append(scale)
        name_list.append(name)
    return test_list, scale_list, name_list


tsp_survey_bench_cost_all = {

    # TSPLIB, 77+4, all optimal, http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html
    "a280": 2579,
    # "ali535": 202339,
    # "att48": 10628,
    # "att532": 27686,
    # "bayg29": 1610,
    # "bays29": 2020,
    "berlin52": 7542,
    "bier127": 118282,
    # "brazil58": 25395,
    "brd14051": 469385,
    # "brg180": 1950,
    # "burma14": 3323,
    "ch130": 6110,
    "ch150": 6528,
    "d198": 15780,
    "d493": 35002,
    "d657": 48912,
    "d1291": 50801,
    "d1655": 62128,
    "d2103": 80450,
    "d15112": 1573084,
    "d18512": 645238,
    # "dantzig42": 699,
    # "dsj1000": 18659688, # (EUC_2D)
    "dsj1000": 18660188, # (CEIL_2D)
    "eil51": 426,
    "eil76": 538,
    "eil101": 629,
    "fl417": 11861,
    "fl1400": 20127,
    "fl1577": 22249,
    "fl3795": 28772,
    "fnl4461": 182566,
    # "fri26": 937,
    "gil262": 2378,
    # "gr17": 2085,
    # "gr21": 2707,
    # "gr24": 1272,
    # "gr48": 5046,
    # "gr96": 55209,
    # "gr120": 6942,
    # "gr137": 69853,
    # "gr202": 40160,
    # "gr229": 134602,
    # "gr431": 171414,
    # "gr666": 294358,
    # "hk48": 11461,
    "kroA100": 21282,
    "kroB100": 22141,
    "kroC100": 20749,
    "kroD100": 21294,
    "kroE100": 22068,
    "kroA150": 26524,
    "kroB150": 26130,
    "kroA200": 29368,
    "kroB200": 29437,
    "lin105": 14379,
    "lin318": 42029,
    "linhp318": 41345,
    "nrw1379": 56638,
    "p654": 34643,
    # "pa561": 2763,
    "pcb442": 50778,
    "pcb1173": 56892,
    "pcb3038": 137694,
    "pla7397": 23260728, # (CEIL_2D)
    "pla33810": 66048945, # (CEIL_2D)
    "pla85900": 142382641, # (CEIL_2D)
    "pr76": 108159,
    "pr107": 44303,
    "pr124": 59030,
    "pr136": 96772,
    "pr144": 58537,
    "pr152": 73682,
    "pr226": 80369,
    "pr264": 49135,
    "pr299": 48191,
    "pr439": 107217,
    "pr1002": 259045,
    "pr2392": 378032,
    "rat99": 1211,
    "rat195": 2323,
    "rat575": 6773,
    "rat783": 8806,
    "rd100": 7910,
    "rd400": 15281,
    "rl1304": 252948,
    "rl1323": 270199,
    "rl1889": 316536,
    "rl5915": 565530,
    "rl5934": 556045,
    "rl11849": 923288,
    # "si175": 21407,
    # "si535": 48450,
    # "si1032": 92650,
    "st70": 675,
    # "swiss42": 1273,
    "ts225": 126643,
    "tsp225": 3916,
    "u159": 42080,
    "u574": 36905,
    "u724": 41910,
    "u1060": 224094,
    "u1432": 152970,
    "u1817": 57201,
    "u2152": 64253,
    "u2319": 234256,
    # "ulysses16": 6859,
    # "ulysses22": 7013,
    "usa13509": 19982859,
    "vm1084": 239297,
    "vm1748": 336556,

    # National TSP, 27, 2 non-optimal, https://www.math.uwaterloo.ca/tsp/world/summary.html
    'ar9152': 837_479,
    'bm33708': 959_289, # gap 0.031%
    'ca4663': 1_290_319,
    'ch71009': 4_566_506, # gap 0.024%
    'dj38': 6_656,
    'eg7146': 172_386,
    'fi10639': 520_527,
    'gr9882': 300_899,
    'ho14473': 177_092,
    'ei8246': 206_171,
    'it16862': 557_315,
    'ja9847': 491_924,
    'kz9976': 1_061_881,
    'lu980': 11_340,
    'mo14185': 427_377,
    'nu3496': 96_132,
    'mu1979': 86_891,
    "pm8079": 114_855,
    'qa194': 9_352,
    'rw1621': 26_051,
    'sw24978': 855_597,
    'tz6117': 394_718,
    'uy734': 79_114,
    'vm22775': 569_288,
    'wi29': 27_603,
    'ym7663': 238_314,
    'zi929': 95_345,

    # VLSI, 102-4, non-optimal when size >= 14233 (xrb14233), https://www.math.uwaterloo.ca/tsp/vlsi/summary.html
    'xqf131': 564,
    'xqg237': 1_019,
    'pma343': 1_368,
    'pka379': 1_332,
    'bcl380': 1_621,
    'pbl395': 1_281,
    'pbk411': 1_343,
    'pbn423': 1_365,
    'pbm436': 1_443,
    'xql662': 2_513,
    'rbx711': 3_115,
    'rbu737': 3_314,
    'dkg813': 3_199,
    'lim963': 2_789,
    'pbd984': 2_797,
    'xit1083': 3_558,
    'dka1376': 4_666,
    'dca1389': 5_085,
    'dja1436': 5_257,
    'icw1483': 4_416,
    'fra1488': 4_264,
    'rbv1583': 5_387,
    'rby1599': 5_533,
    'fnb1615': 4_956,
    'djc1785': 6_115,
    'dcc1911': 6_396,
    'dkd1973': 6_421,
    'djb2036': 6_197,
    'dcb2086': 6_600,
    'bva2144': 6_304,
    'xqc2175': 6_830,
    'bck2217': 6_764,
    'xpr2308': 7_219,
    'ley2323': 8_352,
    'dea2382': 8_017,
    'rbw2481': 7_724,
    'pds2566': 7_643,
    'mlt2597': 8_071,
    'bch2762': 8_234,
    'irw2802': 8_423,
    'lsm2854': 8_014,
    'dbj2924': 10_128,
    'xva2993': 8_492,
    'pia3056': 8_258,
    'dke3097': 10_539,
    'lsn3119': 9_114,
    'lta3140': 9_517,
    'fdp3256': 10_008,
    'beg3293': 9_772,
    'dhb3386': 11_137,
    'fjs3649': 9_272,
    'fjr3672': 9_601,
    'dlb3694': 10_959,
    'ltb3729': 11_821,
    'xqe3891': 11_995,
    'xua3937': 11_239,
    'dkc3938': 12_503,
    'dkf3954': 12_538,
    'bgb4355': 12_723,
    'bgd4396': 13_009,
    'frv4410': 10_711,
    'bgf4475': 13_221,
    'xqd4966': 15_316,
    'fqm5087': 13_029,
    'fea5557': 15_445,
    'xsc6880': 21_535,
    'bnd7168': 21_834,
    'lap7454': 19_535,
    'ida8197': 22_338,
    'dga9698': 27_724,
    'xmc10150': 28_387,
    'xvb13584': 37_083,
    'xrb14233': 45_462, # gap 0.026%
    'xia16928': 52_850, # gap 0.023%
    'pjh17845': 48_092, # gap 0.019%
    'frh19289': 55_798, # gap 0.013%
    'fnc19402': 59_287, # gap 0.020%
    'ido21215': 63_517, # gap 0.028%
    'fma21553': 66_527, # gap unknown
    'lsb22777': 60_977, # gap unknown
    'xrh24104': 69_294, # gap unknown
    'bbz25234': 69_335, # gap unknown
    'irx28268': 72_607, # gap unknown
    'fyg28534': 78_562, # gap unknown
    'icx28698': 78_087, # gap unknown
    'boa28924': 79_622, # gap unknown
    'ird29514': 80_353, # gap unknown
    'pbh30440': 88_313, # gap unknown
    'xib32892': 96_757, # gap unknown
    'fry33203': 97_240, # gap unknown
    'bby34656': 99_159, # gap unknown
    'pba38478': 108_318, # gap unknown
    'ics39603': 106_819, # gap unknown
    'rbz43748': 125_183, # gap unknown
    'fht47608': 125_104, # gap unknown
    'fna52057': 147_789, # gap unknown
    'bna56769': 158_078, # gap unknown
    'dan59296': 165_371, # gap unknown
    # 'sra104815': 251_342, # gap unknown
    # 'ara238025': 578_761, # gap unknown
    # 'lra498378': 2_168_039, # gap unknown
    # 'lrb744710': 1_611_232, # gap unknown

    # DIMACS 8th Challenge, non-optimal, http://dimacs.rutgers.edu/archive/Challenges/TSP/opts.html and http://webhotel4.ruc.dk/~keld/research/LKH/DIMACS_results.html
    "portcgen-1000-1000": 11387430, # gap 0.54% C1k.0
    "portcgen-1000-10001": 11376735, # gap 0.41% C1k.1
    "portcgen-1000-10002": 10855033, # gap 0.42% C1k.2
    "portcgen-1000-10003": 11886457, # gap 0.53% C1k.3
    "portcgen-1000-10004": 11499958, # gap 0.58% C1k.4
    "portcgen-1000-10005": 11394911, # gap 0.58% C1k.5
    "portcgen-1000-10006": 10166701, # gap 0.73% C1k.6
    "portcgen-1000-10007": 10664660, # gap 0.58% C1k.7
    "portcgen-1000-10008": 11605723, # gap 0.34% C1k.8
    "portcgen-1000-10009": 10906997, # gap 0.66% C1k.9
    "portcgen-3162-3162": 19198258, # gap 0.62% C3k.0
    "portcgen-3162-31621": 19017805, # gap 0.61% C3k.1
    "portcgen-3162-31622": 19547551, # gap 0.70% C3k.2
    "portcgen-3162-31623": 19108508, # gap 0.57% C3k.3
    "portcgen-3162-31624": 18864046, # gap 0.57% C3k.4
    "portcgen-10000-10000": 33_001_034, # gap 0.668% C10k.0
    "portcgen-10000-100001": 33_186_248, # gap 0.690% C10k.1
    "portcgen-10000-100002": 33_155_424, # gap 0.694% C10k.2
    "portcgen-31623-31623": 59_545_390, # gap 0.636% C31k.0
    "portcgen-31623-316231": 59_293_266, # gap 0.770% C31k.1
    "portcgen-100000-100000": 104_617_752, # gap 0.675% C100k.0
    "portcgen-100000-1000001": 105_390_777, # gap 0.695% C100k.1
    # "C316k.0": 186_870_839 # gap 0.697% C316k.0
}



if __name__ == "__main__":
    pass
