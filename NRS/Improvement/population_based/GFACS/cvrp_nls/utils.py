import os
import pickle

import numpy as np
import torch
from torch_geometric.data import Data


DEMAND_LOW = 1
DEMAND_HIGH = 9

# ! 新增
def parse_cvrplib_file(path: str):
    """
    仅支持 EUC_2D / CEIL_2D。
    返回：
      name(str), dim(int=客户数), coords(torch.FloatTensor[n+1,2]), demand(torch.FloatTensor[n+1]), capacity(float), edge_type(str), optimal(float|None)
    """
    name, dim, edge_type, capacity = None, None, None, None
    coords, demand = [], []
    started_node, started_demand = False, False

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if started_demand:
                if line.startswith("DEPOT_SECTION"):
                    break
                demand.append(float(line.split()[-1]))
            if started_node:
                if line.startswith("DEMAND_SECTION"):
                    started_node, started_demand = False, True
                else:
                    parts = line.split()
                    coords.append([float(parts[1]), float(parts[2])])

            if line.startswith("NAME"):
                name = line.replace(":", " ").split()[-1]
            elif line.startswith("DIMENSION"):
                dim = int(float(line.replace(":", " ").split()[-1]) - 1)  # depot 不计
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                t = line.replace(":", " ").split()[-1]
                if t not in ("EUC_2D", "CEIL_2D"):
                    return None, None, None, None, None, None, None
                edge_type = t
            elif line.startswith("CAPACITY"):
                capacity = float(line.replace(":", " ").split()[-1])
            elif line.startswith("NODE_COORD_SECTION"):
                started_node = True

    # label（.sol -> Cost）
    cost = None
    sol = path.replace(".vrp", ".sol")
    if os.path.exists(sol):
        with open(sol, "r") as f:
            for raw in f:
                if raw.startswith("Cost"):
                    try:
                        cost = float(raw.split()[1])
                    except:
                        pass

    if name is None or dim is None or edge_type is None or capacity is None:
        return None, None, None, None, None, None, None

    assert len(coords) == dim + 1, f"{path}: coords 行数与 DIMENSION 不匹配"
    assert len(demand) == dim + 1, f"{path}: demand 行数与 DIMENSION 不匹配"

    coords = torch.tensor(coords, dtype=torch.float64)          # [n+1, 2] 含 depot
    demand = torch.tensor(demand, dtype=torch.float64)          # [n+1]   含 depot(0)
    return name, dim, coords, demand, capacity, edge_type, cost

# ! 新增
def cvrp_integer_distance(coords: torch.Tensor, edge_type: str) -> torch.Tensor:
    """
    TSPLIB 口径的整数距离矩阵（EUC_2D: 四舍五入；CEIL_2D: 向上取整）
    coords: [n+1, 2]  返回: [n+1, n+1]，对角线置0
    """
    n = coords.size(0)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_row = torch.linalg.norm(diff, dim=2)
    # if edge_type == "EUC_2D":
    # dist = torch.round(dist)
    dist = torch.floor(dist_row + 0.5)
    # elif edge_type == "CEIL_2D":
        # dist = torch.ceil(dist)
    # else:
    #     raise ValueError(edge_type)
    dist[torch.arange(n), torch.arange(n)] = 0
    return dist

# ! 新增
def build_pyg_cvrp(coords_norm: torch.Tensor, demands_norm: torch.Tensor, k_sparse: int):
    """
    CVRP 的 PyG 稀疏图：客户-客户 KNN（k_sparse），depot 与所有客户双向全连。
    coords_norm: [n+1,2]  demands_norm: [n+1]（已/容量）; 返回 Data(x, edge_index, edge_attr)
    """
    device = coords_norm.device
    n = coords_norm.size(0)  # 含 depot(0)
    # 欧式浮点距（用于 KNN）
    diff = coords_norm[:, None, :] - coords_norm[None, :, :]
    euc = torch.linalg.norm(diff, dim=2)
    euc[torch.arange(n), torch.arange(n)] = 1e9

    k = min(k_sparse, n - 2) if n > 2 else 0  # 只在客户-客户之间 KNN
    topk_vals, topk_idx = torch.topk(euc[1:, 1:], k=k, dim=1, largest=False) if k > 0 \
        else (torch.empty(0, device=device), torch.empty(0, device=device, dtype=torch.long))

    # 客户-客户 KNN 边
    if k > 0:
        edge_u1 = torch.repeat_interleave(torch.arange(1, n, device=device), repeats=k)
        edge_v1 = topk_idx.reshape(-1) + 1
        edge_attr1 = topk_vals.reshape(-1, 1)
        edge_index_1 = torch.stack([edge_u1, edge_v1], dim=0)
    else:
        edge_index_1 = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr1 = torch.empty((0, 1), dtype=torch.float32, device=device)

    # depot 全连
    clients = torch.arange(1, n, device=device)
    edge_index_2 = torch.stack([torch.zeros_like(clients), clients], dim=0)
    edge_index_3 = torch.stack([clients, torch.zeros_like(clients)], dim=0)
    edge_attr2 = euc[0, 1:].reshape(-1, 1)
    edge_index = torch.cat([edge_index_1, edge_index_2, edge_index_3], dim=1)
    edge_attr = torch.cat([edge_attr1, edge_attr2, edge_attr2], dim=0)

    x = demands_norm.unsqueeze(1).float()  # 节点特征：需求
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr.float())

# ! 新增
def load_cvrplib_handles(root_dir: str, dim_ranges=None):
    """
    轻量“句柄”列表：[{name, dim, coords, demand, capacity, edge_type, optimal}, ...]
    仅解析 .vrp 与 .sol，不构图、不算矩阵。
    """
    handles = []
    for root, _, files in os.walk(root_dir):
        for fn in sorted([f for f in files if f.lower().endswith(".vrp")]):
            full = os.path.join(root, fn)
            name, dim, coords, demand, cap, etype, opt = parse_cvrplib_file(full)
            if name is None:
                continue
            if dim_ranges is not None:
                ok = any(lo <= dim < hi for (lo, hi) in dim_ranges)
                if not ok:
                    continue
            handles.append({
                "name": name, "dim": dim, "coords": coords, "demand": demand,
                "capacity": cap, "edge_type": etype, "optimal": opt, "path": full
            })
    return handles

# ! 新增
def prepare_instance_fullmatrix_cvrp(handle, k_sparse_factor: int, device: str):
    """
    输入：handle（见上）
    输出：pyg_data（稀疏图，基于归一化坐标&需求）、normed_demand、normed_dist、positions(normed)、int_dist(整数口径)
    说明：归一化同 ICAM：min-max 到 [0,1]，距离相应缩放；评分用 int_dist（EUC round / CEIL ceil）。
    """
    name   = handle["name"]
    dim    = handle["dim"]
    # 修改：保持在 CPU 计算，避免 OOM
    coords = handle["coords"]   # [n+1,2] 含 depot (CPU)
    demand = handle["demand"]   # [n+1]   含 depot(0) (CPU)
    cap    = float(handle["capacity"])
    etype  = handle["edge_type"]

    # 1) label 距离矩阵（整数口径）
    # 在 CPU 上计算
    int_dist_tensor = cvrp_integer_distance(coords, etype)
    # 转为 numpy int 格式
    int_dist = int_dist_tensor.numpy().astype(int)

    # 2) 归一化坐标（与 load_vrplib_dataset 一致）
    xy_max, _ = coords.max(dim=0, keepdim=True)
    xy_min, _ = coords.min(dim=0, keepdim=True)
    scale = (xy_max - xy_min).max() / 0.98
    coords_norm = (coords - xy_min) / scale + 0.01

    # 3) 归一化距离（求解口径）
    # 在 CPU 上计算
    dist_norm = int_dist_tensor / scale
    dist_norm[torch.arange(dim + 1), torch.arange(dim + 1)] = 1e-10

    # 4) 归一化需求（/capacity）
    demand_norm = demand / cap

    # 5) 稀疏图（客户-客户 KNN + depot 全连）
    k_sparse = (dim + 1) // k_sparse_factor
    
    # 准备移至 GPU 的数据 (与 load_vrplib_dataset 一致)
    demand_norm_gpu = demand_norm.to(device)
    dist_norm_gpu = dist_norm.to(device)
    positions_gpu = coords_norm.to(device)

    # 使用 gen_pyg_data 以保持与 load_vrplib_dataset 一致的构图逻辑
    pyg_data = gen_pyg_data(demand_norm_gpu, dist_norm_gpu, device, k_sparse=k_sparse)

    return pyg_data, demand_norm_gpu, dist_norm_gpu, positions_gpu, int_dist



def get_capacity(n: int, tam=False):
    if tam:
        capacity_list_tam = [
            (1, 10), (20, 30), (50, 40), (100, 50), (400, 150), (1000, 200), (2000, 300)  # (number of nodes, capacity)
        ]
        return list(filter(lambda x: x[0]<=n, capacity_list_tam))[-1][-1]

    capacity_dict = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.,
        200: 50.,
        500: 50.,
        1000: 50.,
        2000: 50.,
    }
    assert n in capacity_dict
    return capacity_dict[n]


def gen_instance(n, device, tam=False):
    """
    Implements data-generation method as described by Kool et al. (2019), Hou et al. (2023), and Son et al. (2023)

    * Kool, W., van Hoof, H., & Welling, M. (2019). Attention, Learn to Solve Routing Problems! (arXiv:1803.08475)
    * Hou, Q., Yang, J., Su, Y., Wang, X., & Deng, Y. (2023, February 1). Generalize Learned Heuristics to Solve Large-scale Vehicle Routing Problems in Real-time. The Eleventh International Conference on Learning Representations. https://openreview.net/forum?id=6ZajpxqTlQ
    * Son, J., et al. (2023). Meta-SAGE: Scale Meta-Learning Scheduled Adaptation with Guided Exploration for Mitigating Scale Shift on Combinatorial Optimization (arXiv:2306.02688)
    """
    locations = torch.rand(size=(n + 1, 2), device=device, dtype=torch.double)
    demands = torch.randint(low=DEMAND_LOW, high=DEMAND_HIGH + 1, size=(n, ), device=device, dtype=torch.double)
    demands_normalized = demands / get_capacity(n, tam)
    all_demands = torch.cat((torch.zeros((1, ), device=device, dtype=torch.double), demands_normalized))
    distances = gen_distance_matrix(locations)
    return all_demands, distances, locations


def gen_distance_matrix(tsp_coordinates):
    n_nodes = len(tsp_coordinates)
    distances = torch.norm(tsp_coordinates[:, None] - tsp_coordinates, dim=2, p=2, dtype=torch.double)
    distances[torch.arange(n_nodes), torch.arange(n_nodes)] = 1e-10  # note here
    return distances


def gen_pyg_data(demands, distances, device, k_sparse):
    n = demands.size(0)
    # First mask out self-loops by setting them to large values
    temp_dists = distances.clone()
    temp_dists[1:, 1:][torch.eye(n - 1, dtype=torch.bool, device=device)] = 1e9
    # sparsify
    # part 1:
    topk_values, topk_indices = torch.topk(temp_dists[1:, 1:], k = k_sparse, dim=1, largest=False)
    edge_index_1 = torch.stack([
        torch.repeat_interleave(torch.arange(n-1).to(topk_indices.device), repeats=k_sparse),
        torch.flatten(topk_indices)
    ]) + 1
    edge_attr_1 = topk_values.reshape(-1, 1)
    # part 2: keep all edges connected to depot
    edge_index_2 = torch.stack([ 
        torch.zeros(n - 1, device=device, dtype=torch.long), 
        torch.arange(1, n, device=device, dtype=torch.long),
    ])
    edge_attr_2 = temp_dists[1:, 0].reshape(-1, 1)
    edge_index_3 = torch.stack([ 
        torch.arange(1, n, device=device, dtype=torch.long),
        torch.zeros(n - 1, device=device, dtype=torch.long), 
    ])
    edge_index = torch.concat([edge_index_1, edge_index_2, edge_index_3], dim=1)
    edge_attr = torch.concat([edge_attr_1, edge_attr_2, edge_attr_2])

    x = demands
    # FIXME: append node type and coordinates into x
    pyg_data = Data(x=x.unsqueeze(1).float(), edge_attr=edge_attr.float(), edge_index=edge_index)
    return pyg_data


def load_test_dataset(n_node, k_sparse, device, tam=False):
    # filename = f"../data/cvrp/testDataset-{'tam-' if tam else ''}{n_node}.pt"
    filename = f"/public/home/bayp/exp_survey_202509/Improvement/population_based/GFACS/data/cvrp/vrp{n_node}_128.pkl"
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"File {filename} not found, please download the test dataset from the original repository."
        )
    # dataset = torch.load(filename, map_location=device)
    test_list = []
    for i in range(len(dataset)):
        demands, position, distances = dataset[i, 0, :], dataset[i, 1:3, :], dataset[i, 3:, :]
        pyg_data = gen_pyg_data(demands, distances, device, k_sparse=k_sparse)
        test_list.append((pyg_data, demands, distances, position.T))
    return test_list

# ! 新加,读取pkl文件
def load_test_dataset_pkl(n_node, k_sparse, device, tam=False):
    filename = f"/public/home/bayp/exp_survey_202509/Improvement/population_based/GFACS/data/cvrp/vrp{n_node}_128.pkl"
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"File {filename} not found, please download the test dataset from the original repository."
        )
    # dataset = torch.load(filename, map_location=device)
    try:
        # 尝试用 torch 加载
        dataset = torch.load(filename, map_location=device)
    except RuntimeError:
        print("torch.load 失败，尝试使用 pickle.load...")
        with open(filename, 'rb') as f:
            dataset = pickle.load(f)
        print("pickle.load 成功！") 
    test_list = []
    out = np.array(dataset, dtype=object)
    raw_data_depot = torch.tensor(out[:, 0].tolist(), dtype=torch.float32).to(device)
    if raw_data_depot.dim() == 2:
        raw_data_depot = raw_data_depot[:, None, :] # shape: (batch, 1, 2)
    raw_data_nodes = torch.tensor(out[:, 1].tolist(), dtype=torch.float32).to(device)
    # shape: (batch, problem, 2)
    raw_data_demand = torch.tensor(out[:, 2].tolist(), dtype=torch.float32)
    # shape: (batch, problem)
    capacity = float(out[0, 3])
    raw_data_demand = (raw_data_demand / capacity).to(device)
    for i in range(raw_data_nodes.size(0)):
        position = torch.cat((raw_data_depot[i], raw_data_nodes[i]), dim=0) # shape: (problem+1, 2)
        demands = torch.cat((torch.zeros((1,), device=device), raw_data_demand[i]), dim=0) # shape: (problem+1,)
        distances = gen_distance_matrix(position) # shape: (problem+1, problem+1)
        pyg_data = gen_pyg_data(demands, distances, device, k_sparse=k_sparse)
        test_list.append((pyg_data, demands, distances, position))
    return test_list
    
    # for i in range(len(dataset)):
    #     demands, position, distances = dataset[i, 0, :], dataset[i, 1:3, :], dataset[i, 3:, :]
    #     pyg_data = gen_pyg_data(demands, distances, device, k_sparse=k_sparse)
    #     test_list.append((pyg_data, demands, distances, position.T))
    # return test_list


def load_val_dataset(n_node, k_sparse, device, tam=False):
    filename = f"../data/cvrp/valDataset-{'tam-' if tam else ''}{n_node}.pt"
    if not os.path.isfile(filename):
        dataset = []
        for i in range(50):
            demand, dist, position = gen_instance(n_node, device, tam)  # type: ignore
            instance = torch.vstack([demand, position.T, dist])
            dataset.append(instance)
        dataset = torch.stack(dataset)
        torch.save(dataset, filename)
    else:
        dataset = torch.load(filename, map_location=device)

    val_list = []
    for i in range(len(dataset)):
        demands, position, distances = dataset[i, 0, :], dataset[i, 1:3, :], dataset[i, 3:, :]
        pyg_data = gen_pyg_data(demands, distances, device, k_sparse=k_sparse)
        val_list.append((pyg_data, demands, distances, position.T))
    return val_list


def load_vrplib_dataset(n_nodes, k_sparse_factor, device, dataset_name="X", filename=None):
    if dataset_name == "X":
        scale_map = {100: "100_299", 200: "100_299", 400: "300_699", 500: "300_699", 1000: "700_1001"}
    elif dataset_name == "M":
        scale_map = {100: "100_200", 200: "100_200"}
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}")

    absolute_path = "/public/home/bayp/exp_survey_202509/Improvement/population_based/GFACS/"
    filename = filename or f"{absolute_path}data/cvrp/vrplib/vrplib_{dataset_name}_{scale_map[n_nodes]}.pkl"
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"File {filename} not found, please download the test dataset from the original repository."
        )
    with open(filename, "rb") as f:
        vrplib_list = pickle.load(f)

    test_list = []
    int_dist_list = []
    name_list = []
    for normed_demand, position, distance, name in vrplib_list:
        # demand is already normalized by capacity
        # normalize the position and distance into [0.01, 0.99] range
        scale = (position.max(0) - position.min(0)).max() / 0.98
        position = position - position.min(0)
        position = position / scale + 0.01
        normed_dist = distance / scale
        np.fill_diagonal(normed_dist, 1e-10)
        # convert all to torch
        normed_demand = torch.tensor(normed_demand, device=device, dtype=torch.float64)
        normed_dist = torch.tensor(normed_dist, device=device, dtype=torch.float64)
        position = torch.tensor(position, device=device, dtype=torch.float64)
        pyg_data = gen_pyg_data(normed_demand, normed_dist, device, k_sparse=position.shape[0] // k_sparse_factor)

        test_list.append((pyg_data, normed_demand, normed_dist, position))
        int_dist_list.append(distance)
        name_list.append(name)
    return test_list, int_dist_list, name_list


if __name__ == '__main__':
    import pathlib
    pathlib.Path('../data/cvrp').mkdir(exist_ok=True)

    # TAM dataset
    for n in [100, 400, 1000]:  # problem scale
        torch.manual_seed(123456)
        inst_list = []
        for _ in range(100):
            demand, dist, position = gen_instance(n, 'cpu', tam=True)  # type: ignore
            instance = torch.vstack([demand, position.T, dist])
            inst_list.append(instance)
        testDataset = torch.stack(inst_list)
        torch.save(testDataset, f'../data/cvrp/testDataset-tam-{n}.pt')

    # main Dataset
    for scale in [200, 500, 1000, 2000]:
        inst_list = []
        try:
            with open(f"../data/cvrp/vrp{scale}_128.pkl", "rb") as f:
                dataset = pickle.load(f)

            for instance in dataset:
                depot_position, positions, demands, capacity = instance

                demands_torch = torch.tensor([0] + [d / capacity for d in demands], dtype=torch.float64)
                positions_torch = torch.tensor([depot_position] + positions, dtype=torch.float64)
                distmat_torch = gen_distance_matrix(positions_torch)
                inst_list.append(torch.vstack([demands_torch, positions_torch.T, distmat_torch]))
        except FileNotFoundError:
            torch.manual_seed(123456)
            for i in range(16):
                demands_torch, distmat_torch, positions_torch = gen_instance(scale, 'cpu')
                inst_list.append(torch.vstack([demands_torch, positions_torch.T, distmat_torch]))

        test_dataset = torch.stack(inst_list)
        torch.save(test_dataset, f"../data/cvrp/testDataset-{scale}.pt")
