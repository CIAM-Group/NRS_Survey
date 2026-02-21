"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import numpy
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import pdist, squareform
import numpy as np
from torch.utils.data.dataloader import default_collate

import os, fnmatch

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class DataSet(Dataset):

    def __init__(self, node_coords, demands, capacities, remaining_capacities, tour_lens=None, via_depots=None):
        self.node_coords = node_coords
        self.demands = demands
        self.capacities = capacities
        self.remaining_capacities = remaining_capacities
        self.via_depots = via_depots
        self.tour_lens = tour_lens

    def __len__(self):
        return len(self.node_coords)

    def __getitem__(self, item):
        node_coords = self.node_coords[item]
        demands = self.demands[item]
        capacity = self.capacities[item]
        if self.tour_lens is not None:
            tour_len = self.tour_lens[item]
        else:
            tour_len = numpy.array([])

        if self.remaining_capacities is not None:
            via_depots = self.via_depots[item]
            current_capacities = self.remaining_capacities[item]
        else:
            via_depots = numpy.array([])
            current_capacities = numpy.array([])

        distance_matrix = squareform(pdist(node_coords, metric='euclidean'))

        # From list to tensors as a DotDict
        item_dict = DotDict()
        item_dict.dist_matrices = torch.Tensor(distance_matrix)
        item_dict.node_coords = torch.Tensor(node_coords)
        item_dict.demands = torch.Tensor(demands)
        item_dict.capacities = torch.tensor(capacity).float()
        item_dict.remaining_capacities = torch.Tensor(current_capacities)
        item_dict.tour_len = torch.tensor(tour_len)
        item_dict.via_depots = torch.Tensor(via_depots).long()
        return item_dict
    
# =========================
# 新增：CVRPLIB 解析工具
# =========================
def _read_cvrplib_header_only(path):
    """
    只读头部拿 name / n_customers(=DIMENSION-1) / capacity / EWT；遇到 NODE_COORD_SECTION 停。
    返回: (name:str|None, n:int|None, capacity:float|None, ewt:str|None)
    """
    name, dimension, capacity, ewt = None, None, None, None
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("NAME"):
                toks = line.replace(":", " ").split()
                name = toks[-1]
            elif line.startswith("DIMENSION"):
                toks = line.replace(":", " ").split()
                try:
                    dimension = int(float(toks[-1]) - 1)  # DIMENSION 含 depot
                except:
                    pass
            elif line.startswith("CAPACITY"):
                toks = line.replace(":", " ").split()
                try:
                    capacity = float(toks[-1])
                except:
                    pass
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                ewt = line.replace(":", " ").split()[-1]
            elif line.startswith("NODE_COORD_SECTION"):
                break
    return name, dimension, capacity, ewt

def _read_cvrplib_full(path):
    """
    完整解析 .vrp：
      - coords: [N+1,2]，含 depot(索引0)
      - demand: [N+1]， demand[0]=0
      - capacity: float
      - ewt: EDGE_WEIGHT_TYPE（无则默认为 EUC_2D）
      - opt_cost: 若同名 .sol 存在，读取 'Cost <val>'；否则 None
    返回: (name, n_customers, coords(np.float32), demand(np.float32), capacity(float), ewt(str), opt_cost(float|None))
    """
    name, n_customers, capacity, ewt = None, None, None, None
    coords, demand = [], []
    started_node = False
    started_demand = False

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if started_demand:
                if line.startswith("DEPOT_SECTION"):
                    started_demand = False
                else:
                    try:
                        demand.append(float(line.split()[-1]))
                    except:
                        pass
            if started_node:
                if line.startswith("DEMAND_SECTION"):
                    started_node = False
                    started_demand = True
                else:
                    toks = line.split()
                    if len(toks) >= 3:
                        try:
                            coords.append([float(toks[1]), float(toks[2])])
                        except:
                            pass
            if line.startswith("NAME"):
                name = line.replace(":", " ").split()[-1]
            elif line.startswith("DIMENSION"):
                n_customers = int(float(line.replace(":", " ").split()[-1]) - 1)
            elif line.startswith("CAPACITY"):
                capacity = float(line.replace(":", " ").split()[-1])
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                ewt = line.replace(":", " ").split()[-1]
            elif line.startswith("NODE_COORD_SECTION"):
                started_node = True

    if ewt is None:
        ewt = "EUC_2D"  # 缺省按 EUC_2D 处理

    coords = np.array(coords, dtype=np.float32)   # [N+1,2]
    demand = np.array(demand, dtype=np.float32)   # [N+1]

    # 读取 label（可选）
    opt_cost = None
    sol_path = path.replace(".vrp", ".sol").replace(".VRP", ".sol")
    if os.path.exists(sol_path):
        with open(sol_path, "r") as f:
            for line in f:
                if line.strip().startswith("Cost"):
                    try:
                        opt_cost = float(line.split()[1])
                    except:
                        pass
                    break

    return name, n_customers, coords, demand, capacity, ewt, opt_cost

def _cvrp_pairwise_dist(coords, ewt):
    """
    VRPLIB 计费：EUC_2D → round；CEIL_2D → ceil；默认 round。
    用原始坐标构造，再转 float32。
    """
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=-1))
    if (ewt or "").upper() == "CEIL_2D":
        d = np.ceil(d)
    else:
        # d = np.rint(d)  # EUC_2D / 默认
        d = np.floor(d+0.5)
    return d.astype(np.float32)


def _collect_vrp_files_filtered(root_dir, min_nodes=0, max_nodes=10**9):
    """
    递归收集 .vrp/.VRP，按客户数 N 过滤 [min_nodes, max_nodes)，并按 N 升序返回路径列表。
    """
    files_with_n = []
    for r, _, fnames in os.walk(root_dir):
        for fn in fnames:
            if fnmatch.fnmatch(fn, "*.vrp") or fnmatch.fnmatch(fn, "*.VRP"):
                path = os.path.join(r, fn)
                name, n, _cap, ewt = _read_cvrplib_header_only(path)
                if name is None or n is None:
                    continue
                # 只接受常用 EWT；没有则默认 EUC_2D，在 full 解析里处理
                if ewt is not None and ewt not in {"EUC_2D", "CEIL_2D"}:
                    continue
                if min_nodes <= n < max_nodes:
                    files_with_n.append((path, n))
    files_with_n.sort(key=lambda x: x[1])
    return [p for p, _ in files_with_n]

# =========================
# 新增：CVRPLIBDataSet（测试/验证用）
# =========================
class CVRPLIBDataSet(Dataset):
    """
    每个样本是一份 .vrp：
      - node_coords: [N+1,2]   归一化到[0,1]（含 depot 在 idx=0）
      - dist_matrices: [N+1,N+1]  用原始坐标按 VRPLIB 计费（EUC_2D→round, CEIL_2D→ceil）
      - demands: [N+1]  demand[0]=0（原值）
      - capacities: []  标量
      - remaining_capacities: []  占位（0.0）
      - via_depots: [2] 占位（0.0, 0.0）
      - tour_len: []  label（若有 .sol）
      - name / problem_size: 方便外层日志
    """
    def __init__(self, file_list):
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        name, n_cust, coords, demand, capacity, ewt, opt_cost = _read_cvrplib_full(path)
        if name is None:
            raise RuntimeError(f"CVRPLIB parse error: {path}")
        # 构造距离（原始坐标 + 计费规则）
        dist_mat = _cvrp_pairwise_dist(coords, ewt)  # [N+1,N+1]

        # 归一化坐标（max-range）
        # xy_min = coords.min(axis=0, keepdims=True)
        # xy_max = coords.max(axis=0, keepdims=True)
        # span = xy_max - xy_min
        # ratio = float(np.max(span))
        # if ratio == 0:
        #     ratio = 1.0
        # coords_norm = (coords - xy_min) / ratio

        # ! 换global
        max_value = np.max(coords)
        min_value = np.min(coords)
        coords_norm = (coords - min_value) / (max_value - min_value)

        item = DotDict()
        item.node_coords = torch.tensor(coords_norm, dtype=torch.float32)        # [N+1,2]
        item.dist_matrices = torch.tensor(dist_mat, dtype=torch.float32)         # [N+1,N+1]
        item.demands = torch.tensor(demand, dtype=torch.float32)                 # [N+1]
        item.capacities = torch.tensor(capacity, dtype=torch.float32)            # []
        item.remaining_capacities = torch.tensor(0.0, dtype=torch.float32)       # []
        item.via_depots = torch.tensor([0.0, 0.0], dtype=torch.float32)          # [2]
        if opt_cost is not None:
            item.tour_len = torch.tensor(opt_cost, dtype=torch.float32)          # []
        else:
            item.tour_len = torch.tensor([], dtype=torch.float32)                # 空 label
        item.name = name
        item.problem_size = int(n_cust)
        item.edge_weight_type = ewt
        return item
    




# def load_dataset(filename, batch_size, shuffle=False, what="test"):
#     data = np.load(filename)

#     if what == "train":
#         assert data["reorder"]

#     node_coords = data["coords"]
#     demands = data["demands"]
#     capacities = data["capacities"]


#     # in training dataset we have via_depots and remaining capacities but not tour lens
#     tour_lens = data["tour_lens"] if "tour_lens" in data.keys() else None
#     remaining_capacities = data["remaining_capacities"] if "remaining_capacities" in data.keys() else None
#     via_depots = data["via_depots"] if "via_depots" in data.keys() else None

#     collate_fn = collate_func_with_sample if what == "train" else None

#     dataset = DataLoader(DataSet(node_coords, demands, capacities,
#                                  remaining_capacities=remaining_capacities,
#                                  tour_lens=tour_lens,
#                                  via_depots=via_depots), batch_size=batch_size,
#                          drop_last=False, shuffle=shuffle, collate_fn=collate_fn)
#     return dataset

# =========================
# Public API
# =========================
def load_dataset(filename, batch_size, shuffle=False, what="test", min_nodes=0, max_nodes=10**9):
    """
    - 训练：保持原 .npz 路径不变（带 collate）
    - 验证/测试：
        * 目录：递归收集 .vrp/.VRP，按 [min_nodes, max_nodes) 过滤并升序排序，CVRPLIBDataSet
        * 单文件 .vrp：CVRPLIBDataSet([filename])
        * 其他：回退到 .npz 测试路径（向后兼容）
    """
    if what == "train":
        data = np.load(filename)
        assert data["reorder"]

        node_coords = data["coords"]
        demands = data["demands"]
        capacities = data["capacities"]

        tour_lens = data["tour_lens"] if "tour_lens" in data.keys() else None
        remaining_capacities = data["remaining_capacities"] if "remaining_capacities" in data.keys() else None
        via_depots = data["via_depots"] if "via_depots" in data.keys() else None

        collate_fn = collate_func_with_sample
        dataset = DataLoader(
            DataSet(node_coords, demands, capacities,
                    remaining_capacities=remaining_capacities,
                    tour_lens=tour_lens,
                    via_depots=via_depots),
            batch_size=batch_size, drop_last=False, shuffle=shuffle,
            collate_fn=collate_fn, num_workers=0
        )
        return dataset

    # 验证/测试：目录（优先 VRPLIB）
    if os.path.isdir(filename):
        vrp_files = _collect_vrp_files_filtered(filename, min_nodes=min_nodes, max_nodes=max_nodes)
        if len(vrp_files) > 0:
            return DataLoader(
                CVRPLIBDataSet(vrp_files),
                batch_size=batch_size, drop_last=False, shuffle=shuffle, collate_fn=None, num_workers=0
            )

    # 验证/测试：单文件 .vrp
    if isinstance(filename, str) and filename.lower().endswith(".vrp"):
        return DataLoader(
            CVRPLIBDataSet([filename]),
            batch_size=batch_size, drop_last=False, shuffle=shuffle, collate_fn=None, num_workers=0
        )

    # 回退：.npz 测试（向后兼容）
    data = np.load(filename)
    node_coords = data["coords"]
    demands = data["demands"]
    capacities = data["capacities"]
    tour_lens = data["tour_lens"] if "tour_lens" in data.keys() else None
    remaining_capacities = data["remaining_capacities"] if "remaining_capacities" in data.keys() else None
    via_depots = data["via_depots"] if "via_depots" in data.keys() else None

    dataset = DataLoader(
        DataSet(node_coords, demands, capacities,
                remaining_capacities=remaining_capacities,
                tour_lens=tour_lens,
                via_depots=via_depots),
        batch_size=batch_size, drop_last=False, shuffle=shuffle, collate_fn=None, num_workers=0
    )
    return dataset



def collate_func_with_sample(l_dataset_items):
    """
    assemble minibatch out of dataset examples.
    For instances of TOUR-CVRP of graph size N (i.e. nb_nodes=N+1 including return to beginning node),
    this function also takes care of sampling a SUB-problem (PATH-TSP) of size 3 to N+1
    """
    nb_nodes = len(l_dataset_items[0].dist_matrices)
    begin_idx = np.random.randint(0, nb_nodes - 3)  # between _ included and nb_nodes + 1 excluded

    l_dataset_items_new = []
    for d in l_dataset_items:
        d_new = {}
        for k, v in d.items():
            if k == "dist_matrices":
                v_ = v[begin_idx:, begin_idx:]
            elif k == "remaining_capacities":
                v_ = v[begin_idx]
            elif k == "capacities":
                v_ = v
            else:
                v_ = v[begin_idx:, ...]

            d_new.update({k + '_s': v_})
        l_dataset_items_new.append({**d, **d_new})

    return default_collate(l_dataset_items_new)

