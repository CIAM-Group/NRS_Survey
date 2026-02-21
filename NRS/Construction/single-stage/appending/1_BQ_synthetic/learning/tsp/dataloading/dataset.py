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
from tqdm import tqdm

# ! 新增：TSPLIB 工具（不归一化坐标）
from utils.tsplib import read_tsplib_tsp, tsplib_pairwise_dist, tsplib_cost
import os

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class DataSet(Dataset):

    def __init__(self, node_coords, tour_lens=None):
        self.node_coords = node_coords
        self.tour_lens = tour_lens

    def __len__(self):
        return len(self.node_coords)

    def __getitem__(self, item):
        node_coords = self.node_coords[item]
        dist_matrix = squareform(pdist(node_coords, metric='euclidean'))

        # From list to tensors as a DotDict
        item_dict = DotDict()
        item_dict.dist_matrices = torch.Tensor(dist_matrix)
        item_dict.nodes_coord = torch.Tensor(node_coords)
        if self.tour_lens is not None and len(self.tour_lens) > 0:
            item_dict.tour_len = self.tour_lens[item]
        else:
            item_dict.tour_len = torch.Tensor([])

        return item_dict

# ! 新增 TSPLIB 读取
class TSPLIBDataSet(Dataset):
    """
    逐 .tsp 文件一个样本：
      - nodes_coord:  (N,2)  —— 原始坐标（不归一化）
      - dist_matrices:(N,N)  —— TSPLIB 计费（EUC_2D round / CEIL_2D ceil）
      - tour_len:     ()     —— 来自 tsplib_cost[name]
    """
    def __init__(self, tsp_file_list):
        self.files = tsp_file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        name, N, coords, ewt = read_tsplib_tsp(path)
        if name is None:
            raise RuntimeError(f"Unsupported EDGE_WEIGHT_TYPE or parse error: {path}")

        # 距离矩阵（TSPLIB 规则，用原始坐标）
        dist_mat = tsplib_pairwise_dist(coords, ewt)  # [N,N]

        # 归一化坐标（ICAM 的输入规范）
        # (x - min) / max(range_x, range_y)，保持长宽比；避免除 0
        xy_min = coords.min(axis=0, keepdims=True)   # [1,2]
        xy_max = coords.max(axis=0, keepdims=True)   # [1,2]
        span = xy_max - xy_min                       # [1,2]
        ratio = np.maximum(span[:, :1], span[:, 1:2]) if span.ndim == 2 else np.max(span)  # 只是稳妥起见
        ratio = np.max(span)                         # 标准实现：max(x_range, y_range)
        if ratio == 0:
            ratio = 1.0
        nodes_norm = (coords - xy_min) / ratio       # [N,2] ∈ [0,1]

        # label（最优长度）
        if name not in tsplib_cost:
            raise KeyError(f"optimal cost not found for TSPLIB instance: {name}")
        optimal = float(tsplib_cost[name])

        item = DotDict()
        # item.nodes_coord = torch.tensor(coords, dtype=torch.float32)       # [N,2] 不归一化
        # item.nodes_coord = torch.tensor(nodes_norm, dtype=torch.float32)     # [N,2]
        item.nodes_coord = torch.tensor(nodes_norm, dtype=torch.float32)     # [N,2]
        item.dist_matrices = torch.tensor(dist_mat, dtype=torch.float32)   # [N,N]
        item.tour_len = torch.tensor(optimal, dtype=torch.float32)         # []
        return item


# def load_dataset(filename, batch_size, shuffle=False, what="test"):
#     data = np.load(filename)

#     if what == "train":
#         assert data["reorder"]

#     tour_lens = data["tour_lens"] if "tour_lens" in data.keys() else None

#     # Do not use collate function in test dataset
#     collate_fn = collate_func_with_sample_suffix if what == "train" else None

#     dataset = DataLoader(DataSet(data["coords"], tour_lens=tour_lens), batch_size=batch_size,
#                          drop_last=False, shuffle=shuffle, collate_fn=collate_fn)
#     return dataset

import fnmatch

def _read_tsplib_header_only(path):
    """只读 .tsp 头部拿 name/dimension/EDGE_WEIGHT_TYPE；遇到 NODE_COORD_SECTION 立即停。"""
    name = None
    dimension = None
    ewt = None
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("NAME"):
                toks = line.replace(":", " ").split()
                name = toks[-1]
            elif line.startswith("DIMENSION"):
                toks = line.replace(":", " ").split()
                try:
                    dimension = int(toks[-1])
                except Exception:
                    pass
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                toks = line.replace(":", " ").split()
                ewt = toks[-1]
            elif line.startswith("NODE_COORD_SECTION"):
                break
    return name, dimension, ewt

def _collect_tsp_files_filtered(root_dir, min_nodes=0, max_nodes=10**9):
    """
    递归收集 .tsp 文件，并按 DIMENSION 过滤 + 按规模从小到大排序。
    返回 [(path, dim), ...]
    """
    files_with_dim = []

    for root, _, fnames in os.walk(root_dir):
        for fn in fnames:
            if fnmatch.fnmatch(fn, "*.tsp") or fnmatch.fnmatch(fn, "*.TSP"):
                path = os.path.join(root, fn)
                name, dim, ewt = _read_tsplib_header_only(path)
                # 过滤不合法文件
                if name is None or dim is None or ewt not in {"EUC_2D", "CEIL_2D"}:
                    continue
                if min_nodes <= dim < max_nodes:
                    files_with_dim.append((path, dim))

    # ✅ 按 dim 从小到大排序
    files_with_dim.sort(key=lambda x: x[1])

    # 只返回路径列表
    files = [p for p, _ in files_with_dim]
    return files

def load_dataset(filename, batch_size, shuffle=False, what="test", min_nodes=0, max_nodes=10**9):
    # 训练分支保持不变 ...
    if what == "train":
        data = np.load(filename)
        assert data["reorder"]
        tour_lens = data["tour_lens"] if "tour_lens" in data.keys() else None
        collate_fn = collate_func_with_sample_suffix
        dataset = DataLoader(
            DataSet(data["coords"], tour_lens=tour_lens),
            batch_size=batch_size, drop_last=False, shuffle=shuffle, collate_fn=collate_fn,
            num_workers=0  # 训练你原来怎么配置就怎么来，这里稳妥起见也可 0
        )
        return dataset

    # 验证/测试：目录
    if os.path.isdir(filename):
        tsp_files = _collect_tsp_files_filtered(filename, min_nodes=min_nodes, max_nodes=max_nodes)
        if len(tsp_files) == 0:
            raise FileNotFoundError(
                f"No .tsp in range [{min_nodes},{max_nodes}] under: {filename}"
            )
        # ! 在这里算的距离矩阵
        dataset = DataLoader(
            TSPLIBDataSet(tsp_files),
            batch_size=batch_size, drop_last=False, shuffle=shuffle, collate_fn=None,
            num_workers=0  # 防止 DataLoader 子进程 OOM 时静默退出
        )
        return dataset

    # 验证/测试：单文件 .tsp
    if filename.lower().endswith(".tsp"):
        # 单文件不需要过滤，但也用 num_workers=0
        dataset = DataLoader(
            TSPLIBDataSet([filename]),
            batch_size=batch_size, drop_last=False, shuffle=shuffle, collate_fn=None,
            num_workers=0
        )
        return dataset

    # 其余走 .npz 测试路径（保持原样）
    data = np.load(filename)
    tour_lens = data["tour_lens"] if "tour_lens" in data.keys() else None
    dataset = DataLoader(
        DataSet(data["coords"], tour_lens=tour_lens),
        batch_size=batch_size, drop_last=False, shuffle=shuffle, collate_fn=None,
        num_workers=0
    )
    return dataset


def collate_func_with_sample_suffix(l_dataset_items):
    """
    assemble minibatch out of dataset examples.
    For instances of TOUR-TSP of graph size N (i.e. nb_nodes=N+1 including return to beginning node),
    this function also takes care of sampling a SUB-problem (PATH-TSP) of size 3 to N+1
    """
    nb_nodes = len(l_dataset_items[0].nodes_coord)
    subproblem_size = np.random.randint(4, nb_nodes + 1)
    begin_idx = nb_nodes + 1 - subproblem_size
    l_dataset_items_new = prepare_dataset_items(l_dataset_items, begin_idx, subproblem_size)
    return default_collate(l_dataset_items_new)

def prepare_dataset_items(l_dataset_items, begin_idx, subproblem_size):
    l_dataset_items_new = []
    for d in l_dataset_items:
        d_new = {}
        for k, v in d.items():
            if type(v) == numpy.float64:
                v_ = 0.
            elif len(v.shape) == 1 or k == 'nodes_coord':
                v_ = v[begin_idx:begin_idx+subproblem_size, ...]
            else:
                v_ = v[begin_idx:begin_idx+subproblem_size, begin_idx:begin_idx+subproblem_size]
            d_new.update({k+'_s': v_})
        l_dataset_items_new.append({**d, **d_new})
    return l_dataset_items_new


def sample_subproblem(nb_nodes):
    subproblem_size = np.random.randint(4, nb_nodes + 1)  # between _ included and nb_nodes + 1 excluded
    begin_idx = np.random.randint(nb_nodes - subproblem_size + 1)
    return begin_idx, subproblem_size

# ! synthetic dataset reader (LEHD style)
def load_dataset_txt(filename, batch_size, shuffle=False, what="test",episode = 10000,device=None):
    if what=='test':
        raw_data_nodes,raw_data_tours = load_raw_data(filename, episioe=episode)
        data={}
        data['coords'] = raw_data_nodes
        # data["tour_lens"] = np.ones(len(raw_data_nodes))*23.1199

        data["tour_lens"] = _get_travel_distance_2(np.array(raw_data_nodes)[:,:-1,:], np.array(raw_data_tours))
    else:
        data = np.load(filename)
        # data["reorder"] = True
    if what == "train":
        assert data["reorder"]

    tour_lens = data["tour_lens"] if "tour_lens" in data.keys() else None

    # Do not use collate function in test dataset
    collate_fn = collate_func_with_sample_suffix if what == "train" else None

    dataset = DataLoader(DataSet(data["coords"], tour_lens=tour_lens), batch_size=batch_size,
                         drop_last=False, shuffle=False, collate_fn=collate_fn)
    return dataset

def load_raw_data(filename, episioe=2):

    data_path = filename

    print('load raw dataset begin!')

    raw_data_nodes = []
    raw_data_tours = []
    for line in tqdm(open(data_path, "r").readlines()[:episioe], ascii=True):
        line = line.split(" ")
        num_nodes = int(line.index('output') // 2)
        nodes = [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)] + [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2, 2)]

        raw_data_nodes.append(nodes)
        tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]]#[:-1]
        raw_data_tours.append(tour_nodes)


    return raw_data_nodes,raw_data_tours


def _get_travel_distance_2(problems, solution):
    problems = torch.tensor(problems)
    solution = torch.tensor(solution,dtype=torch.int64)

    gathering_index = solution.unsqueeze(2).expand(problems.shape[0], problems.shape[1], 2)

    seq_expanded = problems

    ordered_seq = seq_expanded.gather(dim=1, index=gathering_index)

    rolled_seq = ordered_seq.roll(dims=1, shifts=-1)

    segment_lengths = ((ordered_seq - rolled_seq) ** 2)

    segment_lengths = segment_lengths.sum(2).sqrt()

    travel_distances = segment_lengths.sum(1)
    print(travel_distances.clone().detach().cpu().numpy())
    return travel_distances.clone().detach().cpu().numpy()

