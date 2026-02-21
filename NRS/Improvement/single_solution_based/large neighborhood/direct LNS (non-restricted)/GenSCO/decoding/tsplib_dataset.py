# decoding/tsplib_dataset.py

import os
import numpy as np

from LIBUtils import TSPLIBReader, tsplib_cost


def _build_tsplib_dist_mat(coords: np.ndarray, edge_weight_type: str) -> np.ndarray:
    """
    coords: (N, 2) 原始 TSPLIB 坐标（未归一化）
    edge_weight_type: 'EUC_2D' 或 'CEIL_2D' 等

    返回: (N, N) 的距离矩阵（float32），按 TSPLIB 规则离散化
    """
    diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 2)
    d_raw = np.sqrt(np.sum(diff ** 2, axis=-1))     # (N, N)

    if edge_weight_type == "CEIL_2D":
        d = np.ceil(d_raw)
    elif edge_weight_type == "EUC_2D":
        # TSPLIB EUC_2D: floor(d + 0.5)
        d = np.floor(d_raw + 0.5)
    else:
        # 其他类型先当连续距离处理，后续需要可以再扩展
        d = d_raw

    return d.astype(np.float32)


def build_tsplib_dataset(root: str):
    """
    扫描 root 下所有 .tsp 文件，仿 ICAM 的 TSPLIBReader + tsplib_cost，
    构造一个可以直接喂给 GenSCO 的 dataset dict。

    返回的 dataset 结构：
        {
            'coords': (B, N_i, 2) 归一化坐标（用于模型）
            'coords_original': (B, N_i, 2) 原始坐标（可选，调试用）
            'dist_mat': (B, N_i, N_i) TSPLIB 度量离散后的距离矩阵（用于评估）
            'opt_costs': (B,) 最优值（来自 tsplib_cost 字典，没有则为 NaN）
            'edge_weight_type': (B,) 字符串数组
            'names': (B,) 实例名
        }
    注意：不同实例 N_i 可能不同，严格来说这是 ragged 的，这里只适用于
    “同一批 problem_size 一样”的情况（比如你先按维度过滤）。
    """
    coords_list = []
    coords_original_list = []
    dist_mat_list = []
    opt_costs = []
    edge_weight_types = []
    names = []

    for root_dir, _, files in os.walk(root):
        for fname in files:
            if not fname.lower().endswith(".tsp"):
                continue

            full_path = os.path.join(root_dir, fname)
            name, dimension, locs, edge_weight_type = TSPLIBReader(full_path)
            if name is None or dimension is None or locs is None:
                # 被过滤掉的类型（如不是 EUC_2D/CEIL_2D）
                continue

            coords_original = np.asarray(locs, dtype=np.float32)  # (N, 2)
            assert coords_original.shape == (int(dimension), 2)

            # === 按 ICAM 的方式做归一化（只用于模型输入） ===
            xy_max = coords_original.max(axis=0, keepdims=True)  # (1, 2)
            xy_min = coords_original.min(axis=0, keepdims=True)  # (1, 2)
            ratio = np.max(xy_max - xy_min)                      # 标量
            if ratio == 0:
                ratio = 1.0
            coords_norm = (coords_original - xy_min) / ratio     # (N, 2)

            # === 按 TSPLIB 规则在原始坐标上构造距离矩阵 ===
            dist_mat = _build_tsplib_dist_mat(coords_original, edge_weight_type)

            # === 最优 cost 来自 tsplib_cost 字典 ===
            opt = tsplib_cost.get(name, np.nan)

            coords_list.append(coords_norm[None, :, :])
            coords_original_list.append(coords_original[None, :, :])
            dist_mat_list.append(dist_mat[None, :, :])
            opt_costs.append(opt)
            edge_weight_types.append(edge_weight_type)
            names.append(name)

    if len(coords_list) == 0:
        raise RuntimeError(f"No valid TSPLIB .tsp files found under {root!r}")

    coords_arr = np.concatenate(coords_list, axis=0)
    coords_original_arr = np.concatenate(coords_original_list, axis=0)
    dist_mat_arr = np.concatenate(dist_mat_list, axis=0)
    opt_costs_arr = np.asarray(opt_costs, dtype=np.float32)
    edge_weight_types_arr = np.asarray(edge_weight_types, dtype=object)
    names_arr = np.asarray(names, dtype=object)

    dataset = {
        "coords": coords_arr,                  # (B, N, 2) 归一化坐标
        "coords_original": coords_original_arr, # (B, N, 2) 原始坐标
        "dist_mat": dist_mat_arr,              # (B, N, N) TSPLIB 距离矩阵
        "opt_costs": opt_costs_arr,            # (B,)
        "edge_weight_type": edge_weight_types_arr,  # (B,)
        "names": names_arr,                    # (B,)
    }
    return dataset
