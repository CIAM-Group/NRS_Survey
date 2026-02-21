import os
import warnings
from multiprocessing import Pool

import numpy as np
import scipy.sparse
import scipy.spatial
import torch
from diffusion.utils.cython_merge.cython_merge import merge_cython

import math

# ! 取整
# 量化/取整助手：与 TSPLIB 口径一致
def apply_tsplib_rounding(dist_tensor_or_np, ewt_code_or_str):
    """
    输入：torch.Tensor 或 numpy.ndarray（任意形状），表示欧氏距离
    ewt: 0/'continuous' -> 不取整；1/'EUC_2D' -> floor(x+0.5)；2/'CEIL_2D' -> ceil(x)
    输出：与输入同类型/同形状
    """
    # 规范 ewt
    if isinstance(ewt_code_or_str, (int, np.integer)):
        code = int(ewt_code_or_str)
    else:
        s = str(ewt_code_or_str).upper()
        code = 1 if s == "EUC_2D" else 2 if s == "CEIL_2D" else 0

    # torch/numpy 统一处理
    is_torch = isinstance(dist_tensor_or_np, torch.Tensor)
    x = dist_tensor_or_np
    if code == 1:  # EUC_2D
        return torch.floor(x + 0.5) if is_torch else np.floor(x + 0.5)
    elif code == 2:  # CEIL_2D
        return torch.ceil(x) if is_torch else np.ceil(x)
    else:
        return x


# ! 修复，判断写在开头，保证代数默认值正确
def batched_two_opt_torch(points, tour, max_iterations=1000, device="cpu"):
    iterator = 0
    tour = tour.copy()
    with torch.inference_mode():
        cuda_points = torch.from_numpy(points).to(device)
        cuda_tour = torch.from_numpy(tour).to(device)
        batch_size = cuda_tour.shape[0]
        # min_change = -1.0
        # while min_change < 0.0:
        while True:
            # 提前判断：达到/超过最大迭代次数则立刻退出（当 max_iterations==0 时不会进入任何一次迭代）
            if iterator >= max_iterations:
                break

            points_i = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape(
                (batch_size, -1, 1, 2)
            )
            points_j = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape(
                (batch_size, 1, -1, 2)
            )
            points_i_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape(
                (batch_size, -1, 1, 2)
            )
            points_j_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape(
                (batch_size, 1, -1, 2)
            )

            A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
            A_i_plus_1_j_plus_1 = torch.sqrt(
                torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1)
            )
            A_i_i_plus_1 = torch.sqrt(
                torch.sum((points_i - points_i_plus_1) ** 2, axis=-1)
            )
            A_j_j_plus_1 = torch.sqrt(
                torch.sum((points_j - points_j_plus_1) ** 2, axis=-1)
            )

            change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
            valid_change = torch.triu(change, diagonal=2)

            # min_change = torch.min(valid_change)
            min_change = float(torch.min(valid_change).item())

            flatten_argmin_index = torch.argmin(
                valid_change.reshape(batch_size, -1), dim=-1
            )
            min_i = torch.div(flatten_argmin_index, len(points), rounding_mode="floor")
            min_j = torch.remainder(flatten_argmin_index, len(points))

            # 若存在可改进的 2-opt 翻转，则执行；否则退出
            if min_change < -1e-6:
                for i in range(batch_size):
                    cuda_tour[i, min_i[i] + 1 : min_j[i] + 1] = torch.flip(
                        cuda_tour[i, min_i[i] + 1 : min_j[i] + 1], dims=(0,)
                    )
                iterator += 1
            else:
                break

            # if iterator >= max_iterations:
            #     break
        tour = cuda_tour.cpu().numpy()
    return tour, iterator


def numpy_merge(points, adj_mat):
    dists = np.linalg.norm(points[:, None] - points, axis=-1)

    components = np.zeros((adj_mat.shape[0], 2)).astype(int)
    components[:] = np.arange(adj_mat.shape[0])[..., None]
    real_adj_mat = np.zeros_like(adj_mat)
    merge_iterations = 0
    for edge in (-adj_mat / dists).flatten().argsort():
        merge_iterations += 1
        a, b = edge // adj_mat.shape[0], edge % adj_mat.shape[0]
        if not (a in components and b in components):
            continue
        ca = np.nonzero((components == a).sum(1))[0][0]
        cb = np.nonzero((components == b).sum(1))[0][0]
        if ca == cb:
            continue
        cca = sorted(components[ca], key=lambda x: x == a)
        ccb = sorted(components[cb], key=lambda x: x == b)
        newc = np.array([[cca[0], ccb[0]]])
        m, M = min(ca, cb), max(ca, cb)
        real_adj_mat[a, b] = 1
        components = np.concatenate(
            [components[:m], components[m + 1 : M], components[M + 1 :], newc], 0
        )
        if len(components) == 1:
            break
    real_adj_mat[components[0, 1], components[0, 0]] = 1
    real_adj_mat += real_adj_mat.T
    return real_adj_mat, merge_iterations


def cython_merge(points, adj_mat):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        real_adj_mat, merge_iterations = merge_cython(
            points.astype("double"), adj_mat.astype("double")
        )
        real_adj_mat = np.asarray(real_adj_mat)
    return real_adj_mat, merge_iterations


def merge_tours(
    adj_mat, np_points, edge_index_np, sparse_graph=False, parallel_sampling=1
):
    """
    To extract a tour from the inferred adjacency matrix A, we used the following greedy edge insertion
    procedure.
    • Initialize extracted tour with an empty graph with N vertices.
    • Sort all the possible edges (i, j) in decreasing order of Aij/kvi − vjk (i.e., the inverse edge weight,
    multiplied by inferred likelihood). Call the resulting edge list (i1, j1),(i2, j2), . . . .
    • For each edge (i, j) in the list:
      – If inserting (i, j) into the graph results in a complete tour, insert (i, j) and terminate.
      – If inserting (i, j) results in a graph with cycles (of length < N), continue.
      – Otherwise, insert (i, j) into the tour.
    • Return the extracted tour.
    """
    splitted_adj_mat = np.split(adj_mat, parallel_sampling, axis=0)
    if not sparse_graph:
        splitted_adj_mat = [adj_mat[0] + adj_mat[0].T for adj_mat in splitted_adj_mat]
    else:
        splitted_adj_mat = [
            scipy.sparse.coo_matrix(
                (adj_mat, (edge_index_np[0], edge_index_np[1])),
            ).toarray()
            + scipy.sparse.coo_matrix(
                (adj_mat, (edge_index_np[1], edge_index_np[0])),
            ).toarray()
            for adj_mat in splitted_adj_mat
        ]
    splitted_points = [np_points for _ in range(parallel_sampling)]
    if np_points.shape[0] > 1000 and parallel_sampling > 1:
        with Pool(parallel_sampling) as p:
            results = p.starmap(
                cython_merge,
                zip(splitted_points, splitted_adj_mat),
            )
    else:
        results = [
            cython_merge(_np_points, _adj_mat)
            for _np_points, _adj_mat in zip(splitted_points, splitted_adj_mat)
        ]
    splitted_real_adj_mat, splitted_merge_iterations = zip(*results)

    tours = []
    for i in range(parallel_sampling):
        tour = [0]
        while len(tour) < splitted_adj_mat[i].shape[0] + 1:
            n = np.nonzero(splitted_real_adj_mat[i][tour[-1]])[0]
            if len(tour) > 1:
                n = n[n != tour[-2]]
            tour.append(n.max())
        tours.append(tour)

    merge_iterations = np.mean(splitted_merge_iterations)
    return tours, merge_iterations


class TSPEvaluator(object):
    # def __init__(self, points):
    #     self.dist_mat = scipy.spatial.distance_matrix(points, points)

    def __init__(self, points, edge_weight_type=None):
        self.dist_mat = scipy.spatial.distance_matrix(points, points)
        if edge_weight_type is not None:
            # 允许传 str 或 code
            self.dist_mat = apply_tsplib_rounding(self.dist_mat, edge_weight_type)

    def evaluate(self, route):
        total_cost = 0
        for i in range(len(route) - 1):
            total_cost += self.dist_mat[route[i], route[i + 1]]
        return total_cost
