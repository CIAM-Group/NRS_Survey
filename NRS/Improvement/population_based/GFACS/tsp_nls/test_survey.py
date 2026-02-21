import os
import time
import logging
import random

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from net import Net
from aco import ACO
from utils import load_tsplib_dataset
from utils import load_tsplib_from_dir
from utils import load_tsplib_handles, prepare_instance_fullmatrix, build_pyg_from_coords, tsplib_integer_distance
from utils import tsp_survey_bench_cost_all as tsplib_cost

# ! 增加log
EPS = 1e-10
START_NODE = None

# ---------------- Logger ----------------
def setup_logger(save_dir: str, desc: str = "survey"):
    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(save_dir, f"{ts}_{desc}.log")

    logger = logging.getLogger("tsp_survey")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False  # ← 防止重复打印到 root

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"Logger initialized. Log file: {log_path}")
    return logger, log_path


@torch.no_grad()
def infer_instance(model, pyg_data, distances, n_ants, t_aco_diff):
    if model is not None:
        model.eval()
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS
    else:
        heu_mat = None

    aco = ACO(
        distances.cpu(),
        n_ants,
        heuristic=heu_mat.cpu() if heu_mat is not None else heu_mat,
        device='cpu',
        local_search_type='nls',
    )

    results = torch.zeros(size=(len(t_aco_diff),))
    elapsed_time = 0
    # last_path = None
    for i, t in enumerate(t_aco_diff):
        # ! 还原原版
        cost, _, t = aco.run(t, start_node=START_NODE)
        # cost, _, t, path = aco.run_without_ls_last(t, start_node=START_NODE)
        results[i] = cost
        elapsed_time += t
        # last_path = path

    # return results, last_path, elapsed_time
    return results, aco.shortest_path, elapsed_time


# @torch.no_grad()
# def test(dataset, model, n_ants, t_aco):
#     _t_aco = [0] + t_aco
#     t_aco_diff = [_t_aco[i + 1] - _t_aco[i] for i in range(len(_t_aco) - 1)]

#     results_list = []
#     best_paths = []
#     sum_times = 0
#     # ! 不需要复原 TSPlib 的坐标了，直接用整数距离矩阵算
#     for (pyg_data, distances) in tqdm(dataset):
#         ceiled_distances = distances # !这里不需要原版的ceil了，数据集里本来就是整数距离矩阵
#         results, best_path, elapsed_time = infer_instance(model, pyg_data, ceiled_distances, n_ants, t_aco_diff)
#         results_list.append(results)
#         best_paths.append(best_path)
#         sum_times += elapsed_time
#     return results_list, best_paths, sum_times / len(dataset)


# def make_tsplib_data(filename, episode):
#     instance_data = []
#     cost = []
#     instance_name = []
#     for line in open(filename, "r").readlines()[episode: episode + 1]:
#         line = line.rstrip("\n")
#         line = line.replace('[', '')
#         line = line.replace(']', '')
#         line = line.replace('\'', '')
#         line = line.split(sep=',')

#         line_data = np.array(line[2:], dtype=float).reshape(-1, 2)
#         instance_data.append(line_data)
#         cost.append(np.array(line[1], dtype=float))
#         instance_name.append(np.array(line[0], dtype=str))
#     instance_data = np.array(instance_data)  
#     cost = np.array(cost)
#     instance_name = np.array(instance_name)
    
#     return instance_data, cost, instance_name


def main(ckpt_path, k_sparse_factor=10, n_ants=100, n_iter=10, guided_exploration=False, seed=0):
    # ---------- logger ----------
    result_root = './tsp_nls/survey_results'
    logger, log_path = setup_logger(result_root, desc="test_TSPLIB_Survey")
    
    # ! 去掉n_nodes参数
    # test_list, scale_list, name_list = load_tsplib_dataset(n_nodes, k_sparse_factor, DEVICE, start_node=START_NODE)
    # ! 修改读数据
    tsplib_dir = '/public/home/bayp/exp_survey_202509/0_data_survey/survey_bench_tsp'

    # 可选：如果你也想像 ICAM 那样分段测试，可以传 dim_ranges
    # 比如只测 [100,300)：
    # dim_ranges = [[100, 300]]
    # dim_ranges = None  # 不过滤
    # ! 先测0-1000
    # dim_ranges = [(0, 1000)]
    # dim_ranges = [(7397, 7398)]

    # ! 然后测1000-10000
    # dim_ranges = [(1000, 10000)]
    dim_ranges = [(0, 1000), (1000, 10000), (10000, 100001)]




    handles = load_tsplib_handles(tsplib_dir, dim_ranges=dim_ranges)
    if len(handles) == 0:
        logger.error("No instances found under given dim_ranges.")
        return

    dims = [h["dim"] for h in handles]
    if len(set(dims)) == 1:
        dim_tag = f"dim{dims[0]}"
    else:
        dim_tag = f"dim{min(dims)}-{max(dims)}"

    # ---------- env/model meta ----------
    logger.info(f"checkpoint: {ckpt_path}")
    logger.info(f"instances: {len(handles)}, dim range: {min(dims)}–{max(dims)} (median {int(np.median(dims))})")
    logger.info(f"device: {'cpu' if DEVICE == 'cpu' else DEVICE + '+cpu'}")
    logger.info(f"n_ants: {n_ants}")
    logger.info(f"n_iter: {n_iter}")
    logger.info(f"k_sparse_factor: {k_sparse_factor}")
    logger.info(f"guided_exploration: {guided_exploration}")
    logger.info(f"seed: {seed}")

    # ! 固定随机种子
    torch.manual_seed(seed); np.random.seed(seed)

    if ckpt_path is not None and os.path.isfile(ckpt_path):
        net = Net(gfn=True, Z_out_dim=2 if guided_exploration else 1, start_node=START_NODE).to(DEVICE)
        net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        logger.info("Model loaded successfully.")
    else:
        logger.info("No checkpoint provided. Running ACO without neural heuristic.")

    # ---------- run (逐实例 try/except) ----------
    t_aco = list(range(1, n_iter + 1))
    _t = [0] + t_aco
    t_aco_diff = [_t[i+1] - _t[i] for i in range(len(_t)-1)]

    rows = []  # (name, dim, length, optimal, gap or None)
    gaps_all = []
    gaps_lt_1k, gaps_lt_10k, gaps_lt_100k = [], [], []
    times_lt_1k, times_lt_10k, times_lt_100k = [], [], []

    solved, total = 0, 0
    total_elapsed = 0.0

    # for h in tqdm(handles):
    for h in handles:
        total += 1
        name, dim = h["name"], h["dim"]
        try:
            # 1) 准备（保持“全矩阵”）
            print (f"Testing instance {name}, dim={dim} ...")

            # 1) 计时起点：包含“距离矩阵/构图”的准备时间
            t0 = time.perf_counter()

            # ! 手动归一化处理，替代 prepare_instance_fullmatrix
            coords = h["coords"].to(DEVICE)
            edge_typ = h["edge_type"]
            
            # 归一化：min-max (global)
            min_val = coords.min()
            max_val = coords.max()
            scale = max_val - min_val
            if scale == 0: scale = 1.0
            coords_norm = (coords - min_val) / scale
            
            # 构建 PyG 数据 (使用归一化坐标)
            pyg_data, _ = build_pyg_from_coords(coords_norm, k_sparse=k_sparse_factor, start_node=START_NODE)
            
            # 计算距离矩阵 (使用原始坐标，保持原始精度和取整规则)
            dist_int = tsplib_integer_distance(coords, edge_typ)

            # pyg_data, dist_int = prepare_instance_fullmatrix(
            #     h, k_sparse_factor=k_sparse_factor, device=DEVICE, start_node=START_NODE
            # )

            # 2) 推理
            _, path, _ = infer_instance(net, pyg_data, dist_int, n_ants, t_aco_diff)

            t1 = time.perf_counter()
            per_inst_elapsed = t1 - t0

            total_elapsed += per_inst_elapsed

            # 3) 计分（直接查整张矩阵）
            tour_len = 0
            for i in range(len(path)):
                u = path[i]
                v = path[(i + 1) % len(path)]
                tour_len += int(dist_int[u, v].item())

            # 4) gap
            optimal = tsplib_cost.get(name, None)
            if optimal is None:
                logger.warning(f"[Skip GAP] optimal not found for {name}")
                rows.append((name, dim, tour_len, None, None))
                solved += 1  # 算已求解，只是没有 label
                logger.info(f"Instance {name}, dim={dim}, length={tour_len}, optimal=NA, gap=NA")
                continue

            gap = (tour_len - optimal) * 100.0 / optimal
            rows.append((name, dim, tour_len, optimal, gap))
            gaps_all.append(gap)

            if dim < 1000: 
                gaps_lt_1k.append(gap)
                times_lt_1k.append(per_inst_elapsed)
            elif dim < 10000: 
                gaps_lt_10k.append(gap)
                times_lt_10k.append(per_inst_elapsed)
            elif dim <= 100000: 
                gaps_lt_100k.append(gap)
                times_lt_100k.append(per_inst_elapsed)

            solved += 1
            logger.info(
                f"Instance {name}, dim={dim}, length={tour_len}, optimal={optimal}, "
                f"gap={gap:.3f}%, time={per_inst_elapsed:.4f}s"
            )

        except (Exception) as e:
            msg = str(e)
            if "out of memory" in msg.lower() or isinstance(e, MemoryError):
                logger.warning(f"[Skip OOM] instance {name}, dim={dim}. {msg}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                logger.warning(f"[Skip Error] instance {name}, dim={dim}. {msg}")
            continue

    # avg_infer_time = total_elapsed / max(1, solved)
    
    logger.info(f"All scale ranges done, solved instance number: {solved}/{total}, "
                # f"total time: {total_elapsed:.2f}s, avg time per instance: {avg_infer_time:.4f}s"
                )



    # ---------- aggregate logs (ICAM 风格) ----------
    def _avg(xs): 
        return float(np.mean(xs)) if len(xs) > 0 else 0.0

    logger.info("#################  Summary by scale  #################")
    logger.info(f"[0, 1000):      num={len(gaps_lt_1k)},     avg gap={_avg(gaps_lt_1k):.3f}%,  avg time={_avg(times_lt_1k):.4f}s")
    logger.info(f"[1000, 10000):  num={len(gaps_lt_10k)},    avg gap={_avg(gaps_lt_10k):.3f}%,  avg time={_avg(times_lt_10k):.4f}s")
    logger.info(f"[10000, 100000]:num={len(gaps_lt_100k)},  avg gap={_avg(gaps_lt_100k):.3f}%,  avg time={_avg(times_lt_100k):.4f}s")
    logger.info("#################  All Done  #################")
    logger.info(f"All solved instances: {len(gaps_all)}, avg gap={_avg(gaps_all):.3f}%")

if __name__ == "__main__":
    # 固定参数
    ckpt_path = "./pretrained/tsp_nls/200/tsp200_sd0/50.pt"  # 模型文件路径，或者 None
    # n_nodes = 200           # TSP 节点数
    k_sparse_factor = 10    # 稀疏化参数
    n_ants = 100            # 蚂蚁数量
    n_iter = 100             # 迭代次数
    guided_exploration = True   # 是否开启 guided exploration
    seed = 0                # 随机种子
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    main(
        ckpt_path,
        k_sparse_factor,
        n_ants,
        n_iter,
        guided_exploration,
        seed
    )
