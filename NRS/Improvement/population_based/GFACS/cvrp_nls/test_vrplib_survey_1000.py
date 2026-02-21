from typing import Tuple, List
import os
import random
import time

# from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

import logging
from logging import getLogger
from logging import Handler
import sys
from datetime import datetime
from net import Net
from aco import ACO, get_subroutes
from utils import load_vrplib_dataset
from utils import (
    load_cvrplib_handles,
    prepare_instance_fullmatrix_cvrp,
)

EPS = 1e-10
START_NODE = 0  # depot

# ! 新增log
LOG_NAME = "cvrp_survey"
logger = logging.getLogger(LOG_NAME)

def init_logger(save_dir: str, desc: str = "") -> str:
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(
        save_dir, f"cvrp_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{desc}.log"
    )

    logger = logging.getLogger(LOG_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # 文件
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 终端
    th = logging.StreamHandler(sys.stdout)
    th.setLevel(logging.INFO)
    th.setFormatter(fmt)
    logger.addHandler(th)

    logger.info(f"Logger initialized, log file at {log_file}")
    return log_file

def validate_route(int_distances: np.ndarray, demands: torch.Tensor, routes: List[torch.Tensor]) -> Tuple[bool, int]:
    length = 0
    valid = True
    visited = {0}
    for r in routes:
        d = demands[r].sum().item()
        if d>1.000001:
            valid = False
        length += int_distances[r[:-1], r[1:]].sum()
        for i in r:
            i = i.item()
            if i<0 or i >= int_distances.shape[0]:
                valid = False
            else:
                visited.add(i)  # type: ignore
    if len(visited) != int_distances.shape[0]:
        valid = False
    return valid, length


@torch.no_grad()
def infer_instance(model, pyg_data, demands, distances, positions, n_ants, t_aco_diff, k_sparse_factor, int_distances):
    if model is not None:
        model.eval()
        heu_vec = model(pyg_data)
        heu_mat = model.reshape(pyg_data, heu_vec) + EPS
    else:
        heu_mat = None

    k_sparse = positions.shape[0] // k_sparse_factor
    aco = ACO(
        distances=distances.cpu(),
        demand=demands.cpu(),
        positions=positions.cpu(),
        n_ants=n_ants,
        heuristic=heu_mat.cpu() if heu_mat is not None else heu_mat,
        k_sparse=k_sparse,
        elitist=ACOALG == "ELITIST",
        maxmin=ACOALG == "MAXMIN",
        rank_based=ACOALG == "RANK",
        device='cpu',
        local_search_type="nls",
    )

    results = torch.zeros(size=(len(t_aco_diff),), dtype=torch.int64)
    elapsed_time = 0
    for i, t in enumerate(t_aco_diff):
        _, _, t = aco.run(t)
        path = get_subroutes(aco.shortest_path)
        valid, length = validate_route(int_distances, demands, path)  # use int_distances here
        if valid is False:
           print("invalid solution.")
        results[i] = length
        elapsed_time += t
    print("elapsed_time in infer_instance:", elapsed_time)
    return results, elapsed_time


@torch.no_grad()
def test(dataset, model, n_ants, t_aco, k_sparse_factor, int_dist_list):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i + 1] - _t_aco[i] for i in range(len(_t_aco) - 1)]

    results_list = []
    times = []
    for (pyg_data, demands, distances, positions), int_distances in zip(dataset, int_dist_list):
        results, elapsed_time = infer_instance(
            model, pyg_data, demands, distances, positions, n_ants, t_aco_diff, k_sparse_factor, int_distances
        )
        results_list.append(results)
        times.append(elapsed_time)
    return results_list, times


def main(ckpt_path, cvrplib_dir, k_sparse_factor, n_ants=100, n_iter=10, guided_exploration=False, seed=0):
    # ---------- logger ----------
    # result_root = './cvrp_nls/survey_results'
    # log_path = init_logger(result_root, desc="test_CVRPLIB_Survey")
    logger = logging.getLogger(LOG_NAME)

    # 读轻量句柄（逐实例、带 label）
    # handles = load_cvrplib_handles(cvrplib_dir, dim_ranges=None)
    # 刚上10000时，第一个15000就kill了
    # handles = load_cvrplib_handles(cvrplib_dir, dim_ranges=[(10000, 100001)])
    # 尝试[10000, 15000]的
    # handles = load_cvrplib_handles(cvrplib_dir, dim_ranges=[(10000, 15000)])
    handles = load_cvrplib_handles(cvrplib_dir, dim_ranges=[(0, 100001)])
    if len(handles) == 0:
        logger.error("No .vrp instances found under given directory.")
        return

    dims = [h["dim"] for h in handles]
    logger.info(f"instances: {len(handles)}, dim range: {min(dims)}–{max(dims)} (median {int(np.median(dims))})")

    # ---------- env/model meta ----------
    global DEVICE, ACOALG
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"checkpoint: {ckpt_path}")
    logger.info(f"device: {'cpu' if DEVICE == 'cpu' else DEVICE + '+cpu'}")
    logger.info(f"n_ants: {n_ants}")
    logger.info(f"seed: {seed}")

    # 固定随机种子
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # 模型（可选）
    net = None
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        net = Net(gfn=True, Z_out_dim=2 if guided_exploration else 1).to(DEVICE)
        net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        logger.info("Model loaded successfully.")
    else:
        logger.info("No checkpoint provided. Running ACO without neural heuristic.")

    # 迭代分段（与 TSP 一致的增量跑法）
    t_aco = list(range(1, n_iter + 1))
    _t = [0] + t_aco
    t_aco_diff = [_t[i + 1] - _t[i] for i in range(len(_t) - 1)]

    # 汇总器
    def _avg(xs): return float(np.mean(xs)) if len(xs) > 0 else 0.0
    total_elapsed = 0.0
    solved, total = 0, 0
    gaps_all = []
    gaps_lt_1k, gaps_lt_10k, gaps_lt_100k = [], [], []
    times_lt_1k, times_lt_10k, times_lt_100k = [], [], []

    # 规模分段（与 ICAM 一致）
    scale_ranges = [(0, 1000), (1000, 10000), (10000, 100001)]
    for lo, hi in scale_ranges:
        logger.info(f"#################  Test scale range: [{lo},{hi})  #################")
        start_range = time.time()
        count_range = 0
        gaps_this = []
        times_this = []

        handles_in_range = [h for h in handles if lo <= h["dim"] < hi]
        handles_in_range.sort(key=lambda x: x["dim"])

        for h in handles_in_range:
            total += 1
            name, dim = h["name"], h["dim"]
            
            if name != "X-n916-k207":
                continue
            
            # ! 打印当前测试实例
            logger.info(f"Current test instance: {name}, number of nodes: {dim}")
            try:

                t0 = time.perf_counter()

                # 逐实例：构图（归一化口径）+ 整数口径全矩阵
                pyg_data, demand_norm, dist_norm, positions, int_dist = prepare_instance_fullmatrix_cvrp(
                    h, k_sparse_factor=k_sparse_factor, device=DEVICE
                )

                # 推理（保持原 infer_instance 行为）
                results, _ = infer_instance(
                    net, pyg_data, demand_norm, dist_norm, positions, n_ants, t_aco_diff, k_sparse_factor, int_dist
                )

                t1 = time.perf_counter()
                per_inst_elapsed = t1 - t0

                total_elapsed += per_inst_elapsed
                length = results[-1].item()

                optimal = h["optimal"]
                if optimal is None:
                    logger.warning(f"[Skip GAP] optimal not found for {name}")
                    solved += 1; count_range += 1
                    logger.info(f"Instance {name}, dim={dim}, length={length}, optimal=NA, gap=NA")
                    continue

                gap = (length - optimal) * 100.0 / optimal
                solved += 1; count_range += 1
                gaps_this.append(gap); gaps_all.append(gap)

                times_this.append(per_inst_elapsed)

                if dim < 1000: 
                    gaps_lt_1k.append(gap)
                    times_lt_1k.append(per_inst_elapsed)
                elif dim < 10000: 
                    gaps_lt_10k.append(gap)
                    times_lt_10k.append(per_inst_elapsed)
                elif dim <= 100000: 
                    gaps_lt_100k.append(gap)
                    times_lt_100k.append(per_inst_elapsed)

                logger.info(
                    f"Instance {name}, dim={dim}, length={length}, optimal={optimal}, "
                    f"gap={gap:.3f}%, time={per_inst_elapsed:.4f}s)"
                )


            except Exception as e:
                msg = str(e)
                logger.warning(f"[Skip Error] instance {name}, dim={dim}. {msg}")
                continue

        dt_range = time.time() - start_range
        logger.info(" *** Test Done *** ")
        logger.info(f"scale_range: [{lo},{hi}), instance number: {count_range}, total time: {dt_range:.2f}s, "
                    # f"avg time per instance: {dt_range / max(1, count_range):.2f}s"
                    )
        if count_range > 0:
            logger.info(f"avg gap (this range): {_avg(gaps_this):.3f}%"
                        f", avg time (this range): {_avg(times_this):.4f}s")
        logger.info("===============================================================")

    avg_infer_time = total_elapsed / max(1, solved)

    logger = getLogger("cvrp_survey")
    logger.info(f"All scale ranges done, solved instance number: {solved}/{total}, "
                f"total time: {total_elapsed:.2f}s, avg time per instance: {avg_infer_time:.4f}s")

    logger.info("#################  Summary by scale  #################")
    logger.info(f"[0, 1000):      num={len(gaps_lt_1k)},     avg gap={_avg(gaps_lt_1k):.3f}%,  avg time={_avg(times_lt_1k):.4f}s")
    logger.info(f"[1000, 10000):  num={len(gaps_lt_10k)},    avg gap={_avg(gaps_lt_10k):.3f}%,  avg time={_avg(times_lt_10k):.4f}s")
    logger.info(f"[10000, 100000]:num={len(gaps_lt_100k)},  avg gap={_avg(gaps_lt_100k):.3f}%,  avg time={_avg(times_lt_100k):.4f}s")
    logger.info("#################  All Done  #################")
    logger.info(f"All solved instances: {len(gaps_all)}, avg gap={_avg(gaps_all):.3f}%")





if __name__ == "__main__":

    # 固定参数（按需改成你的本地路径/设置）
    ckpt_path = "/public/home/bayp/exp_survey_202509/Improvement/population_based/GFACS/pretrained/cvrp_nls/1000/cvrp1000_sd0_fromckpt-500_cvrp500_sd0_50/20.pt"   # 模型文件路径，或 None
    cvrplib_dir = "/public/home/bayp/exp_survey_202509/0_data_survey/survey_bench_cvrp"  # CVRPLIB 根目录
    k_sparse_factor = 5        # 稀疏化系数：k = (n+1) // k_sparse_factor
    n_ants = 100                # 蚂蚁数量
    n_iter = 100                 # ACO 迭代次数
    guided_exploration = True   # 是否开启 guided exploration（决定 Z_out_dim）
    seed = 1234 #0                    # 随机种子

    # 设备 & ACO 算法（保持和原版一致的全局变量）
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    ACOALG = "AS"               # 也可设为: "ELITIST" / "MAXMIN" / "RANK"

    # 固定随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ---------- logger ----------
    result_root = '/public/home/bayp/exp_survey_202509/Improvement/population_based/GFACS/cvrp_nls/survey_results'
    log_path = init_logger(result_root, desc="test_CVRPLIB_Survey")
    logger = logging.getLogger(LOG_NAME)

    logger.info("===== Start CVRP Testing =====")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Dataset dir: {cvrplib_dir}")
    logger.info(f"n_ants={n_ants}, n_iter={n_iter}, k_sparse_factor={k_sparse_factor}, seed={seed}")

    # 跑起来
    main(
        ckpt_path=ckpt_path,
        cvrplib_dir=cvrplib_dir,
        k_sparse_factor=k_sparse_factor,
        n_ants=n_ants,
        n_iter=n_iter,
        guided_exploration=guided_exploration,
        seed=seed,
    )
