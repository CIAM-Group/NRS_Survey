import os
import time
import math
from datetime import datetime

import numpy as np
import pytz

from LIBUtils import TSPLIBReader, tsplib_cost


# =========================
# 距离计算：对齐 TSPLIB / ICAM
# =========================

def tsplib_distances_vectorized(cities, current_city, unvisited, edge_weight_type: str):
    """
    cities: (n, 2) numpy array，原始 TSPLIB 坐标
    current_city: int，下标
    unvisited: 1D numpy array，未访问节点下标
    edge_weight_type: 'EUC_2D' 或 'CEIL_2D' 等
    """
    current_city_coords = cities[current_city]            # (2,)
    unvisited_coords = cities[unvisited]                 # (k, 2)

    diff = unvisited_coords - current_city_coords        # (k, 2)
    d_raw = np.sqrt(np.sum(diff * diff, axis=1))         # (k,)

    if edge_weight_type == "CEIL_2D":
        d = np.ceil(d_raw)
    elif edge_weight_type == "EUC_2D":
        # TSPLIB: floor(d + 0.5)
        d = np.floor(d_raw + 0.5)
    else:
        d = d_raw

    return d.astype(np.float64)


def tsplib_total_distance(cities, tour, edge_weight_type: str):
    """
    cities: (n, 2)
    tour: list[int]，包含回到起点后的完整 tour，比如 [0, 3, 5, ..., 0]
    """
    total = 0.0
    for i in range(len(tour) - 1):
        a = tour[i]
        b = tour[i + 1]
        diff = cities[b] - cities[a]
        d_raw = math.sqrt(diff[0] * diff[0] + diff[1] * diff[1])

        if edge_weight_type == "CEIL_2D":
            d = math.ceil(d_raw)
        elif edge_weight_type == "EUC_2D":
            d = math.floor(d_raw + 0.5)
        else:
            d = d_raw
        total += d
    return total


# =========================
# 最近邻核心
# =========================

def nearest_neighbor_tsp_tsplib(cities, edge_weight_type: str):
    """
    基于 TSPLIB 距离定义的最近邻 TSP。
    cities: (n, 2) numpy array
    edge_weight_type: 'EUC_2D' / 'CEIL_2D' / others
    """
    num_cities = cities.shape[0]
    visited = np.zeros(num_cities, dtype=bool)
    current_city = 0
    tour = [current_city]
    visited[current_city] = True

    for _ in range(num_cities - 1):
        unvisited = np.where(~visited)[0]
        distances = tsplib_distances_vectorized(cities, current_city, unvisited, edge_weight_type)
        nearest_idx = np.argmin(distances)
        nearest_city = unvisited[nearest_idx]

        tour.append(nearest_city)
        visited[nearest_city] = True
        current_city = nearest_city

    # 回到起点
    tour.append(tour[0])

    total_distance = tsplib_total_distance(cities, tour, edge_weight_type)
    return tour, total_distance


# =========================
# 单实例求解
# =========================

def solve_one_tsplib_instance(tsp_path: str):
    name, dimension, locs, edge_weight_type = TSPLIBReader(tsp_path)
    if name is None:
        # EDGE_WEIGHT_TYPE 不支持等情况
        return None

    optimal = tsplib_cost.get(name, None)
    if optimal is None:
        # 和 ICAM 一样：没有 BKS 就直接报错/跳过
        raise ValueError(f"optimal value (BKS) of instance {name} not found in tsplib_cost")

    bks = float(optimal)
    cities = np.array(locs, dtype=np.float64)  # (n, 2)

    start_time = time.time()
    tour, total_distance = nearest_neighbor_tsp_tsplib(cities, edge_weight_type)
    elapsed = time.time() - start_time

    gap = (total_distance - bks) * 100.0 / bks

    # sanity check：是否访问所有点
    unique = np.unique(np.array(tour[:-1]))
    assert len(unique) == dimension, f"Tour invalid: visited {len(unique)} unique nodes, expected {dimension}"

    return {
        "name": name,
        "dimension": dimension,
        "edge_weight_type": edge_weight_type,
        "bks": bks,
        "nn_cost": total_distance,
        "gap": gap,
        "time": elapsed,
        "tour": tour,
    }


# =========================
# 主流程：分桶统计 & 保存 log
# =========================

if __name__ == "__main__":
    # 和 ICAM 保持一致的 TSPLIB 目录
    lib_path = '/public/home/bayp/exp_survey_202509/0_data_survey/survey_bench_tsp'
    # lib_path = '/public/home/bayp/exp_survey_202509/0_data_survey/tsp_test'


    # 分桶边界，和 ICAM 一致
    scale_ranges = [
        (0, 1000),       # [0, 1000)
        (1000, 10000),   # [1000, 10000)
        (10000, 100001)  # [10000, 100001]
    ]

    # 为每个桶准备容器：gaps & times
    bucket_stats = {
        (0, 1000): {
            "gaps": [],
            "times": [],
        },
        (1000, 10000): {
            "gaps": [],
            "times": [],
        },
        (10000, 100001): {
            "gaps": [],
            "times": [],
        },
    }

    # 全部实例级别的结果
    all_results = []
    all_gaps = []
    all_times = []

    # per-instance 列表（用于写 log 类 ICAM detailed_log）
    inst_names = []
    inst_optimal = []
    inst_problem_size = []
    inst_nn_score = []
    inst_gap = []
    inst_time = []

    # 生成 log 路径（模仿 ICAM）
    tz = pytz.timezone("Asia/Shanghai")
    process_start_time = datetime.now(tz)
    log_dir = os.path.join(
        ".", "./TSP/result_survey_tsp_nn",
        process_start_time.strftime("%Y%m%d_%H%M%S") + "_NN_TSPLIB"
    )
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run_log.txt")

    log_lines = []

    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    # -----------------------------------------------------------------
    # 预扫描：统计实际需要处理的 .tsp 文件数量
    # -----------------------------------------------------------------
    all_tsp_files = []
    for root, dirs, files in os.walk(lib_path):
        for fname in files:
            if fname.endswith(".tsp"):
                all_tsp_files.append(os.path.join(root, fname))

    all_instance_num = len(all_tsp_files)
    solved_instance_num = 0
    current_idx = 0

    log(f"Total TSPLIB instances detected: {all_instance_num}")

    total_start = time.time()
    solved_instance_num = 0

    log("#################  NN Test on TSPLIB_Survey (ICAM style)  #################")
    log(f"TSPLIB folder: {lib_path}")
    log("-----------------------------------------------------------------")

    # -----------------------------------------------------------------
    # 主循环：对 all_tsp_files 遍历
    # -----------------------------------------------------------------
    for tsp_path in all_tsp_files:
        current_idx += 1
        fname = os.path.basename(tsp_path)

        try:
            result = solve_one_tsplib_instance(tsp_path)
            if result is None:
                log(f"[SKIP {current_idx}/{all_instance_num}] {tsp_path}, unsupported format")
                continue
            solved_instance_num += 1
        except Exception as e:
            log(f"[SKIP {current_idx}/{all_instance_num}] {tsp_path}, error: {e}")
            continue

        # ===== 保存结果 =====
        all_results.append(result)
        gap = result["gap"]
        t_ = result["time"]
        dim = result["dimension"]

        all_gaps.append(gap)
        all_times.append(t_)

        inst_names.append(result["name"])
        inst_optimal.append(result["bks"])
        inst_problem_size.append(dim)
        inst_nn_score.append(result["nn_cost"])
        inst_gap.append(gap)
        inst_time.append(t_)

        # ===== 分桶 =====
        for r in scale_ranges:
            lo, hi = r
            if lo <= dim < hi:
                bucket_stats[r]["gaps"].append(gap)
                bucket_stats[r]["times"].append(t_)
                break

        # ===== 加入进度输出 =====
        log(f"[{current_idx}/{all_instance_num}] "
            f"Instance: {result['name']}, dim: {dim}, "
            f"BKS: {result['bks']:.0f}, "
            f"NN cost: {result['nn_cost']:.0f}, "
            f"GAP: {gap:.3f}%, "
            f"time: {t_:.3f}s")



    total_end = time.time()
    total_time = total_end - total_start

    log("-----------------------------------------------------------------")
    log(f"All instances found: {all_instance_num}, solved: {solved_instance_num}")
    log(f"Total time: {total_time:.2f}s, avg time per solved instance: "
        f"{(total_time / solved_instance_num) if solved_instance_num > 0 else 0:.2f}s")

    # 分桶 summary（和 ICAM 同尺度）
    log("#################  Bucket Summary  #################")
    for r in scale_ranges:
        lo, hi = r
        gaps = bucket_stats[r]["gaps"]
        times = bucket_stats[r]["times"]
        num = len(gaps)
        avg_gap = np.mean(gaps) if num > 0 else 0.0
        avg_time = np.mean(times) if num > 0 else 0.0

        # 文本格式模仿 ICAM：
        if (lo, hi) == (0, 1000):
            bracket_str = "[0, 1000)"
        elif (lo, hi) == (1000, 10000):
            bracket_str = "[1000, 10000)"
        else:
            bracket_str = "[10000, 100000]"

        log(f"{bracket_str}, number: {num}, "
            f"avg GAP: {avg_gap:.3f}%, avg time: {avg_time:.3f}s")

    # 全部实例的平均 GAP
    avg_all_gap = np.mean(all_gaps) if len(all_gaps) > 0 else 0.0
    avg_all_time = np.mean(all_times) if len(all_times) > 0 else 0.0
    log("###################################  Overall Summary  ##########################################")
    log(f"All solved instances, number: {len(all_gaps)}, "
        f"avg GAP: {avg_all_gap:.3f}%, avg time: {avg_all_time:.3f}s")

    # # detailed_log 风格：把列表直接写进 log
    # log("#################  Detailed Results  #################")
    # log(f"instance: {inst_names}")
    # log(f"optimal: {inst_optimal}")
    # log(f"problem_size: {inst_problem_size}")
    # log(f"nn_score: {inst_nn_score}")
    # log(f"gap: {inst_gap}")
    # log(f"time: {inst_time}")

    # 写入文本文件
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    # print(f"\n[INFO] Log saved to: {log_path}")
