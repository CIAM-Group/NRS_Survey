# test_survey.py
import os
import time
import logging
from datetime import datetime

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from models import TSPModelConfig
from training import load_ckpt

from helpers.coord_transform import normalize, cdist
from helpers.solution_transform import sol2adj
from decoding.continuous_sampling import (
    sample_dpmpp_2m,
    get_linear_timesteps,
)
from decoding.utils import (
    convert_heatmap_dtype,
    CoordDynamicArgument as DynamicArgument,
)
from lib import tsp_greedy_insert, tsp_double_two_opt, tsp_eval_cost
from LIBUtils import TSPLIBReader, tsplib_cost

import pytz


# ==========================
# 固定参数（对应 tsp100.sh + survey 设定）
# ==========================

# ckpt 与 TSPLIB 文件夹
CKPT_PATH = "ckpts/tsp1000.ckpt"
LIB_PATH = "/public/home/bayp/exp_survey_202509/0_data_survey/survey_bench_tsp"
# LIB_PATH = "/public/home/bayp/exp_survey_202509/0_data_survey/survey_bench_tsp_small_example"


# 解码与搜索参数（基本按 tsp100.sh 写死，batch_size 这里使用 1 做逐实例评测）
SAMPLING_STEPS = 4
TWO_OPT_STEPS = 0
CYCLES = 200
RUNS = 16
BATCH_SIZE = 1

# RANDOM_TWO_OPT_RANGE 现在按每个 instance 动态设置为 (n/4, 3n/4)
# 这里保留一个描述用的常量，仅用于 log
RANDOM_TWO_OPT_RANGE_DESC = "dynamic: (n/4, 3n/4)"

THREADS_OVER_BATCHES = 1         # 本脚本逐实例评测，不再做 over-batch 并行

# Dynamic data augmentation (coord reflections)，所有实验关aug
# ARGUMENT_LEVEL = 1
ARGUMENT_LEVEL = 0

HEATMAP_DTYPE = "float32"        # or "uint8"

# TOPK 为全局上限，实际每个实例会用 min(TOPK, N*N)
TOPK = None                      # 若需要限制 top-k edge，可改为整数
# TOPK = 20000

SEED = 0

# 分桶区间（与 ICAM 一致）
SCALE_RANGES = [
    (0, 1000),
    (1000, 10000),
    (10000, 100001),
]

# ==========================
# logging 相关
# ==========================

def setup_logger():
    tz = pytz.timezone("Asia/Shanghai")
    t_str = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("result_survey_tsp", f"{t_str}_GenSCO_TSPLIB_Survey")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run_log.txt")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 清空已有 handler，避免重复输出
    logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Log file created at: {log_path}")
    return logger, log_dir


def log_parameters(logger: logging.Logger):
    logger.info("=" * 60)
    logger.info("===== GenSCO Survey Test Parameters =====")
    logger.info(f"CKPT_PATH = {CKPT_PATH}")
    logger.info(f"LIB_PATH = {LIB_PATH}")
    logger.info(f"SAMPLING_STEPS = {SAMPLING_STEPS}")
    logger.info(f"TWO_OPT_STEPS = {TWO_OPT_STEPS}")
    logger.info(f"CYCLES = {CYCLES}")
    logger.info(f"RUNS = {RUNS}")
    logger.info(f"BATCH_SIZE = {BATCH_SIZE}")
    logger.info(f"RANDOM_TWO_OPT_RANGE = {RANDOM_TWO_OPT_RANGE_DESC}")
    logger.info(f"THREADS_OVER_BATCHES = {THREADS_OVER_BATCHES}")
    logger.info(f"ARGUMENT_LEVEL = {ARGUMENT_LEVEL}")
    logger.info(f"HEATMAP_DTYPE = {HEATMAP_DTYPE}")
    logger.info(f"TOPK (per instance) = min({TOPK}, N*N)")
    logger.info(f"SEED = {SEED}")
    logger.info(f"SCALE_RANGES = {SCALE_RANGES}")
    logger.info("=========================================")


# ==========================
# TSPLIB metric 相关工具
# ==========================

def tsplib_distance_matrix(locs: np.ndarray, edge_weight_type: str) -> np.ndarray:
    """
    按 TSPLIB 的 EDGE_WEIGHT_TYPE 生成距离矩阵：
    - EUC_2D : floor(sqrt(dx^2 + dy^2) + 0.5)
    - CEIL_2D: ceil(sqrt(dx^2 + dy^2))
    其他类型则直接使用连续距离。
    locs: (N, 2)
    返回: (N, N) float32
    """
    # diff = locs[:, None, :] - locs[None, :, :]
    # d_raw = np.sqrt(np.sum(diff * diff, axis=-1))
    
    # Use GPU for distance calculation (similar to tsp100.sh)
    # Note: This will trigger JIT recompilation for each new N, but it's acceptable for survey.
    d_raw = np.array(cdist(locs))

    if edge_weight_type == "CEIL_2D":
        d = np.ceil(d_raw)
    elif edge_weight_type == "EUC_2D":
        d = np.floor(d_raw + 0.5)
    else:
        d = d_raw

    return d.astype(np.float32)


# ==========================
# GenSCO 推理核心
# ==========================

def build_encode_decode(model, num_nodes: int):
    """
    基于当前 model & num_nodes，构造 encode / decode 的 jitted 函数，
    逻辑基本来自 decoding_gpu_heuristics.tsp.py 的 tsp_flow_searching_decode。
    """
    heatmap_dtype_jax = jnp.dtype(HEATMAP_DTYPE)

    @jax.jit
    def encode(raw_features: jax.Array):
        # 这里依然做一次中心化 / 归一化（多分辨率坐标鲁棒）
        feats = normalize(raw_features, centering_method="mean")
        feats = model.encode(feats)
        return feats

    @jax.jit
    def decode(features: jax.Array, sols: jax.Array):
        timesteps = get_linear_timesteps(SAMPLING_STEPS)

        def denoised_fn(adjmat: jax.Array, timestep: jax.Array):
            logits = model.decode(features, timestep, adjmat)
            # logits -> softmax adjacency
            adj = jax.nn.softmax(logits, axis=-1) * 2.0
            return adj

        adjmat = sol2adj(
            sols,
            dtype=jnp.float16 if num_nodes > 128 else jnp.float32,
        )
        adjmat_pred = sample_dpmpp_2m(denoised_fn, adjmat, timesteps)
        adjmat_pred = convert_heatmap_dtype(adjmat_pred, dtype=heatmap_dtype_jax)

        flat = adjmat_pred.reshape(BATCH_SIZE, -1)
        num_edges_total = flat.shape[-1]

        if TOPK is None or TOPK <= 0:
            candidate_edges = jnp.argsort(
                flat,
                axis=-1,
                descending=True,
                stable=False,
            )
        else:
            # 给 topk 一个基于 dimension 的上限，避免 k > N*N 导致错误
            eff_topk = min(TOPK, int(num_edges_total))
            _, candidate_edges = jax.lax.top_k(flat, k=eff_topk)

        candidate_edges = jnp.stack(jnp.divmod(candidate_edges, num_nodes), axis=-1)
        return candidate_edges

    return encode, decode


def run_instance(
    coords_norm_batch: np.ndarray,
    dist_mat_batch: np.ndarray,
    model,
) -> float:
    """
    对单个 TSPLIB 实例进行多轮搜索，返回最优 tour 的 cost（TSPLIB 度量下）。
    coords_norm_batch: (1, N, 2) 坐标（可归一化），供模型编码使用
    dist_mat_batch:    (1, N, N) TSPLIB 度量的距离矩阵，供 two-opt / eval_cost 使用
    """
    assert coords_norm_batch.shape[0] == 1
    assert dist_mat_batch.shape[0] == 1

    num_nodes = coords_norm_batch.shape[1]
    encode, decode = build_encode_decode(model, num_nodes)

    # DynamicArgument: argument_level=0 or 1
    features_manager = DynamicArgument(
        encode_fn=encode,
        coords=coords_norm_batch,
        argument_level=ARGUMENT_LEVEL,
    )

    num_workers = 1  # C++ 后端内部再做多线程

    storage = {}
    for r in range(RUNS):
        # 初始 tours：随机排列
        base_tour = jnp.arange(num_nodes, dtype=jnp.int32)
        key = jax.random.key(SEED + r)
        keys = jax.random.split(key, BATCH_SIZE)
        tours = jax.vmap(jax.random.permutation, in_axes=(0, None))(keys, base_tour)
        tours = np.array(tours, dtype=np.int32)  # (B, N)

        min_costs = np.full((BATCH_SIZE,), np.finfo(np.float32).max, dtype=np.float32)
        generator = np.random.default_rng(SEED + r * 7)

        # ! 动态 two-opt 随机步数范围：约 (n/4, 3n/4)
        low_steps = max(1, num_nodes // 4)
        high_steps_exclusive = max(low_steps + 1, (3 * num_nodes) // 4)

        for c in range(CYCLES):
            # denoising -> candidate edges
            candidate_edges = decode(
                features_manager(generator),
                jnp.array(tours, dtype=jnp.int32),
            )
            candidate_edges = np.array(candidate_edges, dtype=np.int32)  # (B, K, 2)

            # greedy insert + double two-opt （C++ 实现）
            tours = tsp_greedy_insert(
                candidate_edges,
                num_nodes,
                num_workers=num_workers,
            )

            random_steps = generator.integers(
                low_steps,
                high_steps_exclusive,
                size=BATCH_SIZE,
                dtype=np.int32,
            )
            tours, tours_disrupted = tsp_double_two_opt(
                seed=SEED + r + c * 10,
                tours=tours,
                dist_mat=dist_mat_batch,
                steps=TWO_OPT_STEPS,
                random_steps=random_steps,
                num_workers=num_workers,
            )

            # 计算当前 tours 的 TSPLIB cost
            costs = tsp_eval_cost(
                tours,
                dist_mat_batch,
                num_workers=num_workers,
            )  # (B,)

            min_costs = np.minimum(min_costs, costs)
            tours = tours_disrupted

        storage[r] = min_costs

    best = storage[0]
    for r in range(1, RUNS):
        best = np.minimum(best, storage[r])

    # BATCH_SIZE=1，直接返回标量
    return float(best.mean().item())


# ==========================
# 主流程
# ==========================

def main():
    logger, _ = setup_logger()
    log_parameters(logger)

    np.random.seed(SEED)

    # ========= 1. 加载模型 =========
    logger.info(f"Loading checkpoint from {CKPT_PATH} ...")
    params, _, _, model_config, _, _ = load_ckpt(CKPT_PATH)
    model_config: TSPModelConfig
    model = model_config.construct_model()
    graphdef = nnx.graphdef(model)
    model = nnx.merge(graphdef, params)
    logger.info("Checkpoint loaded and model constructed.")

    # ========= 预扫描 + 排序 =========
    logger.info("Scanning and sorting TSPLIB instances...")
    all_instances_list = []
    for root, dirs, files in os.walk(LIB_PATH):
        for file in files:
            if file.endswith(".tsp"):
                full = os.path.join(root, file)
                name, dimension, locs, ewt = TSPLIBReader(full)
                if name is not None and dimension is not None:
                    all_instances_list.append((dimension, full, name, locs, ewt))

    # 按 dimension 升序排序
    all_instances_list.sort(key=lambda x: x[0])
    logger.info(f"Total {len(all_instances_list)} instances found after sorting.")

    # ========= 2. 分桶遍历 =========
    all_gaps = []
    all_times = []
    all_instances = 0
    all_solved_instances = 0

    start_all = time.time()

    for scale_range in SCALE_RANGES:
        logger.info("")
        logger.info(
            "#################  Test scale range: {}  #################".format(
                scale_range
            )
        )

        bucket_instances = []
        bucket_sizes = []
        bucket_gaps = []
        bucket_times = []

        # ======== 使用已排序列表遍历所有 instance ========
        for (dimension, full_path, name, locs, ewt) in all_instances_list:
            if not (scale_range[0] <= dimension < scale_range[1]):
                continue

            all_instances += 1

            optimal = tsplib_cost.get(name, None)
            if optimal is None:
                logger.info(f"[SKIP] Instance {name}: optimal cost not found.")
                continue

            locs_np = np.array(locs, dtype=np.float32)

            logger.info("=" * 60)
            logger.info(
                f"[Scale {scale_range}] "
                f"Instance #{len(bucket_instances) + 1} | "
                f"Name: {name} | N = {dimension} | EDGE_WEIGHT_TYPE = {ewt}"
            )

            # 这里直接使用原始坐标，归一化在 encode 中处理
            coords_norm_batch = locs_np[None, :, :]

            dist_mat = tsplib_distance_matrix(locs_np, ewt)
            dist_mat_batch = dist_mat[None, :, :]

            inst_start = time.time()
            try:
                pred_cost = run_instance(
                    coords_norm_batch=coords_norm_batch,
                    dist_mat_batch=dist_mat_batch,
                    model=model,
                )
                inst_time = time.time() - inst_start
                all_solved_instances += 1
            except Exception as e:
                logger.info(
                    f"[ERROR] Instance {name} (N={dimension}) failed, skip. Error: {e}"
                )
                continue

            gap = (pred_cost - optimal) * 100.0 / optimal

            bucket_instances.append(name)
            bucket_sizes.append(dimension)
            bucket_gaps.append(gap)
            bucket_times.append(inst_time)

            all_gaps.append(gap)
            all_times.append(inst_time)

            logger.info(f"Instance name: {name}, optimal cost: {optimal:.4f}")
            logger.info(
                f"Pred cost: {pred_cost:.4f}, Gap: {gap:.3f}%, "
                f"Time: {inst_time:.3f}s"
            )

        # ======== 当前 bucket 统计输出 ========
        if len(bucket_instances) == 0:
            logger.info(
                f"No instances found / solved in scale_range {scale_range}."
            )
            continue

        during_range = sum(bucket_times)
        avg_gap = float(np.mean(bucket_gaps))
        avg_time = float(np.mean(bucket_times))

        logger.info("---------------------------------------------------------------")
        logger.info(
            "Scale_range: {} | solved instances: {} | min_N: {} | max_N: {}".format(
                scale_range,
                len(bucket_instances),
                min(bucket_sizes),
                max(bucket_sizes),
            )
        )
        logger.info(
            "Avg gap: {:.3f}% | Avg time per instance: {:.3f}s".format(
                avg_gap, avg_time
            )
        )
        logger.info(
            "Total time in this scale_range: {:.2f}s".format(during_range)
        )
        logger.info("---------------------------------------------------------------")

    # ========= 全局统计 =========
    end_all = time.time()
    total_time_all = end_all - start_all

    logger.info("")
    logger.info("#################  All Done  #################")
    logger.info(
        "All instances: {} | solved: {} | total time: {:.2f}s | avg time per solved instance: {:.2f}s".format(
            all_instances,
            all_solved_instances,
            total_time_all,
            total_time_all / max(all_solved_instances, 1),
        )
    )
    if len(all_gaps) > 0:
        logger.info(
            "Overall avg gap: {:.3f}% | Overall avg time per instance: {:.3f}s".format(
                float(np.mean(all_gaps)),
                float(np.mean(all_times)),
            )
        )
    else:
        logger.info("No instance solved, overall stats not available.")


if __name__ == "__main__":
    main()
