"""
BQ-NCO — CVRP Survey Test (ICAM-style buckets & logging)
- 目录/单文件 .vrp：按 VRPLIB 解析
- 分桶统计：[(0,1000), (1000,10000), (10000,100001)]（最后一桶左闭右闭）
- 距离计费：EUC_2D→round, CEIL_2D→ceil（在 dataloader 里已实现）
- 日志：日期+随机数命名；仅写 log，不写 txt
"""

import os
import random
import logging
from datetime import datetime

import pytz
import torch
from tqdm import tqdm

from utils.exp import setup_exp
from utils.misc import get_opt_gap
from learning.cvrp.dataloading.dataset import load_dataset
from learning.cvrp.decoding import decode  # (node_coords, dist_matrices, demands, capacities, net, beam, knns)
import time

# =====================
# 参数集中设置
# =====================
class Args:
    # --- 模型结构 ---
    dim_emb = 192
    dim_ff = 512
    nb_layers_encoder = 9
    nb_heads = 12
    dropout = 0.0
    batchnorm = False
    activation_ff = "relu"
    activation_attention = "softmax"

    # --- 数据 ---
    train_dataset = None
    val_dataset = None
    test_dataset = "/public/home/bayp/exp_survey_202509/0_data_survey/survey_bench_cvrp"  # 目录或 .vrp
    output_dir = "./cvrp_results/bq_test_cvrplib"

    # --- 预训练模型 ---
    pretrained_model = "./pretrained_models/cvrp"  # 前缀，不带 _best

    # --- 测试设置 ---
    beam_size = 1
    knns = -1
    # ! 按消融给的最好值
    # knns = 250


    # --- 通用 ---
    test_only = True
    seed = 1234
    test_batch_size = 1
    train_batch_size = 512
    val_batch_size = 1024
    debug = False

    # --- LR decay (train-only) ---
    decay_rate = 0.99
    decay_every = 50

    # --- Optim (train-only) ---
    test_every = 100
    nb_total_epochs = 1000
    lr = 2.5e-4


# 分桶：前两段左闭右开，最后一段左闭右闭
BUCKETS = [(0, 1000), (1000, 10000), (10000, 100001)]


def _bucket_to_loader_range(lo, hi):
    """
    把 BUCKETS 的区间语义转换为 dataloader 的 [min_nodes, max_nodes) 过滤；
    最后一桶为 [lo, hi]，其余 [lo, hi)。
    """
    if (lo, hi) == BUCKETS[-1]:
        return lo, hi  # inclusive
    else:
        return lo, hi  # 注意：我们的 dataloader 内部就是 [min, max) 语义，这里传 hi 即可，
                       # 最后一桶因为写了 100001，自然实现左闭右闭（上限 exclusive=100001）


def _init_logger(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tz = pytz.timezone("Asia/Shanghai")
    timestamp = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
    rand_suffix = str(random.randint(10000, 99999))
    log_filename = f"{timestamp}_{rand_suffix}_test_CVRPLIB_Survey.log"
    log_path = os.path.join(out_dir, log_filename)

    logger = logging.getLogger("bq_cvrplib")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("===== BQ-NCO CVRP Test (ICAM-style) =====")
    logger.info(f"Log file: {log_path}")
    return logger, log_path


def main():
    args = Args()

    # 日志
    logger, log_path = _init_logger(args.output_dir)
    logger.info(f"Dataset root: {args.test_dataset}")

    logger.info(f"Model prefix : {args.pretrained_model}")
    logger.info(f"Test dataset : {args.test_dataset}")
    logger.info(f"Output dir   : {args.output_dir}")
    logger.info(f"Beam/knns    : beam_size={args.beam_size}, knns={args.knns}")

    # 初始化模型（只测不训）
    net, module, device, optimizer, checkpointer, other = setup_exp(args, problem="cvrp", is_test=True)
    module.eval()
    logger.info("Model initialized.")

    all_solved_cnt = 0
    all_total_cnt = 0
    all_gaps = []

    total_start = datetime.now()

    for (lo, hi) in BUCKETS:
        rng_lo, rng_hi = _bucket_to_loader_range(lo, hi)
        logger.info("=" * 60)
        bucket_name = f"[{lo},{hi}]" if (lo, hi) == BUCKETS[-1] else f"[{lo},{hi})"
        logger.info(f"### Bucket: {bucket_name}")

        # 构建 dataloader（目录/单 .vrp 都支持；num_workers=0 防止静默挂）
        try:
            test_loader = load_dataset(
                args.test_dataset,
                batch_size=args.test_batch_size,
                shuffle=False,
                what="test",
                min_nodes=rng_lo,
                max_nodes=rng_hi
            )
        except Exception as e:
            logger.info(f"[Bucket {bucket_name}] No instances. Reason: {e}")
            continue

        bucket_gaps = []
        bucket_cnt = 0
        bucket_solved = 0
        bucket_start = datetime.now()

        with torch.no_grad():
            it = iter(test_loader)
            num_batches = len(test_loader)   # batch_size=1 时即实例数
            # for idx, batch in enumerate(tqdm(test_loader, desc=f"Testing {bucket_name}")):
            for idx in tqdm(range(num_batches), desc=f"Testing {bucket_name}"):
                all_total_cnt += 1
                bucket_cnt += 1
                try:
                    # === 逐实例计时开始 ===
                    t_inst0 = time.time()
                    batch = next(it)  # 这里触发 Dataset.__getitem__，会计算 dist_matrices

                    node_coords = batch["node_coords"].to(device)
                    dist_mats = batch["dist_matrices"].to(device)
                    demands = batch["demands"].to(device)
                    capacities = batch["capacities"].to(device)
                    label = batch.get("tour_len", None)
                    label = label.to(device) if label is not None and label.numel() > 0 else None

                    # 实例名与规模（兼容 batch 维度）
                    inst_name = batch.get("name", ["unknown"])
                    if isinstance(inst_name, (list, tuple)):
                        inst_name = inst_name[0]
                    elif torch.is_tensor(inst_name):
                        inst_name = inst_name[0]
                    size_val = batch.get("problem_size", [node_coords.size(1) - 1])
                    if isinstance(size_val, (list, tuple)):
                        size_val = int(size_val[0])
                    elif torch.is_tensor(size_val):
                        size_val = int(size_val[0].item())

                    # 推理
                    pred_lens, _ = decode(node_coords, dist_mats, demands, capacities, net, args.beam_size, args.knns)
                    # 标量化
                    pred_val = float(pred_lens.mean().item())

                    # === 逐实例计时结束 ===
                    dur_inst = time.time() - t_inst0

                    if label is not None:
                        label_val = float(label.mean().item())
                        gap = float(get_opt_gap(pred_lens, label))  # %
                        bucket_gaps.append(gap)
                        all_gaps.append(gap)
                        bucket_solved += 1
                        all_solved_cnt += 1
                        logger.info(
                            f"[{bucket_name}#{idx}] name={inst_name}, size={size_val}, "
                            f"pred={pred_val:.3f}, label={label_val:.3f}, gap={gap:.3f}%  | time={dur_inst:.3f}s"
                        )
                    else:
                        logger.info(f"[{bucket_name}#{idx}] name={inst_name}, size={size_val}, pred={pred_val:.3f} (no label) | time={dur_inst:.3f}s")
                except (RuntimeError, torch.cuda.OutOfMemoryError, MemoryError) as e:
                    msg = str(e)
                    logger.info(f"[{bucket_name}#{idx}] SKIP (OOM/RuntimeError): {type(e).__name__}: {msg}")
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass
                    continue
                except Exception as e:
                    logger.info(f"[{bucket_name}#{idx}] SKIP (Exception): {type(e).__name__}: {e}")
                    continue

        # 桶统计
        dur = (datetime.now() - bucket_start).total_seconds()
        if bucket_gaps:
            logger.info(f"[Bucket {bucket_name}] solved={bucket_solved}/{bucket_cnt}, "
                        f"avg_gap={sum(bucket_gaps)/len(bucket_gaps):.3f}%, "
                        f"avg_time_per_inst={dur/max(bucket_solved,1):.2f}s")
        else:
            logger.info(f"[Bucket {bucket_name}] solved={bucket_solved}/{bucket_cnt}, "
                        f"no labeled instances, time={dur:.2f}s")

    # 总体统计
    total_dur = (datetime.now() - total_start).total_seconds()
    logger.info("=" * 60)
    if all_gaps:
        logger.info(f"[ALL] solved={all_solved_cnt}/{all_total_cnt}, "
                    f"avg_gap={sum(all_gaps)/len(all_gaps):.3f}%, "
                    f"total_time={total_dur:.2f}s, avg_time_per_solved={total_dur/max(all_solved_cnt,1):.2f}s")
    else:
        logger.info(f"[ALL] solved={all_solved_cnt}/{all_total_cnt}, "
                    f"no labeled instances, total_time={total_dur:.2f}s")
    logger.info("===== Done =====")
    print(f"[INFO] Log saved to: {log_path}")


if __name__ == "__main__":
    main()