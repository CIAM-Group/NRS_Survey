"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import argparse
import time
from args import add_common_args
from learning.tsp.data_iterator import DataIterator
from learning.tsp.traj_learner import TrajectoryLearner
from utils.exp import setup_exp
from learning.tsp.dataloading.dataset import load_dataset
import os
import torch
from tqdm import tqdm

from learning.tsp.decoding import decode
from utils.misc import get_opt_gap

import logging, time, random
from datetime import datetime
import pytz

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
    test_dataset = "/public/home/bayp/exp_survey_202509/0_data_survey/survey_bench_tsp"
    output_dir = "./tsp_results/bq_test_tsplib"

    # --- 预训练模型 ---
    pretrained_model = "./pretrained_models/tsp"  # ← 模型前缀路径，不带 _best.pt
    # --- 测试设置 ---
    beam_size = 1
    knns = 250

    # Common
    test_only = True
    seed = 1234
    test_batch_size = 1  # TSPLIB 每次一个实例
    train_batch_size = 512
    val_batch_size = 1024
    # test_batch_size = 1024
    debug = False

    decay_rate = 0.99
    decay_every = 50

    # Optim
    test_every = 100
    nb_total_epochs = 1000
    lr = 2.5e-4

    # ! 筛选范围
    min_nodes = 0
    max_nodes = 1000
    
# def main():

#     args = Args()
#     TEST_DATASET = args.test_dataset
#     OUTPUT_DIR = args.output_dir
#     BATCH_SIZE = args.test_batch_size

#     # === 初始化模型 ===
#     net, module, device, optimizer, checkpointer, other = setup_exp(args, problem="tsp", is_test=True)
#     module.eval()

#     # === 加载测试数据 ===
#     test_loader = load_dataset(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, what="test", min_nodes=args.min_nodes, max_nodes=args.max_nodes)

#     print(f"[INFO] Loaded TSPLIB data from: {TEST_DATASET}")
#     print(f"[INFO] Number of instances: {len(test_loader)}")

#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     out_path = os.path.join(OUTPUT_DIR, "results.txt")

#     # === 推理过程 ===
#     all_results = []
#     total_pred, total_label, total_gap = 0.0, 0.0, 0.0
#     count = 0

#     with torch.no_grad():
#         for idx, batch in enumerate(tqdm(test_loader, desc="Testing TSPLIB")):
#             try:
#                 coords = batch["nodes_coord"].to(device)
#                 dist_mats = batch["dist_matrices"].to(device)
#                 label = batch.get("tour_len", None)
#                 label = label.to(device) if label is not None and label.numel() > 0 else None

#                 # 模型推理
#                 _, pred_lens = decode(coords, dist_mats, net, args.beam_size, args.knns)
#                 pred_len = float(pred_lens.mean().item())

#                 if label is not None:
#                     label_val = float(label.mean().item())
#                     gap = float(get_opt_gap(pred_lens, label))
#                     all_results.append(f"[{idx}] pred={pred_len:.3f}, label={label_val:.3f}, gap={gap:.3f}%")
#                     total_pred += pred_len
#                     total_label += label_val
#                     total_gap += gap
#                 else:
#                     all_results.append(f"[{idx}] pred={pred_len:.3f}")

#                 count += 1
#             except Exception as e:
#                 # 其它任何错误也跳过
#                 all_results.append(f"[{idx}] SKIP (Exception): {type(e).__name__}: {str(e)}")
#                 continue

#     # === 写出结果 ===
#     with open(out_path, "w", encoding="utf-8") as f:
#         for line in all_results:
#             f.write(line + "\n")

#         if count > 0 and total_label > 0:
#             f.write("\n[SUMMARY] instances={0}, avg_pred={1:.3f}, avg_label={2:.3f}, avg_gap={3:.3f}%\n".format(
#                 count, total_pred / count, total_label / count, total_gap / count))
#         else:
#             f.write(f"\n[SUMMARY] instances={count}, avg_pred={total_pred / count if count else 0:.3f}\n")

#     print(f"[INFO] Results saved to: {out_path}")

def main():
    import logging, time
    args = Args()

    # ========= 日志器 =========
    os.makedirs(args.output_dir, exist_ok=True)

    # 生成 ICAM 风格的文件名：日期 + 随机数
    tz = pytz.timezone("Asia/Shanghai")
    timestamp = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
    rand_suffix = str(random.randint(10000, 99999))
    log_filename = f"{timestamp}_{rand_suffix}_test_TSPLIB_Survey.log"
    log_path = os.path.join(args.output_dir, log_filename)

    logger = logging.getLogger("bq_tsplib")
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

    logger.info("===== BQ-NCO TSPLIB Test (ICAM-style buckets & logging) =====")
    logger.info(f"Log file created: {log_path}")

    logger.info(f"Model prefix : {args.pretrained_model}")
    logger.info(f"Test dataset : {args.test_dataset}")
    logger.info(f"Output dir   : {args.output_dir}")
    logger.info(f"Beam/knns    : beam_size={args.beam_size}, knns={args.knns}")

    # === 初始化模型 ===
    net, module, device, optimizer, checkpointer, other = setup_exp(args, problem="tsp", is_test=True)
    module.eval()
    logger.info("Model initialized & loaded (eval mode).")

    # === ICAM 分桶 ===
    buckets = [(0, 1000), (1000, 10000), (10000, 100001)]

    # 总体统计
    all_no_aug_gaps = []
    all_preds, all_labels = [], []
    total_instances = 0
    total_solved = 0
    t0_all = time.time()

    for (lo, hi) in buckets:
        logger.info("=" * 60)
        logger.info(f"### Bucket: [{lo}, {hi})")
        # 为当前分桶创建 DataLoader（按节点数过滤）
        try:
            test_loader = load_dataset(
                args.test_dataset,
                batch_size=args.test_batch_size,
                shuffle=False,
                what="test",
                min_nodes=lo,
                max_nodes=hi
            )
        except Exception as e:
            logger.info(f"[Bucket {lo}-{hi}) No instances. Reason: {e}")
            continue

        # 拿到文件清单（用于打印实例名）
        files = getattr(test_loader.dataset, "files", None)
        logger.info(f"Instances in bucket: {len(test_loader)}")
        if len(test_loader) == 0:
            continue

        # 分桶统计
        bucket_gaps = []
        bucket_preds, bucket_labels = [], []
        solved_this_bucket = 0
        t0_bucket = time.time()

        with torch.no_grad():
            # for idx, batch in enumerate(tqdm(test_loader, desc=f"Bucket [{lo},{hi})")):
            for idx, batch in enumerate(test_loader):
                try:
                    coords = batch["nodes_coord"].to(device, non_blocking=True)      # [1, N, 2]
                    dist_mats = batch["dist_matrices"].to(device, non_blocking=True) # [1, N, N]
                    label = batch.get("tour_len", None)
                    label = label.to(device) if (label is not None and label.numel() > 0) else None

                    # === 逐实例计时开始 ===
                    t_inst0 = time.time()

                    # 预测
                    _, pred_lens = decode(coords, dist_mats, net, args.beam_size, args.knns)
                    pred = float(pred_lens.mean().item())

                    # === 逐实例计时结束 ===
                    dur_inst = time.time() - t_inst0

                    # 打印名称与 N（batch_size=1 时这样拿）
                    name = None
                    N = coords.shape[1]
                    if files is not None and idx < len(files):
                        name = os.path.basename(files[idx])

                    if label is not None:
                        lab = float(label.mean().item())
                        gap = float(get_opt_gap(pred_lens, label))
                        bucket_gaps.append(gap)
                        bucket_preds.append(pred)
                        bucket_labels.append(lab)
                        all_no_aug_gaps.append(gap)
                        all_preds.append(pred)
                        all_labels.append(lab)
                        solved_this_bucket += 1
                        total_solved += 1
                        # ! 2025.10.7 补时间
                        logger.info(f"[{idx:04d}] {name or ''} N={N} | pred={pred:.3f}, label={lab:.3f}, gap={gap:.3f}% | time={dur_inst:.3f}s")
                    else:
                        bucket_preds.append(pred)
                        all_preds.append(pred)
                        logger.info(f"[{idx:04d}] {name or ''} N={N} | pred={pred:.3f} (no label) | time={dur_inst:.3f}s")

                    total_instances += 1

                except Exception as e:
                    logger.info(f"[{idx:04d}] SKIP (Exception {type(e).__name__}): {e}")
                    continue

                t1_bucket = time.time()
                if solved_this_bucket > 0 and len(bucket_gaps) > 0 and len(bucket_labels) > 0:
                    avg_gap = sum(bucket_gaps) / len(bucket_gaps)
                    logger.info(f"Bucket [{lo},{hi}) DONE | solved={solved_this_bucket}/{len(test_loader)} "
                                f"| avg_gap={avg_gap:.3f}% | time={t1_bucket - t0_bucket:.2f}s "
                                f"| avg_time/solved={(t1_bucket - t0_bucket)/max(1, solved_this_bucket):.2f}s")
                else:
                    logger.info(f"Bucket [{lo},{hi}) DONE | solved={solved_this_bucket}/{len(test_loader)} "
                                f"| (no label or all skipped) | time={t1_bucket - t0_bucket:.2f}s")

    # === 总结 ===
    t1_all = time.time()
    logger.info("=" * 60)
    logger.info("#################  ALL DONE  #################")
    if total_solved > 0 and len(all_no_aug_gaps) > 0 and len(all_labels) > 0:
        avg_gap_all = sum(all_no_aug_gaps) / len(all_no_aug_gaps)
        logger.info(f"All instances: solved={total_solved}/{total_instances} | "
                    f"avg_gap(no-aug)={avg_gap_all:.3f}% | "
                    f"total_time={t1_all - t0_all:.2f}s | "
                    f"avg_time/solved={(t1_all - t0_all)/max(1,total_solved):.2f}s")
    else:
        logger.info(f"All instances: solved={total_solved}/{total_instances} | "
                    f"total_time={t1_all - t0_all:.2f}s")

    logger.info(f"Log saved to: {log_path}")

if __name__ == "__main__":
    main()
