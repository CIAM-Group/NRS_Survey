import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusion.consistency.meta import MetaConsistency
from diffusion.utils.diffusion_schedulers import InferenceSchedule
from diffusion.utils.tsp_utils import TSPEvaluator, batched_two_opt_torch, merge_tours

import time
import logging
import os


class TSPConsistency(MetaConsistency):
    def __init__(
        self,
        args,
        sigma_max=1000,
        sigma_min=0,
        weight_schedule="uniform",
        boundary_func="truncate",
    ):
        super(TSPConsistency, self).__init__(
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            weight_schedule=weight_schedule,
            boundary_func=boundary_func,
        )
        self.args = args

    def consistency_losses(self, model, batch):
        edge_index = None
        if model.sparse:
            _, graph_data, point_indicator, edge_indicator, _ = batch
            route_edge_flags = graph_data.edge_attr
            # points: B*N, 2
            # edge_index: B*E, 2
            # adj_matrix: B, N*k
            points = graph_data.x
            edge_index = graph_data.edge_index
            num_edges = edge_index.shape[1]
            batch_size = point_indicator.shape[0]
            adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
            t = torch.randint(
                1,
                model.diffusion.T + 1,
                [point_indicator.shape[0]],
                device=points.device,
            )
            t2 = (model.args.alpha * t).int()
        else:
            # points: B, N, 2
            # adj_matrix: B, N, N
            _, points, adj_matrix, _ = batch
            batch_size = points.shape[0]
            t = torch.randint(
                1, model.diffusion.T + 1, [points.shape[0]], device=points.device
            )
            t2 = (model.args.alpha * t).int()

        x0 = F.one_hot(
            adj_matrix.long(), num_classes=2
        ).float()  # B, N, N, 2 if not sparse else B, N*N, 2
        if model.sparse:
            x0 = x0.unsqueeze(1)  # B, 1, N*K, 2

        x_t = model.diffusion.sample(x0, t.cpu().numpy())
        x_t2 = model.diffusion.sample(x0, t2.cpu().numpy())

        # x_t, x_t2 = model.diffusion.consistency_sample(x0, t.cpu().numpy(), t2.cpu().numpy())

        if model.sparse:
            t = t.reshape(-1, 1).repeat(1, adj_matrix.shape[1]).reshape(-1)
            t2 = t2.reshape(-1, 1).repeat(1, adj_matrix.shape[1]).reshape(-1)
            x_t = x_t.reshape(-1)
            x_t2 = x_t2.reshape(-1)
            adj_matrix = adj_matrix.reshape(-1)
            points = points.reshape(-1, 2)
            edge_index = edge_index.float().to(adj_matrix.device).reshape(2, -1)

        model_output, denoise = self.denoise(model, points, x_t, t, edge_index, x0)
        model_output2, denoise2 = self.denoise(model, points, x_t2, t2, edge_index, x0)
        adj_matrix = adj_matrix.long()
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(denoise, adj_matrix) + loss_func(denoise2, adj_matrix)

        return loss

    def denoise(self, model, points, x_t, t, edge_index, x0):
        x_t = x_t * 2 - 1
        x_t = x_t * (1.0 + 0.05 * torch.rand_like(x_t))
        model_output = model(
            points, x_t, t, edge_index
        )  # B, 2, N, N if not sparse else B*N*K, 2; k is the sparse factor

        c_skip, c_out = [
            self.append_dims(x, model_output.ndim)
            for x in self.get_scalings_for_boundary_condition(t)
        ]  # B, 1, 1, 1 or B*N*K, 1

        if not model.sparse:
            x0 = x0.permute(0, 3, 1, 2)
        else:
            x0 = x0.reshape((-1, 2))
        denoise = c_out * model_output + c_skip * x0

        return model_output, denoise

    # def consistency_test_step(self, model, batch, batch_idx, split="test"):
    #     edge_index = None
    #     np_edge_index = None
    #     device = batch[-1].device
    #     original_edge_index = None

    #     if not model.sparse:
    #         real_batch_idx, points, adj_matrix, gt_tour = batch
    #         np_points = points.cpu().numpy()[0]
    #         np_gt_tour = gt_tour.cpu().numpy()[0]
    #     else:
    #         real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour = batch
    #         route_edge_flags = graph_data.edge_attr
    #         points = graph_data.x
    #         edge_index = graph_data.edge_index
    #         original_edge_index = edge_index.clone()
    #         num_edges = edge_index.shape[1]
    #         batch_size = point_indicator.shape[0]
    #         adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
    #         points = points.reshape((-1, 2))
    #         edge_index = edge_index.reshape((2, -1))
    #         np_points = points.cpu().numpy()
    #         np_gt_tour = gt_tour.cpu().numpy().reshape(-1)
    #         np_edge_index = edge_index.cpu().numpy()

    #     # tsp_solver = TSPEvaluator(np_points)  # np_points: [N, 2] ndarray
    #     # gt_cost = tsp_solver.evaluate(np_gt_tour)  # np_gt_tour: [N+1] ndarray

    #     if model.args.parallel_sampling > 1:
    #         if not model.sparse:
    #             points = points.repeat(model.args.parallel_sampling, 1, 1)
    #         else:
    #             points = points.repeat(model.args.parallel_sampling, 1)
    #             edge_index = model.duplicate_edge_index(
    #                 model.args.parallel_sampling, edge_index, np_points.shape[0], device
    #             )

    #     # Initialize with original diffusion
    #     stacked_tours = []

    #     for _ in range(model.args.sequential_sampling):
    #         xt = torch.randn_like(adj_matrix.float())
    #         if model.args.parallel_sampling > 1:
    #             if not model.sparse:
    #                 xt = xt.repeat(model.args.parallel_sampling, 1, 1)
    #             else:
    #                 xt = xt.repeat(model.args.parallel_sampling, 1)  # [B, E]
    #             xt = torch.randn_like(xt)

    #         xt = (xt > 0).long()

    #         if model.sparse:
    #             xt = xt.reshape(-1)  # [E]

    #         # Diffusion iterations
    #         steps = model.args.inference_diffusion_steps
    #         time_schedule = InferenceSchedule(
    #             inference_schedule=model.args.inference_schedule,
    #             T=model.diffusion.T,
    #             inference_T=steps,
    #         )

    #         for i in range(steps):
    #             t1, t2 = time_schedule(i)
    #             t1 = torch.tensor([t1], device=device).float()

    #             # [B, N, N], heatmap score
    #             xt_scale = xt * 2 - 1
    #             xt_scale = xt_scale * (
    #                 1.0 + 0.05 * torch.rand_like(xt.float(), device=device)
    #             )
    #             x0_pred = model.forward(
    #                 points.float().to(device),  # b*s*n, 2 (4000, 2)
    #                 xt_scale,  # b*s*n*n (200000)
    #                 t1,  # 1
    #                 edge_index.long().to(device) if edge_index is not None else None,
    #             )  # b, 2, n, n

    #             if not model.sparse:
    #                 x0_pred = x0_pred.permute(0, 2, 3, 1).contiguous().softmax(-1)
    #             else:
    #                 x0_pred = x0_pred.reshape((-1, 2)).softmax(
    #                     dim=-1
    #                 )  # n*k, 2; k is the sparse factor

    #             if model.args.use_intermediate:
    #                 adj_mat = x0_pred[..., 1].cpu().detach().numpy() + 1e-6  # [B, N, N]

    #                 tours, merge_iterations = merge_tours(  # [B, N+1], list
    #                     adj_mat,
    #                     np_points,
    #                     np_edge_index,
    #                     sparse_graph=model.sparse,
    #                     parallel_sampling=model.args.parallel_sampling,
    #                 )

    #                 # Refine using 2-opt
    #                 # solver_tours,  [B, N+1] ndarray, the visiting sequence of each city
    #                 solved_tours, ns = batched_two_opt_torch(
    #                     np_points.astype("float64"),
    #                     np.array(tours).astype("int64"),
    #                     max_iterations=model.args.two_opt_iterations,
    #                     device=device,
    #                 )

    #                 stacked_tours.append(solved_tours)

    #             if not t2.item == 0:
    #                 x0 = torch.bernoulli(x0_pred[..., 1].clamp(0, 1))
    #                 x0_onehot = F.one_hot(
    #                     x0.long(), num_classes=2
    #                 ).float()  # [B, N, N, 2]
    #                 Q_bar = (
    #                     torch.from_numpy(model.diffusion.Q_bar[t2])
    #                     .float()
    #                     .to(x0_onehot.device)
    #                 )
    #                 xt_prob = torch.matmul(x0_onehot, Q_bar)  # [B, N, N, 2]
    #                 xt = torch.bernoulli(xt_prob[..., 1].clamp(0, 1))  # [B, N, N]

    #         adj_mat = x0_pred[..., 1].cpu().detach().numpy() + 1e-6  # [B, N, N]

    #         tours, merge_iterations = merge_tours(  # [B, N+1], list
    #             adj_mat,
    #             np_points,
    #             np_edge_index,
    #             sparse_graph=model.sparse,
    #             parallel_sampling=model.args.parallel_sampling,
    #         )

    #         # Refine using 2-opt
    #         # solver_tours,  [B, N+1] ndarray, the visiting sequence of each city
    #         solved_tours, ns = batched_two_opt_torch(
    #             np_points.astype("float64"),
    #             np.array(tours).astype("int64"),
    #             max_iterations=model.args.two_opt_iterations,
    #             device=device,
    #         )

    #         stacked_tours.append(solved_tours)

    #     tsp_solver = TSPEvaluator(np_points)  # np_points: [N, 2] ndarray
    #     gt_cost = tsp_solver.evaluate(np_gt_tour)  # np_gt_tour: [N+1] ndarray

    #     solved_tours = np.concatenate(stacked_tours, axis=0)  # [B, N+1] ndarray
    #     # solved_tours = np.array(stacked_tours)  # [B, N+1] ndarray

    #     all_solved_costs = [
    #         tsp_solver.evaluate(solved_tours[i]) for i in range(solved_tours.shape[0])
    #     ]
    #     best_solved_cost, best_id = np.min(all_solved_costs), np.argmin(
    #         all_solved_costs
    #     )
    #     gap = (best_solved_cost - gt_cost) / gt_cost * 100

    #     g_best_tour = solved_tours[best_id]  # [N+1] ndarray

    #     guided_gap, g_best_solved_cost = -1.0, -1.0

    #     # Local Rewrite
    #     if model.args.rewrite:
    #         g_best_solved_cost = best_solved_cost
    #         for _ in range(model.args.rewrite_steps):
    #             g_stacked_tours = []
    #             # optimal adjacent matrix
    #             g_x0 = model.tour2adj(
    #                 g_best_tour,
    #                 np_points,
    #                 model.sparse,
    #                 model.args.sparse_factor,
    #                 original_edge_index,
    #             )

    #             g_x0 = g_x0.unsqueeze(0).to(device)  # [1, N, N] or [1, N]
    #             if model.args.parallel_sampling > 1:
    #                 if not model.sparse:
    #                     g_x0 = g_x0.repeat(
    #                         model.args.parallel_sampling, 1, 1
    #                     )  # [1, N ,N] -> [B, N, N]
    #                 else:
    #                     g_x0 = g_x0.repeat(model.args.parallel_sampling, 1)

    #             if model.sparse:
    #                 g_x0 = g_x0.reshape(-1)

    #             g_x0_onehot = F.one_hot(
    #                 g_x0.long(), num_classes=2
    #             ).float()  # [B, N, N, 2]
    #             # if self.sparse:
    #             #   g_x0_onehot = g_x0_onehot.unsqueeze(1)

    #             steps_T = int(model.args.diffusion_steps * model.args.rewrite_ratio)

    #             # g_xt = self.diffusion.sample(g_x0_onehot, steps_T)
    #             Q_bar = (
    #                 torch.from_numpy(model.diffusion.Q_bar[steps_T])
    #                 .float()
    #                 .to(g_x0_onehot.device)
    #             )
    #             g_xt_prob = torch.matmul(g_x0_onehot, Q_bar)  # [B, N, N, 2]

    #             t = torch.tensor(
    #                 [model.args.diffusion_steps * model.args.rewrite_ratio]
    #             ).int()

    #             # [1, N, N], denoise, heatmap for edges
    #             if model.args.guided:
    #                 g_x0 = self.guided_denoise_step(
    #                     model, points, g_xt_prob, g_x0, t, device, edge_index
    #                 )
    #             else:
    #                 # add noise for the steps_T samples, namely rewrite
    #                 g_xt = torch.bernoulli(g_xt_prob[..., 1].clamp(0, 1))  # [B, N, N]
    #                 g_x0 = self.denoise_step(model, points, g_xt, t, device, edge_index)

    #             g_adj_mat = g_x0.float().cpu().detach().numpy() + 1e-6
    #             if model.args.save_numpy_heatmap:
    #                 model.run_save_numpy_heatmap(
    #                     g_adj_mat, np_points, real_batch_idx, split
    #                 )

    #             g_tours, g_merge_iterations = merge_tours(
    #                 g_adj_mat,
    #                 np_points,
    #                 np_edge_index,
    #                 sparse_graph=model.sparse,
    #                 parallel_sampling=(
    #                     model.args.parallel_sampling * 2
    #                     if model.args.guided
    #                     else model.args.parallel_sampling
    #                 ),
    #             )

    #             # Refine using 2-opt
    #             g_solved_tours, g_ns = batched_two_opt_torch(
    #                 np_points.astype("float64"),
    #                 np.array(g_tours).astype("int64"),
    #                 max_iterations=model.args.two_opt_iterations,
    #                 device=device,
    #             )

    #             for g_tour in g_solved_tours:
    #                 g_stacked_tours.append(g_tour)

    #             g_solved_tours = np.array(g_stacked_tours)

    #             # tsp_solver = TSPEvaluator(np_points)  # np_points: [N, 2] ndarray
    #             # gt_cost = tsp_solver.evaluate(np_gt_tour)  # np_gt_tour: [N+1] ndarray

    #             g_all_solved_costs = [
    #                 tsp_solver.evaluate(g_solved_tours[i])
    #                 for i in range(g_solved_tours.shape[0])
    #             ]
    #             g_best_solved_cost_tmp, g_best_id = np.min(
    #                 g_all_solved_costs
    #             ), np.argmin(g_all_solved_costs)

    #             if g_best_solved_cost_tmp < g_best_solved_cost:
    #                 g_best_solved_cost = g_best_solved_cost_tmp
    #                 g_best_tour = g_solved_tours[g_best_id]

    #             guided_gap = (g_best_solved_cost - gt_cost) / gt_cost * 100

    #         # print("gap: {}% -> {}%".format(gap, guided_gap))

    #     if model.args.rewrite:
    #         metrics = {
    #             f"{split}/rewrite_ratio": float(model.args.rewrite_ratio),
    #             f"{split}/gap": gap,
    #             f"{split}/guided_gap": guided_gap,
    #             f"{split}/gt_cost": gt_cost,
    #             f"{split}/guided_solved_cost": g_best_solved_cost,
    #         }
    #     else:
    #         metrics = {
    #             f"{split}/gap": gap,
    #             f"{split}/gt_cost": gt_cost,
    #         }

    #     for k, v in metrics.items():
    #         model.log(k, v, on_epoch=True, sync_dist=True)
    #     model.log(
    #         f"{split}/solved_cost",
    #         best_solved_cost,
    #         prog_bar=True,
    #         on_epoch=True,
    #         sync_dist=True,
    #     )

    #     return metrics
    
    # ! 更新版本
    def consistency_test_step(self, model, batch, batch_idx, split="test"):

        logger = logging.getLogger("root")
        inst_start = time.time()

        # 为了识别实例：拿到 idx -> 路径 -> 名称
        try:
            idx_tensor = batch[0]
            idx = int(idx_tensor.view(-1)[0].item())
        except Exception:
            idx = batch_idx
        name = None
        try:
            if getattr(model.args, "use_tsplib", False) and hasattr(model, "test_dataset") and hasattr(model.test_dataset, "paths"):
                tsp_path = model.test_dataset.paths[idx]
                name = os.path.splitext(os.path.basename(tsp_path))[0]
        except Exception:
            pass
        if name is None:
            name = f"inst_{idx}"

        edge_index = None
        np_edge_index = None
        try:
            device = next(model.model.parameters()).device
        except Exception:
            device = torch.device(getattr(model.args, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        # device = batch[-1].device if hasattr(batch[-1], "device") else model.device
        original_edge_index = None

        try:
            # ---------- 识别 TSPLIB 稠密 vs 原 txt ----------
            # TSPLIB: (idx, points_norm[B,N,2], ewt_code[B,], gt_cost_tensor[0 or 1], points_orig[B,N,2])
            # 原 txt 稠密: (idx, points[B,N,2], adj_matrix[B,N,N], gt_tour[B,N+1])
            # 稀疏（原 txt）维持不变
            is_tsplib = False
            ewt_code = None
            gt_cost_from_label = None

            if not model.sparse:
                # 第3个元素是 [B] / [B,1] → TSPLIB；否则是 adj 矩阵 → 原 txt
                if batch[2].dim() <= 2 and batch[2].numel() <= batch[0].shape[0] * 2:
                    # ===== TSPLIB 稠密 =====
                    is_tsplib = True
                    real_batch_idx, points_norm, ewt_tensor, gt_cost_tensor, points_orig = batch

                    # 模型输入：归一化坐标
                    points = points_norm
                    if points.dim() == 2:
                        points = points.unsqueeze(0)

                    # 评测：原坐标
                    np_points_model = points.cpu().numpy()[0]       # 模型/合并/2opt 使用
                    np_points_for_cost = points_orig.cpu().numpy()[0]  # 评测使用

                    ewt_code = int(ewt_tensor[0].item())
                    gt_cost_from_label = (
                        float(gt_cost_tensor[0].item())
                        if gt_cost_tensor.numel() > 0 else None
                    )

                    # 伪造一个 adj_matrix（只为 shape/初始化方便）
                    N = points.shape[1]
                    adj_matrix = torch.zeros((1, N, N), device=device)

                    np_gt_tour = None  # TSPLIB 没有 gt_tour
                else:
                    # ===== 原 txt 稠密 =====
                    real_batch_idx, points, adj_matrix, gt_tour = batch
                    if points.dim() == 2:
                        points = points.unsqueeze(0)
                    np_points_model = points.cpu().numpy()[0]
                    # 如果你保留了未归一化的原坐标，可在这里替换；否则就与原逻辑一致
                    np_points_for_cost = np_points_model
                    np_gt_tour = gt_tour.cpu().numpy()[0]
            else:
                # ===== 原 txt 稀疏 =====（TSPLIB 当前不走稀疏）
                real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour = batch
                route_edge_flags = graph_data.edge_attr
                points = graph_data.x
                edge_index = graph_data.edge_index
                original_edge_index = edge_index.clone()
                num_edges = edge_index.shape[1]
                batch_size = point_indicator.shape[0]
                adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
                points = points.reshape((-1, 2))
                edge_index = edge_index.reshape((2, -1))
                np_points_model = points.cpu().numpy()
                np_points_for_cost = np_points_model
                np_gt_tour = gt_tour.cpu().numpy().reshape(-1)

            # ---------- 并行采样准备 ----------
            if model.args.parallel_sampling > 1:
                if not model.sparse:
                    points = points.repeat(model.args.parallel_sampling, 1, 1)
                else:
                    points = points.repeat(model.args.parallel_sampling, 1)
                    edge_index = model.duplicate_edge_index(
                        model.args.parallel_sampling, edge_index, np_points_model.shape[0], device
                    )

            stacked_tours = []

            # ---------- 采样主循环 ----------
            for _ in range(model.args.sequential_sampling):
                xt = torch.randn_like(adj_matrix.float())
                if model.args.parallel_sampling > 1:
                    if not model.sparse:
                        xt = xt.repeat(model.args.parallel_sampling, 1, 1)
                    else:
                        xt = xt.repeat(model.args.parallel_sampling, 1)
                    xt = torch.randn_like(xt)
                xt = (xt > 0).long()
                if model.sparse:
                    xt = xt.reshape(-1)

                steps = model.args.inference_diffusion_steps
                time_schedule = InferenceSchedule(
                    inference_schedule=model.args.inference_schedule,
                    T=model.diffusion.T,
                    inference_T=steps,
                )

                for i in range(steps):
                    t1, t2 = time_schedule(i)
                    t1 = torch.tensor([t1], device=device).float()

                    xt_scale = xt * 2 - 1
                    xt_scale = xt_scale * (1.0 + 0.05 * torch.rand_like(xt.float(), device=device))

                    # 注意：模型输入是归一化坐标 points
                    x0_pred = model.forward(
                        points.float().to(device),
                        xt_scale.to(device),
                        t1,
                        edge_index.long().to(device) if edge_index is not None else None,
                    )

                    if not model.sparse:
                        x0_pred = x0_pred.permute(0, 2, 3, 1).contiguous().softmax(-1)
                    else:
                        x0_pred = x0_pred.reshape((-1, 2)).softmax(dim=-1)

                    if model.args.use_intermediate:
                        adj_mat = x0_pred[..., 1].cpu().detach().numpy() + 1e-6
                        tours, _ = merge_tours(
                            adj_mat,
                            np_points_model,
                            np_edge_index,
                            sparse_graph=model.sparse,
                            parallel_sampling=model.args.parallel_sampling,
                        )

                        # Refine using 2-opt
                        # solver_tours,  [B, N+1] ndarray, the visiting sequence of each city
                        solved_tours, _ = batched_two_opt_torch(
                            np_points_model.astype("float64"),
                            np.array(tours).astype("int64"),
                            max_iterations=model.args.two_opt_iterations,
                            device=device,
                        )
                        stacked_tours.append(solved_tours)

                    if not t2.item() == 0:
                        x0 = torch.bernoulli(x0_pred[..., 1].clamp(0, 1))
                        x0_onehot = F.one_hot(x0.long(), num_classes=2).float()
                        Q_bar = torch.from_numpy(model.diffusion.Q_bar[t2]).float().to(x0_onehot.device)
                        xt_prob = torch.matmul(x0_onehot, Q_bar)
                        xt = torch.bernoulli(xt_prob[..., 1].clamp(0, 1))

                # 最后一轮 heatmap → 合并 → 2-opt
                adj_mat = x0_pred[..., 1].cpu().detach().numpy() + 1e-6
                tours, _ = merge_tours(
                    adj_mat,
                    np_points_model,
                    np_edge_index,
                    sparse_graph=model.sparse,
                    parallel_sampling=model.args.parallel_sampling,
                )

                # Refine using 2-opt
                # solver_tours,  [B, N+1] ndarray, the visiting sequence of each city
                solved_tours, _ = batched_two_opt_torch(
                    np_points_model.astype("float64"),
                    np.array(tours).astype("int64"),
                    max_iterations=model.args.two_opt_iterations,
                    device=device,
                )
                stacked_tours.append(solved_tours)

            solved_tours = np.concatenate(stacked_tours, axis=0)  # [S, N+1]

            # ---------- 评测：原坐标 + 取整 ----------
            ewt_for_eval = ewt_code if is_tsplib else None
            tsp_solver = TSPEvaluator(np_points_for_cost, edge_weight_type=ewt_for_eval)

            all_solved_costs = [tsp_solver.evaluate(solved_tours[i]) for i in range(solved_tours.shape[0])]
            best_solved_cost = float(np.min(all_solved_costs))
            best_id = int(np.argmin(all_solved_costs))
            g_best_tour = solved_tours[best_id]  # [N+1]

            # Ground truth
            if is_tsplib:
                gt_cost = gt_cost_from_label
            else:
                gt_cost = tsp_solver.evaluate(np_gt_tour)

            # ---------- Local Rewrite（同样用原坐标评测） ----------
            guided_gap, g_best_solved_cost = None, None
            if getattr(model.args, "rewrite", False):
                g_best_solved_cost = best_solved_cost

                for _ in range(model.args.rewrite_steps):
                    g_stacked_tours = []

                    # 当前最优 tour → 邻接
                    g_x0 = model.tour2adj(
                        g_best_tour,
                        np_points_model,         # 这里仍用模型坐标来构造邻接/噪声（与采样坐标系一致）
                        model.sparse,
                        model.args.sparse_factor,
                        original_edge_index,
                    )
                    g_x0 = g_x0.unsqueeze(0).to(device)  # [1, N, N] 或 [1, E]
                    if model.args.parallel_sampling > 1:
                        if not model.sparse:
                            g_x0 = g_x0.repeat(model.args.parallel_sampling, 1, 1)
                        else:
                            g_x0 = g_x0.repeat(model.args.parallel_sampling, 1)

                    if model.sparse:
                        g_x0 = g_x0.reshape(-1)

                    g_x0_onehot = F.one_hot(g_x0.long(), num_classes=2).float()

                    steps_T = int(model.args.diffusion_steps * model.args.rewrite_ratio)

                    Q_bar = torch.from_numpy(model.diffusion.Q_bar[steps_T]).float().to(g_x0_onehot.device)
                    g_xt_prob = torch.matmul(g_x0_onehot, Q_bar)  # [B, N, N, 2] or [B*E, 2]

                    t = torch.tensor([steps_T]).int()

                    # 引导 or 非引导
                    if getattr(model.args, "guided", False):
                        g_x0_like = self.guided_denoise_step(
                            model, points, g_xt_prob, g_x0, t, device, edge_index
                        )  # 返回的是拼接的 prob 向量，这里直接取后半段作为更新后的 prob
                        # # 上面 guided_denoise_step 的返回在你的实现是 concat(p_theta[...,1], x0_pred_prob_u[...,1])
                        # # 这里更稳妥的是直接重新走一次非引导 denoise（或者你按现有实现继续使用 g_xt_prob_u）
                        # # 为避免歧义，简化为：取非引导路径
                        # g_xt = torch.bernoulli(g_xt_prob[..., 1].clamp(0, 1))
                        # g_x0_like = self.denoise_step(model, points, g_xt, t, device, edge_index)
                    else:
                        g_xt = torch.bernoulli(g_xt_prob[..., 1].clamp(0, 1))
                        g_x0_like = self.denoise_step(model, points, g_xt, t, device, edge_index)

                    # heatmap → 合并 → 2-opt（仍在模型坐标系）
                    if not model.sparse:
                        g_adj_mat = g_x0_like.cpu().detach().numpy() + 1e-6
                    else:
                        g_adj_mat = g_x0_like.cpu().detach().numpy() + 1e-6

                    if getattr(model.args, "save_numpy_heatmap", False):
                        model.run_save_numpy_heatmap(g_adj_mat, np_points_model, real_batch_idx, split)

                    g_tours, _ = merge_tours(
                        g_adj_mat,
                        np_points_model,
                        np_edge_index,
                        sparse_graph=model.sparse,
                        parallel_sampling=model.args.parallel_sampling,
                    )

                    # Refine using 2-opt
                    g_solved_tours, _ = batched_two_opt_torch(
                        np_points_model.astype("float64"),
                        np.array(g_tours).astype("int64"),
                        max_iterations=model.args.two_opt_iterations,
                        device=device,
                    )

                    # 评测：原坐标 + 取整
                    g_all_costs = [tsp_solver.evaluate(g_solved_tours[i]) for i in range(g_solved_tours.shape[0])]
                    cand_best = float(np.min(g_all_costs))
                    cand_id = int(np.argmin(g_all_costs))

                    if cand_best < g_best_solved_cost:
                        g_best_solved_cost = cand_best
                        g_best_tour = g_solved_tours[cand_id]

                if gt_cost is not None and gt_cost > 0:
                    guided_gap = (g_best_solved_cost - gt_cost) / gt_cost * 100.0

            # ==== 统计与日志（成功案例）====
            inst_time = time.time() - inst_start
            N = int(np_points_for_cost.shape[0])
            model.icam_stats["all_instance_num"] += 1
            model.icam_stats["all_solved_instance_num"] += 1

            gap = None
            if gt_cost is not None and gt_cost > 0:
                gap = (best_solved_cost - float(gt_cost)) / float(gt_cost) * 100.0

            # 分桶
            bucket_key = "[0,1000)" if N < 1000 else "[1000,10000)" if N < 10000 else "[10000,100000]"
            if gap is not None:
                model.icam_stats["bucket"][bucket_key].append(float(gap))

            model.icam_stats["bucket_time"][bucket_key].append(float(inst_time))

            # 保存 per-instance 记录
            model.icam_stats["per_instance"].append({
                "name": name,
                "N": N,
                "optimal": float(gt_cost) if gt_cost is not None else None,
                "cost": best_solved_cost,
                "gap": float(gap) if gap is not None else None,
                "time": inst_time,
            })

            # === ICAM 风格 per-instance 日志 ===
            logger.info("===============================================================")
            logger.info(f"Instance name: {name}, problem_size: {N}")
            if gt_cost is not None:
                logger.info(f"Instance name: {name}, optimal score: {float(gt_cost):.4f}")
            logger.info(f"No aug score:{best_solved_cost:.3f}, No aug gap:{(gap if gap is not None else float('nan')):.3f}%")
            logger.info(f"Instance time (incl. dist+test): {inst_time:.3f}s")
            # 若做了 rewrite，可追加 guided 结果：
            # logger.info(f"Aug score:{g_best_solved_cost:.3f}, Aug gap:{guided_gap:.3f}%")

            # 返回空字典即可（我们不用 pl 的聚合），或者返回 metrics（不再依赖 wandb）
            return {}
        
        except Exception as e:
            # 失败案例：只记 all_instance_num + 打印错误，不中断
            model.icam_stats["all_instance_num"] += 1
            logger.info(f"Error occurred in instance {name}, skip it!")
            logger.info(f"Error message: {e}")
            logger.info("---------------------------------------------------------------")
            return {}

        # # ---------- 记日志 ----------
        # metrics = {f"{split}/solved_cost": best_solved_cost}

        # if gt_cost is not None and gt_cost > 0:
        #     gap = (best_solved_cost - gt_cost) / gt_cost * 100.0
        #     metrics[f"{split}/gap"] = gap
        #     metrics[f"{split}/gt_cost"] = float(gt_cost)

        # if getattr(model.args, "rewrite", False):
        #     if g_best_solved_cost is not None:
        #         metrics[f"{split}/guided_solved_cost"] = g_best_solved_cost
        #     if guided_gap is not None:
        #         metrics[f"{split}/guided_gap"] = guided_gap
        #     metrics[f"{split}/rewrite_ratio"] = float(model.args.rewrite_ratio)

        # for k, v in metrics.items():
        #     model.log(k, v, on_epoch=True, sync_dist=True, prog_bar=(k.endswith("solved_cost")))

        return metrics


    def denoise_step(self, model, points, xt, t, device, edge_index=None):
        with torch.no_grad():
            xt = xt.float()  # b, n, n
            xt_scale = xt * 2 - 1
            xt_scale = xt_scale * (
                1.0 + 0.05 * torch.rand_like(xt.float(), device=device)
            )

            # [b, 2, n, n]
            x0_pred = model.forward(
                points.float().to(device),
                xt_scale.to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
            )

            if not model.sparse:
                x0_pred_prob = (
                    x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
                )
            else:
                x0_pred_prob = x0_pred.reshape((-1, 2)).softmax(dim=-1)

            return x0_pred_prob[..., 1]

    def guided_denoise_step(
        self, model, points, xt_prob, x0, t, device, edge_index=None
    ):
        # xt_prob [B, N, N, 2]
        torch.set_grad_enabled(True)

        # straight-through gumbel-Softmax
        xt_prob.requires_grad = True
        xt = xt_prob[..., 1].float()
        # xt = F.gumbel_softmax(xt_prob, tau=1, hard=True, dim=-1)[...,1]
        xt = xt.float()

        # b, n, n
        # xt.requires_grad = True

        with torch.inference_mode(False):
            xt_scale = xt * 2 - 1
            xt_scale = xt_scale * (
                1.0 + 0.05 * torch.rand_like(xt.float(), device=device)
            )
            # [b, 2, n, n]
            x0_pred = model.forward(
                points.float().to(device),
                xt_scale.to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
            )

            # Compute loss
            loss_func = nn.CrossEntropyLoss()
            bce_loss = loss_func(x0_pred, x0.long())

            if not model.sparse:
                p_theta = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
            else:
                p_theta = x0_pred.reshape((-1, 2)).softmax(dim=-1)

            if not model.sparse:
                dis_matrix = model.points2adj(points)
                cost_est = (dis_matrix * p_theta[..., 1]).sum()
                # cost_est.requires_grad_(True)
                # cost_est.backward()
            else:
                dis_matrix = torch.sqrt(
                    torch.sum(
                        (points[edge_index.T[:, 0]] - points[edge_index.T[:, 1]]) ** 2,
                        dim=1,
                    )
                )

                cost_est = (dis_matrix * p_theta[..., 1]).sum()
                # cost_est.requires_grad_(True)
                # cost_est.backward()

            loss = self.args.c1 * bce_loss + self.args.c2 * cost_est
            loss.backward()
            # if model.args.norm is True:
            #     xt.grad = torch.nn.functional.normalize(xt.grad, p=2, dim=-1)
        torch.set_grad_enabled(False)
        assert xt_prob.grad is not None

        # compute p_phi
        with torch.no_grad():
            # scale = 50
            p_phi = torch.exp(-xt_prob.grad)
            if model.sparse:
                p_phi = p_phi.reshape(p_theta.shape)
            xt_prob_u = (xt_prob * p_phi) / torch.sum(
                (xt_prob * p_phi), dim=-1, keepdim=True
            )

            xt_u = torch.bernoulli(xt_prob_u[..., 1].clamp(0, 1))
            xt_u = xt_u.float()  # b, n, n
            xt_scale_u = xt_u * 2 - 1
            xt_scale_u = xt_scale_u * (
                1.0 + 0.05 * torch.rand_like(xt_u.float(), device=device)
            )  # b*n

            # [b, 2, n, n]
            x0_pred_u = model.forward(
                points.float().to(device),
                xt_scale_u.to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
            )

            if not model.sparse:
                x0_pred_prob_u = (
                    x0_pred_u.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
                )
            else:
                x0_pred_prob_u = x0_pred_u.reshape((-1, 2)).softmax(dim=-1)
            output = torch.concat((p_theta[..., 1], x0_pred_prob_u[..., 1]), dim=0)
            # output = torch.concat((x0_pred_prob_u[..., 1], x0_pred_prob_u[..., 1]), dim=0)
            # output = torch.concat((p_theta[..., 1], p_theta[..., 1]), dim=0)
            return output
