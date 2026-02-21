import torch

import os
from logging import getLogger

from TSP.TSPEnv import TSPEnv as Env
from TSP.TSPModel_DRHG import TSPModel as Model_DRHG
from TSP.TSPModel_DRHG_aug import TSPModel as Model_DRHG_rp
from utils.utils import *
from utils_for_tester import assemble_solution_for_sorted_problem_batch

from TSP.random_insertion import random_insertion_tsp
from LIBUtils import TSPLIBReader, tsplib_cost

import random
import numpy as np
import time

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(1234)

class TSPTester():
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params,
                 ):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        self.model_params = model_params
        self.iter_budget = tester_params['iter_budget']
        self.destroy_mode    = tester_params['destroy_mode']
        self.destroy_params  = tester_params['destroy_params']

        # ! 取消输入
        # self.initial_solution_path = tester_params['initial_solution_path']
        self.initial_solution_path = None

        # result folder, logger
        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()
        self.knn_k_high = int(self.destroy_params[self.destroy_mode[0]]['knn_k'][1])
        self.recordings = {'tsp_name':[],'solution':[]}

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            # torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        if self.env_params['use_model'] == 'DRHG':
            self.model = Model_DRHG(**self.model_params)
            self.env = Env(**self.env_params)
        elif self.env_params['use_model'] == 'DRHG_rp':
            self.model = Model_DRHG_rp(**self.model_params)
            self.env = Env(**self.env_params)
        else: raise NotImplementedError("{} not implemented".format(self.env_params['use_model']))

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        torch.set_printoptions(precision=20)

        # utility
        self.time_estimator = TimeEstimator()
        self.time_estimator_2 =  TimeEstimator()

    def run(self):
        self.time_estimator.reset()
        self.time_estimator_2.reset()

        # if self.env_params['load_way']=='allin':
        #     self.env.load_raw_data(self.tester_params['test_episodes'] )

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        aug_score_AM = AverageMeter()
        gap_AM = AverageMeter() 

        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        gap_log_all = []

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, score_student_mean, aug_score, problems_size, gap, gap_log = self._test_one_batch(episode, batch_size, clock=self.time_estimator_2)

            gap_log_all.extend(gap_log)
            score_AM.update(score, batch_size)
            score_student_AM.update(score_student_mean, batch_size)
            aug_score_AM.update(aug_score, batch_size)
            gap_AM.update(gap, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d} /{:3d}, Elapsed[{}], Remain[{}], score:{:.3f},Score_studetnt: {:.4f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score,score_student_mean, aug_score))
            self.logger.info("===============================================================")

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info("===============================================================")
                self.logger.info("===============================================================")
                self.logger.info("===============================================================")
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" NO-AUG SCORE student: {:.4f} ".format(score_student_AM.avg))
                self.logger.info(" Gap: {:.4f}%".format((score_student_AM.avg-score_AM.avg) / score_AM.avg * 100))
                gap_ = (score_student_AM.avg-score_AM.avg) / score_AM.avg * 100
                torch.save(self.recordings, 'tsplib_result_b{}.pt'.format(self.iter_budget))
                self.logger.info('Result saved.')

        return score_AM.avg, score_student_AM.avg, gap_, gap_log_all
    
    # ! 新增
    def run_lib(self, lib_path, scale_ranges=None):
        """
        支持:
        - 单一区间: [0, 100001]
        - 多个区间: [[0,1000], [1000,10000], [10000,100001]]
        会对每个区间分别统计，并输出总汇总。
        """
        self.time_estimator.reset()

        # 1) 规范化 scale_ranges 入参
        if scale_ranges is None:
            scale_ranges = [[0, 100001]]
        elif isinstance(scale_ranges[0], (int, float)):     # 传入 [a,b]
            scale_ranges = [scale_ranges]                   # 变成 [[a,b]]
        # 此时一定是 list[list]

        grand_total = 0
        grand_solved = 0

        grand_gap_sum = 0.0
        grand_gap_cnt = 0

        grand_start = time.time()

        # 2) 逐区间处理
        for rng in scale_ranges:
            if (not isinstance(rng, (list, tuple))) or len(rng) != 2:
                self.logger.info(f"[warn] bad range: {rng}, skip.")
                self.logger.info("===============================================================")
                continue
            
            per_inst_times = []  # [(name, dimension, time_sec), ...]
            per_inst_gaps  = []

            start_t = time.time()
            total = 0
            solved = 0

            # ——范围内逐实例——
            for root, _, files in os.walk(lib_path):
                for file in files:
                    if not file.endswith(".tsp"):
                        continue

                    name, dimension, locs, ew_type = TSPLIBReader(os.path.join(root, file))
                    if name is None:
                        continue

                    # 只看当前区间
                    if not (rng[0] <= dimension < rng[1]):
                        continue

                    optimal = tsplib_cost.get(name, None)
                    if optimal is None:
                        self.logger.info(f"[skip] optimal of {name} not found in tsplib_cost")
                        self.logger.info("===============================================================")
                        continue

                    # 构造 dict_instance_info
                    instance_xy = np.array(locs, dtype=np.float32)             # (N,2)
                    node_coord  = torch.from_numpy(instance_xy).unsqueeze(0)   # [1,N,2]

                    # 归一化到 [0,1]
                    xy_max = torch.max(node_coord, dim=1, keepdim=True).values
                    xy_min = torch.min(node_coord, dim=1, keepdim=True).values
                    ratio  = torch.max((xy_max - xy_min), dim=-1, keepdim=True).values
                    ratio[ratio == 0] = 1
                    nodes_xy_normalized = (node_coord - xy_min) / ratio.expand(-1, 1, 2)

                    dict_instance_info = {
                        'name': name,
                        'optimal': float(optimal),
                        'problem_size': dimension,
                        'original_node_xy_lib': node_coord,   # 原坐标
                        'node_xy': nodes_xy_normalized,       # 归一化坐标
                        'edge_weight_type': ew_type
                    }

                    total += 1

                    # ——每个实例开始前——
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    inst_start = time.time()
                    try:
                        score, gap = self._test_one_instance_lib(dict_instance_info)
                        solved += 1

                        # ——每个实例结束后——
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        inst_time = time.time() - inst_start
                        per_inst_times.append((name, dimension, inst_time))
                        per_inst_gaps.append(float(gap))

                        self.logger.info(f"[{name}] score={score:.3f}, opt={optimal:.3f}, gap={gap:.3f}%, time={inst_time:.3f}s")
                        self.logger.info("===============================================================")
                    except Exception as e:
                        self.logger.info(f"[error] {name}: {e}")
                        self.logger.info("===============================================================")

            # ——区间统计输出——
            elapsed = time.time() - start_t

            # 仅对“成功求解的实例”的时间做平均
            if len(per_inst_times) > 0:
                solved_time_sum = sum(t for (_, _, t) in per_inst_times)
                solved_time_avg = solved_time_sum / len(per_inst_times)
            else:
                solved_time_sum = 0.0
                solved_time_avg = 0.0

            # 1) throughput 型平均（区间总耗时/成功个数）：包含遍历/跳过/日志等开销
            throughput_avg = elapsed / max(solved, 1)

            if len(per_inst_gaps) > 0:
                avg_gap = sum(per_inst_gaps) / len(per_inst_gaps)
            else:
                avg_gap = 0.0

            self.logger.info(
                f"[Range {rng[0]}, {rng[1]}) done. solved {solved}/{total}, "
                f"total {elapsed:.2f}s, avg_throughput {throughput_avg:.2f}s/inst, "
                f"avg_solved_only {solved_time_avg:.2f}s/inst, avg_gap {avg_gap:.3f}%"
            )
            self.logger.info("===============================================================")

            grand_total  += total
            grand_solved += solved
            grand_gap_sum += sum(per_inst_gaps)   # <-- 新增
            grand_gap_cnt += len(per_inst_gaps)   # <-- 新增

        # 3) 总汇总
        grand_elapsed = time.time() - grand_start
        overall_avg_gap = (grand_gap_sum / grand_gap_cnt) if grand_gap_cnt > 0 else 0.0  # <-- 新增
        self.logger.info(
            f"LIB all ranges done. solved {grand_solved}/{grand_total}, "
            f"total {grand_elapsed:.2f}s, avg {grand_elapsed/max(grand_solved,1):.2f}s, "
            f"overall_avg_gap {overall_avg_gap:.3f}%"
        )
        self.logger.info("===============================================================")

    
    # ! 新增
    def _test_one_instance_lib(self, dict_instance_info):
        """
        在 LIB/ICAM 格式的数据上跑 1 个实例（batch 固定为 1）：
        1) Env 从 dict_instance_info 载入坐标/最优值/edge_weight_type
        2) 用 Random Insertion 生成初始解
        3) 按 destroy-repair 超图流程迭代改进
        4) 返回 (当前最佳长度, gap%)
        """
        device = self.device
        self.model.eval()

        from TSP.random_insertion import random_insertion_tsp
        import torch, random

        with torch.no_grad():
            # ---------- 1) 载入实例 ----------
            self.env.load_problem_from_lib(dict_instance_info, device=device)
            self.origin_problem = self.env.problems.clone()
            B = 1  # 本方法按实例逐个跑

            # 最优值（TSPLIB 口径）
            self.optimal_length, tsplib_name = self.env._get_travel_distance_2(
                self.origin_problem,
                torch.zeros((B, self.env.problem_size), dtype=torch.long, device=device),
                test_in_tsplib=True,
                need_optimal=True
            )

            # ---------- 2) RI 初始解 ----------
            best_solution = random_insertion_tsp(self.origin_problem).long()  # [1, N]
            self.env.selected_node_list = best_solution.clone()
            current_best_length = self.env._get_travel_distance_2(
                self.origin_problem, best_solution, test_in_tsplib=True
            )

            # 日志
            # init_gap = ((current_best_length.mean() - self.optimal_length.mean()) /
            #             self.optimal_length.mean() * 100).item()
            # escape_time, _ = self.time_estimator_2.get_est_string(1, 1)
            # self.logger.info("===============================================================")
            # self.logger.info(
            #     "initial solution, gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
            #         init_gap, escape_time,
            #         current_best_length.mean().item(),
            #         self.optimal_length.mean().item()
            #     )
            # )
            # self.logger.info("===============================================================")

            # ---------- 3) 破坏-修复超图流程 ----------
            destroy_mode = self.destroy_mode[0]
            destroy_params = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                            for k, v in self.destroy_params[destroy_mode].items()}  # 浅拷贝，避免外部被改

            # 运行前初始化 env
            self.env.problems = self.origin_problem
            self.env.problem_size = self.origin_problem.size(1)
            self.env.solution = best_solution.clone()
            self.logger.info('problem size: {}'.format(self.env.problem_size))

            # 动态收紧 knn_k 上界：min(0.75*n, 预设上限)
            if destroy_mode == 'knn-location':
                hi = int(destroy_params['knn_k'][1])
                hi = min(int(self.env.problem_size * 0.75), self.knn_k_high, hi)
                destroy_params['knn_k'] = [int(destroy_params['knn_k'][0]), int(hi)]

            # 预采样等距中心网格
            iter_budget = int(self.iter_budget)
            num_interval = torch.sqrt(torch.tensor(iter_budget, device=device, dtype=torch.float32)).long().item()
            num_interval = max(1, num_interval)
            center_x = (torch.arange(num_interval, device=device, dtype=torch.float32) + 0.5) / num_interval
            center_y = (torch.arange(num_interval, device=device, dtype=torch.float32) + 0.5) / num_interval

            gap_log = []
            for step_idx in range(iter_budget):
                # 3.1 选择中心（尊重 center_type，不再重复覆盖）
                ct = destroy_params.get('center_type', 'equally')
                if ct == 'equally':
                    cx = center_x[torch.randint(low=0, high=center_x.numel(), size=(1,), device=device)].item()
                    cy = center_y[torch.randint(low=0, high=center_y.numel(), size=(1,), device=device)].item()
                elif ct == 'random':
                    cx = random.random()
                    cy = random.random()
                else:
                    raise NotImplementedError(f"center_type {ct} not implemented")
                destroy_params['center_location'] = (cx, cy)

                # 3.2 采样 reduced problem（注意：函数内部使用的张量设备/类型）
                (destruction_mask, reduced_problems, endpoint_mask, another_endpoint, point_couples,
                padding_mask, new_problem_index_on_sorted_problem, sorted_problems, shift) = \
                    self.env.sampling_reduced_problems(
                        destroy_mode, destroy_params,
                        return_sorted_problem=True, if_return=True, norm_p=2
                    )

                # 3.3 准备重建循环
                reward, done = self.env.reset()
                selected_teacher_all = torch.ones((B, 0), dtype=torch.long, device=device)
                selected_student_all = torch.ones((B, 0), dtype=torch.long, device=device)
                state, reward, reward_student, done = self.env.pre_step()

                if self.tester_params.get('coordinate_transform', False):
                    state.data = self.env.coordinate_transform(state.data.clone())
                    # self.logger.info('coordinate_transform imposed.')

                # 3.4 在 reduced problem 上逐步解码
                current_step = 0
                while not done:
                    if current_step == 0:
                        selected_teacher = torch.zeros((B,), dtype=torch.long, device=device)
                        selected_student = selected_teacher
                        last_selected = selected_student
                        last_is_second_endpoint = torch.zeros((B,), dtype=torch.bool, device=device)
                    else:
                        last_is_padding = padding_mask[:, current_step - 1].to(device=device, dtype=torch.bool)
                        last_is_endpoint = torch.gather(endpoint_mask, dim=1,
                                                        index=last_selected.unsqueeze(1)).squeeze(1).to(device=device)
                        connect_to_another_endpoint = last_is_endpoint & (~last_is_second_endpoint)

                        selected_teacher, _, _, selected_student = self.model(
                            state,
                            self.env.selected_node_list,
                            self.env.solution,
                            current_step,
                            point_couples=point_couples,
                            endpoint_mask=endpoint_mask
                        )

                        # 端点强制接续
                        selected_student[connect_to_another_endpoint] = \
                            another_endpoint.gather(index=last_selected.unsqueeze(1), dim=1).squeeze(1)[
                                connect_to_another_endpoint
                            ]
                        # padding 位置：用步号占位
                        selected_student[last_is_padding] = current_step
                        selected_teacher = selected_student
                        last_selected = selected_student
                        last_is_second_endpoint = connect_to_another_endpoint

                    current_step += 1
                    selected_teacher_all = torch.cat((selected_teacher_all, selected_teacher[:, None]), dim=1)
                    selected_student_all = torch.cat((selected_student_all, selected_student[:, None]), dim=1)
                    state, reward, reward_student, done = self.env.step(selected_teacher, selected_student)

                # 3.5 还原到完整解的索引
                _, complete_solution_on_sorted_problem_batch = assemble_solution_for_sorted_problem_batch(
                    destruction_mask.to(device=device, dtype=torch.long),
                    endpoint_mask.to(device=device, dtype=torch.bool),
                    self.env.selected_node_list.to(device=device, dtype=torch.long),
                    new_problem_index_on_sorted_problem.to(device=device, dtype=torch.long),
                    padding_mask.to(device=device, dtype=torch.bool)
                )

                # 处理整体循环位移
                if destroy_mode != 'knn-location':
                    best_solution = torch.roll(best_solution, shifts=int(shift), dims=1)
                else:
                    problem_size = best_solution.size(1)
                    shift_ist_by_ist = shift.to(device=device).long().unsqueeze(-1)  # [B,1]
                    base = torch.arange(problem_size, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
                    shifted_index = base + shift_ist_by_ist
                    shifted_index = torch.remainder(shifted_index, problem_size)
                    best_solution = best_solution.gather(index=shifted_index, dim=1)

                current_solution = best_solution.gather(1, index=complete_solution_on_sorted_problem_batch.long())
                current_length = self.env._get_travel_distance_2(
                    self.origin_problem, current_solution, test_in_tsplib=True
                )
                is_better = current_length < (current_best_length - 1e-6)

                # self.logger.info("improved: {}".format(torch.sum(is_better).item()))
                best_solution[is_better, :] = current_solution[is_better, :]
                current_best_length[is_better] = current_length[is_better]

                # 重置 env 到完整问题以进入下一轮破坏
                self.env.problems = self.origin_problem
                self.env.problem_size = self.origin_problem.size(1)
                self.env.solution = best_solution

                escape_time, _ = self.time_estimator_2.get_est_string(1, 1)
                cur_gap = ((current_best_length.mean() - self.optimal_length.mean()) /
                        self.optimal_length.mean()).item() * 100
                # self.logger.info(
                #     "repair step {},  gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                #         step_idx, cur_gap, escape_time,
                #         current_best_length.mean().item(),
                #         self.optimal_length.mean().item()
                #     )
                # )

            # ---------- 4) 返回结果 ----------
            final_len = self.env._get_travel_distance_2(
                self.origin_problem, best_solution, test_in_tsplib=True
            ).mean().item()
            final_gap = ((final_len - self.optimal_length.mean().item()) /
                        self.optimal_length.mean().item()) * 100.0

            # 记录
            self.recordings['tsp_name'].append(tsplib_name)
            self.recordings['solution'].append(best_solution.squeeze(0).unsqueeze(0))

            return final_len, final_gap



    def _test_one_batch(self, episode, batch_size, clock=None):

        # Ready
        ###############################################
        self.model.eval()

        with torch.no_grad():

            # load problem and optimal solution
            self.env.load_problems(episode, batch_size)
            self.origin_problem = self.env.problems
            reward, done = self.env.reset()
            self.optimal_length, tsplib_name = self.env._get_travel_distance_2(self.origin_problem, self.env.solution, test_in_tsplib=True, need_optimal=True)
            self.optimal_length, tsplib_name = self.optimal_length[episode], tsplib_name[episode]

            # # load initial solution
            # RI_solutions = torch.load(self.initial_solution_path, map_location=self.device)[1]
            # self.env.selected_node_list = torch.tensor(RI_solutions[episode:episode + batch_size][0]).unsqueeze(0)
            # best_solution = self.env.selected_node_list.clone().long()   #self.env.selected_node_list
            # current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_solution, test_in_tsplib=True)
            # ! 初始化修改为代码生成，而非导入
            # coords: [B, N, 2] already on self.device
            coords = self.origin_problem  # TSPLIB 流程中已是 cuda().float()
            best_solution = random_insertion_tsp(coords).long()           # [B, N], dtype long
            self.env.selected_node_list = best_solution.clone()           # 作为当前解写入 env
            # 计算当前解长度（用于日志 & 初始 gap）
            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_solution, test_in_tsplib=True)

            escape_time, _ = clock.get_est_string(1, 1)
            self.logger.info("initial solution, gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100, escape_time,
            current_best_length.mean().item(), self.optimal_length.mean().item()))
            self.logger.info("===============================================================")

            B_V = batch_size * 1
            ########################################## destroy and repair ########################################
            current_solution = torch.zeros(best_solution.size(), dtype=int)
            # set env
            self.env.problems = self.origin_problem
            self.env.problem_size = self.origin_problem.size(1)
            self.env.solution = best_solution
            self.solution_1 = self.env.solution.clone()
            self.logger.info('problem size: {}'.format(self.env.problem_size))

            destroy_mode = self.destroy_mode[0]
            destroy_params = self.destroy_params[destroy_mode] # 
            destroy_params['knn_k'][1] = min(int(self.env.problem_size * 0.75), self.knn_k_high)

            
            # pre step of destruction
            iter_budget = self.iter_budget
            num_interval = torch.sqrt(torch.tensor(iter_budget)).long()
            center_x = (torch.arange(num_interval) + 0.5)/ num_interval
            center_y = (torch.arange(num_interval) + 0.5) / num_interval
            gap_log = []
            for bbbb in range(iter_budget):
                if destroy_params['center_type'] == "equally":
                    destroy_params['center_location'] = (random.choice(center_x), random.choice(center_y))
                elif destroy_params['center_type'] == "random":
                    destroy_params['center_location'] = (random.uniform(0, 1), random.uniform(0, 1))
                else: 
                    raise NotImplementedError("center_type {} not implemented".format(destroy_params['center_type']))

                # sampling reduced_problem      
                destroy_params['center_location'] = (random.choice(center_x), random.choice(center_y))
                destruction_mask, reduced_problems, endpoint_mask, another_endpoint, point_couples, padding_mask, new_problem_index_on_sorted_problem, sorted_problems, shift = \
                                            self.env.sampling_reduced_problems(destroy_mode, destroy_params, return_sorted_problem=True, if_return=True, norm_p=2 )


                # pre step of reconstruction
                reward, done = self.env.reset() 
                selected_teacher_all = torch.ones(size=(B_V,  0),dtype=torch.int)
                selected_student_all = torch.ones(size=(B_V,  0),dtype=torch.int)               
                state, reward, reward_student, done = self.env.pre_step()  
                if self.tester_params['coordinate_transform']:
                    state.data = self.env.coordinate_transform(state.data.clone())
                    self.logger.info('coordinate_transform imposed.')

                # get solution on reduced problem
                current_step = 0
                while not done:
                    if current_step == 0:
                        selected_teacher= torch.zeros((batch_size),dtype=torch.int64)  # B_V = 1
                        selected_student = selected_teacher
                        last_selected = selected_student
                        last_is_second_endpoint = torch.zeros((batch_size),dtype=bool)

                    else:
                        last_is_padding = padding_mask[:,current_step-1]
                        last_is_endpoint = torch.gather(endpoint_mask, dim=1, index=last_selected.unsqueeze(1)).squeeze()
                        connect_to_another_endpoint = last_is_endpoint & (~last_is_second_endpoint)
                        selected_teacher, _, _, selected_student = self.model(state, 
                                                                              self.env.selected_node_list, 
                                                                              self.env.solution,
                                                                              current_step,
                                                                              point_couples=point_couples, 
                                                                              endpoint_mask=endpoint_mask)
                        
                        selected_student[connect_to_another_endpoint] = \
                                another_endpoint.gather(index=last_selected.unsqueeze(1), dim=1).squeeze(1)[connect_to_another_endpoint]
                        selected_student[last_is_padding] = current_step
                        selected_teacher = selected_student
                        last_selected    = selected_student
                        last_is_second_endpoint = connect_to_another_endpoint

                    current_step += 1

                    selected_teacher_all  = torch.cat((selected_teacher_all, selected_teacher[:,  None]), dim=1)
                    selected_student_all = torch.cat((selected_student_all, selected_student[:, None]), dim=1)
                    
                    state, reward, reward_student, done = self.env.step(selected_teacher, selected_student) 
                
                           
                reduced_solution_indexed_by_sorted_problem = new_problem_index_on_sorted_problem.gather(dim=1, index=self.env.selected_node_list)
                if destroy_mode != 'knn-location':
                    best_solution = torch.roll(best_solution, shifts=shift, dims=1)
                else: 
                    problem_size = best_solution.size(1)
                    shift_ist_by_ist = shift.unsqueeze(-1)
                    shifted_index = torch.arange(problem_size).unsqueeze(0).repeat((batch_size,1)) + shift_ist_by_ist
                    shifted_index[shifted_index >= problem_size] = shifted_index[shifted_index >= problem_size] - problem_size
                    best_solution = best_solution.gather(index=shifted_index, dim=1)

                _, complete_solution_on_sorted_problem_batch = assemble_solution_for_sorted_problem_batch(destruction_mask, 
                                                                                    endpoint_mask, 
                                                                                    self.env.selected_node_list, 
                                                                                    new_problem_index_on_sorted_problem, 
                                                                                    padding_mask)
                
                current_solution = best_solution.gather(1, index=complete_solution_on_sorted_problem_batch)
                
                current_length = self.env._get_travel_distance_2(self.origin_problem, current_solution, test_in_tsplib=True)
                is_better = current_length < current_best_length - 1e-6

                self.logger.info("improved: {}".format(torch.sum(is_better).item()))

                best_solution[is_better,:] = current_solution[is_better,:]
                current_best_length[is_better] = current_length[is_better]

                # Reset env
                self.env.problems = self.origin_problem
                self.env.problem_size = self.origin_problem.size(1)
                self.env.solution = best_solution

                escape_time,_ = clock.get_est_string(1, 1)

                # self.logger.info("repair step{},  gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                #    bbbb, ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100, escape_time,
                #     current_best_length.mean().item(), self.optimal_length.mean().item()))
                # self.logger.info("===============================================================")

            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_solution, test_in_tsplib=True)

            self.logger.info("-------------------------------------------------------------------------------")
            gap = ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100
            gap_log.append(tsplib_name + '\n')
            gap_log.append("repair step {},  gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f} \n".format(
                                                                bbbb,
                                                                gap,
                                                                escape_time,
                                                                current_best_length.mean().item(),
                                                                self.optimal_length.mean().item()))
            
            self.recordings['tsp_name'].append(tsplib_name)
            self.recordings['solution'].append(best_solution.squeeze().unsqueeze(0))


            ####################################### END repair #########################################


            return self.optimal_length.mean().item(), current_best_length.mean().item(), current_best_length.mean().item(), self.env.problem_size, gap, gap_log
