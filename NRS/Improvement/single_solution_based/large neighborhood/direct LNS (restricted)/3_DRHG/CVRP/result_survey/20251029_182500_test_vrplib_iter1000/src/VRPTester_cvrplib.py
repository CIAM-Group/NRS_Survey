
import torch

import os
from logging import getLogger

from CVRP.VRPEnv import VRPEnv as Env
from CVRP.VRPModel import VRPModel as Model
from utils.utils import *
from CVRP.utils_for_vrp_tester import assemble_vrp_solution_for_sorted_problem_batch

from CVRP.sweep import Sweep
from CVRP.trans_flag import tran_to_node_flag

import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(1234)


import os

def CVRPLIBReader(filename):
    with open(filename, 'r') as f:
        dimension = 0
        started_node = False
        started_demand = False
        locs, demand = [], []
        for line in f:
            loc = []
            if started_demand:
                if line.startswith("DEPOT_SECTION"):
                    break
                demand.append(int(line.strip().split()[-1]))
            if started_node:
                if line.startswith("DEMAND_SECTION"):
                    started_node = False
                    started_demand = True
            if started_node:
                loc.append(float(line.strip().split()[1]))
                loc.append(float(line.strip().split()[2]))
                locs.append(loc)

            if line.startswith("NAME"):
                name = line.strip().split()[-1]
            if line.startswith("DIMENSION"):
                dimension = int(float(line.strip().split()[-1]) - 1)  # depot 不计入 dimension
            if line.startswith("EDGE_WEIGHT_TYPE"):
                if line.strip().split()[-1] not in ["EUC_2D", "CEIL_2D"]:
                    return None, None, None, None, None, None
            if line.startswith("CAPACITY"):
                capacity = int(float(line.strip().split()[-1]))
            if line.startswith("NODE_COORD_SECTION"):
                started_node = True

    cost_file = filename.replace('.vrp', '.sol')
    cost = None
    if os.path.exists(cost_file):
        with open(cost_file, 'r') as f:
            for line in f:
                if line.startswith("Cost"):
                    cost = float(line.split()[1])
    assert len(locs) == dimension + 1  # +1 for depot
    return name, int(dimension), locs, demand, capacity, cost



class VRPTester():
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()

        # problem and initial solution load
        self.test_in_lib = self.env_params['test_in_vrplib']
        self.pt_path = self.env_params['vrplib_path']
        # self.initial_solution_path = tester_params['initial_solution_path']
        self.initial_solution_path = None

        # set shortcuts
        self.iter_budget = tester_params['iter_budget']
        self.destroy_mode    = tester_params['destroy_mode']
        self.destroy_params  = tester_params['destroy_params']
        self.knn_k_high = int(self.destroy_params[self.destroy_mode[0]]['knn_k'][1])
        
        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device
        self.env_params['device'] = device

        if self.env_params['use_model'] == 'DRHG':
            self.model = Model(**self.model_params)
            self.env = Env(**self.env_params)
        else:
            raise NotImplementedError("{} not implemented".format(self.env_params['use_model']))

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()

        if self.env_params['load_way']=='txt':
            self.env.load_raw_data(self.tester_params['test_episodes'] )
        elif self.env_params['load_way']=='vrplib':
            self.env.load_raw_data(self.tester_params['test_episodes'], from_pt=True, pt_path=self.pt_path, cvrplib=self.test_in_lib)
        else:
            raise NotImplementedError("load_way {} not implemented".format(self.env_params['load_way']))

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            # score, score_student_mean, problems_size, gap_log = self._test_one_batch(episode, batch_size, clock=self.time_estimator)
            score, score_student_mean, = self._test_one_batch(episode, batch_size, clock=self.time_estimator)
            # gap_log_all.extend(gap_log)
            score_AM.update(score, batch_size)
            score_student_AM.update(score_student_mean, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f},Score_studetnt: {:.4f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score,score_student_mean))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" NO-AUG SCORE student: {:.4f} ".format(score_student_AM.avg))
                self.logger.info(" Gap: {:.4f}%".format((score_student_AM.avg-score_AM.avg) / score_AM.avg * 100))
                gap_ = (score_student_AM.avg-score_AM.avg) / score_AM.avg * 100


        return score_AM.avg, score_student_AM.avg



    def _test_one_batch(self, episode, batch_size, clock=None):

        # Ready
        ###############################################
        self.model.eval()
        self.model.mode = 'test'

        with torch.no_grad():

            # load problem and solution
            self.env.load_problems(episode, batch_size, load_cvrplib=True)
            self.origin_problem = self.env.problems
            self.optimal_length = self.env._get_travel_distance_2(self.origin_problem, self.env.solution)
            self.optimal_solution = self.env.solution.clone()
            # initial_solution = torch.load(self.initial_solution_path, map_location=self.device)[episode] 
            ist_name = self.env.raw_instance_name[episode]
            problem_size = self.optimal_solution.size(-2)


            # ! ——在线生成 Sweep 初始解（node-flag 形式）——
            # problems: [B, V+1, 4]，[:,:,:2] 坐标，[:,:,2] 需求，[:,0,3] 容量
            problems = self.origin_problem.to(self.device)

            # 1) 用 Sweep 得到包含 depot=0 的访问序列（partial_solution 的第一列即为 0）
            sweep_solver = Sweep(problems)
            sweep_solver.set_up()
            done = False
            while not done:
                done, partial_solution = sweep_solver.update()  # partial_solution: [B, t]，首元素是 0

            # 2) 收尾补 0（回仓），再转成 (node, flag) 形式，shape = [B, V, 2]
            solution_seq = torch.cat(
                [partial_solution, torch.zeros((partial_solution.size(0), 1), dtype=torch.int64, device=self.device)],
                dim=1
            )  # [B, t+1]，以 0 结尾
            initial_solution = tran_to_node_flag(solution_seq).to(self.device)  # [B, V, 2] (long)

            # （可选）合法性检查，方便排错
            self.env.valida_solution_legal(self.origin_problem, initial_solution, capacity_=self.env.raw_data_capacity[episode])

            # get ready
            reset_state, _, _ = self.env.reset(self.env_params['mode'])
            best_solution = initial_solution.clone()
            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_solution)
            escape_time, _ = clock.get_est_string(1, 1)

            self.logger.info("initial solution, gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100, escape_time,
                current_best_length.mean().item(), self.optimal_length.mean().item()))

            ################################ destroy and repair ###########################################
            current_solution = torch.zeros(best_solution.size(), dtype=int)
            if self.tester_params['rearrange_solution']:
                best_solution = self.env.Rearrange_solution_clockwise(self.env.problems, best_solution)
                self.logger.info('rearrange solution clockwise')

            # reset env
            self.env.problems = self.origin_problem
            self.env.problem_size = self.origin_problem.size(1)
            self.env.solution = best_solution

            destroy_mode = self.destroy_mode[0]
            destroy_params = self.destroy_params[destroy_mode]
            destroy_params['knn_k'][1] = min(self.knn_k_high, problem_size) # for small-size instances


            # pre step of destruction
            iter_budget = self.iter_budget
            num_interval = torch.sqrt(torch.tensor(iter_budget)).long()
            center_x = (torch.arange(num_interval) + 0.5)/ num_interval
            center_y = (torch.arange(num_interval) + 0.5) / num_interval

            for bbbb in range(iter_budget):
                if destroy_params['center_type'] == "equally":
                    destroy_params['center_location'] = (random.choice(center_x), random.choice(center_y))
                elif destroy_params['center_type'] == "random":
                    destroy_params['center_location'] = (random.uniform(0, 1), random.uniform(0, 1))
                else: 
                    raise NotImplementedError("center_type {} not implemented".format(destroy_params['center_type']))
                
                # destroy
                destruction_mask, reduced_problem_coords, reduced_problem_demand, reduced_problem_capacity, endpoint_mask, another_endpoint, point_couples, padding_mask, new_problem_index_on_sorted_problem, \
                    problem_coords_sorted, demand_sorted= \
                            self.env.sampling_reduced_problems(destroy_mode, destroy_params, True)
                

                self.env.problems[:, :, :2] = self.env.coordinate_transform(self.env.problems[:, :, :2])
                self.logger.info('coordinate_transform imposed')

                state, reward, done = self.env.reset(mode='test')  # state: data, first_node = current_node
                state, reward, reward_student, done = self.env.pre_step()

                current_step = 0
                # repair
                while not done:
                    # print(self.env.raw_data_capacity)
                    loss_node, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = \
                            self.model(state, 
                                       self.env.selected_node_list, 
                                       self.env.solution,
                                       current_step,
                                       raw_data_capacity=self.env.raw_data_capacity, 
                                       point_couples=point_couples, 
                                       endpoint_mask=endpoint_mask)  

                    this_is_padding = padding_mask[:, current_step]
                    if current_step == 0:
                        selected_flag_teacher = torch.ones(batch_size, dtype=torch.int)
                        selected_flag_student = selected_flag_teacher
                        selected_student[this_is_padding] = current_step + 1
                        selected_teacher = selected_student # in testing, no teacher; only for env.step()
                        last_selected = selected_student
                        last_is_second_endpoint = torch.zeros((batch_size), dtype=bool)
                        connect_to_another_endpoint = None
                        connect_to_another_endpoint = torch.zeros((batch_size), dtype=bool)
                        last_is_endpoint = torch.gather(endpoint_mask, dim=1, index=last_selected.unsqueeze(1)).squeeze()

                    else:
                        last_is_padding = padding_mask[:, current_step-1]
                        last_is_endpoint = torch.gather(endpoint_mask, dim=1, index=last_selected.unsqueeze(1)).squeeze()
                        connect_to_another_endpoint = last_is_endpoint & (~last_is_second_endpoint) # segment endpoint A should connected to endpoint B

                        selected_student[connect_to_another_endpoint] = \
                                another_endpoint.gather(index=last_selected.unsqueeze(1), dim=1).squeeze(1)[connect_to_another_endpoint]
                        selected_flag_student[connect_to_another_endpoint] = 0
                        
                        selected_student[this_is_padding] = current_step + 1
                        selected_flag_student[last_is_padding] = 1

                        selected_teacher = selected_student # in testing, no teacher; only for env.step()
                        selected_flag_teacher = selected_flag_student

                        last_selected    = selected_student
                        last_is_second_endpoint = connect_to_another_endpoint
                    
                    current_step += 1
                    state, reward, reward_student, done = \
                        self.env.step(selected_teacher, selected_student, selected_flag_teacher, selected_flag_student, \
                                      is_another_endpoint=connect_to_another_endpoint, is_first_endpoint=None, raw_capacity=self.env.raw_data_capacity[episode])

                reduced_solution_after_repair = torch.cat((self.env.selected_student_list.unsqueeze(2),
                                                           self.env.selected_student_flag.unsqueeze(2)), dim=2)
                
                # restore solution of hyper-graph on original problem
                destruction_mask_after_repair, complete_solution_on_sorted_problem, complete_flag_on_sorted_problem = assemble_vrp_solution_for_sorted_problem_batch(destruction_mask, 
                                                                                        endpoint_mask, 
                                                                                        reduced_solution_after_repair,
                                                                                        new_problem_index_on_sorted_problem,
                                                                                        padding_mask)

                current_solution[:,:,0] = best_solution[:,:,0].gather(1, index=complete_solution_on_sorted_problem - 1) # -1: solution start from 1
                current_solution[:,:,1] = complete_flag_on_sorted_problem
                
                # update if improve
                current_length = self.env._get_travel_distance_2(self.origin_problem, current_solution)
                is_better = current_length < current_best_length - 1e-7
                best_solution[is_better,:] = current_solution[is_better,:]
                current_best_length[is_better] = current_length[is_better]

                # reset Env
                if self.tester_params['rearrange_solution']:
                    best_solution = self.env.Rearrange_solution_clockwise(self.origin_problem, best_solution)
                    self.logger.info('rearrange solution clockwise')

                self.env.problems = self.origin_problem
                self.env.problem_size = self.origin_problem.size(1)
                self.env.solution = best_solution

                # logging
                escape_time,_ = clock.get_est_string(1, 1)
                self.logger.info("step{},  gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                   bbbb, ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100, escape_time,
                    current_best_length.mean().item(), self.optimal_length.mean().item()))
                                
                self.env.valida_solution_legal(self.origin_problem, best_solution, capacity_=self.env.raw_data_capacity[episode])

            # all budgets are run out
            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_solution)
            self.logger.info("-------------------------------------------------------------------------------")

            return self.optimal_length.mean().item(), current_best_length.mean().item() 
        
    # ! 新增
    def run_lib(self, root_dir, detailed_log=True):
        """
        读取 root_dir 下的 .vrp/.sol，逐实例跑 DRHG，并按规模分桶统计与日志输出。
        """
        self.time_estimator.reset()

        # 分桶容器
        buckets = {
            "[0,1000)":         {"no_aug_gap": [], "times": [], "count": 0},
            "[1000,10000)":     {"no_aug_gap": [], "times": [], "count": 0},
            "[10000,100001]":   {"no_aug_gap": [], "times": [], "count": 0},
            "ALL":              {"no_aug_gap": [], "times": [], "count": 0},
        }
        result_detail = {"instances": [], "optimal": [], "problem_size": [], "score": [], "gap": []}
        all_start = time.time()
        solved = 0
        total  = 0

        for root, _, files in os.walk(root_dir):
            for fn in files:
                if not fn.endswith(".vrp"):
                    continue
                total += 1
                path = os.path.join(root, fn)
                try:
                    name, dimension, locs, demand, capacity, optimal = CVRPLIBReader(path)
                    if name is None or optimal is None:
                        self.logger.info(f"Skip {fn} (unsupported EDGE_WEIGHT_TYPE or missing .sol)")
                        continue

                    # 构造 origin_problem（[1, V+1, 4]：xy, demand, capacity）
                    node_xy = torch.tensor(locs, dtype=torch.float32, device=self.device).unsqueeze(0)        # (1, V+1, 2)
                    demand_ = torch.tensor(demand, dtype=torch.float32, device=self.device).unsqueeze(0)      # (1, V+1)
                    cap_    = torch.full((1, node_xy.size(1), 1), float(capacity), device=self.device)        # (1, V+1, 1)
                    origin_problem = torch.cat([node_xy, demand_.unsqueeze(-1), cap_], dim=2)                 # (1, V+1, 4)

                    # ---- 求解前计时 ----
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()  # 确保 GPU 空闲
                    inst_start = time.time()


                    # ==== DRHG 流程（基本照抄 _test_one_batch，但不依赖 env.solution 作为最优）====
                    self.model.eval()
                    with torch.no_grad():
                        # 设置 env 的“当前实例”
                        self.env.problems = origin_problem.clone()
                        self.origin_problem = origin_problem.clone()
                        self.env.problem_size = origin_problem.size(1) - 1

                        # 生成 Sweep 初始解（node-flag）
                        problems = self.origin_problem.to(self.device)
                        sweep_solver = Sweep(problems)
                        sweep_solver.set_up()
                        done = False
                        while not done:
                            done, partial_solution = sweep_solver.update()
                        solution_seq = torch.cat(
                            [partial_solution, torch.zeros((partial_solution.size(0), 1), dtype=torch.int64, device=self.device)],
                            dim=1
                        )
                        initial_solution = tran_to_node_flag(solution_seq).to(self.device)  # [1, V, 2] (long)
                        self.env.valida_solution_legal(self.origin_problem, initial_solution, capacity_=capacity)

                        self.env.batch_size = int(self.origin_problem.size(0))      # 这里通常就是 1
                        self.env.problem_size = self.origin_problem.size(1) - 1     # 已有也行，确保一致

                        # get ready
                        reset_state, _, _ = self.env.reset(self.env_params['mode'])
                        best_solution = initial_solution.clone()
                        current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_solution)

                        # 破坏-修复迭代
                        destroy_mode = self.destroy_mode[0]
                        destroy_params = self.destroy_params[destroy_mode]
                        problem_size = best_solution.size(1)
                        destroy_params['knn_k'][1] = min(self.knn_k_high, problem_size)

                        if self.tester_params.get('rearrange_solution', True):
                            best_solution = self.env.Rearrange_solution_clockwise(self.env.problems, best_solution)

                        self.env.problems = self.origin_problem
                        self.env.problem_size = self.origin_problem.size(1)
                        self.env.solution = best_solution  # 仅用于 env 内部接口（teacher无用）

                        iter_budget = self.iter_budget
                        num_interval = torch.sqrt(torch.tensor(iter_budget)).long()
                        center_x = (torch.arange(num_interval) + 0.5)/ num_interval
                        center_y = (torch.arange(num_interval) + 0.5)/ num_interval

                        for step_id in range(iter_budget):
                            if destroy_params['center_type'] == "equally":
                                destroy_params['center_location'] = (random.choice(center_x), random.choice(center_y))
                            elif destroy_params['center_type'] == "random":
                                destroy_params['center_location'] = (random.uniform(0, 1), random.uniform(0, 1))
                            else:
                                raise NotImplementedError

                            # destroy（返回 reduced & mapping）
                            (destruction_mask, reduced_problem_coords, reduced_problem_demand, reduced_problem_capacity,
                            endpoint_mask, another_endpoint, point_couples, padding_mask,
                            new_problem_index_on_sorted_problem, problem_coords_sorted, demand_sorted) = \
                                self.env.sampling_reduced_problems(destroy_mode, destroy_params, True)

                            # 坐标标准化（只作用于 env 的工作副本）
                            self.env.problems[:, :, :2] = self.env.coordinate_transform(self.env.problems[:, :, :2])

                            # repair
                            state, reward, done = self.env.reset(mode='test')
                            state, reward, reward_student, done = self.env.pre_step()
                            current_step = 0
                            while not done:
                                (_, selected_teacher, selected_student,
                                selected_flag_teacher, selected_flag_student) = \
                                    self.model(state,
                                            self.env.selected_node_list,
                                            self.env.solution,
                                            current_step,
                                            raw_data_capacity=torch.tensor([capacity], dtype=torch.float32, device=self.device),
                                            point_couples=point_couples,
                                            endpoint_mask=endpoint_mask)

                                this_is_padding = padding_mask[:, current_step]
                                if current_step == 0:
                                    selected_flag_teacher = torch.ones(1, dtype=torch.int, device=self.device)
                                    selected_flag_student = selected_flag_teacher
                                    selected_student[this_is_padding] = current_step + 1
                                    selected_teacher = selected_student
                                    last_selected = selected_student
                                    last_is_second_endpoint = torch.zeros((1), dtype=bool, device=self.device)
                                    connect_to_another_endpoint = torch.zeros((1), dtype=bool, device=self.device)
                                else:
                                    last_is_padding = padding_mask[:, current_step-1]
                                    last_is_endpoint = torch.gather(endpoint_mask, dim=1, index=last_selected.unsqueeze(1)).squeeze()
                                    connect_to_another_endpoint = last_is_endpoint & (~last_is_second_endpoint)

                                    selected_student[connect_to_another_endpoint] = \
                                        another_endpoint.gather(index=last_selected.unsqueeze(1), dim=1).squeeze(1)[connect_to_another_endpoint]
                                    selected_flag_student[connect_to_another_endpoint] = 0

                                    selected_student[this_is_padding] = current_step + 1
                                    selected_flag_student[last_is_padding] = 1

                                    selected_teacher = selected_student
                                    selected_flag_teacher = selected_flag_student
                                    last_selected = selected_student
                                    last_is_second_endpoint = connect_to_another_endpoint

                                current_step += 1
                                state, reward, reward_student, done = self.env.step(
                                    selected_teacher, selected_student,
                                    selected_flag_teacher, selected_flag_student,
                                    is_another_endpoint=connect_to_another_endpoint,
                                    is_first_endpoint=None, raw_capacity=capacity
                                )

                            reduced_solution_after_repair = torch.cat(
                                (self.env.selected_student_list.unsqueeze(2),
                                self.env.selected_student_flag.unsqueeze(2)), dim=2)

                            # restore
                            (destruction_mask_after_repair,
                            complete_solution_on_sorted_problem,
                            complete_flag_on_sorted_problem) = assemble_vrp_solution_for_sorted_problem_batch(
                                destruction_mask, endpoint_mask, reduced_solution_after_repair,
                                new_problem_index_on_sorted_problem, padding_mask
                            )

                            current_solution = torch.zeros_like(best_solution)
                            current_solution[:,:,0] = best_solution[:,:,0].gather(1, index=complete_solution_on_sorted_problem - 1)
                            current_solution[:,:,1] = complete_flag_on_sorted_problem

                            # 更新最优
                            current_length = self.env._get_travel_distance_2(self.origin_problem, current_solution)
                            is_better = current_length < current_best_length - 1e-7
                            best_solution[is_better,:] = current_solution[is_better,:]
                            current_best_length[is_better] = current_length[is_better]

                            if self.tester_params.get('rearrange_solution', True):
                                best_solution = self.env.Rearrange_solution_clockwise(self.origin_problem, best_solution)

                            self.env.problems = self.origin_problem
                            self.env.problem_size = self.origin_problem.size(1)
                            self.env.solution = best_solution

                        # 单实例完成：记录成绩
                        student_score = self.env._get_travel_distance_2(self.origin_problem, best_solution).mean().item()
                        # 与 .sol 的 optimal 对比（注意 optimal 已是整数口径）
                        gap = (student_score - optimal) * 100.0 / optimal
                    
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    inst_time = time.time() - inst_start

                    # 分桶
                    if dimension < 1000:
                        bucket_key = "[0,1000)"
                    elif dimension < 10000:
                        bucket_key = "[1000,10000)"
                    else:
                        bucket_key = "[10000,100001]"

                    buckets[bucket_key]["no_aug_gap"].append(gap)
                    buckets[bucket_key]["times"].append(inst_time)      # ✅ 记录时间
                    buckets[bucket_key]["count"] += 1

                    buckets["ALL"]["no_aug_gap"].append(gap)
                    buckets["ALL"]["times"].append(inst_time)           # ✅ 记录时间
                    buckets["ALL"]["count"] += 1
                    solved += 1

                    # detailed
                    result_detail["instances"].append(name)
                    result_detail["optimal"].append(optimal)
                    result_detail["problem_size"].append(dimension)
                    result_detail["score"].append(student_score)
                    result_detail["gap"].append(gap)
                    # 也可以把单实例时间存一下（可选）
                    # result_detail.setdefault("time", []).append(inst_time)

                    self.logger.info(
                        f"Instance {name} (n={dimension}) | optimal={optimal:.0f} | score={student_score:.0f} | "
                        f"gap={gap:.3f}% | time={inst_time:.3f}s"       # ✅ 打印单实例时间
                    )
                    self.logger.info("===============================================================")


                except Exception as e:
                    self.logger.info(f"[Skip] {fn}: {e}")
                    continue

        all_dur = time.time() - all_start
        
        
        # 汇总日志
        def _avg(xs): 
            return float(np.mean(xs)) if len(xs) > 0 else 0.0

        self.logger.info("#################  CVRPLIB (file) TEST DONE  #################")
        self.logger.info(
            f"Solved {solved}/{total}, total time: {all_dur:.2f}s, "
            f"avg time (solved only): {all_dur / max(solved,1):.2f}s"     # ✅ 全局平均时间（按 solved）
        )

        for k in ["[0,1000)","[1000,10000)","[10000,100001]"]:
            self.logger.info(
                f"{k} | num={buckets[k]['count']}, "
                f"avg gap={_avg(buckets[k]['no_aug_gap']):.3f}%, "
                f"avg time={_avg(buckets[k]['times']):.3f}s"               # ✅ 每桶平均时间
            )

        self.logger.info(
            f"ALL | num={buckets['ALL']['count']}, "
            f"avg gap={_avg(buckets['ALL']['no_aug_gap']):.3f}%, "
            f"avg time={_avg(buckets['ALL']['times']):.3f}s"   # ✅ 修复了格式说明符
        )


