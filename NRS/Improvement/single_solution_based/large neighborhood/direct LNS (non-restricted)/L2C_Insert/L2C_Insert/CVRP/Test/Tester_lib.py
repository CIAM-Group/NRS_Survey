from logging import getLogger

from L2C_Insert.CVRP.Test.VRPModel_lib import VRPModel as Model
from L2C_Insert.CVRP.Test.VRPEnv import VRPEnv as Env
from L2C_Insert.CVRP.Test.utils2 import *
from L2C_Insert.CVRP.utils.utils import *

import time
import numpy as np

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        set_seeds(seed=123)

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

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()
        self.time_estimator_2 = TimeEstimator()

    def run(self):
        self.time_estimator.reset()
        self.time_estimator_2.reset()

        self.env.load_raw_data(self.tester_params['test_episodes'])

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = self.tester_params['begin_index']
        problems_100 = []
        problems_100_200 = []
        problems_200_500 = []
        problems_500_1000 = []
        problems_1000 = []

        problems_A = []
        problems_B = []
        problems_E = []
        problems_F = []
        problems_M = []
        problems_P = []
        problems_X = []

        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score_teacher, score_student, problems_size, vrpname = self._test_one_batch(
                episode, batch_size, clock=self.time_estimator_2, logger=self.logger)
            current_gap = (score_student - score_teacher) / score_teacher
            if problems_size < 100:
                problems_100.append(current_gap)
            elif 100 <= problems_size < 200:
                problems_100_200.append(current_gap)
            elif 200 <= problems_size < 500:
                problems_200_500.append(current_gap)
            elif 500 <= problems_size < 1000:
                problems_500_1000.append(current_gap)
            elif 1000 <= problems_size:
                problems_1000.append(current_gap)

            if vrpname[:2] == 'A-':
                problems_A.append(current_gap)
            elif vrpname[:2] == 'B-':
                problems_B.append(current_gap)
            elif vrpname[:2] == 'E-':
                problems_E.append(current_gap)
            elif vrpname[:2] == 'F-':
                problems_F.append(current_gap)
            elif vrpname[:2] == 'M-':
                problems_M.append(current_gap)
            elif vrpname[:2] == 'P-':
                problems_P.append(current_gap)
            elif vrpname[:2] == 'X-':
                problems_X.append(current_gap)

            print('problems_100 mean gap:', np.mean(problems_100), len(problems_100))
            print('problems_100_200 mean gap:', np.mean(problems_100_200), len(problems_100_200))
            print('problems_200_500 mean gap:', np.mean(problems_200_500), len(problems_200_500))
            print('problems_500_1000 mean gap:', np.mean(problems_500_1000), len(problems_500_1000))
            print('problems_1000 mean gap:', np.mean(problems_1000), len(problems_1000))

            self.logger.info(
                " problems_A    mean gap:{:4f}%, num:{}".format(np.mean(problems_A) * 100, len(problems_A)))
            self.logger.info(
                " problems_B    mean gap:{:4f}%, num:{}".format(np.mean(problems_B) * 100, len(problems_B)))
            self.logger.info(
                " problems_E    mean gap:{:4f}%, num:{}".format(np.mean(problems_E) * 100, len(problems_E)))
            self.logger.info(
                " problems_F    mean gap:{:4f}%, num:{}".format(np.mean(problems_F) * 100, len(problems_F)))
            self.logger.info(
                " problems_M    mean gap:{:4f}%, num:{}".format(np.mean(problems_M) * 100, len(problems_M)))
            self.logger.info(
                " problems_P    mean gap:{:4f}%, num:{}".format(np.mean(problems_P) * 100, len(problems_P)))
            self.logger.info(
                " problems_X    mean gap:{:4f}%, num:{}".format(np.mean(problems_X) * 100, len(problems_X)))

            score_AM.update(score_teacher, batch_size)
            score_student_AM.update(score_student, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info(
                "episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], Score_teacher:{:.4f}, Score_studetnt: {:.4f}".format(
                    episode, test_num_episode, elapsed_time_str, remain_time_str, score_teacher, score_student))

            all_done = (episode == test_num_episode)

            if all_done:
                if self.env_params['test_in_vrplib']:
                    self.logger.info(" *** Test Done *** ")
                    all_result_gaps = problems_A + problems_B + problems_E + problems_F + problems_M + problems_P + problems_X
                    gap_ = np.mean(all_result_gaps) * 100
                    self.logger.info(" Gap: {:.4f}%".format(gap_))
                else:
                    self.logger.info(" *** Test Done *** ")
                    self.logger.info(" Teacher SCORE: {:.4f} ".format(score_AM.avg))
                    self.logger.info(" Student SCORE: {:.4f} ".format(score_student_AM.avg))
                    gap_ = (score_student_AM.avg - score_AM.avg) / score_AM.avg * 100
                    self.logger.info(" Gap: {:.4f}%".format(gap_))

        return score_AM.avg, score_student_AM.avg, gap_

    def run_lib(self):
        self.time_estimator.reset()
        self.time_estimator_2.reset()

        # 读取数据（已按你前面改好的 VRPEnv 支持目录排序等）
        self.env.load_raw_data(self.tester_params['test_episodes'])

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = self.tester_params['begin_index']

        # ===== 新：按 ICAM 的规模桶统计 =====
        # 桶： [0,1000), [1000,10000), [10000,100000]
        bucket_ranges = [(0, 1000), (1000, 10000), (10000, 100001)]
        # 每桶：gap 列表、时间列表
        bucket_stats = {
            (0, 1000):      {"gaps": [], "times": []},
            (1000, 10000):  {"gaps": [], "times": []},
            (10000, 100001): {"gaps": [], "times": []},
        }

        # 汇总
        all_gaps = []
        all_times = []
        all_instance_num = 0
        all_solved_instance_num = 0

        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            # ===== 新：对每个 instance 做 try/except + 计时 =====
            # 这里沿用旧接口：一批只测1个（原本就这么用），保持兼容
            inst_start = None
            try:
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                inst_start = time.time()

                score_teacher, score_student, problems_size, vrpname = self._test_one_batch(
                    episode, batch_size, clock=self.time_estimator_2, logger=self.logger)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                inst_time = time.time() - inst_start

                # gap（百分比）
                current_gap = (score_student - score_teacher) * 100.0 / score_teacher

                # ===== 新：按桶统计 =====
                placed = False
                for br in bucket_ranges:
                    lo, hi = br
                    if lo <= problems_size < hi:
                        bucket_stats[br]["gaps"].append(current_gap)
                        bucket_stats[br]["times"].append(inst_time)
                        placed = True
                        break
                if not placed:
                    # 正常不会到这里；若有异常维度，归入最后一桶
                    bucket_stats[bucket_ranges[-1]]["gaps"].append(current_gap)
                    bucket_stats[bucket_ranges[-1]]["times"].append(inst_time)

                # 全部汇总
                all_gaps.append(current_gap)
                all_times.append(inst_time)
                all_instance_num += 1
                all_solved_instance_num += 1

                # 打印当前桶的即时均值（可选）
                self.logger.info(
                    f"[Inst] name:{vrpname}, size:{problems_size}, time:{inst_time:.3f}s, gap:{current_gap:.3f}%, score_teacher:{score_teacher:.4f}, score_student:{score_student:.4f}"
                )

                # 旧版的前缀分组（A-/B-/…）如果还想保留，可以在此继续累加；本文聚焦规模桶，故省略

                score_AM.update(score_teacher, batch_size)
                score_student_AM.update(score_student, batch_size)

            except Exception as e:
                # 出错不中断，记为未解决
                self.logger.info(
                    f"[Inst-ERROR] episode:{episode}, msg:{e}. Skip this instance."
                )
                all_instance_num += 1
                # 不增加 solved 计数
                # 不更新均值/桶
                inst_time = 0.0  # 没跑完就不计时
                # 继续下一条

            # 推进
            episode += batch_size

            # ===== 进度日志 =====
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            # self.logger.info(
            #     "episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], Score_teacher:{:.4f}, Score_studetnt: {:.4f}".format(
            #         episode, test_num_episode, elapsed_time_str, remain_time_str,
            #         score_AM.avg if score_AM.count>0 else 0.0,
            #         score_student_AM.avg if score_student_AM.count>0 else 0.0
            #     )
            # )
            self.logger.info("episode {:3d}/{:3d}".format(episode, test_num_episode))

        # ===== 全部结束后：打印桶统计与整体统计 =====
        self.logger.info(" *** Test Done *** ")

        for br in bucket_ranges:
            lo, hi = br
            gaps = bucket_stats[br]["gaps"]
            times = bucket_stats[br]["times"]
            cnt = len(gaps)
            avg_gap = float(np.mean(gaps)) if cnt > 0 else 0.0
            avg_time = float(np.mean(times)) if cnt > 0 else 0.0
            self.logger.info(f"[{lo}, {hi})  num:{cnt}, avg gap:{avg_gap:.3f}%, avg time:{avg_time:.3f}s")

        # 总体
        overall_cnt = len(all_gaps)
        overall_avg_gap = float(np.mean(all_gaps)) if overall_cnt > 0 else 0.0
        overall_avg_time = float(np.mean(all_times)) if overall_cnt > 0 else 0.0
        self.logger.info(
            f"All instances: solved {all_solved_instance_num}/{all_instance_num}, "
            f"avg gap:{overall_avg_gap:.3f}%, avg time:{overall_avg_time:.3f}s"
        )

        # 兼容原返回（若你上层还有用）
        if score_AM.count > 0 and score_student_AM.count > 0:
            gap_ = (score_student_AM.avg - score_AM.avg) / score_AM.avg * 100
            return score_AM.avg, score_student_AM.avg, gap_
        else:
            return 0.0, 0.0, 0.0


    def decide_whether_to_repair_solution(self, before_repair_solution, before_reward, after_repair_solution, after_reward):

        if_repair = before_reward > after_reward

        before_repair_solution[if_repair] = after_repair_solution[if_repair]

        return before_repair_solution

    def _test_one_batch(self, episode, batch_size, clock=None, logger=None):

        random_seed = 1234
        torch.manual_seed(random_seed)

        ###############################################
        self.model.eval()

        with torch.no_grad():



            self.env.load_problems(episode, batch_size)

            self.origin_problem_size = self.env.problems.shape[1]-1

            reset_state= self.env.reset(self.env_params['mode'])

            current_step = 0

            state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node

            self.origin_problem = self.env.problems.clone().detach()

            if self.env.test_in_vrplib:
                self.origin_problem = self.env.problems.clone().detach()
                self.origin_solution = None
                self.optimal_length, name = self.env._get_travel_distance_2(self.origin_problem, self.env.solution,
                                                                            need_optimal=True)

            else:
                self.origin_problem = self.env.problems.clone().detach()
                self.origin_solution = self.env.solution.clone().detach()
                self.optimal_length = self.env._get_travel_distance_2(self.origin_problem, self.env.solution)
                name = 'vrp' + str(self.env.solution.shape[1])
            B_V = batch_size * 1

            # print(self.optimal_length)

            self.index_gobal = torch.arange(batch_size, dtype=torch.long)[:, None]

            raw_capacity = self.env.raw_data_capacity[0]

            if self.env_params['random_insertion']:
                from utils_insertion.insertion import cvrp_random_insertion

                dataset = self.origin_problem.clone().cpu().numpy()
                problem_size = dataset.shape[1]
                width = 1
                print('random insertion begin!')
                initial_solution = []
                for kk in range(self.origin_problem.shape[0]):
                    pos = self.origin_problem[kk, 1:, :2].clone().cpu().numpy()
                    depotpos = self.origin_problem[kk, 0, :2].clone().cpu().numpy()
                    demands = self.origin_problem[kk, 1:, 2].clone().cpu().numpy()
                    capacity = self.origin_problem[kk, 0, 3].clone().cpu().numpy()
                    capacity = int(capacity)
                    # print(capacity)
                    # print(pos.shape)
                    # print(demands.shape, demands[0])
                    # print(depotpos.shape)

                    # assert False
                    route = cvrp_random_insertion(pos, depotpos, demands, capacity)
                    solution = []
                    for i in range(len(route)):
                        sub_tour = (route[i] + 1).tolist()
                        solution += [0]
                        solution += sub_tour
                        solution += [0]
                    # solution = np.array(solution)
                    solution = torch.tensor(solution).reshape(1, -1)
                    solution = tran_to_node_flag(solution)
                    if initial_solution == []:
                        initial_solution = solution
                    else:
                        initial_solution = torch.cat((initial_solution, solution), dim=0)

                best_select_node_list = initial_solution
            else:

                # from tqdm import tqdm
                # with tqdm(total=self.env.problem_size) as pbar:
                while current_step<self.origin_problem_size :
                    # pbar.update(1)
                # print(' \n 0 ------------------------ current_step ',current_step)

                    if current_step == 0:
                        # if current_step%100==0:
                        #     print(current_step)
                        # 第一步先选出与depot离得最近的scatter，然后这两个点当作partial solution。

                        abs_scatter_solu_1 = torch.arange(start=1, end=self.origin_problem_size + 1,
                                                            dtype=torch.int64).unsqueeze(0).repeat(B_V, 1)
                        abs_partial_solu_2 = torch.zeros(B_V, dtype=torch.int64).unsqueeze(1)

                        last_node_index = abs_partial_solu_2[:, [-1]]

                        # 按照距离，选择距离depot最近的一个customer node，将其与depot进行连接，作为第一条边。
                        partial_end_node_coor = get_encoding(state.data, last_node_index)

                        scatter_node_coors = get_encoding(state.data, abs_scatter_solu_1)


                        random_index = ploar_distance(scatter_node_coors[:, :, :2], state.data[:, [0], :2],
                                                        partial_end_node_coor[:, :, :2]).reshape(batch_size, 1)

                        abs_scatter_solu_1_seleted = abs_scatter_solu_1[self.index_gobal, random_index]

                        index1 = torch.arange(abs_scatter_solu_1.shape[1])[None, :].repeat(batch_size, 1)

                        tmp1 = (index1 < random_index).long()

                        tmp2 = (index1 > random_index).long()

                        tmp3 = tmp1 + tmp2

                        abs_scatter_solu_1_unseleted = abs_scatter_solu_1[tmp3.gt(0.5)].reshape(batch_size,
                                                                                                abs_scatter_solu_1.shape[
                                                                                                    1] - 1)
                        abs_scatter_solu_1 = abs_scatter_solu_1_unseleted
                        abs_partial_solu_2 = torch.cat((abs_partial_solu_2, abs_scatter_solu_1_seleted, abs_partial_solu_2), dim=1)

                        abs_partial_solu_2 = tran_to_node_flag(abs_partial_solu_2)
                        last_node_index = abs_scatter_solu_1_seleted

                    else:
                        # print(
                        #     f" {current_step} Max memory during model run: {torch.cuda.max_memory_allocated(device=self.device) / 1024 ** 2} MB")
                        partial_end_node_coor = get_encoding(state.data, last_node_index)

                        scatter_node_coors = get_encoding(state.data, abs_scatter_solu_1)

                        # print(scatter_node_coors.shape, partial_end_node_coor.shape)
                        # Manhattan_Distance = manhattan_distance(scatter_node_coors[:, :, :2],
                        #                                         partial_end_node_coor[:, :, :2])
                        # random_index = torch.argmin(Manhattan_Distance, dim=1).reshape(batch_size, 1)  # [B]
                        random_index = ploar_distance(scatter_node_coors[:, :, :2], state.data[:, [0], :2],
                                                        partial_end_node_coor[:, :, :2]).reshape(batch_size, 1)

                        # 先计算每条subtour的remaining capacity

                        # print(self.origin_problem.shape, abs_partial_solu_2.shape, raw_capacity)

                        remaining_capacity = cal_remaining_capacity(self.origin_problem, abs_partial_solu_2,capacity=raw_capacity)

                        # print('----------------- remaining_capacity \n',remaining_capacity)
                        # print('----------------- abs_partial_solu_2 \n', node_flag_tran_to_(abs_partial_solu_2))

                        abs_partial_solu_2, abs_scatter_solu_1, abs_scatter_solu_1_seleted = \
                            self.model(state.data, self.env.solution, abs_scatter_solu_1, abs_partial_solu_2,
                                        random_index, current_step, last_node_index, self.env.raw_data_capacity,
                                        remaining_capacity_each_subtour=remaining_capacity)

                        last_node_index = abs_scatter_solu_1_seleted
                        abs_partial_solu_2 = tran_to_node_flag(abs_partial_solu_2)

                    current_step += 1
                best_select_node_list = abs_partial_solu_2

            # print('Get first complete solution!')

            # print(
            #     f"Max memory during model run: {torch.cuda.max_memory_allocated(device=self.device) / 1024 ** 2} MB")

            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)

            escape_time, _ = clock.get_est_string(1, 1)

            # self.logger.info("Greedy, name:{}, gap:{:5f} %, Elapsed[{}], stu_l:{:5f} , opt_l:{:5f}".format(name,
            #      ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100,
            #       escape_time,current_best_length.mean().item(), self.optimal_length.mean().item()))

            # # ! 修正版本
            # self.logger.info("Greedy, name:{}, gap:{:5f} %, Elapsed[{}], stu_l:{:5f} , opt_l:{:5f}".format(name,
            #      ((current_best_length.mean() - self.optimal_length) / self.optimal_length).item() * 100,
            #       escape_time,current_best_length.mean().item(), self.optimal_length))

            ####################################################

            # self.env.valida_solution_legal(self.origin_problem, best_select_node_list,capacity_=raw_capacity)

            budget = self.env_params['RRC_budget']

            length_all = torch.randint(low=4, high=min(self.env_params['RRC_range'],self.origin_problem_size), size=[budget])  # in [4,N]

            for bbbb in range(budget):
                # torch.cuda.empty_cache()
                # print(f" {current_step} Max memory during model run: {torch.cuda.max_memory_allocated(device=self.device) / 1024 ** 2} MB")
                # # 1. The complete solution is obtained, which corresponds to the problems of the current env

                self.env.problems     = self.origin_problem
                self.env.problem_size = self.origin_problem_size
                self.env.solution     = self.origin_solution

                # max_control = 10
                # for i in range(max_control):
                #     begin_index = int(batch_size/max_control)*i
                #     end_index = int(batch_size/max_control)*(i+1)
                #
                #     best_select_node_list[begin_index:end_index] = \
                #         Rearrange_solution_clockwise(self.origin_problem[begin_index:end_index], best_select_node_list[begin_index:end_index])

                # 2. Sample the partial solution, reset env, and assign the first node and last node in env
                best_select_node_list = node_flag_tran_to_(best_select_node_list)
                abs_scatter_solu_1, abs_partial_solu_2  = self.env.sampling_subpaths_by_Proximity(
                                        self.env.problems, best_select_node_list, length_all[bbbb])

                best_select_node_list = tran_to_node_flag(best_select_node_list)

                before_reward = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)

                reset_state = self.env.reset(self.env_params['mode'])

                state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node

                # 3. Generate solution 2 again, compare the path lengths of solution 1 and solution 2,
                # and decide which path to accept.

                raw_capacity = self.env.raw_data_capacity[0]

                last_node_index = abs_partial_solu_2[:, [-1], 0]

                current_step = 0

                while  abs_scatter_solu_1.shape[1] >=1 :

                    partial_end_node_coor = get_encoding(state.data, last_node_index)

                    scatter_node_coors = get_encoding(state.data, abs_scatter_solu_1)

                    random_index = ploar_distance(scatter_node_coors[:, :, :2], state.data[:, [0], :2],
                                                  partial_end_node_coor[:, :, :2]).reshape(batch_size, 1)

                    remaining_capacity = cal_remaining_capacity(self.origin_problem, abs_partial_solu_2,
                                                                capacity=raw_capacity)

                    abs_partial_solu_2, abs_scatter_solu_1, abs_scatter_solu_1_seleted = \
                        self.model(state.data, self.env.solution,abs_scatter_solu_1,abs_partial_solu_2,
                                   random_index, current_step, last_node_index,
                                   self.env.raw_data_capacity, remaining_capacity_each_subtour=remaining_capacity)

                    last_node_index = abs_scatter_solu_1_seleted
                    abs_partial_solu_2 = tran_to_node_flag(abs_partial_solu_2)

                    current_step += 1

                after_repair_solution = abs_partial_solu_2

                after_reward = self.env._get_travel_distance_2(self.origin_problem, after_repair_solution)

                # self.env.valida_solution_legal(self.origin_problem, after_repair_solution)

                best_select_node_list = self.decide_whether_to_repair_solution(best_select_node_list, before_reward,
                    after_repair_solution, after_reward)

                current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)

                escape_time, _ = clock.get_est_string(1, 1)

                # self.logger.info("RRC step{}, name:{}, gap:{:6f} %, Elapsed[{}], stu_l:{:5f} , opt_l:{:5f}".format(
                #         bbbb, name, ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100,
                #         escape_time, current_best_length.mean().item(), self.optimal_length.mean().item()))
                
                # # ! 修正版本
                # self.logger.info("RRC step{}, name:{}, gap:{:6f} %, Elapsed[{}], stu_l:{:5f} , opt_l:{:5f}".format(
                #         bbbb, name, ((current_best_length.mean() - self.optimal_length) / self.optimal_length).item() * 100,
                #         escape_time, current_best_length.mean().item(), self.optimal_length))

                # self.env.valida_solution_legal(self.origin_problem, best_select_node_list)
                # self.env.drawPic_VRP(self.origin_problem[0, :, [0, 1]], best_select_node_list[0, :, 0],
                #                      best_select_node_list[0, :, 1], name=name+f'rrc_{bbbb}')

            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)
            # print(f'current_best_length', (current_best_length.mean() - self.optimal_length)
            #       / self.optimal_length * 100, '%', 'escape time:', escape_time,
            #       f'optimal:{self.optimal_length}, current_best:{current_best_length.mean()}')

            # 4. Cycle until the budget is consumed.
            # self.env.valida_solution_legal(self.origin_problem, best_select_node_list, capacity_=raw_capacity)

            # self.env.drawPic_VRP(self.origin_problem[0, :, [0, 1]], best_select_node_list[0, :, 0],
            #                      best_select_node_list[0, :, 1], name=name)

            return self.optimal_length, current_best_length.mean().item(), self.env.problem_size, name

