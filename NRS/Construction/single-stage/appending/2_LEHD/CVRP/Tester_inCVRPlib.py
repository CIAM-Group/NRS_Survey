from logging import getLogger

import torch

from CVRP.VRPModel import VRPModel as Model
from CVRP.VRPEnv_inCVRPlib import VRPEnv as Env
from utils.utils import *

import time
import numpy as np

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'




cvrplib_opt_cost = {
    # # Rochat and Taillard (1995)
    # "tai75a": 1618.36,
    # "tai75b": 1344.62,
    # "tai75c": 1291.01,
    # "tai75d": 1365.42,
    # "tai100a": 2041.34,
    # "tai100b": 1939.90,
    # "tai100c": 1406.20,
    # "tai100d": 1580.46,
    # "tai150a": 3055.23,
    # "tai150b": 2727.03,
    # "tai150c": 2358.66,
    # "tai150d": 2645.39,
    # "tai385": 24366.41339,

    # # Golden et al. (1998)
    # "Golden_9": 579.702026,
    # "Golden_10": 735.427307,
    # "Golden_11": 911.980164,
    # "Golden_12": 1100.665283,
    # "Golden_13": 857.189,
    # "Golden_14": 1080.55,
    # "Golden_15": 1337.2677,
    # "Golden_16": 1611.2769688292835,
    # "Golden_17": 707.756,
    # "Golden_18": 995.133,
    # "Golden_19": 1365.6,
    # "Golden_20": 1817.59,

    # Arnold, Gendreau and Sörensen (2017)
    "Antwerp1": 477277,
    "Antwerp2": 291350,
    "Brussels1": 501719,
    "Brussels2": 345468,
    "Flanders1": 7240118,
    "Flanders2": 4373244,
    "Ghent1": 469531,
    "Ghent2": 257749,
    "Leuven1": 192848,
    "Leuven2": 111395,

    # Uchoa et al. (2014)
    "X-n101-k25": 27591,
    "X-n106-k14": 26362,
    "X-n110-k13": 14971,
    "X-n115-k10": 12747,
    "X-n120-k6": 13332,
    "X-n125-k30": 55539,
    "X-n129-k18": 28940,
    "X-n134-k13": 10916,
    "X-n139-k10": 13590,
    "X-n143-k7": 15700,
    "X-n148-k46": 43448,
    "X-n153-k22": 21220,
    "X-n157-k13": 16876,
    "X-n162-k11": 14138,
    "X-n167-k10": 20557,
    "X-n172-k51": 45607,
    "X-n176-k26": 47812,
    "X-n181-k23": 25569,
    "X-n186-k15": 24145,
    "X-n190-k8": 16980,
    "X-n195-k51": 44225,
    "X-n200-k36": 58578,
    "X-n204-k19": 19565,
    "X-n209-k16": 30656,
    "X-n214-k11": 10856,
    "X-n219-k73": 117595,
    "X-n223-k34": 40437,
    "X-n228-k23": 25742,
    "X-n233-k16": 19230,
    "X-n237-k14": 27042,
    "X-n242-k48": 82751,
    "X-n247-k50": 37274,
    "X-n251-k28": 38684,
    "X-n256-k16": 18839,
    "X-n261-k13": 26558,
    "X-n266-k58": 75478,
    "X-n270-k35": 35291,
    "X-n275-k28": 21245,
    "X-n280-k17": 33503,
    "X-n284-k15": 20226,
    "X-n289-k60": 95151,
    "X-n294-k50": 47161,
    "X-n298-k31": 34231,
    "X-n303-k21": 21736,
    "X-n308-k13": 25859,
    "X-n313-k71": 94043,
    "X-n317-k53": 78355,
    "X-n322-k28": 29834,
    "X-n327-k20": 27532,
    "X-n331-k15": 31102,
    "X-n336-k84": 139111,
    "X-n344-k43": 42050,
    "X-n351-k40": 25896,
    "X-n359-k29": 51505,
    "X-n367-k17": 22814,
    "X-n376-k94": 147713,
    "X-n384-k52": 65940,
    "X-n393-k38": 38260,
    "X-n401-k29": 66154,
    "X-n411-k19": 19712,
    "X-n420-k130": 107798,
    "X-n429-k61": 65449,
    "X-n439-k37": 36391,
    "X-n449-k29": 55233,
    "X-n459-k26": 24139,
    "X-n469-k138": 221824,
    "X-n480-k70": 89449,
    "X-n491-k59": 66483,
    "X-n502-k39": 69226,
    "X-n513-k21": 24201,
    "X-n524-k153": 154593,
    "X-n536-k96": 94846,
    "X-n548-k50": 86700,
    "X-n561-k42": 42717,
    "X-n573-k30": 50673,
    "X-n586-k159": 190316,
    "X-n599-k92": 108451,
    "X-n613-k62": 59535,
    "X-n627-k43": 62164,
    "X-n641-k35": 63684,
    "X-n655-k131": 106780,
    "X-n670-k130": 146332,
    "X-n685-k75": 68205,
    "X-n701-k44": 81923,
    "X-n716-k35": 43373,
    "X-n733-k159": 136187,
    "X-n749-k98": 77269,
    "X-n766-k71": 114417,
    "X-n783-k48": 72386,
    "X-n801-k40": 73311,
    "X-n819-k171": 158121,
    "X-n837-k142": 193737,
    "X-n856-k95": 88965,
    "X-n876-k59": 99299,
    "X-n895-k37": 53860,
    "X-n916-k207": 329179,
    "X-n936-k151": 132715,
    "X-n957-k87": 85465,
    "X-n979-k58": 118976,
    "X-n1001-k43": 72355,
}


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

        keys = []
        # numbers = []
        values = []
        import re
        # 遍历字典的每一对键和值
        for key, value in cvrplib_opt_cost.items():
            # 1. 将键和值分别添加到列表中
            keys.append(key)
            values.append(value)

            # # 2. 从键中提取数字
            # # re.search(r'\d+', key) 会在字符串中查找第一个连续的数字序列
            # match = re.search(r'\d+', key)
            # if match:
            #     # 如果找到数字，将其转换为整数并添加到列表
            #     number = int(match.group(0))
            #     numbers.append(number)
            # else:
            #     # 如果键中没有数字，可以添加一个默认值，例如 None
            #     numbers.append(None)

        self.inst_names = np.array(keys)
        # self.inst_problem_sizes = np.array(numbers)
        self.inst_opt_values = np.array(values)
        self.inst_total_num = len(self.inst_names)



        # bb = os.path.abspath("../..").replace('\\', '/')
        # test  TSPLIBReader

        validate_name = []
        validate_problem_sizes = []
        validate_opt_values = []

        for i in range(len(self.inst_names)):
            # filename = f'/ba_survey/cvrplib/{self.inst_names[i]}.vrp'
            filename = f'/public/home/bayp/exp_survey_202509/0_data_survey/survey_bench_cvrp/{self.inst_names[i]}.vrp'
            name, dimension, locs, demand, capacity, cost = self.env.CVRPLIBReader(filename)
            #         return name, int(dimension), locs, demand, capacity, cost
            if name is not None:
                validate_name.append(self.inst_names[i])
                validate_problem_sizes.append(dimension)
                validate_opt_values.append(self.inst_opt_values[i])
        print(validate_problem_sizes)

        self.inst_names =  np.array(validate_name)
        self.inst_problem_sizes = np.array(validate_problem_sizes)
        self.inst_opt_values =  np.array(validate_opt_values)
        self.inst_total_num = len(self.inst_names)

        sort_indices = np.argsort(self.inst_problem_sizes)

        self.inst_names = self.inst_names[sort_indices]
        self.inst_problem_sizes = self.inst_problem_sizes[sort_indices]
        self.inst_opt_values = self.inst_opt_values[sort_indices]

        # !==== ICAM 风格分桶：仅 3 桶 ====
        # 区间分别是 [0,1000), [1000,10000), [10000,100001]
        self.scale_buckets = [
            ("[0,1000)",        lambda n: (n >= 0) and (n < 1000)),
            ("[1000,10000)",    lambda n: (n >= 1000) and (n < 10000)),
            ("[10000,100001]",  lambda n: (n >= 10000) and (n < 100001)),
        ]
        # 每桶统计 gap(%) 与 time(s)
        self.bucket_stats = {k: {"gaps": [], "times": [], "count": 0} for k, _ in self.scale_buckets}
        # 逐实例明细（供详细日志/复盘）
        self.per_instance = []   # {name, size, gap_pct, time_sec}
        # 计总
        self.total_instances = self.inst_total_num
        self.solved_instances = 0


        # print(self.inst_names)
        # print(self.inst_problem_sizes)
        # print(self.inst_opt_values)
        # assert False

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()
        self.time_estimator_2 = TimeEstimator()

    def run(self):
        self.time_estimator.reset()
        self.time_estimator_2.reset()


        # self.env.load_raw_data(self.tester_params['test_episodes'])

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()

        test_num_episode = self.inst_total_num # self.tester_params['test_episodes']
        episode = 0 # self.tester_params['begin_index']
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

            try:
                score_teacher, score_student, problems_size, vrpname = self._test_one_batch(
                    episode, batch_size, clock=self.time_estimator_2,logger = self.logger)
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


                if vrpname[:2]=='A-':
                    problems_A.append(current_gap)
                elif vrpname[:2]=='B-':
                    problems_B.append(current_gap)
                elif vrpname[:2]=='E-':
                    problems_E.append(current_gap)
                elif vrpname[:2]=='F-':
                    problems_F.append(current_gap)
                elif vrpname[:2]=='M-':
                    problems_M.append(current_gap)
                elif vrpname[:2]=='P-':
                    problems_P.append(current_gap)
                elif vrpname[:2]=='X-':
                    problems_X.append(current_gap)

                print('problems_100 mean gap:', np.mean(problems_100), len(problems_100))
                print('problems_100_200 mean gap:', np.mean(problems_100_200), len(problems_100_200))
                print('problems_200_500 mean gap:', np.mean(problems_200_500), len(problems_200_500))
                print('problems_500_1000 mean gap:', np.mean(problems_500_1000), len(problems_500_1000))
                print('problems_1000 mean gap:', np.mean(problems_1000), len(problems_1000))

                self.logger.info(" problems_A    mean gap:{:4f}%, num:{}".format(np.mean( problems_A)*100,len( problems_A) ))
                self.logger.info(" problems_B    mean gap:{:4f}%, num:{}".format(np.mean( problems_B)*100,len( problems_B) ))
                self.logger.info(" problems_E    mean gap:{:4f}%, num:{}".format(np.mean( problems_E)*100, len(problems_E)))
                self.logger.info(" problems_F    mean gap:{:4f}%, num:{}".format(np.mean( problems_F)*100, len(problems_F)))
                self.logger.info(" problems_M    mean gap:{:4f}%, num:{}".format(np.mean( problems_M)*100, len(problems_M)))
                self.logger.info(" problems_P    mean gap:{:4f}%, num:{}".format(np.mean( problems_P)*100, len(problems_P)))
                self.logger.info(" problems_X    mean gap:{:4f}%, num:{}".format(np.mean( problems_X)*100, len(problems_X)))


                score_AM.update(score_teacher, batch_size)
                score_student_AM.update(score_student, batch_size)

                episode += batch_size

                elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
                self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], Score_teacher:{:.4f}, Score_studetnt: {:.4f}".format(
                    episode, test_num_episode, elapsed_time_str, remain_time_str, score_teacher, score_student))
            except Exception as e:
                self.logger.error(f"[Error] _test_one_batch failed: {e}", exc_info=True)
                episode += batch_size

            ############################
            # Logs
            ############################
            

            all_done = (episode == test_num_episode)

            if all_done:
                if self.env_params['test_in_vrplib']:
                    self.logger.info(" *** Test Done *** ")
                    all_result_gaps = problems_A + problems_B + problems_E + problems_F + problems_M + problems_P + problems_X
                    gap_ = np.mean(all_result_gaps)*100
                    self.logger.info(" Gap: {:.4f}%".format(gap_))
                else:
                    self.logger.info(" *** Test Done *** ")
                    self.logger.info(" Teacher SCORE: {:.4f} ".format(score_AM.avg))
                    self.logger.info(" Student SCORE: {:.4f} ".format(score_student_AM.avg))
                    gap_ = (score_student_AM.avg - score_AM.avg) / score_AM.avg * 100
                    self.logger.info(" Gap: {:.4f}%".format(gap_))

        return score_AM.avg, score_student_AM.avg, gap_

    def decide_whether_to_repair_solution(self,
                                          after_repair_sub_solution, before_reward, after_reward,
                                          first_node_index, length_of_subpath, double_solution):

        the_whole_problem_size = int(double_solution.shape[1] / 2)
        batch_size = len(double_solution)

        temp = torch.arange(double_solution.shape[1])

        x3 = temp >= first_node_index[:, None].long()
        x4 = temp < (first_node_index[:, None] + length_of_subpath).long()
        x5 = x3 * x4

        origin_sub_solution = double_solution[x5.unsqueeze(2).repeat(1, 1, 2)].reshape(batch_size, length_of_subpath, 2)

        jjj, _ = torch.sort(origin_sub_solution[:, :, 0], dim=1, descending=False)

        index = torch.arange(batch_size)[:, None].repeat(1, jjj.shape[1])

        kkk_2 = jjj[index, after_repair_sub_solution[:, :, 0] - 1]

        after_repair_sub_solution[:, :, 0] = kkk_2

        if_repair = before_reward > after_reward

        need_to_repari_double_solution = double_solution[if_repair]
        need_to_repari_double_solution[x5[if_repair].unsqueeze(2).repeat(1, 1, 2)] = after_repair_sub_solution[if_repair].ravel()
        double_solution[if_repair] = need_to_repari_double_solution

        x6 = temp >= (first_node_index[:, None] + length_of_subpath - the_whole_problem_size).long()

        x7 = temp < (first_node_index[:, None] + length_of_subpath).long()

        x8 = x6 * x7

        after_repair_complete_solution = double_solution[x8.unsqueeze(2).repeat(1, 1, 2)].reshape(batch_size, the_whole_problem_size, -1)

        return after_repair_complete_solution
    
    def run_lib(self):
        self.time_estimator.reset()
        self.time_estimator_2.reset()

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()

        test_num_episode = self.inst_total_num
        episode = 0

        start_time_all = time.time()

        while episode < test_num_episode:
            batch_size = 1  # 单实例测试，便于计时与隔离

            inst_name = self.inst_names[episode]
            inst_size = int(self.inst_problem_sizes[episode])

            # ==== 逐实例计时开始 ====
            try:
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                inst_start = time.time()

                score_teacher, score_student, problems_size, vrpname = self._test_one_batch(
                    episode, batch_size, clock=self.time_estimator_2, logger=self.logger
                )

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                inst_time = time.time() - inst_start

                # gap 百分比
                current_gap_ratio = ((score_student - score_teacher) / score_teacher)
                gap_pct = current_gap_ratio * 100.0

                # 入桶（ICAM 3 桶）
                for label, cond in self.scale_buckets:
                    if cond(problems_size):
                        self.bucket_stats[label]["gaps"].append(gap_pct)
                        self.bucket_stats[label]["times"].append(inst_time)
                        self.bucket_stats[label]["count"] += 1
                        break

                # 逐实例记录
                self.per_instance.append({
                    "name": vrpname,
                    "size": int(problems_size),
                    "gap_pct": float(gap_pct),
                    "time_sec": float(inst_time),
                })
                self.solved_instances += 1

                # 均值器 & 日志
                score_AM.update(score_teacher, batch_size)
                score_student_AM.update(score_student, batch_size)

                elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode + 1, test_num_episode)
                # self.logger.info(
                #     "ep {:3d}/{:3d}, Elapsed[{}], Remain[{}], "
                #     "Teacher:{:.4f}, Student:{:.4f}, Gap:{:.3f}%, Time:{:.3f}s".format(
                #         episode + 1, test_num_episode, elapsed_time_str, remain_time_str,
                #         score_teacher, score_student, gap_pct, inst_time
                #     )
                # )
                self.logger.info(
                    "ep {:3d}/{:3d}, Elapsed[{}], Remain[{}], "
                    "Dim:{:5d}, Teacher:{:.4f}, Student:{:.4f}, Gap:{:.3f}%, Time:{:.3f}s".format(
                        episode + 1, test_num_episode, elapsed_time_str, remain_time_str,
                        int(problems_size), score_teacher, score_student, gap_pct, inst_time
                    )
                )


            except Exception as e:
                # 单实例失败不中断
                self.logger.error(f"[Error] instance '{inst_name}' (size {inst_size}) failed: {e}", exc_info=True)

            # 下一实例（已按尺寸升序）
            episode += 1

        # ==== 汇总 ====
        total_time_all = time.time() - start_time_all
        self.logger.info(" *** Test Done *** ")
        self.logger.info("Solved {}/{} instances, total time: {:.2f}s, avg time per solved: {:.2f}s".format(
            self.solved_instances, self.total_instances,
            total_time_all, (total_time_all / self.solved_instances) if self.solved_instances > 0 else float('nan'))
        )

        # 按 ICAM 3 桶打印
        for label, _ in self.scale_buckets:
            gaps = self.bucket_stats[label]["gaps"]
            times = self.bucket_stats[label]["times"]
            cnt = self.bucket_stats[label]["count"]
            if cnt > 0:
                self.logger.info(f"{label}  num:{cnt}, avg gap(no aug): {np.mean(gaps):.3f}%, avg time/inst: {np.mean(times):.3f}s")
            else:
                self.logger.info(f"{label}  num:0, avg gap(no aug): 0.000%, avg time/inst: 0.000s")

        # # 逐实例明细（与 ICAM 的 detailed_log 类似）
        # self.logger.info("===== Per-instance (name, size, gap%, time s) =====")
        # for rec in self.per_instance:
        #     self.logger.info(f"{rec['name']}, {rec['size']}, {rec['gap_pct']:.3f}%, {rec['time_sec']:.3f}s")

        # 返回（保持原接口）
        if score_AM.count > 0:
            final_gap = (score_student_AM.avg - score_AM.avg) / score_AM.avg * 100
        else:
            final_gap = float('nan')
        return score_AM.avg, score_student_AM.avg, final_gap





    def _test_one_batch(self, episode, batch_size, clock=None,logger = None):

        random_seed = 12
        torch.manual_seed(random_seed)

        ###############################################
        self.model.eval()

        with torch.no_grad():

            self.env.load_problems(episode, batch_size, self.inst_names[episode] )

            reset_state, _, _ = self.env.reset(self.env_params['mode'])

            current_step = 0

            state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node

            self.origin_problem = self.env.problems.clone().detach()

            if self.env.test_in_vrplib:
                self.optimal_length, name  = self.env._get_travel_distance_2(self.origin_problem, self.env.solution,
                                                                      need_optimal=True)
            else:
                self.optimal_length= self.env._get_travel_distance_2(self.origin_problem, self.env.solution)
                name = 'vrp'+str(self.env.solution.shape[1])
            B_V = batch_size * 1

            while not done:

                loss_node, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = \
                    self.model(state, self.env.selected_node_list, self.env.solution, current_step,
                               raw_data_capacity=self.env.raw_data_capacity)  # 更新被选择的点和概率

                if current_step == 0:
                    selected_flag_teacher = torch.ones(B_V, dtype=torch.int)
                    selected_flag_student = selected_flag_teacher
                current_step += 1

                state, reward, reward_student, done = \
                    self.env.step(selected_teacher, selected_student, selected_flag_teacher, selected_flag_student)

            # print('Get first complete solution!')


            best_select_node_list = torch.cat((self.env.selected_student_list.reshape(batch_size, -1, 1),
                                               self.env.selected_student_flag.reshape(batch_size, -1, 1)), dim=2)

            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)

            escape_time, _ = clock.get_est_string(1, 1)

            self.logger.info("Greedy, name:{}, gap:{:5f} %, Elapsed[{}], stu_l:{:5f} , opt_l:{:5f}".format(name,
                ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100, escape_time,
            current_best_length.mean().item(), self.optimal_length.mean().item()))


            ####################################################

            budget = self.env_params['RRC_budget']

            for bbbb in range(budget):
                torch.cuda.empty_cache()

                # 1. The complete solution is obtained, which corresponds to the problems of the current env

                self.env.load_problems(episode, batch_size, self.inst_names[episode])

                # 2. Sample the partial solution, reset env, and assign the first node and last node in env

                best_select_node_list = self.env.vrp_whole_and_solution_subrandom_inverse(best_select_node_list)

                partial_solution_length, first_node_index, length_of_subpath, double_solution = \
                    self.env.destroy_solution(self.env.problems, best_select_node_list)

                before_repair_sub_solution = self.env.solution

                before_reward = partial_solution_length

                current_step = 0

                reset_state, _, _ = self.env.reset(self.env_params['mode'])

                state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node

                # 3. Generate solution 2 again, compare the path lengths of solution 1 and solution 2,
                # and decide which path to accept.

                while not done:
                    if current_step == 0:
                        selected_teacher = self.env.solution[:, 0, 0]
                        selected_flag_teacher = self.env.solution[:, 0, 1]
                        selected_student = selected_teacher
                        selected_flag_student = selected_flag_teacher


                    else:
                        _, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = \
                            self.model(state, self.env.selected_node_list, self.env.solution, current_step,
                                       raw_data_capacity=self.env.raw_data_capacity)

                    current_step += 1

                    state, reward, reward_student, done = \
                        self.env.step(selected_teacher, selected_student, selected_flag_teacher, selected_flag_student)

                ahter_repair_sub_solution = torch.cat((self.env.selected_student_list.unsqueeze(2),
                                                       self.env.selected_student_flag.unsqueeze(2)), dim=2)

                after_reward = - reward_student

                after_repair_complete_solution = self.decide_whether_to_repair_solution(
                     ahter_repair_sub_solution,
                    before_reward, after_reward, first_node_index, length_of_subpath, double_solution)

                best_select_node_list = after_repair_complete_solution

                current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)

                escape_time, _ = clock.get_est_string(1, 1)

                self.logger.info(
                    "RRC step{}, name:{}, gap:{:6f} %, Elapsed[{}], stu_l:{:5f} , opt_l:{:5f}".format(
                         bbbb, name, ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100,
                        escape_time,current_best_length.mean().item(), self.optimal_length.mean().item()))

            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)
            print(f'current_best_length', (current_best_length.mean() - self.optimal_length.mean())
                  / self.optimal_length.mean() * 100, '%', 'escape time:', escape_time,
                  f'optimal:{self.optimal_length.mean()}, current_best:{current_best_length.mean()}')

            # 4. Cycle until the budget is consumed.
            # self.env.valida_solution_legal(self.origin_problem, best_select_node_list)

            # self.env.drawPic_VRP(self.origin_problem[0,:,[0,1]], best_select_node_list[0,:,0],best_select_node_list[0,:,1],name=name)

            return self.optimal_length.mean().item(), current_best_length.mean().item(), self.env.problem_size, name
