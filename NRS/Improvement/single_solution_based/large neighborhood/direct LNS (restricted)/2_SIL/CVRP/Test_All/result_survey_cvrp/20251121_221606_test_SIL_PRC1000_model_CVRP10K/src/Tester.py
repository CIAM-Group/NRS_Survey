import torch

import os
from logging import getLogger

from CVRP.Test_All.VRPEnv import VRPEnv as Env
from CVRP.Test_All.VRPModel import VRPModel as Model

from utils.utils import *
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import random



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
        seed = 123
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

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

        self.inst_names = np.array(keys)
        # self.inst_problem_sizes = np.array(numbers)
        self.inst_opt_values = np.array(values)
        self.inst_total_num = len(self.inst_names)

        # bb = os.path.abspath("../../..").replace('\\', '/')
        # test  TSPLIBReader

        validate_name = []
        validate_problem_sizes = []
        validate_opt_values = []

        for i in range(len(self.inst_names)):
            # filename = bb + f'/ba_survey/cvrplib/{self.inst_names[i]}.vrp'
            filename = f'/public/home/bayp/exp_survey_202509/0_data_survey/survey_bench_cvrp/{self.inst_names[i]}.vrp'
            name, dimension, locs, demand, capacity, cost = self.env.CVRPLIBReader(filename)
            #         return name, int(dimension), locs, demand, capacity, cost
            if name is not None:
                validate_name.append(self.inst_names[i])
                validate_problem_sizes.append(dimension)
                validate_opt_values.append(self.inst_opt_values[i])
        print(validate_problem_sizes)

        self.inst_names = np.array(validate_name)
        self.inst_problem_sizes = np.array(validate_problem_sizes)
        self.inst_opt_values = np.array(validate_opt_values)
        self.inst_total_num = len(self.inst_names)

        sort_indices = np.argsort(self.inst_problem_sizes)

        self.inst_names = self.inst_names[sort_indices]
        self.inst_problem_sizes = self.inst_problem_sizes[sort_indices]
        self.inst_opt_values = self.inst_opt_values[sort_indices]

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname,map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()
        self.time_estimator_2 = TimeEstimator()

        # ! ==== NEW: 统计配置与容器 ====
        # 分桶区间：[0,1000), [1000,10000), [10000,100000]
        self.bucket_ranges = [(0, 1000), (1000, 10000), (10000, 100001)]
        # 每个桶分别记录 gaps（百分比值）与 times（秒）
        self.bucket_gaps = [[] for _ in self.bucket_ranges]
        self.bucket_times = [[] for _ in self.bucket_ranges]
        # 总体统计
        self.overall_gaps = []        # 所有成功实例的 gap（百分比）
        self.per_instance_log = []     # 单实例记录：[(name, size, gap%, time_s)]
        self.all_instance_num = 0
        self.all_solved_instance_num = 0

    def run(self):
        self.time_estimator.reset()
        self.time_estimator_2.reset()

        # if self.env_params['load_way'] == 'allin':
        #     self.env.load_raw_data(self.tester_params['test_episodes'])

        k_nearest = self.env_params['k_nearest']
        beam_width = self.env_params['beam_width']
        decode_method = self.env_params['decode_method']

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.inst_total_num # self.tester_params['test_episodes']
        episode = 0

        problems_le_7000 = []
        problems_gt_7000 = []
        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, score_student_mean, aug_score, problems_size = self._test_one_batch(
                episode, batch_size, k_nearest, decode_method, clock=self.time_estimator_2,logger = self.logger)
            if self.env_params['vrplib_path']:
                if problems_size <= 7000:
                    problems_le_7000.append((score_student_mean - score) / score)
                elif 7000 < problems_size:
                    problems_gt_7000.append((score_student_mean - score) / score)

                print('problems_le_7000 mean gap:', np.mean(problems_le_7000), len(problems_le_7000))
                print('problems_gt_7000 mean gap:', np.mean(problems_gt_7000), len(problems_gt_7000))

            score_AM.update(score, batch_size)
            score_student_AM.update(score_student_mean, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f},Score_studetnt: {:.4f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, score_student_mean, aug_score))

            all_done = (episode == test_num_episode)

            gap_ = 1
            if all_done and not self.env_params['vrplib_path']:
                self.logger.info(" *** Test_All Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" NO-AUG SCORE student: {:.4f} ".format(score_student_AM.avg))
                # self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))
                self.logger.info(" Gap: {:.4f}%".format((score_student_AM.avg - score_AM.avg) / score_AM.avg * 100))
                gap_ = (score_student_AM.avg - score_AM.avg) / score_AM.avg * 100

        return score_AM.avg, score_student_AM.avg, gap_

    def decide_whether_to_repair_solution(self,
                                          before_complete_solution, before_repair_sub_solution,
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

        kkk_1 = jjj[index, before_repair_sub_solution[:, :, 0] - 1]

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

        k_nearest = self.env_params['k_nearest']
        beam_width = self.env_params['beam_width']
        decode_method = self.env_params['decode_method']

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.inst_total_num
        episode = 0

        use_cuda = self.tester_params.get('use_cuda', False)
        logger = self.logger

        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            idx_this = episode
            inst_name = None
            try:
                self.all_instance_num += 1
                inst_name = self.inst_names[idx_this] if len(self.inst_names) > 0 else f'inst_{idx_this}'

                # --- 单实例计时 ---
                if use_cuda:
                    torch.cuda.synchronize(self.device)
                import time
                t0 = time.time()

                score, score_student_mean, aug_score, problems_size = self._test_one_batch(
                    episode, batch_size, k_nearest, decode_method, clock=self.time_estimator_2, logger=self.logger
                )

                if use_cuda:
                    torch.cuda.synchronize(self.device)
                inst_time = time.time() - t0

                # --- gap 与统计 ---
                gap_percent = (score_student_mean - score) / score * 100.0
                for b_idx, (lo, hi) in enumerate(self.bucket_ranges):
                    if lo <= problems_size < hi:
                        self.bucket_gaps[b_idx].append(gap_percent)
                        self.bucket_times[b_idx].append(inst_time)
                        break

                self.overall_gaps.append(gap_percent)
                self.per_instance_log.append((str(inst_name), int(problems_size), float(gap_percent), float(inst_time)))
                self.all_solved_instance_num += 1

                score_AM.update(score, batch_size)
                score_student_AM.update(score_student_mean, batch_size)
                aug_score_AM.update(aug_score, batch_size)

                # 补充当前跑的实例个数
                logger.info(
                    "Episode {:3d}/{:3d} done."
                    "Instance: {name}, size: {n}, "
                    "opt: {opt:.3f}, student: {stu:.3f}, gap: {gap:.3f}%, time: {t:.3f}s".format(
                        episode + batch_size, test_num_episode,
                        name=str(inst_name),
                        n=int(problems_size),
                        opt=score,                   # _test_one_batch 返回的 'score' 为最优长度
                        stu=score_student_mean,      # 学生/当前解长度
                        gap=gap_percent,
                        t=inst_time
                    )
                )

            except Exception as e:
                logger.info(f"[SKIP] Instance {inst_name if inst_name is not None else idx_this} failed with error: {e}")
            finally:
                episode += batch_size
                # elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
                # logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, Score_student:{:.4f}".format(
                #     episode, test_num_episode, elapsed_time_str, remain_time_str,
                #     score_AM.avg if score_AM.count > 0 else float('nan'),
                #     score_student_AM.avg if score_student_AM.count > 0 else float('nan'))
                # )

        # --- 分桶汇总 ---
        logger.info("################ Bucket Summary (gap%% / time s) ################")
        for b_idx, (lo, hi) in enumerate(self.bucket_ranges):
            gaps = np.array(self.bucket_gaps[b_idx], dtype=float)
            times = np.array(self.bucket_times[b_idx], dtype=float)
            if len(gaps) > 0:
                logger.info(f"Bucket [{lo}, {hi}): count={len(gaps)}, avg_gap={np.mean(gaps):.3f}% "
                            f"avg_time={np.mean(times):.3f}s")
            else:
                logger.info(f"Bucket [{lo}, {hi}): count=0")

        overall_avg_gap = float(np.mean(self.overall_gaps)) if len(self.overall_gaps) > 0 else float('nan')
        logger.info("################ Overall ################")
        logger.info(f"All solved instances: {self.all_solved_instance_num}/{self.all_instance_num}, "
                    f"Overall avg gap: {overall_avg_gap:.3f}%")

        # # --- 输出每个实例的时间 CSV ---
        # try:
        #     import csv, os
        #     csv_path = os.path.join(self.result_folder, "per_instance_times.csv")
        #     with open(csv_path, "w", newline="") as f:
        #         writer = csv.writer(f)
        #         writer.writerow(["name", "problem_size", "gap_percent", "time_sec"])
        #         for row in self.per_instance_log:
        #             writer.writerow(row)
        #     logger.info(f"Per-instance times saved to: {csv_path}")
        # except Exception as e:
        #     logger.info(f"[WARN] Failed to save per-instance times CSV: {e}")

        # # --- 打印总结果 ---
        # if self.all_solved_instance_num > 0:
        #     logger.info(" *** Test_All Done *** ")
        #     logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
        #     logger.info(" NO-AUG SCORE student: {:.4f} ".format(score_student_AM.avg))
        #     logger.info(" Gap: {:.4f}%".format(overall_avg_gap))

        return score_AM.avg, score_student_AM.avg, overall_avg_gap



    def _test_one_batch(self, episode, batch_size, k_nearest, decode_method, clock=None,logger = None):

        random_seed = 123
        torch.manual_seed(random_seed)


        self.model.eval()

        max_memory_allocated_before = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024
        # print('max_memory_allocated before',max_memory_allocated_before,'MB')
        torch.cuda.reset_peak_memory_stats(device=self.device)


        with torch.no_grad():

            self.env.load_problems(episode, batch_size, self.inst_names[episode], only_test=True)

            reset_state, _, _ = self.env.reset(self.env_params['mode'])

            current_step = 0

            state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node

            self.origin_problem = self.env.problems.clone().detach()
            # print('self.origin_problem.shape',self.origin_problem.shape)

            if self.env.test_in_vrplib:
                optimal_length_and_names = self.env._get_travel_distance_2(self.origin_problem, self.env.solution,
                                                                           test_in_vrplib=self.env.test_in_vrplib,
                                                                           need_optimal=self.env.test_in_vrplib)
                self.optimal_length = optimal_length_and_names[0]
                self.vrp_names = optimal_length_and_names[1]

            else:
                self.optimal_length = self.env._get_travel_distance_2(self.origin_problem, self.env.solution)

            if self.env_params['random_insertion']:
                from utils.insertion import cvrp_random_insertion

                initial_solution = self.env.random_insert(self.origin_problem)
                best_select_node_list = initial_solution
            else:

                B_V = batch_size * 1
                # from tqdm import tqdm
                # with tqdm(total=self.env.problem_size) as pbar:
                while not done:
                    # pbar.update(1)
                    loss_node, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = \
                        self.model(state, self.env.selected_node_list, self.env.solution, current_step,
                                    raw_data_capacity=self.env.raw_data_capacity, decode_method=decode_method)
                    if current_step == 0:
                        selected_flag_teacher = torch.ones(B_V, dtype=torch.int)
                        selected_flag_student = selected_flag_teacher
                    current_step += 1

                    state, reward, reward_student, done = \
                        self.env.step(selected_teacher, selected_student, selected_flag_teacher, selected_flag_student)
                    # print(current_step)


                # print('Get first complete solution!')

                best_select_node_list = torch.cat((self.env.selected_student_list.reshape(batch_size, -1, 1),
                                                   self.env.selected_student_flag.reshape(batch_size, -1, 1)), dim=2)

            max_memory_allocated_after = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024

            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list,
                                                                  test_in_vrplib=self.env.test_in_vrplib)

            escape_time, _ = clock.get_est_string(1, 1)


            # if self.env.test_in_vrplib:
            #     self.logger.info("curr00, {} gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(self.vrp_names,
            #       (( current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100,
            #      escape_time, current_best_length.mean().item(),  self.optimal_length.mean().item()))
            # else:
            #     self.logger.info("curr00,  gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}, Memory:{:4f}MB,".format(
            #         ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100, escape_time,
            #     current_best_length.mean().item(), self.optimal_length.mean().item(), max_memory_allocated_after))


            budget = self.env_params['budget']

            origin_problem_size = self.origin_problem.shape[1]
            origin_batch_size = batch_size



            max_length = min(origin_problem_size, self.env_params['repair_max_sub_length'])
            if origin_problem_size<=1000:
                length_all = torch.randint(4,
                                           high=max_length, size=[budget])  # in [4,N]
            else:
                length_all  = torch.randint(4,
                                            high=max_length+1, size=[budget])  # in [4,N]
            first_index_all = torch.randint(low=0, high=origin_problem_size, size=[budget])  # in [4,N]


            for bbbb in range(budget):
                torch.cuda.empty_cache()

                # print(length_all[bbbb])

                best_select_node_list = self.env.Rearrange_solution_caller(
                    self.origin_problem, best_select_node_list)

                self.env.load_problems(episode, batch_size, self.inst_names[episode] , only_test=True)

                best_select_node_list = self.env.vrp_whole_and_solution_subrandom_inverse(best_select_node_list)
                fix_length =torch.randint(low=4, high=self.env_params['repair_max_sub_length'], size=[1])[0]  # in [4,N]
                if self.env_params['PRC']:

                    if origin_problem_size<=1000:

                        if int(origin_problem_size / length_all[bbbb])<=1:

                            partial_solution_length, first_node_index, length_of_subpath, double_solution = \
                                self.env.destroy_solution(self.env.problems, best_select_node_list, length_all[bbbb],
                                                          first_index_all[bbbb])
                        else:
                            partial_solution_length, first_node_index, end_node_index, length_of_subpath, \
                            double_solution, origin_sub_solution, index4, factor = \
                                self.env.destroy_solution_PRC(self.env.problems, best_select_node_list,
                                                              length_all[bbbb], first_index_all[bbbb])

                    else:
                        partial_solution_length, first_node_index, end_node_index, length_of_subpath, \
                        double_solution, origin_sub_solution, index4, factor = \
                            self.env.destroy_solution_PRC(self.env.problems, best_select_node_list,
                                                      length_all[bbbb], first_index_all[bbbb])

                else:
                    partial_solution_length, first_node_index, length_of_subpath, double_solution = \
                    self.env.destroy_solution(self.env.problems, best_select_node_list,length_all[bbbb], first_index_all[bbbb])

                before_repair_sub_solution = self.env.solution

                self.env.batch_size = before_repair_sub_solution.shape[0]

                before_reward = partial_solution_length

                current_step = 0

                reset_state, _, _ = self.env.reset(self.env_params['mode'])

                state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node


                while not done:
                    if current_step == 0:

                        selected_teacher = self.env.solution[:, 0, 0]
                        selected_flag_teacher = self.env.solution[:, 0, 1]
                        selected_student = selected_teacher
                        selected_flag_student = selected_flag_teacher


                    else:
                        _, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = \
                            self.model(state, self.env.selected_node_list, self.env.solution, current_step,
                                       raw_data_capacity=self.env.raw_data_capacity, decode_method=decode_method)

                    current_step += 1

                    state, reward, reward_student, done = \
                        self.env.step(selected_teacher, selected_student, selected_flag_teacher, selected_flag_student)

                ahter_repair_sub_solution = torch.cat((self.env.selected_student_list.unsqueeze(2),
                                                       self.env.selected_student_flag.unsqueeze(2)), dim=2)

                after_reward = reward_student

                if self.env_params['PRC']:

                    if origin_problem_size <= 1000:

                        if int(origin_problem_size / length_all[bbbb]) <= 1:
                            after_repair_complete_solution = self.env.decide_whether_to_repair_solution(
                                ahter_repair_sub_solution,
                                before_reward, after_reward, first_node_index, length_of_subpath, double_solution)
                        else:
                            after_repair_complete_solution = self.env.decide_whether_to_repair_solution_V2(
                                ahter_repair_sub_solution, before_reward, after_reward, double_solution,
                                origin_sub_solution,
                                index4, origin_batch_size, factor)
                    else:


                        after_repair_complete_solution = self.env.decide_whether_to_repair_solution_V2(
                            ahter_repair_sub_solution,before_reward, after_reward,double_solution, origin_sub_solution,
                            index4, origin_batch_size, factor)
                else:

                    after_repair_complete_solution = self.env.decide_whether_to_repair_solution(ahter_repair_sub_solution,
                        before_reward, after_reward, first_node_index, length_of_subpath,double_solution)


                best_select_node_list = after_repair_complete_solution

                current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list,
                                                                      test_in_vrplib=self.env.test_in_vrplib)

                escape_time, _ = clock.get_est_string(1, 1)

                max_memory_allocated_after = torch.cuda.max_memory_allocated(device=self.device) / 1024 / 1024

                # if self.env.test_in_vrplib:
                #     self.logger.info( " step{},{}, gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                #             bbbb, self.vrp_names, ((  current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100,
                #             escape_time, current_best_length.mean().item(), self.optimal_length.mean().item()))
                # else:
                #     self.logger.info(
                #         " step{}, gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f},, Memory:{:4f}MB".format(
                #             bbbb, ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100,
                #             escape_time, current_best_length.mean().item(), self.optimal_length.mean().item(),
                #             max_memory_allocated_after))

            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list,
                                                                  test_in_vrplib=self.env.test_in_vrplib)

            # print(f'current_best_length', (current_best_length.mean() - self.optimal_length.mean())
            #       / self.optimal_length.mean() * 100, '%', 'escape time:', escape_time,
            #       f'optimal:{self.optimal_length.mean()}, current_best:{current_best_length.mean()}')

            return self.optimal_length.mean().item(), current_best_length.mean().item(), self.optimal_length.mean().item(), self.env.problem_size
