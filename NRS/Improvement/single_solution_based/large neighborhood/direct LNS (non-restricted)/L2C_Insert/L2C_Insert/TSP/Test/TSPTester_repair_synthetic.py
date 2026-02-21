
from logging import getLogger

import numpy as np
import torch

from L2C_Insert.TSP.Test.TSPModel import TSPModel as Model
from L2C_Insert.TSP.Test.TSPEnv_synthetic import TSPEnv as Env
from L2C_Insert.TSP.utils.utils import *
import random
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class TSPTester():
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
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}'.format(**model_load)

        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        torch.set_printoptions(precision=20)
        # utility
        self.time_estimator = TimeEstimator()
        self.time_estimator_2 =  TimeEstimator()

    def run(self):
        self.time_estimator.reset()
        self.time_estimator_2.reset()
        
        start_time = time.time()
        self.logger.info(" ***** Test Start ***** ")

        if not self.env_params['test_in_tsplib']:
            self.env.load_raw_data(self.tester_params['test_episodes'] )



        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        aug_score_AM = AverageMeter()
        gap_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        problems_100 = []
        problems_100_200 = []
        problems_200_500 = []
        problems_500_1000 = []
        problems_1000 = []
        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score_teacher, score_student,problems_size, batch_gap = self._test_one_batch(episode,batch_size,clock=self.time_estimator_2)
            current_gap = batch_gap / 100
            if problems_size<100:
                problems_100.append(current_gap)
            elif 100<=problems_size<200:
                problems_100_200.append(current_gap)
            elif 200<=problems_size<500:
                problems_200_500.append(current_gap)
            elif 500<=problems_size<1000:
                problems_500_1000.append(current_gap)
            elif 1000<=problems_size:
                problems_1000.append(current_gap)

            print('problems_100 mean gap:',np.mean(problems_100),len(problems_100))
            print('problems_100_200 mean gap:', np.mean(problems_100_200),len(problems_100_200))
            print('problems_200_500 mean gap:', np.mean(problems_200_500),len(problems_200_500))
            print('problems_500_1000 mean gap:', np.mean(problems_500_1000),len(problems_500_1000))
            print('problems_1000 mean gap:', np.mean(problems_1000),len(problems_1000))
            score_AM.update(score_teacher, batch_size)
            score_student_AM.update(score_student, batch_size)
            gap_AM.update(batch_gap, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], Score_teacher:{:.4f},Score_studetnt: {:.4f},".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score_teacher,score_student,))

            all_done = (episode == test_num_episode)

            if all_done:
                end_time = time.time()
                total_time = end_time - start_time
                
                if not self.env_params['test_in_tsplib']:
                    self.logger.info(" *** Test Done *** ")
                    self.logger.info(" Teacher SCORE: {:.4f} ".format(score_AM.avg))
                    self.logger.info(" Student SCORE: {:.4f} ".format(score_student_AM.avg))
                    self.logger.info(" Gap: {:.4f}%".format(gap_AM.avg))
                    gap_ = gap_AM.avg

                else:
                    self.logger.info(" *** Test Done *** ")
                    all_result_gaps = problems_1000 + problems_500_1000 + problems_200_500 + problems_100_200 + problems_100
                    average_gap = np.mean(all_result_gaps)
                    self.logger.info(" Average Gap: {:.4f}%".format(average_gap*100))
                    gap_ = average_gap
                self.logger.info(" Total Test Time: {0:.2f} sec, {1:.2f} min, {2:.2f} hr ".format(
                    total_time, total_time / 60, total_time / 3600))

        return score_AM.avg, score_student_AM.avg, gap_

    def decide_whether_to_repair_solution(self,before_solution,before_reward,after_solution,after_reward):

        if_repair = before_reward>after_reward

        before_solution[if_repair] = after_solution[if_repair]

        return before_solution

    def sampling_subpaths_L2Insert(self, problems, solution, length_fix):

        problems_size = problems.shape[1]

        # mm = torch.randint(low=4, high=problems_size, size=[1])[0].item()  # in [0,N)
        #
        # solution = torch.roll(solution, shifts=mm, dims=1)

        # 2. 将 solution分割成两部分，

        # max_range = min(self.env_params['max_RRC_range'],problems_size)
        #
        # length_fix = torch.randint(low=4, high=max_range, size=[1])[0]  # in [0,N)

        abs_scatter_solu_1 = solution[:, :length_fix]
        abs_partial_solu_2 = solution[:, length_fix:]

        return solution, abs_scatter_solu_1, abs_partial_solu_2

    def sampling_subpaths_by_Proximity(self, problems, solution, length_sub):

        problems_size = problems.shape[1]
        batch_size = problems.shape[0]

        mm = torch.randint(low=4, high=problems_size, size=[1])[0].item()  # in [0,N)
        solution = torch.roll(solution, shifts=mm, dims=1)

        position = torch.randint(low=0, high=self.origin_problem_size, size=[1])[0]  # in [4,N]

        # 选中solution中的一点
        selected_node_index = solution[0, position]

        # 这一点对应的坐标
        selected_one_node = problems[:, [selected_node_index], :]

        # 把instance的所有点的坐标按照solution进行排序
        tmp_index1 = torch.arange(batch_size)[:, None].repeat(1, problems_size)
        problems_sorted_by_solution = problems[tmp_index1, solution]

        # 计算所有点距离被选点的距离
        distance = torch.norm(problems_sorted_by_solution - selected_one_node, dim=-1)

        # distance = manhattan_distance(problems_sorted_by_solution, selected_one_node)

        # 计算所有点距离被选点的距离
        sorted_distance, sorted_index = torch.sort(distance, dim=1, descending=False)
        # print(sorted_distance)

        # zz = torch.randperm(problems_size,dtype=torch.long)
        # sorted_index = sorted_index[:,zz]
        # near_node_num = torch.randint(low=10, high=length_sub, size=[1])[0]  # in [4,N]

        # 这个radius是用来画图的
        # radius = sorted_distance[0, length_sub - 1]

        # 选择 k-nearest 的 index
        sorted_index = sorted_index[:, :length_sub]

        tmp_index = torch.arange(batch_size)[:, None].repeat(1, length_sub)
        selected_solution_index = solution[tmp_index, sorted_index]

        def _get_new_data_v2(data, selected_node_list, prob_size, B_V):
            # print(data[-1,:,0])
            # def sort_solu_index(new_sulution):
            #     new_sulution_ascending, rank = torch.sort(new_sulution, dim=-1, descending=False)  # 升序
            #     _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)  # 升序
            #     return new_sulution_rank

            new_sulution_ascending, rank = torch.sort(data, dim=-1, descending=False)  # 升序
            _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)  # 升序

            list = selected_node_list

            new_list = torch.arange(prob_size)[None, :].repeat(B_V, 1)

            new_list_len = prob_size - list.shape[1]  # shape: [B, V-current_step]

            index_2 = list.type(torch.long)

            index_1 = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2.shape[1])

            new_list[index_1, index_2] = -2

            index_3 = torch.arange(B_V, dtype=torch.long)[:, None].repeat(1, prob_size)

            new_list = new_list[index_3, new_sulution_rank]

            unselect_list = new_list[torch.gt(new_list, -1)].view(B_V, new_list_len)

            return unselect_list

        unselected_solution_index = _get_new_data_v2(solution, selected_solution_index, problems_size, batch_size)

        return solution, selected_solution_index, unselected_solution_index

    def check_legalilty(self,best_select_node_list,origin_problem_size):
        out_student = torch.unique(best_select_node_list[0])
        if len(out_student) != origin_problem_size:
            print(len(out_student),origin_problem_size)
            assert False, 'infeasible solution!'

    def _test_one_batch(self, episode, batch_size,clock=None):

        self.model.eval()
        self.model.mode='test'
        with torch.no_grad():

            if self.env.test_in_tsplib:
                self.env.load_problems_lib(episode, batch_size)
            else:

                self.env.load_problems(episode, batch_size)

            self.origin_problem = self.env.problems
            self.origin_problem_size = self.origin_problem.shape[1]
            self.origin_solution= self.env.solution

            reset_state, _, _ = self.env.reset(self.env_params['mode'])

            if self.env.test_in_tsplib:
                self.optimal_length, name = self.env._get_travel_distance_2(self.origin_problem, self.env.solution,need_optimal=True)
            else:
                self.optimal_length = self.env._get_travel_distance_2(self.origin_problem, self.env.solution)
                name = 'TSP_visual_1'+str(self.origin_problem.shape[1])

            B_V = batch_size * 1

            current_step = 0

            state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node

            IF_random_insertion = self.env_params['random_insertion']


            if IF_random_insertion:
                from utils_insertion.insertion import random_insertion

                dataset = self.origin_problem.clone().cpu().numpy()
                problem_size = dataset.shape[1]
                width = 1
                print('random insertion begin!')
                orders = [torch.randperm(problem_size) for i in range(width)]
                pi_all = [random_insertion(instance, orders[order_id])[0] for order_id in range(len(orders)) for
                          instance in
                          dataset]  # instance: (p
                pi_all = np.array(pi_all, dtype=np.int64)
                best_select_node_list = torch.tensor(pi_all)
            else:
                # from tqdm import tqdm
                # with tqdm(total=self.env.problem_size) as pbar:
                #     while not done:
                #         pbar.update(1)

                # from tqdm import tqdm
                # with tqdm(total=self.origin_problem_size) as pbar:
                while not done:
                    #pbar.update(1)
                    # print('  ')
                    # print('******************************************************************************')
                    # print(f'************************ current step {current_step} ************************')
                    # print('******************************************************************************')
                    if current_step == 0:

                        abs_scatter_solu_1 = torch.arange(start=1, end = self.origin_problem_size,
                                                                    dtype=torch.int64).unsqueeze(0).repeat(B_V,1)
                        abs_partial_solu_2 = torch.zeros(B_V,dtype=torch.int64).unsqueeze(1)
                        last_node_index = abs_partial_solu_2[:, [-1]]

                    else:

                        partial_end_node_coor = self.model.decoder._get_encoding(state.data,
                                                                                    last_node_index.reshape(batch_size, 1))
                        scatter_node_coors = self.model.decoder._get_encoding(state.data, self.env.abs_scatter_solu_1)

                        Manhattan_Distance = manhattan_distance(scatter_node_coors, partial_end_node_coor)

                        # print(Manhattan_Distance.shape)
                        random_index = torch.argmin(Manhattan_Distance, dim=1).reshape(batch_size, 1)  # [B]

                        abs_partial_solu_2, abs_scatter_solu_1, abs_scatter_solu_1_seleted = self.model( state.data,
                                                                                    self.env.solution,
                                                                                    self.env.abs_scatter_solu_1,
                                                                                    self.env.abs_partial_solu_2,
                                                                                    random_index,
                                                                                    current_step,
                                                                                    last_node_index)

                        last_node_index = abs_scatter_solu_1_seleted
                    current_step += 1

                    state, reward,reward_student, done = self.env.step(abs_scatter_solu_1,abs_partial_solu_2,mode='test')

                    # print('Get first complete solution!')


                best_select_node_list = self.env.abs_partial_solu_2
            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)
            escape_time, _ = clock.get_est_string(1, 1)

            gap = ((current_best_length - self.optimal_length) / self.optimal_length).mean().item() * 100
            self.logger.info("greedy, name:{}, gap:{:5f} %,  Elapsed[{}], stu_l:{:5f} , opt_l:{:5f}".format(
                name, gap, escape_time, current_best_length.mean().item(), self.optimal_length.mean().item()))


            # 检查解是否合法。
            # self.check_legalilty(best_select_node_list, self.origin_problem_size)

            ####################################################

            budget = self.env_params['RRC_budget']

            max_range = min(self.env_params['max_RRC_range'], self.origin_problem_size)

            length_fix = torch.randint(low=4, high=max_range, size=[budget])  # in [0,N)

            for bbbb in range(budget):

                curren_length_sub = length_fix[bbbb]
                # #  采样
                # self.env.load_problems(episode, batch_size)
                #
                # random inverse
                best_select_node_list = self.env.insvert_solution(best_select_node_list)
                # best_select_node_list = Re(best_select_node_list)

                # sample partial solution
                if self.env_params['mix_sample_strategy']:
                    mm = torch.randint(low=0, high=100, size=[1])[0].item()  # in [0,N)
                    if mm < 50:
                        abs_solution, abs_scatter_solu_1, abs_partial_solu_2 = self.sampling_subpaths_L2Insert(
                            self.origin_problem, best_select_node_list, curren_length_sub)
                    else:
                        abs_solution, abs_scatter_solu_1, abs_partial_solu_2 = self.sampling_subpaths_by_Proximity(
                            self.origin_problem, best_select_node_list, curren_length_sub)
                else:
                    if self.env_params['turn_to_cluster_strategy']:
                        abs_solution, abs_scatter_solu_1, abs_partial_solu_2 = self.sampling_subpaths_by_Proximity(
                            self.origin_problem, best_select_node_list, curren_length_sub)
                    else:
                        abs_solution, abs_scatter_solu_1, abs_partial_solu_2 = self.sampling_subpaths_L2Insert(
                        self.origin_problem, best_select_node_list, curren_length_sub)



                self.env.solution = abs_solution
                self.env.abs_scatter_solu_1 = abs_scatter_solu_1
                self.env.abs_partial_solu_2 = abs_partial_solu_2

                before_reward = self.env._get_travel_distance_2(self.origin_problem, abs_solution)

                current_step = 0

                self.env.problems = self.origin_problem.clone().detach()

                reset_state, _, _ = self.env.reset(self.env_params['mode'])

                state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node

                # mm = torch.randint(low=0, high=len(self.env.abs_partial_solu_2), size=[1])[0].item()  # in [0,N)
                # solution = torch.roll(solution, shifts=mm, dims=1)

                # last_node_index = self.env.abs_partial_solu_2[:, [mm]]
                last_node_index = self.env.abs_partial_solu_2[:, [-1]]
                while not done:

                    partial_end_node_coor = self.model.decoder._get_encoding(state.data, last_node_index)

                    scatter_node_coors = self.model.decoder._get_encoding(state.data, self.env.abs_scatter_solu_1)

                    Manhattan_Distance = manhattan_distance(scatter_node_coors, partial_end_node_coor)

                    random_index = torch.argmin(Manhattan_Distance, dim=1).reshape(batch_size, 1)  # [B]
                    # print(index.shape)

                    # random_index = torch.randint(low=0, high=len_1, size=[1])[0]  # in [0,N)

                    # print('******************************************************************************')
                    # print(f'************************ current step {current_step} ************************')
                    # print('******************************************************************************')

                    abs_partial_solu_2, abs_scatter_solu_1, abs_scatter_solu_1_seleted = self.model(state.data,
                                                                                                    self.env.solution,
                                                                                                    self.env.abs_scatter_solu_1,
                                                                                                    self.env.abs_partial_solu_2,
                                                                                                    random_index,
                                                                                                    current_step,
                                                                                                    last_node_index)
                    last_node_index = abs_scatter_solu_1_seleted

                    state, reward, reward_student, done = self.env.step(abs_scatter_solu_1, abs_partial_solu_2,
                                                                        mode='test')

                after_reward = self.env._get_travel_distance_2(self.origin_problem, self.env.abs_partial_solu_2)

                best_select_node_list = self.decide_whether_to_repair_solution( best_select_node_list,
                                                                                before_reward,
                                                                                self.env.abs_partial_solu_2,
                                                                                after_reward,
                                                                                    )
                current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)

                jjj = torch.arange(batch_size)

                # print(jjj[(current_best_length-self.optimal_length)>0.001])
                # tensor([27, 41, 45, 48, 52, 53, 56, 58, 59, 60, 68, 75, 76, 83, 90])
                # 检查解是否合法。

                # self.check_legalilty(best_select_node_list, self.origin_problem_size)
                # num_ins = 45
                # self.env.drawPic(self.origin_problem[num_ins], best_select_node_list[num_ins],
                #                  self.origin_problem_size,name=f'{num_ins}_TSP{self.origin_problem_size}step{bbbb}',
                #                  optimal_tour_=self.origin_solution[num_ins])

                escape_time, _ = clock.get_est_string(1, 1)
                gap = ((
                                   current_best_length - self.optimal_length) / self.optimal_length).mean().item() * 100
                self.logger.info("RRC step{}, name:{}, gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                    bbbb, name, gap, escape_time, current_best_length.mean().item(), self.optimal_length.mean().item()))

            # best_select_node_list = torch.load('LEHD_RRC_step1000_episode10000.pt')[episode:episode+batch_size]
            # torch.save(best_select_node_list, f'TSP{self.origin_problem_size}_RRC_step{budget}_{episode}_{episode + batch_size}.pt')

            self.check_legalilty(best_select_node_list, self.origin_problem_size)
            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)
            gap = ((current_best_length - self.optimal_length) / self.optimal_length).mean().item() * 100

            return self.optimal_length.mean().item(),current_best_length.mean().item(), self.origin_problem_size, gap

def manhattan_distance(x, y):
    # x: [B,seq,2]
    # y: [B, 1, 2]
    # difference = torch.abs(x - y).sum(2)
    difference = ((x - y) ** 2).sum(2).sqrt()

    return difference


