from logging import getLogger

import torch

from L2C_Insert.TSP.Train.TSPModel import TSPModel as Model
from L2C_Insert.TSP.Train.TSPEnv import TSPEnv as Env
from L2C_Insert.TSP.utils.utils import *
import random


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
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        torch.set_printoptions(precision=20)
        # utility
        self.time_estimator = TimeEstimator()
        self.time_estimator_2 = TimeEstimator()

    def run(self):
        self.time_estimator.reset()
        self.time_estimator_2.reset()

        if not self.env_params['test_in_tsplib']:
            self.env.load_raw_data(self.tester_params['test_episodes'])

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        aug_score_AM = AverageMeter()

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

            score_teacher, score_student, problems_size = self._test_one_batch(episode, batch_size,
                                                                               clock=self.time_estimator_2)
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

            print('problems_100 mean gap:', np.mean(problems_100), len(problems_100))
            print('problems_100_200 mean gap:', np.mean(problems_100_200), len(problems_100_200))
            print('problems_200_500 mean gap:', np.mean(problems_200_500), len(problems_200_500))
            print('problems_500_1000 mean gap:', np.mean(problems_500_1000), len(problems_500_1000))
            print('problems_1000 mean gap:', np.mean(problems_1000), len(problems_1000))
            score_AM.update(score_teacher, batch_size)
            score_student_AM.update(score_student, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info(
                "episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], Score_teacher:{:.4f},Score_studetnt: {:.4f},".format(
                    episode, test_num_episode, elapsed_time_str, remain_time_str, score_teacher, score_student, ))

            all_done = (episode == test_num_episode)

            if all_done:
                if not self.env_params['test_in_tsplib']:
                    self.logger.info(" *** Test Done *** ")
                    self.logger.info(" Teacher SCORE: {:.4f} ".format(score_AM.avg))
                    self.logger.info(" Student SCORE: {:.4f} ".format(score_student_AM.avg))
                    self.logger.info(" Gap: {:.4f}%".format((score_student_AM.avg - score_AM.avg) / score_AM.avg * 100))
                    gap_ = (score_student_AM.avg - score_AM.avg) / score_AM.avg * 100

                else:
                    self.logger.info(" *** Test Done *** ")
                    all_result_gaps = problems_1000 + problems_500_1000 + problems_200_500 + problems_100_200 + problems_100
                    average_gap = np.mean(all_result_gaps)
                    self.logger.info(" Average Gap: {:.4f}%".format(average_gap * 100))
                    gap_ = average_gap

        return score_AM.avg, score_student_AM.avg, gap_

    def decide_whether_to_repair_solution(self, after_repair_sub_solution, before_reward, after_reward,
                                          first_node_index, length_of_subpath, double_solution):

        the_whole_problem_size = int(double_solution.shape[1] / 2)

        other_part_1 = double_solution[:, :first_node_index]
        other_part_2 = double_solution[:, first_node_index + length_of_subpath:]
        origin_sub_solution = double_solution[:, first_node_index: first_node_index + length_of_subpath]

        jjj, _ = torch.sort(origin_sub_solution, dim=1, descending=False)

        index = torch.arange(jjj.shape[0])[:, None].repeat(1, jjj.shape[1])

        kkk_2 = jjj[index, after_repair_sub_solution]

        if_repair = before_reward > after_reward

        double_solution[if_repair] = torch.cat((other_part_1[if_repair],
                                                kkk_2[if_repair],
                                                other_part_2[if_repair]), dim=1)
        after_repair_complete_solution = double_solution[:, first_node_index:first_node_index + the_whole_problem_size]

        return after_repair_complete_solution

    def _test_one_batch(self, episode, batch_size, clock=None):

        self.model.eval()
        self.model.mode = 'test'
        with torch.no_grad():

            self.env.load_problems(episode, batch_size)
            self.origin_problem = self.env.problems
            self.origin_problem_size = self.origin_problem.shape[1]

            reset_state, _, _ = self.env.reset(self.env_params['mode'])

            if self.env.test_in_tsplib:
                self.optimal_length, name = self.env._get_travel_distance_2(self.origin_problem, self.env.solution,
                                                                            need_optimal=True)
            else:
                self.optimal_length = self.env._get_travel_distance_2(self.origin_problem, self.env.solution)
                name = 'TSP_visual_1' + str(self.origin_problem.shape[1])

            B_V = batch_size * 1

            current_step = 0

            state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node

            while not done:
                # print('  ')
                # print('******************************************************************************')
                # print(f'************************ current step {current_step} ************************')
                # print('******************************************************************************')
                if current_step == 0:

                    abs_scatter_solu_1 = torch.arange(start=1, end=self.origin_problem_size,
                                                      dtype=torch.int64).unsqueeze(0).repeat(B_V, 1)
                    abs_partial_solu_2 = torch.zeros(B_V, dtype=torch.int64).unsqueeze(1)
                    last_node_index = abs_partial_solu_2[:, [-1]]

                else:

                    partial_end_node_coor = self.model.decoder._get_encoding(state.data,
                                                                             last_node_index.reshape(batch_size, 1))
                    scatter_node_coors = self.model.decoder._get_encoding(state.data, self.env.abs_scatter_solu_1)

                    Manhattan_Distance = manhattan_distance(scatter_node_coors, partial_end_node_coor)

                    # print(Manhattan_Distance.shape)
                    random_index = torch.argmin(Manhattan_Distance, dim=1).reshape(batch_size, 1)  # [B]

                    abs_partial_solu_2, abs_scatter_solu_1, abs_scatter_solu_1_seleted = self.model(state.data,
                                                                                                    self.env.solution,
                                                                                                    self.env.abs_scatter_solu_1,
                                                                                                    self.env.abs_partial_solu_2,
                                                                                                    random_index,
                                                                                                    current_step,
                                                                                                    last_node_index)
                    last_node_index = abs_scatter_solu_1_seleted
                current_step += 1

                state, reward, reward_student, done = self.env.step(abs_scatter_solu_1, abs_partial_solu_2, mode='test')

            # print('Get first complete solution!')

            best_select_node_list = self.env.abs_partial_solu_2
            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)
            escape_time, _ = clock.get_est_string(1, 1)

            gap = ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100
            self.logger.info("greedy, name:{}, gap:{:5f} %,  Elapsed[{}], stu_l:{:5f} , opt_l:{:5f}".format(
                name, gap, escape_time, current_best_length.mean().item(), self.optimal_length.mean().item()))

            ####################################################

            out_student = torch.unique(best_select_node_list[0])

            print('selected_student', len(out_student), self.origin_problem_size)

            # assert False
            return self.optimal_length.mean().item(), current_best_length.mean().item(), self.origin_problem_size


def manhattan_distance(x, y):
    # x: [B,seq,2]
    # y: [B, 1, 2]
    # difference = torch.abs(x - y).sum(2)
    difference = torch.norm(x - y, p=2, dim=-1)

    return difference
