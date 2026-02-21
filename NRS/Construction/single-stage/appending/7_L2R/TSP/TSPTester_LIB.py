import time

import numpy as np
import torch

import os
from logging import getLogger

from tqdm import tqdm

from TSPEnv import TSPEnv as Env
from TSPModel_Upper import TSPUpperModel as UpperModel
from TSPModel_Lower import TSPLowerModel as LowerModel
from utils.utils import *


class TSPTester:
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

        self.env_params['device'] = device
        self.model_params['device'] = device
        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.upper_model = UpperModel(**self.model_params)
        self.lower_model = LowerModel(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        if model_load['epoch'] == 'best':
            checkpoint_fullname = '{path}/best_model.pt'.format(**model_load)
        else:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.upper_model.load_state_dict(checkpoint['upper_model_state_dict'])
        self.lower_model.load_state_dict(checkpoint['lower_model_state_dict'])
        total_params = list(self.upper_model.parameters()) + list(self.lower_model.parameters())

        total = sum([param.nelement() for param in total_params])
        self.logger.info("Number of parameters: %.2fM" % (total / 1e6))

        # utility
        self.time_estimator = TimeEstimator()

    def run_tsplib(self):
        self.time_estimator.reset()

        import ast
        result_dict = {}
        result_dict["instances"] = []
        result_dict['optimal'] = []
        result_dict['problem_size'] = []
        result_dict['score'] = []
        result_dict['gap'] = []

        gap_set_less_5k = []
        gap_set_gt_5k = []

        tsplib_path = self.env_params["tsplib_path"]
        start_time = time.time()

        with open(tsplib_path, 'r') as f:
            for line in f.readlines():
                dict_instance_info = {}
                instance = line.strip()  # use strip to remove the '\n' at the end of each line
                instance_list = ast.literal_eval(instance)  # use ast.literal_eval to convert string to list

                name = instance_list[0]  # name of the instance
                optimal = float(instance_list[1])  # optimal value of the instance
                instance_xy = np.array(instance_list[2:]).astype(np.float32)  # shape: (dimension*2,)

                node_coord = torch.from_numpy(instance_xy).reshape(-1, 2).unsqueeze(0)
                # shape: (1,dimension,2)
                dimension = node_coord.size(1)  # node number of the instance
                assert instance_xy.shape[0] == dimension * 2, "dimension error in instance:{}".format(name)

                dict_instance_info['optimal'] = optimal
                dict_instance_info['problem_size'] = dimension

                self.logger.info("===============================================================")
                self.logger.info("Instance name: {0}, problem_size: {1}".format(name, dimension))

                # shape:(1,dimension,2)
                dict_instance_info['original_node_xy_lib'] = node_coord
                # shape:(1,problem_size)
                dict_instance_info['name'] = name

                if self.env_params['lib_norm'] == 'unified_norm':
                    max_value = np.max(instance_xy)
                    min_value = np.min(instance_xy)
                    nodes_xy_normalized = (node_coord - min_value) / (max_value - min_value)
                    nodes_xy_normalized = nodes_xy_normalized.reshape(1, -1, 2)
                    # shape:(1,dimension+1,2)
                elif self.env_params['lib_norm'] == 'separate_norm':
                    min_x = torch.min(node_coord[:, :, 0], dim=-1)[0][:, None]
                    min_y = torch.min(node_coord[:, :, 1], dim=-1)[0][:, None]
                    max_x = torch.max(node_coord[:, :, 0], dim=-1)[0][:, None]
                    max_y = torch.max(node_coord[:, :, 1], dim=-1)[0][:, None]
                    # shape:(1,1)
                    scaled_depot_node_x = (node_coord[:, :, 0] - min_x) / (max_x - min_x)  # shape:(1,dimension+1)
                    scaled_depot_node_y = (node_coord[:, :, 1] - min_y) / (max_y - min_y)  # shape:(1,dimension+1)

                    nodes_xy_normalized = torch.cat((scaled_depot_node_x[:, :, None], scaled_depot_node_y[:, :, None]),dim=2)
                    # shape:(1,dimension+1,2)
                elif self.env_params['lib_norm'] == 'invit_norm':
                    xy_max = torch.max(node_coord, dim=1, keepdim=True).values
                    xy_min = torch.min(node_coord, dim=1, keepdim=True).values
                    # shape: (1, 1, 2)
                    ratio = torch.max((xy_max - xy_min), dim=-1, keepdim=True).values
                    ratio[ratio == 0] = 1
                    # shape: (1, 1, 1)
                    nodes_xy_normalized = (node_coord - xy_min) / ratio.expand(-1, 1, 2)
                    # shape: (1, dimension+1,2)
                else:
                    raise NotImplementedError

                dict_instance_info["node_xy"] = nodes_xy_normalized # shape: (1, dimension, 2)

                score,during_time = self._test_one_batch(batch_size=1,dict_instance_info=dict_instance_info)

                ############################
                # Logs
                ############################
                gap = (score - optimal) * 100 / optimal
                result_dict["instances"].append(name)
                result_dict['optimal'].append(optimal)
                result_dict['problem_size'].append(dimension)
                result_dict['score'].append(score)
                result_dict['gap'].append(gap)

                if dimension <= 5000:
                    gap_set_less_5k.append(gap)
                else:
                    gap_set_gt_5k.append(gap)

                self.logger.info("Instance name: {}, optimal score: {:.4f}".format(name, optimal))
                self.logger.info("score:{:.3f}, gap:{:.3f}%".format(score, gap))
                self.logger.info("During time: {0:.2f}s, {1:.2f}min".format(during_time, during_time / 60))

        end_time = time.time()

        self.logger.info("===============================================================")
        self.logger.info("instance: {0}".format(result_dict['instances']))
        self.logger.info("optimal: {0}".format(result_dict['optimal']))
        self.logger.info("problem_size: {0}".format(result_dict['problem_size']))
        self.logger.info("score: {0}".format(result_dict['score']))
        self.logger.info("gap: {0}".format(result_dict['gap']))
        self.logger.info("===============================================================")
        self.logger.info("===============================================================")

        self.logger.info("size <=5000, number: {0}, avg_gap: {1:.3f}%".
                         format(len(gap_set_less_5k), np.mean(gap_set_less_5k)))
        self.logger.info("size 5000~100000, number: {0}, avg_gap: {1:.3f}%".
                            format(len(gap_set_gt_5k), np.mean(gap_set_gt_5k)))

        self.logger.info("===============================================================")
        self.logger.info("===============================================================")
        avg_all_gap = np.mean(result_dict['gap'])
        all_instance_num = len(result_dict['instances'])
        max_dimension = max(result_dict['problem_size'])
        min_dimension = min(result_dict['problem_size'])
        self.logger.info("Coordinates normalization: {0}".format(self.env_params['lib_norm']))
        self.logger.info("All instances number: {0}, min_dimension: {1}, max_dimension: {2}, avg_gap: {3:.3f}%".
                         format(all_instance_num, min_dimension, max_dimension, avg_all_gap))
        self.logger.info("avg time per instance: {0:.2f}s".format((end_time - start_time) / all_instance_num))


    def _test_one_batch(self, batch_size,dict_instance_info):

        problem_size = dict_instance_info['problem_size']
        # Ready
        ###############################################
        self.upper_model.eval()
        self.lower_model.eval()
        self.upper_model.set_decoder_method('greedy')
        self.lower_model.set_decoder_method('greedy')

        with torch.no_grad():
            self.env.load_problems_tsp(batch_size, problem_size,lib_data=dict_instance_info,device=self.device)
            reset_state, _, _ = self.env.reset()
            self.upper_model.pre_forward(reset_state)

            # AM Rollout
            ###############################################
            state, reward, done = self.env.pre_step()
            start_time = time.time()
            with tqdm(total=0) as pbar:
                while not done:
                    if state.current_node is not None:
                        state = self.env.get_upper_input()
                        upper_scores, _, _ = self.upper_model(state)
                        self.env.update_cur_scores(upper_scores=upper_scores)
                        # upper_score.shape: (batch, unvisited_num)
                    state = self.env.get_lower_transformed_neighbors()
                    low_selected, _ = self.lower_model(state)
                    # shape: (batch,)
                    state, reward, done = self.env.step(low_selected,lib_mode=True)
                    # shape: (batch,)
                    pbar.total += 1
                    pbar.update(1)

        # Return
        ###############################################
        end_time = time.time()
        during_time = end_time - start_time
        avg_score = -reward.float().mean()  # negative sign to make positive value
        #if self.env_params['draw_pic']:
        #    self.env.drawPic_TSP(self.env.original_node_xy_lib[0], self.env.selected_node_list[0], optimal=False,name=dict_instance_info['name'])

        return avg_score.item(), during_time
