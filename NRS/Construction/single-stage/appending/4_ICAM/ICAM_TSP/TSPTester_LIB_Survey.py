import time

import numpy as np
import torch

import os
from logging import getLogger

from tqdm import tqdm

from TSPEnv import TSPEnv as Env
from TSPModel_ICAM import TSPModel as Model

from LIBUtils import TSPLIBReader, tsplib_cost

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

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        if "epoch" in model_load.keys():
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        else:
            checkpoint_fullname = '{path}/{name}.pt'.format(**model_load)

        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info("Model loaded successfully!!!")
        self.logger.info("Model loaded from: {}".format(checkpoint_fullname))

        total = sum([param.nelement() for param in self.model.parameters()])
        self.logger.info("Number of parameters: %.2fM" % (total / 1e6))

        # utility
        self.time_estimator = TimeEstimator()

    def run_lib(self):
        self.time_estimator.reset()



        self.gap_set_less_1000 = []
        self.gap_set_less_10000 = []
        self.gap_set_less_100000 = []

        self.aug_gap_set_less_1000 = []
        self.aug_gap_set_less_10000 = []
        self.aug_gap_set_less_100000 = []

        self.gap_set_all_instances = []
        self.aug_gap_set_all_instances = []

        self.all_instance_num = 0
        self.all_solved_instance_num = 0

        filename = self.tester_params["filename"]
        start_time_all = time.time()

        # scale_range_all = [[0, 1000], [1000, 10000]]
        scale_range_all = [[0, 1000], [1000, 10000], [10000, 100001]]
        # scale_range_all = [[1000, 1001]]
        # scale_range_all = [[70, 71]]




        for scale_range in scale_range_all:
            self.tester_params["scale_range"] = scale_range
            self.logger.info("#################  Test scale range: {0}  #################".format(scale_range))
            self._run_one_scale_range_lib(filename,scale_range)

        end_time_all = time.time()
        self.logger.info("All scale ranges done, solved instance number: {0}/{1}, total time: {2:.2f}s, avg time per instance: {3:.2f}s".
                            format(self.all_solved_instance_num, self.all_instance_num,
                                   end_time_all - start_time_all,
                                   (end_time_all - start_time_all) / self.all_solved_instance_num))

        self.logger.info("[0, 1000), number: {0}, avg gap(no aug): {1:.3f}%, avg gap(aug): {2:.3f}%".
                         format(len(self.gap_set_less_1000),
                                np.mean(self.gap_set_less_1000) if len(self.gap_set_less_1000) > 0 else 0,
                                np.mean(self.aug_gap_set_less_1000) if len(self.aug_gap_set_less_1000) > 0 else 0))
        self.logger.info("[1000, 10000), number: {0}, avg gap(no aug): {1:.3f}%, avg gap(aug): {2:.3f}%".
                            format(len(self.gap_set_less_10000),
                                      np.mean(self.gap_set_less_10000) if len(self.gap_set_less_10000) > 0 else 0,
                                      np.mean(self.aug_gap_set_less_10000) if len(self.aug_gap_set_less_10000) > 0 else 0))
        self.logger.info("[10000, 100000], number: {0}, avg gap(no aug): {1:.3f}%, avg gap(aug): {2:.3f}%".
                            format(len(self.gap_set_less_100000),
                                        np.mean(self.gap_set_less_100000) if len(self.gap_set_less_100000) > 0 else 0,
                                        np.mean(self.aug_gap_set_less_100000) if len(self.aug_gap_set_less_100000) > 0 else 0))
        self.logger.info("#################  All Done  #################")
        self.logger.info("All solved instances, number: {0}, avg gap(no aug): {1:.3f}%, avg gap(aug): {2:.3f}%".
                            format(len(self.gap_set_all_instances),
                                   np.mean(self.gap_set_all_instances) if len(self.gap_set_all_instances) > 0 else 0,
                                   np.mean(self.aug_gap_set_all_instances) if len(self.aug_gap_set_all_instances) > 0 else 0))




    def _run_one_scale_range_lib(self, filename, scale_range):
        num_sample = 0
        start_time_range = time.time()
        result_dict = {}
        result_dict["instances"] = []
        result_dict['optimal'] = []
        result_dict['problem_size'] = []
        result_dict['no_aug_score'] = []
        result_dict['aug_score'] = []
        result_dict['no_aug_gap'] = []
        result_dict['aug_gap'] = []
        for root, dirs, files in os.walk(filename):
            for file in files:
                if file.endswith(".tsp"):
                    name, dimension, locs, ew_type = TSPLIBReader(os.path.join(root, file))
                    if name is None:
                        continue
                    if not (scale_range[0] <= dimension < scale_range[1]):
                        continue

                    # ! check，打印当前处理的文件名，看是缺了哪个label
                    full_path = os.path.join(root, file)
                    self.logger.info(f"当前读取的文件名: {full_path}")  # 推荐用 logger

                    dict_instance_info = {}
                    optimal = float(tsplib_cost.get(name, None))
                    assert optimal is not None, "optimal value of instance {} not found".format(name)
                    instance_xy = np.array(locs).astype(np.float32)  # shape: (dimension,2)
                    node_coord = torch.from_numpy(instance_xy).unsqueeze(0)
                    # shape: (1,problem_size,2)
                    assert node_coord.shape == (1, dimension, 2), "dimension error in instance {}".format(name)

                    num_sample += 1 # 实际总实例个数,包含因为各种原因跳过的实例
                    self.all_instance_num += 1 # 全部实例个数,包含因为各种原因跳过的实例

                    dict_instance_info['optimal'] = optimal
                    dict_instance_info['problem_size'] = dimension
                    dict_instance_info['pomo_size'] = dimension
                    dict_instance_info['original_node_xy_lib'] = node_coord
                    # shape:(1,problem_size,2)
                    dict_instance_info['name'] = name
                    # ! 补充round/ceil
                    dict_instance_info['edge_weight_type'] = ew_type

                    self.logger.info("===============================================================")
                    self.logger.info("Instance name: {0}, problem_size: {1}".format(name, dimension))

                    # normalize data to [0,1] using min-max normalization
                    ################################################################
                    # max_value = np.max(instance_xy)
                    # min_value = np.min(instance_xy)
                    # nodes_xy_normalized = (node_coord - min_value) / (max_value - min_value)

                    xy_max = torch.max(node_coord, dim=1, keepdim=True).values
                    xy_min = torch.min(node_coord, dim=1, keepdim=True).values
                    # shape: (1, 1, 2)
                    ratio = torch.max((xy_max - xy_min), dim=-1, keepdim=True).values
                    ratio[ratio == 0] = 1
                    # shape: (1, 1, 1)
                    nodes_xy_normalized = (node_coord - xy_min) / ratio.expand(-1, 1, 2)
                    # shape: (1, dimension+1,2)

                    dict_instance_info["node_xy"] = nodes_xy_normalized


                    # ! 2025.10.08补充：统计单个实例时间
                    # === 计时开始（含矩阵计算与测试）===
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    inst_start = time.time()

                    # shape:(1,dimension,2)
                    try:
                        score, aug_score = self._test_one_batch_lib(batch_size=1, dict_instance_info=dict_instance_info)
                        self.all_solved_instance_num += 1

                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        inst_time = time.time() - inst_start
                    except Exception as e:
                        self.logger.info("Error occurred in instance {0}, dimension: {1}, skip it!".format(name, dimension))
                        self.logger.info("Error message: {0}".format(e))
                        continue

                    ############################
                    # Logs
                    ############################
                    no_aug_gap = (score - optimal) * 100 / optimal
                    aug_gap = (aug_score - optimal) * 100 / optimal
                    result_dict["instances"].append(name)
                    result_dict['optimal'].append(optimal)
                    result_dict['problem_size'].append(dimension)
                    result_dict['no_aug_score'].append(score)
                    result_dict['aug_score'].append(aug_score)
                    result_dict['no_aug_gap'].append(no_aug_gap)
                    result_dict['aug_gap'].append(aug_gap)

                    self.gap_set_all_instances.append(no_aug_gap)
                    self.aug_gap_set_all_instances.append(aug_gap)

                    if dimension < 1000:
                        self.gap_set_less_1000.append(no_aug_gap)
                        self.aug_gap_set_less_1000.append(aug_gap)
                    elif 1000 <= dimension < 10000:
                        self.gap_set_less_10000.append(no_aug_gap)
                        self.aug_gap_set_less_10000.append(aug_gap)
                    elif 10000 <= dimension <= 100000:
                        self.gap_set_less_100000.append(no_aug_gap)
                        self.aug_gap_set_less_100000.append(aug_gap)

                    self.logger.info("Instance name: {}, optimal score: {:.4f}".format(name, optimal))
                    self.logger.info("No aug score:{:.3f}, No aug gap:{:.3f}%".format(score, no_aug_gap))
                    # ! 补充时间
                    self.logger.info(f"Instance time (incl. dist+test): {inst_time:.3f}s")
                    self.logger.info("Aug score:{:.3f}, Aug gap:{:.3f}%".format(aug_score, aug_gap))

        end_time_range = time.time()
        during_range = end_time_range - start_time_range
        # Logs for all instances
        self.logger.info(" *** Test Done *** ")
        self.logger.info("scale_range: {0}, instance number: {1}, total time: {2:.2f}s, avg time per instance: {3:.2f}s".
                            format(scale_range, num_sample, during_range, during_range / num_sample))
        self.logger.info("===============================================================")
        if self.tester_params["detailed_log"]:
            self.logger.info("instance: {0}".format(result_dict['instances']))
            self.logger.info("optimal: {0}".format(result_dict['optimal']))
            self.logger.info("problem_size: {0}".format(result_dict['problem_size']))
            self.logger.info("no_aug_score: {0}".format(result_dict['no_aug_score']))
            self.logger.info("aug_score: {0}".format(result_dict['aug_score']))
            self.logger.info("no_aug_gap: {0}".format(result_dict['no_aug_gap']))
            self.logger.info("aug_gap: {0}".format(result_dict['aug_gap']))
            self.logger.info("===============================================================")

        self.logger.info("===============================================================")
        avg_solved_no_aug_gap = np.mean(result_dict['no_aug_gap'])  # avg of all instances gap (no aug)
        avg_solved_aug_gap = np.mean(result_dict['aug_gap'])  # avg of all instances gap (aug)
        solved_instance_num = len(result_dict['instances'])
        max_dimension = max(result_dict['problem_size'])
        min_dimension = min(result_dict['problem_size'])
        self.logger.info("Solved_ instances number: {0}, min_dimension: {1}, max_dimension: {2}, avg gap(no aug): {3:.3f}%, avg_gap(aug): {4:.3f}%".
            format(solved_instance_num, min_dimension, max_dimension, avg_solved_no_aug_gap, avg_solved_aug_gap))
        self.logger.info("Avg time per instance: {0:.2f}s".format(during_range / solved_instance_num))

    def _test_one_batch_lib(self, batch_size,dict_instance_info):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            problem_size = dict_instance_info['problem_size']
            self.env.load_problems_tsp(batch_size, problem_size,lib_data=dict_instance_info,
                                       aug_factor=aug_factor,device=self.device)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()
            # with tqdm(total=0) as pbar:
            while not done:
                cur_dist = self.env.get_local_feature()
                selected, prob = self.model(state, cur_dist)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(selected,lib_mode=True)
                # pbar.total += 1
                # pbar.update(1)

        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_aug_score.item(), aug_score.item()


