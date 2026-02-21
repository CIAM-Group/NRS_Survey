import os
import sys
import yaml
import time
import pickle
import json
import torch
import numpy as np
from torch.optim import Adam as Optimizer

from generate_data import generate_tsp_data, TSPDataset
from TSPModel import TSPModel, Att_Local_policy
from TSPEnv import TSPEnv
from utils import rollout, batched_two_opt_torch, check_feasible
from LIBUtils import TSPLIBReader, tsplib_cost


class TSPLib_Tester:

    def __init__(self, config):
        self.config = config
        model_params = config['model_params']
        load_checkpoint = config['load_checkpoint']
        load_checkpoint = os.path.join(os.path.dirname(__file__), load_checkpoint)
        print("Load checkpoint: {}".format(load_checkpoint))

        # cuda
        USE_CUDA = config['use_cuda']
        if USE_CUDA:
            cuda_device_num = config['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        
        # load trained model
        if config['training'] == 'joint':
            self.model = TSPModel(**model_params)
            if model_params['ensemble']:
                self.model.decoder.add_local_policy(self.device)
        elif config['training'] == 'only_local_att':
            self.model = Att_Local_policy(**model_params)

        checkpoint = torch.load(load_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tsplib_path = 'TSPLib'
        self.repeat_times = 1
        # ! augmentation factor set to 1 for survey
        self.aug_factor = 1 #config['params']['aug_factor']
        self.tsplib_results = None
        
    def run_lib(self):
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

        self.tsplib_path = '/public/home/bayp/exp_survey_202509/0_data_survey/survey_bench_tsp'
        print("Start testing on TSPLib instances...")
        print("Loading instances from {}".format(self.tsplib_path))
        start_time_all = time.time()

        # scale_range_all = [[0, 1000], [1000, 10000]]
        scale_range_all = [[0, 1000], [1000, 10000], [10000, 100001]]
        # scale_range_all = [[1000, 1001]]




        for scale_range in scale_range_all:
            print("#################  Test scale range: {0}  #################".format(scale_range))
            self._run_one_scale_range_lib(self.tsplib_path,scale_range)

        end_time_all = time.time()
        print("All scale ranges done, solved instance number: {0}/{1}, total time: {2:.2f}s, avg time per instance: {3:.2f}s".
                            format(self.all_solved_instance_num, self.all_instance_num,
                                end_time_all - start_time_all,
                                (end_time_all - start_time_all) / self.all_solved_instance_num))

        print("[0, 1000), number: {0}, avg gap(no aug): {1:.3f}%, avg gap(aug): {2:.3f}%".
                        format(len(self.gap_set_less_1000),
                                np.mean(self.gap_set_less_1000) if len(self.gap_set_less_1000) > 0 else 0,
                                np.mean(self.aug_gap_set_less_1000) if len(self.aug_gap_set_less_1000) > 0 else 0))
        print("[1000, 10000), number: {0}, avg gap(no aug): {1:.3f}%, avg gap(aug): {2:.3f}%".
                            format(len(self.gap_set_less_10000),
                                    np.mean(self.gap_set_less_10000) if len(self.gap_set_less_10000) > 0 else 0,
                                    np.mean(self.aug_gap_set_less_10000) if len(self.aug_gap_set_less_10000) > 0 else 0))
        print("[10000, 100000], number: {0}, avg gap(no aug): {1:.3f}%, avg gap(aug): {2:.3f}%".
                            format(len(self.gap_set_less_100000),
                                        np.mean(self.gap_set_less_100000) if len(self.gap_set_less_100000) > 0 else 0,
                                        np.mean(self.aug_gap_set_less_100000) if len(self.aug_gap_set_less_100000) > 0 else 0))
        print("#################  All Done  #################")
        print("All solved instances, number: {0}, avg gap(no aug): {1:.3f}%, avg gap(aug): {2:.3f}%".
                            format(len(self.gap_set_all_instances),
                                np.mean(self.gap_set_all_instances) if len(self.gap_set_all_instances) > 0 else 0,
                                np.mean(self.aug_gap_set_all_instances) if len(self.aug_gap_set_all_instances) > 0 else 0))

    def _run_one_scale_range_lib(self, tsplib_path, scale_range):
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
        
        instance_list = []
        for root, dirs, files in os.walk(tsplib_path):
            for file in files:
                if file.endswith(".tsp"):
                    full_path = os.path.join(root, file)
                    name, dimension, locs, ew_type = TSPLIBReader(full_path)
                    if name is None:
                        continue
                    if not (scale_range[0] <= dimension < scale_range[1]):
                        continue
                    instance_list.append({
                        'full_path': full_path,
                        'name': name,
                        'dimension': dimension,
                        'locs': locs,
                        'ew_type': ew_type
                    })
        
        instance_list.sort(key=lambda x: x['dimension'])

        #try:
        for instance in instance_list:
            full_path = instance['full_path']
            name = instance['name']
            dimension = instance['dimension']
            locs = instance['locs']
            ew_type = instance['ew_type']

            # ! check，打印当前处理的文件名，看是缺了哪个label
            print(f"**********当前读取的文件名: {full_path}**********")  # 推荐用 logger

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

            print("===============================================================")
            print("Instance name: {0}, problem_size: {1}".format(name, dimension))

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
                no_aug_gap = (score - optimal) * 100 / optimal
                aug_gap = (aug_score - optimal) * 100 / optimal
                if aug_gap >= 100.0:
                    print("Warning: Aug gap >=100% in instance {0}, dimension: {1}, gap: {2:.3f}%, skip it!".format(name, dimension, aug_gap))
                    continue
                self.all_solved_instance_num += 1
                print("current all solved instance num:", self.all_solved_instance_num)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                inst_time = time.time() - inst_start
            except Exception as e:
                print("Error occurred in instance {0}, dimension: {1}, skip it!".format(name, dimension))
                print("Error message: {0}".format(e))
                continue

            ############################
            # Logs
            ############################
            
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

            print("Instance name: {}, optimal score: {:.4f}".format(name, optimal))
            print("No aug score:{:.3f}, No aug gap:{:.3f}%".format(score, no_aug_gap))
            # ! 补充时间
            print(f"Instance time (incl. dist+test): {inst_time:.3f}s")
            sys.stdout.flush()

        end_time_range = time.time()
        during_range = end_time_range - start_time_range
        # Logs for all instances
        print(" *** Test Done *** ")
        print("scale_range: {0}, instance number: {1}, total time: {2:.2f}s, avg time per instance: {3:.2f}s".
                            format(scale_range, num_sample, during_range, during_range / num_sample))
        print("===============================================================")
        print("instance: {0}".format(result_dict['instances']))
        print("optimal: {0}".format(result_dict['optimal']))
        print("problem_size: {0}".format(result_dict['problem_size']))
        print("no_aug_score: {0}".format(result_dict['no_aug_score']))
        print("aug_score: {0}".format(result_dict['aug_score']))
        print("no_aug_gap: {0}".format(result_dict['no_aug_gap']))
        print("aug_gap: {0}".format(result_dict['aug_gap']))
        print("===============================================================")

        solved_instance_num = len(result_dict['instances'])
        if solved_instance_num > 0:
            avg_solved_no_aug_gap = np.mean(result_dict['no_aug_gap'])  # avg of all instances gap (no aug)
            avg_solved_aug_gap = np.mean(result_dict['aug_gap'])  # avg of all instances gap (aug)
            max_dimension = max(result_dict['problem_size'])
            min_dimension = min(result_dict['problem_size'])
            print("Solved_ instances number: {0}, min_dimension: {1}, max_dimension: {2}, avg gap(no aug): {3:.3f}%, avg_gap(aug): {4:.3f}%".
            format(solved_instance_num, min_dimension, max_dimension, avg_solved_no_aug_gap, avg_solved_aug_gap))
            print("Avg time per instance: {0:.2f}s".format(during_range / solved_instance_num))

        else:
            print("No instances were solved successfully.")

    

    def _test_one_batch_lib(self, batch_size,dict_instance_info):
        # unscaled_points = torch.tensor(instance[0], dtype=torch.float)[None, :, :]
        # points = (instance[0] - np.min(instance[0])) / (np.max(instance[0]) - np.min(instance[0]))
        # # points = instance[0] / np.max(instance[0])
        # test_batch = torch.tensor(points, dtype=torch.float)[None, :, :]
        # optimal = instance[1]

        problem_size = dict_instance_info['problem_size']
        unscaled_points = dict_instance_info['original_node_xy_lib'].to(self.device)
        test_batch = dict_instance_info['node_xy'].to(self.device)
        optimal = dict_instance_info['optimal']
        edge_weight_type = dict_instance_info['edge_weight_type']
        pomo_size = problem_size

        # initialize env
        env = TSPEnv(pomo_size, self.device)
        env.load_tsplib_problem(test_batch, unscaled_points, edge_weight_type,self.aug_factor)
        reset_state, reward, done = env.reset()

        self.model.eval()
        self.model.requires_grad_(False)
        self.model.pre_forward(reset_state)

        policy_solutions, policy_prob, rewards = rollout(self.model, env, 'greedy')

        aug_reward = rewards.reshape(self.aug_factor, 1, pomo_size)
        # shape: (augmentation, batch, pomo)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_cost = -max_aug_pomo_reward.float()  # negative sign to make positive value

        best_cost = aug_cost
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        # if result_dict is not None:
        #     result_dict['best_cost'] = best_cost.cpu().numpy().tolist()[0]
        #     result_dict['scale'] = problem_size
        #     result_dict['gap'] = (result_dict['best_cost'] - optimal) / optimal
        #     # print(best_cost)
        return no_aug_score.item(), aug_cost.mean().item()

# nohup python -u test_tsplib_survey.py > elg_tsplib_survey_output.log 2>&1 &
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    with open(config_path, 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)
    tester = TSPLib_Tester(config=config)
    tester.run_lib()