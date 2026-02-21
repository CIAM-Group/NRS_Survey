import vrplib
import numpy as np
import torch
import yaml
import json
import time
import os
from torch.optim import Adam as Optimizer

from CVRPModel import CVRPModel, CVRPModel_local
from CVRPEnv import CVRPEnv
from utils import rollout, check_feasible

# nohup python -u test_vrplib_survey.py > elg_cvrplib_survey.log 2>&1 &
class VRPLib_Tester:

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
        self.model = CVRPModel(**model_params)
        if model_params['ensemble']:
            self.model.decoder.add_local_policy(self.device)

        checkpoint = torch.load(load_checkpoint, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.vrplib_path = '/public/home/bayp/exp_survey_202509/0_data_survey/survey_bench_cvrp'
        self.repeat_times = 1
        self.aug_factor = 1 #config['params']['aug_factor']
        self.vrplib_results = None
        
    def test_on_vrplib(self):
        files = sorted(os.listdir(self.vrplib_path))
        vrplib_results = []
        total_time = 0.

        # Pre-read and sort instances by problem size
        instance_list = []
        print("Pre-reading instances to sort by size...")
        for name in files:
            if '.sol' in name:
                continue
            
            name_base = name[:-4]
            instance_file = self.vrplib_path + '/' + name_base + '.vrp'
            solution_file = self.vrplib_path + '/' + name_base + '.sol'
            
            if not os.path.exists(instance_file):
                continue

            try:
                # ! 注意我们这里将 compute_edge_weights 设为 False，因为环境中已经处理过了, 避免重复计算且防止内存不足被kill掉
                inst_data = vrplib.read_instance(instance_file,compute_edge_weights=False)
                p_size = inst_data['node_coord'].shape[0] - 1
                instance_list.append({
                    'name': name_base,
                    'instance_file': instance_file,
                    'solution_file': solution_file,
                    'problem_size': p_size
                })
            except Exception as e:
                print(f"Skipping {name}: {e}")

        # Sort
        instance_list.sort(key=lambda x: x['problem_size'])
        print(f"Sorted {len(instance_list)} instances.")
        print("instance list :")
        for item in instance_list:
            print(f"Instance: {item['name']}, Problem Size: {item['problem_size']}")
        print("Start testing on VRPLib dataset...")

        for t in range(self.repeat_times):
            for item in instance_list:
                name = item['name']
                instance_file = item['instance_file']
                solution_file = item['solution_file']
                
                solution = vrplib.read_solution(solution_file)
                optimal = solution['cost']

                result_dict = {}
                result_dict['run_idx'] = t
                
                # ! 如果OOM了，继续
                oom_happened = False

                try:
                    # —— 正常跑该实例 ——
                    self.test_on_one_ins(
                        name=name,
                        result_dict=result_dict,
                        instance=instance_file,
                        solution=solution_file
                    )
                    total_time += result_dict['runtime']

                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        oom_happened = True
                        print(f"[CUDA OOM] Instance {name} 显存不足，跳过该实例（不计入统计）。错误：{e}")
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                    else:
                        raise
                except torch.cuda.OutOfMemoryError as e:
                    oom_happened = True
                    print(f"[CUDA OOM] Instance {name} 显存不足，跳过该实例（不计入统计）。错误：{e}")
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

                # —— OOM：直接跳过记录与统计 —— 
                if oom_happened:
                    continue
                
                vrplib_results.append({
                    'instance': name,
                    'optimal': optimal,
                    'record': [result_dict]
                })
                
                print("-" * 60)   # 分隔线
                
        small_gaps, small_times = [], []
        mid_gaps,   mid_times   = [], []
        large_gaps, large_times = [], []
        all_gaps,   all_times   = [], []
        count_ok = 0
        print("-" * 60)
        print("-" * 60)
        print("Test on VRPLib dataset completed. Now summarizing results...")
        print("Start summarizing results...")
        for result in vrplib_results:
            rec = result['record'][-1]
            g = rec['gap']                # ratio
            s = int(rec['scale'])         # problem size
            rt = float(rec['runtime'])    # seconds

            # 归类
            if g >= 1.0:
                print(f"[Warning] Instance {result['instance']} has gap >= 100% ({g * 100:.2f}%), skip it in statistics.")
                continue
            if s < 1000:
                small_gaps.append(g); small_times.append(rt)
            elif s < 10000:
                mid_gaps.append(g);   mid_times.append(rt)
            else:  # >= 10000
                large_gaps.append(g); large_times.append(rt)

            all_gaps.append(g); all_times.append(rt)
            count_ok += 1
            print("already handled count: {}/{}".format(count_ok, len(vrplib_results)))

        def _pct_mean(arr):
            return "{:.2f}%".format(100 * np.mean(arr)) if len(arr) > 0 else "nan%"
        def _mean(values):
            return "{:.2f}s".format(np.mean(values)) if len(values) > 0 else "nan"


        print("-" * 60)
        print("Average gap [0, 1k): {} (count={})".format(
            _pct_mean(small_gaps), len(small_gaps)))
        print("Average time [0, 1k): {} (count={})".format(
            _mean(small_times), len(small_times)))
        print("-" * 60)

        print("-" * 60)
        print("Average gap [1k, 10k): {} (count={})".format(
            _pct_mean(mid_gaps), len(mid_gaps)))
        print("Average time [1k, 10k): {} (count={})".format(
            _mean(mid_times), len(mid_times)))
        print("-" * 60)

        print("-" * 60)
        print("Average gap [10k, +inf): {} (count={})".format(
            _pct_mean(large_gaps), len(large_gaps)))
        print("Average time [10k, +inf): {} (count={})".format(
            _mean(large_times), len(large_times)))
        print("-" * 60)

        print("-" * 60)
        print("Average gap total: {} (count={})".format(
            _pct_mean(all_gaps), len(all_gaps)))
        print("Average time total: {} (count={})".format(
            _mean(all_times), len(all_times)))
        print("-" * 60)
        print("Test finished. Total successfully completed instances: {}".format(count_ok))
        print("-" * 60)


    def test_on_one_ins(self, name, result_dict, instance, solution):
        start_time = time.time()
        instance = vrplib.read_instance(instance)
        solution = vrplib.read_solution(solution)
        optimal = solution['cost']
        problem_size = instance['node_coord'].shape[0] - 1
        multiple_width = min(problem_size, 1000)
        # multiple_width = problem_size

        # Initialize CVRP state
        env = CVRPEnv(multiple_width, self.device)
        env.load_vrplib_problem(instance, aug_factor=self.aug_factor)

        reset_state, reward, done = env.reset()
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.pre_forward(reset_state)

        with torch.no_grad():
            policy_solutions, policy_prob, rewards = rollout(self.model, env, 'greedy')
        # Return
        aug_reward = rewards.reshape(self.aug_factor, 1, env.multi_width)
        # shape: (augmentation, batch, multi)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_cost = -max_aug_pomo_reward.float()  # negative sign to make positive value

        best_cost = aug_cost
        end_time = time.time()

        elapsed_time = end_time - start_time
        if result_dict is not None:
            result_dict['best_cost'] = best_cost.cpu().numpy().tolist()[0]
            result_dict['scale'] = problem_size
            result_dict['gap'] = (result_dict['best_cost'] - optimal) / optimal
            result_dict['runtime'] = elapsed_time
            print(
                f"Instance {name}: Time {elapsed_time:.4f}s, "
                f"Cost {result_dict['best_cost']}, "
                f"Gap {result_dict['gap'] * 100:.2f}%"
            )

            # print(best_cost)


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    with open(config_path, 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)
    tester = VRPLib_Tester(config=config)
    tester.test_on_vrplib()