import vrplib
import numpy as np
import torch
import yaml
import json
import time
import os
from CVRPModel import CVRPModel
from CVRPEnv import CVRPEnv
# ! inport的rollout和本地方法重名，没用，去掉
# from utils import rollout, check_feasible
from utils import check_feasible
import random

import logging
import datetime
import pytz

# nohup python -u test_cvrplib_survey.py > reld_vrplib_test_survey.log 2>&1 &
def rollout(model, env, eval_type='greedy'):
    env.reset()
    actions = []
    probs = []
    reward = None
    state, reward, done = env.pre_step()

    while not done:
        cur_dist = env.get_cur_feature()
        selected, one_step_prob = model.one_step_rollout(state, cur_dist, eval_type=eval_type)
        # selected, one_step_prob = model(state)
        state, reward, done = env.step(selected)
        actions.append(selected)
        probs.append(one_step_prob)

    actions = torch.stack(actions, 1)
    if eval_type == 'greedy':
        probs = None
    else:
        probs = torch.stack(probs, 1)

    return torch.transpose(actions, 1, 2), probs, reward


class VRPLib_Tester:

    def __init__(self, config):
        self.config = config
        model_params = config['model_params']
        load_checkpoint = config['load_checkpoint']
        self.multiple_width = config['test_params']['pomo_size']

        # ! === 日志设置 ===
        base_dir = os.path.join(os.getcwd(), "/public/home/bayp/exp_survey_202509/Construction/single-stage/appending/9_ReLD/CVRP/Results_survey")
        os.makedirs(base_dir, exist_ok=True)

        # 获取上海时区当前时间
        shanghai_tz = pytz.timezone("Asia/Shanghai")
        now = datetime.datetime.now(shanghai_tz)

        # 格式化：日期+时间（精确到秒）
        date_time_str = now.strftime("%Y%m%d_%H%M%S")
        rand_str = f"{random.randint(0, 99):02d}"

        # 子文件夹名
        sub_dir_name = f"results_survey_cvrp_{date_time_str}_{rand_str}"

        log_dir = os.path.join(base_dir, sub_dir_name)
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "vrplib_test.log")

        self.logger = logging.getLogger("VRPLibTester")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            fmt = logging.Formatter("%(asctime)s - %(message)s")

            # 控制台输出
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            self.logger.addHandler(ch)

            # 文件输出
            fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

        self.log_dir = log_dir  # 存下来，方便之后保存结果

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
        checkpoint = torch.load(load_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # ! 改成自己的benchmark地址
        # self.vrplib_path = 'data/VRPLib/Vrp-Set-X' if config['vrplib_set'] == 'X' else "data/VRPLib/Vrp-Set-XXL"
        self.vrplib_path = '/public/home/bayp/exp_survey_202509/0_data_survey/survey_bench_cvrp'
        self.repeat_times = 1
        self.aug_factor = 1 #config['test_params']['aug_factor']
        self.logger.info("AUG_FACTOR: ", self.aug_factor)
        self.vrplib_results = None
        
    def test_on_vrplib(self):
        files = sorted(os.listdir(self.vrplib_path))
        vrplib_results = []
        total_time = 0.

        for t in range(self.repeat_times):
            for name in files:
                if '.sol' in name:
                    continue

                name = name[:-4]
                instance_file = self.vrplib_path + '/' + name + '.vrp'
                solution_file = self.vrplib_path + '/' + name + '.sol'

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
                        self.logger.info(f"[CUDA OOM] Instance {name} 显存不足，跳过该实例（不计入统计）。错误：{e}")
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                    else:
                        raise
                except torch.cuda.OutOfMemoryError as e:
                    oom_happened = True
                    self.logger.info(f"[CUDA OOM] Instance {name} 显存不足，跳过该实例（不计入统计）。错误：{e}")
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

                # —— OOM：直接跳过记录与统计 —— 
                if oom_happened:
                    continue



                # self.test_on_one_ins(name=name, result_dict=result_dict, instance=instance_file, solution=solution_file)
                # total_time += result_dict['runtime']  # Update total run time

                # —— 记录本实例（仅成功的才进入统计池） ——
                vrplib_results.append({
                    'instance': name,
                    'optimal': optimal,
                    'record': [result_dict]
                })

                # new_instance_dict = {}
                # new_instance_dict['instance'] = name
                # new_instance_dict['optimal'] = optimal
                # new_instance_dict['record'] = [result_dict]
                # vrplib_results.append(new_instance_dict)

                # self.logger.info("Instance Name {}: gap {:.2f}%".format(name, result_dict['gap'] * 100))
                self.logger.info("-" * 60)   # 分隔线
                # if 'XXL' in self.vrplib_path:
                #     print("cost: {}".format(result_dict['best_cost']))
        # if 'XXL' in self.vrplib_path:
        #     avg_gap = []
        #     for result in vrplib_results:
        #         avg_gap.append(result['record'][-1]['gap'])
            
        #     print("{:.2f}%".format(100 * np.array(avg_gap).mean()))
        #     print("Average time: {:.2f}s".format(total_time / 4))
        # else:
            
        # —— 统计（此时 vrplib_results 里已无 OOM 实例） ——
        small_gaps, small_times = [], []
        mid_gaps,   mid_times   = [], []
        large_gaps, large_times = [], []
        all_gaps,   all_times   = [], []
        count_ok = 0


        # avg_gap_small = []
        # avg_gap_medium = []
        # avg_gap_large = []
        # total = []
        # number = 0
        # for result in vrplib_results:
        #     scale = int(result['record'][-1]['scale'])
        #     if scale <= 200:
        #         avg_gap_small.append(result['record'][-1]['gap'])
        #     elif scale <= 500:
        #         avg_gap_medium.append(result['record'][-1]['gap'])
        #     else:
        #         avg_gap_large.append(result['record'][-1]['gap'])
        #     total.append(result['record'][-1]['gap'])
        #     number += 1
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
            print("already handled count: {}".format(count_ok))

        def _pct_mean(arr):
            return "{:.2f}%".format(100 * np.mean(arr)) if len(arr) > 0 else "nan%"
        def _mean(values):
            return "{:.2f}s".format(np.mean(values)) if len(values) > 0 else "nan"


        self.logger.info("-" * 60)
        self.logger.info("Average gap [0, 1k): {} (count={})".format(
            _pct_mean(small_gaps), len(small_gaps)))
        self.logger.info("Average time [0, 1k): {} (count={})".format(
            _mean(small_times), len(small_times)))
        self.logger.info("-" * 60)

        self.logger.info("-" * 60)
        self.logger.info("Average gap [1k, 10k): {} (count={})".format(
            _pct_mean(mid_gaps), len(mid_gaps)))
        self.logger.info("Average time [1k, 10k): {} (count={})".format(
            _mean(mid_times), len(mid_times)))
        self.logger.info("-" * 60)

        self.logger.info("-" * 60)
        self.logger.info("Average gap [10k, +inf): {} (count={})".format(
            _pct_mean(large_gaps), len(large_gaps)))
        self.logger.info("Average time [10k, +inf): {} (count={})".format(
            _mean(large_times), len(large_times)))
        self.logger.info("-" * 60)

        self.logger.info("-" * 60)
        self.logger.info("Average gap total: {} (count={})".format(
            _pct_mean(all_gaps), len(all_gaps)))
        self.logger.info("Average time total: {} (count={})".format(
            _mean(all_times), len(all_times)))
        self.logger.info("-" * 60)
        self.logger.info("Test finished. Total successfully completed instances: {}".format(count_ok))
        self.logger.info("-" * 60)

        # print("Average gap [1, 200]: {:.2f}%".format(100 *(np.array(avg_gap_small).mean())))
        # print("Average gap (200, 500]: {:.2f}%".format(100 *(np.array(avg_gap_medium).mean())))
        # print("Average gap (500, 1000]: {:.2f}%".format(100 *(np.array(avg_gap_large).mean())))
        # print("Average gap total: {:.2f}%".format(100 *(np.array(total).mean())))
        # print("Average time: {:.2f}s".format(total_time / number))


    def test_on_one_ins(self, name, result_dict, instance, solution):
        start_time = time.time()
        instance = vrplib.read_instance(instance)
        solution = vrplib.read_solution(solution)
        optimal = solution['cost']
        problem_size = instance['node_coord'].shape[0] - 1

        multiple_width = min(problem_size, self.multiple_width)
        # multiple_width = problem_size

        # Initialize CVRP state
        env = CVRPEnv(self.multiple_width, self.device)

        aug_reward = None
        sep_augmentation = False
        if sep_augmentation:
            # compute only one augmented version each time to save gpu memory, repeat 8 times for each instance
            for idx in range(8):
                env.load_vrplib_problem(instance, aug_factor=self.aug_factor, aug_idx=idx)

                reset_state, reward, done = env.reset()
                self.model.eval()
                self.model.requires_grad_(False)
                self.model.pre_forward(reset_state)

                with torch.no_grad():
                    policy_solutions, policy_prob, rewards = rollout(self.model, env, 'greedy')
                # Return
                
                if aug_reward is not None:
                    aug_reward = rewards.reshape(self.aug_factor, 1, env.multi_width)
                    # shape: (augmentation, batch, multi)
                else:
                    aug_reward = rewards.reshape(1, 1, env.multi_width)

        else:
            env.load_vrplib_problem(instance, aug_factor=self.aug_factor, aug_idx=-1)

            reset_state, reward, done = env.reset()
            self.model.eval()
            self.model.requires_grad_(False)
            self.model.pre_forward(reset_state)

            with torch.no_grad():
                policy_solutions, policy_prob, rewards = rollout(self.model, env, 'greedy')

            aug_reward = rewards.reshape(self.aug_factor, 1, env.multi_width)
            # shape: (augmentation, batch, multi)    
        

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
            self.logger.info(
                f"Instance {name}: Time {elapsed_time:.4f}s, "
                f"Cost {result_dict['best_cost']}, "
                f"Gap {result_dict['gap'] * 100:.2f}%"
            )



if __name__ == "__main__":
    with open('/public/home/bayp/exp_survey_202509/Construction/single-stage/appending/9_ReLD/CVRP/config.yml', 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)
    tester = VRPLib_Tester(config=config)
    tester.test_on_vrplib()