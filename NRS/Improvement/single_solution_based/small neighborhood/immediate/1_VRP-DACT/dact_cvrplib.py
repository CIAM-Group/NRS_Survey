import os
import json
import sys
import time
import torch
import pprint
import numpy as np
from tensorboard_logger import Logger as TbLogger
import warnings
import random
from options import get_options

from problems.problem_tsp import TSP
from problems.problem_vrp import CVRP
from agent.ppo import PPO
from torch.utils.data import DataLoader

def load_agent(name):
    agent = {
        'ppo': PPO,
    }.get(name, None)
    assert agent is not None, "Currently unsupported agent: {}!".format(name)
    return agent

def load_problem(name):
    problem = {
        'tsp': TSP,
        'vrp': CVRP,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def run(opts):
    start_time = time.time()
    # Pretty print the run args
    pprint.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tb and not opts.distributed:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, 
                                                          opts.graph_size), opts.run_name))
    if not opts.no_saving and not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
        
    # Save arguments so exact configuration can always be found
    if not opts.no_saving:
        with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
    
    
    # Figure out what's the problem
    problem = load_problem(opts.problem)(
                            p_size = opts.graph_size,
                            step_method = opts.step_method,
                            init_val_met = opts.init_val_met,
                            with_assert = opts.use_assert,
                            P = opts.P,
                            DUMMY_RATE = opts.dummy_rate,
                            lib_model = opts.lib_model) # ! 修改的代码：设置lib_model为True
    
    # Figure out the RL algorithm
    agent = load_agent(opts.RL_agent)(problem.NAME, problem.size,  opts)

    # Load data from load_path
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        agent.load(load_path)
    
    # ! 修改的代码
    # Do validation only
    # Load the validation datasets
    #agent.start_inference(problem, opts.val_dataset, tb_logger)
    print("Evaluation with library benchmark datasets.")
    
    # ! 遍历求解每个实例
    gap_set_less_1000 = []
    gap_set_less_10000 = []
    gap_set_less_100000 = []
    
    gap_set_all_instances = []

    all_instance_num = 0
    all_solved_instance_num = 0

    start_time_all = time.time()

    scale_range_all = [[0, 1000], [1000, 10000], [10000, 100001]]

    
    for scale_range in scale_range_all:
        opts.scale_range = scale_range
        print("#################  Test scale range: {0}  #################".format(opts.scale_range))
        # run_one_scale_range_lib(data_file,scale_range)
    
        # Validate mode
        opts = agent.opts
        
        print(f'\nInference with x{opts.val_m} augments...', flush=True)
        
        num_sample = 0
        start_time_range = time.time()
        result_dict = {}
        result_dict["instances"] = []
        result_dict['optimal'] = []
        result_dict['problem_size'] = []
        result_dict['score'] = []
        result_dict['gap'] = []
        
        agent.eval()
        problem.eval()
        
        if opts.eval_only:
            torch.manual_seed(opts.seed)
            np.random.seed(opts.seed)
            random.seed(opts.seed)
        
        val_dataset = problem.make_dataset(size=opts.graph_size,
                                num_samples=opts.val_size,
                                filename = opts.val_dataset,
                                DUMMY_RATE = opts.dummy_rate,
                                lib_model = opts.lib_model,
                                scale_range = opts.scale_range,
                                device = opts.device)

        
        val_dataloader = DataLoader(val_dataset, batch_size=opts.eval_batch_size, shuffle=False,
                                    num_workers=0,
                                    pin_memory=True)
        
        for batch_id, batch in enumerate(val_dataloader):
            
            num_sample += 1 # 实际总实例个数,包含因为各种原因跳过的实例
            all_instance_num += 1 # 全部实例个数,包含因为各种原因跳过的实例

            inst_start = time.time()
            current_start_batch_idx = batch_id * opts.eval_batch_size
            name = batch.get('name', None)[0]
            dimension = batch.get('dimension').item()
            print(f"===================当前读取的文件名: {name}===================")  
            print("===============================================================")
            print("Instance name: {0}, problem_size: {1}".format(name, dimension))
            try:
            
                bv_, cost_hist_, best_hist_, r_, rec_history_ = agent.rollout(problem,
                                                                            opts.val_m,
                                                                            batch,
                                                                            do_sample = True,
                                                                            record = False,
                                                                            show_bar = True,
                                                                            current_start_batch_idx = current_start_batch_idx)
                score = bv_[0].item()
                assert batch.get('optimal_score') is not None, "Optimal score is required for validation!"
                optimal_score = batch.get('optimal_score')[0].item()
                
                gap = (score - optimal_score) * 100 / optimal_score
                if gap >= 100.0:
                    print("Warning: Gap >=100% in instance {0}, dimension: {1}, gap: {2:.3f}%, skip it!".format(name, dimension, gap))
                    continue
                
                all_solved_instance_num += 1
                inst_time = time.time() - inst_start
                
                
            except Exception as e:
                print("Error occurred in instance {0}, dimension: {1}, skip it!".format(name, dimension))
                print("Error message: {0}".format(e))
                continue
            
            ############################
            # Logs
            ############################
            
            result_dict["instances"].append(name)
            result_dict['optimal'].append(optimal_score)
            result_dict['problem_size'].append(dimension)
            result_dict['score'].append(score)
            result_dict['gap'].append(gap)
            
            print("model_name: {0}, already solved instance number: {1}/{2}, failed instances number: {3}".format(
                opts.model_name,
                len(result_dict['instances']),
                num_sample,
                num_sample - len(result_dict['instances'])
            ))
            
            gap_set_all_instances.append(gap)

            if dimension < 1000:
                gap_set_less_1000.append(gap)
            elif 1000 <= dimension < 10000:
                gap_set_less_10000.append(gap)
            elif 10000 <= dimension <= 100000:
                gap_set_less_100000.append(gap)

            print("Instance name: {}, optimal score: {:.4f}".format(name, optimal_score))
            print("score:{:.3f}, gap:{:.3f}%".format(score, gap))
            # ! 补充时间
            print(f"Instance time: {inst_time:.3f}s")
            sys.stdout.flush() # 强制刷新输出缓冲区
            
        end_time_range = time.time()
        during_range = end_time_range - start_time_range
        # Logs for all instances
        print(" *** Test Done *** ")
        print("model name: {0}".format(opts.model_name))
        print("scale_range: {0}, instance number: {1}, total time: {2:.2f}s, avg time per instance (including failed instances): {3:.2f}s".
                            format(scale_range, num_sample, during_range, during_range / num_sample if num_sample >0 else 0))
        print("===============================================================")
        print("instance: {0}".format(result_dict['instances']))
        print("optimal: {0}".format(result_dict['optimal']))
        print("problem_size: {0}".format(result_dict['problem_size']))
        print("score: {0}".format(result_dict['score']))
        print("gap: {0}".format(result_dict['gap']))
        print("===============================================================")

        print("===============================================================")
        if len(result_dict['instances']) == 0:
            print("No instance solved successfully in this scale range!")
            continue
        avg_solved_gap = np.mean(result_dict['gap'])  # avg of all instances gap
        solved_instance_num = len(result_dict['instances'])
        max_dimension = max(result_dict['problem_size'])
        min_dimension = min(result_dict['problem_size'])
        print("Solved instances number: {0}/{1}, min_dimension: {2}, max_dimension: {3}, avg gap: {4:.3f}%".
            format(solved_instance_num, num_sample, min_dimension, max_dimension, avg_solved_gap))
        print("Avg time per instance (excluding failed instances): {0:.2f}s".format(during_range / solved_instance_num))
    
    ###########################

    end_time_all = time.time()
    print("model name: {0}".format(opts.model_name))
    print("All scale ranges done, solved instance number: {0}/{1}, total time: {2:.2f}s, avg time per instance: {3:.2f}s".
                        format(all_solved_instance_num, all_instance_num,
                                end_time_all - start_time_all,
                                (end_time_all - start_time_all) / all_solved_instance_num if all_solved_instance_num >0 else 0))

    print("[0, 1000), number: {0}, avg gap: {1:.3f}%".
                        format(len(gap_set_less_1000),
                            np.mean(gap_set_less_1000) if len(gap_set_less_1000) > 0 else 0))
    print("[1000, 10000), number: {0}, avg gap: {1:.3f}%".
                        format(len(gap_set_less_10000),
                                    np.mean(gap_set_less_10000) if len(gap_set_less_10000) > 0 else 0))
    print("[10000, 100000], number: {0}, avg gap: {1:.3f}%".
                        format(len(gap_set_less_100000),
                                    np.mean(gap_set_less_100000) if len(gap_set_less_100000) > 0 else 0))
    print("#################  All Done  #################")
    print("All solved instances, number: {0}, avg gap: {1:.3f}%".
                        format(len(gap_set_all_instances),
                                np.mean(gap_set_all_instances) if len(gap_set_all_instances) > 0 else 0))

    
    
                


if __name__ == "__main__":
    # nohup python -u dact_cvrplib.py > dact_survey_cvrplib_aug1_T1K.log 2>&1 &
    warnings.filterwarnings("ignore")
    
    model_list = ["cvrp100-epoch-190.pt", "cvrp50-epoch-175.pt", "cvrp20-epoch-198.pt"]
    for model_name in model_list:
        
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        opts = get_options()
        opts.model_name = model_name  # ! 修改的代码
        opts.problem = 'vrp'  # ! 修改的代码
        opts.T_max = 1000  # ! 修改的代码
        current_path = "/public/home/bayp/exp_survey_202509/Improvement/single_solution_based/small neighborhood/immediate/1_VRP-DACT/pretrained"
        opts.load_path = os.path.join(current_path, model_name)
        print("*"*100)
        print("Loading model from: {}".format(opts.load_path))
        print("*"*100)
        
        # figure out whether to use distributed training if needed
        opts.world_size = torch.cuda.device_count()
        opts.distributed = False #(opts.world_size > 1) and (not opts.no_DDP) # we disable DDP for now
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '4869'
        # processing settings
        opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
        
        opts.eval_only = True
        opts.no_saving = True
        opts.no_tb = True
        
        opts.val_size = 10000 # useless for tsplib
        opts.val_dataset = '/public/home/bayp/exp_survey_202509/0_data_survey/survey_bench_cvrp'
        opts.eval_batch_size = 1
        opts.lib_model = True
        opts.val_m = 1
        # opts.scale_range = [0,100]
        opts.init_val_met = 'greedy'
        opts.dummy_rate = 0.5
        opts.P = 20 if opts.eval_only else 1e10 # can set to smaller values e.g., 20 or 10, for generalization 
        opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S")) \
            if not opts.resume else opts.resume.split('/')[-2]
        opts.save_dir = os.path.join(
            opts.output_dir,
            "{}_{}".format(opts.problem, opts.graph_size),
            opts.run_name
        ) if not opts.no_saving else None
        
        # print options
        print("\n[Options]")
        for k, v in sorted(vars(opts).items()):
            print(f'  {k}: {v}')
        
        model_start_time = time.time()
        run(opts)
        model_end_time = time.time()
        print("Model {0} total time used: {1:.2f} seconds".format(model_name, model_end_time - model_start_time))
        print("Model {0} total time used: {1:.2f} mins".format(model_name, (model_end_time - model_start_time)/60))
        print("Model {0} total time used: {1:.2f} hours".format(model_name, (model_end_time - model_start_time)/3600))