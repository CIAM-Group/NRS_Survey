import sys
import torch
import torch.nn as nn
import time
import argparse

import os
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from pathlib import Path

import warnings
from load_data import load_instances_with_baselines,use_saved_problems_tsp_txt
from utils.utils_for_model import run_aug, compute_vrp_tour_length, compute_tsp_tour_length
from utils.utilities import get_dist_matrix,calculate_tour_length_by_dist_matrix,normalize_nodes_to_unit_board,avg_list,load_tsplib_file,load_cvrplib_file,choose_bsz,check_cvrp_solution_validity,parse_tsplib_name,parse_cvrplib_name
warnings.filterwarnings("ignore", category=UserWarning)

from LIBUtils import *


def run_tsplib_test_knn(model,action_k,state_k,path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # root = Path(path)
    tsplib_path = "/public/home/bayp/exp_survey_202509/0_data_survey/survey_bench_tsp"
    # aug = 'mix'
    # main loop
    
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
    # scale_range_all = [[0, 100], [100, 200], [200, 300]]
    # scale_range_all = [[30000, 100001]]
    
    
    print("#################  Start Testing  #################")
    print("scale_range_all: {0}".format(scale_range_all))
    print("===============================================================")
    for scale_range in scale_range_all:
        print("#################  Test scale range: {0}  #################".format(scale_range))
        
        
        num_sample = 0
        start_time_range = time.time()
        result_dict = {}
        result_dict["instances"] = []
        result_dict['optimal'] = []
        result_dict['problem_size'] = []
        result_dict['score'] = []
        result_dict['gap'] = []
        
        # ! 单一scale range下的实例求解
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
        print("ordered instance list collected.")
        print("Total instances in this scale range: {0}".format(len(instance_list)))
        instance_list.sort(key=lambda x: x['dimension'])
        for instance in instance_list:
            full_path = instance['full_path']
            name = instance['name']
            dimension = instance['dimension']
            locs = instance['locs']
            ew_type = instance['ew_type']
            
            # ! check，打印当前处理的文件名，看是缺了哪个label
            print(f"**********当前读取的文件名: {full_path}**********")  # 推荐用 logger

            opt_len = float(tsplib_cost.get(name, None))
            assert opt_len is not None, "optimal value of instance {} not found".format(name)
            num_sample += 1 # 当前scale range下实际总实例个数,包含因为各种原因跳过的实例
            all_instance_num += 1 # 全部实例个数,包含因为各种原因跳过的实例
            
            print("===============================================================")
            print("Instance name: {0}, problem_size: {1}".format(name, dimension))
            try:
                # ! prepare env
                instance = torch.tensor(locs)  # shape: (dimension,2)
                assert instance.shape == (dimension, 2), "dimension error in instance {}".format(name)
                dist_matrix = get_dist_matrix(instance).to(device)
                
                # ! normalize instance for tsplib
                normalized_instance = normalize_nodes_to_unit_board(instance)
                size = normalized_instance.size(0)
                # ! 不使用aug
                # bsz = choose_bsz(size)
                bsz = 1
                normalized_instance = torch.tensor(normalized_instance).float().to(device)
                normalized_instance = normalized_instance.unsqueeze(0)
                normalized_instance = normalized_instance.repeat((bsz,1,1))
                # X = run_aug(aug,normalized_instance)
                X = normalized_instance
                
                # ! 2025.10.08补充：统计单个实例时间
                # === 计时开始（含矩阵计算与测试）===
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                inst_start = time.time()
                # ! model inference
            
                with torch.no_grad():
                    tour, _ = model(X, action_k, state_k, choice_deterministic=True)
                length_by_agent = compute_tsp_tour_length(normalized_instance,tour)
                idx = length_by_agent.min(dim=0).indices.item()
                best_tour = tour[idx,:]

                # evaluate tour length
                tour_len = calculate_tour_length_by_dist_matrix(dist_matrix, best_tour,ew_type).item()
                # ! math.ceil去掉,因为已经在calculate_tour_length_by_dist_matrix里处理过了
                # tour_len = math.ceil(tour_len)
                gap = (tour_len - opt_len) * 100 / opt_len
                if gap >= 100.0:
                    print("Warning: Gap >=100% in instance {0}, dimension: {1}, gap: {2:.3f}%, skip it!".format(name, dimension, gap))
                    continue
                all_solved_instance_num += 1
                print("current all solved instance num:", all_solved_instance_num)
                
                if device.type == 'cuda':
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
            result_dict['optimal'].append(opt_len)
            result_dict['problem_size'].append(dimension)
            result_dict['score'].append(tour_len)
            result_dict['gap'].append(gap)
            
            print("Already solved instance number: {0}/{1}, failed instances number: {2}".format(
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

            print("Instance name: {}, optimal score: {:.4f}".format(name, opt_len))
            print("score:{:.3f}, gap:{:.3f}%".format(tour_len, gap))
            # ! 补充时间
            print(f"Instance time: {inst_time:.3f}s")
            sys.stdout.flush() # 强制刷新输出缓冲区

            
        end_time_range = time.time()
        during_range = end_time_range - start_time_range
        # Logs for all instances
        print(" *** Test Done *** ")
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

def run_cvrplib_test_knn(model,action_k,state_k,path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # root = Path(path)
    problem_type='cvrp'
    cvrplib_path = "/public/home/bayp/exp_survey_202509/0_data_survey/survey_bench_cvrp"
    # aug = 'mix'
    # main loop
    
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
    # scale_range_all = [[0, 100], [100, 200], [200, 300]]
    
    print("#################  Start Testing  #################")
    print("scale_range_all: {0}".format(scale_range_all))
    print("===============================================================")
    for scale_range in scale_range_all:
        print("#################  Test scale range: {0}  #################".format(scale_range))
        
        
        num_sample = 0
        start_time_range = time.time()
        result_dict = {}
        result_dict["instances"] = []
        result_dict['optimal'] = []
        result_dict['problem_size'] = []
        result_dict['score'] = []
        result_dict['gap'] = []
        
        # ! 单一scale range下的实例求解
        instance_list = []
        for root, dirs, files in os.walk(cvrplib_path):
            for file_1 in files:
                if file_1.endswith(".vrp"):
                    name, dimension, locs, demand, capacity, optimal = CVRPLIBReader(
                        os.path.join(root, file_1)
                    )
                    if name is None:
                        continue
                    if not (scale_range[0] <= dimension < scale_range[1]):
                        continue
                    assert optimal is not None, "optimal value of instance {} not found".format(name)
                    instance_xy = torch.from_numpy(np.array(locs).astype(np.float32))
                    # shape: (problem_size+1,2)
                    assert instance_xy.shape == (dimension+1, 2), "dimension error in instance {}".format(name)
                    demand = torch.tensor(demand).float()  # shape: (problem_size+1,)
                    assert demand.shape == (dimension+1,), "demand dimension error in instance {}".format(name)
                    capacity = torch.tensor(capacity).float()  # scalar
                    instance_list.append({
                        'full_path': os.path.join(root, file_1),
                        'name': name,
                        'dimension': dimension,
                        'depot': instance_xy[0], # shape: (2,)
                        'nodes': instance_xy[1:], # shape: (dimension,2)
                        'demand': demand[1:], # ! 在VRP_net中做normalize, shape: (dimension,)
                        'capacity': capacity,
                        'optimal': optimal,
                        'ew_type': 'EUC_2D'  # CVRPLIB instances use EUC_2D distance
                    })
        
        print("ordered instance list collected.")
        print("Total instances in this scale range: {0}".format(len(instance_list)))
        instance_list.sort(key=lambda x: x['dimension'])
        for instance in instance_list:
            full_path = instance['full_path']
            name = instance['name']
            dimension = instance['dimension']
            depot = instance['depot'].to(device)
            nodes = instance['nodes'].to(device)
            demand = instance['demand'].to(device)
            capacity = instance['capacity']
            opt_len = instance['optimal']
            ew_type = instance['ew_type']
            
            # ! check，打印当前处理的文件名，看是缺了哪个label
            print(f"**********当前读取的文件名: {full_path}**********")  # 推荐用 logger

            assert opt_len is not None, "optimal value of instance {} not found".format(name)
            num_sample += 1 # 当前scale range下实际总实例个数,包含因为各种原因跳过的实例
            all_instance_num += 1 # 全部实例个数,包含因为各种原因跳过的实例
            
            print("===============================================================")
            print("Instance name: {0}, problem_size: {1}".format(name, dimension))
            
            try:
                # ! prepare env
                # ! 注意nodes在前depot在后
                depot_nodes = torch.cat((nodes, depot.unsqueeze(dim=0)), dim=0)  # shape: (dimension+1,2)
                assert depot_nodes.shape == (dimension+1, 2), "dimension error in instance {}".format(name)
                dist_matrix = get_dist_matrix(depot_nodes).to(device)
                
                # ! normalize instance for tsplib
                normalized_depot_nodes = normalize_nodes_to_unit_board(depot_nodes)
                size = nodes.size(0)
                # ! 不使用aug
                # bsz = choose_bsz(size)
                bsz = 1
                normalized_instance = torch.tensor(normalized_depot_nodes).float().to(device)
                normalized_instance = normalized_instance.unsqueeze(0)
                normalized_instance = normalized_instance.repeat((bsz,1,1))
                # X = run_aug(aug,normalized_instance)
                X = normalized_instance
                depot_aug = X[:,-1,:] 
                nodes_aug = X[:,0:-1,:] 
                demand_repeat = demand.unsqueeze(dim=0).repeat((bsz,1))
                input_aug = {'loc':nodes_aug,'demand':demand_repeat,'depot':depot_aug}
                
                # ! 2025.10.08补充：统计单个实例时间
                # === 计时开始（含矩阵计算与测试）===
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                inst_start = time.time()
                # ! model inference
            
                with torch.no_grad():
                    tour, _ = model(input_aug, action_k, state_k, capacity, problem_type, choice_deterministic=True)
                length_by_agent = compute_vrp_tour_length(normalized_instance,tour)
                idx = length_by_agent.min(dim=0).indices.item()
                best_tour = tour[idx,:]
                
                if not check_cvrp_solution_validity(best_tour + 1, demand_repeat[0], size, capacity):
                    print(f"Warning: Instance {name}: Failed to be solved!")
                    continue

                # evaluate tour length
                tour_len = calculate_tour_length_by_dist_matrix(dist_matrix, best_tour,ew_type).item()
                # ! math.ceil去掉,因为已经在calculate_tour_length_by_dist_matrix里处理过了
                # tour_len = math.ceil(tour_len)
                gap = (tour_len - opt_len) * 100 / opt_len
                if gap >= 100.0:
                    print("Warning: Gap >=100% in instance {0}, dimension: {1}, gap: {2:.3f}%, skip it!".format(name, dimension, gap))
                    continue
                all_solved_instance_num += 1
                print("current all solved instance num:", all_solved_instance_num)
                
                if device.type == 'cuda':
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
            result_dict['optimal'].append(opt_len)
            result_dict['problem_size'].append(dimension)
            result_dict['score'].append(tour_len)
            result_dict['gap'].append(gap)
            
            print("Already solved instance number: {0}/{1}, failed instances number: {2}".format(
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

            print("Instance name: {}, optimal score: {:.4f}".format(name, opt_len))
            print("score:{:.3f}, gap:{:.3f}%".format(tour_len, gap))
            # ! 补充时间
            print(f"Instance time: {inst_time:.3f}s")
            sys.stdout.flush() # 强制刷新输出缓冲区

            
        end_time_range = time.time()
        during_range = end_time_range - start_time_range
        # Logs for all instances
        print(" *** Test Done *** ")
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
    

def run_tsp_test_knn(local_k,global_k,aug,model,if_use_local_mask,sizes,bszs,data_path,device,
                     file,distributions,num_instance=None,if_aug=True,bs=None):
    problem_type = 'tsp'
    
    for distribution in distributions:
        for i in range(len(sizes)):
            start_time = time.time()
            # ! 添加batch size
            bs_problem = bs[i]
            print("if_aug: {0}, aug method: {1}, aug size: {2}".format(if_aug, aug, bszs[i]))
            file.write("if_aug: {0}, aug method: {1}, aug size: {2}\n".format(if_aug, aug, bszs[i]))
            print('bs_problem',bs_problem)
            file.write('Batch size for problem size {:d}: {:d}\n'.format(sizes[i], bs_problem))
            
            # tsp_instances, _, opt_lens = load_instances_with_baselines(data_path, problem_type, sizes[i], distribution)
            # ! 读取txt
            test_data_path = f'/public/home/bayp/exp_survey_202509/0_data_survey/survey_synthetic_tsp/test_tsp{sizes[i]}_n{num_instance[i]}_lkh.txt'

            tsp_instances, _, opt_lens = use_saved_problems_tsp_txt(test_data_path, device=device)
            opt_lens = torch.tensor(opt_lens).to(device)
            num = tsp_instances.size(0)
            problem_size = tsp_instances.size(1)
            assert problem_size == sizes[i], f'{problem_size} != {sizes[i]}'
            num = num_instance if isinstance(num_instance,int) else num_instance[i]
            assert num == num_instance[i], f'{num} != {num_instance[i]}'
            print("dataset size: {0}, problem size: {1}, num_instance to test: {2}".format(tsp_instances.size(0), problem_size, num))
            file.write("dataset size: {0}, problem size: {1}, num_instance to test: {2}".format(tsp_instances.size(0), problem_size, num)+'\n')
            
            # ! batch process
            all_ins_length = torch.zeros(size=(num,)).to(device)
            ##############################################################
            episode = 0
            while episode < num:
                remaining = num - episode
                batch_size = min(bs_problem, remaining)
                instance = tsp_instances[episode:episode+batch_size]
                # shape: [batch_size, num_nodes, 2]
                instance = torch.tensor(instance).float().to(device)
                # ! 注意更改为repeat_interleave, 格式(111122223333...)
                instance = instance.repeat_interleave(bszs[i], dim=0) # 
                # shape: [batch_size*bszs[i], num_nodes, 2]
                assert instance.size(0) == bszs[i]*batch_size, f'{instance.size(0)} != {bszs[i]*batch_size}'
                if if_aug:
                    X = run_aug(aug,instance,aug_num=bszs[i])
                else:
                    X = instance
                print("X size: {0}, aug used: {1}, aug method: {2}, aug size: {3}".format(
                    X.size(), if_aug, aug, bszs[i]))
                file.write("X size: {0}, aug used: {1}, aug method: {2}, aug size: {3}\n".format(
                    X.size(), if_aug, aug, bszs[i]))
                
                 # ! model inference
                with torch.no_grad():
                    tour, _ = model(X, local_k, global_k, choice_deterministic=True, if_use_local_mask=if_use_local_mask,if_aug=if_aug)                
                
                length_by_agent = compute_tsp_tour_length(instance,tour).view(batch_size, bszs[i]) # shape: [batch_size, bszs[i], 1]
                values, _ = length_by_agent.min(dim=1) # shape: [batch_size, ]
                assert values.shape == (batch_size,), f'{values.shape} != {(batch_size,)}'
                all_ins_length[episode:episode+batch_size] = values
                optimal_length_batch = opt_lens[episode:episode+batch_size]
                gap_batch = (values - optimal_length_batch) * 100 / optimal_length_batch
                
                episode += batch_size
                all_done = (episode == num)
                
                print( " problem size {:4d}, episode {:3d}/{:3d}, score:{:.3f}, gap:{:.3f}%".format(
                            problem_size, episode, num, values.mean().item(), gap_batch.mean().item()))
                file.write( " problem size {:4d}, episode {:3d}/{:3d}, score:{:.3f}, gap:{:.3f}%\n".format(
                            problem_size, episode, num, values.mean().item(), gap_batch.mean().item()))
                if all_done:
                    print(" All instances for problem size {:4d} done.".format(problem_size))
                    file.write(" All instances for problem size {:4d} done.\n".format(problem_size))
            ##############################################################
            
            
            gap_all = (all_ins_length - opt_lens) * 100 / opt_lens
            print( " >>>>> problem size {:4d}, all done, overall score:{:.3f}, overall gap:{:.3f}% <<<<<".format(
                        problem_size, all_ins_length.mean().item(), gap_all.mean().item()))
            file.write( " >>>>> problem size {:4d}, all done, overall score:{:.3f}, overall gap:{:.3f}% <<<<<\n".format(
                        problem_size, all_ins_length.mean().item(), gap_all.mean().item()))

                
            curr_time = time.time()
            elapsed_time = curr_time - start_time   
            avg_time = elapsed_time / num
            time_str = f" Total time: {elapsed_time} sec, {elapsed_time / 60:.3f} min, {elapsed_time / 3600:.3f} hr"
            out_string = 'For ' + distribution + '-tsp-{:d}, optimal: {:.4f}, model mean value: {:.4f}, gap: {:.3f}%'.format(
                sizes[i], opt_lens.mean().item(), all_ins_length.mean().item(),gap_all.mean().item()) # ! todo 修改
            
            print(out_string)
            
            file.write(out_string+'\n')
            
            print(time_str)
            
            print('==================================================')
            print('Time stats for problem size {:d}:'.format(problem_size))
            print('data info and results:')
            print('Number of instances: {:d}'.format(num))
            print('Optimal mean length: {:.4f}'.format(opt_lens.mean().item()))
            print('Model mean length: {:.4f}'.format(all_ins_length.mean().item()))
            print('Mean gap: {:.3f}%'.format(gap_all.mean().item()))
            print('Total time: {:.3f} sec, {:.3f} min, {:.3f} hr'.format(
                elapsed_time, elapsed_time / 60, elapsed_time / 3600))
            print('Average time per instance: {:.6f} sec, {:.6f} min, {:.6f} hr'.format(
                avg_time, avg_time / 60, avg_time / 3600))
            print('==================================================')
            
            # Logs
            file.write('==================================================\n')
            file.write('Time stats for problem size {:d}:\n'.format(problem_size))
            file.write('data info and results:\n')
            file.write('Number of instances: {:d}\n'.format(num))
            file.write('Optimal mean length: {:.4f}\n'.format(opt_lens.mean().item()))
            file.write('Model mean length: {:.4f}\n'.format(all_ins_length.mean().item()))
            file.write('Mean gap: {:.3f}%\n'.format(gap_all.mean().item()))
            file.write('Total time: {:.3f} sec, {:.3f} min, {:.3f} hr\n'.format(
                elapsed_time, elapsed_time / 60, elapsed_time / 3600))
            file.write('Average time per instance: {:.6f} sec, {:.6f} min, {:.6f} hr\n'.format(
                avg_time, avg_time / 60, avg_time / 3600))
            
            file.write('==================================================\n')


def run_vrp_test_knn(local_k,global_k,aug,model,if_use_local_mask,sizes,bszs,data_path,device,file,distributions,num_instance=None,if_aug=True):
    problem_type = 'cvrp'
    for distribution in distributions:
        for i in range(len(sizes)):
            cvrp_instances, _, opt_lens = load_instances_with_baselines(data_path, problem_type, sizes[i], distribution)
            depot, nodes, demands, capacity = cvrp_instances
            instances = torch.cat((nodes,depot.unsqueeze(dim=1)),dim=1).to(device)
            demands = demands.to(device)
            capacity = capacity.to(device)
            opt_lens = opt_lens.to(device)
            model_length = []
            num = num_instance if isinstance(num_instance,int) else num_instance[i]
            for j in range(num):
                instance = instances[j,:,:].to(device)
                instance = instance.unsqueeze(0)
                instance = instance.repeat((bszs[i],1,1))
                cap = capacity[j].item()
                if if_aug:
                    X = run_aug(aug,instance)
                else:
                    X = instance
                depot_aug = X[:,-1,:] 
                nodes_aug = X[:,0:-1,:] 
                demand_repeat = demands[j,:].unsqueeze(dim=0).repeat((bszs[i],1))
                input_aug = {'loc':nodes_aug,'demand':demand_repeat,'depot':depot_aug}
                with torch.no_grad():
                    tour, _ = model(input_aug, local_k, global_k, cap, problem_type, choice_deterministic=True, if_use_local_mask=if_use_local_mask)
                length_by_agent = compute_vrp_tour_length(instance,tour)
                value = length_by_agent.min(dim=0).values.item()
                #idx = length_by_agent.min(dim=0).indices.item()
                #best_tour = tour[idx,:]
                model_length.append(value)
                info = 'For '+distribution+'-'+problem_type+'-{:d} {:d}-th instance, gap is {:.3f}%.'.format(
                    sizes[i], j, 100*(value-opt_lens[j].item())/opt_lens[j].item()) 
                print(info)
            model_length = torch.tensor(model_length).to(device)
            gap = (model_length-opt_lens[0:num])/opt_lens[0:num]
            gap = torch.tensor(gap).to(device)
            out_string = 'For '+distribution+'-'+problem_type+'-{:d}, model provides solution with mean gap is {:.3f}%, min gap is {:.3f}%, max gap is {:.3f}%, std is {:.3f}.'.format(
                sizes[i], 100*gap.mean(dim=0).item(), 100*gap.min(dim=0).values.item(),100*gap.max(dim=0).values.item(),gap.std(dim=0).item())
            print(out_string)  
            file.write(out_string)
            file.write('\n')
