import os
import time
from datetime import datetime
import pytz

import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


# 生成随机二维城市坐标
def generate_random_cities(num_cities):
    return np.random.rand(num_cities, 2) * 100  # 随机生成在100x100范围内的城市坐标

def calculate_path_length(cities, tour):
    coords = cities[tour]
    diffs = coords[1:] - coords[:-1]
    distances = np.linalg.norm(diffs, axis=1)
    distances = np.append(distances, np.linalg.norm(coords[-1] - coords[0]))
    return np.sum(distances)

def use_saved_problems_tsp_txt(filename, total_episodes, start=0):
    nodes_coords = []
    opt_costs = []

    with open(filename, "r") as f:
        lines = f.readlines()[start:start+total_episodes]

    for line in lines:
        line = line.strip().split(" ")
        if 'output' not in line:
            continue
        split_idx = line.index('output')
        
        # Coords
        coord_data = line[:split_idx]
        num_nodes = len(coord_data) // 2
        coords = [[float(coord_data[idx]), float(coord_data[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]
        nodes_coords.append(coords)
        
        # Tour (Optimal)
        tour_data = line[split_idx+1:]
        tour = [int(x) for x in tour_data if x]
        if len(tour) > 0 and min(tour) == 1:
            tour = [x - 1 for x in tour]
            
        # Calculate Opt Cost
        opt_cost = calculate_path_length(np.array(coords), tour)
        opt_costs.append(opt_cost)

    problems = np.array(nodes_coords, dtype=np.float64)
    return problems, opt_costs


# 计算当前城市到其他未访问城市的距离，使用向量化加速
def calculate_distances_vectorized(cities, current_city, unvisited):
    # 提取当前城市坐标
    current_city_coords = cities[current_city]

    # 提取所有未访问城市的坐标
    unvisited_cities_coords = cities[unvisited]

    # 使用广播和向量化计算欧几里得距离
    distances = np.linalg.norm(unvisited_cities_coords - current_city_coords, axis=1)

    return distances


# Nearest Neighbor算法求解TSP，使用动态计算且忽略已访问的节点，使用NumPy向量化
def nearest_neighbor_tsp(cities):
    num_cities = cities.shape[0]
    visited = np.zeros(num_cities, dtype=bool)  # 记录节点是否被访问过
    current_city = 0
    tour = [current_city]
    visited[current_city] = True
    total_distance = 0

    # tqdm用于显示单个实例的进度

    for _ in range(num_cities - 1):
        # 找到所有未访问的城市
        unvisited = np.where(~visited)[0]

        # 使用向量化计算距离
        distances = calculate_distances_vectorized(cities, current_city, unvisited)

        # 找到距离最近的城市
        nearest_city_idx = np.argmin(distances)
        nearest_city = unvisited[nearest_city_idx]
        min_distance = distances[nearest_city_idx]

        # 更新路径
        tour.append(nearest_city)
        visited[nearest_city] = True  # 标记为已访问
        total_distance += min_distance
        current_city = nearest_city  # 移动到下一个城市

    # 返回到起始城市
    total_distance += np.linalg.norm(cities[current_city] - cities[tour[0]])
    tour.append(tour[0])

    return tour, total_distance


# 运行多个TSP实例的函数
def run_tsp_instances(problems, num_instances):

    problem_size = problems.shape[1]
    cities_list = [problems[i] for i in range(num_instances)]

    # 使用ProcessPoolExecutor并行运行多个TSP实例
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(nearest_neighbor_tsp, cities_list), total=num_instances, desc="Computing"))

    tour_list = [r[0] for r in results]
    distance_list = [r[1] for r in results]

    for i in range(num_instances):
        unique_node_list_len = len(torch.unique(torch.tensor(tour_list[i])))
        assert unique_node_list_len == problem_size, \
            f"nn process error, unique_node_list_len:{unique_node_list_len}, problem_size:{problem_size}"

    return tour_list, distance_list


if __name__ == '__main__':
    # Setup Logging
    tz = pytz.timezone("Asia/Shanghai")
    process_start_time = datetime.now(tz)
    log_dir = os.path.join(
        ".", "TSP/result_survey_tsp_nn_synthetic",
        process_start_time.strftime("%Y%m%d_%H%M%S") + "_NN_SYNTHETIC"
    )
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run_log.txt")
    
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    data_dict = {
        100: 'test_tsp100_n10000_lkh.txt',
        1000: 'test_tsp1000_n128_lkh.txt',
        10000: 'test_tsp10000_n16_lkh.txt'
    }

    for problem_size in [100, 1000, 10000]:
        start_time = time.time()
        log(f"Processing problem size: {problem_size}")

        data_path = f'/public/home/bayp/exp_survey_202509/0_data_survey/survey_synthetic_tsp/{data_dict[problem_size]}'
        
        if problem_size == 100:
            test_episodes = 10000
        elif problem_size == 1000:
            test_episodes = 128
        elif problem_size == 10000:
            test_episodes = 16
        else:
            test_episodes = 16

        problems, opt_costs = use_saved_problems_tsp_txt(data_path, test_episodes, start=0)

        log(f"Please wait, computing {test_episodes} TSP{problem_size} instances...")
        tour_list, distance_list = run_tsp_instances(problems, test_episodes)

        # Calculate gaps and log per instance
        gaps = []
        for i in range(test_episodes):
            nn_cost = distance_list[i]
            opt_cost = opt_costs[i]
            gap = (nn_cost - opt_cost) / opt_cost * 100
            gaps.append(gap)
            
            log(f"[{i+1}/{test_episodes}] Instance: TSP{problem_size}_{i}, dim: {problem_size}, BKS: {opt_cost:.4f}, NN cost: {nn_cost:.4f}, GAP: {gap:.4f}%")

        result_dict = {
            "tours": tour_list,
            "distances": distance_list,
            "opt_costs": opt_costs,
            "gaps": gaps
        }
        end_time = time.time()
        avg_gap = np.mean(gaps)
        avg_time = (end_time - start_time) / test_episodes
        
        log(f"TSP{problem_size}, mean distance:{np.mean(distance_list)}")
        log(f"TSP{problem_size}, mean gap:{avg_gap:.4f}%")
        log(f"TSP{problem_size} computation finished, time elapsed: {end_time - start_time:.2f} s")
        log(f"TSP{problem_size}, avg time per instance: {avg_time:.4f} s")
        
        # # Ensure directory exists for saving results
        # os.makedirs('NN/tsp', exist_ok=True)
        # torch.save(result_dict, f'NN/tsp/nearest_neighbor_TSP{problem_size}_results.pt')
