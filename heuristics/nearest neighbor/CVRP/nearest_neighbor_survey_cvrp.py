import os
import time
import random
import numpy as np

import logging
from logging import getLogger


#############################################
# 1. 复用/拷贝自 ICAM 的 CVRPLIBReader
#############################################

def CVRPLIBReader(filename):
    """
    Acquire description of a CVRP problem from a CVRPLIB-formatted file

    Returns:
        name: problem name
        dimension: number of customers (不含 depot)
        locs: coordinates of depot+customers, shape: (dimension+1, 2)
        demand: A list of node demands, length dimension+1 (含 depot)
        capacity: vehicle capacity
        cost: optimal cost read from .sol (如果找不到则为 None)
    """
    with open(filename, 'r') as f:
        dimension = 0
        started_node = False
        started_demand = False
        edge_weight_type = None
        locs = []
        demand = []
        for line in f:
            loc = []
            if started_demand:
                if line.startswith("DEPOT_SECTION"):
                    break
                demand.append(int(line.strip().split()[-1]))
            if started_node:
                if line.startswith("DEMAND_SECTION"):
                    started_node = False
                    started_demand = True
            if started_node:
                loc.append(float(line.strip().split()[1]))
                loc.append(float(line.strip().split()[2]))
                locs.append(loc)

            if line.startswith("NAME"):
                name = line.strip().split()[-1]
            if line.startswith("DIMENSION"):
                # DIMENSION 包含 depot，所以减 1
                dimension = float(line.strip().split()[-1]) - 1
            if line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.strip().split()[-1]
                if edge_weight_type not in ["EUC_2D", "CEIL_2D"]:
                    return None, None, None, None, None, None, None
            if line.startswith("CAPACITY"):
                capacity = float(line.strip().split()[-1])
            if line.startswith("NODE_COORD_SECTION"):
                started_node = True

    cost_file = filename.replace('.vrp', '.sol')
    if os.path.exists(cost_file):
        with open(cost_file, 'r') as f:
            for line in f:
                if line.startswith("Cost"):
                    cost = float(line.split()[1])
    else:
        cost = None

    assert len(locs) == dimension + 1  # +1 for depot
    return name, int(dimension), locs, demand, capacity, cost, edge_weight_type


#############################################
# 2. 最近邻求解 + 取整距离
#############################################

def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def calculate_distances_vectorized(cities, current_city, unvisited, edge_weight_type):
    """用于选择最近邻的距离计算，支持 TSPLIB 取整规则"""
    current_city_coords = cities[current_city]          # (2,)
    unvisited_cities_coords = cities[unvisited]         # (k, 2)
    
    diff = unvisited_cities_coords - current_city_coords
    d_raw = np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float64)

    if edge_weight_type == "CEIL_2D":
        d = np.ceil(d_raw)
    elif edge_weight_type == "EUC_2D":
        # TSPLIB standard: floor(d + 0.5)
        d = np.floor(d_raw + 0.5)
    else:
        d = d_raw

    return d


def nearest_neighbor_cvrp(cities, demands, vehicle_capacity, edge_weight_type):
    """
    最近邻构造 CVRP 路径（只负责生成 tour，不负责最终距离计算）

    cities: np.ndarray, shape (num_nodes, 2)，下标 0 为 depot
    demands: np.ndarray, shape (num_nodes, )，demands[0] = 0
    vehicle_capacity: float/int
    """
    num_cities = cities.shape[0]
    visited = np.zeros(num_cities, dtype=bool)
    current_city = 0
    tour = [current_city]
    visited[current_city] = True
    current_load = 0

    all_indices = np.arange(num_cities)

    while not np.all(visited):
        # 计算当前节点到所有所有节点的距离 (保持维度一致)
        if current_city != 0:
            visited[0] = False  # 允许回 depot
        distances = calculate_distances_vectorized(cities, current_city, all_indices, edge_weight_type)

        # 1. 屏蔽已访问节点
        distances[visited] = np.inf

        # 2. 屏蔽需求超过剩余容量的节点
        remaining_cap = vehicle_capacity - current_load
        distances[demands > remaining_cap] = np.inf

        # 找最近的有效节点
        nearest_city = np.argmin(distances)
        min_dist = distances[nearest_city]

        if min_dist == np.inf:
            # 无合法节点 (所有未访问节点的需求都大于剩余容量)
            if current_city != 0:
                # 回仓库，重置容量
                tour.append(0)
                current_city = 0
                current_load = 0
            else:
                # 已经在仓库但还没合适节点（例如单个需求 > 容量），无法生成合法解
                raise ValueError("Unvisited node demand exceeds vehicle capacity. Cannot generate feasible tour.")
        else:
            # 访问最近的合法节点
            tour.append(nearest_city)
            visited[nearest_city] = True
            current_load += demands[nearest_city]
            current_city = nearest_city
            
        if current_city == 0:
            current_load = 0  # 回 depot 后重置负载

    # 最后一条路回到仓库
    tour.append(0)
    return tour


def compute_tour_length(cities, tour, edge_weight_type):
    """
    cities: (num_nodes, 2)
    tour: list of node indices, e.g. [0, 5, 2, 0, 3, 1, 4, 0]
          允许多车，中间的 0 是 depot 分隔
    返回：total_distance，所有边长按 TSPLIB 规则取整再求和
    """
    tour = np.array(tour, dtype=int)
    coords = cities[tour]                      # (L, 2)
    rolled = np.roll(coords, shift=-1, axis=0)  # (L, 2)
    
    diff = coords - rolled
    d_raw = np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float64)

    if edge_weight_type == "CEIL_2D":
        d = np.ceil(d_raw)
    elif edge_weight_type == "EUC_2D":
        d = np.floor(d_raw + 0.5)
    else:
        d = d_raw
        
    total_distance = d.sum()
    return float(total_distance)


#############################################
# 3. Tester：模仿 ICAM 的 CVRPTester_LIB 风格
#############################################

class CVRPNNTester:
    def __init__(self, tester_params):
        self.tester_params = tester_params
        self.logger = getLogger('trainer')

        # gap 分桶
        self.gap_set_less_1000 = []
        self.gap_set_less_10000 = []
        self.gap_set_less_100000 = []

        # 时间分桶（累加单实例时间）
        self.time_sum_less_1000 = 0.0
        self.time_sum_less_10000 = 0.0
        self.time_sum_less_100000 = 0.0

        self.all_instance_num = 0
        self.all_solved_instance_num = 0

    def run_lib(self):
        filename = self.tester_params["filename"]
        start_time_all = time.time()

        scale_range_all = [[0, 1000], [1000, 10000], [10000, 100001]]

        for scale_range in scale_range_all:
            self.logger.info("#################  Test scale range: {0}  #################".format(scale_range))
            self._run_one_scale_range_lib(filename, scale_range)

        end_time_all = time.time()
        if self.all_solved_instance_num > 0:
            avg_time = (end_time_all - start_time_all) / self.all_solved_instance_num
        else:
            avg_time = 0.0

        # 全局总结
        self.logger.info(
            "All scale ranges done, solved instance number: {0}/{1}, total time: {2:.2f}s, avg time per instance: {3:.2f}s".
            format(self.all_solved_instance_num, self.all_instance_num,
                   end_time_all - start_time_all, avg_time)
        )

        # 各规模区间平均 gap（第一次输出）
        self.logger.info("[0, 1000), number: {0}, avg gap(no aug): {1:.3f}%".
                         format(len(self.gap_set_less_1000),
                                np.mean(self.gap_set_less_1000) if len(self.gap_set_less_1000) > 0 else 0))
        self.logger.info("[1000, 10000), number: {0}, avg gap(no aug): {1:.3f}%".
                         format(len(self.gap_set_less_10000),
                                np.mean(self.gap_set_less_10000) if len(self.gap_set_less_10000) > 0 else 0))
        self.logger.info("[10000, 100000], number: {0}, avg gap(no aug): {1:.3f}%".
                         format(len(self.gap_set_less_100000),
                                np.mean(self.gap_set_less_100000) if len(self.gap_set_less_100000) > 0 else 0))

        # ✅ 2. 最终再次输出各分桶的平均时间
        self.logger.info("#################  Bucket-wise Avg Time  #################")
        if len(self.gap_set_less_1000) > 0:
            self.logger.info("[0, 1000), number: {0}, avg time: {1:.3f}s".format(
                len(self.gap_set_less_1000), self.time_sum_less_1000 / len(self.gap_set_less_1000)
            ))
        if len(self.gap_set_less_10000) > 0:
            self.logger.info("[1000, 10000), number: {0}, avg time: {1:.3f}s".format(
                len(self.gap_set_less_10000), self.time_sum_less_10000 / len(self.gap_set_less_10000)
            ))
        if len(self.gap_set_less_100000) > 0:
            self.logger.info("[10000, 100000], number: {0}, avg time: {1:.3f}s".format(
                len(self.gap_set_less_100000), self.time_sum_less_100000 / len(self.gap_set_less_100000)
            ))

        # ✅ 3. 最终输出所有 instance 的平均 gap
        all_gaps = (
            self.gap_set_less_1000
            + self.gap_set_less_10000
            + self.gap_set_less_100000
        )
        if len(all_gaps) > 0:
            self.logger.info("#######################################################")
            self.logger.info("All instances, number: {0}, avg gap(no aug): {1:.3f}%".format(
                len(all_gaps), np.mean(all_gaps)
            ))
        else:
            self.logger.info("All instances, number: 0, avg gap(no aug): 0.000%")

        self.logger.info("#################  All Done  #################")

    def _run_one_scale_range_lib(self, filename, scale_range):
        num_sample = 0
        start_time_range = time.time()

        result_dict = {
            "instances": [],
            "optimal": [],
            "problem_size": [],
            "no_aug_score": [],
            "no_aug_gap": [],
        }

        for root, dirs, files in os.walk(filename):
            for file in files:
                if not file.endswith(".vrp"):
                    continue

                vrp_path = os.path.join(root, file)
                name, dimension, locs, demand, capacity, optimal, edge_weight_type = CVRPLIBReader(vrp_path)

                if name is None:
                    continue
                if not (scale_range[0] <= dimension < scale_range[1]):
                    continue
                if optimal is None:
                    self.logger.info(f"Instance {name}: .sol not found or cost missing, skip.")
                    continue

                num_sample += 1
                self.all_instance_num += 1

                # 数据转 numpy
                instance_xy = np.array(locs, dtype=np.float64)   # (dimension+1, 2)
                demands = np.array(demand, dtype=np.float64)     # (dimension+1, )

                self.logger.info("===============================================================")
                self.logger.info("Instance name: {0}, problem_size: {1}, edge_weight: {2}".format(name, dimension, edge_weight_type))

                # 计时：仅包含 NN 构造 + 距离计算
                inst_start = time.time()
                #try:
                tour = nearest_neighbor_cvrp(instance_xy, demands, capacity, edge_weight_type)
                score = compute_tour_length(instance_xy, tour, edge_weight_type)
                self.all_solved_instance_num += 1
                inst_time = time.time() - inst_start
                # except Exception as e:
                #     self.logger.info("Error occurred in instance {0}, dimension: {1}, skip it!".format(name, dimension))
                #     self.logger.info("Error message: {0}".format(e))
                #     continue

                # gap 计算
                no_aug_gap = (score - optimal) * 100.0 / optimal

                result_dict["instances"].append(name)
                result_dict["optimal"].append(optimal)
                result_dict["problem_size"].append(dimension)
                result_dict["no_aug_score"].append(score)
                result_dict["no_aug_gap"].append(no_aug_gap)

                # 放入不同规模桶，并累加时间
                if dimension < 1000:
                    self.gap_set_less_1000.append(no_aug_gap)
                    self.time_sum_less_1000 += inst_time
                elif 1000 <= dimension < 10000:
                    self.gap_set_less_10000.append(no_aug_gap)
                    self.time_sum_less_10000 += inst_time
                elif 10000 <= dimension <= 100000:
                    self.gap_set_less_100000.append(no_aug_gap)
                    self.time_sum_less_100000 += inst_time
                else:
                    raise ValueError("dimension should be less than 100000, but got {}".format(dimension))

                # 单实例日志
                self.logger.info("Instance name: {}, optimal score: {:.4f}".format(name, optimal))
                self.logger.info("No aug score:{:.3f}, No aug gap:{:.3f}%".format(score, no_aug_gap))
                self.logger.info(f"Instance time (NN only): {inst_time:.3f}s")

        end_time_range = time.time()
        during_range = end_time_range - start_time_range

        if num_sample > 0:
            self.logger.info(" *** Test Done *** ")
            self.logger.info(
                "scale_range: {0}, instance number: {1}, total time: {2:.2f}s, avg time per instance: {3:.2f}s".
                format(scale_range, num_sample, during_range, during_range / num_sample)
            )

            self.logger.info("instance: {0}".format(result_dict['instances']))
            self.logger.info("optimal: {0}".format(result_dict['optimal']))
            self.logger.info("problem_size: {0}".format(result_dict['problem_size']))
            self.logger.info("no_aug_score: {0}".format(result_dict['no_aug_score']))
            self.logger.info("no_aug_gap: {0}".format(result_dict['no_aug_gap']))

            avg_solved_no_aug_gap = np.mean(result_dict['no_aug_gap'])
            solved_instance_num = len(result_dict['instances'])
            max_dimension = max(result_dict['problem_size'])
            min_dimension = min(result_dict['problem_size'])
            self.logger.info(
                "Solved_ instances number: {0}, min_dimension: {1}, max_dimension: {2}, avg gap(no aug): {3:.3f}%".
                format(solved_instance_num, min_dimension, max_dimension, avg_solved_no_aug_gap)
            )
            self.logger.info("Avg time per instance: {0:.2f}s".format(during_range / solved_instance_num))
        else:
            self.logger.info(
                "scale_range: {0}, instance number: 0, total time: {1:.2f}s".format(scale_range, during_range)
            )
        self.logger.info("===============================================================")


#############################################
# 4. main：带“时间+随机数”的 log 目录
#############################################

if __name__ == "__main__":
    # ✅ 1. log 要带时间 + 随机数的文件夹
    time_str = time.strftime("%Y%m%d_%H%M%S")
    rand_str = f"{random.randint(0, 9999):04d}"
    log_dir = os.path.join("./CVRP/result_nn_cvrp", f"{time_str}_{rand_str}")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run_log.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    logger = getLogger('trainer')
    logger.info("===== Nearest Neighbor CVRP Tester (ICAM-style log) =====")
    logger.info(f"Log directory: {log_dir}")

    tester_params = {
        # 和 ICAM 一样的路径：根目录下面递归找 .vrp
        "filename": "/public/home/bayp/exp_survey_202509/0_data_survey/survey_bench_cvrp",
        # "filename": "/public/home/bayp/exp_survey_202509/0_data_survey/cvrp_test",
    }

    tester = CVRPNNTester(tester_params=tester_params)
    tester.run_lib()
