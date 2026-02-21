"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import argparse
import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
SCALE = 1e7


def create_problem_file(instance_num, nodes, demands, capacity, working_dir):
    with open(os.path.join(working_dir, str(instance_num) + ".vrp"), "w") as file:
        file.write("NAME : " + str(instance_num) + "\n")
        file.write("COMMENT : generated instance No. " + str(instance_num) + "\n")
        file.write("TYPE : CVRP\n")
        file.write("DIMENSION : " + str(len(nodes)) + "\n")
        file.write("EDGE_WEIGHT_TYPE : EUC_2D \n")
        file.write("CAPACITY : " + str(capacity) + " \n")
        file.write("NODE_COORD_SECTION\n")

        for i, node in enumerate(nodes):
            file.write(" " + str(i+1) + " " + str(int(node[0] * SCALE)) + " " + str(int(node[1] * SCALE)) + "\n")
        file.write("DEMAND_SECTION\n")
        for i, demand in enumerate(demands):
            file.write(str(i+1) + " " + str(demand) + "\n")
        file.write("DEPOT_SECTION \n 1 \n -1 \nEOF ")
        file.close()


def create_parameter_file(instance_num, working_dir, num_runs, time_limit):
    with open(os.path.join(working_dir, str(instance_num) + ".par"), "w") as file:
        file.write("PROBLEM_FILE = " + os.path.join(working_dir, str(instance_num) + ".vrp\n"))
        file.write("RUNS = " + str(num_runs) + "\n")
        if time_limit > 0:
            file.write("TIME_LIMIT = " + str(time_limit) + "\n")
        file.write("TOUR_FILE = " + os.path.join(working_dir, str(instance_num) + ".sol\n"))


def read_solution_file(instance_num, working_dir, num_nodes):
    with open(os.path.join(working_dir, str(instance_num) + ".sol"), "r") as file:
        lines = file.readlines()
        tours = list()
        for node in lines[6:-2]:
            tours.append(int(node))
        tours.append(1)
        tours = np.array(tours)
        tours = tours - 1
        tours[tours > num_nodes] = 0
    return tours


def reorder(coordinates, demands, capacity, all_tours):
    tours, subtour = list(), list()

    for node_idx in all_tours[1:]:
        if node_idx == 0:
            tours.append(subtour)
            subtour = list()
        else:
            subtour.append(node_idx)

    reformated_tour, remaining_capacities = list(), list()
    distances, capacities = list(), list()
    for tour in tours:
        tour_capacity = capacity
        for node_idx in tour:
            tour_capacity -= demands[node_idx]
        capacities.append(tour_capacity)

    tour_idxs = np.argsort(capacities)

    for num_tour in tour_idxs:
        reformated_tour.extend(tours[num_tour])
        first = True
        for node in tours[num_tour]:
            if first:
                remaining_capacities.append(capacity - demands[node])
                first = False
            else:
                remaining_capacities.append(remaining_capacities[-1] - demands[node])

    # add depot at the beginning and the end
    remaining_capacities = [capacity] + remaining_capacities + [capacity]
    tour = [0] + reformated_tour + [0]
    tour = np.array(tour)
    via_depot = np.array([0.] * len(tour))
    via_depot[0] = 1.

    for i in range(1, len(remaining_capacities) - 1):
        if remaining_capacities[i] > remaining_capacities[i - 1]:
            via_depot[i] = 1.

    if reorder:
        coordinates = coordinates[tour]
        demands = demands[tour]

    return coordinates, demands, remaining_capacities, via_depot
def make_vrplib_data(filename):
    import vrplib

    node_coords = []
    demands = []
    capacitys = []
    costs = []
    names = []

    from tqdm import tqdm

    for line in tqdm(open(filename, "r").readlines(), ascii=True):
        line = line.split(", ")

        name_index = int(line.index('[\'name\''))
        depot_index = int(line.index('\'depot\''))
        customer_index = int(line.index('\'customer\''))
        capacity_index = int(line.index('\'capacity\''))
        demand_index = int(line.index('\'demand\''))
        cost_index = int(line.index('\'cost\''))

        depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
        customer = [[float(line[idx]), float(line[idx + 1])] for idx in
                    range(customer_index + 1, demand_index, 2)]

        loc = depot + customer
        # 包括 depot 的 location，在第一个

        capacity = int(float(line[capacity_index + 1]))
        # demand = [0] + [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
        demand = [int(line[idx]) for idx in range(demand_index + 1, capacity_index)]
        # [0] + 包括depot的demand，其为 0，在第一个

        cost = float(line[cost_index + 1])

        node_coords.append(loc)
        demands.append(demand)
        capacitys.append(capacity)
        costs.append(cost)
        names.append(line[name_index+1][1:-1])
        # print(node_coords,demands,capacitys,costs,names)
        # assert False

    # 每一行的数据表示一个instance，每一个instance的size不一样
    node_coords = np.array(node_coords,dtype=object)
    demands = np.array(demands,dtype=object)
    capacitys = np.array(capacitys)
    costs = np.array(costs)
    names = np.array(names)
    # print(instance_data.shape)

    return node_coords, demands, capacitys, costs, names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generate and solve CVRP")
    parser.add_argument("--num_instances", type=int, default=10, help="Number instances")
    parser.add_argument("--num_nodes", type=int, default=100, help="Number of nodes")
    parser.add_argument("--capacity", type=int, default=50, help="Capacity")
    parser.add_argument("--working_dir", type=str, default='work', )
    parser.add_argument("--num_runs", type=int, default=10, help="LKH num runs")
    parser.add_argument("--time_limit", type=int, default=10, help="LKH time limit")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reorder", dest="reorder", action="store_true",
                        help="Reorder nodes/tours. Must be reordered in training dataset")

    args = parser.parse_args()



    filename = 'CVRPlib_XXL_instances.txt'

    cvrp_node_coords, cvrp_demands, cvrp_capacitys, \
    vrplib_cost, vrplib_name = make_vrplib_data(filename)
    factors = []
    for kkk in range(len(cvrp_node_coords)):
        data = {}

        args.output_filename = f'CVRPlib_XXL_instances_n{kkk}'

        np.random.seed(args.seed)
        all_coords, all_demands, all_capacities, all_remaining_capacities = list(), list(), list(), list()
        all_via_depots, all_tour_lens = list(), list()


        for instance_num in range(kkk,kkk+1):
            coords = cvrp_node_coords[instance_num]
            factors.append(np.max(coords)-np.min(coords))
            demands = cvrp_demands[instance_num]
            capacity = cvrp_capacitys[instance_num]
            create_problem_file(instance_num, coords, demands, args.capacity, args.working_dir)
            create_parameter_file(instance_num, args.working_dir, num_runs=args.num_runs,   time_limit=args.time_limit)

            tour_len = vrplib_cost[instance_num]

            # add first node to the end

            coords.append(coords[0])

            demands.append(demands[0])
            coords = np.array(coords)
            demands = np.array(demands)



            remaining_capacities, via_depots = None, None

            all_coords.append(coords)
            all_demands.append(demands)
            all_remaining_capacities.append(remaining_capacities)
            all_via_depots.append(via_depots)
            all_tour_lens.append(tour_len)
            all_capacities.append(args.capacity)

        capacities = np.stack(all_capacities)
        coords = np.stack(all_coords)
        demands = np.stack(all_demands)

        tour_lens = np.stack(all_tour_lens)
        np.savez_compressed(args.output_filename, capacities=capacities, coords=coords, demands=demands,
                            tour_lens=tour_lens, reorder=False)

        print("Result saved in " + args.output_filename)
    print(factors)
