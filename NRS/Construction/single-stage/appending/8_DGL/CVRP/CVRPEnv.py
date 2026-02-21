
from dataclasses import dataclass
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from utils.utils import *

import time
import functools
import math

def time_count(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        wrapper.total_time += run_time
        wrapper.calls += 1
        print(f"Function {func.__name__} called {wrapper.calls} times. Total run time: {wrapper.total_time:.4f} seconds. Last run time: {run_time:.4f} seconds.")
        return result
    wrapper.total_time = 0
    wrapper.calls = 0
    return wrapper


def get_random_problems(batch_size, problem_size):

    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 500
    else:
        demand_scaler = 500
        # raise NotImplementedError

    node_demand = torch.randint(1, 200, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)

    return depot_xy, node_xy, node_demand


def read_solutions_from_file(file_path):
    tour_storage = []
    tour_len_storage = []
    ellapsed_time_storage = []
    with open(file_path, 'r', encoding='utf8') as read_file:
        line_text = read_file.readline()
        while line_text:
            tour_text, tour_len_text, ellapsed_time_text = line_text.strip().split(" ")

            tour = [int(val) for val in tour_text.split(",")]
            tour_storage.append(tour)

            tour_len = float(tour_len_text)
            tour_len_storage.append(tour_len)

            ellapsed_time = float(ellapsed_time_text)
            ellapsed_time_storage.append(ellapsed_time)

            line_text = read_file.readline()

    # for tour in tour_storage:
    #     print(len(tour))

    tours = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in tour_storage], batch_first=True, padding_value=0)
    tour_lens = torch.tensor(tour_len_storage)
    time_consumptions = torch.tensor(ellapsed_time_storage)
    return tours, tour_lens, time_consumptions


def read_cvrp_instances_from_file(file_path):
    depot = []
    nodes = []
    demands = []
    capacity = []

    with open(file_path, 'r', encoding='utf8') as read_file:
        line_text = read_file.readline()
        while line_text:
            splitted_text = line_text.strip().split(" .|. ")

            instance_depot = [float(x) for x in splitted_text[0].strip().split(",")]

            instance_nodes = []
            for node_text in splitted_text[1].strip().split(" "):
                instance_nodes.append([float(x) for x in node_text.split(",")])

            instance_demands = [int(x) for x in splitted_text[2].strip().split(" ")]
            instance_capacity = int(splitted_text[3])

            instance_demands = (torch.Tensor(instance_demands) / instance_capacity).tolist()

            depot.append(instance_depot)
            nodes.append(instance_nodes)
            demands.append(instance_demands)

            line_text = read_file.readline()

    return torch.Tensor(depot), torch.Tensor(nodes), torch.Tensor(demands)

def load_cvrp_instances_with_baselines(root, problem_type, size, distribution):
    assert problem_type == "cvrp"
    assert size in (50, 500, 5000)
    assert distribution in ('uniform', 'clustered1', 'clustered2', 'explosion', 'implosion')

    # TODO: compare HGS and LKH3 after finishing LKH3 baselines and set final baseline
    baseline = "HGS"

    instance_root = Path(root)
    instance_dir = f"data_farm/{problem_type}/{problem_type}{size}/"
    instance_name = f"{problem_type}{size}_{distribution}.txt"
    instance_file = instance_root.joinpath(instance_dir).joinpath(instance_name)

    cvrp_instances = read_cvrp_instances_from_file(instance_file)
    # num = tsp_instances.size(0)
    # print(tsp_instances.size())

    solution_root = Path(root)
    solution_dir = f"solution_farm/{problem_type}{size}_{distribution}/"
    solution_name = f"{baseline}.txt"
    solution_file = solution_root.joinpath(solution_dir).joinpath(solution_name)
    baseline_tours, baseline_lens, _ = read_solutions_from_file(solution_file)

    return cvrp_instances, baseline_tours, baseline_lens


def load_cvrplib(file_path):
    # assert problem_type == "cvrp"
    # assert size in (50, 500, 5000)
    # assert distribution in ('uniform', 'clustered1', 'clustered2', 'explosion', 'implosion')

    # # TODO: compare HGS and LKH3 after finishing LKH3 baselines and set final baseline
    # baseline = "HGS"

    # instance_root = Path(root)
    # instance_dir = f"data_farm/{problem_type}/{problem_type}{size}/"
    # instance_name = f"{problem_type}{size}_{distribution}.txt"
    instance_file = file_path

    cvrp_instances = read_cvrp_instances_from_file(instance_file)
    # num = tsp_instances.size(0)
    # print(tsp_instances.size())

    # solution_root = Path(root)
    # solution_dir = f"solution_farm/{problem_type}{size}_{distribution}/"
    # solution_name = f"{baseline}.txt"
    # solution_file = solution_root.joinpath(solution_dir).joinpath(solution_name)
    # baseline_tours, baseline_lens, _ = read_solutions_from_file(solution_file)

    return cvrp_instances


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)

@dataclass
class Step_State:
    data: torch.Tensor
    avg_unselect_distence: torch.Tensor = None
    std_dev_unselect_distence: torch.Tensor = None
    capacity: torch.Tensor = None
    ninf_mask: torch.Tensor = None
    selected_flag: torch.Tensor = None
    distance_to_depot: torch.Tensor = None
    # sum_demand_aggregation: torch.Tensor = None
    # knn_count: torch.Tensor = None


class CVRPEnv:
    def __init__(self, **env_params):

        self.env_params = env_params
        self.problem_size = None
        self.mode = env_params['mode']
        if self.mode == 'test':
            if env_params['test_mode'] == 'pomo_test':
                self.pomo_size = env_params['aug_size']
            elif env_params['test_mode'] == 'aug_test':
                self.aug_size = env_params['aug_size']
                self.pomo_size = 1
        else:
            self.pomo_size = env_params["pomo_size"]

        self.append_information = env_params['append_information']
        self.batch_size = None
        self.problems = None
        self.raw_data_nodes = []
        self.raw_data_tours = []
        self.selected_count = None
        self.selected_node_list = None
        self.selected_student_list = None
        self.episode = None
        self.solution = None
        self.solution_flag = None
        self.solution_len = None
        self.step_state = None
        self.best_solution = []
        self.best_solution_len = []
        self.aggregation_nums = env_params['aggregation_nums']
    
    def load_data(self, test_episode, problem_size):
        if self.env_params["baseline_path"] == "":
             (self.raw_depot_node_xy, self.raw_node_xy, self.raw_node_demand) = load_cvrplib(self.env_params["data_path"])
        else:
            (self.raw_depot_node_xy, self.raw_node_xy, self.raw_node_demand), self.baseline_tours, self.baseline_len = load_cvrp_instances_with_baselines(self.env_params["data_path"],'cvrp',self.problem_size,self.env_params['distribution'])
            print("baseline_len avg", torch.mean(self.baseline_len))

    def load_problems(self, episode, batch_size, problem_size):
        self.episode = episode

        self.batch_size = batch_size

        self.problem_size = problem_size

        if self.mode == 'train':
            depot_xy, node_xy, node_demand = get_random_problems(batch_size, problem_size)

            self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
            # shape: (batch, problem+1, 2)
            depot_demand = torch.zeros(size=(self.batch_size, 1))
            # shape: (batch, 1)
            self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
            # shape: (batch, problem+1)
        elif self.mode == 'test':
            depot_xy = self.raw_depot_node_xy[episode:episode + batch_size]
            node_xy = self.raw_node_xy[episode:episode + batch_size]

            self.depot_node_xy = torch.cat((depot_xy[:, None,:], node_xy), dim=1)

            depot_demand = torch.zeros(size=(self.batch_size, 1))
            node_demand = self.raw_node_demand[episode:episode + batch_size]
            self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)

            self.origin_problems = torch.cat((self.depot_node_xy,self.depot_node_demand[:,:,None]),dim=2)

            if self.env_params['test_mode'] == "aug_test":
                self.depot_node_xy = self.depot_node_xy.repeat_interleave(self.aug_size, dim=0)
                self.depot_node_demand = self.depot_node_demand.repeat_interleave(self.aug_size, dim=0) 
                # print("self.depot_node_xy", self.depot_node_xy.shape)
                # print("self.depot_node_demand", self.depot_node_demand.shape)
                # print("self.aug_size", self.aug_size)

                self.batch_size = self.batch_size * self.aug_size


        self.problems = torch.cat((self.depot_node_xy,self.depot_node_demand[:,:,None]),dim=2)

        self.dis_matrix = torch.cdist(self.depot_node_xy, self.depot_node_xy, p=2).detach()
        
        self.reset_solution()

    def randowm_augment_xy_data_by_8_fold(self):
        # xy_data.shape: (batch, N, 2)

        x = self.problems[:, :, [0]]
        y = self.problems[:, :, [1]]
        # x,y shape: (batch, N, 1)

        random_int = torch.randint(0, 8, (1,)).item()

        if random_int == 0:
            pass
        elif random_int == 1:
            x = 1 - x
        elif random_int == 2:
            y = 1 - y
        elif random_int == 3:
            x = 1 - x
            y = 1 - y
        elif random_int == 4:
            x, y = y, x
        elif random_int == 5:
            x, y = 1 - y, x
        elif random_int == 6:
            x, y = y, 1 - x
        elif random_int == 7:
            x, y = 1 - y, 1 - x
        else:
            raise Exception("random_int should be in [0, 7]")
        
        self.problems[:, :, [0]] = x
        self.problems[:, :, [1]] = y

    def random_replace(self):

        random_index = torch.randperm(self.batch_size)[:self.replace_size]

        depot_xy, node_xy, node_demand = get_random_problems(self.replace_size, self.problem_size)

        depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)

        self.depot_node_xy[random_index] = depot_node_xy

        depot_demand = torch.zeros(size=(self.replace_size, 1))
        # shape: (batch, 1)
        depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)

        self.depot_node_demand[random_index] = depot_node_demand

        replace_problems = torch.cat((depot_node_xy, depot_node_demand[:,:,None]),dim=2)

        self.problems[random_index] = replace_problems
        
        self.solution_len[random_index] = torch.tensor(float('inf'), dtype=torch.float32).repeat(self.replace_size)
        self.dis_matrix = torch.cdist(self.depot_node_xy, self.depot_node_xy, p=2).detach()

        return random_index

    def data_augmentation(self):
        # pass
        # self.randowm_augment_xy_data_by_8_fold()

        for i in range(self.batch_size):
            indices = torch.nonzero(self.solution_flag[i], as_tuple=True)[0]
            for j in range(1, len(indices)):
                if torch.rand(1) > 0.5:
                    self.solution[i][indices[j-1]:indices[j]] = torch.flip(self.solution[i][indices[j-1]:indices[j]], [0])

            random_shift_num = torch.randint(0, indices.shape[0], (1,))[0].item()

            self.solution[i] = torch.cat((self.solution[i][indices[random_shift_num]:], self.solution[i][:indices[random_shift_num]]))
            self.solution_flag[i] = torch.cat((self.solution_flag[i][indices[random_shift_num]:], self.solution_flag[i][:indices[random_shift_num]]))

    def greedy_search(self):
        selected_node_list = torch.zeros((self.batch_size, 0), dtype=torch.long)

        last_node = torch.zeros(self.batch_size, dtype=torch.int64).unsqueeze(-1)
        distance = self.dis_matrix.gather(1, last_node.view(self.batch_size, 1, 1).expand(-1, -1, self.problem_size + 1)).squeeze(1)
        mask = torch.zeros_like(distance)
        index_1 = torch.arange(self.batch_size, dtype=torch.long)
        mask[index_1, 0] = 1e5
        selected_node = torch.argsort(distance + mask, dim=1)[:, :1]

        selected_node_list = torch.cat((selected_node_list, selected_node), dim=1)

        selected_flag_list = torch.ones(self.batch_size, dtype=torch.int64).unsqueeze(-1)

        capacity = torch.ones(self.batch_size, dtype=torch.int64)

        index = torch.arange(self.batch_size)

        capacity = capacity - self.problems[index,selected_node.squeeze(-1),2]

        step = 1

        while step < self.problem_size:

            last_node = selected_node_list[:,-1]

            distance = self.dis_matrix.gather(1, last_node.view(self.batch_size, 1, 1).expand(-1, -1, self.problem_size + 1)).squeeze(1)

            mask = torch.zeros_like(distance)

            index_1 = torch.arange(self.batch_size, dtype=torch.long)[:, None].expand(self.batch_size, selected_node_list.shape[1])

            mask[index_1, selected_node_list] = 1e5

            index_1 = torch.arange(self.batch_size, dtype=torch.long)

            mask[index_1, 0] = 1e5

            selected_node = torch.argsort(distance + mask, dim=1)[:, :1]

            selected_flag = torch.zeros(self.batch_size, dtype=torch.int64).unsqueeze(-1)

            round_error_epsilon = 0.000001
            via_depot = (capacity + round_error_epsilon) < self.problems[index,selected_node.squeeze(-1),2]

            capacity[~via_depot] = capacity[~via_depot] - self.problems[index,selected_node.squeeze(-1),2][~via_depot]

            capacity[via_depot] = 1 - self.problems[index,selected_node.squeeze(-1),2][via_depot]

            selected_flag[via_depot] = 1

            selected_node_list = torch.cat((selected_node_list, selected_node), dim=1)

            selected_flag_list = torch.cat((selected_flag_list, selected_flag), dim=1)

            step += 1

        self.solution = selected_node_list

        self.solution_flag = selected_flag_list

        # print("selected_node_list:", selected_node_list)

        # print("self.solution_flag", self.solution_flag)

        index = torch.arange(self.batch_size)[:,None].expand_as(selected_node_list)

        self.solution_len = self.cal_length(self.problems[:,:,:2], self.solution, self.solution_flag)

        return self.solution_len.mean().item()

    # ! 没有CVRPSolver，注释掉
    # @torch.no_grad()
    # def get_best_solution(self):

    #     self.best_solution = []
    #     self.best_solution_len = []

    #     for problem in self.problems:
    #         xs = []
    #         ys = []
    #         for (x, y) in problem:
    #             xs.append(int(x * 10000))
    #             ys.append(int(y * 10000))
    #         solver = CVRPSolver.from_data(xs, ys, norm="EUC_2D")
    #         solution = solver.solve()
    #         print(solution.found_tour)
    #         self.best_solution.append(solution.tour)
    #         self.best_solution_len.append(solution.optimal_value/10000)

    #     self.best_solution = torch.tensor(self.best_solution, dtype=torch.long)
    #     self.best_solution_len = torch.tensor(self.best_solution_len, dtype=torch.float)

    @torch.no_grad()
    def get_unselect_list(self):

        batch_size = self.selected_node_list.shape[0]

        new_list = torch.arange(self.problem_size)[None, :].repeat(batch_size, 1)

        new_list_len = self.problem_size - self.selected_node_list.shape[1]  # shape: [B, V-current_step]

        index_2 = self.selected_node_list.type(torch.long)

        index_1 = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, index_2.shape[1])

        new_list[index_1, index_2] = -2

        unselect_list = new_list[torch.gt(new_list, -1)].view(batch_size, new_list_len)

        return unselect_list
        
    @torch.no_grad()
    def reset(self, mode,):

        self.selected_count = 0

        if mode == 'test':
            self.mode = mode
            with torch.no_grad():

                self.selected_node_list = torch.zeros((self.batch_size*self.pomo_size*self.beam_size, 0), dtype=torch.long)
                
                if self.env_params['test_mode'] == "aug_test":
                    # print("self.problems", self.problems.shape)
                    # print("self.problems[self.start_idx:self.start_idx + self.step_size]", self.problems[self.start_idx:self.start_idx + self.batch_size].shape)
                    self.step_state = Step_State(data=self.problems[self.start_idx:self.start_idx + self.batch_size].clone())

                    self.step_state.data[:,:,:2] = run_aug("mix", self.step_state.data[:,:,:2])
                else:
                    self.step_state = Step_State(data=self.problems[self.start_idx:self.start_idx + self.batch_size])

                self.step_state.capacity = torch.ones(size=(self.batch_size*self.pomo_size*self.beam_size,))
                self.step_state.ninf_mask = torch.zeros(size=(self.batch_size*self.pomo_size*self.beam_size, self.problem_size+1))
                self.step_state.selected_flag = torch.zeros((self.batch_size*self.pomo_size*self.beam_size, 0), dtype=torch.long)

                self.dis_matrix = torch.cdist(self.step_state.data[:,:,:2], self.step_state.data[:,:,:2], p=2).to(torch.float32)

                self.step_state.data = torch.repeat_interleave(self.step_state.data ,self.pomo_size ,0)

                self.step_state.data = torch.repeat_interleave(self.step_state.data ,self.beam_size ,0)

                self.step_state.avg_unselect_distence = torch.mean(self.dis_matrix, dim=-1).repeat_interleave(self.pomo_size*self.beam_size, dim=0)
                self.step_state.std_dev_unselect_distence = torch.std(self.dis_matrix, unbiased=False, dim=-1).repeat_interleave(self.pomo_size*self.beam_size, dim=0)
                
        if mode == 'valid':
            self.mode = mode
            with torch.no_grad():

                self.selected_node_list = torch.zeros((self.step_size*self.pomo_size*self.beam_size, 0), dtype=torch.long)
                

                self.step_state = Step_State(data=self.problems[self.start_idx:self.start_idx + self.step_size])


                self.step_state.capacity = torch.ones(size=(self.step_size*self.pomo_size*self.beam_size,))
                self.step_state.ninf_mask = torch.zeros(size=(self.step_size*self.pomo_size*self.beam_size, self.problem_size+1))
                self.step_state.selected_flag = torch.zeros((self.step_size*self.pomo_size*self.beam_size, 0), dtype=torch.long)

                self.dis_matrix = torch.cdist(self.step_state.data[:,:,:2], self.step_state.data[:,:,:2], p=2).to(torch.float32)

                self.step_state.data = torch.repeat_interleave(self.step_state.data ,self.pomo_size ,0)

                self.step_state.data = torch.repeat_interleave(self.step_state.data ,self.beam_size ,0)

                self.step_state.avg_unselect_distence = torch.mean(self.dis_matrix, dim=-1).repeat_interleave(self.pomo_size*self.beam_size, dim=0)
                self.step_state.std_dev_unselect_distence = torch.std(self.dis_matrix, unbiased=False, dim=-1).repeat_interleave(self.pomo_size*self.beam_size, dim=0)


        if mode == 'train':
            self.mode = mode
            self.selected_node_list = torch.zeros((self.batch_size, 0), dtype=torch.long)

            self.step_state = Step_State(data=self.problems.clone())

            self.step_state.data[:,:,:2] = run_aug("mix", self.step_state.data[:,:,:2])

            self.step_state.capacity = torch.ones(size=(self.batch_size,))
            self.step_state.ninf_mask = torch.zeros(size=(self.batch_size, self.problem_size+1))
            self.step_state.selected_flag = torch.zeros((self.batch_size, 0), dtype=torch.long)

            
            self.dis_matrix = torch.cdist(self.step_state.data, self.step_state.data, p=2).to(torch.float32)

            self.step_state.avg_unselect_distence = torch.mean(self.dis_matrix, dim=-1)
            self.step_state.std_dev_unselect_distence = torch.std(self.dis_matrix, unbiased=False, dim=-1)

        reward = None
        done = False
        return Reset_State(self.problems), reward, done


    def reset_solution(self):
        self.solution = None
        self.solution_len = None

    # @time_count
    def perm_attr(self, attr_name, prev_k):
        # attr = getattr(self.step_state, attr_name)

        # attr = attr.view(self.batch_size*self.pomo_size, self.beam_size, self.problem_size, -1)

        # perm_attr = prev_k.unsqueeze(2).unsqueeze(3).expand_as(attr)
        # attr = attr.gather(1, perm_attr)

        # attr = attr.view(self.batch_size*self.pomo_size*self.beam_size, self.problem_size, -1)

        # setattr(self.step_state, attr_name, attr)
        attr = getattr(self.step_state, attr_name)

        if self.mode == 'test' or self.mode == 'train':
            batch_size = self.batch_size
        elif self.mode == 'valid':
            batch_size = self.step_size

        if attr == None:
            return 
        
        if len(attr.shape) == 1:
            attr = attr.view(batch_size*self.pomo_size, self.beam_size)
            perm_attr = prev_k
            attr = attr.gather(1, perm_attr)
            attr = attr.view(batch_size*self.pomo_size*self.beam_size)
            setattr(self.step_state, attr_name, attr)

        elif len(attr.shape) == 2:
            
            attr = attr.view(batch_size*self.pomo_size, self.beam_size,-1)
            perm_attr = prev_k.unsqueeze(2).expand_as(attr)
            attr = attr.gather(1, perm_attr)
            attr = attr.view(batch_size*self.pomo_size*self.beam_size, -1)
            setattr(self.step_state, attr_name, attr)

        elif len(attr.shape) == 3:
            shape1 = attr.shape[1]
            attr = attr.view(batch_size*self.pomo_size, self.beam_size, shape1 ,-1)
            perm_attr = prev_k.unsqueeze(2).unsqueeze(3).expand_as(attr)
            attr = attr.gather(1, perm_attr)
            attr = attr.view(batch_size*self.pomo_size*self.beam_size, shape1, -1)
            setattr(self.step_state, attr_name, attr)
        else:
            raise Exception("attr shape error")
            exit(0)


    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    @torch.no_grad()
    # @TimeCount
    def step(self, selected):
        
        # if self.selected_count < 5:
        #     if self.selected_count == 4:
        #         exit(0)
            # print("self.selected_count", self.selected_count)
            # print("self.selected_node_list", self.selected_node_list[:10])
            # print("self.step_state.selected_flag", self.step_state.selected_flag[:10])

        if self.mode == "valid" or self.mode == "test":
            pomo_size = self.pomo_size
            beam_size = self.beam_size
        elif self.mode == "train":
            pomo_size = 1
            beam_size = 1

        if self.mode == 'test' or self.mode == 'train':
            batch_size = self.batch_size
        elif self.mode == 'valid':
            batch_size = self.step_size

        via_depot = (selected > self.problem_size)

        # print("self.selected_node_list:", self.selected_node_list)
        # print("self.step_state.selected_flag:", self.step_state.selected_flag)

        # print("via_depot: ", via_depot)

        selected[via_depot] = selected[via_depot] - (self.problem_size + 1)

        selected_flag = torch.zeros_like(selected)
        selected_flag[via_depot] = 1

        index = torch.arange(batch_size).repeat_interleave(pomo_size*beam_size)

        if self.mode == "valid":
            self.step_state.capacity[via_depot] = 1 - self.problems[self.start_idx:self.start_idx + self.step_size][index,selected,2][via_depot]
            self.step_state.capacity[~via_depot] = self.step_state.capacity[~via_depot] - self.problems[self.start_idx:self.start_idx + self.step_size][index,selected,2][~via_depot]
        else:
            self.step_state.capacity[via_depot] = 1 - self.problems[index,selected,2][via_depot]
            self.step_state.capacity[~via_depot] = self.step_state.capacity[~via_depot] - self.problems[index,selected,2][~via_depot]

        round_error_epsilon = 0.000001
        demand_too_large = (self.step_state.capacity[:, None] + round_error_epsilon) < self.step_state.data[:, :, 2]

        self.step_state.ninf_mask.zero_()
        self.step_state.ninf_mask[demand_too_large] = float('-inf')

        self.selected_count += 1

        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, None]), dim=1)  # shape: [B, current_step]

        # print("self.select_list", self.selected_node_list)

        self.step_state.selected_flag = torch.cat((self.step_state.selected_flag, selected_flag[:, None]), dim=1)  # shape: [B, current_step]

        # print("self.step_state.selected_flag", self.step_state.selected_flag)

        # print("self.selected_node_list", self.selected_node_list)
        # print("self.mode:", self.mode)
        # print("self.selected_node_list==0", torch.sum(self.selected_node_list == 0))
        # print("self.step_state.selected_flag", self.step_state.selected_flag)

        range_index = torch.arange(self.step_state.data.shape[0], dtype=torch.long)

        dest_node_idx = None
        distance_to_dest = None
        last_node = None

        bs = self.step_state.data.shape[0]

        unselect_count = self.problem_size + 1 - self.selected_count

        problem_size = self.problem_size + 1

        if self.append_information[2] == True:
            before_avg_unselect_distence = self.step_state.avg_unselect_distence

        feature_num = 0

        if self.append_information[0] == True:

            current_node = self.selected_node_list[:,-1].view(-1)

            current_node = current_node.repeat_interleave(problem_size)
            index1 = torch.arange(batch_size).repeat_interleave(pomo_size*beam_size*problem_size)
            index2 = torch.arange(problem_size).repeat(bs)

            distance_to_current = self.dis_matrix[index1, index2, current_node].view(bs, problem_size)

            if self.selected_count == 1:
                current_node = torch.zeros(self.selected_node_list.shape[0]).to(torch.int64)
                current_node = current_node.repeat_interleave(problem_size)
                index1 = torch.arange(batch_size).repeat_interleave(pomo_size*beam_size*problem_size)
                index2 = torch.arange(problem_size).repeat(bs)

                distance_to_depot = self.dis_matrix[index1, index2, current_node].view(bs, problem_size)
                self.step_state.distance_to_depot = distance_to_depot
                # print("self.self.step_state.distance_to_depot:", self.step_state.distance_to_depot.shape)


            feature_num = feature_num + 1
            if self.selected_count == 1:
                self.step_state.data = torch.cat((self.step_state.data, distance_to_current.unsqueeze(-1)), dim=-1)
            else:
                self.step_state.data[:,:,2 + feature_num] = distance_to_current

        if self.append_information[1] == True:
            
            if self.append_information[0] == False:

                current_node = self.selected_node_list[:,-1].view(-1)

                current_node = current_node.repeat_interleave(problem_size)
                index1 = torch.arange(batch_size).repeat_interleave(pomo_size*beam_size*problem_size)
                index2 = torch.arange(problem_size).repeat(bs)

                distance_to_current = self.dis_matrix[index1, index2, current_node].view(bs, problem_size)


            self.step_state.avg_unselect_distence = (self.step_state.avg_unselect_distence * (unselect_count + 1) - distance_to_current) / unselect_count            


            feature_num = feature_num + 1
            if self.selected_count == 1:
                self.step_state.data = torch.cat((self.step_state.data, self.step_state.avg_unselect_distence.unsqueeze(-1).to(torch.float32)), dim=-1)
            else:
                self.step_state.data[:,:,2 + feature_num] = self.step_state.avg_unselect_distence.to(torch.float32)


        if self.append_information[2] == True:

            if self.append_information[0] == False:

                current_node = self.selected_node_list[:,-1].view(-1)

                current_node = current_node.repeat_interleave(problem_size)
                index1 = torch.arange(batch_size).repeat_interleave(pomo_size*beam_size*problem_size)
                index2 = torch.arange(problem_size).repeat(bs)

                distance_to_current = self.dis_matrix[index1, index2, current_node].view(bs, problem_size)
            
            sqrt_item = (torch.square(self.step_state.std_dev_unselect_distence)*(unselect_count + 1) - torch.square(distance_to_current - before_avg_unselect_distence)) / unselect_count

            mask = sqrt_item < 0
            sqrt_item[mask] = 0

            self.step_state.std_dev_unselect_distence = torch.sqrt(sqrt_item)

            # self.step_state.std_dev_unselect_distence = torch.sqrt(sqrt_item) / unselect_count

            feature_num = feature_num + 1
            if self.selected_count == 1:
                self.step_state.data = torch.cat((self.step_state.data, self.step_state.std_dev_unselect_distence.unsqueeze(-1).to(torch.float32)), dim=-1)
            else:
                self.step_state.data[:,:,2 + feature_num] = self.step_state.std_dev_unselect_distence.to(torch.float32)

        if self.selected_count == 1 and self.append_information[3] == True:

            dest_node_idx = torch.zeros(self.selected_node_list.shape[0]).to(torch.int64)

            dest_node_idx = dest_node_idx.repeat_interleave(problem_size)
            index1 = torch.arange(batch_size).repeat_interleave(pomo_size*beam_size*problem_size)
            index2 = torch.arange(problem_size).repeat(bs)

            distance_to_dest = self.dis_matrix[index1, index2, dest_node_idx].view(bs, problem_size, 1)

            self.step_state.data = torch.cat((self.step_state.data, distance_to_dest), dim=-1)


        if self.selected_count == 1 and self.append_information[4] == True:

            dest_node_idx = torch.zeros(self.selected_node_list.shape[0]).to(torch.int64)

            dest_node_idx_repeat = dest_node_idx.repeat_interleave(problem_size)
            index1 = torch.arange(batch_size).repeat_interleave(pomo_size*beam_size*problem_size)
            index2 = torch.arange(problem_size).repeat(bs)

            distance_to_dest = self.dis_matrix[index1, index2, dest_node_idx_repeat].view(bs, problem_size, 1)


            last_node = self.step_state.data[range_index,dest_node_idx,:2]

            delta_y = last_node[:,1].unsqueeze(-1).repeat(1, problem_size) - self.step_state.data[:,:,1]

            # print("delta_y", delta_y)

            sin_to_dest = delta_y.unsqueeze(-1) / (distance_to_dest + 1e-7)

            # sin_to_dest = (sin_to_dest + 1) / 2

            self.step_state.data = torch.cat((self.step_state.data, sin_to_dest), dim=-1)

        if self.selected_count == 1 and self.append_information[5] == True:

            dest_node_idx = torch.zeros(self.selected_node_list.shape[0]).to(torch.int64)

            dest_node_idx_repeat = dest_node_idx.repeat_interleave(problem_size)
            index1 = torch.arange(batch_size).repeat_interleave(pomo_size*beam_size*problem_size)
            index2 = torch.arange(problem_size).repeat(bs)

            distance_to_dest = self.dis_matrix[index1, index2, dest_node_idx_repeat].view(bs, problem_size, 1)

            last_node = self.step_state.data[range_index,dest_node_idx,:2]
            delta_x = last_node[:,0].unsqueeze(-1).repeat(1, problem_size) - self.step_state.data[:,:,0]

            cos_to_dest = delta_x.unsqueeze(-1) / (distance_to_dest + 1e-7)

            # cos_to_dest = (cos_to_dest + 1) / 2

            self.step_state.data = torch.cat((self.step_state.data, cos_to_dest), dim=-1)

        if self.selected_count == 1 and self.append_information[6] == True:

            avg_distence_all = torch.mean(self.dis_matrix, dim=-1).repeat_interleave(pomo_size*beam_size, dim=0)

            self.step_state.data = torch.cat((self.step_state.data, avg_distence_all.unsqueeze(-1)), dim=-1)

        if self.selected_count == 1 and self.append_information[7] == True:

            std_dev_distence_all = torch.std(self.dis_matrix, unbiased=False, dim=-1).repeat_interleave(pomo_size*beam_size, dim=0)

            self.step_state.data = torch.cat((self.step_state.data, std_dev_distence_all.unsqueeze(-1)), dim=-1)


        if self.selected_count == 1 and self.append_information[9] == True:

            avg_distence_aggregation = torch.mean(self.dis_matrix.topk(k=self.aggregation_nums, dim=-1, largest=False)[0], dim=-1).repeat_interleave(pomo_size*beam_size, dim=0)

            self.step_state.data = torch.cat((self.step_state.data, avg_distence_aggregation.unsqueeze(-1)), dim=-1)


        if self.selected_count == 1 and self.append_information[10] == True:

            std_dev_distence_aggregation = torch.std(self.dis_matrix.topk(k=self.aggregation_nums, dim=-1, largest=False)[0], unbiased=False, dim=-1).repeat_interleave(pomo_size*beam_size, dim=0)

            self.step_state.data = torch.cat((self.step_state.data, std_dev_distence_aggregation.unsqueeze(-1)), dim=-1)


        if self.append_information[11] == True:


            if self.selected_count == 1:
                all_demand = self.problems[:,:,2].unsqueeze(1).repeat_interleave(self.problem_size + 1, dim=1)
                self.knn_index = self.dis_matrix.topk(k=self.aggregation_nums, dim=-1, largest=False)[1]
                self.step_state.sum_demand_aggregation = torch.sum(all_demand.gather(dim=-1, index=self.knn_index), dim=-1).repeat_interleave(pomo_size*beam_size, dim=0)  
                self.step_state.knn_count = torch.full((self.step_state.sum_demand_aggregation.shape[0], self.problem_size + 1), self.aggregation_nums + 0.01)
                self.step_state.data = torch.cat((self.step_state.data, (self.step_state.sum_demand_aggregation/self.step_state.knn_count).unsqueeze(-1)), dim=-1)

                self.knn_index = self.knn_index.repeat_interleave(pomo_size*beam_size, dim=0)  
            else:
                in_knn = torch.any(torch.eq(self.knn_index.view(-1, self.aggregation_nums), selected[:, None].repeat_interleave(self.problem_size + 1, dim=0)), dim=1).view(bs, self.problem_size + 1)
                demand = self.step_state.data[torch.arange(bs), selected, 2]
                self.step_state.sum_demand_aggregation[in_knn] -= demand.repeat_interleave(self.problem_size + 1, dim=0).view(bs, self.problem_size + 1)[in_knn]
                self.step_state.knn_count[in_knn] -= 1
                self.step_state.data[:,:,-1] = (self.step_state.sum_demand_aggregation/self.step_state.knn_count)


        if self.selected_count == 1 and self.append_information[12] == True:
            
            if self.append_information[11] == False:
                all_demand = self.problems[:,:,2].unsqueeze(1).repeat_interleave(self.problem_size + 1, dim=1)
                index = self.dis_matrix.topk(k=self.aggregation_nums, dim=-1, largest=False)[1]

            std_demand_aggregation = torch.std(all_demand.gather(dim=-1, index=index), dim=-1).repeat_interleave(pomo_size*beam_size, dim=0) 

            self.step_state.data = torch.cat((self.step_state.data, std_demand_aggregation.unsqueeze(-1)), dim=-1)


        done = (self.selected_count == self.problem_size)
        if done:
            # self.drawPic_VRP(self.problems[:,:,:2], self.selected_node_list, self.step_state.selected_flag, name=str(1))
            # reward = self._get_travel_distance()
            if self.mode == "test" and self.env_params['test_mode'] == "aug_test":
                reward = self.cal_length(self.problems[:,:,:2].repeat_interleave(self.beam_size, dim=0), self.selected_node_list, self.step_state.selected_flag)
            else:
                reward = self.cal_length(self.step_state.data[:,:,:2], self.selected_node_list, self.step_state.selected_flag)

            # self.drawPic_VRP(self.problems, self.selected_node_list,self.step_state.selected_flag, name="1",index = 22)
        else:
            reward = None, None

        return self.step_state, reward, done

    def make_dir(self,path_destination):
        isExists = os.path.exists(path_destination)
        if not isExists:
            os.makedirs(path_destination)
        return


    def drawPic_VRP(self, coor_, order_node_,order_flag_,name='xx', index = 0):
        # coor: shape (V,2)
        # order_node_: shape (V)
        # order_flag_: shape (V)

        coor = coor_[index].clone().cpu().numpy()
        order_node =  order_node_[index].clone().cpu().numpy()
        order_flag = order_flag_[index].clone().cpu().numpy()

        tour = []
        for i in range(len(order_node)):
            if order_flag[i]==1:
                tour.append(0)
                tour.append(order_node[i])
            if order_flag[i]==0:
                tour.append(order_node[i])

        arr_max = np.max(coor)
        arr_min = np.min(coor)
        arr = (coor - arr_min) / (arr_max - arr_min)

        fig, ax = plt.subplots(figsize=(20, 20))

        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.axis('off')
        plt.scatter(arr[0, 0], arr[0, 1], color='red', linewidth=15,marker='v')

        col_counter = order_flag.sum()
        colors = plt.cm.turbo(np.linspace(0, 1, col_counter)) # turbo
        np.random.seed(123)
        np.random.shuffle(colors)

        count = -1
        for i in range(len(tour) - 1):
            if tour[i]==0:
                count+=1

            tour = np.array(tour, dtype=int)

            start = [arr[tour[i], 0], arr[tour[i + 1], 0]]
            end = [arr[tour[i], 1], arr[tour[i + 1], 1]]
            plt.plot(start, end, color=colors[count], linewidth=3)  # ,linestyle ="dashed"

            plt.scatter(arr[tour[i], 0], arr[tour[i], 1], color='gray', linewidth=2)
            plt.scatter(arr[tour[i+1], 0], arr[tour[i+1], 1], color='gray', linewidth=2)

        b = os.path.abspath(".")
        path = b+'/figure'
        self.make_dir(path)
        plt.savefig(path+f'/{name}.pdf',bbox_inches='tight', pad_inches=0)
        # plt.show()



    def _get_travel_distance(self):
        # trained model's distance
        gathering_index = self.selected_node_list.unsqueeze(2).expand(-1, self.problems.shape[1], 2)
        ordered_seq = self.step_state.data.gather(dim=1, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2)
        segment_lengths = segment_lengths.sum(2).sqrt()
        # shape: (batch,problem)
        travel_distances = segment_lengths.sum(1)
        # shape: (batch)
        return travel_distances
    

    def cal_length(self, problems, order_node, order_flag):
        # problems:   [B,V+1,2]
        # order_node: [B,V]
        # order_flag: [B,V]
        order_node_ = order_node.clone()

        order_flag_ = order_flag.clone()

        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)

        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = 0

        # print("order_flag_", order_flag_[0].tolist())

        roll_node = order_node_.roll(dims=1, shifts=1)

        problem_size = problems.shape[1] - 1

        # print("problem_size", problem_size)

        order_gathering_index = order_node_.unsqueeze(2).expand(-1, problem_size, 2)
        order_loc = problems.gather(dim=1, index=order_gathering_index)

        roll_gathering_index = roll_node.unsqueeze(2).expand(-1, problem_size, 2)
        roll_loc = problems.gather(dim=1, index=roll_gathering_index)

        flag_gathering_index = order_flag_.unsqueeze(2).expand(-1, problem_size, 2)
        flag_loc = problems.gather(dim=1, index=flag_gathering_index)

        order_lengths = ((order_loc - flag_loc) ** 2)

        order_flag_[:,0]=0
        # print("2order_flag_", order_flag_[0].tolist())
        flag_gathering_index = order_flag_.unsqueeze(2).expand(-1, problem_size, 2)
        flag_loc = problems.gather(dim=1, index=flag_gathering_index)

        roll_lengths = ((roll_loc - flag_loc) ** 2)

        length = (order_lengths.sum(2).sqrt() + roll_lengths.sum(2).sqrt()).sum(1)

        return length


