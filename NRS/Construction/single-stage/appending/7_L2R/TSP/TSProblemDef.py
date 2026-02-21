import pickle

import math
import torch
import numpy as np


def get_random_problems_tsp(batch_size, problem_size, device):
    problems = torch.rand(size=(batch_size, problem_size, 2),device=device)
    # problems.shape: (batch, problem, 2)
    return problems


def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems

def get_saved_tsp_data(filename, total_episodes,device, start=0, solution_file=None):

    data_type = filename.split('.')[-1]

    data_loader = {
        'pt': use_saved_problems_tsp_pt,
        'pkl': use_saved_problems_tsp_pkl,
        'txt': use_saved_problems_tsp_txt,
    }

    if data_type not in data_loader.keys():
        assert False, f"Unsupported file type: {data_type}. Supported types are: {list(data_loader.keys())}"

    return data_loader[data_type](filename, total_episodes, device, start, solution_file)

def use_saved_problems_tsp_pt(filename, total_episodes, device, start=0, solution_file=None):

    loaded_dict = torch.load(filename, map_location=device)
    problems = loaded_dict['node_xy'][start:start+total_episodes]
    optimal = None
    if 'optimal' in loaded_dict.keys():
        optimal = loaded_dict['optimal']
    if 'solutions' in loaded_dict.keys():
        solution = loaded_dict['solutions']
        gathering_index =  solution.unsqueeze(2).expand(-1, -1, 2)
        # shape: (batch, problem, 2)
        ordered_seq = problems.gather(dim=1, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(2).sqrt()
        # shape: (batch, problem)
        travel_distances = segment_lengths.sum(1)
        # shape: (batch,)
        optimal = travel_distances
    else:
        solution = None

    assert optimal is not None, 'optimal score is not given'

    #self.drawPic_TSP(self.saved_node_xy[0],self.solution[0], optimal=True,distribution=distribution)
    return problems, solution, optimal

def use_saved_problems_tsp_pkl(filename, total_episodes, device, start=0, solution_file=None):
    with open(filename, 'rb') as f:
        out_1 = pickle.load(f)[start:start+total_episodes]
        problems = torch.tensor(out_1, dtype=torch.float32,device=device)
        # shape: (batch, problem, 2)
        problem_size = problems.size(1)
    if solution_file is not None:
        with open(solution_file, 'rb') as f2:
            out_2 = pickle.load(f2)[start:start+total_episodes]
            out_2 = np.array(out_2, dtype=object)[:, 0].tolist()
            optimal_score_all = torch.tensor(out_2, dtype=torch.float32, device=device)
            optimal = optimal_score_all
    elif problem_size == 100 * 10000:
        optimal = 713.97143555
    else:
        assert False, 'optimal score is not given, please provide the solution file or set the optimal score'
    solution = None
    return problems, solution, optimal

def use_saved_problems_tsp_txt(filename, total_episodes,device, start=0, solution_file=None):
    nodes_coords = []
    solution = []

    for line in open(filename, "r").readlines()[start:start+total_episodes]:
        line = line.split(" ")
        num_nodes = int(line.index('output') // 2)
        nodes_coords.append(
            [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]
        )
        tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]]
        solution.append(tour_nodes)

    problems = torch.tensor(nodes_coords,device=device) # shape: (batch, problem, 2)

    solution = torch.tensor(solution,device=device)  # shape: (batch, problem)

    gathering_index = solution.unsqueeze(2).expand(-1, -1, 2)
    # shape: (batch, problem, 2)
    ordered_seq = problems.gather(dim=1, index=gathering_index)
    rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
    segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(2).sqrt()
    # shape: (batch, problem)
    travel_distances = segment_lengths.sum(1)
    # shape: (batch,)
    optimal = travel_distances

    return problems, solution, optimal
