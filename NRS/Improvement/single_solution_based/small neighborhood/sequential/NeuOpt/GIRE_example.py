import os
import json
import torch
import pprint
import numpy as np
import random
from tensorboard_logger import Logger as TbLogger
import warnings

from problems.problem_tsp import TSP
from problems.problem_cvrp import CVRP
from agent.ppo import PPO
from tqdm.notebook import tqdm

from options import get_options
opts = get_options('')
opts

opts.problem = 'cvrp' 
opts.dummy_rate = 0.4
opts.wo_feature1 = False
opts.wo_feature3 = False
opts.wo_regular = False
opts.wo_bonus = False
opts.wo_RNN = False
opts.wo_MDP = True
opts.graph_size = 50
opts.val_dataset='./datasets/cvrp_50.pkl'
opts.init_val_met = 'random'
opts.no_saving = True
opts.no_tb = True
opts.val_size = 10
opts.load_path = './pre-trained/cvrp50.pt'
opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
opts.no_progress_bar = True

def load_problem(name):
    problem = {
        'tsp': TSP,
        'cvrp': CVRP,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem

# Disable Tensorboard logging
tb_logger = None

# Figure out what's the problem
problem = load_problem(opts.problem)(
                        p_size = opts.graph_size,
                        init_val_met = opts.init_val_met,
                        with_assert = opts.use_assert,
                        DUMMY_RATE = opts.dummy_rate,
                        k = opts.k,
                        with_bonus = not opts.wo_bonus,
                        with_regular = not opts.wo_regular)

# Figure out the RL algorithm
agent = PPO(problem, opts)

# Load data from load_path
assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
load_path = opts.load_path if opts.load_path is not None else opts.resume
if load_path is not None:
    agent.load(load_path)

from torch.utils.data import DataLoader
from problems.problem_cvrp import CVRPDataset
dataset = CVRPDataset(size = 50, num_samples = 10, DUMMY_RATE=0.4)
batch = next(iter(DataLoader(dataset, batch_size=10)))
coordinates_first = batch['coordinates'][0]
demand_first = batch['demand'][0]

rec = problem.get_initial_solutions(batch)
print(rec[0])

import torch
from matplotlib import pyplot as plt

def plot_tour(rec, coordinates, dpi = 300):
    real_seq = problem.get_order(rec.unsqueeze(0), return_solution = True)
    plt.figure(figsize=(8,6))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axis([-0.05, 1.05]*2)
    # plot the nodes
    plt.scatter(coordinates[:,0], coordinates[:,1], marker = 'H', s = 55, c = 'blue', zorder = 2)
    # plot the tour
    real_seq_coordinates = coordinates.gather(0,real_seq[0].unsqueeze(1).repeat(1,2))
    real_seq_coordinates = torch.cat((real_seq_coordinates, real_seq_coordinates[:1]),0)
    plt.plot(real_seq_coordinates[:,0], real_seq_coordinates[:,1], color = 'black', zorder = 1)
    # mark node
    for i,txt in enumerate(range(rec.size(0))):
        plt.annotate(txt,(coordinates[i,0]+0.01, coordinates[i,1]+0.01),)
    
    
    plt.show()

plot_tour(rec[0], coordinates_first)
print('Linked list format (rec variable):\n', rec[0])
print('\nHere, the linked list format means:')
for i in range(10):
    print(f'edge {i}-{rec[0,i]} is in the solution')

real_seq = problem.get_order(rec, return_solution = True)
print('\nReal solution after decoding (node visited in sequence):\n', real_seq[0])