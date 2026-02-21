"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import argparse
import time
from args import add_common_args,add_common_training_args
from learning.cvrp.data_iterator import DataIterator
from learning.cvrp.traj_learner import TrajectoryLearner
from utils.exp import setup_exp
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cvrp_test')
    add_common_args(parser)  # (only need common args)
    add_common_training_args(parser)
    args = parser.parse_args()

    b = os.path.abspath(".").replace('\\', '/')


    args.pretrained_model = b + '/pretrained_models/cvrp.best'
    args.test_dataset = b + f'/data/vrplib_instances_X.txt'
    args.episode = 100
    args.test_batch_size = 1
    args.cuda_device_num = 0
    args.knns = 250
    args.beam_size = 1

    net, module, device, _, checkpointer, _ = setup_exp(args, "cvrp", True)

    data_iterator = DataIterator(args)

    traj_learner = TrajectoryLearner(args, net, module, device, data_iterator, checkpointer=checkpointer)
    start_time = time.time()
    traj_learner.val_test()
    print(f"Inference time {(time.time() - start_time):.3f}s")