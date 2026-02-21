"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import argparse
import time
import torch
import numpy as np
from args import add_common_args,add_common_training_args
from learning.cvrp.data_iterator import DataIterator
from learning.cvrp.traj_learner import TrajectoryLearner
from utils.exp import setup_exp
from utils.misc import EpochMetrics
import os
import sys

class FilteredTrajectoryLearner(TrajectoryLearner):
    def val_test(self, what="test"):
        if what == "test":
            dataloader = self.data_iterator.test_trajectories
        else:
            dataloader = self.data_iterator.val_trajectories

        self.net.eval()
        epoch_metrics = EpochMetrics()
        factors = self.data_iterator.factors
        vrplib_names = self.data_iterator.names
        
        # Buckets for statistics
        buckets = {
            '0_1000': [],
            '1000_10000': [],
            '10000_100000': [],
            'total': []
        }
        
        skipped_count = 0
        total_instances = 0

        with torch.no_grad():
            for batch_num, data in enumerate(dataloader):
                total_instances += 1
                problem_name = vrplib_names[batch_num] if batch_num < len(vrplib_names) else f"Instance_{batch_num}"
                
                try:
                    problem_size = data['node_coords'].shape[1] - 1

                    val_test_metrics, execution_time, predicted_tour_lens, opt_value = self.get_minibatch_val_test_metrics(
                        data, factors[batch_num])

                    epoch_metrics.update(val_test_metrics)
                    
                    pred_cost = predicted_tour_lens.item()
                    optimal_cost = opt_value.item()
                    
                    if optimal_cost > 0:
                        gap = (pred_cost - optimal_cost) / optimal_cost * 100
                    else:
                        gap = 0.0

                    if gap > 100.0:
                        print(f"Skipping Instance: {problem_name}, Gap: {gap:.4f}% > 100%")
                        skipped_count += 1
                        continue

                    # Output per instance with Time
                    print(f"Instance: {problem_name}, Cost: {pred_cost:.2f}, Optimal: {optimal_cost:.2f}, Gap: {gap:.4f}%, Time: {execution_time:.4f}s")
                    
                    stats = [execution_time, pred_cost, gap]

                    buckets['total'].append(stats)
                    
                    if problem_size < 1000:
                        buckets['0_1000'].append(stats)
                    elif 1000 <= problem_size < 10000:
                        buckets['1000_10000'].append(stats)
                    elif 10000 <= problem_size:
                        buckets['10000_100000'].append(stats)
                        
                except Exception as e:
                    print(f"Error checking Instance: {problem_name}: {e}")
                    continue

        # Print Final Statistics
        print("\n" + "="*100)
        print(f"{'Bucket':<20} | {'Valid Count':<12} | {'Avg Time (s)':<15} | {'Avg Gap (%)':<15}")
        print("-" * 100)
        
        display_order = ['0_1000', '1000_10000', '10000_100000', 'total']
        
        for name in display_order:
            data_list = buckets[name]
            if len(data_list) > 0:
                arr = np.array(data_list)
                avg_time = np.mean(arr[:, 0])
                avg_gap = np.mean(arr[:, 2])
                count = len(data_list)
                print(f"{name:<20} | {count:<12} | {avg_time:<15.4f} | {avg_gap:<15.4f}")
            else:
                print(f"{name:<20} | {'0':<12} | {'N/A':<15} | {'N/A':<15}")
        
        print("-" * 100)
        print(f"Total Instances Processed: {total_instances}")
        print(f"Valid Instances (Gap <= 100%): {total_instances - skipped_count}")
        print(f"Skipped Instances (Gap > 100%): {skipped_count}")
        print("="*100 + "\n")

        return {f'{k}_{what}': v for k, v in epoch_metrics.get_means().items()}

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
    args.test_dataset = b + f'/data/CVRPlib_survey_instances_converted.txt'
    args.episode = 100
    args.test_batch_size = 1
    args.cuda_device_num = 0
    args.knns = 250
    args.beam_size = 1

    net, module, device, _, checkpointer, _ = setup_exp(args, "cvrp", True)

    data_iterator = DataIterator(args)

    traj_learner = FilteredTrajectoryLearner(args, net, module, device, data_iterator, checkpointer=checkpointer)
    start_time = time.time()
    traj_learner.val_test()
    print(f"Inference time {(time.time() - start_time):.3f}s")