"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import time
import torch
from torch import nn
from learning.cvrp.decoding import decode
from utils.misc import do_lr_decay, EpochMetrics, get_opt_gap
import numpy as np
DEBUG_NUM_BATCHES = 3

class TrajectoryLearner:

    def __init__(self, args, net, module, device, data_iterator, optimizer=None, checkpointer=None):
        # same supervisor is used for training and testing, during testing we do not have optimizer, mlflow etc.

        self.net = net
        self.module = module
        self.device = device
        self.knns = args.knns
        self.beam_size = args.beam_size
        self.data_iterator = data_iterator
        self.optimizer = optimizer
        self.checkpointer = checkpointer

        self.output_dir = args.output_dir
        self.test_only = args.test_only

        self.debug = args.debug

        if not args.test_only:
            try:
                self.test_every = args.test_every if args.test_every > 0 else None
            except AttributeError:
                self.test_every = None
            self.decay_rate = args.decay_rate
            self.decay_every = args.decay_every
            self.loss = nn.CrossEntropyLoss()
            self.best_current_val_metric = float('inf')
            self.epoch_done = 0
            self.nb_epochs = args.nb_total_epochs

    def train(self):
        assert not self.test_only

        for _ in range(self.nb_epochs):
            # Train one epoch
            start = time.time()
            self.net.train()
            epoch_metrics_train = EpochMetrics()

            for batch_num, data in enumerate(self.data_iterator.train_trajectories):

                self.optimizer.zero_grad()
                node_coords, _, demands, capacities, remaining_capacities, via_depots, _ = self.prepare_batch(data)
                inputs = torch.cat([node_coords, (demands / capacities.unsqueeze(-1)).unsqueeze(-1),
                                    (remaining_capacities / capacities).unsqueeze(-1).repeat(1, node_coords.shape[1]).unsqueeze(-1)], dim=-1)
                output_scores = self.net(inputs, demands=demands, remaining_capacities=remaining_capacities)

                ground_truth = torch.full((output_scores.shape[0], ), 2, dtype=torch.long, device=output_scores.device)
                # update ground truth for edges via depot
                ground_truth[via_depots[:, 1] == 1.] += 1
                loss = self.loss(output_scores, ground_truth)
                epoch_metrics_train.update({"training loss": loss})
                loss.backward()
                self.optimizer.step()
                if batch_num == DEBUG_NUM_BATCHES and self.debug:
                    break

            metrics = {f'{k}_train': v for k, v in epoch_metrics_train.get_means().items()}

            print("[EPOCH {:03d}] Time: {:.3f}s ".format(self.epoch_done, time.time() - start))
            for k, v in metrics.items():
                print(k, f"{v:.5f}")

            # Val test
            val_metrics = self.val_test("val")

            if val_metrics['opt_gap_val'] < self.best_current_val_metric:
                self.best_current_val_metric = val_metrics['opt_gap_val']
                self.checkpointer.save(self.module, None, 'best')  # only model

            # test
            if self.test_every is not None:
                if self.epoch_done % self.test_every == 0:
                    self.save_model("current")
                    self.load_model("best")
                    self.val_test("test")
                    self.load_model("current")
                    self.remove_model("current")

            # lr decay
            if self.epoch_done % self.decay_every == 0:
                do_lr_decay(self.optimizer, self.decay_rate)

            self.epoch_done += 1

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

        with torch.no_grad():

            for batch_num, data in enumerate(dataloader):
                problem_name = vrplib_names[batch_num] if batch_num < len(vrplib_names) else f"Instance_{batch_num}"
                
                try:
                    problem_size = data['node_coords'].shape[1] - 1
                    # print('problem_size:',problem_size, problem_name)

                    val_test_metrics, execution_time, predicted_tour_lens, opt_value = self.get_minibatch_val_test_metrics(
                        data, factors[batch_num])

                    # val_test_metrics = self.get_minibatch_val_test_metrics(data)
                    epoch_metrics.update(val_test_metrics)
                    
                    # Calculate Stats
                    # If we are using exact distance matrix (from original coords), predicted_tour_lens is already Real Cost.
                    # We should check if we need to apply factor or not.
                    # Since we updated DataSet to use Real Coords for Distance Matrix calculation, predicted_tour_lens is Real.
                    # So we ignore factor multiplication here.
                    
                    pred_cost = predicted_tour_lens.item()
                    optimal_cost = opt_value.item()
                    
                    if optimal_cost > 0:
                        gap = (pred_cost - optimal_cost) / optimal_cost * 100
                    else:
                        gap = 0.0 # Handle case where optimal might be 0 (unlikely for CVRP but good for safety)
                    
                    # Output per instance
                    print(f"Instance: {problem_name}, Cost: {pred_cost:.2f}, Optimal: {optimal_cost:.2f}, Gap: {gap:.4f}%")
                    
                    stats = [execution_time, pred_cost, gap]

                    # Add to buckets
                    buckets['total'].append(stats)
                    
                    if problem_size < 1000:
                        buckets['0_1000'].append(stats)
                    elif 1000 <= problem_size < 10000:
                        buckets['1000_10000'].append(stats)
                    elif 10000 <= problem_size:
                        buckets['10000_100000'].append(stats)
                        
                except Exception as e:
                    print(f"Skipping Instance: {problem_name} due to error: {e}")
                    continue

                if batch_num == DEBUG_NUM_BATCHES and self.debug:
                    break

        # Print Final Statistics
        print("\n" + "="*80)
        print(f"{'Bucket':<20} | {'Count':<10} | {'Avg Time (s)':<15} | {'Avg Gap (%)':<15}")
        print("-" * 80)
        
        # Order of display
        display_order = ['0_1000', '1000_10000', '10000_100000', 'total']
        
        for name in display_order:
            data_list = buckets[name]
            if len(data_list) > 0:
                arr = np.array(data_list)
                # arr columns: 0:time, 1:cost, 2:gap
                avg_time = np.mean(arr[:, 0])
                avg_gap = np.mean(arr[:, 2])
                count = len(data_list)
                print(f"{name:<20} | {count:<10} | {avg_time:<15.4f} | {avg_gap:<15.4f}")
            else:
                print(f"{name:<20} | {'0':<10} | {'N/A':<15} | {'N/A':<15}")
        print("="*80 + "\n")

        res = {f'{k}_{what}': v for k, v in epoch_metrics.get_means().items()}

        # for k, v in res.items():
        #     print(k, f"{v:.3f}")

        return res

    def load_model(self, label, allow_not_exist=False):
        assert label in ['current', 'current_FULL', 'best']
        self.checkpointer.load(self.module, None, label, allow_not_exist=allow_not_exist)

    def save_model(self, label, complete=False):
        assert label in ['current', 'best']
        args = {'module': self.module}
        if not complete:
            args_ = {'optimizer': None, 'label': label}
        else:
            assert not self.eval_only
            assert label == 'current'
            args_ = {'optimizer': self.optimizer,
                     'label': label+'_FULL',
                     'other': {
                         'epoch_done': self.epoch_done,
                         'best_current_val_metric': self.best_current_val_metric,
                         'data_iterator': self.data_iterator}
                     }
        self.checkpointer.save(**args, **args_)

    def remove_model(self, label):
        assert label in ['current', 'best']
        self.checkpointer.delete(label)

    def get_minibatch_val_test_metrics(self, data, factor):

        metrics = {}
        start_time = time.time()
        # autoregressive decoding
        node_coords, dist_matrices, demands, capacities, _, _, tour_lens = self.prepare_batch(data, sample=False)
        decoding_metrics = {}
        predicted_tour_lens, _ = decode(node_coords, dist_matrices, demands, capacities, self.net, self.beam_size, self.knns)
        end_time = time.time()
        execution_time = end_time - start_time
        # print(f"Execution time: {execution_time} seconds", 'predicted_tour_lens', predicted_tour_lens * factor, 'optimal_len',tour_lens,
        #       'factor:', factor)

        # Assuming predicted_tour_lens is now Real Cost because dist_matrices is Real.
        opt_gap = get_opt_gap(predicted_tour_lens, tour_lens)
        decoding_metrics.update({'opt_gap': opt_gap})

        # Return predicted_tour_lens without factor (assuming it is already Real)
        return {**metrics, **decoding_metrics}, execution_time, predicted_tour_lens, tour_lens

    def prepare_batch(self, data, sample=True):
        ks = '_s' if sample else ''
        node_coords = data[f"node_coords{ks}"].to(self.device)
        distance_matrices = data[f"dist_matrices{ks}"].to(self.device)
        demands = data[f"demands{ks}"].to(self.device)
        capacities = data[f"capacities{ks}"].to(self.device)
        remaining_capacities = data[f"remaining_capacities{ks}"].to(self.device)
        via_depots = data[f"via_depots{ks}"].to(self.device)
        tour_len = data[f"tour_len{ks}"].to(self.device)

        return node_coords, distance_matrices, demands, capacities, remaining_capacities, via_depots, tour_len
