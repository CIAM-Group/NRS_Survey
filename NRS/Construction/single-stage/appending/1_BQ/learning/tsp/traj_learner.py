"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import time
import torch
from torch import nn
from learning.tsp.decoding import decode
from utils.misc import do_lr_decay, EpochMetrics, get_opt_gap
import time

import numpy as np

DEBUG_NUM_BATCHES = 2

class TrajectoryLearner:

    def __init__(self, args, net, module, device, data_iterator, optimizer=None, checkpointer=None):
        # same supervisor is used for training and testing, during testing we do not have optimizer etc.

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
                node_coords_ordered, _, _ = self.prepare_batch(data)
                output_scores = self.net(node_coords_ordered)
                loss = self.loss(output_scores,
                                 torch.ones((output_scores.shape[0]), dtype=torch.long, device=output_scores.device))
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

            if val_metrics["opt_gap_val"] < self.best_current_val_metric:
                # monitoring on current trajectories, in order to see if we are training enough on them or not
                self.best_current_val_metric = val_metrics["opt_gap_val"]
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
        factors = self.data_iterator.factors
        self.net.eval()
        epoch_metrics = EpochMetrics()

        problems_lt_1000 = []
        problems_1000_10000 = []
        problems_10000_100000 = []
        problems_total = []
        
        def print_stats(name, lst):
            if len(lst) > 0:
                arr = np.array(lst).reshape(-1, 3)
                print(f'[{name}] count: {len(arr)}, mean time: {np.mean(arr[:, 0]):.3f}s, mean gap: {np.mean(arr[:, 2]):.4f}%')
            else:
                print(f'[{name}] count: 0, mean time: NaN s, mean gap: NaN %')

        with torch.no_grad():

            for batch_num, data in enumerate(dataloader):
                try:
                    # Try to get instance name
                    instance_name = "Unknown"
                    if 'name' in data:
                        instance_name = data['name'][0]
                    
                    problem_size = data['nodes_coord'].shape[1]-1

                    val_test_metrics, execution_time, predicted_tour_lens, opt_value = self.get_minibatch_val_test_metrics(data,factors[batch_num])
                    epoch_metrics.update(val_test_metrics)
                    
                    cost = predicted_tour_lens.item()
                    opt = opt_value.item()
                    gap = (cost - opt) / opt * 100
                    
                    # 1. Output per instance
                    print(f"Instance: {instance_name}, Dimension: {problem_size}, Cost: {cost:.2f}, Optimal: {opt:.2f}, Gap: {gap:.4f}%, Time: {execution_time:.3f}s")
                    
                    if batch_num == DEBUG_NUM_BATCHES and self.debug:
                        break
                    
                    # 2. Skip if gap > 100%
                    if gap > 100.0:
                        print(f"  -> Gap {gap:.2f}% > 100%, excluded from statistics.")
                    else:
                        result_entry = [execution_time, cost, gap]
                        problems_total.append(result_entry)

                        if 0 < problem_size < 1000:
                            problems_lt_1000.append(result_entry)
                        elif 1000 <= problem_size < 10000:
                            problems_1000_10000.append(result_entry)
                        elif 10000 <= problem_size <= 100000:
                            problems_10000_100000.append(result_entry)
                    
                    # 3. Print cumulative stats
                    print_stats('0-1000', problems_lt_1000)
                    print_stats('1000-10000', problems_1000_10000)
                    print_stats('10000-100000', problems_10000_100000)
                    print_stats('Total', problems_total)
                    print("-" * 50)

                except Exception as e:
                    print(f"Error processing batch {batch_num}, instance {instance_name if 'instance_name' in locals() else 'Unknown'}: {e}")
                    continue

        res = {f'{k}_{what}': v for k, v in epoch_metrics.get_means().items()}

        for k, v in res.items():
            print(k, f"{v:.3f}")

        return res

    def load_model(self, label, allow_not_exist=False):
        assert label in ["current", "current_FULL", "best"]
        self.checkpointer.load(self.module, None, label, allow_not_exist=allow_not_exist)

    def save_model(self, label, complete=False):
        assert label in ["current", "best"]
        args = {"module": self.module}
        if not complete:
            args_ = {"optimizer": None, 'label': label}
        else:
            assert not self.eval_only
            assert label == "current"
            args_ = {"optimizer": self.optimizer,
                     "label": label+"_FULL",
                     "other": {
                         "epoch_done": self.epoch_done,
                         "best_current_val_metric": self.best_current_val_metric,
                         "data_iterator": self.data_iterator}
                     }
        self.checkpointer.save(**args, **args_)

    def remove_model(self, label):
        assert label in ["current", "best"]
        self.checkpointer.delete(label)

    def get_minibatch_val_test_metrics(self, data,factor):
        metrics = {}
        # 开始计时
        start_time = time.time()
        # autoregressive decoding
        node_coords, dist_matrices, opt_value = self.prepare_batch(data, sample=False)
        decoding_metrics = {}
        predicted_tours, predicted_tour_lens =\
            decode(node_coords, dist_matrices, self.net, self.beam_size, self.knns)
        # 结束计时
        end_time = time.time()
        # 计算并打印执行时间
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds",'predicted_tour_lens',predicted_tour_lens,'factor:',factor)


        opt_gap = get_opt_gap(predicted_tour_lens, opt_value)
        decoding_metrics.update({"opt_gap": opt_gap})

        return {**metrics, **decoding_metrics}, execution_time, predicted_tour_lens, opt_value

    def prepare_batch(self, data, sample=True):
        ks = "_s" if sample else ""
        node_coords = data[f"nodes_coord{ks}"].to(self.device)
        dist_matrices = data[f"dist_matrices{ks}"].to(self.device)
        tour_len = data[f"tour_len{ks}"].to(self.device)

        return node_coords, dist_matrices, tour_len