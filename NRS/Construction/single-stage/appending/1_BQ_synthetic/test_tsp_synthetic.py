"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import argparse
import time
from args import add_common_args,add_common_training_args
from learning.tsp.data_iterator import DataIterator
from learning.tsp.traj_learner import TrajectoryLearner
from utils.exp import setup_exp

# nohup python -u test_tsp_synthetic.py > bq_test_tsp_synthetic_100_1000.log 2>&1 &
if __name__ == "__main__":
    problem_size_list = [100, 1000]
    for problem_size in problem_size_list:
        start_time = time.time()
        parser = argparse.ArgumentParser(description='test_tsp')
        add_common_args(parser)  # (only need common args)
        args = parser.parse_args()
        
        # 修改args
        batch_dict = {
            100: 10000,
            1000: 128,
        }
        # batch_bs_dict = {
        #     100: 256,
        #     200: 16,
        #     500: 16,
        #     1000: 16,
        # }
        episode_dict = {
            100: 10000,
            1000: 128,
        }
        test_episodes = episode_dict[problem_size]
        test_batch_size = batch_dict[problem_size]
        args.test_dataset = f'/public/home/bayp/exp_survey_202509/0_data_survey/survey_synthetic_tsp/test_tsp{problem_size}_n{test_episodes}_lkh.txt'
        args.pretrained_model = '/public/home/bayp/exp_survey_202509/Construction/single-stage/appending/1_BQ/pretrained_models/tsp.best'
        args.episode = test_episodes
        args.test_batch_size = test_batch_size
        args.knns = -1 if problem_size < 1000 else 250
        args.cuda_device_num = 0
        # print all args
        for k, v in vars(args).items():
            print(f"{k}: {v}")

        net, module, device, _, checkpointer, _ = setup_exp(args, is_test=True)

        data_iterator = DataIterator(args)

        traj_learner = TrajectoryLearner(args, net, module, device, data_iterator, checkpointer=checkpointer)
        # (for eval, no need for optimizer, watcher, checkpointer)

        # start_time = time.time()
        traj_learner.val_test()
        print(f"Inference time {(time.time() - start_time):.3f}s")
        print('Inference time: {}mins'.format((time.time() - start_time)/60))
        print('episode: {}'.format(args.episode))
        print('beam_size: {}'.format(args.beam_size))
        print('batch_size: {}'.format(args.test_batch_size))
        print(f"Inference time {(time.time() - start_time):.3f}s")
