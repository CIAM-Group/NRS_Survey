##########################################################################################
# Machine Environment Config
import random

import numpy as np
import torch

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from TSPTester import TSPTester as Tester


##########################################################################################
# parameters
# need to be modified
distribution = "uniform" # "uniform", "clustered1" or "explosion" or "implosion"
problem_size = 10000
neighbors_num = 20
budget = 0
##########################################################################################
data_details= { 100: {'test_episodes': 10000},
               1000: {'test_episodes': 128},
               10000: {'test_episodes': 16},
              }
test_episodes = data_details[problem_size]['test_episodes']

test_batch_size = test_episodes
data_path = f'/public/home/bayp/exp_survey_202509/0_data_survey/survey_synthetic_tsp/test_tsp{problem_size}_n{test_episodes}_lkh.txt'

solution_filename = None
model_path_absolute = '/public/home/bayp/exp_survey_202509/Construction/single-stage/appending/7_L2R/000_Test_L2R_Adam_LN_TSP/result_tsp_models_L2R'
model_path = f"{model_path_absolute}/20250418_103804_rollout_tsp100_C10_Longtrain100"


reduction_percentage = 0.1

env_params = {
    'problem_size': problem_size,
    'lower_neighbors_num': neighbors_num,
    'reduction_percentage': reduction_percentage,
    'distribution': distribution,
    'test_in_tsplib': False,
    'repair_max_sub_length': 1000,
    'budget': budget,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'logit_clipping': 10,
    'decoder_layer_num': 6,
    'ff_hidden_dim': 512,
    'eval_type': 'greedy',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': model_path,
        # directory path of pre-trained model and log files saved.
        'epoch': 'best', #'best', 100
    },
    'test_episodes': test_episodes,
    'test_batch_size': test_batch_size,
    'test_data_load': {
        'enable': True,
        'filename': data_path,
        'solution_filename': solution_filename,
    },
}


##########################################################################################
from datetime import datetime
import pytz

process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))

epoch = tester_params['model_load']['epoch']
logger_params = {
    'log_file': {
        'desc':  f"L2R_test_{distribution}_epoch{epoch}_K{neighbors_num}_tsp{problem_size}_PRC{budget}",
        'filename': 'run_log.txt',
        'filepath': f'./result_survey_tsp_synthetic/tsp{problem_size}/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
    }
}

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    tester.run()


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

def seed_everything(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    # HASHSEED是 HASH映射函数时，保证每一次新的运行进程，对于一个相同的object生成的hash是相同的。
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。

    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子（只用一块GPU）
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed) # 为所有GPU设置随机种子（多块GPU）

##########################################################################################

if __name__ == "__main__":
    seed_everything(1234)
    main()
