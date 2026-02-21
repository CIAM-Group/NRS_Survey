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

from CVRPTester_LIB_Survey import CVRPTester as Tester


##########################################################################################
# parameters
# need to be modified
CVRPLIB_MODE = True
lib_norm = 'invit_norm' # 'unified_norm' or 'separate_norm',invit_norm
neighbors_num = 50
budget = 0
##########################################################################################
test_episodes = 1
test_batch_size = 1

# cvrplib_path = '/public/home/bayp/exp_survey_202509/0_data_survey/cvrp_test'
cvrplib_path = '/public/home/bayp/exp_survey_202509/0_data_survey/survey_bench_cvrp'

# model_path_absolute = '/public/home/zhoucl/000_NCO_Codes_v20241213/NeurIPS2025_L2R_Revised/CVRP/v0_C50_TwoMask/result_cvrp_models_L2R'
# model_path = f"{model_path_absolute}/20250422_110455_rollout_Adam_FC50_LD_Upper98_Lower98_NoReLD_ComTrain_weighted1_UpperLowerAlpha_reduction10_k50_cvrp100_bs60_Longtrain100"
model_path_absolute = '/public/home/bayp/exp_survey_202509/Construction/single-stage/appending/7_L2R/000_CVRP_Test_L2R_FixedC50_TwoMask/result_cvrp_models_L2R'
model_path = f"{model_path_absolute}/20250422_110455_rollout_Adam_FC50_LD_Upper98_Lower98_NoReLD_ComTrain_weighted1_UpperLowerAlpha_reduction10_k50_cvrp100_bs60_Longtrain100"


env_params = {
    'problem_size': None,
    'lower_neighbors_num': neighbors_num,
    'reduction_percentage': 0.1,
    'test_in_vrplib': CVRPLIB_MODE,
    'lib_norm': lib_norm,
    'cvrplib_path': cvrplib_path,
    #'draw_pic': False,
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
        'epoch': 'best',
    },
    'test_episodes': test_episodes,
    'test_batch_size': test_batch_size,
    'test_data_load': {
        'enable': False,
        'filename': None,
    },
}


##########################################################################################
from datetime import datetime
import pytz

process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
epoch = tester_params['model_load']['epoch']
logger_params = {
    'log_file': {
        'desc':  f"L2R_test_epoch{epoch}_K{neighbors_num}_{lib_norm}",
        'filename': 'run_log.txt',
        'filepath': './result_cvrp_models_Test_Survey/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
    }
}

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()
    if CVRPLIB_MODE:
        _set_CVRPLIB_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)
    if CVRPLIB_MODE:
        tester.run_lib()
    # else:
    #     tester.run()

def _set_CVRPLIB_mode():
    global tester_params
    tester_params['test_episodes'] = 1
    tester_params['test_batch_size'] = 1
    tester_params['test_data_load']['enable'] = False
    env_params['cvrplib_path'] = cvrplib_path


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
    seed_everything(seed=1234)
    main()
