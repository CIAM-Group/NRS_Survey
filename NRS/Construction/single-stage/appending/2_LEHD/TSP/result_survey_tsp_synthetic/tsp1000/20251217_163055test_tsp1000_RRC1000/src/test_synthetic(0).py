
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

# Path Config
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
import logging
import numpy as np
from utils.utils import create_logger, copy_all_src
from TSP.TSPTester import TSPTester as Tester

#############################################################

# testing problem size
problem_size = 100

# decode method: use RRC or not (greedy)
Use_RRC = True

# RRC budget
RRC_budget = 1000

model_load_path = 'result/20230509_153705_train'
model_load_epoch = 150

test_paras = {
    # problem_size: [filename, episode, batch]
    100: {"test_episodes": 10000, "test_batch_size": 10000},
    1000: {"test_episodes": 128, "test_batch_size": 128},
}

if not Use_RRC:
    RRC_budget = 0

mode = 'test'
test_episodes = test_paras[problem_size]["test_episodes"]
test_batch_size = test_paras[problem_size]["test_batch_size"]
##########################################################################################

b = os.path.abspath(".").replace('\\', '/')

env_params = {
    'mode': mode,
    'data_path': f'/public/home/bayp/exp_survey_202509/0_data_survey/survey_synthetic_tsp/test_tsp{problem_size}_n{test_episodes}_lkh.txt',

    'sub_path': False,
    'RRC_budget': RRC_budget
}

model_params = {
    'mode': mode,
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** (1 / 2),
    'decoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'ff_hidden_dim': 512,
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'test_episodes': test_episodes,
    'test_batch_size': test_batch_size,
}

logger_params = {
    'log_file': {
        'desc': f'test__tsp{problem_size}',
        'filename': 'log.txt'
    }
}

from datetime import datetime
import pytz
process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
logger_params = {
    'log_file': {
        'desc': f'test_tsp{problem_size}_RRC{RRC_budget}',
        'filename': 'run_log.txt',
        'filepath': f'./result_survey_tsp_synthetic/tsp{problem_size}/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
    }
}

##########################################################################################
# main

def main_test(epoch, path, use_RRC=None,cuda_device_num=None):
    if DEBUG_MODE:
        _set_debug_mode()
    if use_RRC is not None:
        env_params['RRC_budget'] = 0
    if cuda_device_num is not None:
        tester_params['cuda_device_num'] = cuda_device_num
    create_logger(**logger_params)
    _print_config()

    tester_params['model_load'] = {
        'path': path,
        'epoch': epoch,
    }

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    score_optimal, score_student, gap = tester.run()
    return score_optimal, score_student, gap


def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    score_optimal, score_student, gap = tester.run()
    return score_optimal, score_student, gap


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    # conda activate survey
    # cd ~/exp_survey_202509/Construction/single-stage/appending/2_LEHD/TSP
    # nohup python -u test_synthetic.py --problem_size 1000 --RRC_budget 1000 > log_tsp1000_rrc1000.txt 2>&1 &
    # nohup python -u test_synthetic.py --problem_size 1000 --RRC_budget 0 > log_tsp1000_greedy.txt 2>&1 &
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_size", type=int, default=1000, help="problem size of TSP")
    parser.add_argument("--RRC_budget", type=int, default=0, help="RRC budget, 0 means no RRC, i.e., greedy decoding")
    args = parser.parse_args()

    problem_size = args.problem_size
    RRC_budget = args.RRC_budget
    if RRC_budget == 0:
        Use_RRC = False
    else:
        Use_RRC = True

    if problem_size in test_paras:
        test_episodes = test_paras[problem_size]["test_episodes"]
        test_batch_size = test_paras[problem_size]["test_batch_size"]
    
    env_params['data_path'] = f'/public/home/bayp/exp_survey_202509/0_data_survey/survey_synthetic_tsp/test_tsp{problem_size}_n{test_episodes}_lkh.txt'
    env_params['RRC_budget'] = RRC_budget
    
    tester_params['test_episodes'] = test_episodes
    tester_params['test_batch_size'] = test_batch_size
    
    logger_params['log_file']['desc'] = f'test_tsp{problem_size}_RRC{RRC_budget}'
    logger_params['log_file']['filepath'] = f'./result_survey_tsp_synthetic/tsp{problem_size}/' + process_start_time.strftime("%Y%m%d_%H%M%S") + logger_params['log_file']['desc']

    path = model_load_path
    allin = []
    for i in [model_load_epoch]:
        score_optimal, score_student, gap = main_test(i, path)
        allin.append([score_optimal, score_student, gap])
    np.savetxt('result.txt', np.array(allin), delimiter=',')
