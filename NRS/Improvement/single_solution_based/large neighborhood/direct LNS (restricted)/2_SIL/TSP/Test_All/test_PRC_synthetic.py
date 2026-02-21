##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 3

##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
sys.path.insert(0, "../../..")  # for utils
sys.path.insert(0, "../../../..")  # for utils
sys.path.insert(0, "../../../../..")  # for utils

##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from TSP.Test_All.TSPTester_Synthetic import TSPTester as Tester

##########################################################################################
# parameters

b = os.path.abspath(".").replace('\\', '/')
b2 = os.path.abspath("..").replace('\\', '/')

env_params = {
    # 'problem_size': 100,
    'pomo_size': 1,
    'k_nearest': 1,
    'beam_width': 16,
    'decode_method': 'greedy',
    'mode': 'test',
    'test_in_tsplib': False,
    'tsplib_path': None,
    'data_path': None,
    'load_way': 'allin',
    'sub_path': False,
    'budget': 10,
    'PRC': True,
    'repair_max_sub_length': 1000,
    'random_insertion': False
}

model_params = {
    'mode': 'test',
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** (1 / 2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
    'use_k_nearest': True,
    'k_nearest_num':1000
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': None,  # directory path of pre-trained model and log files saved.
    },
    'test_episodes': 16,  # 65
    'test_batch_size': 4,
}


from datetime import datetime
import pytz
process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
logger_params = {
    'log_file': {
        'desc': f'test_RandomInsertion_PRC0',
        'filename': 'run_log.txt',
        'filepath': f'./result_survey_tsp_synthetic/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
    }
}


##########################################################################################
# main

def main_test(path, prob_size, PRC, repair_max_sub_length, budget, random_insertion):
    logger_params['log_file']['desc']=f'test_{prob_size}_RandomInsertion_{random_insertion}_PRC_{PRC}_Budget{budget}'

    create_logger(**logger_params)
    tester_params['model_load'] = {
        'path': path,
    }

    tester_params['test_episodes'] = datas[prob_size][0]
    tester_params['test_batch_size'] = datas[prob_size][1]
    env_params['data_path'] = f'/public/home/bayp/exp_survey_202509/0_data_survey/survey_synthetic_tsp/test_tsp{prob_size}_n{tester_params["test_episodes"]}_lkh.txt'

    env_params['PRC'] = PRC
    env_params['repair_max_sub_length'] = repair_max_sub_length
    env_params['budget'] = budget
    env_params['random_insertion'] = random_insertion

    # ! TSPlib test is not used in this script
    if prob_size==-1:
        env_params['test_in_tsplib'] = True
        env_params['tsplib_path'] = b2 + datas[prob_size][0]
        env_params['data_path'] = b2 + datas[prob_size][0]


    if not random_insertion and budget==0: # (purely greedy search)
        model_params['use_k_nearest'] = False

    _print_config()
    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    score_optimal, score_student, gap = tester.run()
    return score_optimal, score_student, gap



def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":


    datas = {
        100: [10000, 10000],
        1000: [128, 128],
        10000: [16, 16],
        # 50000: ['/data/test_set/test_tsp50000_lkh3_n16.txt', 16, 8],
        # 100000: ['/data/test_set/test_tsp100000_lkh3_n16.txt', 16, 4],
        #-1:["/data/test_set/TSPlib_scale_ge_1K_n33_ascending.txt", 205, 1, 0]
    }
    

    model_path = {
        100:   './result/checkpoint-tsp1k.pt',
        1000:   './result/checkpoint-tsp1k.pt',
        5000:   './result/checkpoint-tsp5k.pt',
        10000:  './result/checkpoint-tsp10k.pt',
        50000:  './result/checkpoint-tsp50k.pt',
        100000: './result/checkpoint-tsp100k.pt',
        -1:     './result/checkpoint-tsp1k.pt',
    }

    PRC = True # True or False
    repair_max_sub_length = 1000

    # purely greedy search: random_insertion=False, budget=0
    # Initilize the solution by random insertion and refine it by PRC: random_insertion=True, budget>0
    budget = 1000
    random_insertion = True

    # for prob_size in [1000, 5000, 10000, 50000, 100000, -1]:
    # nohup python -u test_PRC_synthetic.py > SIL_test_PRC1000_synthetic.txt 2>&1 &
    for prob_size in [1000, 10000]:
        score_optimal, score_student, gap = main_test(
                                                      model_path[prob_size],
                                                      prob_size, PRC, repair_max_sub_length,
                                                      budget, random_insertion)
