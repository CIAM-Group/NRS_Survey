##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = True
CUDA_DEVICE_NUM = 3


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

import pytz
from datetime import datetime
import time

##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from TSP.TSPTester import TSPTester as Tester


##########################################################################################
# parameters

data_folder = 'data'
data_load_way = 'txt' # pt, txt
                     # if pt, load data from .pt file, you should set the 'pt_data_path' in env_params
                     # if txt, load data from .txt file, you should set the 'data_path' in env_params 
                     
# model_load =  {'path': './result/finetuned',
#                     'epoch': 120}
model_load =  {'path': './result/no_finetune',
                    'epoch': 100}
use_model = 'DRHG' # DRHG; DRHG_rp
                   # DRHG: default model
                   # DRHG_rp: you can set the number of repeated first node and last node

knn_k = [20, 100]
problem_size = 100  
BUDGET = 1000

test_paras = {
    # problem_size: [filename, episode, batch]
    100: {"test_episodes": 10000, "test_batch_size": 10000},
    1000: {"test_episodes": 128, "test_batch_size": 128},
    10000: {"test_episodes": 16, "test_batch_size": 16},
}
test_episodes = test_paras[problem_size]["test_episodes"]
test_batch_size = test_paras[problem_size]["test_batch_size"]

env_params = {
    'mode': 'test',
    'use_model': use_model, 
    'load_way': data_load_way,
    'data_path': f'/public/home/bayp/exp_survey_202509/Improvement/single_solution_based/large neighborhood/direct LNS (restricted)/3_DRHG/TSP/data/drhg_test_LKH3_pop_TSP10000_n16.txt',
    'test_in_tsplib':False,
    'tsplib_path': None,       
}



model_params = {
    'mode': 'test',
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
    # 'repeated_first_node': 8, # for DRHG_rp
    # 'repeated_last_node': 8, # for DRHG_rp
}


tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': model_load,
    # 'initial_solution_path': "./RI_solution/RI_{}.pt".format(problem_size),
    'test_episodes': test_episodes,   # 65
    'test_batch_size': test_batch_size,

    'destroy_mode': ['knn-location'],
    'destroy_params': { 
        'knn-location':{
            'center_type': 'equally', # random or equally; 
                                      # equally: precompute centers depending on budget, then sample from them
                                      # random: sample centers from the whole space 
            'knn_k': knn_k,
            },
        },
    'iter_budget': BUDGET,
    'coordinate_transform': True, # please do not change this

}



from datetime import datetime
import pytz
process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
logger_params = {
    'log_file': {
        'desc': f'test_tsp1000_10000_iter{tester_params["iter_budget"]}',
        'filename': 'run_log.txt',
        'filepath': f'./debug_result_survey_tsp_synthetic/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    logger = _print_config()
    begin = time.time()
    logger.info(f'================================ Starting Testing with Synthetic TSP{problem_size} ================================')
    logger.info(f'Testing synthetic TSP with problem size {problem_size}')
    logger.info(f'knn_k: {tester_params["destroy_params"]["knn-location"]["knn_k"]}')
    logger.info(f'Environment data path: {env_params["data_path"]}')
    logger.info(f'Using model from: {tester_params["model_load"]["path"]}, epoch: {tester_params["model_load"]["epoch"]}')
    logger.info(f'Start testing problem size {problem_size} with {test_episodes} episodes and batch size {test_batch_size}')

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    score_optimal, score_student, gap, _ = tester.run()

    end = time.time()
    logger.info('total time: {}s'.format(int(end-begin)))
    logger.info(f'================================ Finished Testing with Synthetic TSP{problem_size} ================================')
    return score_optimal, score_student,gap


def _set_debug_mode():
    global tester_params   
    tester_params['test_episodes'] = 8
    tester_params['test_batch_size'] = 4
    tester_params['iter_budget'] = 2

def _print_config():
    logger = logging.getLogger('root')
    logger.info('use model {}, epoch {}'.format(tester_params['model_load']['path'], tester_params['model_load']['epoch']))
    # logger.info('initial solution: {}'.format(tester_params['initial_solution_path']))
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

    return logger



##########################################################################################
# nohup python -u test_synthetic_drhg_no_finetune_debug.py > DRHG_test_synthetic_10000_debug.log 2>&1 &
if __name__ == "__main__":
    problem_size_list = [10000]
    for problem_size in problem_size_list:
        k_max = min(1000, problem_size)
        knn_k = [20, k_max]
        test_episodes = test_paras[problem_size]["test_episodes"]
        test_batch_size = test_paras[problem_size]["test_batch_size"]
        env_params['data_path'] = f'/public/home/bayp/exp_survey_202509/0_data_survey/survey_synthetic_tsp/test_tsp{problem_size}_n{test_episodes}_lkh.txt'

        # env_params['data_path'] = f'/public/home/bayp/exp_survey_202509/Improvement/single_solution_based/large neighborhood/direct LNS (restricted)/3_DRHG/TSP/data/drhg_test_LKH3_pop_TSP10000_n16.txt'
        tester_params['test_episodes'] = test_episodes
        tester_params['iter_budget'] = 1000
        tester_params['test_batch_size'] = test_batch_size
        tester_params['destroy_params']['knn-location']['knn_k'] = knn_k

        main()


