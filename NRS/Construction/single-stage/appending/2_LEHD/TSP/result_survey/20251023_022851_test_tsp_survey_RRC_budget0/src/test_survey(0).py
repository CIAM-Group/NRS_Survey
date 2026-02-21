##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0
##########################################################################################
# Path Config
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
##########################################################################################
# import
import logging
import numpy as np
from utils.utils import create_logger, copy_all_src
from TSP.TSPTester_inTSPlib import TSPTester as Tester

########### Frequent use parameters  ##################################################

problem_size = 100      # testing problem size
test_in_tsplib = True  # test in tsplib or not
Use_RRC = False          # decode method: use RRC or not (greedy)
RRC_budget = 0         # RRC budget

########### model to load ###############

model_load_path = 'result/20230509_153705_train'
model_load_epoch = 150

# ! === 新增：TSPLIB 遍历与分桶控制（和 ICAM 一致，可自定义） ===
tsplib_dir = '/public/home/bayp/exp_survey_202509/0_data_survey/survey_bench_tsp'
# 例：与 ICAM 一致的三档（默认）
scale_ranges = [[0, 1000], [1000, 10000], [10000, 100001]]
# scale_ranges = [[0, 1000]]

# 例：只测某一小段（可临时改）
# scale_ranges = [[70, 71]]

detailed_log = True

##########################################################################################
mode = 'test'
test_paras = {
   # problem_size: [filename, episode, batch]
    100: [ 're_generate_test_TSP100_0423_n1w.txt',10000,10000],
    200: ['re_generate_test_TSP200_0423_n128.txt', 128, 128],
    500: ['re_generate_test_TSP500_0423_n128.txt', 128, 128],
    1000: ['re_generate_test_TSP1000_0423_n128.txt', 128, 128],
    0: ['TSPlib_70instances.txt', 205, 1]
}

if test_in_tsplib == True:
    problem_size = 0
if not Use_RRC:
    RRC_budget = 0

##########################################################################################

b = os.path.abspath(".").replace('\\', '/')

env_params = {
    'mode': mode,
    'test_in_tsplib':test_in_tsplib,
    'tsplib_path':  b + f"/data/{test_paras[problem_size][0]}",
    'data_path':  b + f"/data/{test_paras[problem_size][0]}",
    'sub_path': False,
    'RRC_budget':RRC_budget
}


model_params = {
    'mode': mode,
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'decoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'ff_hidden_dim': 512,
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'test_episodes': test_paras[problem_size][1],
    'test_batch_size': test_paras[problem_size][2],

    # ! === 新增：TSPLIB 遍历与分桶配置 ===
    'tsplib_dir': tsplib_dir,
    'scale_ranges': scale_ranges,
    'detailed_log': detailed_log,
}

tester_params['model_load'] = {
    'path': model_load_path,
    'epoch': model_load_epoch,
}


logger_params = {
    'log_file': {
        'desc': f'test_tsp_survey_RRC_budget{RRC_budget}',
        'filename': 'log.txt'
    }
}

##########################################################################################
# main

def main_test(epoch,path,use_RRC=None):
    if DEBUG_MODE:
        _set_debug_mode()
    if use_RRC is not None:
        env_params['RRC_budget'] = 0

    create_logger(**logger_params)
    _print_config()

    tester_params['model_load']={
        'path': path,
        'epoch': epoch,
    }

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    score_optimal, score_student, gap = tester.run()
    return score_optimal, score_student,gap

# def main():
#     if DEBUG_MODE:
#         _set_debug_mode()

#     create_logger(**logger_params)
#     _print_config()


#     tester = Tester(env_params=env_params,
#                     model_params=model_params,
#                     tester_params=tester_params)

#     copy_all_src(tester.result_folder)

#     score_optimal, score_student, gap = tester.run()
#     return score_optimal, score_student,gap

def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

# if __name__ == "__main__":

#     path = model_load_path
#     allin = []
#     for i in [model_load_epoch]:
#         score_optimal, score_student,gap = main_test(i,path)
#         allin.append([ score_optimal, score_student,gap])
#     np.savetxt('result.txt',np.array(allin),delimiter=',')

if __name__ == "__main__":
    create_logger(**logger_params)  # 若未创建
    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)
    copy_all_src(tester.result_folder)
    tester.run_lib()  # <<< 新增的入口