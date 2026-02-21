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
from L2C_Insert.TSP.utils.utils import create_logger, copy_all_src
from L2C_Insert.TSP.Train.TSPTester import TSPTester as Tester

########### Frequent use parameters  ##################################################

problem_size = 100      # testing problem size
test_in_tsplib = False  # test in tsplib or not
Use_RRC = False          # decode method: use RRC or not (greedy)
RRC_budget = 1000         # RRC budget

########### model to load ###############

model_load_path = './result/20240817_232506_train'
model_load_epoch = 100

##########################################################################################
mode = 'test'
test_paras = {
    100: [ 'validation_TSP100.txt',2000,2000],
}

if test_in_tsplib == True:
    problem_size = 0
if not Use_RRC:
    RRC_budget = 0

##########################################################################################

b = os.path.abspath("..//..//..").replace('\\', '/')

env_params = {
    'mode': mode,
    'test_in_tsplib':test_in_tsplib,
    'tsplib_path':  b + f"/data/TSP/{test_paras[problem_size][0]}",
    'data_path':  b + f"/data/TSP/{test_paras[problem_size][0]}",
    'sub_path': False,
    'RRC_budget':RRC_budget
}


model_params = {
    'mode': 'test',
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'decoder_layer_num':9,
    'qkv_dim': 16,
    'head_num': 8,
    'ff_hidden_dim': 512,
    'knearest': False,
    'k_nearest_edges': 100,
    'k_nearest_scatter': 100,
    'coor_norm': False
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'test_episodes': test_paras[problem_size][1],
    'test_batch_size': test_paras[problem_size][2],
}


logger_params = {
    'log_file': {
        'desc': f'test__tsp{problem_size}_RRC{RRC_budget}',
        'filename': 'log.txt'
    }
}

##########################################################################################
# main

def main_test(epoch,path,use_RRC=None,cuda_num=None):
    if DEBUG_MODE:
        _set_debug_mode()
    if use_RRC is not None:
        env_params['RRC_budget'] = 0
    if cuda_num is not None:
        tester_params['cuda_device_num']=cuda_num
    create_logger(**logger_params)


    tester_params['model_load']={
        'path': path,
        'epoch': epoch,
    }

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)
    if cuda_num is None:
        copy_all_src(tester.result_folder)
    _print_config()
    score_optimal, score_student, gap = tester.run()
    return score_optimal, score_student,gap

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
    return score_optimal, score_student,gap

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

    path = model_load_path
    allin = []
    for i in [model_load_epoch]:
        score_optimal, score_student,gap = main_test(i,path)
        allin.append([ score_optimal, score_student,gap])
    np.savetxt('result.txt',np.array(allin),delimiter=',')
