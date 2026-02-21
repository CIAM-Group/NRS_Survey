##########################################################################################
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = None

##########################################################################################
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")  # for problem_def
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
sys.path.insert(0, "../../..")  # for utils
sys.path.insert(0, "../../../..")  # for utils
sys.path.insert(0, "../../../../..")  # for utils
##########################################################################################
import logging
import numpy as np
from L2C_Insert.CVRP.utils.utils import create_logger, copy_all_src
from L2C_Insert.CVRP.Test.Tester import VRPTester as Tester

import argparse

##########################################################################################
# parameters

problem_size = 100  # testing problem size
test_in_vrplib = False  # test in vrplib or not
Use_RRC = True  # decode method: use RRC or not (greedy)
RRC_budget = 0  # RRC budget
RRC_range = 100
########### model ###############

model_load_path = './result/pretrain/cvrp_model.pt'

knearest = True
k_nearest_nodes = 200
coor_norm = True
random_insertion = False
if test_in_vrplib == True:
    problem_size = 0

if not Use_RRC:
    RRC_budget = 0

mode = 'test'
test_paras = {
    # problem_size: [filename, episode, batch]
    # 100: ['vrp100_test_lkh.txt', 10000, 5000, 0],
    100: ['test_vrp100_n10000_C50_uniform_hgs.txt', 10000, 5000, 0],
    1000: ['test_vrp1000_n16_C200_uniform_hgs.txt', 100, 100, 0],
    10000: ['test_vrp10000_n16_C300_uniform_hgs.txt', 16, 16, 0],
    50000: ['test_vrp50000_n16_C300_uniform_hgs.txt', 16, 16, 0],
    100000: ['test_vrp100000_n16_C300_uniform_hgs.txt', 16, 16, 0],
}

##########################################################################################
# parameters
b = os.path.abspath("../../..").replace('\\', '/')

env_params = {
    'mode': mode,
    'test_in_vrplib': test_in_vrplib,
    'vrplib_path': b + f'/data/CVRP/{test_paras[problem_size][0]}',
    'data_path': b + f"/data/CVRP/{test_paras[problem_size][0]}",
    'sub_path': False,
    'RRC_budget': RRC_budget,
    'RRC_range': RRC_range,
    'random_insertion': random_insertion
}

model_params = {
    'mode': mode,
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** (1 / 2),
    'decoder_layer_num': 9,
    'qkv_dim': 16,
    'head_num': 8,
    'ff_hidden_dim': 512,
    'knearest': knearest,
    'k_nearest_edges': 100,
    'k_nearest_scatter': 100,
    'coor_norm': coor_norm
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'begin_index': test_paras[problem_size][3],
    'test_episodes': test_paras[problem_size][1],  # 65
    'test_batch_size': test_paras[problem_size][2],
}

logger_params = {
    'log_file': {
        'desc': f'test__vrp{problem_size}_RRC{RRC_budget}',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main_test(path, args, file_name, use_RRC=None):
    if DEBUG_MODE:
        _set_debug_mode()

    tester_params['model_load'] = {
        'path': path,
    }
    if use_RRC is not None:
        env_params['RRC_budget'] = 0

    logger_params['log_file'][
        'desc'] = file_name

    tester_params['cuda_device_num'] = args.cuda_device_num
    tester_params['test_episodes'] = test_paras[args.problem_size][1]
    tester_params['test_batch_size'] = test_paras[args.problem_size][2]
    tester_params['begin_index'] = test_paras[args.problem_size][3]
    model_params['k_nearest_edges'] = args.k_nearest_edges
    model_params['k_nearest_scatter'] = args.k_nearest_scatter
    model_params['knearest'] = args.knearest
    model_params['coor_norm'] = args.coor_norm
    env_params['data_path'] = b + f"/data/CVRP/{test_paras[args.problem_size][0]}"
    env_params['vrplib_path'] = b + f"/data/CVRP/{test_paras[args.problem_size][0]}"
    env_params['test_in_vrplib'] = args.test_in_vrplib
    env_params['RRC_budget'] = args.RRC_budget
    env_params['random_insertion'] = args.random_insertion
    env_params['RRC_range'] = args.RRC_range

    create_logger(**logger_params)
    _print_config()

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


def add_common_args(parser):
    parser.add_argument("--cuda_device_num", type=int, default=0, help="None")
    parser.add_argument("--problem_size", type=int, default=200, help="None")
    parser.add_argument("--test_in_vrplib", type=int, default=0, help="None")
    parser.add_argument("--RRC_budget", type=int, default=0, help="None")
    parser.add_argument("--RRC_range", type=int, default=100, help="None")
    parser.add_argument("--random_insertion", type=int, default=0, help="None")
    parser.add_argument("--knearest", type=int, default=1, help="None")
    parser.add_argument("--k_nearest_nodes", type=int, default=250, help="None")
    parser.add_argument("--coor_norm", type=int, default=0, help="None")
    parser.add_argument("--counter_current", type=int, default=0, help="None")


##########################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='test')
    add_common_args(parser)
    args = parser.parse_args()


    cuda_num = 0
    problem_scales = [1000, 10000, 50000, 100000, 100] # 1000, 10000, 50000, 100000, 100

    RRC_budgets = [1000]
    RI_inites = [0]
    coords_norms = [0]
    knearest_ifs = [1]
    rrc_ranges = [300]
    k_nearest_edge_nums = [200]
    k_nearest_scatter_nums = [200]

    test_in_tsplib_if = 0

    file_name = f'scales={str(problem_scales)}, RRC_budgets={str(RRC_budgets)}, RI_inites={str(RI_inites)},' \
                f' coords_norms={str(coords_norms)}, knearest={str(knearest_ifs)}, rrc_ranges={str(rrc_ranges)},' \
                f' k_edge_nums={str(k_nearest_edge_nums)}, k_scatter_nums={str(k_nearest_scatter_nums)}'

    for problem_scale in problem_scales:


        for RI_init in RI_inites:
            for knearest_if in knearest_ifs:
                for k_nearest_edge_num in k_nearest_edge_nums:
                    for k_nearest_scatter_num in k_nearest_scatter_nums:
                        for rrc_budget in RRC_budgets:
                            for rrc_range in rrc_ranges:
                                for coords_norm in coords_norms:


                                    args.cuda_device_num = cuda_num
                                    args.problem_size = problem_scale
                                    args.random_insertion = RI_init
                                    args.RI_init = RI_init

                                    args.knearest = knearest_if

                                    if problem_scale == 100:
                                        args.knearest = 0

                                    args.k_nearest_edges = k_nearest_edge_num
                                    args.k_nearest_scatter = k_nearest_scatter_num
                                    args.coor_norm = coords_norm

                                    if problem_scale >= 2000:
                                        args.coor_norm = 1

                                    args.test_in_tsplib = test_in_tsplib_if

                                    args.RRC_budget = rrc_budget
                                    args.RRC_range = rrc_range

                                    path = f'./{model_load_path}'

                                    score_optimal, score_student, gap = main_test( path, args,file_name)



