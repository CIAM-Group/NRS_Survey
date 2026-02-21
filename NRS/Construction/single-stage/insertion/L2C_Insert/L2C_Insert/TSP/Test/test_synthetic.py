##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = None
##########################################################################################
# Path Config
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
sys.path.insert(0, "../../..")  # for utils
##########################################################################################
# import
import logging
import numpy as np
from L2C_Insert.TSP.utils.utils import create_logger, copy_all_src
from L2C_Insert.TSP.Test.TSPTester_repair_synthetic import TSPTester as Tester
import argparse



def add_common_args(parser):
    parser.add_argument("--cuda_device_num", type=int, default=0, help="None")
    parser.add_argument("--problem_size", type=int, default=500, help="None")
    parser.add_argument("--test_in_tsplib", type=int, default=0, help="None")
    parser.add_argument("--RRC_budget", type=int, default=0, help="None")
    parser.add_argument("--RRC_range", type=int, default=100, help="None")
    parser.add_argument("--random_insertion", type=int, default=0, help="None")
    parser.add_argument("--knearest", type=int, default=0, help="None")
    parser.add_argument("--k_nearest_edges", type=int, default=100, help="None")
    parser.add_argument("--k_nearest_scatter", type=int, default=100, help="None")
    parser.add_argument("--coor_norm", type=int, default=0, help="None")
    parser.add_argument("--counter_current", type=int, default=0, help="None")


########### Frequent use parameters  ##################################################

problem_size = 100      # testing problem size

model_load_path = './result/pretrain/tsp_model.pt'

Use_RRC = False          # decode method: use RRC or not (greedy)
RRC_budget = 1000       # RRC budget

# test_paras = {
#    # problem_size: [filename, episode, batch]
#     100: [ 'test_TSP100_n10000.txt', 10000, 5000],
#     1000: ['MCTS_tsp1000_test_concorde.txt', 128, 128],
#     10000: ['MCTS_tsp10000_test_concorde.txt', 16, 16],
#     50000: ['test_tsp50000_lkh3_n16.txt', 16, 16],
#     100000: ['test_tsp100000_lkh3_n16.txt', 16, 16],

# }
test_paras = {
    # problem_size: [episode, batch]
    100: {"test_episodes": 10000, "test_batch_size": 10000},
    1000: {"test_episodes": 128, "test_batch_size": 128},
    10000: {"test_episodes": 16, "test_batch_size": 16},
}
test_episodes = test_paras[problem_size]["test_episodes"]
test_batch_size = test_paras[problem_size]["test_batch_size"]

mode = 'test'
test_in_tsplib = False  # test in tsplib or not
mix_sample_strategy = False
turn_to_cluster_strategy = True

if test_in_tsplib == True:
    problem_size = 0
if not Use_RRC:
    RRC_budget = 0


##########################################################################################

b = os.path.abspath("../../..").replace('\\', '/')

env_params = {
    'mode': mode,
    'test_in_tsplib':test_in_tsplib,
    'tsplib_path':  None, #b + f"/data/{test_paras[problem_size][0]}",
    'data_path':  f'/public/home/bayp/exp_survey_202509/0_data_survey/survey_synthetic_tsp/test_tsp{problem_size}_n{test_episodes}_lkh.txt',
    'sub_path': False,
    'RRC_budget':RRC_budget,
    'max_RRC_range':200,
    'mix_sample_strategy':mix_sample_strategy,
    'turn_to_cluster_strategy':turn_to_cluster_strategy,
    'random_insertion': False
}


model_params = {
    'mode': mode,
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'decoder_layer_num':9,
    'qkv_dim': 16,
    'head_num': 8,
    'ff_hidden_dim': 512,
    'knearest': True,
    'k_nearest_edges': 100,
    'k_nearest_scatter': 100,
    'coor_norm': False
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'test_episodes': test_episodes,
    'test_batch_size': test_batch_size,
}


from datetime import datetime
import pytz
process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
logger_params = {
    'log_file': {
        'desc': f'test_tsp100_1000_10000_RRC{RRC_budget}',
        'filename': 'run_log.txt',
        'filepath': f'./result_survey_tsp_synthetic/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
    }
}

##########################################################################################
# main

def main_test(path,args,file_name,use_RRC=None,cuda_num=None):
    if DEBUG_MODE:
        _set_debug_mode()
    if use_RRC is not None:
        env_params['RRC_budget'] = 0
    if cuda_num is not None:
        tester_params['cuda_device_num']=cuda_num


    tester_params['model_load']={
        'path': path,
    }

    logger_params['log_file']['desc'] = file_name

    tester_params['cuda_device_num'] = args.cuda_device_num
    tester_params['test_episodes'] = test_paras[args.problem_size]['test_episodes']
    tester_params['test_batch_size'] = test_paras[args.problem_size]['test_batch_size']
    model_params['k_nearest_edges'] = args.k_nearest_edges
    model_params['k_nearest_scatter'] = args.k_nearest_scatter
    model_params['knearest'] = args.knearest
    model_params['coor_norm'] = args.coor_norm
    # env_params['data_path'] = b + f"/data/TSP/{test_paras[args.problem_size][0]}"
    env_params['data_path'] = f'/public/home/bayp/exp_survey_202509/0_data_survey/survey_synthetic_tsp/test_tsp{args.problem_size}_n{test_paras[args.problem_size]["test_episodes"]}_lkh.txt'

    env_params['tsplib_path'] = None #b + f"/data/TSP/{test_paras[args.problem_size][0]}" # unused if test_in_tsplib is False
    env_params['test_in_tsplib'] = args.test_in_tsplib
    env_params['RRC_budget'] = args.RRC_budget
    env_params['random_insertion'] = args.random_insertion
    env_params['max_RRC_range'] = args.RRC_range

    create_logger(**logger_params)

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

    parser = argparse.ArgumentParser(description='test')
    add_common_args(parser)
    args = parser.parse_args()

    cuda_num = 0

    problem_scales = [100, 1000, 10000]

    RRC_budgets = [0] # see it
    RI_inites = [0]
    coords_norms = [0]
    knearest_ifs = [1]
    rrc_ranges = [300]
    k_nearest_edge_nums = [100]
    k_nearest_scatter_nums = [100]

    test_in_tsplib_if = 0

    file_name = f'scales={str(problem_scales)}, RRC_b={str(RRC_budgets)}, RIinit={str(RI_inites)},' \
                f' norms={str(coords_norms)}, k_if={str(knearest_ifs)}, rrc_ranges={str(rrc_ranges)},' \
                f' k_edges={str(k_nearest_edge_nums)}, k_scatters={str(k_nearest_scatter_nums)}'
    # log文件夹的名字。命名太长可能会导致系统无法创建此文件夹，需要系统允许创建长命名文件夹


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
                                    args.test_in_tsplib = test_in_tsplib_if

                                    if problem_scale <= 1000:
                                        args.coor_norm = 0

                                    if problem_scale > 1000:
                                        args.coor_norm = 1

                                    args.RRC_budget = rrc_budget
                                    args.RRC_range = rrc_range

                                    path = model_load_path

                                    score_optimal, score_student,gap = main_test(path,args,file_name)
