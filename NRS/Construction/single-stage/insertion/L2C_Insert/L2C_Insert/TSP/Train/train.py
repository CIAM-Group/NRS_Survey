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
sys.path.insert(0, "../../..")  # for utils

##########################################################################################
# import

import logging

from L2C_Insert.TSP.utils.utils import create_logger, copy_all_src

from L2C_Insert.TSP.Train.TSPTrainer import TSPTrainer as Trainer

##########################################################################################
# parameters

b = os.path.abspath("../../..").replace('\\', '/')

########### model ###############
mode = 'train'
model_load_enable = False
model_load_path = ''
model_load_epoch = 0


env_params = {
    'test_in_tsplib': False,
    'tsplib_path': b +'',
    'data_path': b + "/data/TSP/train_TSP100_n1000000.txt",
    'mode': mode,
    'sub_path': True
}


model_params = {
    'mode': 'train',
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

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
                 },
    'scheduler': {
        'milestones': [1 * i for i in range(1, 150)],
        'gamma': 0.97
                 }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 15,
    'train_episodes': 1000000,
    'train_batch_size': 1024,
    'logging': {
        'model_save_interval': 1,
        'img_save_interval': 3000,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_100.json'
               },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
               },
               },
    'model_load': {
        'enable': model_load_enable,  # enable loading pre-trained model
        'path': f'.{model_load_path}',  # directory path of pre-trained model and log files saved.
        'epoch': model_load_epoch,  # epoch version of pre-trained model to laod.
                  }
    }

logger_params = {
    'log_file': {
        'desc': 'train',
        'filename': 'log.txt'
    }
}

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params

    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 8
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()


