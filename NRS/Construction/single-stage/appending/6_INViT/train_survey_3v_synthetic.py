import torch
import torch.nn as nn
import time
import argparse

import os
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from TSP_net import TSP_net
from VRP_net import VRP_net
from utils.utils_for_model import create_parser, read_from_logs
from training_loop import train_model_with_knn
from test_function_survey import run_tsp_test_knn,run_tsplib_test_knn,run_vrp_test_knn, run_cvrplib_test_knn



###################
# Hardware : CPU / GPU(s)
###################

device = torch.device("cpu"); gpu_id = -1 # select CPU

gpu_id = '0' # select a single GPU  
#gpu_id = '2,3' # select multiple GPUs  
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU name: {:s}, gpu_id: {:s}'.format(torch.cuda.get_device_name(0),gpu_id))   
    
print(device)



### parser ###


config_dict = {
    'aug': 'mix',
    'bsz': 128,
    'nb_nodes':100,
    'model_lr': 2e-5,
    'nb_batch_per_epoch': 300,
    'data_path':'./',
    'checkpoint_model': 'n',
    'aug_num': 16,
    'test_aug_num': 16,
    'num_state_encoder': 3, # ! changed
    'dim_emb': 128,
    'dim_ff':512,
    'nb_heads': 8,
    'action_k': 15,
    'nb_layers_state_encoder': 2,
    'nb_layers_action_encoder': 4, # ! changed, cvrp 2 tsp 4
    'nb_layers_decoder': 3,
    'nb_epochs': 400,
    'problem': 'tsp',
    'gamma': 0.99,
    'dim_input_nodes': 2,
    'batchnorm':False,
    'gpu_id': 0,
    'loss_type':'n',
    'train_joint':'n',
    'nb_batch_eval': 80,
    'if_use_local_mask':False,
    'if_agg_whole_graph':False,
    'tol':1e-3,
}

state_k = [35,50,65]
custom_parser, args = create_parser(config_dict)
config = custom_parser.parse_args(namespace=args)

# ! 加载模型
args.problem = "tsp"
if args.problem == 'tsp':
    args.checkpoint_model = '24-04-24--14-34-54-n100-gpu0'
elif args.problem == 'cvrp':
    args.checkpoint_model = '24-04-18--14-14-20-n100-gpu0'
else:
    raise ValueError('Unsupported Problem Type')

if args.checkpoint_model != 'n':
    read_from_logs(args)

args.state_k = state_k[:args.num_state_encoder]

if args.problem == 'cvrp' or args.problem == 'sdvrp':
    args.CAPACITIES = {
                10: 20.,
                20: 30.,
                50: 40.,
                100: 50.
            }
    


# print(args)
# print options
print("\n[Options]")
for k, v in sorted(vars(args).items()):
    print(f'  {k}: {v}')


if args.problem == 'tsp':

    model_train = TSP_net(args.dim_input_nodes, args.dim_emb, args.dim_ff, args.num_state_encoder, 
                args.nb_layers_state_encoder, args.nb_layers_action_encoder, args.nb_layers_decoder, args.nb_heads, batchnorm = args.batchnorm, if_agg_whole_graph = args.if_agg_whole_graph)
    model_baseline = TSP_net(args.dim_input_nodes, args.dim_emb, args.dim_ff, args.num_state_encoder, 
                args.nb_layers_state_encoder, args.nb_layers_action_encoder, args.nb_layers_decoder, args.nb_heads, batchnorm = args.batchnorm, if_agg_whole_graph = args.if_agg_whole_graph)
    
elif args.problem == 'cvrp' or args.problem == 'sdvrp':

    model_train = VRP_net(args.dim_input_nodes, args.dim_emb, args.dim_ff, args.num_state_encoder,
                args.nb_layers_state_encoder,args.nb_layers_action_encoder, args.nb_layers_decoder, args.nb_heads, batchnorm = args.batchnorm, if_agg_whole_graph = args.if_agg_whole_graph)
    model_baseline = VRP_net(args.dim_input_nodes, args.dim_emb, args.dim_ff, args.num_state_encoder,
                args.nb_layers_state_encoder,args.nb_layers_action_encoder, args.nb_layers_decoder, args.nb_heads, batchnorm = args.batchnorm, if_agg_whole_graph = args.if_agg_whole_graph)

else:

    raise ValueError('Unsupported Problem Type')

optimizer_model = torch.optim.AdamW( model_train.parameters() , lr = args.model_lr ) 
scheduler_model = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_model, gamma=args.gamma)
model_train = model_train.to(device)
model_baseline = model_baseline.to(device)
if args.checkpoint_model != 'n':
    save_addr_model = args.data_path+'ckpt/'+args.problem+'/train/model/checkpoint_'
    checkpoint_file_model = save_addr_model + args.checkpoint_model+'.pkl'
    checkpoint_model = torch.load(checkpoint_file_model, map_location=device)
    tot_time_ckpt_model = checkpoint_model['tot_time']
    model_baseline.load_state_dict(checkpoint_model['model_baseline'])
    model_train.load_state_dict(checkpoint_model['model_train'])
    optimizer_model.load_state_dict(checkpoint_model['optimizer'])
    print('Model checkpoint loaded from {:s}'.format(checkpoint_file_model))
model_baseline.eval()

print(args); print('')

# Logs
#os.system("mkdir logs")
time_stamp=datetime.datetime.now().strftime("%y-%m-%d--%H-%M-%S")
saved_path = args.data_path+'survey/synthetic_'+args.problem
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
file_name = saved_path+'/'+time_stamp + "-n{}".format(args.nb_nodes) + "-gpu{}".format(args.gpu_id) + ".txt"
file = open(file_name,"w",1) 
file.write(time_stamp+'\n\n') 
for arg in vars(args):
    file.write(arg)
    hyper_param_val="={}".format(getattr(args, arg))
    file.write(hyper_param_val)
    file.write('\n')
file.write('\n\n') 
plot_performance_train = []
plot_performance_baseline = []
all_strings = []
epoch_ckpt = 0
tot_time_ckpt = 0


# # Uncomment these lines to re-start training with saved checkpoint

###################
# Main training loop 
###################

# train_model_with_knn(args,model_train,model_baseline,optimizer_model,scheduler_model,device,file,time_stamp)    


## final evaluation part

if args.problem == 'tsp':

    # sizes = [100,1000,5000,10000]
    # bszs = [64,32,16,8]
    # num_instance = [500,50,5,5]
    # distributions = ['uniform', 'clustered1', 'clustered2', 'explosion', 'implosion']
    
    sizes = [100,1000,10000]
    bszs = [64,32,8]
    num_instance = [10000,128, 16]
    bs = [100,50,1]
    distributions = ['uniform']
    local_k = args.action_k
    global_k = args.state_k
    if_use_local_mask = False
    data_path = None
    run_tsp_test_knn(local_k,global_k,args.aug,model_baseline,if_use_local_mask,sizes,bszs,
                     data_path,device,file,distributions,num_instance=num_instance,bs=bs)
    # run_tsplib_test_knn(model_baseline,args.action_k,args.state_k)

elif args.problem == 'cvrp':

    capacity = 50
    sizes = [50,500,5000]
    bszs = [64,32,16]
    num_instance = [500,50,5]
    distributions = ['uniform', 'clustered1', 'clustered2', 'explosion', 'implosion']
    local_k = args.action_k
    global_k = args.state_k
    if_use_local_mask = False
    data_path = args.data_path +'data/'
    # run_vrp_test_knn(local_k,global_k,args.aug,model_baseline,if_use_local_mask,sizes,bszs,data_path,device,file,distributions,num_instance)
    # lib_path = "/public/home/bayp/exp_survey_202509/0_data_survey/survey_bench_cvrp"
    run_cvrplib_test_knn(model_baseline,args.action_k,args.state_k)
    
# nohup python -u train_survey_3v.py --problem tsp > tsplib_survey.log 2>&1 &
# nohup python -u train_survey_3v.py --problem cvrp > cvrplib_survey.log 2>&1 &

# nohup python -u train_survey_3v_synthetic.py --problem tsp > synthetic_tsp100_1000_10000.log 2>&1 &