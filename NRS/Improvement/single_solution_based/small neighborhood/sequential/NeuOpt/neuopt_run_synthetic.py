import os
import json
import time
import torch
import pprint
import numpy as np
import random
from tensorboard_logger import Logger as TbLogger
import warnings
from options import get_options

from problems.problem_tsp import TSP
from problems.problem_cvrp import CVRP
from agent.ppo import PPO

def load_problem(name):
    problem = {
        'tsp': TSP,
        'cvrp': CVRP,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def run(opts):
    start_time = time.time()
    # Pretty print the run args
    pprint.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tb and not opts.distributed:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, 
                                                          opts.graph_size), opts.run_name))
    if not opts.no_saving and not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
        
    # Save arguments so exact configuration can always be found
    if not opts.no_saving:
        with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
    
    # Figure out what's the problem
    problem = load_problem(opts.problem)(
                            p_size = opts.graph_size,
                            init_val_met = opts.init_val_met,
                            with_assert = opts.use_assert,
                            DUMMY_RATE = opts.dummy_rate,
                            k = opts.k,
                            with_bonus = not opts.wo_bonus,
                            with_regular = not opts.wo_regular)
    
    # Figure out the RL algorithm
    agent = PPO(problem, opts)

    # Load data from load_path
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        agent.load(load_path)

    # Do validation only
    if opts.eval_only:
        # Load the validation datasets
        agent.start_inference(problem, opts.val_dataset, tb_logger)
        
    else:
        if opts.resume:
            epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
            print("Resuming after {}".format(epoch_resume))
            agent.opts.epoch_start = epoch_resume + 1
    
        # Start the actual training loop
        agent.start_training(problem, opts.val_dataset, tb_logger)
    end_time = time.time()
    # print seconds and mins
    print("Total time used: {:.2f} seconds".format(end_time - start_time))
    print("Total time used: {:.2f} mins".format((end_time - start_time)/60))
    print("Total time used: {:.2f} hours".format((end_time - start_time)/3600))

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    
    warnings.filterwarnings("ignore")
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ! 在这里写死参数，相当于命令行输入
    # predefined_args = [
    #     '--problem', 'tsp',
    #     '--graph_size', '20',
    #     '--run_name', 'debug_tsp20',
    #     '--epoch_end', '10',
    #     '--no_tb',   # 举例：关闭 TensorBoard
    # ]    
    
    opts = get_options()
    
    if opts.problem == 'tsp':
        opts.wo_feature1 = opts.wo_feature2 = opts.wo_feature3 = opts.wo_bonus = opts.wo_regular = True
    
    ### figure out whether to use distributed training
    opts.world_size = torch.cuda.device_count()
    opts.distributed = (torch.cuda.device_count() > 1) and (not opts.no_DDP)
    # ! 不管训练的服务器和端口
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '4869'
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    
    opts.eval_only = True
    opts.no_saving = True
    opts.no_tb = True
    opts.load_path = 'pre-trained/tsp100.pt'
    opts.T_max = 10000
    opts.val_batch_size = 10000
    opts.val_size = 10000
    opts.val_dataset = '/public/home/bayp/exp_survey_202509/0_data_survey/survey_synthetic_tsp/test_tsp100_n10000_lkh.txt'
    opts.val_m = 1
    opts.init_val_met = 'random'
    opts.k = 4
    
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S")) \
        if not opts.resume else opts.resume.split('/')[-2]
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    ) if not opts.no_saving else None
    
    # print options
    print("\n[Options]")
    for k, v in sorted(vars(opts).items()):
        print(f'  {k}: {v}')
        
    run(opts)
    # run(get_options(predefined_args))

