import os
import json
import time
import torch
import pprint
import numpy as np
from tensorboard_logger import Logger as TbLogger
import warnings
import random
from options import get_options

from problems.problem_tsp import TSP
from problems.problem_vrp import CVRP
from agent.ppo import PPO

def load_agent(name):
    agent = {
        'ppo': PPO,
    }.get(name, None)
    assert agent is not None, "Currently unsupported agent: {}!".format(name)
    return agent

def load_problem(name):
    problem = {
        'tsp': TSP,
        'vrp': CVRP,
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
                            step_method = opts.step_method,
                            init_val_met = opts.init_val_met,
                            with_assert = opts.use_assert,
                            P = opts.P,
                            DUMMY_RATE = opts.dummy_rate)
    
    # Figure out the RL algorithm
    agent = load_agent(opts.RL_agent)(problem.NAME, problem.size,  opts)

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
    # nohup python -u dact_run_synthetic.py --no_saving --no_tb > dact_tsp100_synthetic_aug1.log 2>&1 &
    warnings.filterwarnings("ignore")
    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    opts = get_options()
    
    # figure out whether to use distributed training if needed
    opts.world_size = torch.cuda.device_count()
    opts.distributed = False #(opts.world_size > 1) and (not opts.no_DDP) # we disable DDP for now
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '4869'
    # processing settings
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.eval_only = True
    opts.no_saving = True
    opts.no_tb = True
    opts.load_path = 'pretrained/tsp100-epoch-195.pt' 
    opts.T_max = 10000
    opts.val_size = 10000
    opts.eval_batch_size = 5000
    opts.val_dataset = '/public/home/bayp/exp_survey_202509/0_data_survey/survey_synthetic_tsp/test_tsp100_n10000_lkh.txt'
    # ! We do not use data augmentation during testing here
    opts.val_m = 1 # 
    opts.scale_range = None # only useful for tsplib
    opts.init_val_met = 'greedy'
    opts.P = 250 if opts.eval_only else 1e10 # can set to smaller values e.g., 20 or 10, for generalization 
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
    
    run(opts=opts)
