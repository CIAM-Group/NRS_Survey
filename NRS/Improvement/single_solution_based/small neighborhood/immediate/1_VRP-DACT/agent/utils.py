import time
import torch
import os
import random
import numpy as np
from utils.logger import log_to_screen, log_to_tb_val
import torch.distributed as dist
from torch.utils.data import DataLoader
from tensorboard_logger import Logger as TbLogger

def gather_tensor_and_concat(tensor):
    gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, tensor)
    return torch.cat(gather_t)

def validate(rank, problem, agent, val_dataset, tb_logger, distributed = False, _id = None):
            
    # Validate mode
    opts = agent.opts
    if rank==0: print(f'\nInference with x{opts.val_m} augments...', flush=True)
    
    agent.eval()
    problem.eval()
    
    if opts.eval_only:
        torch.manual_seed(opts.seed)
        np.random.seed(opts.seed)
        random.seed(opts.seed)
    
    val_dataset = problem.make_dataset(size=opts.graph_size,
                               num_samples=opts.val_size,
                               filename = val_dataset,
                               DUMMY_RATE = opts.dummy_rate,
                               lib_model = opts.lib_model,
                               scale_range = opts.scale_range,
                               device = opts.device)

    if distributed and opts.distributed:
        assert False, "Distributed validation is not supported yet!"
        device = torch.device("cuda", rank)
        torch.distributed.init_process_group(backend='nccl', world_size=opts.world_size, rank = rank)
        torch.cuda.set_device(rank)
        agent.actor.to(device)
        agent.actor = torch.nn.parallel.DistributedDataParallel(agent.actor, device_ids=[rank])
        
        if not opts.no_tb and rank == 0:
            tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, 
                                                          opts.graph_size), opts.run_name))

        assert opts.val_size % opts.world_size == 0
        train_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size = opts.eval_batch_size, shuffle=False,
                                    num_workers=0,
                                    pin_memory=True,
                                    sampler=train_sampler)
    else:
        val_dataloader = DataLoader(val_dataset, batch_size=opts.eval_batch_size, shuffle=False,
                                   num_workers=0,
                                   pin_memory=True)
    
    s_time = time.time()
    optimal_score = []
    bv = []
    cost_hist = []
    best_hist = []
    r = []

    for batch_id, batch in enumerate(val_dataloader):
        current_start_batch_idx = batch_id * opts.eval_batch_size
        bv_, cost_hist_, best_hist_, r_, rec_history_ = agent.rollout(problem,
                                                                 opts.val_m,
                                                                 batch,
                                                                 do_sample = True,
                                                                 record = False,
                                                                 show_bar = rank==0,
                                                                 current_start_batch_idx = current_start_batch_idx)
        bv.append(bv_)
        cost_hist.append(cost_hist_)
        best_hist.append(best_hist_)
        r.append(r_)
        assert batch.get('optimal_score') is not None, "Optimal score is required for validation!"
        if batch.get('optimal_score') is not None:
            optimal_score.append(batch.get('optimal_score'))
    
    bv = torch.cat(bv, 0)
    cost_hist = torch.cat(cost_hist, 0)
    best_hist = torch.cat(best_hist, 0)
    r = torch.cat(r, 0)
    if len(optimal_score) > 0:
        optimal_score = torch.cat(optimal_score, 0)
    else:
        optimal_score = None

    if distributed and opts.distributed:
        assert False, "Distributed validation is not supported yet!"
        dist.barrier()
        initial_cost = gather_tensor_and_concat(cost_hist[:,0].contiguous())
        time_used = gather_tensor_and_concat(torch.tensor([time.time() - s_time]).cuda())
        bv = gather_tensor_and_concat(bv.contiguous())
        costs_history = gather_tensor_and_concat(cost_hist.contiguous())
        search_history = gather_tensor_and_concat(best_hist.contiguous())
        reward = gather_tensor_and_concat(r.contiguous())
        if optimal_score is not None:
             optimal_score = gather_tensor_and_concat(optimal_score.to(bv.device).contiguous())
        dist.barrier()
    else:
        initial_cost = cost_hist[:,0]
        time_used = torch.tensor([time.time() - s_time])
        bv = bv
        costs_history = cost_hist
        search_history = best_hist
        reward = r
        if optimal_score is not None:
            optimal_score = optimal_score.to(bv.device)
    
    # log to screen  
    if rank == 0: log_to_screen(time_used, 
                                initial_cost, 
                                bv, 
                                reward, 
                                costs_history,
                                search_history,
                                batch_size = opts.val_size, 
                                dataset_size = len(val_dataset), 
                                T = opts.T_max,
                                optimal_score = optimal_score)
    
    # log to tb
    if(not opts.no_tb) and rank == 0:
        assert False, "Tensorboard logging during validation is not supported yet!"
        log_to_tb_val(tb_logger,
                      time_used, 
                      initial_cost, 
                      bv, 
                      reward, 
                      costs_history,
                      search_history,
                      batch_size = opts.val_size,
                      val_size =  opts.val_size,
                      dataset_size = len(val_dataset), 
                      T = opts.T_max,
                      epoch = _id)
        
    if distributed and opts.distributed: 
        assert False, "Distributed validation is not supported yet!"
        dist.barrier()
    
