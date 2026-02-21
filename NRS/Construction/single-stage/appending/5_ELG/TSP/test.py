import torch
from torch.optim import Adam as Optimizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import yaml
import time
import os

from generate_data import generate_tsp_data, TSPDataset
from TSPModel import TSPModel
from TSPEnv import TSPEnv
from utils import rollout, check_feasible, Logger


def test(dataloader, model, env, aug_factor,dataset_num):
    # test
    model.eval()
    model.requires_grad_(False)
    avg_cost_total = 0.
    no_avg_cost_total = 0.
    avg_gap_total = 0.
    all_gap_tensor = torch.zeros(dataset_num)
    t = 0
    start = time.time()
    for batch in dataloader:
        if isinstance(batch, list) and len(batch) == 2:
            problems, optimal_scores = batch
            if optimal_scores is not None:
                optimal_scores = optimal_scores.to(env.device)
        else:
            problems = batch
            optimal_scores = None

        env.load_random_problems(problems, aug_factor)
        reset_state, _, _ = env.reset()
        
        with torch.no_grad():
            model.pre_forward(reset_state)
            solutions, probs, rewards = rollout(model=model, env=env, eval_type='greedy')

        # Return
        aug_reward = rewards.reshape(aug_factor, problems.shape[0], env.pomo_size)
        # shape: (augmentation, batch, pomo)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_cost = -max_pomo_reward[0, :].float()  # negative sign to make positive value
        no_aug_cost_mean = no_aug_cost.mean()

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_cost = -max_aug_pomo_reward.float()  # negative sign to make positive value
        aug_cost_mean = aug_cost.mean()

        if optimal_scores is not None and optimal_scores.sum() > 0:
             gap = (aug_cost - optimal_scores) * 100 / optimal_scores
             avg_gap_total += gap.mean().item()
             all_gap_tensor[t * dataloader.batch_size: t * dataloader.batch_size + problems.shape[0]] = gap
             print("Batch {}: Avg.optimal score: {:.4f}, Aug.cost: {:.4f}, Gap: {:.4f}%, No aug.cost: {:.4f}".format(
                 (t+1) * dataloader.batch_size, optimal_scores.mean().item(), aug_cost_mean.item(), gap.mean().item(), no_aug_cost_mean.item()))

        avg_cost_total += aug_cost_mean.item()
        no_avg_cost_total += no_aug_cost_mean.item()
        # best_idx = rewards.max(1)[1]
        # best_sols = torch.take_along_dim(solutions, best_idx[:, None, None].expand(solutions.shape), dim=1)
        # # check feasible
        # check_feasible(best_sols[0:1], reset_state.node_demand[0:1])
        t += 1
    end = time.time()
    avg_cost_total /= t
    no_avg_cost_total /= t
    avg_gap_total /= t
    avg_gap_2 = all_gap_tensor.mean().item()
    print("Aug cost: {:.4f}".format(avg_cost_total))
    print("Aug gap: {:.4f}%".format(avg_gap_total))
    print("Aug gap (tensor): {:.4f}%".format(avg_gap_2))
    print("no aug Avg cost: {:.4f}, Wall-clock time: {:.2f}s".format(no_avg_cost_total, float(end - start)))
    
    return avg_cost_total


if __name__ == "__main__":
    start_time = time.time()
    config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    with open(config_path, 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)

    # params
    device = "cuda:{}".format(config['cuda_device_num']) if config['use_cuda'] else 'cpu'
    multiple_width = config['params']['multiple_width']
    print("Test device: {}".format(device))
    print("Test multiple width: {}".format(multiple_width))
    test_size = config['params']['test_size']
    print("Test size: {}".format(test_size))
    test_batch_size = config['params']['test_batch_size']
    print("Test batch size: {}".format(test_batch_size))
    load_checkpoint = config['load_checkpoint']
    load_checkpoint = os.path.join(os.path.dirname(__file__), load_checkpoint)
    print("Load checkpoint: {}".format(load_checkpoint))
    # test_data = config['test_filename']
    test_data = f'/public/home/bayp/exp_survey_202509/0_data_survey/survey_synthetic_tsp/test_tsp100_n{test_size}_lkh.txt'
    print("Test data: {}".format(test_data))
    model_params = config['model_params']
    print("Model params: {}".format(model_params))
    aug_factor = config['params']['aug_factor']
    print("Test aug factor: {}".format(aug_factor))

    # load checkpoint
    model = TSPModel(**model_params)
    if model_params['ensemble']:
        print("Add ensemble policy")
        model.decoder.add_local_policy(device)
    checkpoint = torch.load(load_checkpoint, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Initialize env
    env = TSPEnv(multi_width=multiple_width, device=device)

    # Dataset
    test_set = TSPDataset(test_data, num_samples=test_size)
    test_loader = DataLoader(test_set, batch_size=test_batch_size)

    # test
    test(test_loader, model, env, aug_factor=aug_factor, dataset_num=test_size)
    end_time = time.time()
    print("Total test time: {0:.3f}s, {1:.3f}min, {2:.3f}hr".format(end_time - start_time, (end_time - start_time)/60, (end_time - start_time)/3600))
    print("Total test time per instance: {:.4f}s".format((end_time - start_time)/test_size))