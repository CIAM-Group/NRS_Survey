import time
import torch
import argparse
import numpy as np
from hydra import initialize
from omegaconf import OmegaConf
from h_tsp import (
    readDataFile,
    use_saved_problems_tsp_txt,
    HTSP_PPO,
    utils,
    VecEnv,
    RLSolver,
)

# nohup python -u evaluate_synthetic.py --graph_size 1000 --batch_size 128 > htsp_tsp1000_synthetic_eval.txt 2>&1 &
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lower_model", type=str, help="Path to the lower level model checkpoint"
    )
    parser.add_argument(
        "--upper_model", type=str, help="Path to the upper level model checkpoint"
    )
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--graph_size", type=int, default=1000)
    parser.add_argument("--frag_len", type=int, default=200, help="Sub-problem size")
    parser.add_argument(
        "--max_new_cities",
        type=int,
        default=190,
        help="Maximum number of new cities in sub-problem",
    )
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--data_augment", default=False, action="store_true")
    parser.add_argument(
        "--improvement_step", type=int, default=0, help="Number of improvement steps"
    )
    parser.add_argument("--time_limit", type=float, default=100.0, help="Time limit")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Evaluate batch size"
    )

    return parser.parse_args()


def main(args):
    start_t = time.time()
    graph_size = args.graph_size
    frag_len = args.frag_len
    max_new_cities = args.max_new_cities
    k = args.k
    bsz = args.batch_size
    
    args.upper_model = f"/public/home/bayp/exp_survey_202509/Construction/two-stage/1_H-TSP/h-tsp-ckpt/upper-level/tsp{graph_size}/best.ckpt"
    args.lower_model = f"/public/home/bayp/exp_survey_202509/Construction/two-stage/1_H-TSP/h-tsp-ckpt/lower-level/lower{frag_len}/best.ckpt"

    ckpt = torch.load(args.upper_model)
    with initialize(config_path="."):
        cfg = OmegaConf.create(ckpt["hyper_parameters"])

    cfg.low_level_load_path = args.lower_model

    model = HTSP_PPO(cfg).cuda()
    model.load_state_dict(ckpt["state_dict"])
    rl_solver = RLSolver(model.low_level_model, frag_len)

    # print all args
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
        
    # data_file = f"data/cluster/tsp{graph_size}_test_concorde.txt"
    data_file = f'/public/home/bayp/exp_survey_202509/0_data_survey/survey_synthetic_tsp/test_tsp{graph_size}_n{bsz}_lkh.txt'
    print(f"Loading data from {data_file}")
    #data = readDataFile(data_file)
    data, optimal_score = use_saved_problems_tsp_txt(data_file, total_episodes=10000, device='cpu', start=0)
    print("optimal_score loaded, shape:", optimal_score.shape)
    print("optimal_score:", optimal_score.mean().item())
    
    
    sample_nums = data.shape[0]
    if args.data_augment:
        data = utils.augment_xy_data_by_8_fold(data)
    print(f"{data.shape=}")

    vec_env = VecEnv(
        k=k, frag_len=frag_len, max_new_nodes=max_new_cities, max_improvement_step=0
    )
    results = np.array([])
    all_tours = torch.zeros(size=(data.shape[0],graph_size), dtype=torch.long)
    
    for i in range(0, data.shape[0], bsz):
        batch_start = time.time()
        batch_time_limit = args.time_limit * bsz
        batch_data = data[i : i + bsz]
        print(f"{i}/{batch_data.shape[0]}")
        s = vec_env.reset(batch_data.to(model.device))
        while not vec_env.done:
            a = model(s).detach()
            # random action for comparison
            # a = vec_env.random_action().to(model.device)
            s, r, d, info = vec_env.step(
                a, rl_solver, frag_buffer=model.val_frag_buffer
            )
        print(np.array([e.state.current_tour_len.item() for e in vec_env.envs]).mean())
        if args.improvement_step > 0:
            for env in vec_env.envs:
                env.max_improvement_step = args.improvement_step
            while not vec_env.done:
                if time.time() - batch_start > batch_time_limit:
                    break
                a = vec_env.random_action().to(model.device)
                s, r, d, info = vec_env.step(
                    a, rl_solver, frag_buffer=model.val_frag_buffer
                )
        length = np.array([e.state.current_tour_len.item() for e in vec_env.envs])
        tours = torch.tensor([e.state.current_tour for e in vec_env.envs], dtype=torch.long) # list of tensors
        results = np.concatenate((results, length))
        all_tours[i : i + bsz] = tours

    end_time = time.time() - start_t
    
    if args.data_augment:
        results = results.reshape(8, -1).min(axis=0)

    assert results.shape[0] == sample_nums, f"{length.shape[0]=}, {sample_nums=}"

    optimal_score = optimal_score.numpy()
    gap = (results - optimal_score) * 100 / optimal_score
    print(f"gap (%): {gap.mean()}")
    # output time second mins and hour
    print(f"Total time: {end_time} seconds, {end_time/60} mins, {end_time/3600} hours")
    
    # ! 下列操作是额外的验证步骤, 不在计算时间内
    # 1. 先检查物理长度
    assert all_tours.shape[1] == graph_size, "Tour length error: expected Node num {0}, got {1}".format(graph_size, all_tours.shape[1])
    # 2. 再检查是否首尾相连
    assert (all_tours[:, 0] != all_tours[:, -1]).all(), "Tour is not a closed loop"

    for i in range(sample_nums):
        unique_node_list_len = len(torch.unique(all_tours[i]))
        assert unique_node_list_len == graph_size, \
            f"refinement process error, unique_node_list_len:{unique_node_list_len}, problem_size:{graph_size}"
            
    double_check_distances = _get_travel_distance(all_tours.to(model.device), data.to(model.device)).cpu().numpy()
    # 3. 最后检查计算的路径长度和之前记录的长度是否一致
    assert np.allclose(double_check_distances, results, atol=1e-5), "Calculated tour lengths do not match recorded lengths."
    gap_check = (double_check_distances - optimal_score) * 100 / optimal_score
    print(f"Double check gap (%): {gap_check.mean()}")

    return end_time, gap.mean(), results.mean()

def _get_travel_distance(selected_node_list, problems):
    # selected_node_list: shape (batch, problem)
    # problems: shape (batch, problem, 2)
    gathering_index = selected_node_list.unsqueeze(2).expand(selected_node_list.size(0), selected_node_list.size(1), 2)
    # shape: (batch, problem, 2)

    ordered_seq = problems.gather(dim=1, index=gathering_index)
    # shape: (batch, problem, 2)

    rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
    segment_lengths = ((ordered_seq-rolled_seq)**2).sum(2).sqrt()
    # shape: (batch, problem)

    travel_distances = segment_lengths.sum(1)
    # shape: (batch,)
    return travel_distances


if __name__ == "__main__":
    # 
    args = parse_args()
    durations = []
    gaps = []
    results = []
    assert args.repeat_times == 1, "Only support repeat_times=1 for now."
    
    for i in range(args.repeat_times):
        duration, gap,result = main(args)
        durations.append(duration)
        gaps.append(gap)
        results.append(result)
    print(f"average duration: {np.average(durations)}")
    print(f"average gap: {np.average(gaps)}")
    print(f"average result: {np.average(results)}")
