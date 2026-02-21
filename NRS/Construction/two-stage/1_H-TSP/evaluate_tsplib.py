import os
import sys
import time
import torch
import argparse
import numpy as np
from hydra import initialize
from omegaconf import OmegaConf
from LIBUtils import *
from h_tsp import (
    readDataFile,
    use_saved_problems_tsp_txt,
    HTSP_PPO,
    utils,
    VecEnv,
    RLSolver,
)


# nohup python -u evaluate_tsplib.py > htsp_tsplib_eval_multiModels.log 2>&1 &
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lower_model", type=str, help="Path to the lower level model checkpoint"
    )
    parser.add_argument(
        "--upper_model", type=str, help="Path to the upper level model checkpoint"
    )
    parser.add_argument("--repeat_times", type=int, default=1)
    # parser.add_argument("--graph_size", type=int, default=1000)
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
    #parser.add_argument(
        # "--batch_size", type=int, default=1, help="Evaluate batch size"
    #)

    return parser.parse_args()


def main(args):
    # start_t = time.time()
    graph_size = args.graph_size
    print("graph_size:", graph_size)
    frag_len = args.frag_len
    max_new_cities = args.max_new_cities
    k = args.k
    bsz = 1 #args.batch_size
    
    args.upper_model = f"/public/home/bayp/exp_survey_202509/Construction/two-stage/1_H-TSP/h-tsp-ckpt/upper-level/tsp{graph_size}/best.ckpt"
    args.lower_model = f"/public/home/bayp/exp_survey_202509/Construction/two-stage/1_H-TSP/h-tsp-ckpt/lower-level/lower{frag_len}/best.ckpt"

    ckpt = torch.load(args.upper_model)
    with initialize(config_path="."):
        cfg = OmegaConf.create(ckpt["hyper_parameters"])

    cfg.low_level_load_path = args.lower_model

    model = HTSP_PPO(cfg).cuda()
    model.load_state_dict(ckpt["state_dict"])
    rl_solver = RLSolver(model.low_level_model, frag_len) # 实例化rl_solver
    
    device = model.device

    # print all args
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
        
    # data_file = f"data/cluster/tsp{graph_size}_test_concorde.txt"
    data_file = f'/public/home/bayp/exp_survey_202509/0_data_survey/survey_bench_tsp'
    print(f"Loading data from {data_file}")
    
    #data = readDataFile(data_file)
    # data, optimal_score = use_saved_problems_tsp_txt(data_file, total_episodes=10000, device='cpu', start=0)
    # print("optimal_score loaded, shape:", optimal_score.shape)
    # print("optimal_score:", optimal_score.mean().item())
    
    
    # ! 遍历求解每个实例
    gap_set_less_1000 = []
    gap_set_less_10000 = []
    gap_set_less_100000 = []
    
    gap_set_all_instances = []

    all_instance_num = 0
    all_solved_instance_num = 0

    start_time_all = time.time()

    scale_range_all = [[0, 1000], [1000, 10000], [10000, 100001]]


    for scale_range in scale_range_all:
        print("#################  Test scale range: {0}  #################".format(scale_range))
        # run_one_scale_range_lib(data_file,scale_range)
        
        num_sample = 0
        start_time_range = time.time()
        result_dict = {}
        result_dict["instances"] = []
        result_dict['optimal'] = []
        result_dict['problem_size'] = []
        result_dict['score'] = []
        result_dict['gap'] = []
        for root, dirs, files in os.walk(data_file):
            for file in files:
                if file.endswith(".tsp"):
                    name, dimension, locs, ew_type = TSPLIBReader(os.path.join(root, file))
                    if name is None:
                        continue
                    if not (scale_range[0] <= dimension < scale_range[1]):
                        continue

                    # ! check，打印当前处理的文件名，看是缺了哪个label
                    full_path = os.path.join(root, file)
                    print(f"===================当前读取的文件名: {full_path}===================")  # 推荐用 logger

                    dict_instance_info = {}
                    optimal = float(tsplib_cost.get(name, None))
                    assert optimal is not None, "optimal value of instance {} not found".format(name)
                    instance_xy = np.array(locs).astype(np.float32)  # shape: (dimension,2)
                    node_coord = torch.from_numpy(instance_xy).unsqueeze(0)
                    # shape: (1,problem_size,2)
                    assert node_coord.shape == (1, dimension, 2), "dimension error in instance {}".format(name)

                    num_sample += 1 # 实际总实例个数,包含因为各种原因跳过的实例
                    all_instance_num += 1 # 全部实例个数,包含因为各种原因跳过的实例

                    dict_instance_info['optimal'] = optimal
                    dict_instance_info['problem_size'] = dimension
                    dict_instance_info['pomo_size'] = dimension
                    dict_instance_info['original_node_xy_lib'] = node_coord
                    # shape:(1,problem_size,2)
                    dict_instance_info['name'] = name
                    # ! 补充round/ceil
                    dict_instance_info['edge_weight_type'] = ew_type

                    print("===============================================================")
                    print("Instance name: {0}, problem_size: {1}".format(name, dimension))

                    # normalize data to [0,1] using min-max normalization
                    ################################################################

                    xy_max = torch.max(node_coord, dim=1, keepdim=True).values
                    xy_min = torch.min(node_coord, dim=1, keepdim=True).values
                    # shape: (1, 1, 2)
                    ratio = torch.max((xy_max - xy_min), dim=-1, keepdim=True).values
                    ratio[ratio == 0] = 1
                    # shape: (1, 1, 1)
                    nodes_xy_normalized = (node_coord - xy_min) / ratio.expand(-1, 1, 2)
                    # shape: (1, dimension+1,2)

                    dict_instance_info["node_xy"] = nodes_xy_normalized


                    # ! 2025.10.08补充：统计单个实例时间
                    # === 计时开始（含矩阵计算与测试）===
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    inst_start = time.time()

                    # shape:(1,dimension,2)
                    try:
                        score = test_one_instance(args,model,rl_solver,dict_instance_info)
                        gap = (score - optimal) * 100 / optimal
                        if gap >= 100.0:
                            print("Warning: Aug gap >=100% in instance {0}, dimension: {1}, gap: {2:.3f}%, skip it!".format(name, dimension, gap))
                            continue
                        all_solved_instance_num += 1
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                        inst_time = time.time() - inst_start
                    except Exception as e:
                        print("Error occurred in instance {0}, dimension: {1}, skip it!".format(name, dimension))
                        print("Error message: {0}".format(e))
                        continue

                    ############################
                    # Logs
                    ############################
                    
                    result_dict["instances"].append(name)
                    result_dict['optimal'].append(optimal)
                    result_dict['problem_size'].append(dimension)
                    result_dict['score'].append(score)
                    result_dict['gap'].append(gap)

                    gap_set_all_instances.append(gap)

                    if dimension < 1000:
                        gap_set_less_1000.append(gap)
                    elif 1000 <= dimension < 10000:
                        gap_set_less_10000.append(gap)
                    elif 10000 <= dimension <= 100000:
                        gap_set_less_100000.append(gap)

                    print("Instance name: {}, optimal score: {:.4f}".format(name, optimal))
                    print("score:{:.3f}, gap:{:.3f}%".format(score, gap))
                    # ! 补充时间
                    print(f"Instance time: {inst_time:.3f}s")
                    sys.stdout.flush() # 强制刷新输出缓冲区

        end_time_range = time.time()
        during_range = end_time_range - start_time_range
        # Logs for all instances
        print(" *** Test Done *** ")
        print("scale_range: {0}, instance number: {1}, total time: {2:.2f}s, avg time per instance (including failed instances): {3:.2f}s".
                            format(scale_range, num_sample, during_range, during_range / num_sample))
        print("===============================================================")
        print("instance: {0}".format(result_dict['instances']))
        print("optimal: {0}".format(result_dict['optimal']))
        print("problem_size: {0}".format(result_dict['problem_size']))
        print("score: {0}".format(result_dict['score']))
        print("gap: {0}".format(result_dict['gap']))
        print("===============================================================")

        print("===============================================================")
        avg_solved_gap = np.mean(result_dict['gap'])  # avg of all instances gap
        solved_instance_num = len(result_dict['instances'])
        max_dimension = max(result_dict['problem_size'])
        min_dimension = min(result_dict['problem_size'])
        print("Solved_ instances number: {0}/{1}, min_dimension: {2}, max_dimension: {3}, avg gap: {4:.3f}%".
            format(solved_instance_num, num_sample, min_dimension, max_dimension, avg_solved_gap))
        print("Avg time per instance (excluding failed instances): {0:.2f}s".format(during_range / solved_instance_num))
    
    
    
    ###########################

    end_time_all = time.time()
    print("All scale ranges done, solved instance number: {0}/{1}, total time: {2:.2f}s, avg time per instance: {3:.2f}s".
                        format(all_solved_instance_num, all_instance_num,
                                end_time_all - start_time_all,
                                (end_time_all - start_time_all) / all_solved_instance_num))

    print("[0, 1000), number: {0}, avg gap: {1:.3f}%".
                        format(len(gap_set_less_1000),
                            np.mean(gap_set_less_1000) if len(gap_set_less_1000) > 0 else 0))
    print("[1000, 10000), number: {0}, avg gap: {1:.3f}%".
                        format(len(gap_set_less_10000),
                                    np.mean(gap_set_less_10000) if len(gap_set_less_10000) > 0 else 0))
    print("[10000, 100000], number: {0}, avg gap: {1:.3f}%".
                        format(len(gap_set_less_100000),
                                    np.mean(gap_set_less_100000) if len(gap_set_less_100000) > 0 else 0))
    print("#################  All Done  #################")
    print("All solved instances, number: {0}, avg gap: {1:.3f}%".
                        format(len(gap_set_all_instances),
                                np.mean(gap_set_all_instances) if len(gap_set_all_instances) > 0 else 0))

    if args.data_augment:
        raise ValueError("Data augment not supported in TSPLIB evaluation.")
        results = results.reshape(8, -1).min(axis=0)

    

    # optimal_score = optimal_score.numpy()
    # gap = (results - optimal_score) * 100 / optimal_score
    # print(f"gap (%): {gap.mean()}")
    # output time second mins and hour

    # return end_time, gap.mean(), results.mean()

# ! 新增求解单个实例的函数框架
@torch.no_grad()
def test_one_instance(args,model,rl_solver,dict_instance_info):
    problem_size = dict_instance_info['problem_size']
    data = dict_instance_info['node_xy']  # shape:(1,problem_size,2)
    sample_nums = data.shape[0]
    assert sample_nums == 1, "Only support single instance evaluation."
    if args.data_augment:
        data = utils.augment_xy_data_by_8_fold(data)
    print(f"data.shape => {data.shape}")

    vec_env = VecEnv(
        k=args.k, frag_len=args.frag_len, max_new_nodes=args.max_new_cities, max_improvement_step=0
    )
    
    results = np.array([])
    bsz = 1 #args.batch_size
    
    # for i in range(0, data.shape[0], bsz):
    batch_start = time.time()
    batch_time_limit = args.time_limit * bsz
    batch_data = data
    s = vec_env.reset(batch_data.to(model.device))
    while not vec_env.done:
        a = model(s).detach()
        # random action for comparison
        # a = vec_env.random_action().to(model.device)
        s, r, d, info = vec_env.step(
            a, rl_solver, frag_buffer=model.val_frag_buffer
        )
    # print(np.array([e.state.current_tour_len.item() for e in vec_env.envs]).mean())
    if args.improvement_step > 0:
        # ! improvement step is prohibited for now
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
    results = np.concatenate((results, length))
    
    tours = torch.tensor([e.state.current_tour for e in vec_env.envs], dtype=torch.long) # list of tensors
    assert tours.shape == (1, problem_size), f"Tour shape error: expected (1, {problem_size}), got {tours.shape}"
    
    # ! 虽然我们新增了额外的验证步骤, 但该计算时间极少, 可忽略不计
    # 1. 先检查物理长度
    assert tours.shape[1] == problem_size, "Tour length error: expected Node num {0}, got {1}".format(problem_size, tours.shape[1])
    # 2. 再检查是否首尾相连
    assert (tours[:, 0] != tours[:, -1]).all(), "Tour is not a closed loop"
    # 3. 再检查是否包含所有节点
    assert len(torch.unique(tours[0])) == problem_size
            
    
    original_node_xy_lib = dict_instance_info['original_node_xy_lib']
    real_distances = get_travel_distance(tours.to(model.device), 
                                          original_node_xy_lib.to(model.device), 
                                          dict_instance_info['edge_weight_type'])[0].mean().cpu().item()
    assert results.shape[0] == 1, f"{length.shape[0]=}, {sample_nums=}"
    
    return real_distances

def get_travel_distance(selected_node_list, problems,ewt):
    # selected_node_list: shape (batch, problem)
    # problems: shape (batch, problem, 2)
    gathering_index = selected_node_list.unsqueeze(2).expand(selected_node_list.size(0), selected_node_list.size(1), 2)
    # shape: (batch, problem, 2)

    ordered_seq = problems.gather(dim=1, index=gathering_index)
    # shape: (batch, problem, 2)

    rolled_seq = ordered_seq.roll(dims=1, shifts=-1)
    segment_lengths_raw = ((ordered_seq-rolled_seq)**2).sum(2).sqrt()
    # shape: (batch, problem)
    
    # 逐边离散化：按 TSPLIB 的度量来
    if ewt == 'CEIL_2D':
        segment_lengths = torch.ceil(segment_lengths_raw)
    elif ewt == 'EUC_2D':
        # TSPLIB 定义：floor(x + 0.5)，避免银行家舍入
        # segment_lengths = (segment_lengths_raw + 0.5).floor()
        segment_lengths = torch.floor(segment_lengths_raw + 0.5)
    else:
        # 其他类型就不取整，保持连续距离（或按需扩展）
        segment_lengths = segment_lengths_raw

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
    
    upper_model_scale_list = [1000, 2000, 5000, 10000]
    for scale in upper_model_scale_list:
        print("="*40)
        print(f"Evaluating model for scale {scale}")
        print("="*40)
        args.graph_size = scale
        main(args)
    
    # for i in range(args.repeat_times):
    #     duration, gap,result = main(args)
    #     durations.append(duration)
    #     gaps.append(gap)
    #     results.append(result)
    # print(f"average duration: {np.average(durations)}")
    # print(f"average gap: {np.average(gaps)}")
    # print(f"average result: {np.average(results)}")
