import argparse
import numpy as np
from data.generate_tsp_with_multirun_2opt import tsp_two_opt_multirun
from helpers.coord_transform import cdist
from lib import tsp_eval_cost



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    

    dataset: dict[str, np.ndarray] = dict(np.load(args.data))
    runs: int = args.runs
    num_workers: int = args.num_workers
    seed: int = args.seed
    generator = np.random.default_rng(seed)

    coords = dataset['coords']

    if 'opt_costs' in dataset.keys():
        opt_costs = dataset['opt_costs']
        mean_opt_cost = opt_costs.mean().item()
    elif 'opt_tours' in dataset.keys():
        opt_tours = dataset['opt_tours']
        opt_costs = tsp_eval_cost(opt_tours, np.array(cdist(coords)), num_workers=1)
        mean_opt_cost = opt_costs.mean().item()
    else:
        opt_costs = None
        mean_opt_cost = None


    _, costs = tsp_two_opt_multirun(coords, runs, generator, num_workers=num_workers)
    mean_cost = costs.mean().item()

    print(f'mean cost: {mean_cost:.6f}')
    if mean_opt_cost is not None:
        print(f'opt cost: {mean_opt_cost:.6f}')
        print(f'Gap: {(mean_cost - mean_opt_cost) / mean_opt_cost * 100:.6f} %')

