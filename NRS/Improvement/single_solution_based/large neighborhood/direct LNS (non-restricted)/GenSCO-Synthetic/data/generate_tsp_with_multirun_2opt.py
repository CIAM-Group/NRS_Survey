from lib import tsp_two_opt, tsp_eval_cost
import numpy as np
import jax
from helpers.coord_transform import cdist
import tqdm


cdist = jax.jit(cdist)


def tsp_two_opt_multirun(
    coords: np.ndarray,
    runs: int,
    generator: np.random.Generator,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    assert runs >= 1
    assert coords.ndim == 3
    assert coords.shape[-1] == 2

    batch_size, num_nodes, _ = coords.shape

    dist_mat = np.array(cdist(coords))    

    tours_bsf = None
    costs_bsf = np.full([batch_size], np.finfo(np.float32).max, dtype=np.float32)
    
    for _ in range(runs):
        tours = np.arange(num_nodes, dtype=np.int32)
        tours = np.broadcast_to(
            tours[None],
            [batch_size, num_nodes],
        )
        tours = generator.permuted(tours, axis=-1)
        tours = tsp_two_opt(
            tours, dist_mat, num_steps=2 * num_nodes,
            num_workers=num_workers,
        )
        costs = tsp_eval_cost(
            tours, dist_mat, num_workers=num_workers,
        )
        should_update = costs < costs_bsf
        costs_bsf = np.minimum(costs_bsf, costs)
        if tours_bsf is not None:
            tours_bsf = np.where(
                should_update[:, None],
                tours, tours_bsf,
            )
        else:
            tours_bsf = tours
    
    return tours_bsf, costs_bsf


def generate_and_save(
    num_nodes: int, num_instances: int,
    runs: int, batch_size: int, 
    num_workers: int, seed: int,
    save_path: str, overwrite: bool = False,
):
    if not overwrite:
        import os
        assert not os.path.exists(save_path)

    assert num_instances % batch_size == 0

    generator = np.random.default_rng(seed)

    coords = generator.uniform(
        0., 1., size=[num_instances, num_nodes, 2],
    ).astype(np.float32)
    
    results_list = [
        tsp_two_opt_multirun(
            coords[i * batch_size:(i + 1) * batch_size],
            runs, generator, num_workers,
        ) for i in tqdm.tqdm(range(num_instances // batch_size))
    ]
    tours = np.concat([result[0] for result in results_list], axis=0)
    costs = np.concat([result[1] for result in results_list], axis=0)
    mean_cost = costs.mean().item()
    print(f'Mean Cost: {mean_cost:.6f}')

    np.savez_compressed(
        save_path, coords=coords, opt_tours=tours,
    )



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, required=True)
    parser.add_argument('--num_instances', type=int, required=True)
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--num_workers', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)

    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()
    args = vars(args)

    generate_and_save(**args)
