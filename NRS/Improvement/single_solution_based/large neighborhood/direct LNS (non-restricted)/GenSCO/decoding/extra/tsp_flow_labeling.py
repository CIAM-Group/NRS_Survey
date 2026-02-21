# TODO: test directly casting params to bf16 instead of casting every pass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import threading
import tqdm
from concurrent.futures import ThreadPoolExecutor

from models import TSPModel, TSPModelConfig
from helpers.coord_transform import (
    normalize,
    cdist,
)
from helpers.tsp_transform import get_costs
from helpers import sol2adj
from decoding.continuous_sampling import (
    sample_euler,
    sample_dpmpp_2m,
    get_linear_timesteps,
)
from training import load_ckpt
from lib import tsp_greedy_insert, tsp_double_two_opt, tsp_eval_cost, tsp_random_two_opt
from decoding.utils import (
    convert_heatmap_dtype, 
    CoordDynamicArgument as DynamicArgument,
)
from decoding.gpu_heuristics.tsp import two_opt as gpu_tsp_two_opt
from functools import partial


def tsp_flow_searching_decode(
    dataset: dict[str, np.ndarray],
    model: TSPModel, model_config: TSPModelConfig,
    sampling_steps: int,
    two_opt_steps: int,
    random_two_opt_steps_range: tuple[int, int],
    cycles: int,
    batch_size: int,
    runs: int,
    heatmap_dtype: jax.typing.DTypeLike,
    topk: int | None,
    argument_level: int,
    threads_over_batches: int | None,
    seed: int,
):      
    np.random.seed(seed)

    num_workers = 8

    coords = dataset['coords']
    dist_mat_np = np.array(jax.jit(cdist)(coords))

    
    num_instances, num_nodes, _ = coords.shape

    jitted_gpu_two_opt = jax.jit(partial(gpu_tsp_two_opt, num_steps=two_opt_steps))

    @jax.jit
    def _encode(raw_features: jax.Array):
        features = normalize(raw_features, centering_method='mean')
        features = model.encode(features)
        return features
    
    @jax.jit
    def _decode(
        features: jax.Array, 
        sols: jax.Array,
    ):
        timesteps = get_linear_timesteps(sampling_steps)
        def denoised_fn(adjmat: jax.Array, timestep: jax.Array):
            logits = model.decode(features, timestep, adjmat)
            adjmat = jax.nn.softmax(logits, axis=-1) * 2
            return adjmat
        adjmat = sol2adj(sols, dtype=jnp.float16 if num_nodes > 128 else jnp.float32)
        adjmat_pred = sample_dpmpp_2m(denoised_fn, adjmat, timesteps)
        adjmat_pred = convert_heatmap_dtype(adjmat_pred, dtype=heatmap_dtype)
        if topk is None or topk <= 0:
            candidate_edges = jnp.argsort(
                adjmat_pred.reshape(batch_size, -1),
                axis=-1, descending=True, stable=False,
            )
        else:
            _, candidate_edges = jax.lax.top_k(
                adjmat_pred.reshape(batch_size, -1),
                k=topk,
            )
        candidate_edges = jnp.stack(jnp.divmod(candidate_edges, num_nodes), axis=-1)
        return candidate_edges
    
    encode = _encode
    decode = _decode

    _get_costs_cpu = partial(tsp_eval_cost, num_workers=num_workers)

    def inference_fn(seed: int, coords: np.ndarray, dist_mat: np.ndarray):
        # features: jax.Array = encode(coords)
        features_manager = DynamicArgument(encode, coords, argument_level=argument_level)
        storage: dict[int, np.ndarray] = {}
        dist_mat_gpu = jnp.array(dist_mat)
        def _single_run(r: int):
            tours = [i for i in range(num_nodes)]
            tours = jnp.array(tours, dtype=jnp.int32)
            key = jax.random.key(seed + r)
            keys = jax.random.split(key, batch_size)
            tours = jax.vmap(jax.random.permutation, in_axes=(0, None))(keys, tours)
            tours = np.array(tours)

            tours_bsf = tours

            min_costs = np.full((batch_size,), np.finfo(np.float32).max, dtype=np.float32)
            generator = np.random.default_rng(seed + r * 7)
            for i in range(cycles):
                candidate_edges = decode(features_manager(generator), tours)
                candidate_edges = np.array(candidate_edges)
                tours = tsp_greedy_insert(candidate_edges, num_nodes, num_workers=num_workers)

                tours = jitted_gpu_two_opt(
                    tours, dist_mat_gpu,
                )
                tours = np.array(tours)

                tours_disrupted = tsp_random_two_opt(
                    seed + r + i * 10, 
                    tours, generator.integers(*random_two_opt_steps_range, size=batch_size, dtype=np.int32), 
                    num_workers=num_workers,
                )
                costs = _get_costs_cpu(tours, dist_mat)

                tours_bsf = np.where(
                    (costs < min_costs)[:, None],
                    tours, tours_bsf,
                )

                min_costs = np.minimum(min_costs, costs)
                tours = tours_disrupted
            storage[r] = (min_costs, tours_bsf)
        
        threads = [threading.Thread(target=_single_run, args=[r]) for r in range(runs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        min_costs, best_tours = storage[0]
        for i in range(1, runs):
            this_costs, this_tours = storage[i]
            best_tours = np.where(
                (this_costs < min_costs)[:, None],
                this_tours,
                best_tours,                    
            )
            min_costs = np.minimum(min_costs, this_costs)
        return min_costs, best_tours
    
    def run_batch(i: int) -> float:
        cost, tours = inference_fn(
            seed + i * 888,
            coords[i * batch_size:(i + 1) * batch_size], 
            dist_mat_np[i * batch_size:(i + 1) * batch_size],
        )
        return cost.mean().item(), tours
    
    # triton autotune and warmup
    for _ in range(5):
        i = 0
        features = encode(coords[i * batch_size:(i + 1) * batch_size])
        tours = [i for i in range(num_nodes)]
        tours = jnp.array(tours, dtype=jnp.int32)
        key = jax.random.key(0)
        keys = jax.random.split(key, batch_size)
        tours = jax.vmap(jax.random.permutation, in_axes=(0, None))(keys, tours)
        candidate_edges = decode(features, tours)
        del features, candidate_edges
    
    costs: list[float] = []
    tours: list[np.ndarray] = []
    assert num_instances % (batch_size * threads_over_batches) == 0
    with ThreadPoolExecutor(max_workers=threads_over_batches) as pool:
        for j in tqdm.tqdm(range(num_instances // batch_size // threads_over_batches)):
            batch_id_start = j * threads_over_batches
            batch_id_end = batch_id_start + threads_over_batches
            submit_results = list(pool.map(run_batch, range(batch_id_start, batch_id_end)))
            for i, (batch_cost, batch_tours) in zip(range(batch_id_start, batch_id_end), submit_results):
                print(f'batch {i} cost mean: {batch_cost:.6f}')
                costs.append(batch_cost)
                tours.append(batch_tours)
    mean_cost = sum(costs) / len(costs)
    print(f'mean cost: {mean_cost:.6f}')
    
    tours = np.concatenate(tours, axis=0)

    return mean_cost, tours


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--sampling_steps', type=int, required=True)
    parser.add_argument('--two_opt_steps', type=int, required=True)
    # parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cycles', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--random_two_opt_steps_range', type=eval, required=True)
    parser.add_argument('--heatmap_dtype', type=str, default='float32')
    parser.add_argument('--topk', type=eval, default=None)
    parser.add_argument('--argument_level', type=int, default=0)
    parser.add_argument('--threads_over_batches', type=int, default=1)


    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--num_nodes', type=int, required=True)
    parser.add_argument('--num_instances', type=int, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--submit_batch_size', type=int, required=True)

    args = parser.parse_args()

    params, _, _, model_config, _, _ = load_ckpt(args.ckpt)
    model_config: TSPModelConfig
    model = model_config.construct_model()
    graphdef = nnx.graphdef(model)
    model = nnx.merge(graphdef, params)
    # dataset = dict(np.load(args.data))


    generator = np.random.default_rng(args.seed)
    num_instances: int = args.num_instances
    num_nodes: int = args.num_nodes
    save_path: str = args.save_path
    submit_batch_size: int = args.submit_batch_size
    import os
    assert not os.path.exists(save_path) 
    assert num_instances % submit_batch_size == 0

    # args = vars(args)
    # del args['ckpt']
    # del args['seed'], args['num_instances'], args['save_path'], args['num_nodes']
    # del args['submit_batch_size']

    coords = generator.uniform(0., 1., size=[num_instances, num_nodes, 2]).astype(np.float32)
    tours = []
    costs = []

    for i in range(num_instances // submit_batch_size):
        submit_cost, submit_tours = tsp_flow_searching_decode(
            dict(coords=coords[i * submit_batch_size:(i + 1) * submit_batch_size]), 
            model, model_config, 
            sampling_steps=args.sampling_steps, 
            two_opt_steps=args.two_opt_steps,
            random_two_opt_steps_range=args.random_two_opt_steps_range,
            cycles=args.cycles,
            batch_size=args.batch_size,
            runs=args.runs,
            heatmap_dtype=args.heatmap_dtype,
            topk=args.topk,
            argument_level=args.argument_level,
            threads_over_batches=args.threads_over_batches,
            seed=args.seed,
        )
        tours.append(submit_tours)
        costs.append(submit_cost)

    tours = np.concatenate(tours, axis=0)
    mean_cost = sum(costs) / len(costs)
    print(f'Labeled Mean Cost: {mean_cost:.6f}')

    np.savez_compressed(
        save_path,
        coords=coords, opt_tours=tours,
    )        
    
