# TODO: test directly casting params to bf16 instead of casting every pass

# ! 
import os
from decoding.tsplib_dataset import build_tsplib_dataset

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
from lib import tsp_greedy_insert, tsp_double_two_opt, tsp_eval_cost
from decoding.utils import (
    convert_heatmap_dtype, 
    CoordDynamicArgument as DynamicArgument,
)
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

    if threads_over_batches is None:
        threads_over_batches = 1

    num_workers = max(1, 32 // (runs * threads_over_batches))

    # coords = dataset['coords']
    # dist_mat_np = np.array(jax.jit(cdist)(coords))

    # !
    coords = dataset['coords']

    if 'dist_mat' in dataset:
        # 如果是 TSPLIB 构造的 dataset，直接用离散后的距离矩阵
        dist_mat_np = np.array(dataset['dist_mat'])
    else:
        # 兼容原来的 .npz 数据集：连续欧氏距离
        dist_mat_np = np.array(jax.jit(cdist)(coords))


    if 'opt_costs' in dataset.keys():
        opt_costs = dataset['opt_costs']
        mean_opt_cost = opt_costs.mean().item()
    elif 'opt_tours' in dataset.keys():
        opt_tours = dataset['opt_tours']
        opt_costs = jax.jit(get_costs, backend='cpu')(opt_tours, dist_mat_np)
        mean_opt_cost = opt_costs.mean().item()
    else:
        opt_costs = None
        mean_opt_cost = None
    
    num_instances, num_nodes, _ = coords.shape
    cpu_device = jax.devices('cpu')[0]

    @jax.jit
    def encode(raw_features: jax.Array):
        features = normalize(raw_features, centering_method='mean')
        features = model.encode(features)
        return features
    
    @jax.jit
    def decode(
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
    
    _get_costs_cpu = partial(tsp_eval_cost, num_workers=num_workers)

    def inference_fn(seed: int, coords: np.ndarray, dist_mat: np.ndarray):
        # features: jax.Array = encode(coords)
        features_manager = DynamicArgument(encode, coords, argument_level=argument_level)
        storage: dict[int, np.ndarray] = {}
        def _single_run(r: int):
            tours = [i for i in range(num_nodes)]
            tours = jnp.array(tours, dtype=jnp.int32)
            key = jax.random.key(seed + r)
            keys = jax.random.split(key, batch_size)
            tours = jax.vmap(jax.random.permutation, in_axes=(0, None))(keys, tours)
            tours = np.array(tours)

            min_costs = np.full((batch_size,), np.finfo(np.float32).max, dtype=np.float32)
            generator = np.random.default_rng(seed + r * 7)
            for i in range(cycles):
                candidate_edges = decode(features_manager(generator), tours)
                candidate_edges = np.array(candidate_edges)
                tours = tsp_greedy_insert(candidate_edges, num_nodes, num_workers=num_workers)
                tours, tours_disrupted = tsp_double_two_opt(
                    seed + r + i * 10, 
                    tours, dist_mat,
                    two_opt_steps, 
                    random_steps=generator.integers(*random_two_opt_steps_range, size=batch_size, dtype=np.int32), 
                    num_workers=num_workers,
                )
                costs = _get_costs_cpu(tours, dist_mat)
                min_costs = np.minimum(min_costs, costs)
                tours = tours_disrupted
            storage[r] = min_costs
        
        threads = [threading.Thread(target=_single_run, args=[r]) for r in range(runs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        min_costs = storage[0]
        for i in range(1, runs):
            this_costs = storage[i]
            min_costs = np.minimum(min_costs, this_costs)
        return min_costs
    
    def run_batch(i: int) -> float:
        cost = inference_fn(
            seed + i * 888,
            coords[i * batch_size:(i + 1) * batch_size], 
            dist_mat_np[i * batch_size:(i + 1) * batch_size],
        )
        return cost.mean().item()
    
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
    if threads_over_batches is None or threads_over_batches == 1:
        for i in tqdm.tqdm(range(num_instances // batch_size)):
            batch_cost = run_batch(i)
            print(f'batch {i} cost mean: {batch_cost:.6f}')
            costs.append(batch_cost)
    else:
        assert num_instances % (batch_size * threads_over_batches) == 0
        with ThreadPoolExecutor(max_workers=threads_over_batches) as pool:
            for j in tqdm.tqdm(range(num_instances // batch_size // threads_over_batches)):
                batch_id_start = j * threads_over_batches
                batch_id_end = batch_id_start + threads_over_batches
                submit_costs = list(pool.map(run_batch, range(batch_id_start, batch_id_end)))
                for i, batch_cost in zip(range(batch_id_start, batch_id_end), submit_costs):
                    print(f'batch {i} cost mean: {batch_cost:.6f}')
                    costs.append(batch_cost)

    mean_cost = sum(costs) / len(costs)
    print(f'mean cost: {mean_cost:.6f}')
    if mean_opt_cost is not None:
        print(f'opt cost: {mean_opt_cost:.6f}')
        print(f'Gap: {(mean_cost - mean_opt_cost) / mean_opt_cost * 100:.6f} %')

    return mean_cost


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--sampling_steps', type=int, required=True)
    parser.add_argument('--two_opt_steps', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cycles', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--random_two_opt_steps_range', type=eval, required=True)
    parser.add_argument('--heatmap_dtype', type=str, default='float32')
    parser.add_argument('--topk', type=eval, default=None)
    parser.add_argument('--argument_level', type=int, default=0)
    parser.add_argument('--threads_over_batches', type=int, default=1)

    args = parser.parse_args()

    params, _, _, model_config, _, _ = load_ckpt(args.ckpt)
    model_config: TSPModelConfig
    model = model_config.construct_model()
    graphdef = nnx.graphdef(model)
    model = nnx.merge(graphdef, params)

    # dataset = dict(np.load(args.data))

    # ! 支持两种 data 形式：
    # 1) .npz：原有随机数据集
    # 2) 目录：按 ICAM 的方式读取 TSPLIB .tsp
    if os.path.isdir(args.data):
        # TSPLIB 目录：比如 /public/home/.../survey_bench_tsp
        dataset = build_tsplib_dataset(args.data)
    elif args.data.endswith(".npz"):
        dataset = dict(np.load(args.data))
    else:
        raise ValueError(f"Unsupported --data input: {args.data!r}. "
                         f"Expect a .npz file or a directory containing .tsp files.")


    tsp_flow_searching_decode(
        dataset, model, model_config,
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
    
