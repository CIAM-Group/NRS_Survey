import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor
from models import MISModel, MISModelConfig
from training import load_ckpt
from helpers.mis_transform import edges2adj
from lib import mis_greedy_insert, mis_edges2neighbors, MISBatchedNeighbors
from decoding.continuous_sampling import sample_dpmpp_2m, sample_euler


def mis_flow_decode(
    dataset: dict[str, np.ndarray],
    model: MISModel,
    sampling_steps: int,
    flip_rate: tuple[float, float],
    cycles: int,
    runs: int,
    decode_intermediate: bool,
    batch_size: int,
    threads_over_batches: int | None,
    seed: int,
):
    num_workers = 1

    np.random.seed(seed)

    edges = dataset['edges']
    num_edges = np.any(edges != 0, axis=-1).astype(np.int32).sum(axis=-1)
    num_nodes = dataset['num_nodes']
    del dataset

    num_nodes_padded: int = num_nodes.max().item()
    num_instances = edges.shape[0]
    assert num_instances % (batch_size * threads_over_batches) == 0

    edges_list, num_edges_list, num_nodes_list = tuple(map(
        lambda x: np.split(x, num_instances // batch_size, axis=0),
        (edges, num_edges, num_nodes),
    ))
    del edges, num_edges, num_nodes

    adjmat_list = [
        edges2adj(edges, num_nodes_padded, dtype=jnp.float16)
        for edges in edges_list
    ]
    neighbors_list = [
        mis_edges2neighbors(
            edges, num_nodes, num_edges, tolist=False,
        ) for edges, num_nodes, num_edges in zip(edges_list, num_nodes_list, num_edges_list)
    ]

    @jax.jit
    def encode(
        adjmat: jax.Array,
        key: jax.Array,
    ):
        return model.encode(adjmat, key)

    @jax.jit
    def decode(
        features: jax.Array,
        solution: jax.Array,
        adjmat: jax.Array,
        num_nodes: jax.Array,
    ):
        timesteps = jnp.linspace(0., 1., sampling_steps + 1)
        def denoised_fn(solution: jax.Array, timestep: jax.Array):
            sigmoid_logits, _ = model.decode(features, solution, timestep, adjmat, num_nodes)
            return jax.nn.sigmoid(sigmoid_logits)
        if not decode_intermediate:
            solution_pred = sample_euler(denoised_fn, solution, timesteps)
            candidate_nodes = jnp.argsort(
                solution_pred, axis=-1, 
                stable=False, descending=True,
            )
            return candidate_nodes
        else:
            solution_pred, denoised_history = sample_euler(denoised_fn, solution, timesteps, need_history=True)
            candidate_nodes, history_candidate_nodes_group = jax.tree.map(
                lambda x: jnp.argsort(
                    x, axis=-1, 
                    stable=False, descending=True,
                ),
                (solution_pred, denoised_history),
            )
            return candidate_nodes, history_candidate_nodes_group
    
    def inference_fn(
        seed: int, num_nodes: np.ndarray,
        adjmat: jax.Array, neighbors: MISBatchedNeighbors,
        *, cycles: int,
    ):
        generator = np.random.default_rng(seed)
        features = encode(
            adjmat, 
            generator.integers(0, np.iinfo(np.uint32).max, size=[2], dtype=np.uint32)
        )
        num_nodes_np = num_nodes
        num_nodes = jax.device_put(num_nodes)
        storage: dict[int, np.ndarray] = {}
        def _single_run(r: int):
            generator = np.random.default_rng(seed + r * 888)

            candidate_nodes = np.broadcast_to(
                np.arange(num_nodes_padded, dtype=np.int32)[None],
                [batch_size, num_nodes_padded],
            )
            candidate_nodes = np.ascontiguousarray(generator.permuted(candidate_nodes, axis=-1))

            solutions, sizes = mis_greedy_insert(
                neighbors, num_nodes_np, num_nodes_padded, candidate_nodes,
                num_workers=num_workers,
            )
            
            best_sizes = sizes
            for _ in range(cycles):
                one_to_zero_rate = generator.uniform(*flip_rate, size=[batch_size, 1]).astype(np.float32)
                whether_flip = generator.binomial(1, one_to_zero_rate, size=[batch_size, num_nodes_padded]).astype('bool')
                solutions = np.where(
                    solutions,
                    np.logical_xor(solutions, whether_flip),
                    False,
                )
                if not decode_intermediate:
                    candidate_nodes = decode(features, solutions, adjmat, num_nodes)
                    candidate_nodes = np.array(candidate_nodes)
                    solutions, sizes = mis_greedy_insert(
                        neighbors, num_nodes_np, num_nodes_padded, candidate_nodes,
                        num_workers=num_workers,
                    )
                    best_sizes = np.maximum(best_sizes, sizes)
                else:
                    candidate_nodes, history_candidate_nodes_group = decode(features, solutions, adjmat, num_nodes)
                    candidate_nodes = np.array(candidate_nodes)
                    solutions, sizes = mis_greedy_insert(
                        neighbors, num_nodes_np, num_nodes_padded, candidate_nodes,
                        num_workers=num_workers,
                    )
                    best_sizes = np.maximum(best_sizes, sizes)
                    for history_candidate_nodes in history_candidate_nodes_group:
                        history_candidate_nodes = np.array(history_candidate_nodes)
                        _, sizes = mis_greedy_insert(
                            neighbors, num_nodes_np, num_nodes_padded, history_candidate_nodes,
                            num_workers=num_workers,
                        )
                        best_sizes = np.maximum(best_sizes, sizes)

            storage[r] = best_sizes

        threads = [threading.Thread(target=_single_run, args=[r]) for r in range(runs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        best_sizes = storage[0]
        for i in range(1, runs):
            this_sizes = storage[i]
            best_sizes = np.maximum(best_sizes, this_sizes)
        return best_sizes
    
    def run_batch(i: int) -> float:
        sizes = inference_fn(
            i * 888,
            num_nodes_list[i],
            adjmat_list[i],
            neighbors_list[i],
            cycles=cycles,
        )
        return sizes.mean().item()
    
    # warmup and triton autotune
    inference_fn(
        1234,
        num_nodes_list[0],
        adjmat_list[0],
        neighbors_list[0],
        cycles=5, 
    )
    
    sizes: list[float] = []
    if threads_over_batches is None or threads_over_batches == 1:
        for i in tqdm.tqdm(range(num_instances // batch_size)):
            batch_cost = run_batch(i)
            print(f'batch {i} cost mean: {batch_cost:.6f}')
            sizes.append(batch_cost)
    else:
        assert num_instances % (batch_size * threads_over_batches) == 0
        with ThreadPoolExecutor(max_workers=threads_over_batches) as pool:
            for j in tqdm.tqdm(range(num_instances // batch_size // threads_over_batches)):
                batch_id_start = j * threads_over_batches
                batch_id_end = batch_id_start + threads_over_batches
                submit_sizes = list(pool.map(run_batch, range(batch_id_start, batch_id_end)))
                for i, batch_size in zip(range(batch_id_start, batch_id_end), submit_sizes):
                    print(f'batch {i} size mean: {batch_size:.6f}')
                    sizes.append(batch_size)

    mean_size = sum(sizes) / len(sizes)
    print(f'Avg Size: {mean_size:.6f}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--sampling_steps', type=int, required=True)
    parser.add_argument('--flip_rate', type=eval, default=(0.25, 0.4))
    parser.add_argument('--cycles', type=int, required=True)
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--threads_over_batches', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--decode_intermediate', action='store_true', default=False)

    args = parser.parse_args()

    params, _, _, model_config, _, _ = load_ckpt(args.ckpt)
    model_config: MISModelConfig
    model = model_config.construct_model()
    graphdef = nnx.graphdef(model)
    model = nnx.merge(graphdef, params)
    dataset = dict(np.load(args.data))

    assert model_config.sigmoid_head

    mis_flow_decode(
        dataset, model,
        args.sampling_steps,
        args.flip_rate,
        args.cycles,
        args.runs,
        args.decode_intermediate,
        args.batch_size,
        args.threads_over_batches,
        args.seed,
    )
