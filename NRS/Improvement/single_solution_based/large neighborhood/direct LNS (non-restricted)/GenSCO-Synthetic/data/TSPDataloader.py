import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from jax.experimental import jax2tf
from functools import partial
from typing import Optional, Literal, Any, Mapping, Iterable, Sequence

PyTree = Any


def apply_two_opt(sol: jax.Array, step: jax.Array) -> jax.Array:
    def _apply_two_opt_unbatched(sol: jax.Array, step: jax.Array) -> jax.Array:
        num_nodes = sol.size
        idxs = (jnp.arange(num_nodes) - step.sum() - 2) % num_nodes
        sol_flipped_rearranged = jnp.flip(sol).at[idxs].get(unique_indices=True)
        new_sol = jnp.arange(num_nodes)
        new_sol = jnp.where(
            jnp.logical_and(step[0] < new_sol, new_sol <= step[1]),
            sol_flipped_rearranged,
            sol,
        )
        return new_sol
    
    assert jnp.issubdtype(step.dtype, jnp.signedinteger)
    if sol.ndim == step.ndim == 1:
        return jax.jit(_apply_two_opt_unbatched)(sol, step)
    elif sol.ndim == step.ndim == 2:
        return jax.vmap(_apply_two_opt_unbatched, in_axes=(0, 0))(sol, step)
    else:
        raise ValueError('Invalid input shape.')


def tour_shuffle(key: jax.typing.ArrayLike, tour: jax.Array, times: jax.Array | int) -> jax.Array:
    ''' Shuffle the tour by applying 2-opt operation for `times` times.
    '''

    if isinstance(times, jax.Array):
        assert times.size == 1
    assert tour.ndim == 1

    num_nodes = tour.size
    assert num_nodes >= 4

    def get_random_step(key: jax.typing.ArrayLike) -> jax.Array:
        key_1, key_2 = jax.random.split(key)
        first_step = jax.random.randint(key_1, shape=(), minval=0, maxval=num_nodes, dtype='int32')
        diff = jax.random.randint(key_2, shape=(), minval=0, maxval=num_nodes - 3, dtype='int32')
        second_step = (first_step + diff + 2) % num_nodes
        return jnp.sort(jnp.array([first_step, second_step], dtype=jnp.int32), stable=False)
    
    def shuffle_step(_i: int, state: dict[str, jax.typing.ArrayLike]) -> dict[str, jax.typing.ArrayLike]:
        key, tour = state['key'], state['tour']
        key, subkey = jax.random.split(key)
        step = get_random_step(subkey)
        new_tour = apply_two_opt(tour, step)
        return {'key': key, 'tour': new_tour}
    
    return jax.lax.fori_loop(
        0, times,
        shuffle_step,
        {'key': key, 'tour': tour},
    )['tour']


def reversed_mapping(mapping: jax.Array):
    assert mapping.ndim == 1
    L = mapping.size

    arange = jnp.arange(L, dtype=jnp.int32)
    result = jnp.zeros([L], dtype=jnp.int32)
    result = result.at[mapping].set(arange)
    return result


def sol2var(sol: jax.Array, normalization_base_idx: jax.Array | int, shift: jax.Array | float):
    assert sol.ndim == 1
    assert jnp.issubdtype(sol.dtype, jnp.signedinteger)
    num_nodes = sol.size
    with jax.experimental.enable_x64(False):
        dtype = jnp.float32
        var = reversed_mapping(sol)
        var = (var - var[normalization_base_idx] + num_nodes) % num_nodes
        var = (var.astype(dtype) + shift.astype(dtype)) * (2 * jnp.pi / num_nodes)
        return jnp.stack([jnp.cos(var), jnp.sin(var)], axis=-1).astype(jnp.float32)


def sample_from_poly(key: jax.Array, shape: Sequence[int], k: int | float, dtype: jnp.dtype | str = jnp.float32):
    ''' Sample from PDF: p(x) = (k + 1)(1 - x)^k .
    '''

    if k != 0 and k != 0.:
        def inverse_cdf(u, k):
            return 1 - (1 - u) ** (1 / (k + 1))
        
        uniform_samples = jax.random.uniform(key, shape=shape, dtype=dtype)
        samples = inverse_cdf(uniform_samples, k)
        return samples
    else:
        return jax.random.uniform(key, shape=shape, dtype=jnp.float32)


@partial(
    jax.jit, inline=True, backend='cpu', 
    static_argnames=[
        'noise_type', 
        'timestep_distribution_k',
        'target_disruption',
    ],
)
def _preprocess_sol_type_edge_unbatched(
    key: jax.Array,
    target: jax.Array,
    shuffle_range: jax.Array,
    noise_type: Literal['shuffle', 'randperm', 'none'], 
    timestep_distribution_k: int | float,
    target_disruption: tuple[int, int] | None,
) -> tuple[jax.Array, jax.Array]:
    assert target.ndim == 1
    assert shuffle_range.shape == (2,)

    keys = jax.random.split(key, num=5)
    num_shuffle = jax.random.randint(keys[0], shape=(), minval=shuffle_range[0], maxval=shuffle_range[1], dtype=jnp.int32)
    
    if_flip_current = jax.random.randint(keys[2], shape=(), minval=0, maxval=2, dtype=jnp.int8)
    if_flip_target = jax.random.randint(keys[3], shape=(), minval=0, maxval=2, dtype=jnp.int8)
    if noise_type == 'shuffle':
        current = tour_shuffle(key=keys[1], tour=target, times=num_shuffle)
        current = jax.lax.cond(
            if_flip_current,
            lambda: jnp.flip(current),
            lambda: current,
        )
    elif noise_type == 'randperm':
        current = jax.random.permutation(key=keys[1], x=target.size)
    elif noise_type == 'none':
        current = target.copy()
    else:
        raise
    target = jax.lax.cond(
        if_flip_target,
        lambda: jnp.flip(target),
        lambda: target,
    )
    # timestep = jax.random.uniform(key=keys[4], shape=(), dtype=jnp.float32)
    timestep = sample_from_poly(keys[4], (), timestep_distribution_k, dtype=jnp.float32)

    # to prevent over-confidence
    if target_disruption is not None:
        assert target_disruption[1] > target_disruption[0] >= 0
        if target_disruption[0] + 1 == target_disruption[1]:
            noisy_target = tour_shuffle(key=key, tour=target, times=target_disruption[0])
        else:
            _, subkey = jax.random.split(keys[-1])
            noisy_target = tour_shuffle(
                key=key, tour=target, 
                times=jax.random.randint(subkey, (), target_disruption[0], target_disruption[1], dtype=jnp.int32),
            )
        target = jnp.concat([target, noisy_target], axis=-1)

    return current, target, timestep


@partial(
    jax.jit, inline=True, backend='cpu', 
    static_argnames=[
        'noise_type', 
        'timestep_distribution_k',
        'target_disruption',
    ],
)
def preprocess_sol_type_edge(
    key: jax.Array,
    targets: jax.Array,
    random_two_opt_steps_range: jax.Array,
    noise_type: Literal['shuffle', 'randperm', 'none'],
    timestep_distribution_k: int | float,
    target_disruption: tuple[int, int] | None,
) -> tuple[jax.Array, ...]:
    assert random_two_opt_steps_range.shape == (2,)
    if targets.ndim == 2:
        batch_size = targets.shape[0]
        if batch_size > 1:
            keys = jax.random.split(key=key, num=batch_size)
            return jax.vmap(
                partial(
                    _preprocess_sol_type_edge_unbatched, 
                    noise_type=noise_type, 
                    timestep_distribution_k=timestep_distribution_k,
                    target_disruption=target_disruption,
                ), 
                in_axes=(0, 0, None),
            )(keys, targets, random_two_opt_steps_range)
        else:
            targets = targets.squeeze(0)
            result = _preprocess_sol_type_edge_unbatched(
                key, targets, random_two_opt_steps_range, 
                noise_type, timestep_distribution_k, target_disruption,
            )
            return jax.tree.map(lambda x: x[None, ...], result)
    else:
        assert targets.ndim == 1
        return _preprocess_sol_type_edge_unbatched(
            key, targets, random_two_opt_steps_range, 
            noise_type, timestep_distribution_k, target_disruption,
        )


def rotate_coords(key: jax.Array, coords: jax.Array) -> jax.Array:
    assert coords.ndim in [2, 3]
    def _get_rot_matrix(theta: jax.Array) -> jax.Array:
        return jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]], dtype='float32')
    if coords.ndim == 3:
        batch_size, _num_nodes, _dim = coords.shape
        thetas = jax.random.uniform(key=key, shape=(batch_size,), minval=0, maxval=2 * jnp.pi, dtype='float32')
        return jax.vmap(lambda coord, theta: jnp.dot(coord, _get_rot_matrix(theta), precision=jax.lax.Precision.HIGHEST), in_axes=(0, 0))(coords, thetas)
    else:
        theta = jax.random.uniform(key=key, shape=(), minval=0, maxval=2 * jnp.pi, dtype='float32')
        return jnp.dot(coords, _get_rot_matrix(theta), precision=jax.lax.Precision.HIGHEST)

def reflect_x(inputs: jax.Array) -> jax.Array:
    return inputs.at[..., 0].set(- inputs[..., 0])


def reflect_y(inputs: jax.Array) -> jax.Array:
    return inputs.at[..., 1].set(- inputs[..., 1])


def median(xs: jax.Array) -> jax.Array:
    ''' Calculate the median along the last axis.
    :param xs: shape (..., d)
    :return: median of xs, with shape (...)
    '''
    xs = xs.sort(axis=-1)
    d = xs.shape[-1]
    return xs[..., d // 2]


def normalize(inputs: jax.Array, centering_method: Literal['mean', 'median'] = 'mean') -> jax.Array:
    # center: for every input feature, subtract the mean of all nodes
    if centering_method == 'mean':
        inputs = inputs - inputs.mean(axis=-2, keepdims=True)
    elif centering_method == 'median':
        inputs = inputs - jax.vmap(median, in_axes=-1, out_axes=-1)(inputs)[:, None]
    else:
        raise

    # scale by the mean L2-norm of all nodes
    inputs_norm = jnp.linalg.norm(inputs, axis=-1, keepdims=True, ord=2)
    inputs_norm_mean = inputs_norm.mean(axis=-2, keepdims=True)
    inputs = inputs / inputs_norm_mean

    return inputs


class TSPDataloader:
    def __init__(
        self,
        datasets: Mapping[str, np.ndarray],
        batch_size: int,
        noise_type: Literal['shuffle', 'randperm', 'none'] = 'shuffle',
        timestep_distribution_k: float | int = 0,
        random_two_opt_steps_range: tuple[int, int] | None = None,
        target_disruption: tuple[int, int] | None = None,
        data_argument: Literal[0, 1, 2, 3] = 3,
        divide_first_axis_into_n_parts: Optional[int] = None,
        num_workers: int | None = 16,
    ) -> None:
        datasets = dict(datasets)
        self.coords = datasets['coords']
        self.opt_tours = datasets['opt_tours']
        assert self.coords.ndim == 3 and self.opt_tours.ndim == 2
        assert self.coords.shape[2] == 2
        self.num_nodes = self.coords.shape[1]
        self.num_instances = self.coords.shape[0]
        if random_two_opt_steps_range is None:
            random_two_opt_steps_range = (1, max(2, self.num_nodes))
        assert random_two_opt_steps_range[1] > random_two_opt_steps_range[0] > 0
        self.random_two_opt_steps_range = np.array(random_two_opt_steps_range, dtype='int32')
        self.data_argument = int(data_argument)

        assert batch_size > 0
        self.batch_size = batch_size
        self.steps_per_epoch = self.num_instances // batch_size

        if divide_first_axis_into_n_parts is not None:
            assert divide_first_axis_into_n_parts >= 1
            assert batch_size % divide_first_axis_into_n_parts == 0
            num_parts = divide_first_axis_into_n_parts
            def divide_first_axis_fn(inputs: PyTree) -> None:
                a, b, c = inputs
                def _inner(x: np.ndarray):
                    shape = x.shape
                    assert shape[0] % num_parts == 0
                    new_shape = (num_parts, shape[0] // num_parts) + shape[1:]
                    return x.resize(new_shape)
                _inner(a), _inner(b), _inner(c)
            self.divide_first_axis_fn = divide_first_axis_fn
        self.divide_first_axis_into_n_parts = divide_first_axis_into_n_parts

        def _preprocess(
            key: jax.Array,
            targets: jax.Array,
            coords: jax.Array,
            random_two_opt_steps_range: jax.Array,
            data_argument: Literal[0, 1, 2, 3],
            divide_first_axis_into_n_parts: Optional[int],  # unused
        ) -> tuple[jax.Array, jax.Array]:
            del divide_first_axis_into_n_parts

            match data_argument:
                case 3:
                    key, subkey = jax.random.split(key)
                    coords = rotate_coords(subkey, coords)
                    key, subkey = jax.random.split(key)
                    whether_flip_coords = jax.random.randint(subkey, shape=(), minval=0, maxval=2, dtype='int32')
                    coords = jax.lax.cond(
                        whether_flip_coords,
                        lambda: reflect_x(coords),
                        lambda: coords,
                    )
                case 2:
                    key, subkey = jax.random.split(key)
                    coords = rotate_coords(subkey, coords)
                case 1:
                    key, subkey = jax.random.split(key)
                    whether_flip_coords = jax.random.randint(subkey, shape=(), minval=0, maxval=2, dtype='int32')
                    coords = jax.lax.cond(
                        whether_flip_coords,
                        lambda: reflect_x(coords),
                        lambda: coords,
                    )
                    key, subkey = jax.random.split(key)
                    coords_rotate = jax.random.randint(subkey, shape=(), minval=0, maxval=4, dtype='int32')
                    coords = jax.lax.switch(
                        index=coords_rotate,
                        branches=[
                            lambda: coords,
                            lambda: jnp.stack([-coords[..., 1], coords[..., 0]], axis=-1),
                            lambda: -coords, 
                            lambda: jnp.stack([coords[..., 1], -coords[..., 0]], axis=-1),
                        ],
                    )
                case 0:
                    pass
                case _:
                    raise ValueError(f'Got invalid data argument level: {data_argument}.')
            coords = normalize(coords, centering_method='mean')

            key, subkey = jax.random.split(key)
            current, target, timestep = preprocess_sol_type_edge(
                key, targets, random_two_opt_steps_range, 
                noise_type=noise_type,
                timestep_distribution_k=timestep_distribution_k,
                target_disruption=target_disruption,
            )

            return coords, current, target, timestep

        self.preprocess = partial(
            jax.jit(_preprocess, static_argnums=(4, 5), backend='cpu'),
            random_two_opt_steps_range=self.random_two_opt_steps_range,
            data_argument=self.data_argument,
            divide_first_axis_into_n_parts=self.divide_first_axis_into_n_parts,
        )
        self.preprocess = jax2tf.convert(self.preprocess, with_gradient=False, native_serialization_platforms=['cpu'])
        self.preprocess = tf.function(self.preprocess, autograph=False, jit_compile=True)
        self.num_workers = num_workers
        
    def __iter__(self):
        keys = np.random.randint(0, np.iinfo(np.uint32).max, size=(self.num_instances, 2), dtype=np.uint32)
        datasets = tf.data.Dataset.from_tensor_slices((keys, self.opt_tours, self.coords))
        datasets = datasets.shuffle(buffer_size=100, seed=np.random.randint(0, np.iinfo(np.uint32).max, size=tuple()))
        datasets = datasets.map(self.preprocess, num_parallel_calls=self.num_workers)
        datasets = datasets.batch(self.batch_size)
        datasets = datasets.take(self.steps_per_epoch)
        if self.divide_first_axis_into_n_parts is None:
            return iter(datasets.as_numpy_iterator())
        else:
            def iterator_wrapper(iterator: Iterable):
                iterator = iter(iterator)
                while True:
                    try:
                        result = next(iterator)
                        self.divide_first_axis_fn(result)
                        yield result
                    except StopIteration:
                        return
            return iter(iterator_wrapper(datasets.as_numpy_iterator()))
    