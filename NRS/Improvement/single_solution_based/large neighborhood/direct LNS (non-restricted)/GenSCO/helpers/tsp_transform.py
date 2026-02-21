import jax
import jax.numpy as jnp
import numpy as np


def _two_opt_unbatched(sol: jax.Array, step: jax.Array) -> jax.Array:
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


def two_opt(sol: jax.Array, step: jax.Array) -> jax.Array:
    assert (
        step.dtype in {jnp.dtype('int8'), jnp.dtype('int16'), jnp.dtype('int32'), jnp.dtype('int64')}
    ), f'step must be signed integer, got dtype {step.dtype}'

    if sol.ndim == step.ndim == 1:
        return jax.jit(_two_opt_unbatched)(sol, step)
    elif sol.ndim == step.ndim == 2:
        return jax.vmap(_two_opt_unbatched, in_axes=(0, 0))(sol, step)
    else:
        raise ValueError('Invalid input shape.')
    

def _get_step_improvement_unbatched(sol: jax.Array, step: jax.Array, dist_mat: jax.Array) -> jax.Array:
    num_nodes = dist_mat.shape[0]
    return (
        dist_mat[sol[step[0]], sol[step[0] + 1]] + dist_mat[sol[step[1]], sol[(step[1] + 1) % num_nodes]]
        - dist_mat[sol[step[0]], sol[step[1]]] - dist_mat[sol[step[0] + 1], sol[(step[1] + 1) % num_nodes]]
    )


def get_step_improvement(sol: jax.Array, step: jax.Array, dist_mat: jax.Array) -> jax.Array:
    if sol.ndim == 1 and step.ndim == 1 and dist_mat.ndim == 2:
        return jax.jit(_get_step_improvement_unbatched)(sol, step, dist_mat)
    elif sol.ndim == 2 and step.ndim == 2 and dist_mat.ndim == 3:
        return jax.vmap(_get_step_improvement_unbatched, in_axes=(0, 0, 0))(sol, step, dist_mat)
    else:
        raise ValueError('Invalid input shape.')


def two_opt_with_improvement_value(sol: jax.Array, step: jax.Array, dist_mat: jax.Array) -> tuple[jax.Array, jax.Array]:
    return two_opt(sol, step), get_step_improvement(sol, step, dist_mat)


# alias
two_opt_with_cost_reduction = two_opt_with_improvement_value


def _get_costs_unbatched(sol: jax.Array, dist_mat: jax.Array) -> jax.Array:
    assert (
        sol.ndim == 1 and dist_mat.ndim == 2
    ), f'sol must be 1D, got shape {sol.shape} | dist_mat must be 2D, got shape {dist_mat.shape}'

    return jnp.sum(dist_mat[sol, jnp.roll(sol, shift=1, axis=0)])


def get_costs(sols: jax.Array, dist_mat: jax.Array) -> jax.Array:
    if sols.ndim == 1 and dist_mat.ndim == 2:
        return jax.jit(_get_costs_unbatched)(sols, dist_mat)
    elif sols.ndim == 2 and dist_mat.ndim == 3:
        return jax.vmap(_get_costs_unbatched, in_axes=(0, 0))(sols, dist_mat)
    else:
        raise ValueError('Invalid input shape.')


def get_cost_numpy(sol: np.ndarray, dist_mat: np.ndarray) -> np.ndarray:
    return dist_mat[sol, np.roll(sol, shift=1, axis=0)].sum()


def _get_costs_with_coords_unbatched(sol: jax.Array, coord: jax.Array) -> jax.Array:
    assert (
        sol.ndim == 1 and coord.ndim == 2
    ), f'sol must be 1D, got shape {sol.shape} | coord must be 2D, got shape {coord.shape}'

    coord_reordered = coord[sol]
    return jnp.sum(jnp.linalg.norm(coord_reordered - jnp.roll(coord_reordered, shift=1, axis=0), axis=-1))


def get_costs_with_coords(sols: jax.Array, coords: jax.Array) -> jax.Array:
    if sols.ndim == 1 and coords.ndim == 2:
        return jax.jit(_get_costs_with_coords_unbatched)(sols, coords)
    elif sols.ndim == 2 and coords.ndim == 3:
        return jax.vmap(_get_costs_with_coords_unbatched, in_axes=(0, 0))(sols, coords)
    else:
        raise ValueError('Invalid input shape.') 


def tour_shuffle(key: jax.typing.ArrayLike, tour: jax.Array, times: jax.Array) -> jax.Array:
    '''
    Shuffle the tour by applying 2-opt operation for `times` times.
    '''
    assert times.size == 1
    assert tour.ndim == 1

    num_nodes = tour.size
    assert num_nodes >= 4

    def get_random_step(key: jax.typing.ArrayLike) -> jax.Array:
        key_1, key_2 = jax.random.split(key)
        first_step = jax.random.randint(key_1, shape=(), minval=0, maxval=num_nodes, dtype='int32')
        diff = jax.random.randint(key_2, shape=(), minval=0, maxval=num_nodes - 3, dtype='int32')
        second_step = (first_step + diff + 2) % num_nodes
        return jnp.sort(jnp.array([first_step, second_step], dtype='int32'), stable=False)
    
    def shuffle_step(_i: int, state: dict[str, jax.typing.ArrayLike]) -> dict[str, jax.typing.ArrayLike]:
        key, tour = state['key'], state['tour']
        key, subkey = jax.random.split(key)
        step = get_random_step(subkey)
        new_tour = two_opt(tour, step)
        return {'key': key, 'tour': new_tour}
    
    return jax.lax.fori_loop(
        0, times,
        shuffle_step,
        {'key': key, 'tour': tour},
    )['tour']


def get_two_opt_mask(num_nodes: int):
    _arange = jnp.arange(num_nodes, dtype='int16')
    mask = jnp.abs(_arange.reshape(-1, 1) - _arange.reshape(1, -1))
    mask = jnp.logical_or(mask <= 1, mask == num_nodes - 1)
    mask_lower_left = _arange.reshape(-1, 1) > _arange.reshape(1, -1)
    mask = jnp.logical_or(mask, mask_lower_left)
    return mask

