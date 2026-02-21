import jax
import jax.numpy as jnp

from functools import partial


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


def _two_opt_post_processing_argmax_impl_unbatched(sol: jax.Array, dist_mat: jax.Array, num_steps: int, unroll: int = 1) -> jax.Array:
    assert sol.ndim == 1 and dist_mat.ndim == 2
    num_nodes = sol.size
    assert num_nodes >= 5, 'for num_nodes < 5, use enumeration instead.'
    assert dist_mat.shape == (num_nodes, num_nodes)
    assert num_steps >= 0

    def _update_once(_i, sol: jax.Array):
        # apply
        # r[i, j] = m[i, i + 1] + m[j, j + 1] - m[i, j] - m[i + 1, j + 1]
        # r       = a           + b           - m       - n  

        m = dist_mat[sol, :][:, sol]
        t = dist_mat[sol, jnp.roll(sol, shift=-1)]      # avoid relying on m
        a = t.reshape(-1, 1)    # implicit broadcasting
        b = t.reshape(1, -1)
        n = jnp.roll(m, shift=(-1, -1), axis=(0, 1))
        r = a + b - m - n

        _arange = jnp.arange(num_nodes, dtype='int16')
        mask = jnp.abs(_arange.reshape(-1, 1) - _arange.reshape(1, -1))
        mask = jnp.logical_or(mask <= 1, mask == num_nodes - 1)
        mask_lower_left = _arange.reshape(-1, 1) > _arange.reshape(1, -1)
        mask = jnp.logical_or(mask, mask_lower_left)

        cost_reductions = jnp.where(mask, -jnp.inf, r)
        cost_reductions_flat = cost_reductions.reshape(-1)
        action = cost_reductions.argmax()
        cost_reduction = cost_reductions_flat[action]
        action = jnp.array(jnp.divmod(action, num_nodes))

        action = jax.lax.select(
            cost_reduction > 0,
            action,
            jnp.zeros_like(action),     # no-op
        )

        new_sol = apply_two_opt(sol, step=action)
        return new_sol

    new_sol = jax.lax.fori_loop(
        0, num_steps,
        body_fun=_update_once,
        init_val=sol,
        unroll=unroll,
    )

    return new_sol

        
def two_opt(
    sol: jax.Array, dist_mat: jax.Array, 
    num_steps: int = 1000, *, unroll: int = 1, 
) -> jax.Array:
    batched = sol.ndim == 2
 
    if not batched:
        return jax.jit(_two_opt_post_processing_argmax_impl_unbatched, static_argnums=(2, 3))(sol, dist_mat, num_steps, unroll)
    else:
        return jax.vmap(partial(_two_opt_post_processing_argmax_impl_unbatched, num_steps=num_steps, unroll=unroll), in_axes=(0, 0))(sol, dist_mat)
