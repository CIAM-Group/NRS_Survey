'''
Mainly adapted from https://github.com/Stability-AI/sd3.5/blob/main/sd3_impls.py.
'''

import jax
import jax.numpy as jnp
import typing as tp


def get_linear_timesteps(scheduled_steps: int):
    return jnp.linspace(0., 1., scheduled_steps + 1, dtype=jnp.float32)


def sample_euler(
    denoised_fn: tp.Callable[[jax.Array, jax.Array], jax.Array], 
    x: jax.Array, timesteps: jax.Array,
    *, ascending: bool = True, force_fp32: bool = True,
    need_history: bool = False,
):
    if not ascending:
        timesteps = 1 - timesteps

    denoised_history: list[jax.Array] = []

    batch_size = x.shape[0]
    for i in range(len(timesteps) - 1):
        cur_timestep = timesteps[i]
        denoised = denoised_fn(x, jax.lax.broadcast_in_dim(cur_timestep, shape=[batch_size], broadcast_dimensions=[]))
        denoised_history.append(denoised)
        if force_fp32:
            denoised = denoised.astype(jnp.float32)
        v = denoised - x
        dt = (timesteps[i + 1] - cur_timestep) / (1 - cur_timestep)
        # Euler method
        x = x + v * dt
    
    if not need_history:
        return x
    else:
        return x, denoised_history


def sample_dpmpp_2m(
    denoised_fn: tp.Callable[[jax.Array, jax.Array], jax.Array], 
    x: jax.Array, timesteps: jax.Array,
    *, ascending: bool = True, force_fp32: bool = True, 
):
    """DPM-Solver++(2M)."""

    if ascending:
        sigmas = 1 - timesteps
    else:
        sigmas = timesteps
    
    batch_size = x.shape[0]

    if force_fp32:
        x = x.astype(jnp.float32)
    
    sigma_fn = lambda t: jnp.exp(-t)
    t_fn = lambda sigma: -jnp.log(sigma)

    old_denoised = None
    num_steps = len(sigmas) - 1 
    for i in range(num_steps):
        denoised = denoised_fn(x, jax.lax.broadcast_in_dim(1 - sigmas[i], shape=[batch_size], broadcast_dimensions=[]))
        if force_fp32:
            denoised = denoised.astype(jnp.float32)
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if i == 0 or i == num_steps - 1: 
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - jnp.expm1(-h) * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - jnp.expm1(-h) * denoised_d
        old_denoised = denoised
    return x

