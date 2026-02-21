import jax
import jax.numpy as jnp
from typing import Literal


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