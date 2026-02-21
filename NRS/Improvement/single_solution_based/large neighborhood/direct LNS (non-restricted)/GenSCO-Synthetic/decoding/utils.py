import jax
import jax.numpy as jnp
import numpy as np
from typing import Literal
import typing as tp


def heatmap_quant(
    heatmap: jax.Array,
    *,
    strategy: Literal['constant', 'adaptive'],
    eps: float = 1e-6,
):
    heatmap = heatmap.astype(jnp.float32)
    match strategy:
        case 'constant':
            return (heatmap * 254).astype(jnp.uint8)
        case 'adaptive':
            return_shape = heatmap.shape
            heatmap = heatmap.reshape(heatmap.shape[0], -1)
            minvals = heatmap.min(axis=-1, keepdims=True)
            maxvals = heatmap.max(axis=-1, keepdims=True)
            scales = 254 / (maxvals - minvals + eps)
            heatmap = (heatmap - minvals) * scales
            heatmap = heatmap.astype(jnp.uint8).reshape(return_shape)
            return heatmap
        case _:
            raise ValueError()


def convert_heatmap_dtype(
    heatmap: jax.Array,
    dtype: jax.typing.DTypeLike,
    *,
    quant_strategy: Literal['constant', 'adaptive'] = 'constant',
    quant_eps: float = 1e-6,
):
    ''' Convert the heatmap to a low precision to accelerate `jnp.argsort` or `jax.lax.top_k`.
    '''

    dtype = jnp.dtype(dtype)
    assert jnp.issubdtype(dtype, jnp.floating) or dtype == jnp.uint8

    if jnp.issubdtype(dtype, jnp.floating):
        return heatmap.astype(dtype)
    else:
        return heatmap_quant(heatmap, strategy=quant_strategy, eps=quant_eps)
    


class CoordDynamicArgument:
    def __init__(
        self, 
        encode_fn: tp.Callable[[jax.Array], jax.Array],
        coords: np.ndarray,
        argument_level: int = 0,
    ):
        def reflect_x(coords: np.ndarray):
            return np.stack([-coords[..., 0], coords[..., 1]], axis=-1)
        def reflect_y(coords: np.ndarray):
            return np.stack([coords[..., 0], -coords[..., 1]], axis=-1)

        self.feature_list: list[jax.Array]
        match argument_level:
            case 0:
                self.feature_list = [
                    encode_fn(coords)
                ]
            case 1:
                self.feature_list = [
                    encode_fn(coords),
                    encode_fn(-coords),
                    encode_fn(reflect_x(coords)),
                    encode_fn(reflect_y(coords)),
                ]
                coords = np.flip(coords, axis=-1)
                self.feature_list += [
                    encode_fn(coords),
                    encode_fn(-coords),
                    encode_fn(reflect_x(coords)),
                    encode_fn(reflect_y(coords)),
                ]
            case _:
                raise ValueError()
            
        self.argument_level = argument_level
        self.argument_num = len(self.feature_list)
        
    def __call__(self, generator: np.random.Generator | None = None):
        if self.argument_level == 0:
            return self.feature_list[0]
        else:
            return self.feature_list[generator.integers(0, self.argument_num)]

