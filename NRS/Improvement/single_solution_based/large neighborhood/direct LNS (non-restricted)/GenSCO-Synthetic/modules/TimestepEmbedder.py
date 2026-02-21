'''
Adapted from https://github.com/Stability-AI/sd3.5/blob/main/mmditx.py.
'''

import math
import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial


class TimestepEmbedder(nnx.Module):
    def __init__(
        self,
        timestep_embed_dim: int = 128, 
        frequency_embed_dim: int = 256,
        timestep_scale: float = 1000.,
        *,
        dtype: jax.typing.DTypeLike,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        in_kernel_init = nnx.initializers.xavier_uniform()
        in_bias_init = nnx.initializers.normal(1e-6)
        out_kernel_init = nnx.initializers.xavier_uniform()
        out_bias_init = nnx.initializers.normal(1e-6)
        Linear = partial(
            nnx.Linear,
            dtype=dtype, rngs=rngs,
            use_bias=True,
        )
        self.mlp = nnx.Sequential(
            Linear(frequency_embed_dim, timestep_embed_dim, kernel_init=in_kernel_init, bias_init=in_bias_init),
            nnx.silu,
            Linear(timestep_embed_dim, timestep_embed_dim, kernel_init=out_kernel_init, bias_init=out_bias_init),
        )
        self.timestep_scale = timestep_scale
        self.frequency_embed_dim = frequency_embed_dim
    
    @staticmethod
    def timestep_embedding(t: jax.Array, dim: int, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        assert dim % 2 == 0
        t = t.astype(jnp.float32)
        half = dim // 2
        with jax.ensure_compile_time_eval():
            freqs = jnp.exp(
                -math.log(max_period)
                * jnp.arange(start=0, stop=half, dtype=jnp.float32)
                / half
            )
        args = t[:, None] * freqs[None]
        embedding = jnp.concat([jnp.cos(args), jnp.sin(args)], axis=-1)
        return embedding
    
    def __call__(self, timestep: jax.Array):
        timestep = timestep.astype(jnp.float32)
        if self.timestep_scale != 1.:
            timestep = timestep * self.timestep_scale
        t_freq = self.timestep_embedding(timestep, self.frequency_embed_dim)
        t_emb = self.mlp(t_freq)
        return t_emb


