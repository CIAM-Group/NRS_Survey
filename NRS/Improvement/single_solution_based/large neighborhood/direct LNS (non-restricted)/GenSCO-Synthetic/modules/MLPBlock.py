import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial

from .get_activation import get_activation


class MLPBlock(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        out_embed_dim: int | None = None,
        use_bias: bool = True,
        activation: str = 'silu',
        *,
        kernel_init = nnx.initializers.xavier_uniform(),
        bias_init = nnx.initializers.normal(1e-6),
        dtype: jax.typing.DTypeLike,
        param_dtype: jax.typing.DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim * 4
        if out_embed_dim is None:
            out_embed_dim = embed_dim
        activation = get_activation(activation)
        
        Linear = partial(
            nnx.Linear,
            use_bias=use_bias, param_dtype=param_dtype,
            dtype=dtype, rngs=rngs,
        )
        self.inner = nnx.Sequential(
            Linear(
                embed_dim, hidden_dim, 
                kernel_init=kernel_init,
                bias_init=bias_init,
            ),
            activation,
            Linear(
                hidden_dim, out_embed_dim, 
                kernel_init=kernel_init,
                bias_init=bias_init,
            ),
        )

    def __call__(self, x: jax.Array):
        return self.inner(x)

