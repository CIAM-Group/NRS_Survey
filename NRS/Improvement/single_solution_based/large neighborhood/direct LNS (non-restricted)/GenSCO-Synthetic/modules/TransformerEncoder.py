import jax
import jax.numpy as jnp
from flax import nnx
import typing as tp

from .TransformerLayer import TransformerLayer


class TransformerEncoder(nnx.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int | None = None,
        activation: str = 'silu',
        use_swiglu: bool = False,
        *,
        remat_layernorm: bool = True,
        qkv_packed: bool = True,
        qk_norm: bool = False,
        default_attn_options: dict[str, tp.Any] = {},
        dtype: jax.typing.DTypeLike,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.layers = [
            TransformerLayer(
                embed_dim, num_heads,
                hidden_dim, activation, use_swiglu,
                remat_layernorm=remat_layernorm,
                qkv_packed=qkv_packed, dtype=dtype,
                qk_norm=qk_norm,
                rngs=rngs,
            ) for _ in range(num_layers)
        ]
        self.default_attn_options = default_attn_options
    
    def __call__(self, features: jax.Array, attn_options: dict[str, tp.Any] | None = None):
        if attn_options is None:
            attn_options = self.default_attn_options
        for layer in self.layers:
            features = layer(features, attn_options)
        return features

