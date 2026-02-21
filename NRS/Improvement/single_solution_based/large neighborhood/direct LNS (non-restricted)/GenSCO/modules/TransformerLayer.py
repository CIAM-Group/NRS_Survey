import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial
import typing as tp

from .MultiHeadAttention import MultiHeadAttention
from .SwiGLU import SwiGLU
from .MLPBlock import MLPBlock


class TransformerLayer(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int | None = None,
        activation: str = 'silu',
        use_swiglu: bool = True,
        *,
        remat_layernorm: bool = True,
        qkv_packed: bool = True,
        qk_norm: bool = False,
        dtype: jax.typing.DTypeLike,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.dtype = dtype

        if hidden_dim is None:
            if not use_swiglu:
                hidden_dim = embed_dim * 4
            else:
                hidden_dim = embed_dim * 3
        self.remat_layernorm = remat_layernorm

        LayerNorm = partial(
            nnx.LayerNorm,
            epsilon=1e-6, 
            dtype=dtype, 
            rngs=rngs,
            use_bias=False,
            scale_init=nnx.initializers.ones_init(),
            bias_init=nnx.initializers.zeros_init(),
        )

        self.ln1 = LayerNorm(embed_dim)
        if not use_swiglu:
            self.ffn = MLPBlock(
                embed_dim, hidden_dim, 
                activation=activation, 
                use_bias=True, 
                dtype=dtype,
                rngs=rngs,
            )
        else:
            self.ffn = SwiGLU(
                embed_dim, hidden_dim, 
                use_bias=True, 
                dtype=dtype, 
                rngs=rngs,
            )
        self.ln2 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim, num_heads, 
            dtype=dtype, qkv_packed=qkv_packed, 
            normalize_qk=qk_norm,
            rngs=rngs,
        )

    def __call__(self, features: jax.Array, attn_options: dict[str, tp.Any] = {}):
        features = features.astype(self.dtype)
        residual = features
        if self.remat_layernorm:
            features = nnx.remat(lambda x: self.ln1(x))(features)
        else:
            features = self.ln1(features)
        features = self.attn(features, attn_options)
        features = features + residual
        residual = features
        if self.remat_layernorm:
            features = nnx.remat(lambda x: self.ln2(x))(features)
        else:
            features = self.ln2(features)
        features = self.ffn(features)
        features = features + residual
        return features
