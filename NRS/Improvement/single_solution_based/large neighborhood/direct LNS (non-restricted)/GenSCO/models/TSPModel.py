import jax
import jax.numpy as jnp
from flax import nnx
from dataclasses import dataclass
from modules import TransformerEncoder, TimestepEmbedder, MLPBlock, Linear
from typing import Any, Literal
from helpers import with_invalid_kwargs_filtered


@dataclass
class TSPModelConfig:
    num_layers: tuple[int, int] = (16, 6) 
    embed_dim: int | tuple[int, int] = 256
    hidden_dim: int | None = None
    num_heads: int | tuple[int, int] = 8
    activation: str = 'silu'
    use_swiglu: bool = True

    dtype: str = 'bfloat16'

    encoder_input_dim: int = 2
    softcap: float | None = 30.
    qk_norm: bool = False
    sm_scale: float | None = None

    final_norm: bool = True
    final_norm_use_scale: bool = False
    final_softcap: float | None = 50.
    final_logit_scale: float = 1.

    rngs: int = 888


    def __post_init__(self):
        if isinstance(self.embed_dim, int):
            self.embed_dim = (self.embed_dim, self.embed_dim)
        if isinstance(self.num_heads, int):
            self.num_heads = (self.num_heads, self.num_heads)

    @staticmethod
    def get_config(which: str = ''):
        match which:
            case '' | 'default':
                return TSPModelConfig()
            case 'norm_use_scale':
                return TSPModelConfig(
                    final_norm=True, final_norm_use_scale=True,
                    final_softcap=None, final_logit_scale=16 / 256,
                )
            case 'qk_norm_small' | 'qk_norm_final_norm_use_scale':
                return TSPModelConfig(
                    qk_norm=True, softcap=None,
                    final_norm=True, final_norm_use_scale=True,
                    final_softcap=None, final_logit_scale=16 / 256,
                )
            case 'qk_norm_base':
                return TSPModelConfig(
                    qk_norm=True, softcap=None,
                    final_norm=True, final_norm_use_scale=True,
                    final_softcap=None, final_logit_scale=16 / 256,
                    num_layers=(32, 12),
                )
            case 'qk_norm_base_dim512':
                return TSPModelConfig(
                    embed_dim=512, num_heads=8,
                    qk_norm=True, softcap=None,
                    final_norm=True, final_norm_use_scale=True,
                    final_softcap=None, final_logit_scale=16 / 512,
                    num_layers=(16, 6),
                )
            case _:
                raise ValueError()

    def construct_model(self):
        return TSPModel(**vars(self))
        
    

class TSPModel(nnx.Module):
    def __init__(
        self,
        num_layers: tuple[int, int] = (16, 6) ,
        embed_dim: tuple[int, int] = 256,
        hidden_dim: int | None = None,
        num_heads: tuple[int, int] = 8,
        activation: str = 'silu',
        use_swiglu: bool = True,
        *,
        dtype: jax.typing.DTypeLike = jnp.bfloat16,
        encoder_input_dim: int = 2,
        softcap: float | None = 30.,
        qk_norm: bool = False,
        sm_scale: float | None = None,
        final_norm: bool = True,
        final_norm_use_scale: bool = False,
        final_softcap: float | None = 50.,
        final_logit_scale: float = 1.,
        rngs: nnx.Rngs | int,
    ):
        super().__init__()

        if isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)
        
        encoder_num_layers, decoder_num_layers = num_layers
        encoder_embed_dim, decoder_embed_dim = embed_dim
        self.init_proj = Linear(encoder_input_dim, encoder_embed_dim, use_bias=True, rngs=rngs)
        self.encoder = TransformerEncoder(
            encoder_num_layers, encoder_embed_dim, num_heads[0], 
            hidden_dim, activation, use_swiglu,
            remat_layernorm=False, qkv_packed=True, qk_norm=qk_norm,
            dtype=dtype, rngs=rngs
        )
        self.mid_proj = Linear(
            encoder_embed_dim, 
            decoder_embed_dim, 
            use_bias=True,
            rngs=rngs,
        )
        self.timestep_embedder = TimestepEmbedder(
            decoder_embed_dim, frequency_embed_dim=256, 
            timestep_scale=1000.,
            dtype=dtype, rngs=rngs,
        )
        self.decoder = TransformerEncoder(
            decoder_num_layers, decoder_embed_dim, num_heads[1], 
            hidden_dim, activation, use_swiglu,
            remat_layernorm=False, qkv_packed=True, qk_norm=qk_norm,
            dtype=dtype, rngs=rngs
        )
        self.final_proj = MLPBlock(
            decoder_embed_dim, decoder_embed_dim, decoder_embed_dim,
            use_bias=True, activation='silu', dtype=dtype, rngs=rngs,
        )

        self.softcap = softcap
        self.sm_scale = sm_scale
        self.final_softcap = final_softcap
        self.final_logit_scale = final_logit_scale
        self.dtype = dtype

        self.final_norm = nnx.RMSNorm(
            decoder_embed_dim, dtype=None, 
            use_scale=final_norm_use_scale, epsilon=1e-6, rngs=rngs,
        )
        self.apply_final_norm = final_norm

    def encode(self, raw_features: jax.Array, attn_options: dict[str, Any] = {}):
        attn_options.update(softcap=self.softcap, sm_scale=self.sm_scale)
        features = self.init_proj(raw_features)
        features = self.encoder(features, attn_options=attn_options)
        return features
    
    def decode(
        self, 
        features: jax.Array, 
        timestep: jax.Array, 
        adjmat: jax.Array | None,
        attn_options: dict[str, Any] = {},
        target: Literal['logit', 'embed'] = 'logit',
        force_fp32_logit: bool = True,
    ):
        if adjmat is not None:
            if adjmat.shape[-1] > 128 or not jnp.issubdtype(adjmat.dtype, jnp.floating):
                adjmat = adjmat.astype(jnp.float16)
        attn_options.update(softcap=self.softcap, bias=adjmat, sm_scale=self.sm_scale)
        features = self.mid_proj(features)
        features = features + self.timestep_embedder(timestep).astype(features.dtype)[:, None]
        features = self.decoder(features, attn_options=attn_options)
        features = self.final_proj(features)
        if target == 'logit':
            return self.feature2logit(features, force_fp32_logit=force_fp32_logit)
        elif target == 'embed':
            return features
        else:
            raise ValueError()
    
    def feature2logit(
        self, 
        features: jax.Array, 
        *, 
        mask_value: float | None | Literal['default'] = 'default',
        force_fp32_logit: bool = True,
    ):
        assert features.ndim == 3
        if self.apply_final_norm:
            features = self.final_norm(features)
        logits = jnp.matmul(features, features.transpose(0, 2, 1))
        if force_fp32_logit:
            logits = logits.astype(jnp.float32)
        if self.final_logit_scale != 1.:
            logits = logits * self.final_logit_scale
        if self.final_softcap is not None:
            logits = self.final_softcap * jnp.tanh(logits / self.final_softcap)
        if mask_value is not None:
            if mask_value == 'default':
                mask_value = 0.5 * jnp.finfo(logits.dtype).min
            _arange = jnp.arange(logits.shape[1], dtype=jnp.int32)
            logits = logits.at[:, _arange, _arange].set(mask_value)
        return logits
