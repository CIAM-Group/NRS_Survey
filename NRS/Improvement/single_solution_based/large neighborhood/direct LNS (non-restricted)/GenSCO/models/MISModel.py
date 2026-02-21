import jax
import jax.numpy as jnp
from flax import nnx
from dataclasses import dataclass
from modules import TransformerEncoder, TimestepEmbedder, MLPBlock, Linear
from typing import Any, Callable, Literal
from functools import partial


@dataclass
class MISModelConfig:
    num_layers: tuple[int, int] = (16, 6) 
    embed_dim: int | tuple[int, int] = 128
    hidden_dim: int | None = None
    num_heads: int | tuple[int, int] = 4
    activation: str = 'silu'
    use_swiglu: bool = True

    dtype: str = 'bfloat16'

    softcap: float | None = 30.
    qk_norm: bool = False

    final_softcap: float | None = 50.

    sigmoid_head: bool = True
    softmax_head: bool = False

    init_method: Literal['normal', 'sincos'] = 'normal'

    rngs: int = 888


    def __post_init__(self):
        if isinstance(self.embed_dim, int):
            self.embed_dim = (self.embed_dim, self.embed_dim)
        if isinstance(self.num_heads, int):
            self.num_heads = (self.num_heads, self.num_heads)
        assert self.sigmoid_head or self.softmax_head

    @staticmethod
    def get_config(which: str = ''):
        match which:
            case '' | 'default':
                return MISModelConfig()
            case 'dim256_qknorm_sigmoid_decoder_only_layer12':
                return MISModelConfig(
                    embed_dim=256, sigmoid_head=True, softmax_head=False, softcap=None,
                    num_layers=(0, 12), num_heads=4, qk_norm=True,
                )
            case _:
                raise ValueError()

    def construct_model(self):
        return MISModel(**vars(self))
        
    

class MISModel(nnx.Module):
    def __init__(
        self,
        num_layers: tuple[int, int] = (16, 6) ,
        embed_dim: tuple[int, int] = (128, 128),
        hidden_dim: int | None = None,
        num_heads: tuple[int, int] = 4,
        activation: str = 'silu',
        use_swiglu: bool = True,
        *,
        dtype: jax.typing.DTypeLike = jnp.bfloat16,
        softcap: float | None = 30.,
        qk_norm: bool = False,
        final_softcap: float | None = 50.,
        sigmoid_head: bool = True,
        softmax_head: bool = False,
        init_method: Literal['normal', 'sincos'] = 'normal',
        rngs: nnx.Rngs | int,
    ):
        super().__init__()

        if isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)
        
        encoder_num_layers, decoder_num_layers = num_layers
        encoder_embed_dim, decoder_embed_dim = embed_dim
        self.encoder_embed_dim = encoder_embed_dim
        if init_method == 'normal':
            self.init_proj = Linear(
                encoder_embed_dim, 
                encoder_embed_dim, 
                use_bias=True,
                rngs=rngs,
            )
        elif init_method == 'sincos':
            self.init_proj = TimestepEmbedder(
                encoder_embed_dim, frequency_embed_dim=256, 
                timestep_scale=1.,
                dtype=dtype, rngs=rngs,
            )
        else:
            raise ValueError()
        self.init_method = init_method
        self.encoder = TransformerEncoder(
            encoder_num_layers, encoder_embed_dim, num_heads[0], 
            hidden_dim, activation, use_swiglu,
            remat_layernorm=True, qkv_packed=True,
            dtype=dtype, rngs=rngs, qk_norm=qk_norm,
        )
        self.mid_proj = Linear(
            encoder_embed_dim, 
            decoder_embed_dim, 
            use_bias=True,
            rngs=rngs,
        )
        self.chosen_bias = nnx.Param(
            nnx.initializers.normal(1e-6)(rngs.params(), [decoder_embed_dim], jnp.float32)
        )
        self.timestep_embedder = TimestepEmbedder(
            decoder_embed_dim, frequency_embed_dim=256, 
            timestep_scale=1000.,
            dtype=dtype, rngs=rngs,
        )
        self.decoder = TransformerEncoder(
            decoder_num_layers, decoder_embed_dim, num_heads[1], 
            hidden_dim, activation, use_swiglu,
            remat_layernorm=True, qkv_packed=True,
            dtype=dtype, rngs=rngs, qk_norm=qk_norm,
        )

        final_proj_cls = partial(
            MLPBlock,
            decoder_embed_dim, decoder_embed_dim, 1,
            use_bias=True, activation='silu', dtype=dtype, rngs=rngs,
        )
        self.sigmoid_final_proj = lambda _: None
        self.softmax_final_proj = lambda _: None
        if sigmoid_head:
            self.sigmoid_final_proj = final_proj_cls()
        if softmax_head:
            self.softmax_final_proj = final_proj_cls()
        # self.final_proj = self.sigmoid_final_proj   # for backward campatibility

        self.softcap = softcap
        self.final_softcap = final_softcap
        self.dtype = dtype

    @staticmethod
    def get_segment_ids(num_nodes: jax.Array, num_nodes_padded: int):
        batch_size = num_nodes.shape[0]
        segment_ids = jnp.arange(num_nodes_padded, dtype=jnp.int32)
        segment_ids = jnp.broadcast_to(segment_ids[None], [batch_size, num_nodes_padded])
        segment_ids = (segment_ids >= num_nodes[..., None]).astype(jnp.int8)
        return segment_ids

    def encode(
        self, 
        adjmat: jax.Array,
        key: jax.Array,
        num_nodes: jax.Array | None = None,
        attn_options: dict[str, Any] = {},
    ):
        adjmat = adjmat.astype(jnp.float16)

        batch_size, _, num_nodes_padded = adjmat.shape

        # segment_ids = self.get_segment_ids(num_nodes, num_nodes_padded)
        # attn_options.update(softcap=self.softcap, segment_ids=segment_ids, bias=adjmat)

        attn_options.update(softcap=self.softcap, bias=adjmat)

        if self.init_method == 'normal':
            init_features = jax.random.normal(key, shape=[batch_size, num_nodes_padded, self.encoder_embed_dim], dtype=self.dtype)
            features = self.init_proj(init_features)
        elif self.init_method == 'sincos':
            indices = jax.random.choice(key, 10000, shape=[num_nodes_padded], replace=False)
            features = self.init_proj(indices)
            features = jnp.broadcast_to(features[None], (batch_size,) + features.shape)
        features = self.encoder(features, attn_options=attn_options)
        return features
    
    def decode(
        self, 
        features: jax.Array, 
        current_state: jax.Array,
        timestep: jax.Array, 
        adjmat: jax.Array,
        num_nodes: jax.Array,
        attn_options: dict[str, Any] = {},
        force_fp32_logit: bool = True,
    ):
        adjmat = adjmat.astype(jnp.float16)
        
        num_nodes_padded = adjmat.shape[-1]
        segment_ids = self.get_segment_ids(num_nodes, num_nodes_padded)
        # attn_options.update(softcap=self.softcap, segment_ids=segment_ids, bias=adjmat)
        attn_options.update(softcap=self.softcap, bias=adjmat)

        features = self.mid_proj(features)
        features = features + self.timestep_embedder(timestep).astype(features.dtype)[:, None]
        chosen_bias = self.chosen_bias.value[None, None, :] * current_state[:, :, None]
        features = features + chosen_bias.astype(features.dtype)

        features = self.decoder(features, attn_options=attn_options)

        def decode_single_head(final_proj: nnx.Module | Callable[..., None], features: jax.Array):
            logits = final_proj(features)
            if logits is None:
                return None
            if force_fp32_logit:
                logits = logits.astype(jnp.float32)
            if self.final_softcap is not None:
                logits = self.final_softcap * jnp.tanh(logits / self.final_softcap)
            logits = logits.squeeze(-1)
            logits = jnp.where(
                segment_ids,
                0.5 * jnp.finfo(logits.dtype).min,
                logits,
            )
            return logits

        return tuple(decode_single_head(head, features) for head in [self.sigmoid_final_proj, self.softmax_final_proj])
