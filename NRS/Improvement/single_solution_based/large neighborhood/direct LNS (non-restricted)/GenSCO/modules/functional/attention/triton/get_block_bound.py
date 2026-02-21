import jax
import jax.numpy as jnp
import triton as tl
from functools import partial


def get_block_bound(segment_ids: jax.Array, block: int, need_q_bound: bool = False):
    '''
    :param segment_ids: with shape [B, S]
    :return kv_bound: with shape [B, num_blocks, 2]
    :return q_bound: None | jax.Array with shape [B, num_blocks, 2]
    '''

    B, S = segment_ids.shape
    S_padded = tl.cdiv(S, block) * block
    segment_ids = jnp.pad(segment_ids, pad_width=[(0, 0), (0, S_padded - S)], constant_values=jnp.iinfo(segment_ids.dtype).max)
    B, S = segment_ids.shape
    segment_ids = segment_ids.reshape(B, -1, block)
    num_blocks = segment_ids.shape[1]
    block_min_segment_ids = segment_ids[..., 0]
    block_max_segment_ids = segment_ids[..., -1]
    block_ids = jnp.arange(num_blocks, dtype=jnp.int16)
    
    @partial(jax.vmap, in_axes=(0, 0, None))
    def _get_block_mask(block_min_segment_ids, block_max_segment_ids, block_ids):
        block_mask = jnp.logical_or(
            jnp.logical_and(
                block_ids.reshape(1, -1) <= block_ids.reshape(-1, 1),
                block_max_segment_ids.reshape(1, -1) >= block_min_segment_ids.reshape(-1, 1)
            ),
            jnp.logical_and(
                block_ids.reshape(1, -1) >= block_ids.reshape(-1, 1),
                block_min_segment_ids.reshape(1, -1) <= block_max_segment_ids.reshape(-1, 1)
            ),
        )   # [num_blocks, num_blocks]
        kv_lb = jax.vmap(partial(jnp.nonzero, size=1))(block_mask)[0]
        kv_ub = num_blocks - jax.vmap(partial(jnp.nonzero, size=1))(jnp.flip(block_mask, axis=-1))[0]
        kv_bound = jnp.concat([kv_lb, kv_ub], axis=-1)
        if not need_q_bound:
            return kv_bound, None
        else:
            block_mask = block_mask.transpose(0, 1)
            q_lb = jax.vmap(partial(jnp.nonzero, size=1))(block_mask)[0]
            q_ub = num_blocks - jax.vmap(partial(jnp.nonzero, size=1))(jnp.flip(block_mask, axis=-1))[0]
            q_bound = jnp.concat([q_lb, q_ub], axis=-1)
            return kv_bound, q_bound
    return _get_block_mask(block_min_segment_ids, block_max_segment_ids, block_ids)
