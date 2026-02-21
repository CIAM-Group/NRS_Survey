import jax
import triton
import math
from functools import partial
from common.jax_triton import triton_call
from common.array_view import ArrayView
from .fa_kernel_softcap_then_bias import _fwd_kernel, _bwd_kernel, _bwd_preprocess_do_o_dot


def strides_from_shape(shape: tuple[int, ...]):
    size = math.prod(shape)
    for s in shape:
        size = size // s
        yield int(size)


def process_bias(bias: jax.Array | None):
    if bias is None:
        return 'none', [0, 0, 0], bias
    assert bias.ndim in [3, 4]
    if bias.ndim == 3:
        bias = bias[:, None]
    # bias: [B, H/1, S/1, S]
    if bias.shape[-2] == 1:
        bias_type = 'vector'
    else:
        bias_type = 'matrix'
    bias_strides = list(strides_from_shape(bias.shape))[:3]
    if bias.shape[1] == 1:
        bias_strides[1] = 0     # shared across heads
    if bias.shape[2] == 1:
        bias_strides[2] == 0     # shared across queries
    return bias_type, bias_strides, bias


@partial(jax.jit, static_argnames=['causal', 'sm_scale', 'softcap', 'use_alibi'])
def _forward_impl(
    q: jax.Array | ArrayView, k: jax.Array | ArrayView, v: jax.Array | ArrayView, 
    bias: jax.Array | None = None, segment_ids: jax.Array | None = None,
    causal: bool = False, sm_scale: float | None = None, softcap: float | None = None,
    use_alibi: bool = False,
) -> tuple[jax.Array, jax.Array]:
    if softcap is None:
        softcap = 0.
    if segment_ids is None:
        segment_ids = jax.numpy.array([], dtype=jax.numpy.int8)

    B, S, H, D = q.shape

    seqlen_rounded = triton.cdiv(S, 128) * 128

    if sm_scale is None:
        sm_scale = D ** (-0.5)

    bias_type, bias_strides, bias = process_bias(bias)
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)

    q_stride = k_stride = v_stride = o_stride = list(strides_from_shape(q.shape))
    if isinstance(q, ArrayView):
        q_stride = q.strides
    if isinstance(k, ArrayView):
        k_stride = k.strides
    if isinstance(v, ArrayView):
        v_stride = v.strides

    k_offset = v_offset = 0
    if isinstance(k, ArrayView):
        k_offset = k.offset
    if isinstance(v, ArrayView):
        v_offset = v.offset

    metaparams = dict(
        TMP=None,
        softmax_scale=sm_scale,
        stride_qb=q_stride[0],
        stride_qh=q_stride[2],
        stride_qm=q_stride[1],
        stride_kb=k_stride[0],
        stride_kh=k_stride[2],
        stride_kn=k_stride[1],
        stride_vb=v_stride[0],
        stride_vh=v_stride[2],
        stride_vn=v_stride[1],
        stride_bb=bias_strides[0],
        stride_bh=bias_strides[1],
        stride_bm=bias_strides[2],
        stride_ob=o_stride[0],
        stride_oh=o_stride[2],
        stride_om=o_stride[1],
        nheads=H,
        seqlen_q=S,
        seqlen_k=S,
        seqlen_q_rounded=seqlen_rounded,
        headdim=D,
        CACHE_KEY_SEQLEN_Q=S // 32,
        CACHE_KEY_SEQLEN_K=S // 32,
        BIAS_TYPE=bias_type,
        IS_CAUSAL=causal,
        BLOCK_HEADDIM=D,
        softcap=softcap,
        k_ptr_offset=k_offset,
        v_ptr_offset=v_offset,
        alibi_scale=None if not use_alibi else 2 ** (-8 / H),
    )

    lse = jax.ShapeDtypeStruct(
        shape=[B, H, seqlen_rounded],
        dtype='float32',
    )
    o = jax.ShapeDtypeStruct(q.shape, q.dtype)

    kv_bound = jax.numpy.array([], dtype=jax.numpy.int8)
    if segment_ids.size > 0:
        from .get_block_bound import get_block_bound
        kv_bound, _ = get_block_bound(segment_ids, 64)

    o, lse = triton_call(
        q, k, v, bias if bias is not None else jax.numpy.array([], dtype=jax.numpy.int8),
        segment_ids, kv_bound,
        grid=grid, kernel=_fwd_kernel,
        out_shape=[o, lse],
        has_segment_ids=segment_ids.size > 0,
        has_kv_bound=kv_bound.size > 0,
        **metaparams,
    )

    return o, lse


@partial(jax.jit, static_argnames=['causal', 'sm_scale', 'softcap', 'use_alibi'])
def _backward_impl(
    q: jax.Array | ArrayView, k: jax.Array | ArrayView, v: jax.Array | ArrayView,
    do: jax.Array, o: jax.Array, lse: jax.Array, 
    bias: jax.Array | None = None, segment_ids: jax.Array | None = None, causal: bool = False, 
    sm_scale: float | None = None, softcap: float | None = None,
    use_alibi: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if softcap is None:
        softcap = 0.
    if segment_ids is None:
        segment_ids = jax.numpy.array([], dtype=jax.numpy.int8)

    B, S, H, D = q.shape

    seqlen_rounded = triton.cdiv(S, 128) * 128

    if sm_scale is None:
        sm_scale = D ** (-0.5)

    bias_type, bias_strides, bias = process_bias(bias)

    dq_stride = dk_stride = dv_stride = q_stride = k_stride = v_stride = o_stride = list(strides_from_shape(q.shape))
    if isinstance(q, ArrayView):
        q_stride = q.strides
    if isinstance(k, ArrayView):
        k_stride = k.strides
    if isinstance(v, ArrayView):
        v_stride = v.strides
    do_stride = o_stride

    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    delta = jax.ShapeDtypeStruct(lse.shape, lse.dtype)
    delta = triton_call(
        o, do,
        kernel=_bwd_preprocess_do_o_dot,
        out_shape=[delta], grid=grid,

        stride_ob=o_stride[0],
        stride_oh=o_stride[2],
        stride_om=o_stride[1],
        stride_dob=do_stride[0],
        stride_doh=do_stride[2],
        stride_dom=do_stride[1],
        nheads=H,
        seqlen_q=S,
        seqlen_q_rounded=seqlen_rounded,
        headdim=D,
        BLOCK_M=128,
        BLOCK_HEADDIM=D,
    )


    # after preprocess
    metaparams = dict(
        softmax_scale=sm_scale,

        stride_qb=q_stride[0],
        stride_qh=q_stride[2],
        stride_qm=q_stride[1],
        stride_kb=k_stride[0],
        stride_kh=k_stride[2],
        stride_kn=k_stride[1],
        stride_vb=v_stride[0],
        stride_vh=v_stride[2],
        stride_vn=v_stride[1],
        stride_bb=bias_strides[0],
        stride_bh=bias_strides[1],
        stride_bm=bias_strides[2],

        stride_dob=do_stride[0],
        stride_doh=do_stride[2],
        stride_dom=do_stride[1],

        stride_dqb=dq_stride[0],
        stride_dqh=dq_stride[2],
        stride_dqm=dq_stride[1],
        stride_dkb=dk_stride[0],
        stride_dkh=dk_stride[2],
        stride_dkn=dk_stride[1],
        stride_dvb=dv_stride[0],
        stride_dvh=dv_stride[2],
        stride_dvn=dv_stride[1],

        nheads=H,
        seqlen_q=S,
        seqlen_k=S,
        seqlen_q_rounded=seqlen_rounded,
        headdim=D,
        CACHE_KEY_SEQLEN_Q=S // 32,
        CACHE_KEY_SEQLEN_K=S // 32,
        BIAS_TYPE=bias_type,
        IS_CAUSAL=causal,
        BLOCK_HEADDIM=D,

        softcap=softcap,

        alibi_scale=None if not use_alibi else 2 ** (-8 / H),
    )

    dq_accum = jax.ShapeDtypeStruct(shape=q.shape, dtype='float32')
    dk = jax.ShapeDtypeStruct(shape=k.shape, dtype=k.dtype)
    dv = jax.ShapeDtypeStruct(shape=v.shape, dtype=q.dtype)

    grid = lambda META: (
        triton.cdiv(S, META["BLOCK_N"]) if META["SEQUENCE_PARALLEL"] else 1,
        B * H,
    )

    k_offset = v_offset = 0
    if isinstance(k, ArrayView):
        k_offset = k.offset
    if isinstance(v, ArrayView):
        v_offset = v.offset

    q_bound = jax.numpy.array([], dtype=jax.numpy.int8)
    from .get_block_bound import get_block_bound
    if segment_ids.size > 0:
        _, q_bound = get_block_bound(segment_ids, 64, need_q_bound=True)

    dq_accum, dk, dv = triton_call(
        q, k, v, bias if bias is not None else jax.numpy.zeros([], dtype='int8'),
        do, segment_ids, q_bound, lse, delta,
        kernel=_bwd_kernel, grid=grid,
        out_shape=[dq_accum, dk, dv],
        zeroed_outputs=[0],
        k_ptr_offset=k_offset, v_ptr_offset=v_offset,
        has_segment_ids=segment_ids.size > 0, has_q_bound=q_bound.size > 0,
        **metaparams,
    )
    return dq_accum.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)


@partial(jax.custom_vjp, nondiff_argnums=[5, 6, 7, 8])
@partial(jax.jit, static_argnames=['causal', 'sm_scale', 'softcap', 'use_alibi'])
def qkv_unpacked_attention_fn(
    q: jax.Array, k: jax.Array, v: jax.Array, bias: jax.Array | None = None, segment_ids: jax.Array | None = None,
    causal: bool = False, sm_scale: float | None = None, softcap: float | None = None, use_alibi: bool = False,
):
    o, lse = _forward_impl(q, k, v, bias, segment_ids, causal, sm_scale, softcap, use_alibi)
    return o


def _forward(
    q: jax.Array, k: jax.Array, v: jax.Array, bias: jax.Array | None, segment_ids: jax.Array | None,
    causal: bool, sm_scale: float | None, softcap: float | None = None, use_alibi: bool = False,
):
    o, lse = _forward_impl(q, k, v, bias, segment_ids, causal, sm_scale, softcap, use_alibi)
    return o, (q, k, v, bias, o, lse, segment_ids)


def _backward(
    causal: bool, sm_scale: float | None, softcap: float | None, use_alibi: bool,
    res, do,
):
    q, k, v, bias, o, lse, segment_ids = res
    dq, dk, dv = _backward_impl(q, k, v, do, o, lse, bias, segment_ids, causal, sm_scale, softcap, use_alibi)
    return dq, dk, dv, None, None


qkv_unpacked_attention_fn.defvjp(_forward, _backward)


@partial(jax.jit, static_argnames=['causal', 'sm_scale', 'softcap', 'use_alibi'])
def _qkv_packed_forward_impl(
    qkv: jax.Array, bias: jax.Array | None = None, segment_ids: jax.Array | None = None,
    causal: bool = False, sm_scale: float | None = None, softcap: float | None = None,
    use_alibi: bool = False,
) -> jax.Array:
    qkv = ArrayView(qkv, flatten_base=False)
    assert qkv.shape[2] == 3
    batch_size, seqlen, _, num_heads, head_dim = qkv.shape
    q = ArrayView(
        qkv, shape=(batch_size, seqlen, num_heads, head_dim),
        strides=(qkv.strides[0], qkv.strides[1], qkv.strides[-2], qkv.strides[-1]),
        offset=0,
    )
    k = ArrayView(
        qkv, shape=(batch_size, seqlen, num_heads, head_dim),
        strides=(qkv.strides[0], qkv.strides[1], qkv.strides[-2], qkv.strides[-1]),
        offset=1 * qkv.strides[2],
    )
    v = ArrayView(
        qkv, shape=(batch_size, seqlen, num_heads, head_dim),
        strides=(qkv.strides[0], qkv.strides[1], qkv.strides[-2], qkv.strides[-1]),
        offset=2 * qkv.strides[2],
    )
    return _forward_impl(q, k, v, bias, segment_ids, causal, sm_scale, softcap, use_alibi)


@partial(jax.jit, static_argnames=['causal', 'sm_scale', 'softcap', 'use_alibi'])
def _qkv_packed_backward_impl(
    qkv: jax.Array,
    do: jax.Array, o: jax.Array, lse: jax.Array, 
    bias: jax.Array | None = None, segment_ids: jax.Array | None = None, causal: bool = False, 
    sm_scale: float | None = None, softcap: float | None = None, use_alibi: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    qkv = ArrayView(qkv, flatten_base=False)
    batch_size, seqlen, _, num_heads, head_dim = qkv.shape
    q = ArrayView(
        qkv, shape=(batch_size, seqlen, num_heads, head_dim),
        strides=(qkv.strides[0], qkv.strides[1], qkv.strides[-2], qkv.strides[-1]),
        offset=0,
    )
    k = ArrayView(
        qkv, shape=(batch_size, seqlen, num_heads, head_dim),
        strides=(qkv.strides[0], qkv.strides[1], qkv.strides[-2], qkv.strides[-1]),
        offset=1 * qkv.strides[2],
    )
    v = ArrayView(
        qkv, shape=(batch_size, seqlen, num_heads, head_dim),
        strides=(qkv.strides[0], qkv.strides[1], qkv.strides[-2], qkv.strides[-1]),
        offset=2 * qkv.strides[2],
    )
    dq, dk, dv = _backward_impl(q, k, v, do, o, lse, bias, segment_ids, causal, sm_scale, softcap, use_alibi)
    return jax.numpy.stack([dq, dk, dv], axis=2)
    

@partial(jax.custom_vjp, nondiff_argnums=[3, 4, 5, 6])
@partial(jax.jit, static_argnames=['causal', 'sm_scale', 'softcap', 'use_alibi'])
def qkvpacked_attention_fn(
    qkv: jax.Array, bias: jax.Array | None = None, segment_ids: jax.Array | None = None,
    causal: bool = False, sm_scale: float | None = None, softcap: float | None = None,
    use_alibi: bool = False,
):
    o, lse = _qkv_packed_forward_impl(qkv, bias, segment_ids, causal, sm_scale, softcap, use_alibi)
    return o


def _qkvpacked_forward(
    qkv: jax.Array, bias: jax.Array | None, segment_ids: jax.Array | None,
    causal: bool, sm_scale: float | None, softcap: float | None = None,
    use_alibi: bool = False,
):
    o, lse = _qkv_packed_forward_impl(qkv, bias, segment_ids, causal, sm_scale, softcap, use_alibi)
    return o, (qkv, bias, o, lse, segment_ids)


def _qkvpacked_backward(
    causal: bool, sm_scale: float | None, softcap: float | None, use_alibi: bool,
    res, do,
):
    qkv, bias, o, lse, segment_ids = res
    dqkv = _qkv_packed_backward_impl(qkv, do, o, lse, bias, segment_ids, causal, sm_scale, softcap, use_alibi)
    return dqkv, None, None


qkvpacked_attention_fn.defvjp(_qkvpacked_forward, _qkvpacked_backward)



def attention_fn(
    q: jax.Array | None = None, k: jax.Array | None = None, v: jax.Array | None = None, 
    qkv: jax.Array | None = None, bias: jax.Array | None = None, segment_ids: jax.Array | None = None,
    causal: bool = False, sm_scale: float | None = None, softcap: float | None = None,
    use_alibi: bool = False,
) -> jax.Array:
    '''
    :param segment_ids: with shape (B, S), must be ascending on axis 1
    '''
    if qkv is None:
        return qkv_unpacked_attention_fn(q, k, v, bias, segment_ids, causal, sm_scale, softcap, use_alibi)
    else:
        assert q is None and k is None and v is None
        return qkvpacked_attention_fn(qkv, bias, segment_ids, causal, sm_scale, softcap, use_alibi)
    

