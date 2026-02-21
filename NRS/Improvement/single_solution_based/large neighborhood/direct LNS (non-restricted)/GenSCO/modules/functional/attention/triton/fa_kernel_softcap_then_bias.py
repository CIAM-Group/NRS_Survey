"""
*Experimental* implementation of FlashAttention in Triton.
Tested with triton==2.0.0.dev20221202.
Triton 2.0 has a new backend (MLIR) but seems like it doesn't yet work for head dimensions
other than 64:
https://github.com/openai/triton/blob/d376020f90002757eea3ea9475d4f7cfc2ec5ead/python/triton/ops/flash_attention.py#L207
We'll update this implementation with the new Triton backend once this is fixed.

We use the FlashAttention implementation from Phil Tillet a starting point.
https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py

Changes:
- Implement both causal and non-causal attention.
- Implement both self-attention and cross-attention.
- Support arbitrary seqlens (not just multiples of 128), for both forward and backward.
- Support all head dimensions up to 128 (not just 16, 32, 64, 128), for both forward and backward.
- Support attention bias.
- Speed up the forward pass a bit, and only store the LSE instead of m and l.
- Make the backward for d=128 much faster by reducing register spilling.
- Optionally parallelize the backward pass across seqlen_k, to deal with the case of
small batch size * nheads.

Caution:
- This is an *experimental* implementation. The forward pass should be quite robust but
I'm not 100% sure that the backward pass doesn't have race conditions (due to the Triton compiler).
- This implementation has only been tested on A100.
- If you plan to use headdim other than 64 and 128, you should test for race conditions
(due to the Triton compiler), as done in tests/test_flash_attn.py
"test_flash_attn_triton_race_condition". I've tested and fixed many race conditions
for different head dimensions (40, 48, 64, 128, 80, 88, 96), but I'm still not 100% confident
that there are none left for other head dimensions.

Differences between this Triton version and the CUDA version:
- Triton version doesn't support dropout.
- Triton forward is generally faster than CUDA forward, while Triton backward is
generally slower than CUDA backward. Overall Triton forward + backward is slightly slower
than CUDA forward + backward.
- Triton version doesn't support different sequence lengths in a batch (i.e., RaggedTensor/NestedTensor).
- Triton version supports attention bias, while CUDA version doesn't.
"""


'''
Taken from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py,
and apply some modifications:
1. Ravel both ptr and val in atomic_add to improve performace. 
2. Add softcap support. (Also use large negative number instead of -inf to mask to avoid numeric problem when applying softcap.)
'''

import math

import triton
import triton.language as tl
from triton.language.extra.libdevice import pow

from .tanh import tanh


from jax._src import test_util as jtu
# Disabling autotune for now, set num_warps=4 if headdim=64 and num_warps=8 if headdim=128
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        # This config has a race condition when EVEN_M == False, disabling it for now.
        triton.Config({"BLOCK_M": 64 if not jtu.is_cuda_compute_capability_at_least('9.0') else 128, "BLOCK_N": 64}, num_warps=4, num_stages=1),
    ],
    key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K', 'BIAS_TYPE', 'IS_CAUSAL', 'BLOCK_HEADDIM']
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
        "use_fp16_accum": lambda args: False,
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Bias,
    segment_ids,
    kv_bound,       # [B, cdiv(seqlen, 64), 2]
    Out,
    Lse,
    TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,

    softcap: tl.constexpr,

    k_ptr_offset: tl.constexpr,
    v_ptr_offset: tl.constexpr,

    use_fp16_accum: tl.constexpr,

    has_segment_ids: tl.constexpr,
    has_kv_bound: tl.constexpr,

    alibi_scale: tl.constexpr,
):
    if k_ptr_offset != 0:
        K += k_ptr_offset
    if v_ptr_offset != 0:
        V += v_ptr_offset

    if has_kv_bound:
        tl.static_assert(has_segment_ids)

    compute_dtype: tl.constexpr = Q.dtype.element_ty
    accum_dtype: tl.constexpr = tl.float16 if compute_dtype == tl.float16 and use_fp16_accum else tl.float32

    DEFAULT_MASK_VALUE: tl.constexpr = -2.381976426469702e+38
    log2e: tl.constexpr = 1.4426950408889634

    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # off_b = tl.program_id(1)
    # off_h = tl.program_id(2)
    # off_hb = off_b * nheads + off_h
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # Initialize pointers to Q, K, V
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    # initialize pointer to m and l
    # t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
            )
    if has_segment_ids:
        q_segment_ids = tl.load(segment_ids + off_b * seqlen_q + offs_m)
    # loop over k, v and update accumulator
    start_n = 0
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    if has_kv_bound:        # [B, cdiv(seqlen, 64), 2]
        if BLOCK_M == 64:
            kv_lower_bound = tl.load(kv_bound + (off_b * tl.cdiv(seqlen_q, 64) + start_m) * 2) * 64
            kv_upper_bound = tl.load(kv_bound + (off_b * tl.cdiv(seqlen_q, 64) + start_m) * 2 + 1) * 64
            if BLOCK_M == 128:
                kv_lower_bound -= (kv_lower_bound % 128 != 0) * 64
                kv_upper_bound += (kv_upper_bound % 128 != 0) * 64
        elif BLOCK_M == 128:
            kv_lower_bound = tl.load(kv_bound + (off_b * tl.cdiv(seqlen_q, 64) * 2) + start_m * 4) * 64
            kv_upper_bound = tl.load(kv_bound + (off_b * tl.cdiv(seqlen_q, 64) * 2) + start_m * 4 + 3) * 64
            if BLOCK_N == 128:
                kv_lower_bound -= (kv_lower_bound % 128 != 0) * 64
                kv_upper_bound += (kv_upper_bound % 128 != 0) * 64
        else:
            tl.static_assert(False)
        start_n = tl.maximum(start_n, kv_lower_bound)
        end_n = tl.minimum(end_n, kv_upper_bound)

    for start_n in range(start_n, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )

        qk = tl.dot(q, k.T, out_dtype=accum_dtype)

        mask = None
        if has_segment_ids:
            kv_segment_ids = tl.load(segment_ids + (off_b * seqlen_k + start_n) + offs_n)
            mask = q_segment_ids[:, None] == kv_segment_ids[None, :]
            if not EVEN_M:
                mask = tl.where(
                    offs_m[:, None] >= seqlen_q,
                    1,
                    mask,
                )   # or numerical error may occur

        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                    ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            if alibi_scale is not None:
                bias *= pow(alibi_scale, off_h + 1)
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.


            # qk = qk * softmax_scale + bias
            # if softcap > 0.:
            #     qk = softcap * tanh(qk * (1. / softcap))
            if softcap > 0.:
                qk = softcap * tanh(qk * (softmax_scale * 1. / softcap)) + bias
            else:
                qk = qk * softmax_scale + bias


            # Trying to combine the two masks seem to make the result wrong
            if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
                if mask is None:
                    mask = (start_n + offs_n)[None, :] < seqlen_k
                else:
                    mask &= (start_n + offs_n)[None, :] < seqlen_k
            if IS_CAUSAL:
                if mask is None:
                    mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
                else:
                    mask &= offs_m[:, None] >= (start_n + offs_n)[None, :]
            if mask is not None:
                qk = tl.where(mask, qk, DEFAULT_MASK_VALUE)
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp2((qk - m_ij[:, None]) * log2e)
        else:
            if softcap > 0.:
                qk *= softmax_scale * (1. / softcap)
                qk = tanh(qk) * softcap
            else:
                qk *= softmax_scale
            if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
                if mask is None:
                    mask = (start_n + offs_n)[None, :] < seqlen_k
                else:
                    mask &= (start_n + offs_n)[None, :] < seqlen_k
            if IS_CAUSAL:
                if mask is None:
                    mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
                else:
                    mask &= offs_m[:, None] >= (start_n + offs_n)[None, :]
            if mask is not None:
                qk = tl.where(mask, qk, DEFAULT_MASK_VALUE)
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp2((qk - m_ij[:, None]) * log2e)


        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp2((m_i - m_ij) * log2e)

        # # -- update output accumulator --
        # BUG: have to store and immediately load
        # tl.store(t_ptrs, acc_o_scale)
        # acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]
        # update acc_o
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v, out_dtype=accum_dtype)

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp2((lse_i - m_ij) * log2e) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp2((m_i - lse_i) * log2e)
    # BUG: have to store and immediately load
    # tl.store(t_ptrs, o_scale)
    # o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )


@triton.jit
def _bwd_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    nheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # load
    o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO
        + off_b * stride_dob
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.jit
def _bwd_store_dk_dv(
    dk_ptrs,
    dv_ptrs,
    dk,
    dv,
    offs_n,
    offs_d,
    seqlen_k,
    headdim,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
):
    # [2022-11-01] TD: Same bug. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.store(dv_ptrs), there's a race condition
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))


@triton.jit
def _bwd_kernel_one_col_block(
    start_n,
    Q,
    K,
    V,
    Bias,
    DO,
    segment_ids,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_bm,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    seqlen_q,
    seqlen_k,
    headdim,
    ATOMIC_ADD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,

    softcap: tl.constexpr,

    use_fp16_accum: tl.constexpr,

    has_segment_ids: tl.constexpr,
    off_b,
    q_lower_bound, q_upper_bound,

    alibi_scale: tl.constexpr,
    off_h,
):
    compute_dtype: tl.constexpr = Q.dtype.element_ty
    accum_dtype: tl.constexpr = tl.float16 if compute_dtype == tl.float16 and use_fp16_accum else tl.float32

    DEFAULT_MASK_VALUE: tl.constexpr = -2.381976426469702e+38
    log2e: tl.constexpr = 1.4426950408889634

    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)
    begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
    # initialize row/col offsets
    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # initialize pointers to value-like data
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = Bias + (offs_qm[:, None] * stride_bm + offs_n[None, :])
    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    # There seems to be some problem with Triton pipelining that makes results wrong for
    # headdim=64, seqlen=(113, 255), bias_type='matrix'. In this case the for loop
    # may have zero step, and pipelining with the bias matrix could screw it up.
    # So we just exit early.
    if begin_m >= seqlen_q:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
        _bwd_store_dk_dv(
            dk_ptrs,
            dv_ptrs,
            dk,
            dv,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
        )
        return
    # k and v stay in SRAM throughout
    # [2022-10-30] TD: Same bug as the fwd. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.load(k_ptrs), we get the wrong output!
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(
                k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
            v = tl.load(
                v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
    if has_segment_ids:
        kv_segment_ids = tl.load(segment_ids + off_b * seqlen_k + offs_n)
    # loop over rows
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    if q_lower_bound is not None:
        begin_m = tl.maximum(begin_m, q_lower_bound)
    end_m = num_block_m * BLOCK_M
    if q_upper_bound is not None:
        end_m = tl.minimum(end_m, q_upper_bound)
    for start_m in range(begin_m, end_m, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        # load q, k, v, do on-chip
        # Same bug as below. Otherwise gives wrong result for headdim=40, seqlen=(128, 117)
        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(
                    q_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        # recompute p = softmax(qk, dim=-1).T
        qk = tl.dot(q, k.T, out_dtype=accum_dtype)

        mask = None
        if has_segment_ids:
            q_segment_ids = tl.load(segment_ids + off_b * seqlen_q + offs_m_curr)
            mask = q_segment_ids[:, None] == kv_segment_ids[None, :]

        if BIAS_TYPE != "none":
            # tl.debug_barrier()  # Race condition otherwise
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs, mask=offs_n < seqlen_k, other=0.0).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            if alibi_scale is not None:
                bias *= pow(alibi_scale, off_h + 1)
            # qk = qk * softmax_scale + bias
        # There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
        # Also wrong for headdim=64.
        if not (EVEN_M & EVEN_HEADDIM):
            # tl.debug_barrier()
            ...
        lse_i = tl.load(LSE + offs_m_curr)
        if BIAS_TYPE == "none":
            if softcap > 0.:
                qk_capped = (softcap * log2e) * tanh(qk * (softmax_scale * (1. / softcap)))
                # Trying to combine the two masks seem to make the result wrong
                if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
                    if mask is None:
                        mask = offs_n[None, :] < seqlen_k
                    else:
                        mask &= offs_n[None, :] < seqlen_k
                if IS_CAUSAL:
                    if mask is None:
                        mask = offs_m_curr[:, None] >= (offs_n[None, :])
                    else:
                        mask &= offs_m_curr[:, None] >= (offs_n[None, :])
                if mask is not None:
                    qk_capped = tl.where(mask, qk_capped, DEFAULT_MASK_VALUE)
                p = tl.exp2(qk_capped - lse_i[:, None] * log2e)
            else:
                # Trying to combine the two masks seem to make the result wrong
                if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
                    if mask is None:
                        mask = offs_n[None, :] < seqlen_k
                    else:
                        mask &= offs_n[None, :] < seqlen_k
                if IS_CAUSAL:
                    if mask is None:
                        mask = offs_m_curr[:, None] >= (offs_n[None, :])
                    else:
                        mask &= offs_m_curr[:, None] >= (offs_n[None, :])
                if mask is not None:
                    qk = tl.where(mask, qk, DEFAULT_MASK_VALUE)
                p = tl.exp(qk * softmax_scale - lse_i[:, None])
        else:
            if softcap > 0.:
                # qk = qk * softmax_scale       # move into next line
                qk_capped = softcap * tanh(qk * (softmax_scale * 1. / softcap)) + bias
                qk_capped *= log2e
                # Trying to combine the two masks seem to make the result wrong
                if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
                    if mask is None:
                        mask = offs_n[None, :] < seqlen_k
                    else:
                        mask &= offs_n[None, :] < seqlen_k
                if IS_CAUSAL:
                    if mask is None:
                        mask = offs_m_curr[:, None] >= (offs_n[None, :])
                    else:
                        mask &= offs_m_curr[:, None] >= (offs_n[None, :])
                if mask is not None:
                    qk_capped = tl.where(mask, qk_capped, DEFAULT_MASK_VALUE)
                p = tl.exp2(qk_capped - lse_i[:, None] * log2e)
            else:
                qk = qk * softmax_scale + bias
                # Trying to combine the two masks seem to make the result wrong
                if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
                    if mask is None:
                        mask = offs_n[None, :] < seqlen_k
                    else:
                        mask &= offs_n[None, :] < seqlen_k
                if IS_CAUSAL:
                    if mask is None:
                        mask = offs_m_curr[:, None] >= (offs_n[None, :])
                    else:
                        mask &= offs_m_curr[:, None] >= (offs_n[None, :])
                if mask is not None:
                    qk = tl.where(mask, qk, DEFAULT_MASK_VALUE)
                p = tl.exp(qk - lse_i[:, None])
        # compute dv
        # [2022-10-30] TD: A Triton bug: if EVEN_M=True and EVEN_HEADDIM=False, if we call
        # do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0), we get wrong outputs
        # in the case of headdim=48/96, seqlen_q & seqlen_k >= 512. If headdim=40 or seqlen < 512,
        # the output is correct.
        if EVEN_M & EVEN_HEADDIM:
            do = tl.load(do_ptrs)
        else:
            # [2022-11-01] TD: Triton bug, there's a race condition if we just use m_mask and not d_mask.
            do = tl.load(
                do_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )
        # if EVEN_M:
        #     if EVEN_HEADDIM:
        #         do = tl.load(do_ptrs)
        #     else:
        #         do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        # else:
        #     if EVEN_HEADDIM:
        #         do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
        #     else:
        #         do = tl.load(do_ptrs, mask=(offs_m_curr[:, None] < seqlen_q)
        #                                    & (offs_d[None, :] < headdim), other=0.0)
        dv += tl.dot(p.to(do.dtype).T, do, out_dtype=accum_dtype)
        # compute dp = dot(v, do)
        # There seems to be a race condition when headdim=48/96, and dq, dk are wrong.
        # Also wrong for headdim=128, seqlen=(108, 256), and ATOMIC_ADD=True
        # Also wrong for headdim=64, seqlen=(1023, 1024), and ATOMIC_ADD=False
        if not (EVEN_M & EVEN_HEADDIM):
            # tl.debug_barrier()
            pass
        dp = tl.dot(do, v.T, out_dtype=accum_dtype)
        # There's a race condition for headdim=48
        if not EVEN_HEADDIM:
            # tl.debug_barrier()
            pass
        # compute ds = p * (dp - delta[:, None])
        # Putting the subtraction after the dp matmul (instead of before) is slightly faster
        Di = tl.load(D + offs_m_curr)
        # Converting ds to q.dtype here reduces register pressure and makes it much faster
        # for BLOCK_HEADDIM=128
        ds = (p * (dp - Di[:, None]) * softmax_scale)

        # softcap backward
        if softcap > 0.:
            normalized = qk * (softmax_scale * (1. / softcap))
            d = tanh(normalized)
            g = ds * (1 - d)
            ds = g + g * d

        ds = ds.to(q.dtype)
        # compute dk = dot(ds.T, q)
        dk += tl.dot(ds.T, q, out_dtype=accum_dtype)
        # compute dq
        if not (
            EVEN_M & EVEN_HEADDIM
        ):  # Otherewise there's a race condition when BIAS_TYPE='matrix'
            # tl.debug_barrier()
            pass
        if not ATOMIC_ADD:
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                dq = tl.load(dq_ptrs, eviction_policy="evict_last")
                dq += tl.dot(ds, k, out_dtype=accum_dtype)
                tl.store(dq_ptrs, dq, eviction_policy="evict_last")
            else:
                if EVEN_HEADDIM:
                    dq = tl.load(
                        dq_ptrs,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k, out_dtype=accum_dtype)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        eviction_policy="evict_last",
                    )
                else:
                    dq = tl.load(
                        dq_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k, out_dtype=accum_dtype)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        eviction_policy="evict_last",
                    )
        else:  # If we're parallelizing across the seqlen_k dimension
            dq = tl.dot(ds, k, out_dtype=accum_dtype)
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                tl.atomic_add(tl.ravel(dq_ptrs), tl.ravel(dq), sem='relaxed')
            else:
                if EVEN_HEADDIM:
                    tl.atomic_add(
                        tl.ravel(dq_ptrs), tl.ravel(dq), 
                        mask=tl.ravel(tl.broadcast_to(offs_m_curr[:, None] < seqlen_q, BLOCK_M, BLOCK_HEADDIM)), 
                        sem='relaxed'
                    )
                else:
                    tl.atomic_add(
                        tl.ravel(dq_ptrs),
                        tl.ravel(dq),
                        mask=tl.ravel((offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim)),
                        sem='relaxed'
                    )
        # increment pointers
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
        if BIAS_TYPE == "matrix":
            b_ptrs += BLOCK_M * stride_bm
    # write-back
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    _bwd_store_dk_dv(
        dk_ptrs,
        dv_ptrs,
        dk,
        dv,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        EVEN_HEADDIM=EVEN_HEADDIM,
    )


# def init_to_zero(name):
#     return lambda nargs: nargs[name].zero_()


@triton.jit
def handle_q_bound(q_bound, off_b, BLOCK_M, BLOCK_N, seqlen_k, start_n):
    if BLOCK_N == 64:
        q_lower_bound = tl.load(q_bound + (off_b * tl.cdiv(seqlen_k, 64) + start_n) * 2) * 64
        q_upper_bound = tl.load(q_bound + (off_b * tl.cdiv(seqlen_k, 64) + start_n) * 2 + 1) * 64
        if BLOCK_M == 128:
            q_lower_bound -= (q_lower_bound % 128 != 0) * 64
            q_upper_bound += (q_upper_bound % 128 != 0) * 64
    elif BLOCK_N == 128:
        q_lower_bound = tl.load(q_bound + (off_b * tl.cdiv(seqlen_k, 64) * 2) + start_n * 4) * 64
        q_upper_bound = tl.load(q_bound + (off_b * tl.cdiv(seqlen_k, 64) * 2) + start_n * 4 + 3) * 64
        if BLOCK_M == 128:
            q_lower_bound -= (q_lower_bound % 128 != 0) * 64
            q_upper_bound += (q_upper_bound % 128 != 0) * 64
    else:
        tl.static_assert(False)
    return q_lower_bound, q_upper_bound


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False},
            num_warps=8,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False},
            num_warps=4,
            num_stages=1,
        ),
    ],
    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "BIAS_TYPE", "IS_CAUSAL", "BLOCK_HEADDIM"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
        "use_fp16_accum": lambda args: False,
    }
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    Bias,
    DO,
    segment_ids,
    q_bound,
    
    LSE,
    D,

    DQ,
    DK,
    DV,

    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,

    softcap: tl.constexpr,

    k_ptr_offset: tl.constexpr,
    v_ptr_offset: tl.constexpr,

    use_fp16_accum: tl.constexpr,

    has_segment_ids: tl.constexpr,
    has_q_bound: tl.constexpr,

    alibi_scale: tl.constexpr,
):
    if k_ptr_offset != 0:
        K += k_ptr_offset
    if v_ptr_offset != 0:
        V += v_ptr_offset
    
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    if BIAS_TYPE != "none":
        Bias += off_b * stride_bb + off_h * stride_bh
    # pointer to row-wise quantities in value-like data
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded

    if has_q_bound:
        tl.static_assert(has_segment_ids)

    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            q_lower_bound, q_upper_bound = None, None
            if has_q_bound:
                q_lower_bound, q_upper_bound = handle_q_bound(q_bound, off_b, BLOCK_M, BLOCK_N, seqlen_k, start_n)
            _bwd_kernel_one_col_block(
                start_n,
                Q,
                K,
                V,
                Bias,
                DO,
                segment_ids,
                DQ,
                DK,
                DV,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_bm,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                seqlen_q,
                seqlen_k,
                headdim,
                ATOMIC_ADD=False,
                BIAS_TYPE=BIAS_TYPE,
                IS_CAUSAL=IS_CAUSAL,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,

                softcap=softcap,

                use_fp16_accum=use_fp16_accum,

                has_segment_ids=has_segment_ids,
                off_b=off_b,

                q_lower_bound=q_lower_bound,
                q_upper_bound=q_upper_bound,

                alibi_scale=alibi_scale,
                off_h=off_h,
            )
    else:
        start_n = tl.program_id(0)
        q_lower_bound, q_upper_bound = None, None
        if has_q_bound:
            q_lower_bound, q_upper_bound = handle_q_bound(q_bound, off_b, BLOCK_M, BLOCK_N, seqlen_k, start_n)
        _bwd_kernel_one_col_block(
            start_n,
            Q,
            K,
            V,
            Bias,
            DO,
            segment_ids,
            DQ,
            DK,
            DV,
            LSE,
            D,
            softmax_scale,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_bm,
            stride_dom,
            stride_dqm,
            stride_dkn,
            stride_dvn,
            seqlen_q,
            seqlen_k,
            headdim,
            ATOMIC_ADD=True,
            BIAS_TYPE=BIAS_TYPE,
            IS_CAUSAL=IS_CAUSAL,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,

            softcap=softcap,

            use_fp16_accum=use_fp16_accum,

            has_segment_ids=has_segment_ids,
            off_b=off_b,

            q_lower_bound=q_lower_bound,
            q_upper_bound=q_upper_bound,

            alibi_scale=alibi_scale,
            off_h=off_h,
        )


try:
    import torch

    def _flash_attn_forward(q, k, v, bias=None, causal=False, softmax_scale=None, softcap=0.):
        # shape constraints
        batch, seqlen_q, nheads, d = q.shape
        _, seqlen_k, _, _ = k.shape
        assert k.shape == (batch, seqlen_k, nheads, d)
        assert v.shape == (batch, seqlen_k, nheads, d)
        assert d <= 128, "FlashAttention only support head dimensions up to 128"
        assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
        assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
        assert q.is_cuda and k.is_cuda and v.is_cuda
        softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

        has_bias = bias is not None
        bias_type = "none"
        if has_bias:
            assert bias.dtype in [q.dtype, torch.float]
            assert bias.is_cuda
            assert bias.dim() == 4
            if bias.stride(-1) != 1:
                bias = bias.contiguous()
            if bias.shape[2:] == (1, seqlen_k):
                bias_type = "vector"
            elif bias.shape[2:] == (seqlen_q, seqlen_k):
                bias_type = "matrix"
            else:
                raise RuntimeError(
                    "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
                )
            bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
        bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

        seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
        lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
        tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
        o = torch.empty_like(q)

        BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
        # BLOCK = 128
        # num_warps = 4 if d <= 64 else 8
        grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
        _fwd_kernel[grid](
            q,
            k,
            v,
            bias,
            o,
            lse,
            tmp,
            softmax_scale,
            q.stride(0),
            q.stride(2),
            q.stride(1),
            k.stride(0),
            k.stride(2),
            k.stride(1),
            v.stride(0),
            v.stride(2),
            v.stride(1),
            *bias_strides,
            o.stride(0),
            o.stride(2),
            o.stride(1),
            nheads,
            seqlen_q,
            seqlen_k,
            seqlen_q_rounded,
            d,
            seqlen_q // 32,
            seqlen_k // 32,  # key for triton cache (limit number of compilations)
            # Can't use kwargs here because triton autotune expects key to be args, not kwargs
            # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
            bias_type,
            causal,
            BLOCK_HEADDIM,
            # BLOCK_M=BLOCK,
            # BLOCK_N=BLOCK,
            # num_warps=num_warps,
            # num_stages=1,
            softcap=softcap,
        )
        return o, lse, softmax_scale  # softmax_scale could have been updated


    def _flash_attn_backward(
        do, q, k, v, o, lse, dq, dk, dv, bias=None, causal=False, softmax_scale=None, softcap=0.,
    ):
        # Make sure that the last dimension is contiguous
        if do.stride(-1) != 1:
            do = do.contiguous()
        batch, seqlen_q, nheads, d = q.shape
        _, seqlen_k, _, _ = k.shape
        # assert d in {16, 32, 64, 128}
        assert d <= 128
        seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
        assert lse.shape == (batch, nheads, seqlen_q_rounded)
        assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
        assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
        softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
        dq_accum = torch.zeros_like(q, dtype=torch.float32)
        # dq_accum = torch.empty_like(q, dtype=torch.float32)
        delta = torch.empty_like(lse)
        # delta = torch.zeros_like(lse)

        BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
        grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
        _bwd_preprocess_do_o_dot[grid](
            o,
            do,
            delta,
            o.stride(0),
            o.stride(2),
            o.stride(1),
            do.stride(0),
            do.stride(2),
            do.stride(1),
            nheads,
            seqlen_q,
            seqlen_q_rounded,
            d,
            BLOCK_M=128,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
        )

        has_bias = bias is not None
        bias_type = "none"
        if has_bias:
            assert bias.dtype in [q.dtype, torch.float]
            assert bias.is_cuda
            assert bias.dim() == 4
            assert bias.stride(-1) == 1
            if bias.shape[2:] == (1, seqlen_k):
                bias_type = "vector"
            elif bias.shape[2:] == (seqlen_q, seqlen_k):
                bias_type = "matrix"
            else:
                raise RuntimeError(
                    "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
                )
            bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
        bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

        # BLOCK_M = 128
        # BLOCK_N = 64
        # num_warps = 4
        grid = lambda META: (
            triton.cdiv(seqlen_k, META["BLOCK_N"]) if META["SEQUENCE_PARALLEL"] else 1,
            batch * nheads,
        )
        _bwd_kernel[grid](
            q,
            k,
            v,
            bias,
            do,
            
            lse,
            delta,

            dq_accum,
            dk,
            dv,
            
            softmax_scale,
            q.stride(0),
            q.stride(2),
            q.stride(1),
            k.stride(0),
            k.stride(2),
            k.stride(1),
            v.stride(0),
            v.stride(2),
            v.stride(1),
            *bias_strides,
            do.stride(0),
            do.stride(2),
            do.stride(1),
            dq_accum.stride(0),
            dq_accum.stride(2),
            dq_accum.stride(1),
            dk.stride(0),
            dk.stride(2),
            dk.stride(1),
            dv.stride(0),
            dv.stride(2),
            dv.stride(1),
            nheads,
            seqlen_q,
            seqlen_k,
            seqlen_q_rounded,
            d,
            seqlen_q // 32,
            seqlen_k // 32,  # key for triton cache (limit number of compilations)
            # Can't use kwargs here because triton autotune expects key to be args, not kwargs
            # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
            bias_type,
            causal,
            BLOCK_HEADDIM,
            # SEQUENCE_PARALLEL=False,
            # BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            # num_warps=num_warps,
            # num_stages=1,
            softcap=softcap,
        )
        dq.copy_(dq_accum)


    class FlashAttnFunc(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, bias=None, causal=False, softmax_scale=None, softcap=0.):
            """
            q: (batch_size, seqlen_q, nheads, headdim)
            k, v: (batch_size, seqlen_k, nheads, headdim)
            bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
                For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
                ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
            """
            # Make sure that the last dimension is contiguous
            q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
            o, lse, ctx.softmax_scale = _flash_attn_forward(
                q, k, v, bias=bias, causal=causal, softmax_scale=softmax_scale, softcap=softcap
            )
            ctx.save_for_backward(q, k, v, o, lse, bias)
            ctx.causal = causal
            ctx.softcap = softcap
            return o

        @staticmethod
        def backward(ctx, do):
            q, k, v, o, lse, bias = ctx.saved_tensors
            assert not ctx.needs_input_grad[3], "FlashAttention does not support bias gradient yet"
            # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
            # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
            with torch.inference_mode():
                dq = torch.empty_like(q)
                dk = torch.empty_like(k)
                dv = torch.empty_like(v)
                _flash_attn_backward(
                    do,
                    q,
                    k,
                    v,
                    o,
                    lse,
                    dq,
                    dk,
                    dv,
                    bias=bias,
                    causal=ctx.causal,
                    softmax_scale=ctx.softmax_scale,
                    softcap=ctx.softcap,
                )
            return dq, dk, dv, None, None, None, None


    flash_attn_func = FlashAttnFunc.apply
except:
    pass


if __name__ == '__main__':
    from functools import partial
    
    import time
    import torch
    torch.manual_seed(0)

    device = 'cuda:0'
    dtype = torch.bfloat16

    softcap = 20.

    B, S, H, D = 16, 1024 * 4 // 2, 16, 32
    attention = flash_attn_func
    flash_attn_func = lambda q, k, v: attention(q, k, v, None, False, D ** (-0.5), softcap)
    q = torch.randn([B, S, H, D], dtype=dtype, device=device, requires_grad=True)
    k = torch.randn([B, S, H, D], dtype=dtype, device=device, requires_grad=True)
    v = torch.randn([B, S, H, D], dtype=dtype, device=device, requires_grad=True)

    times = 20
    for _ in range(5):
        result: torch.Tensor = flash_attn_func(q, k, v,)    
        result.mean().backward()
        torch.cuda.synchronize(device=device)
        # q.grad, k.grad, v.grad = None, None, None

    q.grad, k.grad, v.grad = None, None, None




    for _ in range(20):
        start_time = time.time_ns()
        for _ in range(times):
            result = flash_attn_func(q, k, v)
            torch.cuda.synchronize(device=device)
        end_time = time.time_ns()
        print(f'Time Fwd: {(end_time - start_time) / 1e6} ms')

    for _ in range(20):
        start_time = time.time_ns()
        for _ in range(times):
            result = flash_attn_func(q, k, v)
            result.mean().backward()
            torch.cuda.synchronize(device=device)
            # q.grad, k.grad, v.grad = None, None, None
        end_time = time.time_ns()
        print(f'Time FwdBwd: {(end_time - start_time) / 1e6} ms')


