import jax
import jax.numpy as jnp
from flax import nnx


@jax.jit
def xla_swiglu(x, w, v, w2, bias_w=None, bias_v=None, bias_w2=None):
    xw, xv = jnp.dot(x, w), jnp.dot(x, v)
    if bias_w is not None:
        xw = xw + bias_w.reshape((1,) * (xw.ndim - 1) + (-1,))
    if bias_v is not None:
        xv = xv + bias_v.reshape((1,) * (xv.ndim - 1) + (-1,))
    @jax.remat
    def compute_gates(xw, xv):
        return jax.nn.silu(xw) * xv
    x = compute_gates(xw, xv)
    x = jnp.dot(x, w2)
    if bias_w2 is not None:
        x = x + bias_w2.reshape((1,) * (x.ndim - 1) + (-1,))
    return x


class SwiGLU(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        use_bias: bool = True,
        *,
        kernel_init = nnx.initializers.xavier_uniform(),
        bias_init = nnx.initializers.normal(1e-6),
        dtype: jax.typing.DTypeLike,
        param_dtype: jax.typing.DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        if hidden_dim is None:
            hidden_dim = embed_dim * 3
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias

        self.dtype = dtype
        self.param_dtype = param_dtype

        wv_kernel_init = kernel_init
        wv_bias_init = bias_init
        w2_kernel_init = kernel_init
        w2_bias_init = bias_init
        self.w = nnx.Param(
            wv_kernel_init(
                rngs.params(), (embed_dim, hidden_dim), param_dtype,
            )
        )
        self.v = nnx.Param(
            wv_kernel_init(
                rngs.params(), (embed_dim, hidden_dim), param_dtype,
            )
        )
        self.w2 = nnx.Param(
            w2_kernel_init(
                rngs.params(), (hidden_dim, embed_dim), param_dtype,
            )
        )

        if not use_bias:
            self.w_bias = nnx.Param(None)
            self.v_bias = nnx.Param(None)
            self.w2_bias = nnx.Param(None)
        else:
            self.w_bias = nnx.Param(wv_bias_init(rngs.params(), (hidden_dim,), param_dtype))
            self.v_bias = nnx.Param(wv_bias_init(rngs.params(), (hidden_dim,), param_dtype))
            self.w2_bias = nnx.Param(w2_bias_init(rngs.params(), (embed_dim,), param_dtype))

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.astype(self.dtype)
        w, v, w2 = jax.tree.map(lambda x: x.value.astype(self.dtype), (self.w, self.v, self.w2))
        if not self.use_bias:
            w_bias, v_bias, w2_bias = None, None, None
        else:
            w_bias, v_bias, w2_bias = jax.tree.map(lambda x: x.value.astype(self.dtype), (self.w_bias, self.v_bias, self.w2_bias))

        return xla_swiglu(x, w, v, w2, w_bias, v_bias, w2_bias)

