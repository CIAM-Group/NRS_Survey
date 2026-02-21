import jax
from flax import nnx


def Linear(
    in_features: int,
    out_features: int,
    *,
    use_bias: bool = True,
    dtype: jax.typing.DTypeLike = None,
    rngs: nnx.Rngs,
) -> nnx.Linear:
    return nnx.Linear(
        in_features, out_features,
        use_bias=use_bias, dtype=dtype, rngs=rngs,
        kernel_init=nnx.initializers.xavier_uniform(),
        bias_init=nnx.initializers.normal(stddev=1e-6),
    )
