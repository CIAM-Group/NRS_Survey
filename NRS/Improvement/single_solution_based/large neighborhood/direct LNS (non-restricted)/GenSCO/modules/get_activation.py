import jax


def get_activation(activation: str):
    match activation:
        case 'silu' | 'swish':
            return jax.remat(jax.nn.silu)
        case 'relu':
            return jax.nn.relu
        case _:
            raise ValueError()
