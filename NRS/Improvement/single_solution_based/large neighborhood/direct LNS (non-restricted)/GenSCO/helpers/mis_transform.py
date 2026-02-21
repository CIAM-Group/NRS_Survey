import jax
import jax.numpy as jnp


def edges2adj(edges: jax.Array, num_nodes_padded: int, dtype: jax.typing.DTypeLike = jnp.float32):
    def _unbatched_impl(edges: jax.Array):
        adj = jnp.zeros([num_nodes_padded, num_nodes_padded], dtype=dtype)
        adj = adj.at[edges[:, 0], edges[:, 1]].set(1)
        adj = jnp.fill_diagonal(adj, 0, inplace=False)
        return adj
    return jax.vmap(_unbatched_impl)(edges)
