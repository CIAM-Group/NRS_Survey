import jax
import jax.numpy as jnp
import optax

from functools import partial
import math
import typing as tp


def zeropower_via_newtonschulz5(
    G: jax.Array, steps=10, eps=1e-7, 
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
    qkv_special_case: bool = True,
) -> jax.Array:
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.astype(dtype)
    if qkv_special_case:
        if X.shape[0] * 3 == X.shape[1]:
            embed_dim = X.shape[0]
            Xq, Xk, Xv = jax.tree.map(
                partial(zeropower_via_newtonschulz5, steps=steps, eps=eps, dtype=dtype),
                (X[:, :embed_dim], X[:, embed_dim:embed_dim * 2], X[:, embed_dim * 2:]),
            )
            return jnp.concat([Xq, Xk, Xv], axis=-1)
    X /= (jnp.linalg.norm(X) + eps) # ensure top singular value <= 1
    trans = G.shape[0] > G.shape[1]
    if trans:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if trans:
        X = X.T
    return X


def if_dispatch_to_muon(x: jax.Array | None):
    if x is None:
        return False
    if x.ndim != 2:
        return False
    return True
    

def merge(x: jax.Array | None, y: jax.Array | None):
    assert x is None or y is None
    return x if x is not None else y


def set_leaves_to_false(pytree, match_keys: tp.Sequence[str] = []):
    from jax.tree_util import DictKey, GetAttrKey, SequenceKey, FlattenedIndexKey
    if len(match_keys) == 0:
        return pytree
    
    def if_set_to_false(keypath: tuple[DictKey | GetAttrKey | SequenceKey | FlattenedIndexKey]):
        for key in keypath:
            if isinstance(key, DictKey):
                if key.key in match_keys:
                    return True
            elif isinstance(key, GetAttrKey):
                if key.name in match_keys:
                    return True
            elif isinstance(key, SequenceKey):
                pass
            elif isinstance(key, FlattenedIndexKey):
                pass
            else:
                raise ValueError()
        return False
    
    return jax.tree_util.tree_map_with_path(lambda keypath, x: False if if_set_to_false(keypath) else x, pytree)


def multiplied_by_adjustment_ratio_for_muon(p: jax.Array):
    ''' Perform lr adjustment described in https://arxiv.org/pdf/2502.16982v1.
    '''
    assert p.ndim == 2
    adjustment_ratio = 0.2 * math.sqrt(max(*p.shape))
    return p * adjustment_ratio


def muon(
    lr: optax.ScalarOrSchedule = 1e-3,
    momentum: float = 0.95, ns_steps: int = 5,
    weight_decay: float = 0.,
    adamw_lr: optax.ScalarOrSchedule | None = None,
    adamw_b1: float = 0.95,
    adamw_b2: float = 0.95,
    adamw_eps: float = 1e-8,        # if loss blows up when using fp16, try 1e-7
    adamw_eps_root: float = 0.,     # if loss blows up when using fp16, try 1e-7
    adamw_weight_decay: float = 1e-4,
    adamw_clip_norm: float | None = None,
    *,
    nesterov: bool = True,
    adamw_param_keys: tp.Sequence[str] = [],
    qkv_special_case: bool = False,
    adjust_lr: bool = True,
) -> optax.GradientTransformation:
    if adamw_lr is None:
        adamw_lr = lr
    tx_adamw = optax.adamw(
        learning_rate=adamw_lr,
        b1=adamw_b1, b2=adamw_b2,
        eps=adamw_eps, eps_root=adamw_eps_root,
        weight_decay=adamw_weight_decay, nesterov=nesterov,
    )
    if adamw_clip_norm is not None:
        tx_adamw = optax.chain(
            optax.clip_by_global_norm(adamw_clip_norm),
            tx_adamw,
        )
    tx_lr = optax.scale_by_learning_rate(learning_rate=lr, flip_sign=True)
    if weight_decay != 0.:
        tx_lr = optax.chain(
            optax.add_decayed_weights(weight_decay, None),
            tx_lr,
        )

    def init_fn(params):
        to_muon = jax.tree.map(if_dispatch_to_muon, params)
        to_muon = set_leaves_to_false(to_muon, match_keys=adamw_param_keys)
        params_muon_part = jax.tree.map(lambda p, to_moun: p if to_moun else None, params, to_muon)
        params_adamw_part = jax.tree.map(lambda p, to_moun: p if not to_moun else None, params, to_muon)
        opt_state = {
            'adamw': tx_adamw.init(params_adamw_part),
            'muon': jax.tree.map(lambda x: jnp.zeros_like(x) if x is not None else None, params_muon_part),
            'lr': tx_lr.init(params_muon_part),
        }
        return opt_state

    def update_fn(updates, state: dict[str, tp.Any], params=None):
        assert state.keys() == {'adamw', 'muon', 'lr'}

        grad = updates
        del updates

        to_muon = jax.tree.map(if_dispatch_to_muon, grad)
        to_muon = set_leaves_to_false(to_muon, match_keys=adamw_param_keys)
        grad_adamw_part = jax.tree.map(lambda p, to_moun: p if not to_moun else None, grad, to_muon)
        grad_muon_part = jax.tree.map(lambda p, to_moun: p if to_moun else None, grad, to_muon)
        state_adamw_part = state['adamw']
        state_muon_part = state['muon']
        state_lr_part = state['lr']
        params_adamw_part = jax.tree.map(lambda p, to_moun: p if not to_moun else None, params, to_muon)
        params_muon_part = jax.tree.map(lambda p, to_moun: p if to_moun else None, params, to_muon)

        update_adamw_part, state_adamw_part = tx_adamw.update(grad_adamw_part, state_adamw_part, params_adamw_part)

        # moun process
        state_muon_part = jax.tree.map(lambda B, G: momentum * B + G if B is not None else None, state_muon_part, grad_muon_part)
        def _process_fn(x: jax.Array | None):
            if x is None:
                return None
            x = zeropower_via_newtonschulz5(x, steps=ns_steps, eps=1e-7, dtype=jnp.bfloat16, qkv_special_case=qkv_special_case).astype(x.dtype)
            if adjust_lr:
                x = multiplied_by_adjustment_ratio_for_muon(x)
            return x
        if not nesterov:
            update_muon_part = jax.tree.map(_process_fn, state_muon_part)
        else:
            update_muon_part = jax.tree.map(
                _process_fn,
                jax.tree.map(
                    lambda B, G: G + momentum * B if B is not None else None,
                    state_muon_part, grad_muon_part,
                )
            )
        update_muon_part, state_lr_part = tx_lr.update(update_muon_part, state_lr_part, params_muon_part)

        # merge
        update = jax.tree.map(
            lambda x, y: merge(x, y), 
            update_adamw_part, update_muon_part, 
            is_leaf=lambda x: x is None,
        )
        state = {
            'adamw': state_adamw_part,
            'muon': state_muon_part,
            'lr': state_lr_part,
        }
        
        return update, state

    return optax.GradientTransformation(init_fn, update_fn)
