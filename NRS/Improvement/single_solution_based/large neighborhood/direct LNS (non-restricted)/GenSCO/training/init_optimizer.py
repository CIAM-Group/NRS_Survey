import optax
from .optimizers import muon
from typing import Literal


def init_optimizer(
    num_steps: int, num_warmup_steps: int,
    peak_lr: float, init_lr: float = 1e-6, end_lr: float = 1e-6,
    weight_decay: float = 1e-4, clip_norm: float | None = None,
    *,
    optimizer_type: Literal['adamw', 'muon', 'muon_decay'],
):
    lr_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=init_lr,
        peak_value=peak_lr,
        end_value=end_lr,
        warmup_steps=num_warmup_steps,
        decay_steps=num_steps, 
    )                                                
    tx: optax.GradientTransformation
    if optimizer_type == 'adamw':
        tx = optax.adamw(lr_scheduler, weight_decay=weight_decay, b2=0.95)
    elif optimizer_type == 'muon':
        tx = muon(lr_scheduler, adamw_param_keys=[])
    elif optimizer_type == 'muon_decay':
        tx = muon(lr_scheduler, weight_decay=weight_decay, adamw_weight_decay=weight_decay, adamw_param_keys=[])
    else:
        raise ValueError()
    if clip_norm is not None:
        tx = optax.chain(
            optax.clip_by_global_norm(clip_norm),
            tx,
        )
    return tx

