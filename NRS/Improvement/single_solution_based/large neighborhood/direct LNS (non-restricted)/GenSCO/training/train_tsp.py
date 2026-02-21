import jax
import jax.numpy as jnp
from flax import nnx
from flax.jax_utils import replicate
from models import TSPModelConfig, TSPModel
from modules.functional import coord_normalize
from training.TrainConfig import TrainConfig
from training import Logger, load_ckpt, save_ckpt
from data import TSPDataloader
from helpers import sol2adj
import optax
import argparse
import numpy as np
from functools import partial


def train_tsp(
    dataloader: TSPDataloader, model_config: TSPModelConfig, train_config: TrainConfig,
    model: TSPModel, opt_state: optax.OptState | None, 
    save_interval: int, logdir: str, savedir: str,
    step: int = 0,
):
    tx = train_config.init_optimizer()
    graphdef, params = nnx.split(model)
    if opt_state is None:
        opt_state = tx.init(params)
    num_nodes = train_config.num_nodes
    target_disruption = train_config.target_disruption
    logger = Logger(logdir, step=step)

    num_devices = jax.device_count()
    assert train_config.batch_size % num_devices == 0
    is_replicated = num_devices > 1
    batch_size_per_device = train_config.batch_size // num_devices
    if is_replicated:
        params = replicate(params)
        opt_state = replicate(opt_state)
    
    @partial(jax.jit, donate_argnums=[1, 2])
    def train_step(
        graphdef: nnx.GraphDef[TSPModel], params: nnx.State, opt_state: optax.OptState,
        raw_features: jax.Array, current: jax.Array, target: jax.Array, timestep: jax.Array,
    ):
        raw_features = coord_normalize(raw_features)
        if target_disruption is not None:
            target, noisy_target = target[..., :num_nodes], target[..., num_nodes:]
        cur_adjmat = sol2adj(current)
        tgt_adjmat = sol2adj(target)
        if target_disruption is not None:
            noisy_tgt_adjmat = sol2adj(noisy_target)
            cur_adjmat = (1 - timestep)[:, None, None] * cur_adjmat + timestep[:, None, None] * noisy_tgt_adjmat
        else:
            cur_adjmat = (1 - timestep)[:, None, None] * cur_adjmat + timestep[:, None, None] * tgt_adjmat
        def loss_fn(params: nnx.State):
            model = nnx.merge(graphdef, params)
            features = model.encode(raw_features)
            logits = model.decode(features, timestep, cur_adjmat)
            logits = jax.nn.log_softmax(logits)
            return - (logits * tgt_adjmat).mean() * (num_nodes / 2)
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(params)
        if is_replicated:
            grads = jax.lax.pmean(grads, axis_name='data')
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state
    
    if not is_replicated:
        train_step = jax.jit(train_step, donate_argnums=[1, 2])
    else:
        train_step = jax.pmap(train_step, donate_argnums=[1, 2], axis_name='data')
    
    while True:
        for coords, current, target, timestep in dataloader:
            step += 1

            if is_replicated:
                coords, current, target, timestep = tuple(map(
                    lambda x: x.reshape((num_devices, batch_size_per_device) + x.shape[1:]),
                    (coords, current, target, timestep),
                ))
            
            loss, params, opt_state = train_step(graphdef, params, opt_state, coords, current, target, timestep)
            logger(**{'loss/loss': loss})

            if step % save_interval == 0:
                save_ckpt(
                    params, opt_state, np.random.get_state(),
                    model_config, train_config, step, savedir,
                    is_replicated=is_replicated,
                )

        if step >= train_config.num_steps:
            return



if __name__ == '__main__':
    np.random.seed(42)

    from helpers import with_invalid_kwargs_filtered, maybe_eval
    parser = argparse.ArgumentParser()
    # train config
    parser.add_argument('--num_nodes', type=int, required=True)
    parser.add_argument('--num_steps', type=int, default=10 ** 6)
    parser.add_argument('--num_warmup_steps', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--peak_lr', type=float, default=1e-3)
    parser.add_argument('--end_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--clip_norm', type=float, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise_type', type=str, default='randperm')
    parser.add_argument('--target_disruption', type=partial(maybe_eval, should_keep=['default']), default='default')

    parser.add_argument('--optimizer_type', type=str, default='adamw')

    parser.add_argument('--save_interval', type=int, default=10 ** 10)  # default value leads to never save
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--savedir', type=str, default=None)
    
    parser.add_argument('--model_config', type=str, default='')

    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--data', type=str, required=True)

    parser.add_argument('--ignore_ckpt_train_config', action='store_true', default=False)

    args = parser.parse_args()

    
    
    params, opt_state, np_rd_state, model_config, train_config, step = load_ckpt(args.ckpt)
    if args.ignore_ckpt_train_config:
        opt_state, np_rd_state, train_config, step = None, None, None, 0
    if np_rd_state is not None:
        np.random.set_state(np_rd_state)
    if train_config is None:
        train_config = with_invalid_kwargs_filtered(TrainConfig)(**vars(args))
    if model_config is None:
        model_config = TSPModelConfig.get_config(args.model_config)
    model = model_config.construct_model()
    if params is not None:
        graphdef = nnx.graphdef(model)
        model = nnx.merge(graphdef, params)
    dataloader = TSPDataloader(
        np.load(args.data),
        batch_size=train_config.batch_size,
        noise_type=train_config.noise_type,
        target_disruption=train_config.target_disruption,
        data_argument=3,
        num_workers=32,
    )
    
    train_tsp(
        dataloader, model_config, train_config, model, opt_state, 
        save_interval=args.save_interval,
        logdir=args.logdir, savedir=args.savedir,
        step=step,
    )

