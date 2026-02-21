import jax
import jax.numpy as jnp
from flax import nnx
from flax.jax_utils import replicate
from models import MCLModelConfig, MCLModel
from training.TrainConfig import TrainConfig
from training import load_ckpt
from data import MCLDataloader
import argparse
import numpy as np


from .train_mis import train_mis as train_mcl



if __name__ == '__main__':
    np.random.seed(42)

    from helpers import with_invalid_kwargs_filtered
    parser = argparse.ArgumentParser()
    # train config
    parser.add_argument('--num_steps', type=int, default=10 ** 6)
    parser.add_argument('--num_warmup_steps', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--peak_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--clip_norm', type=float, default=None)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--optimizer_type', type=str, default='adamw')

    parser.add_argument('--save_interval', type=int, default=10 ** 10)  # default value leads to never save
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--savedir', type=str, default=None)
    
    parser.add_argument('--model_config', type=str, default='')

    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--data', type=str, required=True)
    # parser.add_argument('--label', type=str, required=True)

    parser.add_argument('--ignore_ckpt_train_config', action='store_true', default=False)
    parser.add_argument('--target_disruption', type=float, default=0.)

    parser.add_argument('--fake_data', action='store_true', default=False)
    
    parser.add_argument('--allow_zero_to_one', action='store_true', default=False)

    args = parser.parse_args()

    a = jnp.ones([])
    a = replicate(a)
    del a

    params, opt_state, np_rd_state, model_config, train_config, step = load_ckpt(args.ckpt)
    if args.ignore_ckpt_train_config:
        opt_state, np_rd_state, train_config, step = None, None, None, 0
    if np_rd_state is not None:
        np.random.set_state(np_rd_state)
    if train_config is None:
        train_config = with_invalid_kwargs_filtered(TrainConfig)(**vars(args))
    if model_config is None:
        model_config = MCLModelConfig.get_config(args.model_config)
    model = model_config.construct_model()
    if params is not None:
        graphdef = nnx.graphdef(model)
        model = nnx.merge(graphdef, params)
    if not args.fake_data:
        data = dict(np.load(args.data))
        # label = dict(np.load(args.label))
        # data.update(**label)
    else:
        num_nodes = num_nodes_padded = 77
        num_instances = 4096
        edges = np.random.randint(0, num_nodes, size=[num_instances, num_nodes ** 2 // 10, 2], dtype=np.int16)
        labels = np.random.randint(0, 2, size=[num_instances, num_nodes], dtype=np.int8)
        data = {'edges': edges, 'labels': labels, 'num_nodes': np.full(num_instances, num_nodes, dtype=np.int32)}
    dataloader = MCLDataloader(
        data,
        batch_size=train_config.batch_size,
        seed=None,
    )
    
    train_mcl(
        dataloader, model_config, train_config, model, opt_state, 
        save_interval=args.save_interval,
        logdir=args.logdir, savedir=args.savedir,
        step=step,
        allow_zero_to_one=args.allow_zero_to_one,
    )

