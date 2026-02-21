import jax
import jax.numpy as jnp
from flax import nnx
from flax.jax_utils import replicate
from models import MISModelConfig, MISModel
from training.TrainConfig import TrainConfig
from training import Logger, load_ckpt, save_ckpt
from data import MISDataloader
from helpers.mis_transform import edges2adj
import optax
import argparse
import numpy as np
from functools import partial


def train_mis(
    dataloader: MISDataloader, model_config: MISModelConfig, train_config: TrainConfig,
    model: MISModel, opt_state: optax.OptState | None, 
    save_interval: int, logdir: str, savedir: str,
    step: int = 0, *, allow_zero_to_one: bool = False,
):
    tx = train_config.init_optimizer()
    graphdef, params = nnx.split(model)
    if opt_state is None:
        opt_state = tx.init(params)
    num_nodes = train_config.num_nodes
    logger = Logger(logdir, step=step)

    num_devices = jax.device_count()
    assert train_config.batch_size % num_devices == 0
    is_replicated = num_devices > 1
    batch_size_per_device = train_config.batch_size // num_devices
    if is_replicated:
        params = replicate(params)
        opt_state = replicate(opt_state)

    num_nodes_padded = dataloader.num_nodes_padded
    target_disruption = train_config.target_disruption
    
    @partial(jax.jit, donate_argnums=[1, 2])
    def train_step(
        graphdef: nnx.GraphDef[MISModel], params: nnx.State, opt_state: optax.OptState,
        num_nodes: jax.Array, edges: jax.Array, labels: jax.Array,
        key: jax.Array,
    ):
        key, subkey = jax.random.split(key)
        timestep = jax.random.uniform(subkey, shape=[batch_size_per_device], dtype=jnp.float32)
        key, subkey = jax.random.split(key)
        one_to_zero_rate = jax.random.uniform(
            subkey, shape=[batch_size_per_device],
            dtype=jnp.float32, minval=0.25, maxval=0.75,
        )
        key, subkey = jax.random.split(key)
        whether_flip = jax.random.bernoulli(
            subkey, one_to_zero_rate[..., None],
            shape=[batch_size_per_device, num_nodes_padded],
        )
        if not allow_zero_to_one:
            src_sols = jnp.where(
                labels,
                jnp.logical_xor(labels, whether_flip),
                False,
            )
        else:
            src_sols = jnp.logical_xor(labels, whether_flip)
        if target_disruption == 0.:
            current = (1 - timestep[:, None]) * src_sols.astype(jnp.float32) + timestep[:, None] * labels.astype(jnp.float32) 
        else:
            key, subkey = jax.random.split(key)
            if_flip = jax.random.bernoulli(
                subkey, 
                target_disruption, 
                shape=[batch_size_per_device, num_nodes_padded],
            )
            noisy_labels = jnp.logical_xor(labels, if_flip)
            current = (1 - timestep[:, None]) * src_sols.astype(jnp.float32) + timestep[:, None] * noisy_labels.astype(jnp.float32)
        current = jnp.where(
            jnp.arange(num_nodes_padded).reshape(1, -1) < num_nodes.reshape(-1, 1),
            current,
            0,
        ) 
        adjmat = edges2adj(edges, num_nodes_padded=num_nodes_padded, dtype=jnp.float16)
        def loss_fn(params: nnx.State):
            model = nnx.merge(graphdef, params)
            features = model.encode(adjmat, key)
            sigmoid_logits, softmax_logits = model.decode(features, current, timestep, adjmat, num_nodes)
            if sigmoid_logits is not None:
                sigmoid_loss = optax.sigmoid_binary_cross_entropy(sigmoid_logits, labels).mean()
            else:
                sigmoid_loss = 0.
            if softmax_logits is not None:
                softmax_loss = optax.softmax_cross_entropy(softmax_logits, labels / labels.sum(axis=-1, keepdims=True)).mean()
            else:
                softmax_loss = 0.
            loss = sigmoid_loss + softmax_loss
            return loss, (sigmoid_loss, softmax_loss)
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (sigmoid_loss, softmax_loss)), grads = grad_fn(params)
        if is_replicated:
            grads = jax.lax.pmean(grads, axis_name='data')
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # finite = jnp.array(True)
        # for g in jax.tree_util.tree_leaves(new_params):
        #     finite &= jnp.all(jax.lax.is_finite(g))
        # for g in jax.tree_util.tree_leaves(new_opt_state):
        #     if jnp.issubdtype(g, jnp.floating):
        #         finite &= jnp.all(jax.lax.is_finite(g))
        # new_params, new_opt_state = jax.tree.map(
        #     lambda x_updated, x: jnp.where(finite, x_updated, x),
        #     (new_params, new_opt_state),
        #     (params, opt_state),
        # )

        return (loss, sigmoid_loss, softmax_loss), new_params, new_opt_state, key
    
    if not is_replicated:
        train_step = jax.jit(train_step, donate_argnums=[1, 2])
    else:
        train_step = jax.pmap(train_step, donate_argnums=[1, 2], axis_name='data')
    
    while True:
        key = jax.random.wrap_key_data(np.random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max, size=[2], dtype=np.uint32))
        if num_devices > 1:
            key = jax.random.split(key, num_devices)
        for num_nodes, edges, labels in dataloader:
            step += 1

            if is_replicated:
                num_nodes, edges, labels = tuple(map(
                    lambda x: x.reshape((num_devices, batch_size_per_device) + x.shape[1:]),
                    (num_nodes, edges, labels),
                ))
            
            (loss, sigmoid_loss, softmax_loss), params, opt_state, key = train_step(graphdef, params, opt_state, num_nodes, edges, labels, key)
            logger(**{'loss/loss': loss, 'loss/sigmoid': sigmoid_loss, 'loss/softmax': softmax_loss})

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
    parser.add_argument('--label', type=str, required=True)

    parser.add_argument('--ignore_ckpt_train_config', action='store_true', default=False)
    parser.add_argument('--target_disruption', type=float, default=0.)

    parser.add_argument('--fake_data', action='store_true', default=False)

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
        model_config = MISModelConfig.get_config(args.model_config)
    model = model_config.construct_model()
    if params is not None:
        graphdef = nnx.graphdef(model)
        model = nnx.merge(graphdef, params)
    if not args.fake_data:
        data = dict(np.load(args.data))
        label = dict(np.load(args.label))
        data.update(**label)
    else:
        num_nodes = num_nodes_padded = 77
        num_instances = 4096
        edges = np.random.randint(0, num_nodes, size=[num_instances, num_nodes ** 2 // 10, 2], dtype=np.int16)
        labels = np.random.randint(0, 2, size=[num_instances, num_nodes], dtype=np.int8)
        data = {'edges': edges, 'labels': labels, 'num_nodes': np.full(num_instances, num_nodes, dtype=np.int32)}
    dataloader = MISDataloader(
        data,
        batch_size=train_config.batch_size,
        seed=None,
    )
    
    train_mis(
        dataloader, model_config, train_config, model, opt_state, 
        save_interval=args.save_interval,
        logdir=args.logdir, savedir=args.savedir,
        step=step,
    )

