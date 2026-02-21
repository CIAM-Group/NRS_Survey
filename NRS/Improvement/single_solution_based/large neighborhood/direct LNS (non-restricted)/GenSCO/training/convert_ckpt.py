from training.ckpt_utils import load_ckpt, save_ckpt
import argparse
import jax
import jax.numpy as jnp


def load_convert_save(
    load_path: str,
    save_path: str,
    dtype: jax.typing.DTypeLike | None,
    delete_train_info: bool,
):
    assert load_path is not None
    params, opt_state, np_rd_state, model_config, train_config, step = load_ckpt(load_path)
    if delete_train_info:
        np_rd_state, train_config, opt_state = [None] * 3
        step = 0
    if dtype is not None:
        params = jax.tree.map(
            lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            params,
        )
    save_ckpt(
        params, opt_state, np_rd_state, model_config, train_config, step,
        savepath=save_path,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--dtype', type=str, default=None)
    parser.add_argument('--delete_train_info', action='store_true', default=False)

    load_convert_save(**vars(parser.parse_args()))
