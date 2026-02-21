import pickle
import os
from flax.serialization import to_state_dict, from_state_dict
from flax.jax_utils import unreplicate
from .TrainConfig import TrainConfig

# ! 改为inference only版
# def load_ckpt(ckpt_path: str | None):
#     if ckpt_path is None:
#         return None, None, None, None, None, 0
#     with open(ckpt_path, 'rb') as f:
#         ckpt = pickle.load(f)
#     params = ckpt['params']
#     opt_state = ckpt['opt_state']
#     np_rd_state = ckpt['np_rd_state']
#     model_config = ckpt['model_config']
#     train_config: TrainConfig = ckpt['train_config']
#     try:
#         opt_state_target = train_config.init_opimizer().init(params)
#         opt_state = from_state_dict(opt_state_target, opt_state)
#         del opt_state_target
#     except:
#         opt_state = None
#     step = ckpt['step']
#     return params, opt_state, np_rd_state, model_config, train_config, step

def load_ckpt(ckpt_path):
    import pickle
    import jax.numpy as jnp

    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    # 必须包含 params 和 model_config
    params = ckpt["params"]
    model_config = ckpt["model_config"]

    # inference 不需要 opt_state 或 train_config
    # 结构保持一致返回 None
    return params, None, None, model_config, None, None



def save_ckpt(
    params, opt_state, np_rd_state, 
    model_config, train_config, step, 
    savedir: str | None = None, savepath: str | None = None, 
    *, is_replicated: bool = False,
):
    assert savedir is None or savepath is None
    assert savedir is not None or savepath is not None
    if savepath is None:
        savepath = os.path.join(savedir, f'step{step}.ckpt')
    if savedir is not None:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
    if is_replicated:
        params = unreplicate(params)
        opt_state = unreplicate(opt_state)
    with open(savepath, 'wb') as f:
        pickle.dump(
            {
                'params': params,
                'opt_state': to_state_dict(opt_state),
                'np_rd_state': np_rd_state,
                'model_config': model_config,
                'train_config': train_config,
                'step': step,
            }, f
        )
