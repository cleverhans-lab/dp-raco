from omegaconf import DictConfig
from functools import partial
import jax


def get_model(cfg: DictConfig, rng: jax.random.PRNGKey):
    if cfg.model.name == "logistic_regression":
        from dpraco.models.logistic_regression_flax import (
            create_train_state as create_train_state_lr,
        )
        init_fn = partial(create_train_state_lr, cfg, rng)
    elif cfg.model.name == "cnn":
        from dpraco.models.cnn_flax import create_train_state

        init_fn = partial(create_train_state, cfg, rng)
    elif cfg.model.name in ["resnet16", "resnet50"]:
        from dpraco.models.resnet import create_train_state as create_train_state_resnet
        init_fn = partial(create_train_state_resnet, cfg, rng)
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    # device_array = np.array(jax.devices())
    # mesh = Mesh(device_array, "batch")
    # sharded_map = shard_map(init_fn, mesh, in_specs=P(), out_specs=P(), check_rep=True)
    # func = jax.jit(sharded_map)
    return init_fn()
