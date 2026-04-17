from typing import Iterator
from jax import numpy as jnp
from jax._src.random import PRNGKey
from omegaconf import DictConfig
from typing import Tuple

DataStream = Iterator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]


def get_data_stream(
    cfg: DictConfig, rng: PRNGKey, seed
) -> Tuple[DataStream, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    if cfg.dataset.name in ["adult", "credit-card", "parkinsons", "folktables", "heart"]:
        from dpraco.data.tabular import tabular_data_stream
        return tabular_data_stream(cfg, rng, seed)
    elif cfg.dataset.name == "celeba":
        from dpraco.data.celeba import celeba_data_stream
        return celeba_data_stream(cfg, rng)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")
