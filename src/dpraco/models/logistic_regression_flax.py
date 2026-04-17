import optax
from flax import linen as nn
from flax.training.train_state import TrainState

from jax import numpy as jnp


class LogisticRegression(nn.Module):
    """A simple Logistic Regession model."""

    num_outputs: int = 2

    @nn.compact
    def __call__(self, x, dropout_rng=None, train=False):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.num_outputs)(x)
        return x


def create_train_state(cfg, rng):
    """Creates initial `TrainState`."""
    num_features = cfg.dataset.num_features
    model = LogisticRegression(num_outputs=cfg.dataset.num_classes)
    params = model.init(rng, jnp.ones([1, num_features]), {})["params"]
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.sgd(cfg.training_params.lr, cfg.training_params.momentum),
    )
