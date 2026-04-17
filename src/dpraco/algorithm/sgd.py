from typing import Tuple, Callable
import jax
from jax import value_and_grad
from jax import numpy as jnp
from jax._src.random import PRNGKey

from flax.core import FrozenDict
from flax.struct import PyTreeNode
from flax.training.train_state import TrainState
from omegaconf import DictConfig

from dpraco.utils.jax_utils import jit_except_first

Metadata = Tuple[DictConfig, Callable, Callable]
Sample = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]

def sgd_grad(
    train_metadata: Metadata,
    params: FrozenDict,
    batch: Sample,
    rng: PRNGKey,
    artifacts: dict,
) -> Tuple[PyTreeNode, dict]:
    """
    Returns the average gradient over the entire batch (plain SGD),
    along with a metadata dict containing the loss, etc.
    """
    cfg, training_loss, _ = train_metadata
    images, labels, sensitive = batch

    # Compute gradient and unpack auxiliary outputs
    (loss_value, (lagrangian_value, train_loss)), grads = value_and_grad(
        training_loss, has_aux=True
    )(params, (images, labels, sensitive), artifacts, rng)

    batch_size = cfg.training_params.batch_size
    grads = jax.tree_map(lambda g: g / batch_size, grads)
    with jax.named_scope("sync_gradients"):
        grads = jax.lax.psum(grads, axis_name="batch")

    # Mirror the metadata dict from private_grad
    sgd_metadata = dict(
        loss_values=jnp.expand_dims(loss_value, 0),
        regularizer_values=jnp.expand_dims(lagrangian_value, 0),
        train_loss=jnp.expand_dims(train_loss, 0),
    )
    return grads, sgd_metadata

@jit_except_first
def sgd_update_step(
    train_metadata: Metadata,
    state: TrainState,
    batch: Sample,
    rng: PRNGKey,
    i: int,
    algorithm_artifacts: dict,
    number_of_samples: int,
):
    """
    A plain-SGD update step analogous to dpsgd_update_step.
    Has the same signature for easy drop-in comparison.
    """
    grads, sgd_metadata = sgd_grad(
        train_metadata=train_metadata,
        params=state.params,
        batch=batch,
        rng=rng,
        artifacts=algorithm_artifacts,
    )
    new_state = update_model(state, grads)
    return new_state, sgd_metadata, {}

@jax.jit
def update_model(state: TrainState, grads: PyTreeNode) -> TrainState:
    # Same final model update as in your DP code
    return state.apply_gradients(grads=grads)
