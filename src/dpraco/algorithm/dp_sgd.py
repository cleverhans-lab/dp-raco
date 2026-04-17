
from typing import Tuple
from typing import Callable
import jax
from flax.core import FrozenDict
from flax.struct import PyTreeNode
from jax import value_and_grad, vmap, random
from jax.tree import flatten, unflatten
from jax import numpy as jnp
from jax._src.random import PRNGKey
from omegaconf import DictConfig

from dpraco.utils.jax_utils import jit_except_first
from jax import tree_util

Metadata = Tuple[DictConfig, Callable, Callable]
Sample = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]

def zero_out_after_num_samples(leaf, n):
    # leaf has shape [batch_size, ...]
    batch_size = leaf.shape[0]

    # Create a boolean mask of shape [batch_size], True for indices < n
    mask = jnp.arange(batch_size) < n
    # Reshape mask to [batch_size, 1, 1, ...] so it broadcasts over the rest
    mask = mask.reshape((batch_size,) + (1,) * (leaf.ndim - 1))

    # Zero everything outside the mask
    return jnp.where(mask, leaf, 0)

def clipped_grad(
    train_metadata: Metadata,
    params: FrozenDict,
    single_example_batch: Sample,
    artifacts: dict,
    rng: PRNGKey,
) -> Tuple[PyTreeNode, Tuple[float, float, float]]:
    """Evaluate gradient for a single-example batch and clip its grad norm."""
    cfg, training_loss, _ = train_metadata
    image, label, sensitive = single_example_batch

    (loss_value, (lagrangian_value, train_loss)), loss_grads = value_and_grad(
        training_loss, has_aux=True
    )(params, (image, label, sensitive), artifacts, rng)

    nonempty_loss_grads, tree_def1 = flatten(loss_grads)

    total_grad_norm = jnp.linalg.norm(
        jnp.array([jnp.linalg.norm(neg.ravel()) for neg in nonempty_loss_grads])
    )
    divisor = jnp.maximum(total_grad_norm / cfg.algorithm.C, 1.0)
    normalized_nonempty_grads = [g / divisor for g in nonempty_loss_grads]
    return (
        unflatten(tree_def1, normalized_nonempty_grads),
        (loss_value, lagrangian_value, train_loss),
    )


def private_grad(
    train_metadata: Metadata, params: FrozenDict, batch: Sample, rng: PRNGKey, artifacts, number_of_samples: int
) -> Tuple[PyTreeNode, dict]:
    """Return differentially private gradients for params, evaluated on batch."""
    noise_rng, _ = random.split(rng, 2)
    batch_noise_rng = random.split(noise_rng, batch[1].shape[0])

    clipped_grads, (loss_value, lagrangian, train_loss) = vmap(
        clipped_grad, (None, None, 0, None, 0)
    )(train_metadata, params, batch, artifacts, batch_noise_rng)

    clipped_grads = tree_util.tree_map(
        lambda leaf: zero_out_after_num_samples(leaf, number_of_samples),
        clipped_grads
    )
    cfg, _, _ = train_metadata
    summed_grads = jax.tree_util.tree_map(
        lambda x: x.sum(0) / cfg.training_params.batch_size, clipped_grads
    )

    with jax.named_scope("sync_gradients"):
        agg_grads = jax.lax.psum(summed_grads, axis_name="batch")

    train_metadata = dict(
        loss_values=loss_value, regularizer_values=lagrangian, train_loss=train_loss
    )

    num_leaves = len(jax.tree.leaves(agg_grads))  # Get the number of leaves
    keys = jax.random.split(rng, num_leaves)  # Split the key into num_leaves subkeys
    rng_tree = jax.tree_unflatten(
        jax.tree_structure(agg_grads), keys
    )  # Rebuild the tree structure with the keys
    noisy_grads = jax.tree_map(
        lambda leaf, key: leaf
        + cfg.algorithm.C
        * cfg.algorithm.sigma
        / cfg.training_params.batch_size
        * jax.random.normal(key, shape=leaf.shape),
        agg_grads,
        rng_tree,
    )

    return (noisy_grads, train_metadata)


@jit_except_first
def dpsgd_update_step(train_metadata, state, batch, rng, i, algorithm_artifacts, number_of_samples):
    priv_grad, train_metadata = private_grad(
        train_metadata=train_metadata,
        params=state.params,
        batch=batch,
        rng=rng,
        artifacts=algorithm_artifacts,
        number_of_samples=number_of_samples
    )
    state = update_model(state, priv_grad)
    return state, train_metadata, {}


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)
