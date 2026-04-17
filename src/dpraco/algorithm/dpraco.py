from typing import Tuple
from typing import Callable
import jax
from flax.core import FrozenDict
from flax.struct import PyTreeNode
from flax.training.train_state import TrainState
from jax import value_and_grad, vmap, random
from jax.tree import flatten, unflatten
from jax import numpy as jnp
from jax._src.random import PRNGKey
from omegaconf import DictConfig
from functools import partial

from dpraco.algorithm.compute_denominator import compute_denominator
from dpraco.algorithm.hist_estimation import get_histogram
from dpraco.algorithm.update_lambdas import update_lambdas
from dpraco.utils.jax_utils import jit_except_first

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
    noise_rng, loss_rng = random.split(rng, 2)
    batch_noise_rng = random.split(noise_rng, batch[0].shape[0])

    clipped_grads, (loss_value, lagrangian, train_loss) = vmap(
        clipped_grad, (None, None, 0, None, 0)
    )(train_metadata, params, batch, artifacts, batch_noise_rng)

    clipped_grads = jax.tree.map(
        lambda leaf: zero_out_after_num_samples(leaf, number_of_samples),
        clipped_grads
    )
    cfg, _, _ = train_metadata
    summed_grads = jax.tree.map(
        lambda x: x.sum(0) / cfg.training_params.batch_size, clipped_grads
    )

    with jax.named_scope("sync_gradients"):
        agg_grads = jax.lax.psum(summed_grads, axis_name="batch")

    train_metadata = dict(
        loss_values=loss_value, regularizer_values=lagrangian, train_loss=train_loss
    )

    num_leaves = len(jax.tree.leaves(agg_grads))  # Get the number of leaves
    keys = jax.random.split(rng, num_leaves)  # Split the key into num_leaves subkeys
    rng_tree = jax.tree.unflatten(
        jax.tree.structure(agg_grads), keys
    )  # Rebuild the tree structure with the keys
    noisy_grads = jax.tree.map(
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
def _dpraco_update_step(
    train_metadata,
    rng: jax.random.PRNGKey,
    state: TrainState,
    batch: Sample,
    artifacts: dict,
    number_of_samples
) -> Tuple[TrainState, dict]:
    priv_grad, train_metadata = private_grad(
        train_metadata=train_metadata,
        params=state.params,
        batch=batch,
        rng=rng,
        artifacts=artifacts,
        number_of_samples=number_of_samples
    )
    state = update_model(state, priv_grad)
    return state, train_metadata


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)

@partial(jax.jit, static_argnums=(0,))
def dpraco_train_step(metadata, state, batch, rng, i, algorithm_artifacts, number_of_samples):
    # if i % k == 0:  this is taken care of in the get_fairness_hist function
    cfg = metadata[0]
    hist_eval, dpsgd_rng, denominator_rng = jax.random.split(rng, 3)
    (c_hat_soft, c_soft, c_hat_hard, c_hard) = get_histogram(
        train_metadata=metadata,
        params=state.params,
        batch=batch,
        rng=hist_eval,
        artifacts=algorithm_artifacts,
    )
    if cfg.algorithm.lambda_update_type == "soft":
        algorithm_artifacts["c"] = c_soft
        algorithm_artifacts["c_hat"] = c_hat_soft
    else:
        algorithm_artifacts["c"] = c_hard
        algorithm_artifacts["c_hat"] = c_hat_hard

    algorithm_artifacts = compute_denominator(
        cfg,
        algorithm_artifacts,
        batch=batch,
        c=algorithm_artifacts["c_hat"],
        rng=denominator_rng,
    )

    # DP-SGD will ignore the regularizer term (which would be NaN)
    state, train_metadata = _dpraco_update_step(
        train_metadata=metadata,
        state=state,
        batch=batch,
        rng=dpsgd_rng,
        artifacts=algorithm_artifacts,
        number_of_samples=number_of_samples
    )

    # update lambdas
    (lambdas, momentum_lambdas) = update_lambdas(
        cfg=cfg,
        lambdas=algorithm_artifacts["lambdas"],
        c_hat=algorithm_artifacts["c"]
        if cfg.algorithm.use_non_private_histogram
        else algorithm_artifacts["c_hat"],
        artifacts=algorithm_artifacts,
        v_lambdas=algorithm_artifacts["momentum_lambdas"],
    )

    algorithm_artifacts["lambdas"] = lambdas
    algorithm_artifacts["momentum_lambdas"] = momentum_lambdas

    train_metadata["c_hat_soft"] = c_hat_soft
    train_metadata["c_hat_hard"] = c_hat_hard
    train_metadata["c_soft"] = c_soft
    train_metadata["c_hard"] = c_hard
    return state, train_metadata, algorithm_artifacts
