from typing import Callable, Tuple

import jax
from jax import numpy as jnp
from omegaconf import DictConfig

from dpraco.training.loss import softmax_temperature
from dpraco.utils.jax_utils import jit_except_first

def sensitive_confusion_matrix_equalized_odds(cfg, predictions, labels, sensitive_attribute, rng):
    c = jnp.einsum("bi,bj,bk->ijk", predictions, labels, sensitive_attribute)

    # Sync gradients as before
    with jax.named_scope("sync_gradients"):
        c = jax.lax.psum(c, axis_name="batch")

    laplace_samples = 1 / cfg.algorithm.laplace_parameter_b  * jax.random.laplace(
        rng, (cfg.dataset.num_classes, cfg.dataset.num_fairness_classes)
    )
    c_hat = c + laplace_samples
    return c_hat, c


def sensitive_confusion_matrix_equal_opportunity(cfg, predictions, labels, sensitive_attribute, rng):
    is_positive_true_label = labels[:, 1]

    predictions_masked = predictions * is_positive_true_label[:, None]

    c = (predictions_masked.T @ sensitive_attribute)

    with jax.named_scope("sync_gradients"):
        c = jax.lax.psum(c, axis_name="batch")

    num_fairness_classes = len(c)

    if cfg.algorithm.laplace_parameter_b <= 0:
        laplace_samples = jnp.zeros_like(c)
    else:
        laplace_samples = (1.0 / cfg.algorithm.laplace_parameter_b) * jax.random.laplace(
            rng, shape=(num_fairness_classes)
        )

    c_hat = c + laplace_samples

    return c_hat, c

def sensitive_confusion_matrix(cfg, predictions, sensitive_attribute, rng):
    c = predictions.T @ sensitive_attribute
    # aggregate across all samples
    with jax.named_scope("sync_gradients"):
        c = jax.lax.psum(c, axis_name="batch")

    laplace_samples = 1 / cfg.algorithm.laplace_parameter_b  * jax.random.laplace(
        rng, (cfg.dataset.num_classes, cfg.dataset.num_fairness_classes)
    )
    c_hat = c + laplace_samples
    return c_hat, c

def balanced_accuracy(cfg, predictions, labels, sensitive_attribute, rng):
    other_labels = 1 - labels #(batch_size, labels)
    acc_err = predictions * other_labels #(batch_size, labels)
    acc_correct = predictions * labels
    total_err = jnp.sum(acc_err, axis=1) #(labels,)
    total_correct = jnp.sum(acc_correct, axis=1) #(labels,)
    c_err = total_err.T @ sensitive_attribute #(fairness_sens_attr)
    c_correct = total_correct.T @ sensitive_attribute # (fairness_sens_attr)
    c = jnp.stack([c_err, c_correct]) #(2, fairness_sens_attr)
    with jax.named_scope("sync_gradients"):
        c = jax.lax.psum(c, axis_name="batch")

    laplace_samples = 1 / cfg.algorithm.laplace_parameter_b  * jax.random.laplace(
        rng, (2, cfg.dataset.num_fairness_classes)
    )
    c_hat = c + laplace_samples
    return c_hat, c



def confusion_matrix(cfg, labels, predictions, rng):
    """ Compute the confusion matrix for the given predictions and labels. 
    
    Args:
        labels: The true labels of the samples. Shape: (batch_size, num_classes)
        predictions: The predicted labels of the samples. Shape: (batch_size, num_classes)
        rng: The random number generator.

    Returns:
        c_hat: The noisy confusion matrix Y^ x Y. Shape: (num_classes, num_classes)
        c: The true confusion matrix Y^ x Y. Shape: (num_classes, num_classes)
    """
    c = predictions.T @ labels

    # aggregate across all samples
    with jax.named_scope("sync_gradients"):
        c = jax.lax.psum(c, axis_name="batch")
    laplace_samples = 1 / cfg.algorithm.laplace_parameter_b * jax.random.laplace(
        rng, (cfg.dataset.num_classes,)
    )
    c_hat = c + laplace_samples
    return c_hat, c


@jit_except_first
def get_histogram(
    train_metadata: Tuple[DictConfig, Callable, Callable],
    params: dict,
    batch: jnp.ndarray,
    rng: jax.random.PRNGKey,
    artifacts: dict,
):
    cfg, _, inference_function = train_metadata
    (data, labels, sensitives) = batch  # label and sensitive are one-hot encoded

    # partition the data into groups based on the sensitive attribute
    batch_logits = inference_function(params, data, artifacts).squeeze()
    # apply softmax
    soft_logits = softmax_temperature(batch_logits, cfg.training_params.softmax_temperature)

    if cfg.algorithm.constraint_type == "EqualizedOdds":
        c_hat_soft, c_soft = sensitive_confusion_matrix_equalized_odds(cfg, soft_logits, labels, sensitives, rng)
        argmax_predictions = jnp.argmax(soft_logits, axis=-1)
        predictions = jax.nn.one_hot(argmax_predictions, cfg.dataset.num_classes)
        c_hat_hard, c_hard = sensitive_confusion_matrix_equalized_odds(
            cfg, predictions, labels, sensitives, rng
        )
    elif cfg.algorithm.constraint_type == "DemographicParity":
        c_hat_soft, c_soft = sensitive_confusion_matrix(
            cfg, soft_logits, sensitives, rng
        )
        argmax_predictions = jnp.argmax(soft_logits, axis=-1)
        predictions = jax.nn.one_hot(argmax_predictions, cfg.dataset.num_classes)
        c_hat_hard, c_hard = sensitive_confusion_matrix(
            cfg, predictions, sensitives, rng
        )
    elif cfg.algorithm.constraint_type in ["FalseNegativeRate", "FalseDiscoveryRate"]:
        c_hat_soft, c_soft = confusion_matrix(cfg, labels, soft_logits, rng)
        argmax_predictions = jnp.argmax(soft_logits, axis=-1)
        predictions = jax.nn.one_hot(argmax_predictions, cfg.dataset.num_classes)
        c_hat_hard, c_hard = confusion_matrix(cfg, labels, predictions, rng)
    else:
        raise ValueError(f"Unknown constraint type: {cfg.algorithm.constraint_type}")

    return c_hat_soft, c_soft, c_hat_hard, c_hard
