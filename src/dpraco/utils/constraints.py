import enum
from dpraco.utils.jax_utils import jit_except_first
from jax import nn
import jax.numpy as jnp


class DecisionType(enum.Enum):
    SOFT = "soft"
    HARD = "hard"


def get_one_hot_preds(logits, num_classes, decision_type=DecisionType.HARD):
    if logits.shape[-1] == 1:
        _val = (
            nn.softmax(logits)
            if decision_type == DecisionType.SOFT
            else jnp.argmax(logits)
        )
        preds = jnp.c_[_val, 1 - _val]
    else:
        reshaped_logits = logits.reshape((-1, num_classes))
        preds = (
            nn.softmax(reshaped_logits, axis=1)
            if decision_type == DecisionType.SOFT
            else nn.one_hot(jnp.argmax(reshaped_logits, axis=-1), num_classes)
        )
    return preds


@jit_except_first
def constraint_value(cfg, counts, artifacts=None):
    if cfg.algorithm.constraint_type == "DemographicParity":
        return demographic_parity_loss(counts, artifacts)
    elif cfg.algorithm.constraint_type == "EqualizedOdds":
        return equalized_odds_loss(counts, artifacts)
    elif cfg.algorithm.constraint_type == "FalseNegativeRate":
        return false_negative_rate(counts).ravel()
    else:
        msg = f"Unknown constraint type: {cfg.algorithm.constraint_type}"
        raise msg


def recall(counts):
    MASK = jnp.array([0, 1])
    true_positives = jnp.diag(counts)
    false_negatives = (counts - jnp.diag(jnp.diag(counts))).sum(axis=0)
    recall = true_positives / (true_positives + false_negatives + 1e-8)  # Add small epsilon to avoid division by zero
    recall = jnp.sum(recall * MASK)
    recall = jnp.clip(recall, 0, 1)
    return recall


def false_negative_rate(counts):
    return 1 - recall(counts)


def demographic_parity_loss(counts, artifacts=None):
    # counts is a matrix of shape (num_labels, num_sensitive_classes)
    N_z_array = artifacts["N_z"] if artifacts else jnp.sum(counts, axis=0)  # Shape: (num_sensitive_classes,)
    N_neq_z_array = artifacts["N_neq_z"] if artifacts else jnp.sum(counts) - N_z_array  # Shape: (num_sensitive_classes,)
    c_k = jnp.sum(counts, axis=1)  # Shape: (num_labels,)

    c_neqz_k = c_k[:, None] - counts  # Shape: (num_labels, num_sensitive_classes)
    # Reshape N_z_array and N_neq_z for broadcasting
    N_z = jnp.where(N_z_array[None, :] == 0, 1, N_z_array[None, :])  # Shape: (1, num_sensitive_classes)
    N_neq_z = jnp.where(N_neq_z_array[None, :] == 0, 1, N_neq_z_array[None, :])  # Shape: (1, num_sensitive_classes)

    IN = (counts / N_z)
    OUT = (c_neqz_k / N_neq_z)

    # Compute demographic parity differences
    demographic_parity_zk = IN - OUT  # Shape: (num_labels, num_sensitive_classes)

    # truncate each element to be in [0, 1]
    demographic_parity_zk = jnp.clip(demographic_parity_zk, 0, 1)
    return demographic_parity_zk


def equalized_odds_loss(counts, artifacts=None):
    """
    counts is: (k,y,z). Validated. See notebooks (test_equalized_odds.ipynb) for more details.
    """
    if artifacts is not None and "N_yz" in artifacts and "N_neq_yz" in artifacts:
        N_yz = artifacts["N_yz"]
        N_neq_yz = artifacts["N_neq_yz"]
    else:
        N_yz = jnp.sum(counts, axis=0)
        N_neq_yz = N_yz.sum(axis=1, keepdims=True) - N_yz

    c_kyz = counts
    c_ky_notz = c_kyz.sum(axis=2, keepdims=True) - c_kyz

    IN = c_kyz / N_yz  # Shape: (k, y, z)
    OUT = c_ky_notz / N_neq_yz

    eq_odds_kyz = IN - OUT

    # truncate each element to be in [0, 1]
    eq_odds_kyz = jnp.clip(eq_odds_kyz, 0, 1)

    return eq_odds_kyz
