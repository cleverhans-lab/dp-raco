from typing import Tuple
from omegaconf import DictConfig
from dpraco.utils.jax_utils import jit_except_first
from dpraco.utils.constraints import constraint_value
from jax import numpy as jnp
import jax

EPS = 1e-6


@jit_except_first
def init_lambdas(cfg: DictConfig, key: jax.random.PRNGKey) -> jnp.array:
    if cfg.algorithm.dual_update_type == "gradient_ascent_with_momentum":
        init_function = jnp.zeros

    if cfg.algorithm.constraint_type in ["DemographicParity"]:
        return init_function(
            (cfg.dataset.num_classes, cfg.dataset.num_fairness_classes)
        )
    elif cfg.algorithm.constraint_type == "EqualizedOdds":
        return init_function(
            (cfg.dataset.num_classes, cfg.dataset.num_classes, cfg.dataset.num_fairness_classes)
        )
    elif cfg.algorithm.constraint_type in ["FalseNegativeRate", "FalseDiscoveryRate"]:
        # return init_function(1) # make it work only for binary classification for now
        return init_function(cfg.dataset.num_classes)
    else:
        msg = f"Unknown constraint type: {cfg.algorithm.constraint_type}"
        raise ValueError(msg)


@jit_except_first
def update_lambdas(
    cfg: DictConfig, lambdas: jnp.array, v_lambdas: jnp.array, c_hat: jnp.array, artifacts: dict
) -> Tuple[jnp.array, jnp.array]:
    train_constraint = constraint_value(cfg=cfg, counts=c_hat, artifacts=artifacts)
    grad = train_constraint - cfg.algorithm.gamma
    return update_lambdas_with_grad(cfg, lambdas, v_lambdas, grad)


def update_lambdas_with_grad(
    cfg: DictConfig, lambdas: jnp.array, v_lambdas: jnp.array, grad: jnp.array
) -> jnp.array:
    if cfg.algorithm.dual_update_type == "gradient_ascent_with_momentum":
        # Update momentum
        momentum = cfg.training_params.momentum
        v_lambdas = momentum * v_lambdas + cfg.training_params.lr_lambda * grad
        # Update lambdas with the momentum term
        lambdas = lambdas + v_lambdas
        # Ensure lambdas are non-negative
        lambdas = jnp.maximum(0, lambdas)
    elif cfg.algorithm.dual_update_type == "exponentiated_gradient_ascent":
        # no update to v_lambdas (no momentum)
        # lambdas = (lambdas * jnp.exp(cfg.training_params.lr_lambda * grad))/jnp.sum(lambdas * grad)
        lambdas = lambdas * jnp.exp(cfg.training_params.lr_lambda * grad)
        # Ensure lambdas are non-negative
        lambdas = jnp.maximum(EPS, lambdas)
    else:
        msg = f"Unknown dual update type: {cfg.algorithm.dual_update_type}"
        raise ValueError(msg)
    return lambdas, v_lambdas
