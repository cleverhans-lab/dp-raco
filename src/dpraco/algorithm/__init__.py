from typing import Callable
from omegaconf import DictConfig
from dpraco.algorithm.dpraco import dpraco_train_step
from .dp_sgd import dpsgd_update_step
from .update_lambdas import init_lambdas
from .sgd import sgd_update_step
from jax import numpy as jnp


def get_update_rule(cfg: DictConfig) -> Callable:
    if cfg.algorithm.name == "dp_sgd":
        return dpsgd_update_step
    elif cfg.algorithm.name == "dpraco":
        return dpraco_train_step
    elif cfg.algorithm.name == "sgd":
        return sgd_update_step
        raise NotImplementedError(
            f"Update rule for {cfg.algorithm.name} not implemented."
        )


def get_algorithm_artifacts(cfg: DictConfig, key):
    if cfg.algorithm.name == "dp_sgd" or cfg.algorithm.name == "sgd":
        return {"c": None, "c_hat": None, "constraint_value": []}
    elif cfg.algorithm.name == "dpraco":
        lambdas = init_lambdas(cfg, key)
        momentum_lambdas = jnp.zeros_like(lambdas)

        if cfg.algorithm.constraint_type in ["DemographicParity", "EqualOpportunity"]:
            c_hat, c = 2 * [jnp.zeros_like(lambdas)]
            additional_context = {
                "N_z": jnp.zeros((cfg.dataset.num_fairness_classes,)),
                "N_neq_z": jnp.zeros((cfg.dataset.num_fairness_classes,)),
                "constraint_value": []
            }
        elif cfg.algorithm.constraint_type == "EqualizedOdds":
            c_hat, c = 2 * [
                jnp.zeros((cfg.dataset.num_classes, cfg.dataset.num_classes, cfg.dataset.num_fairness_classes))
            ]
            additional_context = {
                "N_yz": jnp.zeros((cfg.dataset.num_fairness_classes,)),
                "N_neq_yz": jnp.zeros((cfg.dataset.num_fairness_classes,)),
            }
        elif cfg.algorithm.constraint_type == "BalancedAccuracy":
            c_hat, c = 2 * [jnp.zeros_like(lambdas)]
            additional_context = {
                "N_z": jnp.zeros((cfg.dataset.num_fairness_classes,)),
                "N_neq_z": jnp.zeros((cfg.dataset.num_fairness_classes,)),
                "constraint_value": []
            }
        elif cfg.algorithm.constraint_type in [
            "FalseNegativeRate",
            "FalseDiscoveryRate",
        ]:
            c_hat, c = 2 * [
                jnp.zeros((cfg.dataset.num_classes, cfg.dataset.num_classes))
            ]
            additional_context = {}

        return {
            "c": c,
            "c_hat": c_hat,
            "lambdas": lambdas,
            "momentum_lambdas": momentum_lambdas,
            "constraint_value": [],
            **additional_context,
        }
    else:
        raise NotImplementedError(
            f"Algorithm artifacts for {cfg.algorithm.name} not implemented."
        )
