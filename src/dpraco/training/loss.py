import optax
from flax.training.train_state import TrainState
from omegaconf import DictConfig

import jax
import jax.numpy as jnp

def log_softmax_temperature(logits, temperature):
    scaled_logits = logits * temperature
    stable_logits = scaled_logits - jnp.max(scaled_logits, axis=-1, keepdims=True)
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(stable_logits), axis=-1, keepdims=True))
    log_probs = stable_logits - log_sum_exp  # shape (batch_size, num_classes)
    return log_probs

def softmax_temperature(logits, temperature):
    return jnp.exp(log_softmax_temperature(logits, temperature))

def cross_entropy_with_temperature(logits, labels, temperature=1.0):
    log_probs = log_softmax_temperature(logits, temperature)  # shape (batch_size, num_classes)
    nll = -jnp.sum(labels * log_probs, axis=-1)
    loss = jnp.mean(nll)
    return loss


def focal_loss(
        logits,
        labels,
        temperature: float = 1.0,
        alpha: float = 0.85,
        gamma: float = 2.0,
):
    return jnp.sum(optax.sigmoid_focal_loss(jnp.expand_dims(logits, axis=0), jnp.expand_dims(labels, axis=0), alpha=alpha))

def get_loss(cfg: DictConfig, model: TrainState):
    if cfg.algorithm.name == "dpraco":
        if cfg.algorithm.constraint_type == "DemographicParity":
            return get_crossentropy_regularized_demographic_parity(cfg, model)
        elif cfg.algorithm.constraint_type == "EqualizedOdds":
            return get_crossentropy_regularized_equalized_odds(cfg, model)
        elif cfg.algorithm.constraint_type == "FalseNegativeRate":
            return get_crossentropy_regularized_fnr(cfg, model)
        elif cfg.algorithm.constraint_type == "FalseDiscoveryRate":
            return get_crossentropy_regularized_fdr(cfg, model)
        elif cfg.algorithm.constraint_type == "EqualOpportunity":
            return get_crossentropy_regularized_equal_opportunity(cfg, model)
        elif cfg.algorithm.constraint_type == "BalancedAccuracy":
            return get_balanced_accuracy(cfg, model)
        else:
            msg = f"Unknown constraint type: {cfg.algorithm.constraint_type}"
            raise ValueError(msg)
    elif cfg.algorithm.name == "dp_sgd":
        if cfg.model.name == "bert":
            return get_crossentropy_bert(model)
        return get_crossentropy(model)
    elif cfg.algorithm.name == "sgd":
        if cfg.model.name == "bert":
            return get_crossentropy_nosqueeze_bert(model)
        return get_crossentropy_nosqueeze(model)
    else:
        raise ValueError(f"Unknown algorithm name: {cfg.algorithm.name}")

def get_crossentropy_nosqueeze_bert(model: TrainState):
    def _crossentropy_loss(params: TrainState, batch, artifacts, rng, train=True):
        inputs, targets, _ = batch
        logits = model.apply_fn(
            **inputs,
            params=params,
            dropout_rng=rng,
            train=train,
        ).logits
        loss = optax.softmax_cross_entropy(logits=logits, labels=targets).squeeze()
        return jnp.sum(loss), (0, jnp.sum(loss))

    return _crossentropy_loss

def get_crossentropy_nosqueeze(model: TrainState):
    def _crossentropy_loss(params: TrainState, batch, artifacts, rng, train=True):
        inputs, targets, _ = batch
        logits = model.apply_fn(
            inputs,
            params=params,
            dropout_rng=rng,
            train=train,
        )
        loss = optax.softmax_cross_entropy(logits=logits, labels=targets).squeeze()
        return jnp.sum(loss), (0, jnp.sum(loss))

    return _crossentropy_loss


def get_crossentropy_bert(model: TrainState):
    def _crossentropy_loss(params: TrainState, batch, artifacts, rng, train=True):
        inputs, targets, _ = batch
        expanded_inputs = {}

        for (k, v) in inputs.items():
            expanded_inputs[k] = jnp.expand_dims(v, axis=0)

        logits = model.apply_fn(
            **expanded_inputs,
            params=params,
            dropout_rng=rng,
            train=train,
        ).logits
        loss = optax.softmax_cross_entropy(logits=logits, labels=targets).squeeze()
        return loss, (0, loss)

    return _crossentropy_loss

def get_crossentropy(model: TrainState):
    def _crossentropy_loss(params: TrainState, batch, artifacts, rng, train=True):
        inputs, targets, _ = batch
        logits = model.apply_fn(
            {"params": params},
            jnp.expand_dims(inputs, axis=0),
            dropout_rng=rng,
            train=train,
        )
        loss = optax.softmax_cross_entropy(logits=logits, labels=targets).squeeze()
        return loss, (0, loss)

    return _crossentropy_loss

def get_crossentropy_regularized_fdr(cfg: DictConfig, model: TrainState):
    # Note: This function assumes binary classification and therefore a single constraint
    MASK = jnp.array([0, 1])

    def per_sample_loss_fn(params: TrainState, batch, artifacts, rng, train=True):

        lambdas = artifacts["lambdas"]
        # c_hat is Y^ x Y
        c_hat = (
            artifacts["c"]
            if cfg.algorithm.use_non_private_histogram
            else artifacts["c_hat"]
        )

        inputs, yi, _ = batch

        # apply per-sample augmentation
        if cfg.training_params.num_aug == 1:
            # Forward pass to get logits
            logits = model.apply_fn(
                {"params": params}, jnp.expand_dims(inputs, 0), train=train, dropout_rng=rng
            ).squeeze()  # Shape: (num_classes,)
            # Compute the main training loss
            train_loss = cross_entropy_with_temperature(logits=logits, labels=yi, temperature=cfg.training_params.softmax_temperature)

            # Compute per-class probabilities using softmax
            per_class_probs = jax.nn.softmax(logits)  # Shape: (num_classes,)
        else:
            raise ValueError(f"Invalid number of augmentations: {cfg.training_params.num_aug}")

        N_yhat = c_hat.sum(axis=1)  # Shape: (num_predicted_classes,)
        # Avoid division by zero
        N_yhat = jnp.where(N_yhat == 0, 1.0, N_yhat)

        # Membership soft-indicator
        m_i = (1 - yi) * per_class_probs

        # Compute the per-sample lagrangian terms
        lagrangian_terms = lambdas.T * (m_i / N_yhat)

        # Sum over output classes
        lagrangian = jnp.sum(lagrangian_terms * MASK)

        return (
            train_loss + (cfg.training_params.batch_size * lagrangian),
            (lagrangian, train_loss),
        )

    return per_sample_loss_fn

def get_crossentropy_regularized_fnr(cfg: DictConfig, model: TrainState):
    # Note: This function assumes binary classification and therefore a single constraint
    MASK = jnp.array([0, 1])

    def per_sample_loss_fn(params: TrainState, batch, artifacts, rng, train=True):
        # c_hat is Y^ x Y
        c_hat = (
            artifacts["c"]
            if cfg.algorithm.use_non_private_histogram
            else artifacts["c_hat"]
        )
        lambdas = artifacts["lambdas"]

        inputs, yi, _ = batch  # Unpack the batch
        # Forward pass to get logits
        logits = model.apply_fn(
            {"params": params}, jnp.expand_dims(inputs, 0), train=train, dropout_rng=rng
        ).squeeze()  # Shape: (num_classes,)
        # Compute the main training loss
        train_loss = cross_entropy_with_temperature(logits=logits, labels=yi, temperature=cfg.training_params.softmax_temperature)

        # Compute per-class probabilities using softmax
        per_class_probs = jax.nn.softmax(logits)  # Shape: (num_classes,)

        N_y = jnp.sum(c_hat, axis=0)  # Shape: (num_ground_truth_classes,) 

        # Avoid division by zero
        N_y = jnp.where(N_y == 0, 1.0, N_y)

        # Membership soft-indicator
        m_i = yi * (1 - per_class_probs)  # 1_{y_i = k} soft1_{y_i != k}
        # fnr = false_negative_rate(None, None, c_hat)  # Shape: (num_classes,)

        # Compute the per-sample lagrangian terms
        lagrangian_terms = lambdas.T * (m_i / N_y)  # Shape: (num_classes,)

        # Sum over output classes
        # lagrangian = jnp.sum(lagrangian_terms)  # scalar
        lagrangian = jnp.sum(lagrangian_terms * MASK)  # scalar

        return (
            train_loss + (cfg.training_params.batch_size * lagrangian),
            (lagrangian, train_loss),
        )

    return per_sample_loss_fn
#

def get_crossentropy_regularized_equalized_odds(cfg: DictConfig, model):
    def per_sample_loss_fn(params, batch, artifacts, rng, train=True):
        lambdas = artifacts["lambdas"]
        N_yz_array = artifacts["N_yz"]
        N_neq_yz_array = artifacts["N_neq_yz"]  # same shape

        x, y, z = batch
        x_expanded = jnp.expand_dims(x, axis=0)
        logits = model.apply_fn(
            {"params": params},
            x_expanded,
            train=train,
            dropout_rng=rng
        ).squeeze()

        ce_loss = cross_entropy_with_temperature(
            logits=logits,
            labels=y,
            temperature=cfg.training_params.softmax_temperature
        )

        IN_yz = jnp.outer(y, z)
        OUT_yz = 1.0 - IN_yz

        weights_yz = (IN_yz / N_yz_array) - (OUT_yz / N_neq_yz_array)
        per_class_probs = softmax_temperature(
            logits,
            temperature=cfg.training_params.softmax_temperature
        )
        lagrangian_terms = (
            lambdas
            * weights_yz[..., None]
            * per_class_probs[None, None, :]
        )
        lagrangian = jnp.sum(lagrangian_terms)
        total_loss = ce_loss + cfg.training_params.batch_size * lagrangian
        return total_loss, (lagrangian, ce_loss)
    return per_sample_loss_fn


def get_crossentropy_regularized_demographic_parity(cfg: DictConfig, model: TrainState):
    def per_sample_loss_fn(params: TrainState, batch, artifacts, rng, train=True):
        lambdas = artifacts["lambdas"]
        N_z_array = artifacts["N_z"]
        N_neq_z_array = artifacts["N_neq_z"]

        inputs, targets, zi = batch  # Unpack the batch

        logits = model.apply_fn(
            {"params": params}, jnp.expand_dims(inputs, 0), train=train, dropout_rng=rng
        ).squeeze()  # Shape: (num_classes,)
        train_loss = cross_entropy_with_temperature(logits=logits, labels=targets, temperature=cfg.training_params.softmax_temperature)

        IN = zi  # Shape: (num_sensitive_classes,)
        OUT = 1.0 - zi  # Shape: (num_sensitive_classes,)

        weights = (IN / N_z_array) - (
            OUT / N_neq_z_array
        )  # Shape: (num_sensitive_classes,)

        per_class_probs = softmax_temperature(logits, cfg.training_params.softmax_temperature)  # Shape: (num_classes,)
        per_class_probs = per_class_probs[None, :]
        per_class_probs_masked = per_class_probs

        # Compute the per-sample lagrangian terms
        lagrangian_terms = (
            lambdas.T * weights[:, None] * per_class_probs_masked
        )  # Shape: (num_sensitive_classes, num_classes)

        # Sum over sensitive classes and output classes
        lagrangian = jnp.sum(lagrangian_terms)  # Scalar

        return (
            train_loss + (cfg.training_params.batch_size * lagrangian),
            (lagrangian, train_loss),
        )
    return per_sample_loss_fn



def get_crossentropy_regularized_equal_opportunity(cfg: DictConfig, model: TrainState):
    def per_sample_loss_fn(params: TrainState, batch, artifacts, rng, train=True):
        lambdas = artifacts["lambdas"]
        N_z = artifacts["N_z"]

        inputs, targets, zi = batch  # Unpack the batch
        pos_factor = jnp.argmax(targets)

        logits = model.apply_fn(
            {"params": params}, jnp.expand_dims(inputs, 0), train=train, dropout_rng=rng
        ).squeeze()  # Shape: (num_classes,)
        train_loss = focal_loss(logits=logits, labels=targets, temperature=cfg.training_params.softmax_temperature)
        per_class_probs = softmax_temperature(logits, cfg.training_params.softmax_temperature)  # Shape: (num_classes,)

        zi = jnp.argmax(zi)

        regularizer_term = pos_factor * per_class_probs[1] * lambdas[0][0] * ((1 - zi) / N_z[0] - zi / N_z[1])
        return (
            train_loss + (cfg.training_params.batch_size * regularizer_term),
            (regularizer_term, train_loss),  # Return individual components for logging
        )

    return per_sample_loss_fn


def get_balanced_accuracy(cfg: DictConfig, model: TrainState):
    def per_sample_loss_fn(params: TrainState, batch, artifacts, rng, train=True):
        lambdas = artifacts["lambdas"]
        N_z_array = artifacts["N_z"]
        N_neq_z_array = artifacts["N_neq_z"]

        inputs, targets, zi = batch  # Unpack the batch

        logits = model.apply_fn(
            {"params": params}, jnp.expand_dims(inputs, 0), train=train, dropout_rng=rng
        ).squeeze()  # Shape: (num_classes,)
        train_loss = cross_entropy_with_temperature(logits=logits, labels=targets,
                                                    temperature=cfg.training_params.softmax_temperature)

        IN = zi  # Shape: (num_sensitive_classes,)
        OUT = 1.0 - zi  # Shape: (num_sensitive_classes,)

        weights = (IN / N_z_array) - (
                OUT / N_neq_z_array
        )

        per_class_probs = softmax_temperature(logits, cfg.training_params.softmax_temperature)  # Shape: (num_classes,)
        reg_term = (1 - per_class_probs[jnp.argmax(targets)])

        lagrangian_terms = lambdas.T * weights[:, None] * reg_term

        lagrangian = jnp.sum(lagrangian_terms)  # Scalar

        return (
            train_loss + (cfg.training_params.batch_size * lagrangian),
            (lagrangian, train_loss),
        )
    return per_sample_loss_fn