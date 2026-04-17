from dp_accounting.pld.privacy_loss_distribution import (
    from_gaussian_mechanism,
)
from omegaconf import DictConfig


def dp_sgd_accounting(cfg: DictConfig, step: int) -> float:
    q = cfg.training_params.batch_size / cfg.dataset.num_train_samples
    sigma_grad = cfg.algorithm.sigma
    delta = cfg.algorithm.delta

    # we multiply the noise variance by C in the grad step, so we need to divide it by C here, giving the sensitivity of 1.
    gaussian_pld = from_gaussian_mechanism(
        standard_deviation=sigma_grad,
        sensitivity=1,
        sampling_prob=q,
              use_connect_dots=True,
    ).self_compose(step)
    return gaussian_pld.get_epsilon_for_delta(delta=delta)
