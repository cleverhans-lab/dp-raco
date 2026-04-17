from dp_accounting.pld.privacy_loss_distribution import (
    from_laplace_mechanism,
    from_gaussian_mechanism,
)
from omegaconf import DictConfig


def dpraco_accounting(cfg: DictConfig, step: int) -> float:
    # we multiply the noise variance by C in the grad step, so we need to divide it by C here, giving the sensitivity of 1.
    q_training = cfg.training_params.batch_size / cfg.dataset.num_train_samples

    sigma_grad = cfg.algorithm.sigma
    b = cfg.algorithm.laplace_parameter_b
    delta = cfg.algorithm.delta
    number_of_hist_estimations = step

    acountant = from_gaussian_mechanism(
        standard_deviation=sigma_grad,
        sensitivity=1,
        sampling_prob=q_training,
        use_connect_dots=True,
    ).self_compose(step)

    dlaplace_pld = from_laplace_mechanism(
        parameter=1 / b,
        sensitivity=1,
        sampling_prob=q_training,
        use_connect_dots=True,
    ).self_compose(number_of_hist_estimations)
    acountant = acountant.compose(dlaplace_pld)

    return acountant.get_epsilon_for_delta(delta=delta)
