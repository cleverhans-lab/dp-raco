import logging

from omegaconf import DictConfig
from time import time
from dpraco.accounting.dp_sgd import dp_sgd_accounting
from dpraco.accounting.dpraco import dpraco_accounting


def get_accounting_step(cfg: DictConfig, step) -> float:
    step = int(step)
    if cfg.algorithm.name == "dpraco":
        return dpraco_accounting(cfg=cfg, step=step)
    elif cfg.algorithm.name == "dp_sgd":
        return dp_sgd_accounting(cfg, step=step)
    else:
        raise ValueError(f"Unknown accounting method: {cfg.accounting}")


def binary_search_closest_to_target(f, target, low, high):
    if low > high:
        return 0

    while low <= high:
        mid = low + (high - low) // 2

        if f(mid) <= target:
            result = mid
            low = mid + 1
        else:
            high = mid - 1

    return result


def get_number_of_steps_for_target_epsilon(cfg: DictConfig) -> int:
    MIN_STEPS = 1

    target_epsilon = cfg.training_params.target_epsilon
    # check if we can perform at least 1 step
    min_eps = get_accounting_step(cfg, MIN_STEPS)
    if min_eps > target_epsilon:
        logging.info(
            f"Target epsilon is too low, can't perform one step, the epsilon for the first step is {min_eps:.2f}."
        )
        return 0

    MAX_STEPS = 2

    while get_accounting_step(cfg, MAX_STEPS) <= target_epsilon:
        MIN_STEPS = MAX_STEPS
        MAX_STEPS = MAX_STEPS * 2
        if MAX_STEPS > cfg.training_params.MAX_STEPS:
            MAX_STEPS = cfg.training_params.MAX_STEPS
            break

    if MAX_STEPS == cfg.training_params.MAX_STEPS:
        logging.warning(
            f"Using fewer steps than maximal number of steps allowed by the privacy accounting, truncating to {MAX_STEPS} steps."
        )

    start_time = time()
    solution = binary_search_closest_to_target(
        lambda steps: get_accounting_step(cfg, steps),
        target_epsilon,
        MIN_STEPS,
        MAX_STEPS,
    )
    logging.info(f"Number of steps is {solution}, took {time() - start_time :.2f}")

    return int(solution)
