from argparse import Namespace

import numpy as np
from omegaconf import DictConfig

from dpraco.data.args import process_arguments
from dpraco.data.utils import process_data

import logging


def one_hot(x, num_class=2):
    return np.eye(num_class)[x]


def adult(cfg: DictConfig, permute_train=False):
    args = Namespace()

    if cfg.dataset.name == "adult":
        args.dataset = "adult"
    else:
        raise ValueError(f"Unknown dataset: {cfg.training_params.name}")

    args.num_classes = 2
    args.output_col_name = "income"
    args.split = 0.75
    args.seed = cfg.dataset.rng

    args = process_arguments(args, cfg.dataset.data_path)
    np_rng = np.random.default_rng(cfg.dataset.rng)
    train_features, train_labels, train_sensitives, test_features, test_labels, test_sensitives = process_data(
        np_rng, args, log=logging.info
    )

    train_labels = one_hot(train_labels, num_class=2)
    train_sensitives = one_hot(train_sensitives, num_class=2)
    test_labels = one_hot(test_labels, num_class=2)
    test_sensitives = one_hot(test_sensitives, num_class=2)

    cfg.dataset.num_train_samples = train_labels.shape[0]

    return (
        train_features,
        train_labels,
        train_sensitives,
        test_features,
        test_labels,
        test_sensitives,
    )
