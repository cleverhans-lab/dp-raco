from functools import partial
from argparse import Namespace

import jax
from jax import random
from jax._src.random import PRNGKey
from jax.sharding import Mesh
import numpy as np
from scipy.stats import poisson
from sklearn.model_selection import StratifiedKFold
from omegaconf import DictConfig

from dpraco.data.dataloader import GeneralData
from dpraco.data.args import process_arguments
from dpraco.data.ftables import get_folkstable
from dpraco.data.heart import get_heart

def get_poisson_upper_bound(lambda_param: float, q: float, alpha: float = 0.001) -> int:
    thinned_rate = lambda_param * q
    k = 0
    while poisson.cdf(k, thinned_rate) < 1 - alpha:
        k += 1
    return k

@partial(jax.jit, static_argnums=(0, 1))
def poisson_sample(size, sampling_rate, rng):
    bern_rng, rng = random.split(rng)
    batch_idxs = random.bernoulli(key=bern_rng, p=sampling_rate, shape=(size,))
    return batch_idxs, rng


def one_hot(x, num_class=2):
    return np.eye(num_class)[x]

def split_into_folds(
    features: np.ndarray,
    labels: np.ndarray,
    sensitives: np.ndarray,
    K: int,
    fairness_constraint: str,
    rng: np.random.Generator
):
    num_samples = features.shape[0]
    if fairness_constraint == "DemographicParity":
        combined_stratify = np.array([
            f"{labels[i].argmax()}_{sensitives[i].argmax()}"
            for i in range(num_samples)
        ])
    else:
        combined_stratify = np.array([labels[i].argmax() for i in range(num_samples)])

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)

    folds = []
    for train_index, test_index in skf.split(features, combined_stratify):
        folds.append((train_index, test_index))

    return folds

def get_data_tables(cfg: DictConfig, permute_train=False, seed=0):
    args = Namespace()
    args.dataset = cfg.dataset.name
    args.num_classes = 2
    args.split = 1
    args.seed = 0

    args = process_arguments(args, cfg.dataset.data_path)
    np_rng = np.random.default_rng(cfg.dataset.rng)

    full_data = GeneralData(
        path=args.path,
        random_state=np_rng,
        sensitive_attributes=args.sensitive_attributes,
        cols_to_norm=args.cols_to_norm,
        output_col_name=args.output_col_name,
        split=0.75,  # or other
    )

    test_data = full_data.getTest(return_tensor=False)
    test_features = np.concatenate([x[0][None, :] for x in test_data], axis=0)
    test_labels = np.array([x[2] for x in test_data])
    test_sensitives = np.array([x[3] for x in test_data])
    test_labels = one_hot(test_labels, num_class=2)
    test_sensitives = one_hot(test_sensitives, num_class=2)

    dataset_train = full_data.getTrain(return_tensor=False)
    features = np.concatenate([x[0][None, :] for x in dataset_train], axis=0)
    labels = np.array([x[2] for x in dataset_train])
    sensitives = np.array([x[3] for x in dataset_train])
    labels = one_hot(labels, num_class=2)
    sensitives = one_hot(sensitives, num_class=2)

    # Folds
    K = cfg.training_params.num_folds
    fairness_constraint = cfg.algorithm.constraint_type
    folds = split_into_folds(
        features=features,
        labels=labels,
        sensitives=sensitives,
        K=K,
        fairness_constraint=fairness_constraint,
        rng=np.random.default_rng(cfg.dataset.rng)
    )

    if hasattr(cfg.training_params, "seeds"):
        seeds = cfg.training_params.seeds
        idx = seeds.index(seed)
    else:
        idx = 0

    train_index, val_index = folds[idx]
    train_features = features[train_index]
    train_labels = labels[train_index]
    train_sensitives = sensitives[train_index]
    val_features = features[val_index]
    val_labels = labels[val_index]
    val_sensitives = sensitives[val_index]

    return (
        train_features,
        train_labels,
        train_sensitives,
        val_features,
        val_labels,
        val_sensitives,
        test_features,
        test_labels,
        test_sensitives
    )

class TrainDataGenerator:
    def __init__(
        self,
        cfg: DictConfig,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        train_sensitives: np.ndarray,
        rng: PRNGKey,
        poisson_upperbound: int,
        inference_mode: bool = False
    ):
        self.cfg = cfg
        self.inference_mode = inference_mode

        self.train_features = train_features
        self.train_labels = train_labels
        self.train_sensitives = train_sensitives

        self.num_train = train_features.shape[0]
        self.rng = rng
        self.poisson_upperbound = poisson_upperbound

        if inference_mode:
            self.poisson_mode = False
            self.batch_size = cfg.training_params.eval_batch_size
        else:
            self.poisson_mode = cfg.training_params.poisson
            self.batch_size = cfg.training_params.batch_size

        self.num_complete_batches, self.leftover = divmod(self.num_train, self.batch_size)
        self.num_batches = self.num_complete_batches + bool(self.leftover)
        self.q_B = self.batch_size / self.num_train

    def __iter__(self):
        if self.poisson_mode:
            while True:
                batch_idxs, self.rng = poisson_sample(self.num_train, self.q_B, self.rng)
                batch_train_features = self.train_features[batch_idxs]
                batch_train_labels = self.train_labels[batch_idxs]
                batch_train_sensitives = self.train_sensitives[batch_idxs]

                if self.cfg.training_params.pad_poisson:
                    # Pad or truncate so that each batch is exactly self.poisson_upperbound
                    current_size = batch_train_features.shape[0]
                    if current_size < self.poisson_upperbound:
                        # pad with zeros
                        num_to_add = self.poisson_upperbound - current_size
                        feat_dim = batch_train_features.shape[1]
                        lab_dim = batch_train_labels.shape[1]
                        sens_dim = batch_train_sensitives.shape[1]

                        pad_feats = np.zeros((num_to_add, feat_dim), dtype=batch_train_features.dtype)
                        pad_labels = np.zeros((num_to_add, lab_dim), dtype=batch_train_labels.dtype)
                        pad_sens = np.zeros((num_to_add, sens_dim), dtype=batch_train_sensitives.dtype)

                        batch_train_features = np.concatenate([batch_train_features, pad_feats], axis=0)
                        batch_train_labels = np.concatenate([batch_train_labels, pad_labels], axis=0)
                        batch_train_sensitives = np.concatenate([batch_train_sensitives, pad_sens], axis=0)

                    elif current_size > self.poisson_upperbound:
                        # truncate
                        batch_train_features = batch_train_features[: self.poisson_upperbound]
                        batch_train_labels = batch_train_labels[: self.poisson_upperbound]
                        batch_train_sensitives = batch_train_sensitives[: self.poisson_upperbound]

                yield (batch_train_features, batch_train_labels, batch_train_sensitives, len(batch_train_features))

        elif self.inference_mode:
            for i in range(self.num_batches):
                batch_idx = slice(i * self.batch_size, (i + 1) * self.batch_size)
                yield (self.train_features[batch_idx],
                       self.train_labels[batch_idx],
                       self.train_sensitives[batch_idx], len(self.train_features[batch_idx]))
        else:
            while True:
                perm = random.permutation(key=self.rng, x=self.num_train)
                _, self.rng = random.split(self.rng)

                for i in range(self.num_batches):
                    batch_idx = perm[i * self.batch_size : (i + 1) * self.batch_size]
                    X = self.train_features[batch_idx]
                    Y = self.train_labels[batch_idx]
                    s = self.train_sensitives[batch_idx]
                    yield (X, Y, s, len(X))

def tabular_data_stream(cfg: DictConfig, rng: PRNGKey, seed: int):
    import tensorflow as tf

    if cfg.dataset.name == "folktables":
        dataset_fn = get_folkstable
    elif cfg.dataset.name in ["adult", "retired-adult", "credit-card", "chit-default-small", "parkinsons"]:
        dataset_fn = get_data_tables
    elif cfg.dataset.name == "heart":
        dataset_fn = get_heart
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")
    (
        train_features,
        train_labels,
        train_sensitives,
        val_features,
        val_labels,
        val_sensitives,
        test_features,
        test_labels,
        test_sensitives,
    ) = dataset_fn(cfg=cfg, permute_train=False, seed=seed)
    cfg.dataset.num_train_samples = len(train_features)

    batch_size = cfg.training_params.batch_size
    q_B = batch_size / train_features.shape[0]
    poisson_upperbound = get_poisson_upper_bound(
        train_features.shape[0],
        q_B,
        alpha=0.001
    )

    train_gen = TrainDataGenerator(
        cfg=cfg,
        train_features=train_features,
        train_labels=train_labels,
        train_sensitives=train_sensitives,
        rng=rng,
        poisson_upperbound=poisson_upperbound,
        inference_mode=False,  # training mode
    )


    train_gen_eval = TrainDataGenerator(
        cfg=cfg,
        train_features=train_features,
        train_labels=train_labels,
        train_sensitives=train_sensitives,
        rng=rng,
        poisson_upperbound=poisson_upperbound,
        inference_mode=True,  # training mode
    )


    feature_dim = train_features.shape[1]
    label_dim = train_labels.shape[1]
    sens_dim = train_sensitives.shape[1]

    output_signature = (
        tf.TensorSpec(shape=(None, feature_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(None, label_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(None, sens_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    train_ds = tf.data.Dataset.from_generator(
        generator=lambda: train_gen,
        output_signature=output_signature
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    train_val_ds = tf.data.Dataset.from_generator(
        generator=lambda: train_gen_eval,
        output_signature=output_signature
    )
    train_val_ds = train_val_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels, val_sensitives))
    val_ds = val_ds.batch(cfg.training_params.eval_batch_size, drop_remainder=False)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_features, test_labels, test_sensitives))
    test_ds = test_ds.batch(cfg.training_params.eval_batch_size, drop_remainder=False)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    mesh = Mesh(jax.devices(), ("batch",))

    def shard_batch(batch, batch_size):
        features, labels, sens = batch
        devices = jax.devices()
        num_devices = len(devices)
        remainder = features.shape[0] % num_devices

        if remainder != 0:
            features = features[:-remainder]
            labels = labels[:-remainder]
            sens = sens[:-remainder]

        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))
        actual_size = features.shape[0]
        f_shape = (actual_size,) + features.shape[1:]
        l_shape = (actual_size,) + labels.shape[1:]
        s_shape = (actual_size,) + sens.shape[1:]

        def cb_feats(idx):
            return features[idx]

        def cb_labels(idx):
            return labels[idx]

        def cb_sens(idx):
            return sens[idx]

        sharded_feats = jax.make_array_from_callback(f_shape, sharding, cb_feats)
        sharded_labels = jax.make_array_from_callback(l_shape, sharding, cb_labels)
        sharded_sens = jax.make_array_from_callback(s_shape, sharding, cb_sens)

        return sharded_feats, sharded_labels, sharded_sens

    def train_iterator(inference_mode=False):
        if inference_mode:
            for *batch, _ in train_val_ds.as_numpy_iterator():
                yield shard_batch(batch, cfg.training_params.eval_batch_size)
        else:
            for *batch, batchsize in train_ds.as_numpy_iterator():
                yield shard_batch(batch, cfg.training_params.batch_size), batchsize

    def val_iterator(inference_mode=False):
        for batch in val_ds.as_numpy_iterator():
            yield shard_batch(batch, cfg.training_params.eval_batch_size)

    def test_iterator(inference_mode=False):
        for batch in test_ds.as_numpy_iterator():
            yield shard_batch(batch, cfg.training_params.eval_batch_size)

    return train_iterator, val_iterator, test_iterator
