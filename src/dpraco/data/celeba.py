import logging
from typing import Callable, Tuple, Optional
from functools import partial

import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import tensorflow as tf
import tensorflow_datasets as tfds
from omegaconf import DictConfig
from jax.random import PRNGKey


def celeba_data_stream(
    cfg: DictConfig,
    rngs: PRNGKey,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
) -> Tuple[Callable[[], Tuple], Callable[[], Tuple], Callable[[], Tuple]]:
    """
    Optimized CelebA data pipeline for JAX with improved performance.
    """
    if max_train_samples is None:
        max_train_samples = getattr(cfg.dataset, "max_train_samples", None)
    if max_val_samples is None:
        max_val_samples = getattr(cfg.dataset, "max_val_samples", None)
    if max_test_samples is None:
        max_test_samples = getattr(cfg.dataset, "max_test_samples", None)

    # ------------------------------------------------------------------
    # 1. Load TFDS splits with optimizations
    # ------------------------------------------------------------------
    read_config = tfds.ReadConfig(
        shuffle_seed=int(rngs[0]),
        skip_prefetch=True,  # We'll handle prefetching ourselves
    )
    
    splits = ["train", "validation", "test"]
    train_ds, val_ds, test_ds = tfds.load(
        "celeb_a",
        split=splits,
        shuffle_files=True,
        data_dir=getattr(cfg.dataset, "data_path", None),
        read_config=read_config,
    )

    # ------------------------------------------------------------------
    # 2. Optimized preprocessing with GPU acceleration
    # ------------------------------------------------------------------
    @tf.function
    def preprocess_batch(batch):
        """Vectorized preprocessing for entire batch"""
        imgs = tf.cast(batch["image"], tf.float32) / 255.0
        imgs = tf.image.resize(imgs, (cfg.dataset.image_size, cfg.dataset.image_size))

        label_vals = tf.cast(batch["attributes"][cfg.dataset.label_attr], tf.int32)
        sens_vals = tf.cast(batch["attributes"][cfg.dataset.sensitive_attr], tf.int32)

        labels = tf.one_hot(label_vals, cfg.dataset.num_classes)
        sens = tf.one_hot(sens_vals, cfg.dataset.num_fairness_classes)
        
        return imgs, labels, sens

    # ------------------------------------------------------------------
    # 3. JAX sharding setup (done once)
    # ------------------------------------------------------------------
    n_devices = len(jax.devices())
    mesh = Mesh(jax.devices(), "batch")
    sharding = NamedSharding(mesh, PartitionSpec("batch"))
    
    # Pre-compile the device_put operation
    @partial(jax.jit, static_argnames=['target_shape'])
    def shard_batch(imgs, labels, sens, target_shape):
        """JIT-compiled sharding operation"""
        return (
            jax.device_put(imgs, sharding),
            jax.device_put(labels, sharding), 
            jax.device_put(sens, sharding)
        )

    # ------------------------------------------------------------------
    # 4. Optimized TFDS pipeline builder
    # ------------------------------------------------------------------
    def make_tf_pipeline(ds, *, batch_size, training, max_samples=None, repeat: bool = False):
        # Apply sample limit early if specified
        if max_samples is not None:
            ds = ds.take(max_samples)

        if training:
            # More aggressive shuffling for training
            buffer_size = min(cfg.dataset.num_train_samples, 10000)
            ds = ds.shuffle(buffer_size, seed=int(rngs[0]), reshuffle_each_iteration=True)

        # Ensure batch size is divisible by number of devices
        effective_batch_size = (batch_size // n_devices) * n_devices
        if effective_batch_size == 0:
            raise ValueError(
                f"batch_size={batch_size} is too small for {n_devices} JAX devices; "
                f"increase training/eval batch size to at least {n_devices}."
            )
        if effective_batch_size != batch_size:
            logging.warning(
                f"Adjusting batch size from {batch_size} to {effective_batch_size} "
                f"to be divisible by {n_devices} devices"
            )

        ds = (
            ds.batch(effective_batch_size, drop_remainder=training)  # Drop remainder for training stability
            .map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )
        # Only the live training loop needs an infinite stream. Val/test/eval must be finite
        # or ``evaluate_dataset`` never terminates (see ``training_routine.evaluate_dataset``).
        if repeat:
            ds = ds.repeat()

        return ds.as_numpy_iterator()  # Convert to numpy iterator for faster JAX interop

    # ------------------------------------------------------------------
    # 5. Create optimized iterators
    # ------------------------------------------------------------------
    train_iter = make_tf_pipeline(
        train_ds,
        batch_size=cfg.training_params.batch_size,
        training=True,
        max_samples=max_train_samples,
        repeat=True,
    )
    # One epoch over train for metrics (matches tabular ``train_val_ds`` semantics).
    train_iter_eval = make_tf_pipeline(
        train_ds,
        batch_size=cfg.training_params.eval_batch_size,
        training=False,
        max_samples=max_train_samples,
        repeat=False,
    )
    val_iter = make_tf_pipeline(
        val_ds,
        batch_size=cfg.training_params.eval_batch_size,
        training=False,
        max_samples=max_val_samples,
        repeat=False,
    )
    test_iter = make_tf_pipeline(
        test_ds,
        batch_size=cfg.training_params.eval_batch_size,
        training=False,
        max_samples=max_test_samples,
        repeat=False,
    )

    # ------------------------------------------------------------------
    # 6. Fast batch loading with minimal copying
    # ------------------------------------------------------------------
    def load_and_shard_batch(numpy_batch):
        """Efficiently load and shard a numpy batch"""
        imgs, labels, sens = numpy_batch
        
        # Verify shapes are compatible with device count
        batch_size = imgs.shape[0]
        if batch_size % n_devices != 0:
            # This should rarely happen with our pipeline adjustments
            trim_size = (batch_size // n_devices) * n_devices
            imgs, labels, sens = imgs[:trim_size], labels[:trim_size], sens[:trim_size]
            batch_size = trim_size
        
        # Single JIT-compiled sharding operation
        sharded_imgs, sharded_labels, sharded_sens = shard_batch(
            imgs, labels, sens, target_shape=imgs.shape
        )
        
        return sharded_imgs, sharded_labels, sharded_sens, batch_size

    # ------------------------------------------------------------------
    # 7. Memory-efficient iterator wrappers
    # ------------------------------------------------------------------
    def make_jax_iterator(tf_iter, inference_mode: bool):
        """Convert TF iterator to JAX iterator with efficient batching.

        Matches ``tabular_data_stream``: training yields ``(batch_3tuple, batch_size)``;
        evaluation yields a single 3-tuple ``(images, labels, sensitives)`` per step.
        """
        for numpy_batch in tf_iter:
            imgs, labels, sens, batch_size = load_and_shard_batch(numpy_batch)

            if inference_mode:
                yield imgs, labels, sens
            else:
                yield (imgs, labels, sens), batch_size
            

    def train_iterator(inference_mode=False):
        tf_iter = train_iter_eval if inference_mode else train_iter
        return make_jax_iterator(tf_iter, inference_mode)

    def val_iterator(inference_mode=False):
        return make_jax_iterator(val_iter, inference_mode)

    def test_iterator(inference_mode=False):
        return make_jax_iterator(test_iter, inference_mode)

    return train_iterator, val_iterator, test_iterator


# ------------------------------------------------------------------
# Additional utility functions for further optimization
# ------------------------------------------------------------------

def create_async_data_loader(iterator_fn, prefetch_size: int = 2):
    """
    Create an asynchronous data loader with prefetching.
    Use this wrapper around your iterators for even better performance.
    """
    import threading
    import queue
    
    def async_loader():
        data_queue = queue.Queue(maxsize=prefetch_size)
        
        def producer():
            try:
                for batch in iterator_fn():
                    data_queue.put(batch)
                data_queue.put(None)  # Sentinel value
            except Exception as e:
                data_queue.put(e)
        
        producer_thread = threading.Thread(target=producer)
        producer_thread.start()
        
        while True:
            item = data_queue.get()
            if item is None:  # End of data
                break
            elif isinstance(item, Exception):
                raise item
            else:
                yield item
        
        producer_thread.join()
    
    return async_loader


def benchmark_data_loader(iterator_fn, num_batches: int = 100):
    """Benchmark the data loader performance"""
    import time
    
    print(f"Benchmarking data loader for {num_batches} batches...")
    start_time = time.time()
    
    count = 0
    for i, batch in enumerate(iterator_fn()):
        if i >= num_batches:
            break
        count += 1
        if i % 10 == 0:
            print(f"Processed {i+1} batches...")
    
    end_time = time.time()
    total_time = end_time - start_time
    batches_per_sec = count / total_time
    
    print(f"Processed {count} batches in {total_time:.2f}s")
    print(f"Throughput: {batches_per_sec:.2f} batches/sec")
    
    return batches_per_sec