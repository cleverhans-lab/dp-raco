import pandas as pd
import tracemalloc

from typing import Literal, Union
from flax.training.train_state import TrainState
from jax._src.random import PRNGKey
from jax.experimental.shard_map import shard_map
from omegaconf import DictConfig, OmegaConf
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from sklearn.metrics import classification_report

from dpraco.utils.jax_utils import get_inference_function
from dpraco.training.loss import get_loss
from dpraco.algorithm import get_update_rule, get_algorithm_artifacts
from jax import numpy as jnp, jit
import jax
import optax
from functools import partial
from ..algorithm.hist_estimation import get_histogram
from ..utils.constraints import constraint_value

def get_one_hot_preds(logits: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    preds = jnp.argmax(logits, axis=1)
    return jax.nn.one_hot(preds, num_classes)


@partial(jax.jit, static_argnums=(0,))
def test_step(cfg, state: TrainState, images: jnp.ndarray, labels: jnp.ndarray):
    if "params" in state.params:
        params = state.params["params"]
    else:
        params = state.params

    if cfg.model.name == "bert":
        logits = state.apply_fn(**images, params=params, train=False).logits
    else:
        logits = state.apply_fn({"params": params}, images, train=False)

    correct = jnp.sum(jnp.argmax(logits, axis=1) == jnp.argmax(labels, axis=1))
    loss = optax.softmax_cross_entropy(logits=logits, labels=labels).sum()
    return logits, correct, loss


def format_constraint(constraint):
    if isinstance(constraint, (np.ndarray, jnp.ndarray)):
        if len(constraint.flatten()) == 1:
            return constraint.item()
        else:
            return constraint.max()
    else:
        return float(constraint)

def evaluate_dataset(*args, **kwargs):
    return evaluate_dataset_new(*args, **kwargs)


@partial(jax.jit, static_argnames=['cfg', 'step_fn'])
def eval_step_compiled(cfg, state, images, labels, step_fn):
    return step_fn(cfg, state, images, labels)

@partial(jax.jit, static_argnames=['sharded_hist_fn'])
def hist_step_compiled(params, batch_data, rng, artifacts, sharded_hist_fn):
    return sharded_hist_fn(params, batch_data, rng, artifacts)

@jax.jit
def extract_predictions_batch(logits, labels):
    batch_preds = jnp.argmax(logits, axis=1)
    batch_targets = jnp.argmax(labels, axis=1)
    return batch_preds, batch_targets


def evaluate_dataset_new(
    cfg: DictConfig,
    state,
    dataset_fn,
    rng: PRNGKey,
    artifacts,
    step_fn,
    sharded_hist_fn,
    prefix: str = "test",
    verbose: bool = False,
):
    """Clean evaluation function without side effects."""
    max_batches = OmegaConf.select(cfg, "eval.max_batches_per_dataset", default=None)
    log_eval = OmegaConf.select(cfg, "eval.log_eval_batches", default=False) or (
        max_batches is not None
    )

    # Local state - properly scoped
    total_samples = 0
    sum_correct = 0.0
    sum_loss = 0.0
    c_soft_total = None
    c_hard_total = None
    
    # Local accumulators
    all_predictions = []
    all_targets = []
    current_batch_preds = []
    current_batch_targets = []
    
    for batch_idx, (images, labels, sensitives) in enumerate(dataset_fn(inference_mode=True)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch_size = labels.shape[0]
        if log_eval:
            print(f"[eval:{prefix}] batch {batch_idx} (n={batch_size})", flush=True)
        
        if verbose and batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx}, samples: {total_samples}")
        
        # Evaluation step
        logits, correct_batch, loss_batch = eval_step_compiled(cfg, state, images, labels, step_fn)
        
        # Histogram computation
        _, c_soft, _, c_hard = hist_step_compiled(
            state.params, (images, labels, sensitives), rng, artifacts, sharded_hist_fn
        )
        
        # Accumulate metrics
        sum_correct += correct_batch
        sum_loss += loss_batch
        total_samples += batch_size
        
        if c_soft_total is None:
            c_soft_total = c_soft
            c_hard_total = c_hard
        else:
            c_soft_total += c_soft
            c_hard_total += c_hard
        
        # if end_of_epoch:
        #     break

        batch_preds, batch_targets = extract_predictions_batch(logits, labels)
        current_batch_preds.append(np.asarray(batch_preds))
        current_batch_targets.append(np.asarray(batch_targets))
    
    # Handle remaining batches
    if current_batch_preds:
        preds_cpu = [np.asarray(pred) for pred in current_batch_preds]
        targets_cpu = [np.asarray(target) for target in current_batch_targets]
        all_predictions.extend(preds_cpu)
        all_targets.extend(targets_cpu)
    
    # Compute final metrics
    accuracy = float(sum_correct) / float(total_samples) if total_samples > 0 else 0.0
    average_loss = float(sum_loss) / float(total_samples) if total_samples > 0 else 0.0
    
    soft_constraint = format_constraint(constraint_value(cfg, c_soft_total))
    hard_constraint = format_constraint(constraint_value(cfg, c_hard_total))
    
    # Process predictions
    additional_metrics = {}
    if all_predictions:
        y_pred = np.concatenate(all_predictions)
        y_true = np.concatenate(all_targets)
        
        try:
            class_report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            if "1" in class_report_dict:
                additional_metrics = class_report_dict["1"]
            else:
                additional_metrics = class_report_dict.get("macro avg", {})
            additional_metrics = {f"{prefix}_{k}": v for k, v in additional_metrics.items()}
        except Exception:
            pass
    
    return {
        f"{prefix}_accuracy": accuracy,
        f"{prefix}_loss": average_loss,
        f"{prefix}_soft_constraint": soft_constraint,
        f"{prefix}_hard_constraint": hard_constraint,
        f"{prefix}_learning_loss": average_loss,
        **additional_metrics,
    }

def train_and_evaluate(
        cfg: DictConfig,
        state: TrainState,
        train_stream,
        test_data,
        val_data,
        rng: PRNGKey,
        return_: Literal["state", "metrics"] = "metrics"
) -> Union[TrainState, pd.DataFrame]:
    """
    Main training loop, which periodically evaluates on both test_data and val_data.

    - We store metrics in a single pandas DataFrame, appending a new row each time
      we evaluate. This avoids keeping a list of dictionaries in memory.
    - If `return_ == "metrics"`, the final returned object is a DataFrame.
    - If `return_ == "state"`, we return the final `TrainState`.
    """

    metrics_df = pd.DataFrame()

    tracemalloc.start()

    print("[train] building loss, update rule, and shard_map jit...", flush=True)
    update_rule = get_update_rule(cfg)
    per_sample_loss = get_loss(cfg, state)
    per_sample_training_loss = jax.jit(partial(per_sample_loss, train=True))
    inference_function = get_inference_function(cfg, state)

    key, rng = jax.random.split(rng)
    metadata = (cfg, per_sample_training_loss, inference_function)
    _train_stream = iter(train_stream())
    artifacts = get_algorithm_artifacts(cfg, key)

    device_array = np.array(jax.devices())
    mesh = Mesh(device_array, ("batch",))
    in_specs = (P(), P("batch"), P(), P(), P(), P())
    out_specs = (P(), P("batch"), P())

    in_spec_estimate_classes = (P(), P("batch"), P(), P())
    out_spec_estimate_classes = (P(), P(), P(), P())
    sharded_hist = partial(get_histogram, metadata)
    sharded_hist = shard_map(
        sharded_hist, mesh, in_spec_estimate_classes, out_spec_estimate_classes
    )
    update_rule = partial(update_rule, metadata)
    sharded_map = shard_map(update_rule, mesh, in_specs, out_specs)
    update_rule = jit(sharded_map)
    print("[train] entering training loop (first batch next)...", flush=True)

    def next_training_batch():
        """Unpack (batch, n) from the train stream; tolerate inference-style (x, y, z) tuples."""
        raw = next(_train_stream)
        if isinstance(raw, tuple) and len(raw) == 2:
            return raw[0], int(raw[1])
        if isinstance(raw, tuple) and len(raw) == 3:
            b = raw
            return b, int(b[0].shape[0])
        raise ValueError(
            "Train stream must yield (batch, num_samples) or a 3-tuple batch; "
            f"got {type(raw)} with length {len(raw) if isinstance(raw, tuple) else 'n/a'}."
        )

    for i in range(1, cfg.training_params.number_of_steps + 1):
        batch, number_of_samples = next_training_batch()

        try:
            if i == 1:
                print(
                    "[train] step 1: first gradient update (XLA compile; often minutes on CPU)...",
                    flush=True,
                )
            next_rng, rng = jax.random.split(rng, 2)
            state, train_metadata, artifacts = update_rule(
                state, batch, next_rng, i, artifacts, number_of_samples
            )
            # breakpoint()

            if (
                    i == 1
                    or i % cfg.eval.eval_every_k == 0
                    or i == cfg.training_params.number_of_steps
            ):
                print(f"[train] step {i}: running eval on train/val/test...", flush=True)
                train_results = evaluate_dataset(
                    cfg,
                    state,
                    train_stream,
                    rng,
                    artifacts,
                    test_step,
                    sharded_hist,
                    prefix="train",
                )
                # Evaluate on test data
                test_results = evaluate_dataset(
                    cfg,
                    state,
                    test_data,
                    rng,
                    artifacts,
                    test_step,
                    sharded_hist,
                    prefix="test",
                )

                # Evaluate on val data
                val_results = evaluate_dataset(
                    cfg,
                    state,
                    val_data,
                    rng,
                    artifacts,
                    test_step,
                    sharded_hist,
                    prefix="val",
                )
                # Summaries of losses and regularizer values
                loss_avg = jnp.mean(train_metadata["loss_values"])
                if "regularizer_values" in train_metadata:
                    regularizer_avg = jnp.sum(train_metadata["regularizer_values"])
                else:
                    regularizer_avg = 0.0

                step_metrics = {
                    "step": i,
                    "train_loss": float(loss_avg),
                    "regularizer_avg": float(regularizer_avg),
                    **train_results,
                    **test_results,
                    **val_results,
                }

                row_df = pd.DataFrame([step_metrics])
                metrics_df = pd.concat([metrics_df, row_df], ignore_index=True)

                import gc
                gc.collect()

        except KeyboardInterrupt:
            break

    metrics_df["gamma"] = cfg.algorithm.gamma
    metrics_df["algorithm"] = cfg.eval.name

    if return_ == "state":
        return state
    else:
        return metrics_df


