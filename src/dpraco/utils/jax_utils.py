from flax.training.train_state import TrainState
from functools import partial
import jax

jit_except_first = partial(jax.jit, static_argnums=(0,))


def get_inference_function(cfg, state: TrainState):
    def _inference_function(params, batch, artifacts):
        return state.apply_fn({"params": params}, batch, train=False)

    def _inference_function_bert(params, batch, artifacts):
        return state.apply_fn(**batch, params=params, train=False).logits

    if cfg.model.name == "bert":
        return _inference_function_bert

    return _inference_function
