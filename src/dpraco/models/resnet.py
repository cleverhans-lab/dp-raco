from flax.training.train_state import TrainState
from jax import numpy as jnp
from functools import partial
import optax
from flax import linen as nn
from typing import Any, Callable, Sequence
import jax
from omegaconf import DictConfig


class BasicBlock(nn.Module):
    out_channels: int
    norm: Any
    expansion: int = 1
    stride: int = 1
    dropout_rate: float = 0.3
    dtype: Any = jnp.bfloat16
    kernel_init: Callable = nn.initializers.kaiming_normal()

    @nn.compact
    def __call__(self, x, train=True, dropout_rng=None):
        residual = x

        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            strides=self.stride,
            padding="SAME",
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)
        x = self.norm()(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(
            x, deterministic=not train, rng=dropout_rng
        )

        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            strides=1,
            padding="SAME",
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)
        x = self.norm()(x)

        if residual.shape != x.shape:
            residual = nn.Conv(
                features=self.out_channels * self.expansion,
                kernel_size=(1, 1),
                strides=self.stride,
                padding="SAME",
                use_bias=False,
                kernel_init=self.kernel_init,
                dtype=self.dtype,
            )(residual)
            residual = self.norm()(residual)

        x = x + residual
        x = nn.relu(x)
        return x


class BottleneckBlock(nn.Module):
    out_channels: int
    norm: Any
    expansion: int = 4  # For BottleneckBlock, expansion is 4
    stride: int = 1
    dropout_rate: float = 0.3
    dtype: Any = jnp.bfloat16
    kernel_init: Callable = nn.initializers.kaiming_normal()

    @nn.compact
    def __call__(self, x, train=True, dropout_rng=None):
        residual = x

        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(1, 1),
            strides=1,
            padding="VALID",
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)
        x = self.norm()(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(
            x, deterministic=not train, rng=dropout_rng
        )

        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            strides=self.stride,
            padding="SAME",
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)
        x = self.norm()(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(
            x, deterministic=not train, rng=dropout_rng
        )

        x = nn.Conv(
            features=self.out_channels * self.expansion,
            kernel_size=(1, 1),
            strides=1,
            padding="VALID",
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)
        x = self.norm()(x)

        if residual.shape[-1] != x.shape[-1] or self.stride != 1:
            residual = nn.Conv(
                features=self.out_channels * self.expansion,
                kernel_size=(1, 1),
                strides=self.stride,
                padding="VALID",
                use_bias=False,
                kernel_init=self.kernel_init,
                dtype=self.dtype,
            )(residual)
            residual = self.norm()(residual)

        x = x + residual
        x = nn.relu(x)
        return x


class ResNet(nn.Module):
    block: Any  # BasicBlock or BottleneckBlock
    layers: Sequence[int]  # Number of blocks in each stage
    num_classes: int
    norm: Any  # Non-default argument
    dropout_rate: float = 0.3
    dtype: Any = jnp.bfloat16
    kernel_init: Callable = nn.initializers.kaiming_normal()

    @nn.compact
    def __call__(self, x, dropout_rng=None, train=True):
        norm = self.norm

        # Initial convolutional layer
        x = nn.Conv(
            features=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x)
        x = norm()(x)
        x = nn.relu(x)
        if train:
            randomness_split = jax.random.split(dropout_rng, 6)
        else:
            randomness_split = [None] * 6

        x = nn.Dropout(rate=self.dropout_rate)(
            x, deterministic=not train, rng=randomness_split[0]
        )

        # Max pooling
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        # Define channels for each stage
        channels = [64, 128, 256, 512]
        x = self._make_layer(
            x,
            channels[0],
            self.layers[0],
            stride=1,
            norm=norm,
            train=train,
            dropout_rng=randomness_split[1],
        )
        x = self._make_layer(
            x,
            channels[1],
            self.layers[1],
            stride=2,
            norm=norm,
            train=train,
            dropout_rng=randomness_split[2],
        )
        x = self._make_layer(
            x,
            channels[2],
            self.layers[2],
            stride=2,
            norm=norm,
            train=train,
            dropout_rng=randomness_split[3],
        )
        x = self._make_layer(
            x,
            channels[3],
            self.layers[3],
            stride=2,
            norm=norm,
            train=train,
            dropout_rng=randomness_split[4],
        )

        # Global Average Pooling
        x = jnp.mean(x, axis=(1, 2))

        # Apply dropout before final fully connected layer
        x = nn.Dropout(rate=self.dropout_rate)(
            x, deterministic=not train, rng=randomness_split[5]
        )

        # Fully connected layer
        x = nn.Dense(
            features=self.num_classes, kernel_init=self.kernel_init, dtype=self.dtype
        )(x)

        return x

    def _make_layer(self, x, out_channels, blocks, stride, norm, train, dropout_rng):
        strides = [stride] + [1] * (blocks - 1)
        for stride in strides:
            x = self.block(
                out_channels=out_channels,
                norm=norm,
                stride=stride,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
            )(x, train=train, dropout_rng=dropout_rng)
        return x


def get_resnet(model_depth, num_classes=10, dtype=jnp.bfloat16):
    norm = partial(nn.GroupNorm, num_groups=16, epsilon=1e-5, dtype=jnp.float32)

    if model_depth == 16:
        # ResNet16 configuration
        layers = [2, 2, 2, 1]  # Number of blocks in each stage
        block = BasicBlock
    elif model_depth == 50:
        # ResNet50 configuration
        layers = [3, 4, 6, 3]
        block = BottleneckBlock
    else:
        raise ValueError(f"Unsupported model depth: {model_depth}")

    model = ResNet(
        block=block,
        layers=layers,
        num_classes=num_classes,
        norm=norm,
        dropout_rate=0.3,
        dtype=dtype,
    )

    return model


# Use SGD with momentum and weight decay
def create_train_state(cfg, rng):
    if cfg.model.name == "resnet16":
        cnn = get_resnet(model_depth=16, num_classes=cfg.dataset.num_classes)
    elif cfg.model.name == "resnet50":
        cnn = get_resnet(model_depth=50, num_classes=cfg.dataset.num_classes)
    else:
        raise ValueError(f"Unsupported model: {cfg.model.name}")

    model_vars = cnn.init(
        rng,
        jnp.ones(
            [
                1,
                cfg.dataset.img_height,
                cfg.dataset.img_width,
                cfg.dataset.img_channels,
            ],
            dtype=jnp.bfloat16,
        ),
        train=False,
    )

    # Use SGD optimizer with momentum
    tx = optax.chain(
        optax.add_decayed_weights(cfg.training_params.weight_decay),
        optax.sgd(
            learning_rate=cfg.training_params.lr,
            momentum=cfg.training_params.momentum,
            nesterov=True,
        ),
    )

    state = TrainState.create(apply_fn=cnn.apply, params=model_vars["params"], tx=tx)
    return state


def warmup_scheduler(cfg: DictConfig):
    return optax.warmup_cosine_decay_schedule(
        init_value=cfg.training_params.lr,
        peak_value=cfg.training_params.lr,
        warmup_steps=0,  # TODO: adjust this
        decay_steps=cfg.training_params.number_of_steps,
        end_value=0.001,
    )
