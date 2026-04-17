import dm_pix as pix
import jax
from omegaconf import DictConfig
from jax import numpy as jnp
from jax.random import PRNGKey
from functools import partial

@partial(jax.jit, static_argnums=(0,))
def image_augmentation(cfg: DictConfig, img: jnp.array, key: PRNGKey) -> jnp.array:
    max_delta = cfg.dataset.max_delta
    lower, upper = cfg.dataset.contrast
    key_brightness, key_contrast, key_crop, key_flip_left_right = jax.random.split(key, num=4)
    img = jax.image.resize(img, (cfg.dataset.shape_aug, cfg.dataset.shape_aug, cfg.dataset.num_channels), "bilinear")
    img = pix.random_crop(key_crop, img, (cfg.dataset.shape_img, cfg.dataset.shape_img, cfg.dataset.num_channels))
    img = pix.random_flip_left_right(key_flip_left_right, img)
    img = pix.random_brightness(key_brightness, img, max_delta)
    img = pix.random_contrast(key_contrast, img, lower, upper)
    # img = pix.random_flip_up_down(key_flip_up_down, img)
    return img
