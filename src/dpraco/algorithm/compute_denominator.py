import jax
from jax import numpy as jnp


def _compute_equalized_odds_denominator(cfg, artifacts, batch, c, rng):
    denominator_type = cfg.algorithm.denominator_type
    lap_b = cfg.algorithm.laplace_parameter_b
    (x, y, z) = batch

    if denominator_type == "independent" or denominator_type == "public":
        N_yz = y.T @ z
    elif denominator_type == "post-processing":
        N_yz = jnp.sum(c, axis=0)

    else:
        raise ValueError(f"Unknown denominator type: {denominator_type}")

    with jax.named_scope("sync_gradients"):
        N_yz = jax.lax.psum(N_yz, axis_name="batch")

    if denominator_type == "independent":
        noise = (1.0 / lap_b) * jax.random.laplace(
            rng, shape=N_yz.shape
        )
        N_yz = N_yz + noise

    total = jnp.sum(N_yz)
    N_neq_yz = total - N_yz

    N_yz = jnp.where(N_yz == 0.0, 1e-4, N_yz)
    N_neq_yz = jnp.where(N_neq_yz == 0.0, 1e-4, N_neq_yz)

    artifacts["N_yz"] = jnp.where(artifacts["N_yz"] == 0, N_yz, artifacts["N_yz"])
    artifacts["N_neq_yz"] = jnp.where(
        artifacts["N_neq_yz"] == 0, N_neq_yz, artifacts["N_neq_yz"]
    )

    return artifacts


def _compute_demographic_parity_denominator(cfg, artifacts, batch, c, rng):
    if cfg.algorithm.denominator_type == "independent":
        # compute the laplace mechanism to know N_z and N_neq_z
        (x, y, z) = batch
        N_z = jnp.sum(z, axis=0)
    elif cfg.algorithm.denominator_type == "public":
        (x, y, z) = batch
        N_z = jnp.sum(z, axis=0)
    elif cfg.algorithm.denominator_type == "post-processing":
        N_z = jnp.sum(c, axis=0)
    else:
        raise ValueError(f"Unknown denominator type: {cfg.algorithm.denominator_type}")

    # sync N_z and N
    N_z = jax.lax.psum(N_z, axis_name="batch")
    if cfg.algorithm.denominator_type == "independent":
        N_z += 1 / cfg.algorithm.laplace_parameter_b * jax.random.laplace(rng, N_z.shape)

    N_neq_z = jnp.sum(N_z) - N_z

    N_z = jnp.where(N_z == 0, 1e-4, N_z)
    N_neq_z = jnp.where(N_neq_z == 0, 1e-4, N_neq_z)

    artifacts["N_z"] = jnp.where(artifacts["N_z"] == 0, N_z, artifacts["N_z"])
    artifacts["N_neq_z"] = jnp.where(
        artifacts["N_neq_z"] == 0, N_neq_z, artifacts["N_neq_z"]
    )



    return artifacts

def _compute_equal_opportunity(cfg, artifacts, batch, c, rng):
    if cfg.algorithm.denominator_type == "independent":
        (x, y, z) = batch
        y = y[:, 1]
        mask_1d_bool = (y == 1)
        mask_1d_float = mask_1d_bool.astype(z.dtype)
        masked_z = z * mask_1d_float[:, None]

        N_z = jnp.sum(masked_z, axis=0)
    else:
        raise ValueError(f"Unknown denominator type: {cfg.algorithm.denominator_type}")

    N_z = jax.lax.psum(N_z, axis_name="batch")
    if cfg.algorithm.denominator_type == "independent":
        N_z += 1 / cfg.algorithm.laplace_parameter_b * jax.random.laplace(rng, N_z.shape)

    N_neq_z = jnp.sum(N_z) - N_z

    N_z = jnp.where(N_z == 0, 1e-4, N_z)
    N_neq_z = jnp.where(N_neq_z == 0, 1e-4, N_neq_z)

    artifacts["N_z"] = jnp.where(artifacts["N_z"] == 0, N_z, artifacts["N_z"])
    artifacts["N_neq_z"] = jnp.where(
        artifacts["N_neq_z"] == 0, N_neq_z, artifacts["N_neq_z"]
    )

    return artifacts

def compute_denominator(cfg, artifacts, batch=None, c=None, rng=None):
    """
        Computes denomitor when needed
    """
    if cfg.algorithm.constraint_type == "DemographicParity":
        func = _compute_demographic_parity_denominator
    elif cfg.algorithm.constraint_type == "EqualizedOdds":
        func = _compute_equalized_odds_denominator
    elif cfg.algorithm.constraint_type == "FalseNegativeRate":
        func = lambda **kwargs: artifacts
    else:
        msg = f"Unknown constraint type: {cfg.algorithm.constraint_type}"
        raise ValueError(msg)

    return func(cfg=cfg, artifacts=artifacts, c=c, batch=batch, rng=rng)
