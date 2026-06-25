"""Eight schools (non-centered) sampled with a vmapped multi-chain HMC, exported to ArviZ.

The model is written with ``distrax`` and differentiated with ``jax``; the parameters are
kept as a **pytree** ``{"eta": (8,), "mu": (), "log_tau": ()}`` and sampled directly --
``sample_vmapped_chains`` runs its leapfrog kernel leafwise via ``tree_map``, so there is
no flat-vector packing/unpacking. ``tau`` must be positive, so we sample ``log_tau``
(unconstrained, which is what HMC needs) and add the change-of-variables Jacobian
``log_tau`` so the target density stays exactly the one defined by ``log_likelihood``.

Unlike ``littlemcmc.sample`` -- which runs chains in separate OS processes -- here the
chains are the batch axis of ``jax.vmap`` via ``littlemcmc.hmc_jax.sample_vmapped_chains``:
every chain advances in lockstep inside one JIT-compiled, on-device program (``lax.scan``
over draws), so there are no processes, no pickling, and no per-draw Python loop. Its
warmup tunes a shared step size (dual averaging) and a diagonal mass matrix (from the
warmup variance) so the well-known funnel-ish ``mu``/``tau`` geometry still mixes.
"""

import os

# Run the vmapped trajectory on the GPU.
#os.environ.setdefault("JAX_PLATFORMS", "cuda")

import arviz as az
import distrax
import jax
from jax import value_and_grad
import jax.numpy as jnp
import numpy as np
import pymc as pm
from littlemcmc.hmc_jax import sample_vmapped_chains
from littlemcmc.nuts_jax import sample_vmapped_nuts_chains

# x64 is left off (jax default float32) to keep the GPU trajectory fast; the data is
# cast to match so distrax doesn't emit float64->float32 truncation warnings.
y = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float32)
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float32)
J = len(y)

# --- Sampler settings ---------------------------------------------------------
N_CHAINS = 4
N_WARMUP = 1000
N_DRAWS = 1000
N_LEAPFROG = 16
MAX_TREEDEPTH = 10
TARGET_ACCEPT = 0.9
EMAX = 1000.0
SEED = 42
INIT_STEP = 0.1


def log_likelihood(test_point):
    log_prior_eta = distrax.Normal(0.0, 1.0).log_prob(test_point["eta"]).sum()
    log_prior_mu = distrax.Normal(0.0, 10.0).log_prob(test_point["mu"])
    log_prior_tau = distrax.Transformed(
        distrax.Normal(loc=0.0, scale=1.0), distrax.Lambda(lambda x: jnp.exp(x))
    ).log_prob(test_point["tau"])
    log_like = (
        distrax.Independent(
            distrax.Normal(test_point["mu"] + test_point["tau"] * test_point["eta"], sigma)
        )
        .log_prob(y)
        .sum()
    )
    return log_prior_eta + log_like + log_prior_mu + log_prior_tau


def logp(q):
    # q is the sampled pytree in unconstrained space: {"eta", "mu", "log_tau"}.
    log_tau = q["log_tau"]
    test_point = {"eta": q["eta"], "mu": q["mu"], "tau": jnp.exp(log_tau)}
    # log_likelihood evaluates the LogNormal density of tau; adding log_tau converts the
    # change of variables to the (unconstrained) log_tau space we actually sample in.
    return log_likelihood(test_point) + log_tau


# The vmapped sampler traces this inside its compiled trajectory, so it must return
# JAX arrays (no np.asarray) built purely from jax.numpy ops.
logp_dlogp_func = value_and_grad(logp)

def test_hmc_jax():
    # Per-chain standard-normal starting points, one leaf per parameter. The pytree
    # structure here defines what the sampler advances; model_ndim is unused for pytrees.
    k_eta, k_mu, k_tau = jax.random.split(jax.random.PRNGKey(SEED), 3)
    start = {
        "eta": jax.random.normal(k_eta, (N_CHAINS, J)),
        "mu": jax.random.normal(k_mu, (N_CHAINS,)),
        "log_tau": jax.random.normal(k_tau, (N_CHAINS,)),
    }

    trace, stats = sample_vmapped_chains(
        logp_dlogp_func,
        None,
        draws=N_DRAWS,
        tune=N_WARMUP,
        chains=N_CHAINS,
        n_leapfrog=N_LEAPFROG,
        target_accept=TARGET_ACCEPT,
        init_step=INIT_STEP,
        Emax=EMAX,
        random_seed=SEED,
        start=start,
    )

    # trace is a pytree of (chains, draws, *event) arrays matching the position.
    log_tau = trace["log_tau"]
    posterior = {
        "eta": trace["eta"],
        "mu": trace["mu"],
        "log_tau": log_tau,
        "tau": np.exp(log_tau),
    }
    sample_stats = {
        "acceptance_rate": stats["acceptance_rate"],
        "diverging": stats["diverging"],
    }
    idata = az.from_dict({"posterior": posterior, "sample_stats": sample_stats})
    return idata, stats, sample_stats

def test_nuts_jax():
    # Same pytree start as the HMC backend; NUTS adapts the path length itself, so
    # there is no n_leapfrog -- max_treedepth caps the trajectory instead.
    k_eta, k_mu, k_tau = jax.random.split(jax.random.PRNGKey(SEED), 3)
    start = {
        "eta": jax.random.normal(k_eta, (N_CHAINS, J)),
        "mu": jax.random.normal(k_mu, (N_CHAINS,)),
        "log_tau": jax.random.normal(k_tau, (N_CHAINS,)),
    }

    trace, stats = sample_vmapped_nuts_chains(
        logp_dlogp_func,
        None,
        draws=N_DRAWS,
        tune=N_WARMUP,
        chains=N_CHAINS,
        max_treedepth=MAX_TREEDEPTH,
        target_accept=TARGET_ACCEPT,
        init_step=INIT_STEP,
        Emax=EMAX,
        random_seed=SEED,
        start=start,
    )

    # trace is a pytree of (chains, draws, *event) arrays matching the position.
    log_tau = trace["log_tau"]
    posterior = {
        "eta": trace["eta"],
        "mu": trace["mu"],
        "log_tau": log_tau,
        "tau": np.exp(log_tau),
    }
    sample_stats = {
        "acceptance_rate": stats["acceptance_rate"],
        "diverging": stats["diverging"],
    }
    idata = az.from_dict({"posterior": posterior, "sample_stats": sample_stats})
    return idata, stats, sample_stats

def test_pymc():
    with pm.Model() as model:
        mu = pm.Normal('mu', 0., 10.)
        tau = pm.LogNormal('tau', 0., 1.)
        eta = pm.Normal('eta', shape=8)
        obs = pm.Normal('observed', mu + tau * eta, sigma, observed=y)
        idata = pm.sample(
            draws=N_DRAWS,
            tune=N_WARMUP,
            chains=N_CHAINS,
            cores=1,
            target_accept=TARGET_ACCEPT,
            random_seed=SEED,
            return_inferencedata=True)
    return idata

def main():
    idata, stats, sample_stats = test_hmc_jax()
    print("vmapped HMC summary:")
    print(az.summary(idata, var_names=["mu", "tau", "eta"]))
    print("\nFinal step size:", stats["step_size"])
    print("Mean acceptance:", float(np.mean(sample_stats["acceptance_rate"])))
    print("Divergences:", int(sample_stats["diverging"].sum()))
    print("-------------\n")
    idata_nuts, stats_nuts, sample_stats_nuts = test_nuts_jax()
    print("vmapped NUTS summary:")
    print(az.summary(idata_nuts, var_names=["mu", "tau", "eta"]))
    print("\nNUTS final step size:", stats_nuts["step_size"])
    print("NUTS mean acceptance:", float(np.mean(sample_stats_nuts["acceptance_rate"])))
    print("NUTS divergences:", int(sample_stats_nuts["diverging"].sum()))
    print("-------------\n")
    ## The following is a PyMC3 model that is equivalent to the above, for comparison.
    idata_pm = test_pymc()
    print("\nPyMC3 summary:")
    print(az.summary(idata_pm, var_names=["mu", "tau", "eta"]))
    print("\nPyMC3 final step size:", idata_pm.sample_stats.step_size.values[:,-1])
    print("PyMC3 mean acceptance:", float(np.mean(idata_pm.sample_stats.acceptance_rate.values)))
    print("\nPyMC3 divergences:", int(idata_pm.sample_stats.diverging.sum()))

    return


if __name__ == "__main__":
    main()
