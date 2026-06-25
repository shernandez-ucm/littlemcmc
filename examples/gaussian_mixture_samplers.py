#  Copyright 2019-2020 George Ho
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Sample a 2D Gaussian mixture with littlemcmc's samplers.

This exercises, on a single shared target:

* ``HamiltonianMC``                       -- the NumPy HMC step method
* ``hmc_jax.sample_vmapped_chains``       -- the vmapped multi-chain JAX HMC
* ``NUTS``                                -- the NumPy No-U-Turn sampler
* ``nuts_jax.sample_vmapped_nuts_chains`` -- the vmapped multi-chain JAX NUTS

The target is a balanced mixture of two isotropic 2D Gaussians with means at
``(-2, 0)`` and ``(2, 0)``. The barrier between the modes is shallow enough that
the chains mix across both, so each sampler should recover the full bimodal
marginal. For every sampler we pool all chains and report:

* the marginal mean        -- target ``[0, 0]`` (the mixture is symmetric),
* the marginal std         -- target ``[sqrt(5), 1] ~= [2.236, 1.0]``,
* the mode coverage        -- fraction of samples on each side of ``x0 = 0``
  (target ``0.5`` / ``0.5``) and the conditional mean of each side (target
  ``-2`` and ``+2``), i.e. both modes are actually populated,
* the mean acceptance statistic and the number of post-tuning divergences,
* the wall-clock sampling time (per draw, including tuning and, for the vmapped
  JAX sampler, the one-off JIT compilation).

Run with ``python examples/gaussian_mixture_samplers.py``. The vmapped JAX
sampler requires ``jax``/``jaxlib`` to be installed; if they are missing the
example runs the two NumPy samplers and skips the JAX one.
"""

import time

import numpy as np

import littlemcmc as lmc

# --- Target: balanced mixture of two isotropic 2D Gaussians (identity cov) ----
WEIGHTS = np.array([0.5, 0.5])
MEANS = np.array([[-2.0, 0.0], [2.0, 0.0]])
MODEL_NDIM = 2

# Sampling settings shared across samplers.
DRAWS = 1000
TUNE = 1000
CHAINS = 4
SEED = 42


def numpy_logp_dlogp_func(x):
    """NumPy mixture log-density and gradient.

    Returns a *scalar* log-density (and a length-2 gradient), which is what the
    NumPy HMC/NUTS step methods expect.
    """
    x = np.asarray(x, dtype=np.float64)
    diff = x[None, :] - MEANS  # (n_components, ndim)
    sq = np.sum(diff**2, axis=1)  # (n_components,)
    # log of each weighted component density (the 2*pi term cancels in the
    # gradient and shifts logp by a constant, but we keep it for honest logp).
    log_comp = np.log(WEIGHTS) - 0.5 * sq - np.log(2 * np.pi)
    c = log_comp.max()
    logp = c + np.log(np.sum(np.exp(log_comp - c)))
    resp = np.exp(log_comp - logp)  # responsibilities, sum to 1
    grad = -np.sum(resp[:, None] * diff, axis=0)  # d/dx logsumexp
    return float(logp), grad


def make_jax_logp_dlogp_func():
    """Build a JAX-traceable mixture log-density + gradient (or None if no JAX).

    The gradient is obtained by ``jax.grad`` so it independently corroborates the
    analytic NumPy gradient above.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return None

    means = jnp.asarray(MEANS)
    weights = jnp.asarray(WEIGHTS)

    def logp(x):
        diff = x[None, :] - means
        sq = jnp.sum(diff**2, axis=1)
        log_comp = jnp.log(weights) - 0.5 * sq - jnp.log(2 * jnp.pi)
        return jax.scipy.special.logsumexp(log_comp)

    @jax.jit
    def value_and_grad(x):
        return logp(x), jax.grad(logp)(x)

    return value_and_grad


def summarize(name, trace, stats, elapsed):
    """Print pooled-chain diagnostics for one sampler run."""
    samples = trace.reshape(-1, trace.shape[-1])  # (chains * draws, ndim)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)

    # Mode coverage along the separating axis x0.
    x0 = samples[:, 0]
    frac_left = float(np.mean(x0 < 0))
    left_mean = float(x0[x0 < 0].mean()) if np.any(x0 < 0) else float("nan")
    right_mean = float(x0[x0 >= 0].mean()) if np.any(x0 >= 0) else float("nan")

    # HMC reports "accept"; NUTS reports "mean_tree_accept"; the vmapped JAX
    # sampler reports "acceptance_rate".
    if "accept" in stats:
        accept = float(np.mean(stats["accept"]))
    elif "mean_tree_accept" in stats:
        accept = float(np.mean(stats["mean_tree_accept"]))
    elif "acceptance_rate" in stats:
        accept = float(np.mean(stats["acceptance_rate"]))
    else:
        accept = float("nan")

    # The trace excludes tuned samples by default, so every recorded diverging
    # flag is a post-tuning divergence.
    n_div = int(np.sum(stats["diverging"])) if "diverging" in stats else 0

    print("=" * 70)
    print("%s" % name)
    print("-" * 70)
    print("trace shape          : %s" % (trace.shape,))
    print("marginal mean        : [% .3f, % .3f]   target [ 0, 0]" % (mean[0], mean[1]))
    print("marginal std         : [% .3f, % .3f]   target [ 2.236, 1.0]" % (std[0], std[1]))
    print(
        "mode coverage (L/R)  : %.2f / %.2f          target 0.50 / 0.50"
        % (frac_left, 1 - frac_left)
    )
    print("mode means (L/R)     : [% .3f, % .3f]   target [-2, +2]" % (left_mean, right_mean))
    print("mean accept stat     : %.3f" % accept)
    print("divergences          : %d" % n_div)
    n_eval = trace.shape[0] * (DRAWS + TUNE)
    print(
        "wall time            : %.3f s  (%.2f ms / draw, incl. tuning & any JIT)"
        % (elapsed, 1000.0 * elapsed / n_eval)
    )


def run(name, logp_dlogp_func, step):
    # NUTS' internal log1mexp legitimately hits log(0) = -inf at the start of a
    # tree; the result is handled, so silence the benign divide warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        start = time.perf_counter()
        trace, stats = lmc.sample(
            logp_dlogp_func=logp_dlogp_func,
            model_ndim=MODEL_NDIM,
            draws=DRAWS,
            tune=TUNE,
            step=step,
            chains=CHAINS,
            cores=1,  # single process: keeps JAX closures out of multiprocessing
            progressbar=False,
            random_seed=SEED,
        )
        elapsed = time.perf_counter() - start
    summarize(name, trace, stats, elapsed)


def run_vmapped(name, logp_dlogp_func, sampler):
    """Run a vmapped multi-chain JAX sampler and summarize it like ``run``.

    ``sampler`` is ``hmc_jax.sample_vmapped_chains`` or
    ``nuts_jax.sample_vmapped_nuts_chains``; both share the same
    ``(logp_dlogp_func, model_ndim, ...) -> (trace, stats)`` contract.
    """
    start = time.perf_counter()
    trace, stats = sampler(
        logp_dlogp_func,
        MODEL_NDIM,
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        random_seed=SEED,
    )
    elapsed = time.perf_counter() - start
    summarize(name, trace, stats, elapsed)


def main():
    np.random.seed(SEED)

    # 1. HMC, NumPy step method.
    run(
        "HamiltonianMC (numpy)",
        numpy_logp_dlogp_func,
        lmc.HamiltonianMC(logp_dlogp_func=numpy_logp_dlogp_func, model_ndim=MODEL_NDIM),
    )

    # 2. Vmapped multi-chain JAX HMC (skipped if JAX is unavailable).
    jax_func = make_jax_logp_dlogp_func()
    if jax_func is None:
        print("=" * 70)
        print("sample_vmapped_chains (jax): SKIPPED -- jax/jaxlib not installed")
    else:
        from littlemcmc.hmc_jax import sample_vmapped_chains

        run_vmapped("sample_vmapped_chains (jax)", jax_func, sample_vmapped_chains)

    # 3. NUTS, NumPy step method.
    run(
        "NUTS (numpy)",
        numpy_logp_dlogp_func,
        lmc.NUTS(logp_dlogp_func=numpy_logp_dlogp_func, model_ndim=MODEL_NDIM),
    )

    # 4. Vmapped multi-chain JAX NUTS (skipped if JAX is unavailable).
    if jax_func is None:
        print("=" * 70)
        print("sample_vmapped_nuts_chains (jax): SKIPPED -- jax/jaxlib not installed")
    else:
        from littlemcmc.nuts_jax import sample_vmapped_nuts_chains

        run_vmapped(
            "sample_vmapped_nuts_chains (jax)", jax_func, sample_vmapped_nuts_chains
        )


if __name__ == "__main__":
    main()
