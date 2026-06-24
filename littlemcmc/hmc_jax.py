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

"""Vmapped multi-chain Hamiltonian Monte Carlo using JAX.

This module provides :func:`sample_vmapped_chains`, a self-contained multi-chain
HMC that runs *all* chains as the batch axis of ``jax.vmap`` -- every chain
advances in lockstep inside one ``jax.jit``-compiled program (``lax.scan`` over
draws, ``lax.fori_loop`` over leapfrog steps), with no OS processes, no pickling,
and no per-draw Python loop. It is the GPU/TPU-friendly alternative to
``littlemcmc.sample``'s process-per-chain parallelism, and is what the
``examples/`` scripts use for multi-chain sampling.

Two requirements follow from compiling the trajectory:

* The ``logp_dlogp_func`` must be **JAX-traceable** -- it must be built from
  ``jax.numpy`` ops and return JAX arrays. A function that returns
  ``np.asarray(...)`` (as the NumPy ``HamiltonianMC`` and the PyTorch/PyMC3
  cookbook examples do) cannot be traced and will not work here.
* Only **diagonal** mass matrices are supported (a per-dimension ``var``).

Divergence handling: the compiled trajectory lets non-finite energies propagate
and flags a divergence when the final energy is non-finite or the energy change
exceeds ``Emax``.
"""

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
except ImportError as err:  # pragma: no cover
    raise ImportError(
        "littlemcmc.hmc_jax requires JAX. Install it with `pip install jax jaxlib`."
    ) from err


# Dual-averaging defaults, matching littlemcmc.step_sizes.DualAverageAdaptation.
_DA_GAMMA = 0.05
_DA_T0 = 10.0
_DA_KAPPA = 0.75


def _build_vmapped_move(logp_dlogp_func, n_leapfrog, Emax):
    """Build a ``vmap``-over-chains single fixed-path-length HMC proposal.

    The returned callable maps ``(q, key, step_size, var)`` over the leading
    (chain) axis of ``q`` and ``key``; ``step_size`` and the diagonal mass-matrix
    ``var`` (with ``velocity = var * p`` and ``p ~ N(0, 1/var)``, the
    ``QuadPotentialDiag`` convention) are shared across chains.
    """

    def move(q, key, step_size, var):
        key_p, key_a = jax.random.split(key)
        logp0, grad0 = logp_dlogp_func(q)
        p0 = jax.random.normal(key_p, q.shape) / jnp.sqrt(var)
        energy0 = 0.5 * jnp.dot(p0, var * p0) - logp0

        dt = 0.5 * step_size

        def leapfrog(_, carry):
            q, p, grad = carry
            p = p + dt * grad
            q = q + step_size * (var * p)
            _, grad = logp_dlogp_func(q)
            p = p + dt * grad
            return (q, p, grad)

        q1, p1, _ = lax.fori_loop(0, n_leapfrog, leapfrog, (q, p0, grad0))
        logp1, _ = logp_dlogp_func(q1)
        energy1 = 0.5 * jnp.dot(p1, var * p1) - logp1

        energy_change = energy0 - energy1
        energy_change = jnp.where(jnp.isnan(energy_change), -jnp.inf, energy_change)
        diverging = (~jnp.isfinite(energy1)) | (jnp.abs(energy_change) > Emax)

        accept_prob = jnp.minimum(1.0, jnp.exp(energy_change))
        accepted = (~diverging) & (jax.random.uniform(key_a) < accept_prob)
        q_next = jnp.where(accepted, q1, q)
        return q_next, accept_prob, diverging

    return jax.vmap(move, in_axes=(0, 0, None, None))


def _build_phase(move_all_chains, target_accept, gamma, t0, kappa):
    """Build a JIT-compiled ``lax.scan`` that advances all chains over draws.

    The returned ``run_phase(q0, key, var, n_steps, init_logstep, adapt)`` returns
    ``(q_final, positions, accept_prob, diverging, logstep_bar)`` with ``positions``
    of shape ``(n_steps, n_chains, ndim)``. When ``adapt`` is True a single step
    size shared across chains is tuned by dual averaging on the mean accept
    statistic; ``logstep_bar`` is the averaged log step to carry into the next
    phase. ``n_steps`` and ``adapt`` are static so each combination compiles once.
    """

    def run_phase(q0, key, var, n_steps, init_logstep, adapt):
        mu_da = jnp.log(10.0) + init_logstep

        def body(carry, _):
            q, key, logstep, logstep_bar, hbar, t = carry
            key, sub = jax.random.split(key)
            chain_keys = jax.random.split(sub, q.shape[0])
            step_size = jnp.exp(logstep)
            q, accept_prob, diverging = move_all_chains(q, chain_keys, step_size, var)

            if adapt:
                t = t + 1.0
                w = 1.0 / (t + t0)
                hbar = (1.0 - w) * hbar + w * (target_accept - jnp.mean(accept_prob))
                logstep = mu_da - jnp.sqrt(t) / gamma * hbar
                eta = t ** (-kappa)
                logstep_bar = eta * logstep + (1.0 - eta) * logstep_bar

            return (q, key, logstep, logstep_bar, hbar, t), (q, accept_prob, diverging)

        init = (q0, key, init_logstep, init_logstep, 0.0, 0.0)
        (q_final, _, _, logstep_bar, _, _), (positions, accept_prob, diverging) = lax.scan(
            body, init, xs=None, length=n_steps
        )
        return q_final, positions, accept_prob, diverging, logstep_bar

    return jax.jit(run_phase, static_argnums=(3, 5))


def sample_vmapped_chains(
    logp_dlogp_func,
    model_ndim,
    draws=1000,
    tune=1000,
    chains=4,
    n_leapfrog=16,
    target_accept=0.8,
    init_step=0.1,
    Emax=1000.0,
    adapt_mass=True,
    random_seed=0,
    start=None,
):
    """Sample multiple chains with a vmapped, fully on-device fixed-path-length HMC.

    All ``chains`` advance together as the ``jax.vmap`` batch axis inside one
    JIT-compiled ``lax.scan``. Warmup tunes a single (shared) step size by dual
    averaging and, if ``adapt_mass`` is True, a diagonal mass matrix estimated from
    the warmup draws; both are then frozen for the sampling phase.

    Parameters
    ----------
    logp_dlogp_func : callable
        JAX-traceable ``x -> (logp, dlogp)`` (built from ``jax.numpy`` ops and
        returning JAX arrays, not ``np.asarray``), as required by the JAX backend.
    model_ndim : int
        Number of parameters (length of ``x``).
    draws, tune : int
        Post-warmup draws and warmup steps. Warmup is split in half when
        ``adapt_mass`` is True (estimate mass, then re-tune the step).
    chains : int
        Number of chains (the ``vmap`` batch size).
    n_leapfrog : int
        Fixed number of leapfrog steps per proposal (``vmap`` needs a uniform
        trip-count across the batch, so the path length cannot be randomized).
    target_accept : float
        Target mean acceptance for dual-averaging step-size adaptation.
    init_step : float
        Initial leapfrog step size before adaptation.
    Emax : float
        Energy-change threshold above which a draw is flagged divergent.
    adapt_mass : bool
        If True, estimate a diagonal mass matrix from the first warmup window.
    random_seed : int
        Seed for ``jax.random.PRNGKey``.
    start : array-like, optional
        Initial positions, shape ``(chains, model_ndim)`` or ``(model_ndim,)``
        (broadcast across chains). Defaults to standard-normal draws.

    Returns
    -------
    trace : np.ndarray
        Posterior draws, shape ``[chains, draws, model_ndim]``.
    stats : dict
        ``acceptance_rate`` and ``diverging`` (each ``[chains, draws]``), plus the
        frozen ``step_size`` (float) and ``mass_matrix_inv`` (``[model_ndim]``).
    """
    move_all_chains = _build_vmapped_move(logp_dlogp_func, int(n_leapfrog), Emax)
    run_phase = _build_phase(move_all_chains, target_accept, _DA_GAMMA, _DA_T0, _DA_KAPPA)

    key = jax.random.PRNGKey(random_seed)
    key, key_init = jax.random.split(key)
    if start is None:
        q0 = jax.random.normal(key_init, (chains, model_ndim))
    else:
        q0 = jnp.asarray(start)
        if q0.ndim == 1:
            q0 = jnp.broadcast_to(q0, (chains, model_ndim))

    init_logstep = jnp.log(jnp.asarray(init_step, dtype=q0.dtype))
    var = jnp.ones(model_ndim, dtype=q0.dtype)
    logstep_bar = init_logstep

    if tune > 0:
        n_w1 = tune // 2 if adapt_mass else tune
        key, sub = jax.random.split(key)
        q0, w1_pos, _, _, logstep_bar = run_phase(q0, sub, var, n_w1, init_logstep, True)
        if adapt_mass and tune - n_w1 > 0:
            # Estimate the diagonal mass matrix from the second half of window 1.
            var = jnp.clip(jnp.var(w1_pos[n_w1 // 2 :], axis=(0, 1)), 1e-8, None)
            key, sub = jax.random.split(key)
            q0, _, _, _, logstep_bar = run_phase(q0, sub, var, tune - n_w1, logstep_bar, True)

    key, sub = jax.random.split(key)
    _, positions, accept_prob, diverging, _ = run_phase(q0, sub, var, draws, logstep_bar, False)

    # positions: (draws, chains, ndim) -> ArviZ-friendly (chains, draws, ndim).
    trace = np.asarray(positions).transpose(1, 0, 2)
    stats = {
        "acceptance_rate": np.asarray(accept_prob).T,  # (chains, draws)
        "diverging": np.asarray(diverging).T.astype(bool),
        "step_size": float(jnp.exp(logstep_bar)),
        "mass_matrix_inv": np.asarray(var),
    }
    return trace, stats
