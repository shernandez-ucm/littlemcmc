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

The position ``q`` may be any JAX **pytree** (a single array, or an arbitrarily
nested dict/list/tuple of arrays). The leapfrog kernel is written entirely with
``jax.tree_util.tree_map``, so the momentum, gradient and diagonal mass matrix
all share ``q``'s structure and a plain array is just the single-leaf case.

Two requirements follow from compiling the trajectory:

* The ``logp_dlogp_func`` must be **JAX-traceable** -- it must be built from
  ``jax.numpy`` ops and return ``(logp, dlogp)`` where ``logp`` is a scalar JAX
  array and ``dlogp`` is a JAX-array pytree matching ``q``'s structure. A
  function that returns ``np.asarray(...)`` (as the NumPy ``HamiltonianMC`` and
  the PyTorch/PyMC3 cookbook examples do) cannot be traced and will not work
  here.
* Only **diagonal** mass matrices are supported (a per-leaf, per-dimension
  ``var`` pytree matching ``q``).

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


tree_map = jax.tree_util.tree_map


# Dual-averaging defaults, matching littlemcmc.step_sizes.DualAverageAdaptation.
_DA_GAMMA = 0.05
_DA_T0 = 10.0
_DA_KAPPA = 0.75


def _tree_n_chains(q):
    """Read the chain count from the leading axis of any leaf of ``q``."""
    return jax.tree_util.tree_leaves(q)[0].shape[0]


def _tree_random_momentum(key, target, var):
    """Draw ``p ~ N(0, 1/var)`` as a pytree matching ``target``.

    One ``key`` is split per leaf so every leaf gets independent noise. With the
    ``QuadPotentialDiag`` convention (``velocity = var * p``), the momentum has
    per-dimension variance ``1 / var``.
    """
    leaves, treedef = jax.tree_util.tree_flatten(target)
    keys = jax.random.split(key, len(leaves))
    noise = jax.tree_util.tree_unflatten(
        treedef, [jax.random.normal(k, leaf.shape) for k, leaf in zip(keys, leaves)]
    )
    return tree_map(lambda n, v: n / jnp.sqrt(v), noise, var)


def _kinetic_energy(p, var):
    """Diagonal kinetic energy ``0.5 * sum_i var_i * p_i**2`` summed over leaves."""
    per_leaf = tree_map(lambda pp, v: jnp.sum(v * pp**2), p, var)
    return 0.5 * sum(jax.tree_util.tree_leaves(per_leaf))


def _build_vmapped_move(logp_dlogp_func, n_leapfrog, Emax):
    """Build a ``vmap``-over-chains single fixed-path-length HMC proposal.

    The returned callable maps ``(q, key, step_size, var)`` over the leading
    (chain) axis of the ``q`` pytree and ``key``; ``step_size`` and the diagonal
    mass-matrix ``var`` pytree (with ``velocity = var * p`` and ``p ~ N(0,
    1/var)``, the ``QuadPotentialDiag`` convention) are shared across chains.
    """

    def move(q, key, step_size, var):
        key_p, key_a = jax.random.split(key)
        logp0, grad0 = logp_dlogp_func(q)
        p0 = _tree_random_momentum(key_p, q, var)
        energy0 = _kinetic_energy(p0, var) - logp0

        dt = 0.5 * step_size

        def leapfrog(_, carry):
            q, p, grad = carry
            p = tree_map(lambda pp, g: pp + dt * g, p, grad)
            q = tree_map(lambda qq, pp, v: qq + step_size * (v * pp), q, p, var)
            _, grad = logp_dlogp_func(q)
            p = tree_map(lambda pp, g: pp + dt * g, p, grad)
            return (q, p, grad)

        q1, p1, _ = lax.fori_loop(0, n_leapfrog, leapfrog, (q, p0, grad0))
        logp1, _ = logp_dlogp_func(q1)
        energy1 = _kinetic_energy(p1, var) - logp1

        energy_change = energy0 - energy1
        energy_change = jnp.where(jnp.isnan(energy_change), -jnp.inf, energy_change)
        diverging = (~jnp.isfinite(energy1)) | (jnp.abs(energy_change) > Emax)

        accept_prob = jnp.minimum(1.0, jnp.exp(energy_change))
        accepted = (~diverging) & (jax.random.uniform(key_a) < accept_prob)
        q_next = tree_map(lambda qq1, qq: jnp.where(accepted, qq1, qq), q1, q)
        return q_next, accept_prob, diverging

    return jax.vmap(move, in_axes=(0, 0, None, None))


def _build_phase(move_all_chains, target_accept, gamma, t0, kappa):
    """Build a JIT-compiled ``lax.scan`` that advances all chains over draws.

    The returned ``run_phase(q0, key, var, n_steps, init_logstep, adapt)`` returns
    ``(q_final, positions, accept_prob, diverging, logstep_bar)`` where ``positions``
    is a pytree matching ``q0`` with each leaf prefixed by an ``n_steps`` axis
    (shape ``(n_steps, n_chains, *event)``). When ``adapt`` is True a single step
    size shared across chains is tuned by dual averaging on the mean accept
    statistic; ``logstep_bar`` is the averaged log step to carry into the next
    phase. ``n_steps`` and ``adapt`` are static so each combination compiles once.
    """

    def run_phase(q0, key, var, n_steps, init_logstep, adapt):
        mu_da = jnp.log(10.0) + init_logstep
        n_chains = _tree_n_chains(q0)

        def body(carry, _):
            q, key, logstep, logstep_bar, hbar, t = carry
            key, sub = jax.random.split(key)
            chain_keys = jax.random.split(sub, n_chains)
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


def _broadcast_start(start, chains):
    """Coerce ``start`` to a pytree whose leaves lead with a ``chains`` axis.

    A leaf that already leads with ``chains`` is used as-is; otherwise it is
    broadcast to ``(chains, *leaf.shape)`` so a single shared starting point is
    replicated across chains.
    """

    def broadcast(leaf):
        leaf = jnp.asarray(leaf)
        if leaf.shape[:1] == (chains,):
            return leaf
        return jnp.broadcast_to(leaf, (chains,) + leaf.shape)

    return tree_map(broadcast, start)


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

    The position can be any JAX pytree of arrays. The kernel operates leafwise via
    ``tree_map``; a plain ``(model_ndim,)`` array is the single-leaf special case.

    Parameters
    ----------
    logp_dlogp_func : callable
        JAX-traceable ``q -> (logp, dlogp)`` where ``logp`` is a scalar JAX array
        and ``dlogp`` is a JAX-array pytree matching ``q`` (built from
        ``jax.numpy`` ops, not ``np.asarray``), as required by the JAX backend.
    model_ndim : int
        Number of parameters in the flat-array default case (length of ``q``).
        Used only when ``start`` is None; ignored for pytree inputs, which must
        be supplied via ``start``.
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
    start : pytree, optional
        Initial positions. Either a plain array of shape ``(chains, model_ndim)``
        or ``(model_ndim,)`` (broadcast across chains), or a pytree whose leaves
        each lead with a ``chains`` axis (or are broadcast to one). Defaults to
        standard-normal draws of shape ``(chains, model_ndim)``.

    Returns
    -------
    trace : pytree of np.ndarray
        Posterior draws, a pytree matching the position with each leaf of shape
        ``[chains, draws, *event]``. For the flat-array default this is a single
        array of shape ``[chains, draws, model_ndim]``.
    stats : dict
        ``acceptance_rate`` and ``diverging`` (each ``[chains, draws]``), plus the
        frozen ``step_size`` (float) and ``mass_matrix_inv`` (a pytree matching
        the position).
    """
    move_all_chains = _build_vmapped_move(logp_dlogp_func, int(n_leapfrog), Emax)
    run_phase = _build_phase(move_all_chains, target_accept, _DA_GAMMA, _DA_T0, _DA_KAPPA)

    key = jax.random.PRNGKey(random_seed)
    key, key_init = jax.random.split(key)
    if start is None:
        q0 = jax.random.normal(key_init, (chains, model_ndim))
    else:
        q0 = _broadcast_start(start, chains)

    first_leaf = jax.tree_util.tree_leaves(q0)[0]
    init_logstep = jnp.log(jnp.asarray(init_step, dtype=first_leaf.dtype))
    # Diagonal mass matrix as a pytree of per-chain (event-shaped) ones.
    var = tree_map(lambda leaf: jnp.ones(leaf.shape[1:], dtype=leaf.dtype), q0)
    logstep_bar = init_logstep

    if tune > 0:
        n_w1 = tune // 2 if adapt_mass else tune
        key, sub = jax.random.split(key)
        q0, w1_pos, _, _, logstep_bar = run_phase(q0, sub, var, n_w1, init_logstep, True)
        if adapt_mass and tune - n_w1 > 0:
            # Estimate the diagonal mass matrix from the second half of window 1.
            var = tree_map(
                lambda pos: jnp.clip(jnp.var(pos[n_w1 // 2 :], axis=(0, 1)), 1e-8, None),
                w1_pos,
            )
            key, sub = jax.random.split(key)
            q0, _, _, _, logstep_bar = run_phase(q0, sub, var, tune - n_w1, logstep_bar, True)

    key, sub = jax.random.split(key)
    _, positions, accept_prob, diverging, _ = run_phase(q0, sub, var, draws, logstep_bar, False)

    # positions leaves: (draws, chains, *event) -> ArviZ-friendly (chains, draws, *event).
    trace = tree_map(lambda pos: np.asarray(pos).swapaxes(0, 1), positions)
    stats = {
        "acceptance_rate": np.asarray(accept_prob).T,  # (chains, draws)
        "diverging": np.asarray(diverging).T.astype(bool),
        "step_size": float(jnp.exp(logstep_bar)),
        "mass_matrix_inv": tree_map(np.asarray, var),
    }
    return trace, stats
