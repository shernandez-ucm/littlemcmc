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

"""Vmapped multi-chain No-U-Turn Sampler using JAX.

This module is the NUTS counterpart of :mod:`littlemcmc.hmc_jax`. It provides
:func:`sample_vmapped_nuts_chains`, a self-contained multi-chain NUTS that runs
*all* chains as the batch axis of ``jax.vmap`` -- every chain advances in
lockstep inside one ``jax.jit``-compiled program (``lax.scan`` over draws,
``lax.while_loop`` over the recursive trajectory doubling), with no OS
processes, no pickling, and no per-draw Python loop. Like ``hmc_jax`` it is the
GPU/TPU-friendly alternative to ``littlemcmc.sample``'s process-per-chain
parallelism, but it replaces ``hmc_jax``'s fixed leapfrog path length with the
adaptive No-U-Turn termination of "Algorithm 6" of Hoffman & Gelman (2014),
using the iterative tree-doubling and generalized U-turn criterion of Betancourt
(2017) as implemented in NumPyro.

It shares ``hmc_jax``'s conventions exactly and reuses its helpers
(:func:`~littlemcmc.hmc_jax._tree_random_momentum`,
:func:`~littlemcmc.hmc_jax._kinetic_energy`,
:func:`~littlemcmc.hmc_jax._build_phase`,
:func:`~littlemcmc.hmc_jax._broadcast_start`):

* The position ``q`` may be any JAX **pytree**; the leapfrog and the U-turn
  checks are written entirely with ``jax.tree_util.tree_map``, so a plain array
  is just the single-leaf case.
* The ``logp_dlogp_func`` must be **JAX-traceable** -- built from ``jax.numpy``
  ops and returning ``(logp, dlogp)`` where ``logp`` is a scalar JAX array and
  ``dlogp`` is a JAX-array pytree matching ``q``. ``np.asarray``-returning
  functions (as the NumPy ``HamiltonianMC`` and the PyTorch/PyMC3 cookbook
  examples use) cannot be traced and will not work here.
* Only **diagonal** mass matrices are supported (a per-leaf, per-dimension
  ``var`` pytree matching ``q``, with the ``QuadPotentialDiag`` convention
  ``velocity = var * p`` and ``p ~ N(0, 1/var)``).

Divergence handling: the compiled trajectory lets non-finite energies propagate
and flags a divergence when a leaf's energy change is non-finite or exceeds
``Emax``; divergent subtrees are never selected as the next proposal.
"""

from typing import Any, NamedTuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
except ImportError as err:  # pragma: no cover
    raise ImportError(
        "littlemcmc.nuts_jax requires JAX. Install it with `pip install jax jaxlib`."
    ) from err

from .hmc_jax import (
    _DA_GAMMA,
    _DA_KAPPA,
    _DA_T0,
    _broadcast_start,
    _build_phase,
    _kinetic_energy,
    _tree_random_momentum,
)

tree_map = jax.tree_util.tree_map


def _tree_add(a, b):
    """Leafwise ``a + b`` for two matching pytrees."""
    return tree_map(jnp.add, a, b)


def _tree_sub(a, b):
    """Leafwise ``a - b`` for two matching pytrees."""
    return tree_map(jnp.subtract, a, b)


def _velocity(p, var):
    """Diagonal velocity ``var * p`` (the ``QuadPotentialDiag`` convention)."""
    return tree_map(jnp.multiply, var, p)


def _dot(a, b):
    """Euclidean inner product of two matching pytrees, summed over all leaves."""
    per_leaf = tree_map(lambda x, y: jnp.sum(x * y), a, b)
    return sum(jax.tree_util.tree_leaves(per_leaf))


def _where_tree(pred, a, b):
    """Leafwise ``jnp.where(pred, a, b)``; ``pred`` is a scalar broadcast per leaf."""
    return tree_map(lambda x, y: jnp.where(pred, x, y), a, b)


class _Tree(NamedTuple):
    """A (sub)tree of the NUTS trajectory, mirroring NumPyro's ``TreeInfo``.

    Carries the two boundary integrator states (``*_left`` / ``*_right``), the
    currently selected ``z_proposal``, and the running statistics needed to
    combine subtrees: the log ``weight`` (a log-sum-exp of leaf weights
    ``-delta_energy``), the momentum sum ``r_sum`` used by the U-turn check, the
    ``turning`` / ``diverging`` flags, and the accept-probability sum / proposal
    count used for the reported acceptance statistic.
    """

    z_left: Any
    r_left: Any
    grad_left: Any
    z_right: Any
    r_right: Any
    grad_right: Any
    z_proposal: Any
    depth: Any
    weight: Any
    r_sum: Any
    turning: Any
    diverging: Any
    sum_accept_probs: Any
    num_proposals: Any


def _leapfrog_step(logp_dlogp_func, z, r, grad, signed_step, var):
    """One velocity-Verlet leapfrog step, leafwise via ``tree_map``.

    ``signed_step`` carries the trajectory direction (positive to extend right,
    negative to extend left). Returns ``(z, r, grad, logp)`` at the new state.
    """
    half = 0.5 * signed_step
    r = tree_map(lambda rr, g: rr + half * g, r, grad)
    z = tree_map(lambda zz, rr, v: zz + signed_step * (v * rr), z, r, var)
    logp, grad = logp_dlogp_func(z)
    r = tree_map(lambda rr, g: rr + half * g, r, grad)
    return z, r, grad, logp


def _build_basetree(
    logp_dlogp_func, z, r, grad, var, step_size, going_right, energy_current, Emax
):
    """Take one leapfrog step and wrap the new state as a depth-0 ``_Tree``."""
    signed_step = jnp.where(going_right, step_size, -step_size)
    z_new, r_new, grad_new, logp_new = _leapfrog_step(
        logp_dlogp_func, z, r, grad, signed_step, var
    )
    energy_new = _kinetic_energy(r_new, var) - logp_new
    delta_energy = energy_new - energy_current
    delta_energy = jnp.where(jnp.isnan(delta_energy), jnp.inf, delta_energy)
    weight = -delta_energy
    diverging = delta_energy > Emax
    accept_prob = jnp.minimum(1.0, jnp.exp(-delta_energy))
    return _Tree(
        z_new,
        r_new,
        grad_new,
        z_new,
        r_new,
        grad_new,
        z_new,
        jnp.array(0, jnp.int32),
        weight,
        r_new,
        jnp.array(False),
        diverging,
        accept_prob,
        jnp.array(1, jnp.int32),
    )


def _is_turning(r_left, r_right, r_sum, var):
    """Generalized U-turn criterion (Betancourt 2017, section A.4.2).

    Returns True if the trajectory subtended by momenta ``r_left`` / ``r_right``
    (summed momentum ``r_sum``) is turning at either end.
    """
    r_mid = tree_map(lambda x: 0.5 * x, _tree_add(r_left, r_right))
    r_adj = _tree_sub(r_sum, r_mid)
    left_angle = _dot(_velocity(r_left, var), r_adj)
    right_angle = _dot(_velocity(r_right, var), r_adj)
    return (left_angle <= 0) | (right_angle <= 0)


def _popcount(x):
    """Vectorized 32-bit population count (number of set bits)."""
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    return ((x * 0x01010101) >> 24) & 0x3F


def _leaf_idx_to_ckpt_idxs(n):
    """Range of checkpoint slots to U-turn-check when adding leaf index ``n``.

    Mirrors NumPyro's bit trick: ``idx_max`` is the number of set bits of ``n``
    above the last bit, and the number of contiguous trailing set bits gives how
    many balanced subtrees complete at ``n`` (hence ``idx_min``).
    """
    n = n.astype(jnp.int32)
    idx_max = _popcount(n >> 1)
    num_subtrees = _popcount((~n & (n + 1)) - 1)
    idx_min = idx_max - num_subtrees + 1
    return idx_min, idx_max


def _is_iterative_turning(r, r_sum, r_ckpts, r_sum_ckpts, idx_min, idx_max, var):
    """U-turn-check the new leaf against every completed balanced subtree.

    ``r_ckpts`` / ``r_sum_ckpts`` are pytrees whose leaves lead with a
    checkpoint axis. For each checkpoint ``i`` in ``[idx_min, idx_max]`` the
    subtree momentum sum is reconstructed and tested against the new leaf.
    """

    def _body_fn(state):
        i, _ = state
        r_ckpt = tree_map(lambda a: a[i], r_ckpts)
        r_sum_ckpt = tree_map(lambda a: a[i], r_sum_ckpts)
        subtree_r_sum = _tree_add(_tree_sub(r_sum, r_sum_ckpt), r_ckpt)
        return i - 1, _is_turning(r_ckpt, r, subtree_r_sum, var)

    _, turning = lax.while_loop(
        lambda it: (it[0] >= idx_min) & (~it[1]), _body_fn, (idx_max, jnp.array(False))
    )
    return turning


def _combine_tree(current_tree, new_tree, var, going_right, rng_key, biased_transition):
    """Merge ``new_tree`` into ``current_tree`` in the doubling direction.

    The outer boundaries are taken from whichever tree is on each side. The
    proposal is resampled with a biased kernel between main trees (favouring the
    fresh half) and a uniform kernel within a subtree, matching NumPyro.
    """
    z_left, r_left, grad_left = _where_tree(
        going_right,
        (current_tree.z_left, current_tree.r_left, current_tree.grad_left),
        (new_tree.z_left, new_tree.r_left, new_tree.grad_left),
    )
    z_right, r_right, grad_right = _where_tree(
        going_right,
        (new_tree.z_right, new_tree.r_right, new_tree.grad_right),
        (current_tree.z_right, current_tree.r_right, current_tree.grad_right),
    )
    r_sum = _tree_add(current_tree.r_sum, new_tree.r_sum)
    weight = jnp.logaddexp(current_tree.weight, new_tree.weight)

    if biased_transition:
        transition_prob = jnp.exp(new_tree.weight - current_tree.weight)
        transition_prob = jnp.where(
            new_tree.turning | new_tree.diverging,
            0.0,
            jnp.minimum(1.0, transition_prob),
        )
        turning = new_tree.turning | _is_turning(r_left, r_right, r_sum, var)
    else:
        transition_prob = jax.nn.sigmoid(new_tree.weight - current_tree.weight)
        turning = current_tree.turning

    transition = jax.random.bernoulli(rng_key, transition_prob)
    z_proposal = _where_tree(transition, new_tree.z_proposal, current_tree.z_proposal)

    return _Tree(
        z_left,
        r_left,
        grad_left,
        z_right,
        r_right,
        grad_right,
        z_proposal,
        current_tree.depth + 1,
        weight,
        r_sum,
        turning,
        new_tree.diverging,
        current_tree.sum_accept_probs + new_tree.sum_accept_probs,
        current_tree.num_proposals + new_tree.num_proposals,
    )


def _get_leaf(tree, going_right):
    """The boundary integrator state of ``tree`` in the doubling direction."""
    z = _where_tree(going_right, tree.z_right, tree.z_left)
    r = _where_tree(going_right, tree.r_right, tree.r_left)
    grad = _where_tree(going_right, tree.grad_right, tree.grad_left)
    return z, r, grad


def _iterative_build_subtree(
    current_tree,
    logp_dlogp_func,
    var,
    step_size,
    going_right,
    rng_key,
    energy_current,
    Emax,
    r_ckpts,
    r_sum_ckpts,
):
    """Grow a subtree of ``2 ** current_tree.depth`` leaves, one leaf at a time.

    Leaves are added in the doubling direction with ``lax.while_loop``, stopping
    early on a U-turn or divergence; momentum checkpoints are maintained so every
    completed balanced subtree is U-turn-checked as it closes.
    """
    max_num_proposals = jnp.left_shift(jnp.array(1, jnp.int32), current_tree.depth)

    def _cond_fn(state):
        tree, turning, _, _, _ = state
        return (tree.num_proposals < max_num_proposals) & (~turning) & (~tree.diverging)

    def _body_fn(state):
        cur_tree, _, r_ckpts, r_sum_ckpts, rng_key = state
        rng_key, transition_key = jax.random.split(rng_key)
        z, r, grad = _get_leaf(cur_tree, going_right)
        new_leaf = _build_basetree(
            logp_dlogp_func, z, r, grad, var, step_size, going_right, energy_current, Emax
        )
        new_tree = lax.cond(
            cur_tree.num_proposals == 0,
            lambda _: new_leaf,
            lambda _: _combine_tree(
                cur_tree, new_leaf, var, going_right, transition_key, False
            ),
            operand=None,
        )

        leaf_idx = cur_tree.num_proposals
        ckpt_idx_min, ckpt_idx_max = _leaf_idx_to_ckpt_idxs(leaf_idx)

        def _store(ckpts):
            r_ckpts, r_sum_ckpts = ckpts
            r_ckpts = tree_map(
                lambda a, v: a.at[ckpt_idx_max].set(v), r_ckpts, new_leaf.r_right
            )
            r_sum_ckpts = tree_map(
                lambda a, v: a.at[ckpt_idx_max].set(v), r_sum_ckpts, new_tree.r_sum
            )
            return r_ckpts, r_sum_ckpts

        # We only need to checkpoint a subtree's left endpoint (even leaf index).
        r_ckpts, r_sum_ckpts = lax.cond(
            leaf_idx % 2 == 0, _store, lambda ckpts: ckpts, (r_ckpts, r_sum_ckpts)
        )
        turning = _is_iterative_turning(
            new_leaf.r_right,
            new_tree.r_sum,
            r_ckpts,
            r_sum_ckpts,
            ckpt_idx_min,
            ckpt_idx_max,
            var,
        )
        return new_tree, turning, r_ckpts, r_sum_ckpts, rng_key

    basetree = current_tree._replace(num_proposals=jnp.array(0, jnp.int32))
    tree, turning, _, _, _ = lax.while_loop(
        _cond_fn, _body_fn, (basetree, jnp.array(False), r_ckpts, r_sum_ckpts, rng_key)
    )
    return tree._replace(depth=current_tree.depth, turning=turning)


def _double_tree(
    current_tree,
    logp_dlogp_func,
    var,
    step_size,
    going_right,
    rng_key,
    energy_current,
    Emax,
    r_ckpts,
    r_sum_ckpts,
):
    """Double ``current_tree``: build a same-size subtree and merge it in."""
    key, transition_key = jax.random.split(rng_key)
    new_tree = _iterative_build_subtree(
        current_tree,
        logp_dlogp_func,
        var,
        step_size,
        going_right,
        key,
        energy_current,
        Emax,
        r_ckpts,
        r_sum_ckpts,
    )
    return _combine_tree(
        current_tree, new_tree, var, going_right, transition_key, True
    )


def _build_tree(logp_dlogp_func, z, r, logp, grad, var, step_size, rng_key, max_treedepth, Emax):
    """Build a full NUTS trajectory by recursive doubling from one start state.

    Returns the final ``_Tree``; ``z_proposal`` is the next sample, and
    ``sum_accept_probs / num_proposals`` is the reported acceptance statistic.
    """
    energy_current = _kinetic_energy(r, var) - logp
    r_ckpts = tree_map(
        lambda leaf: jnp.zeros((max_treedepth,) + leaf.shape, leaf.dtype), r
    )
    r_sum_ckpts = tree_map(
        lambda leaf: jnp.zeros((max_treedepth,) + leaf.shape, leaf.dtype), r
    )

    tree = _Tree(
        z,
        r,
        grad,
        z,
        r,
        grad,
        z,
        jnp.array(0, jnp.int32),
        jnp.zeros(()),
        r,
        jnp.array(False),
        jnp.array(False),
        jnp.zeros(()),
        jnp.array(0, jnp.int32),
    )

    def _cond_fn(state):
        tree, _ = state
        return (tree.depth < max_treedepth) & (~tree.turning) & (~tree.diverging)

    def _body_fn(state):
        tree, rng_key = state
        rng_key, direction_key, doubling_key = jax.random.split(rng_key, 3)
        going_right = jax.random.bernoulli(direction_key)
        tree = _double_tree(
            tree,
            logp_dlogp_func,
            var,
            step_size,
            going_right,
            doubling_key,
            energy_current,
            Emax,
            r_ckpts,
            r_sum_ckpts,
        )
        return tree, rng_key

    tree, _ = lax.while_loop(_cond_fn, _body_fn, (tree, rng_key))
    return tree


def _build_vmapped_nuts_move(logp_dlogp_func, max_treedepth, Emax):
    """Build a ``vmap``-over-chains single NUTS transition.

    The returned callable maps ``(q, key, step_size, var)`` over the leading
    (chain) axis of the ``q`` pytree and ``key``; ``step_size`` and the diagonal
    mass-matrix ``var`` pytree are shared across chains. It returns
    ``(q_next, accept_prob, diverging)``, matching ``hmc_jax``'s move so the same
    ``_build_phase`` dual-averaging driver can advance it.
    """

    def move(q, key, step_size, var):
        key_p, key_tree = jax.random.split(key)
        logp0, grad0 = logp_dlogp_func(q)
        p0 = _tree_random_momentum(key_p, q, var)
        tree = _build_tree(
            logp_dlogp_func, q, p0, logp0, grad0, var, step_size, key_tree, max_treedepth, Emax
        )
        accept_prob = tree.sum_accept_probs / tree.num_proposals
        return tree.z_proposal, accept_prob, tree.diverging

    return jax.vmap(move, in_axes=(0, 0, None, None))


def sample_vmapped_nuts_chains(
    logp_dlogp_func,
    model_ndim,
    draws=1000,
    tune=1000,
    chains=4,
    max_treedepth=10,
    target_accept=0.8,
    init_step=0.1,
    Emax=1000.0,
    adapt_mass=True,
    random_seed=0,
    start=None,
):
    """Sample multiple chains with a vmapped, fully on-device NUTS.

    All ``chains`` advance together as the ``jax.vmap`` batch axis inside one
    JIT-compiled ``lax.scan``. Each draw builds an adaptive No-U-Turn trajectory
    (recursive doubling, generalized U-turn termination) instead of a fixed
    leapfrog path. Warmup tunes a single (shared) step size by dual averaging
    and, if ``adapt_mass`` is True, a diagonal mass matrix estimated from the
    warmup draws; both are then frozen for the sampling phase.

    The position can be any JAX pytree of arrays. The kernel operates leafwise
    via ``tree_map``; a plain ``(model_ndim,)`` array is the single-leaf case.

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
    max_treedepth : int
        Maximum NUTS tree depth (a trajectory has at most ``2 ** max_treedepth``
        leapfrog steps before being forced to stop).
    target_accept : float
        Target mean acceptance for dual-averaging step-size adaptation.
    init_step : float
        Initial leapfrog step size before adaptation.
    Emax : float
        Energy-change threshold above which a leaf is flagged divergent.
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
    move_all_chains = _build_vmapped_nuts_move(logp_dlogp_func, int(max_treedepth), Emax)
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
