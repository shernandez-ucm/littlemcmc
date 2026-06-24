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

"""Benchmark: NumPy process-parallel chains vs JAX ``vmap``-parallel chains.

littlemcmc runs multiple chains by spawning one OS process per chain
(``lmc.sample(..., cores=C)``, backed by ``parallel_sampling.ParallelSampler``).
JAX offers a different kind of parallelism: ``jax.vmap`` vectorizes a single
chain's update across a batch axis, so all chains advance in lockstep inside one
JIT-compiled, on-device program -- no processes, no pickling, no Python loop per
draw.

This script measures the wall-clock throughput of both as the number of chains
grows, on a shared target: a ``D``-dimensional standard normal (identity mass
matrix is exact, so both samplers are correct and we can check recovery of
``mean = 0`` / ``std = 1``).

Caveats (this is an honest throughput comparison, not an identical-FLOPs one):

* The NumPy side is the real library path: ``HamiltonianMC`` with step-size and
  mass-matrix adaptation and a *randomized* path length per draw.
* The JAX side is a compact, hand-written HMC with a *fixed* number of leapfrog
  steps and a fixed step size -- ``vmap`` requires a uniform leapfrog trip-count
  across the batch, so a randomized path length cannot be vectorized this way.
  It uses an identity mass matrix and no adaptation.

So the two do slightly different per-draw work; the comparison is
"valid draws per second of each parallelism strategy", plus a correctness check,
not a controlled FLOP-for-FLOP race. JAX's one-off compilation time is reported
separately from steady-state run time.

Usage::

    python examples/benchmark_parallel_vs_vmap.py
    python examples/benchmark_parallel_vs_vmap.py --dim 50 --chains 2,4,8,16,32

The JAX rows are skipped if ``jax``/``jaxlib`` are not installed.
"""

import argparse
import time

import numpy as np

import littlemcmc as lmc


# --- Shared target: D-dimensional standard normal. ----------------------------
# Defined at module scope so it pickles to the spawned worker processes that the
# NumPy parallel path uses.
def numpy_logp_dlogp_func(x):
    """Standard-normal log-density (scalar) and gradient (length-D)."""
    x = np.asarray(x, dtype=np.float64)
    return float(-0.5 * np.dot(x, x)), -x


def benchmark_numpy(dim, chains, draws, tune, seed):
    """Time littlemcmc's process-parallel HMC over ``chains`` chains."""
    step = lmc.HamiltonianMC(logp_dlogp_func=numpy_logp_dlogp_func, model_ndim=dim)
    t0 = time.perf_counter()
    trace, stats = lmc.sample(
        logp_dlogp_func=numpy_logp_dlogp_func,
        model_ndim=dim,
        draws=draws,
        tune=tune,
        step=step,
        chains=chains,
        cores=chains,  # one process per chain -- the path under test
        progressbar=False,
        random_seed=seed,
    )
    elapsed = time.perf_counter() - t0
    samples = trace.reshape(-1, dim)
    accept = float(np.mean(stats["accept"]))
    return {
        "elapsed": elapsed,
        "draws_per_s": chains * draws / elapsed,
        "mean_abs_err": float(np.abs(samples.mean(axis=0)).mean()),
        "std": float(samples.std(axis=0).mean()),
        "accept": accept,
    }


def build_vmapped_hmc(dim, step_size, n_leapfrog, n_total):
    """Build a JIT-compiled, ``vmap``-over-chains fixed-path-length HMC.

    Returns ``run(q0, key) -> (positions, accept_probs)`` where ``q0`` has shape
    ``(chains, dim)`` and ``positions`` has shape ``(n_total, chains, dim)``. The
    whole chain loop runs on-device via ``lax.scan``; chains are the ``vmap``
    batch axis.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    def logp_grad(q):
        return -0.5 * jnp.dot(q, q), -q

    def leapfrog(q, p, grad):
        dt = 0.5 * step_size

        def body(_, carry):
            q, p, grad = carry
            p = p + dt * grad
            q = q + step_size * p  # identity mass matrix: velocity == momentum
            _, grad = logp_grad(q)
            p = p + dt * grad
            return (q, p, grad)

        return lax.fori_loop(0, n_leapfrog, body, (q, p, grad))

    def one_chain_step(q, key):
        key_p, key_a = jax.random.split(key)
        logp0, grad0 = logp_grad(q)
        p0 = jax.random.normal(key_p, (dim,))
        energy0 = 0.5 * jnp.dot(p0, p0) - logp0

        qn, pn, _ = leapfrog(q, p0, grad0)
        logpn, _ = logp_grad(qn)
        energyn = 0.5 * jnp.dot(pn, pn) - logpn

        accept_prob = jnp.minimum(1.0, jnp.exp(energy0 - energyn))
        accept = jax.random.uniform(key_a) < accept_prob
        return jnp.where(accept, qn, q), accept_prob

    step_all_chains = jax.vmap(one_chain_step)  # over the leading (chain) axis

    @jax.jit
    def run(q0, key):
        def scan_body(carry, _):
            q, key = carry
            key, sub = jax.random.split(key)
            chain_keys = jax.random.split(sub, q.shape[0])
            q, accept_prob = step_all_chains(q, chain_keys)
            return (q, key), (q, accept_prob)

        (_, _), (positions, accept_probs) = lax.scan(
            scan_body, (q0, key), xs=None, length=n_total
        )
        return positions, accept_probs

    return run


def benchmark_jax(dim, chains, draws, tune, seed, step_size, n_leapfrog):
    """Time the JAX ``vmap`` HMC over ``chains`` chains (compile vs run split)."""
    import jax
    import jax.numpy as jnp

    n_total = draws + tune
    run = build_vmapped_hmc(dim, step_size, n_leapfrog, n_total)

    key = jax.random.PRNGKey(seed)
    key, key_init = jax.random.split(key)
    q0 = jax.random.normal(key_init, (chains, dim))

    # First call compiles. Block to exclude JAX's async dispatch from the timing.
    key, sub = jax.random.split(key)
    t0 = time.perf_counter()
    positions, _ = run(q0, sub)
    positions.block_until_ready()
    compile_elapsed = time.perf_counter() - t0

    # Steady-state run (already compiled).
    key, sub = jax.random.split(key)
    t0 = time.perf_counter()
    positions, accept_probs = run(q0, sub)
    positions.block_until_ready()
    run_elapsed = time.perf_counter() - t0

    samples = np.asarray(positions[tune:]).reshape(-1, dim)  # drop burn-in
    return {
        "compile": compile_elapsed,
        "elapsed": run_elapsed,
        "draws_per_s": chains * draws / run_elapsed,
        "mean_abs_err": float(np.abs(samples.mean(axis=0)).mean()),
        "std": float(samples.std(axis=0).mean()),
        "accept": float(np.mean(np.asarray(accept_probs))),
    }


def has_jax():
    try:
        import jax  # noqa: F401

        return True
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=30, help="target dimensionality")
    parser.add_argument(
        "--chains", type=str, default="2,4,8,16", help="comma-separated chain counts"
    )
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--step-size", type=float, default=0.2, help="JAX HMC step size")
    parser.add_argument("--leapfrog", type=int, default=10, help="JAX HMC leapfrog steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    chain_counts = [int(c) for c in args.chains.split(",")]
    jax_available = has_jax()

    print(
        "Target: %d-D standard normal | draws=%d tune=%d | JAX step=%g, L=%d"
        % (args.dim, args.draws, args.tune, args.step_size, args.leapfrog)
    )
    print(
        "%-7s | %-26s | %-34s | %-8s"
        % ("chains", "numpy (process-parallel)", "jax (vmap-parallel)", "speedup")
    )
    print(
        "%-7s | %-9s %-7s %-7s | %-9s %-8s %-7s %-6s | %-8s"
        % ("", "time(s)", "draw/s", "std", "run(s)", "cmpl(s)", "draw/s", "std", "draw/s")
    )
    print("-" * 92)

    for chains in chain_counts:
        npr = benchmark_numpy(args.dim, chains, args.draws, args.tune, args.seed)
        if jax_available:
            jxr = benchmark_jax(
                args.dim, chains, args.draws, args.tune, args.seed, args.step_size, args.leapfrog
            )
            speedup = jxr["draws_per_s"] / npr["draws_per_s"]
            print(
                "%-7d | %-9.3f %-7.0f %-7.3f | %-9.4f %-8.3f %-7.0f %-6.3f | %-6.1fx"
                % (
                    chains,
                    npr["elapsed"],
                    npr["draws_per_s"],
                    npr["std"],
                    jxr["elapsed"],
                    jxr["compile"],
                    jxr["draws_per_s"],
                    jxr["std"],
                    speedup,
                )
            )
        else:
            print(
                "%-7d | %-9.3f %-7.0f %-7.3f | %-34s | %-8s"
                % (chains, npr["elapsed"], npr["draws_per_s"], npr["std"], "skipped (no jax)", "-")
            )

    print("-" * 92)
    print("std should be ~1.0 for both (recovering the standard normal).")
    if jax_available:
        print("jax 'cmpl(s)' is one-off compilation; 'run(s)' is steady-state for `draws+tune`.")


if __name__ == "__main__":
    main()
