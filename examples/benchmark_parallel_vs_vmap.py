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
* The JAX side is ``littlemcmc.hmc_jax.sample_vmapped_chains`` -- a vmapped
  multi-chain HMC with dual-averaging step size and diagonal mass adaptation, but
  a *fixed* number of leapfrog steps (``vmap`` requires a uniform leapfrog
  trip-count across the batch, so a randomized path length cannot be vectorized
  this way). Its reported time includes the one-off JIT compilation.

So the two do slightly different per-draw work; the comparison is
"valid draws per second of each parallelism strategy", plus a correctness check,
not a controlled FLOP-for-FLOP race.

Usage::

    python examples/benchmark_parallel_vs_vmap.py
    python examples/benchmark_parallel_vs_vmap.py --dim 50 --chains 2,4,8,16,32

The JAX rows are skipped if ``jax``/``jaxlib`` are not installed.
"""

import argparse
import multiprocessing as mp
import time

# Use "spawn" globally: this script initializes JAX (which is multithreaded) and also
# spawns worker processes for the NumPy parallel path. A fork() with live JAX threads
# can deadlock (Python 3.12 warns about exactly this), so start fresh interpreters
# instead. Must run before importing JAX or littlemcmc.
mp.set_start_method("spawn", force=True)

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


def benchmark_jax(dim, chains, draws, tune, seed, n_leapfrog):
    """Time ``sample_vmapped_chains`` over ``chains`` chains (incl. JIT compile)."""
    import jax.numpy as jnp

    from littlemcmc.hmc_jax import sample_vmapped_chains

    def jax_logp_dlogp_func(x):
        """Standard-normal log-density and gradient as JAX arrays."""
        return -0.5 * jnp.dot(x, x), -x

    # The returned trace is a NumPy array (a device->host copy), so the timed call
    # already blocks on completion; no explicit block_until_ready is needed.
    t0 = time.perf_counter()
    trace, stats = sample_vmapped_chains(
        jax_logp_dlogp_func,
        dim,
        draws=draws,
        tune=tune,
        chains=chains,
        n_leapfrog=n_leapfrog,
        random_seed=seed,
    )
    elapsed = time.perf_counter() - t0  # includes one-off JIT compilation

    samples = trace.reshape(-1, dim)
    return {
        "elapsed": elapsed,
        "draws_per_s": chains * draws / elapsed,
        "mean_abs_err": float(np.abs(samples.mean(axis=0)).mean()),
        "std": float(samples.std(axis=0).mean()),
        "accept": float(np.mean(stats["acceptance_rate"])),
    }


def has_jax():
    import importlib.util

    return importlib.util.find_spec("jax") is not None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=30, help="target dimensionality")
    parser.add_argument(
        "--chains", type=str, default="2,4,8,16", help="comma-separated chain counts"
    )
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--leapfrog", type=int, default=10, help="JAX HMC leapfrog steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    chain_counts = [int(c) for c in args.chains.split(",")]
    jax_available = has_jax()

    print(
        "Target: %d-D standard normal | draws=%d tune=%d | JAX L=%d"
        % (args.dim, args.draws, args.tune, args.leapfrog)
    )
    print(
        "%-7s | %-26s | %-26s | %-8s"
        % ("chains", "numpy (process-parallel)", "jax (vmap-parallel)", "speedup")
    )
    print(
        "%-7s | %-9s %-7s %-7s | %-9s %-7s %-7s | %-8s"
        % ("", "time(s)", "draw/s", "std", "time(s)", "draw/s", "std", "draw/s")
    )
    print("-" * 84)

    for chains in chain_counts:
        npr = benchmark_numpy(args.dim, chains, args.draws, args.tune, args.seed)
        if jax_available:
            jxr = benchmark_jax(args.dim, chains, args.draws, args.tune, args.seed, args.leapfrog)
            speedup = jxr["draws_per_s"] / npr["draws_per_s"]
            print(
                "%-7d | %-9.3f %-7.0f %-7.3f | %-9.3f %-7.0f %-7.3f | %-6.1fx"
                % (
                    chains,
                    npr["elapsed"],
                    npr["draws_per_s"],
                    npr["std"],
                    jxr["elapsed"],
                    jxr["draws_per_s"],
                    jxr["std"],
                    speedup,
                )
            )
        else:
            print(
                "%-7d | %-9.3f %-7.0f %-7.3f | %-26s | %-8s"
                % (chains, npr["elapsed"], npr["draws_per_s"], npr["std"], "skipped (no jax)", "-")
            )

    print("-" * 84)
    print("std should be ~1.0 for both (recovering the standard normal).")
    if jax_available:
        print("jax 'time(s)' includes one-off JIT compilation and the warmup/tuning phase.")


if __name__ == "__main__":
    main()
