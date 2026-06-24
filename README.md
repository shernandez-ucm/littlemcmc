<p align="center"><img src="docs/_static/logo/default-cropped.png"></p>

> *Warning:* `littlemcmc`'s behavior is unstable in Jupyter notebooks - for best
> results and support, please use `littlemcmc` in Python scripts. Furthermore,
> despite best efforts, `littlemcmc` is not guaranteed to be up-to-date with the
> [current PyMC3 HMC/NUTS
> samplers](https://github.com/pymc-devs/pymc3/tree/master/pymc3/step_methods/hmc) -
> please consult [our GitHub
> issues](https://github.com/shernandez-ucm/littlemcmc/issues).

---

![Tests Status](https://github.com/shernandez-ucm/littlemcmc/workflows/tests/badge.svg)
![Lint Status](https://github.com/shernandez-ucm/littlemcmc/workflows/lint/badge.svg)
![Up to date with PyMC3 Status](https://github.com/shernandez-ucm/littlemcmc/workflows/even-with-pymc3/badge.svg)
[![Coverage Status](https://codecov.io/gh/shernandez-ucm/littlemcmc/branch/master/graph/badge.svg)](https://codecov.io/gh/shernandez-ucm/littlemcmc)
[![Documentation Status](https://readthedocs.org/projects/littlemcmc/badge/?version=latest)](https://littlemcmc.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/shernandez-ucm/littlemcmc)](https://github.com/shernandez-ucm/littlemcmc/blob/master/LICENSE.txt)

> littlemcmc &nbsp; &nbsp; /lɪtəl ɛm si ɛm si/ &nbsp; &nbsp; _noun_
>
> A lightweight and performant implementation of HMC and NUTS in Python, spun
> out of [the PyMC project](https://github.com/pymc-devs). Not to be confused
> with [minimc](https://github.com/ColCarroll/minimc).

## Installation

The latest release of LittleMCMC can be installed from PyPI using `pip`:

```bash
pip install littlemcmc
```

The current development branch of LittleMCMC can be installed directly from
GitHub, also using `pip`:

```bash
pip install git+https://github.com/shernandez-ucm/littlemcmc.git
```

## What's new in this fork

This fork extends the upstream
[`eigenfoo/littlemcmc`](https://github.com/eigenfoo/littlemcmc) with a
JIT-compiled JAX backend for HMC and several correctness fixes.

### JAX backend for Hamiltonian Monte Carlo

`HamiltonianMC` takes a `backend` argument. With `backend="jax"`, the entire
leapfrog trajectory and the Metropolis accept/reject are fused into a single
`jax.jit`-compiled function (optionally running on GPU/TPU):

```python
import jax.numpy as jnp
import littlemcmc as lmc

def logp_dlogp_func(x):
    return -0.5 * jnp.sum(x ** 2), -x   # must be JAX-traceable

step = lmc.HamiltonianMC(logp_dlogp_func=logp_dlogp_func, model_ndim=2, backend="jax")
trace, stats = lmc.sample(logp_dlogp_func, model_ndim=2, step=step)
```

The JAX backend requires `jax`/`jaxlib`, a `logp_dlogp_func` that is
JAX-traceable (built from `jax.numpy` and returning JAX arrays — *not* wrapped in
`np.asarray`), and a diagonal mass matrix. The default `backend="numpy"` is
unchanged, so existing NumPy / PyTorch / PyMC3 log-probability functions keep
working as before.

### Bug fixes

- **Parallel sampling.** With `cores > 1`, each worker process now writes its
  draws into the shared-memory buffer the main process reads. Previously every
  chain stayed frozen at its starting point.
- **Array-valued log-probabilities.** A `logp_dlogp_func` returning a
  shape-`(1,)` log-density no longer corrupts step-size adaptation in
  `HamiltonianMC` (the joint log-probability is now coerced to a scalar).

### Examples

See the [`examples/`](examples/) directory:

- `gaussian_mixture_samplers.py` — samples a 2D Gaussian mixture with NumPy HMC,
  JAX HMC, and NUTS, comparing posterior recovery and runtime.
- `benchmark_parallel_vs_vmap.py` — benchmarks NumPy process-parallel chains
  against JAX `vmap`-parallel chains.

> **Jupyter note:** multi-chain sampling defaults to multiprocessing, which is
> unreliable in notebooks because the `spawn` start method cannot re-import
> functions defined in interactive cells. In notebooks, pass `cores=1` to
> `lmc.sample`, or define your `logp_dlogp_func` in an importable `.py` module.

## Contributors

LittleMCMC was originally developed by [George Ho](https://eigenfoo.xyz/). This
fork is maintained by [shernandez-ucm](https://github.com/shernandez-ucm). For a
full list of contributors, please see the [GitHub contributor
graph](https://github.com/shernandez-ucm/littlemcmc/graphs/contributors).

## License

LittleMCMC is modified from [the PyMC3 and PyMC4
projects](https://github.com/pymc-devs/), both of which are distributed under
the Apache-2.0 license. A copy of both projects' license files are distributed
with LittleMCMC. All modifications from PyMC are distributed under [an identical
Apache-2.0 license](https://github.com/shernandez-ucm/littlemcmc/blob/master/LICENSE.txt).
