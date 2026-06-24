# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

LittleMCMC is a standalone, dependency-light implementation of Hamiltonian Monte Carlo (HMC)
and the No-U-Turn Sampler (NUTS), carved out of PyMC3/PyMC4's HMC step methods. The whole point
is to let users bring their own log-probability function (and its gradient) and run PyMC's
battle-tested samplers without buying into the entire PyMC ecosystem.

The user-facing contract is small: you call `littlemcmc.sample(logp_dlogp_func, model_ndim, ...)`,
where `logp_dlogp_func(x: np.ndarray) -> Tuple[logp: float, dlogp: np.ndarray]`. `sample` returns
`(trace, stats)` where `trace` has shape `[chains, samples, model_ndim]` and `stats` is a dict of
per-sample sampler statistics. Because the user supplies the gradient themselves, the model can be
written in *any* autodiff framework (JAX, PyTorch, Theano, etc.) — see `docs/_static/scripts/` and
`docs/tutorials/framework_cookbook.rst` for examples.

> Project status: this is a maintenance-mode fork. The original is at `eigenfoo/littlemcmc`.
> Behavior is unstable in Jupyter notebooks; develop and test against plain Python scripts.

## Commands

All common workflows are in the `Makefile`:

- `make venv` / `make conda` — create a dev environment (installs `requirements.txt`,
  `requirements-dev.txt`, and `pip install -e .`).
- `make test` — run the full pytest suite (includes `--doctest-modules`, so docstring examples
  are executed as tests, plus coverage).
- `make lint` — runs all four checks: `blackstyle` (black --check), `pylintstyle` (pylint),
  `pydocstyle` (numpy convention, excludes `parallel_sampling.py`), `mypytypes` (mypy).
- `make black` — auto-format in place.
- `make check` — `lint` then `test`.
- `make package` / `make clean` — build for PyPI / clean artifacts.

Run a single test directly with pytest, e.g.:

```bash
pytest -v tests/test_sampling.py
pytest -v tests/test_sampling.py::test_sample   # single test
```

CI (`.github/workflows/`) runs `tests.yml` and `lint.yml`. A separate `even-with-pymc3.yml`
workflow runs `scripts/check-for-pymc3-commits.sh`, which fails if upstream PyMC3's HMC/NUTS
code changed — a signal that this fork may need to be re-synced from upstream.

## Architecture

The code mirrors PyMC3's `step_methods/hmc/` layout, split into composable modules. When porting
fixes from upstream PyMC3, expect a near-1:1 file correspondence.

- `sampling.py` — the driver (`sample`, `init_nuts`). Owns chain/core orchestration, initialization
  methods (`jitter+adapt_diag`, `adapt_full`, etc.), and the reshaping of raw per-chain output into
  the public `(trace, stats)` arrays. **Unrelated to PyMC3's `sampling.py`** despite the name.
- `base_hmc.py` — `BaseHMC`, the shared base class. Holds the integrator, the quadpotential (mass
  matrix), and step-size adaptation; defines the `_hamiltonian_step` hook that subclasses implement.
- `hmc.py` — `HamiltonianMC`, fixed-path-length HMC (`_hamiltonian_step`). Constructed with
  `backend="numpy"` (default) or `backend="jax"`; the latter dispatches the trajectory to
  `hmc_jax.py`.
- `hmc_jax.py` — optional JIT-compiled HMC trajectory. `build_trajectory()` fuses the leapfrog loop
  (`lax.fori_loop`) and Metropolis accept/reject into one `jax.jit` function. Requires a
  JAX-traceable `logp_dlogp_func` (returns JAX arrays, *not* `np.asarray`) and a diagonal mass
  matrix. JAX is a lazy/optional import — only `backend="jax"` pulls it in, so the rest of the
  package (and NUTS) stay numpy-only. Randomness (path length, accept threshold) is drawn with
  `np.random` *outside* the compiled region, so the trajectory stays a pure function and existing
  seeding is preserved.
- `nuts.py` — `NUTS`, the default step method. `_Tree` implements the recursive trajectory doubling
  and U-turn termination.
- `integration.py` — `CpuLeapfrogIntegrator`, the leapfrog symplectic integrator. Raises
  `IntegrationError` on non-finite energies (divergences).
- `quadpotential.py` — mass-matrix / kinetic-energy implementations and their online adaptation:
  `QuadPotentialDiag(Adapt)`, `QuadPotentialFull(Inv/Adapt)`. `quad_potential()` is the factory.
- `step_sizes.py` — `DualAverageAdaptation`, dual-averaging step-size tuning targeting an accept rate.
- `parallel_sampling.py` — multiprocess chain execution (`ParallelSampler`, `ProcessAdapter`).
  Uses the `spawn` context (set as `littlemcmc.ctx` in `__init__.py`). Pickling logp functions across
  processes is the main constraint here (`pickle_backend`). Exempt from pydocstyle.
- `report.py` / `exceptions.py` — `SamplerWarning`/`WarningType` and `SamplingError`. The sampler
  surfaces divergences, low acceptance, bad energy, etc. as warnings rather than failing silently.
- `math.py` — small numerically-stable log-space helpers used by NUTS.

## Conventions

- Python 3.6/3.7 era code; type hints are checked by mypy (`--ignore-missing-imports`).
- Formatting is `black` and docstrings follow the **numpy** convention — `make lint` enforces both,
  so run it before considering a change done.
- Docstring code examples are run as tests via `--doctest-modules`; keep them correct and runnable.
- This is a fork of Apache-2.0 PyMC code. Preserve the existing license headers on source files.
