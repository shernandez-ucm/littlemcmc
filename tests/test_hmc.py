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

import numpy as np
import numpy.testing as npt
import pytest

import littlemcmc as lmc
from littlemcmc import HamiltonianMC
from test_utils import logp_dlogp_func


def test_leapfrog_reversible():
    np.random.seed(42)
    model_ndim = 1
    scaling = np.random.rand(model_ndim)
    step = HamiltonianMC(logp_dlogp_func=logp_dlogp_func, model_ndim=model_ndim, scaling=scaling)
    p = step.potential.random()
    q = np.random.randn(model_ndim)
    start = step.integrator.compute_state(p, q)

    for epsilon in [0.01, 0.1]:
        for n_steps in [1, 2, 3, 4, 20]:
            state = start
            for _ in range(n_steps):
                state = step.integrator.step(epsilon, state)
            for _ in range(n_steps):
                state = step.integrator.step(-epsilon, state)
            npt.assert_allclose(state.q, start.q, rtol=1e-5)
            npt.assert_allclose(state.p, start.p, rtol=1e-5)


def test_array_valued_logp_is_coerced_to_scalar():
    # Regression test: a logp_dlogp_func that returns a shape-(1,) log-density
    # must not make `energy` array-valued. Previously this propagated into an
    # array-valued `accept_stat` and `step_size`, crashing HMC on the 2nd draw.
    def array_logp_dlogp_func(x):
        return np.atleast_1d(-0.5 * np.dot(x, x)), -x

    step = HamiltonianMC(logp_dlogp_func=array_logp_dlogp_func, model_ndim=2)
    state = step.integrator.compute_state(np.zeros(2), np.ones(2))
    assert np.ndim(state.energy) == 0
    assert np.ndim(state.model_logp) == 0

    state = step.integrator.step(0.1, state)
    assert np.ndim(state.energy) == 0

    # And a full HMC run with array-valued logp completes and recovers N(0, I).
    trace, _ = lmc.sample(
        array_logp_dlogp_func, 2, draws=500, tune=500, step=step, chains=1, cores=1
    )
    npt.assert_allclose(trace.mean(axis=(0, 1)), np.zeros(2), atol=0.3)
    npt.assert_allclose(trace.std(axis=(0, 1)), np.ones(2), atol=0.3)


def test_nuts_tuning():
    model_ndim = 1
    draws = 5
    tune = 5
    step = lmc.NUTS(logp_dlogp_func=logp_dlogp_func, model_ndim=model_ndim)
    chains = 1
    cores = 1
    trace, stats = lmc.sample(
        logp_dlogp_func, model_ndim, draws, tune, step=step, chains=chains, cores=cores
    )

    assert not step.tune
    # FIXME revisit this test once trace object has been stabilized.
    # assert np.all(trace['step_size'][5:] == trace['step_size'][5])


def test_sample_vmapped_chains():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from littlemcmc.hmc_jax import sample_vmapped_chains

    def jax_logp_dlogp_func(x):
        # Standard normal: logp = -0.5 * sum(x ** 2), grad = -x.
        return -0.5 * jnp.sum(x**2), -x

    model_ndim = 2
    draws = 500
    tune = 500
    chains = 2
    trace, stats = sample_vmapped_chains(
        jax_logp_dlogp_func,
        model_ndim,
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=42,
    )

    assert trace.shape == (chains, draws, model_ndim)
    assert np.all(np.isfinite(trace))
    assert stats["acceptance_rate"].shape == (chains, draws)
    assert stats["diverging"].shape == (chains, draws)
    # Recovered posterior should be roughly standard normal.
    npt.assert_allclose(trace.mean(axis=(0, 1)), np.zeros(model_ndim), atol=0.2)
    npt.assert_allclose(trace.std(axis=(0, 1)), np.ones(model_ndim), atol=0.3)
