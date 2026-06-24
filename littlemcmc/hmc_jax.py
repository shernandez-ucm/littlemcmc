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

"""JIT-compiled Hamiltonian Monte Carlo trajectory using JAX.

This module backs ``HamiltonianMC(..., backend="jax")``. The whole leapfrog
trajectory and the Metropolis accept/reject are fused into a single
``jax.jit``-compiled function, so the per-draw hot path runs as compiled XLA
(optionally on GPU/TPU) instead of dispatching one NumPy op at a time.

Two requirements follow from compiling the trajectory:

* The ``logp_dlogp_func`` must be **JAX-traceable** -- it must be built from
  ``jax.numpy`` ops and return JAX arrays. A function that returns
  ``np.asarray(...)`` (as the NumPy backend and the PyTorch/PyMC3 cookbook
  examples do) cannot be traced and will not work here.
* Only **diagonal** mass matrices are supported. This covers the default
  ``QuadPotentialDiagAdapt`` and ``QuadPotentialDiag``; full-matrix potentials
  raise ``NotImplementedError`` (use the NumPy backend for those).

Divergence handling differs from the NumPy integrator: rather than catching a
``scipy.linalg`` ``IntegrationError``, the compiled trajectory lets non-finite
energies propagate and flags a divergence when the final energy is non-finite
or the energy change exceeds ``Emax`` -- the same criteria the NumPy path
applies after the fact.
"""

from typing import Callable, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
except ImportError as err:  # pragma: no cover
    raise ImportError(
        "HamiltonianMC(backend='jax') requires JAX. Install it with "
        "`pip install jax jaxlib`."
    ) from err


def diagonal_of(potential) -> np.ndarray:
    """Return the mass-matrix diagonal of a diagonal quadpotential.

    The leapfrog only needs the diagonal ``var`` such that ``velocity(p) =
    var * p``. ``QuadPotentialDiagAdapt`` stores it as ``_var`` and
    ``QuadPotentialDiag`` as ``v``. Anything else is unsupported by the JAX
    backend.
    """
    if hasattr(potential, "_var"):
        return potential._var
    if hasattr(potential, "v"):
        return potential.v
    raise NotImplementedError(
        "The JAX backend for HamiltonianMC only supports diagonal mass "
        "matrices (QuadPotentialDiagAdapt / QuadPotentialDiag), got "
        "{}.".format(type(potential).__name__)
    )


def build_trajectory(
    logp_dlogp_func: Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]
):
    """Build a JIT-compiled HMC trajectory closing over ``logp_dlogp_func``.

    The returned callable takes only arrays/scalars so that ``step_size``, the
    mass-matrix ``var``, and ``n_steps`` can vary across draws (during step-size
    and mass-matrix adaptation) without triggering recompilation.
    """

    def trajectory(
        q0, p0, q_grad0, logp0, start_energy, step_size, var, n_steps, Emax, accept_unif
    ):
        # The position arrives as float64 (NumPy default) while the momentum and
        # mass-matrix diagonal arrive as float32 (the quadpotential's default
        # dtype). Promote everything to a single dtype so the leapfrog carry is
        # type-stable -- `lax.fori_loop` requires identical input/output types.
        dtype = jnp.result_type(q0, p0, q_grad0, var)
        q0 = q0.astype(dtype)
        p0 = p0.astype(dtype)
        q_grad0 = q_grad0.astype(dtype)
        logp0 = logp0.astype(dtype)
        var = var.astype(dtype)
        step_size = step_size.astype(dtype)
        start_energy = start_energy.astype(dtype)

        dt = 0.5 * step_size

        def leapfrog(_, carry):
            q, p, q_grad, _logp = carry
            # Half momentum step, full position step, half momentum step --
            # identical to integration.CpuLeapfrogIntegrator._step.
            p = p + dt * q_grad
            q = q + step_size * (var * p)
            logp, q_grad = logp_dlogp_func(q)
            p = p + dt * q_grad
            return (q, p, q_grad, logp)

        q, p, q_grad, logp = lax.fori_loop(0, n_steps, leapfrog, (q0, p0, q_grad0, logp0))

        kinetic = 0.5 * jnp.dot(p, var * p)
        energy = kinetic - logp

        energy_change = start_energy - energy
        energy_change = jnp.where(jnp.isnan(energy_change), -jnp.inf, energy_change)

        diverging = (~jnp.isfinite(energy)) | (jnp.abs(energy_change) > Emax)
        accept_stat = jnp.minimum(1.0, jnp.exp(energy_change))
        accepted = (~diverging) & (accept_unif < accept_stat)

        # Proposal is the integrated state if accepted, else the start state.
        # `energy` / `logp` below are always the *integrated* values, matching
        # the stats reported by the NumPy backend.
        q_end = jnp.where(accepted, q, q0)
        p_end = jnp.where(accepted, p, p0)
        grad_end = jnp.where(accepted, q_grad, q_grad0)

        return (
            q_end,
            p_end,
            var * p_end,
            grad_end,
            energy,
            logp,
            energy_change,
            accept_stat,
            accepted,
            diverging,
        )

    return jax.jit(trajectory)
