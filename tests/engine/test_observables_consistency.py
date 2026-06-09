"""Enforce that the two parallel observable registries stay in sync.

Frhodo carries two pieces of observable information that must match:

1. :data:`frhodo.simulation.shock.reactor_output.REACTOR_VARS` carries the
   ``observable_default: bool`` flag on each entry, used to build
   ``BY_DISPLAY_OBSERVABLE`` (the GUI dropdown).
2. :data:`frhodo.simulation.shock.observables.OBSERVABLES` is the
   sensitivity-pipeline registry.

They sit in separate modules because consolidating into one would
create a circular import (observables needs reactor physics; the
reactor file needs the output registry from reactor_output;
reactor_output would need observables). Two registries, one test:
drift caught the moment any new observable lands in only one of
the two places.

The drhodz formula has two implementations — trajectory-shape in
reactor_output.py (for ReactorOutput post-hoc display) and
single-state in observables.py (for SUNDIALS sensitivity callbacks).
The second test below evaluates both at the same gas state and
asserts they agree.
"""
import pathlib
import shutil

import cantera as ct
import numpy as np
import pytest

from frhodo.simulation.shock.observables import (
    OBSERVABLES, drhodz_at_state, drhodz_per_rxn_at_state,
)
from frhodo.simulation.shock.reactor_output import REACTOR_VARS, drhodz, drhodz_per_rxn


def test_observable_default_matches_OBSERVABLES_keys():
    """Every ``OBSERVABLES`` entry has a matching ``observable_default=True``
    ``ReactorVar``, and no extras on either side.
    """
    flagged = {v.sim_name for v in REACTOR_VARS if v.observable_default}
    registered = set(OBSERVABLES)
    assert flagged == registered, (
        f"observable_default vs OBSERVABLES drift detected:\n"
        f"  in OBSERVABLES but not flagged: {sorted(registered - flagged)}\n"
        f"  flagged but not in OBSERVABLES: {sorted(flagged - registered)}"
    )


def test_drhodz_trajectory_and_at_state_agree(loaded_cycloheptane):
    """``drhodz(states)`` and ``drhodz_at_state(gas, …)`` are two
    implementations of the same formula at different shapes. Evaluate
    both at the same gas state and assert agreement to machine
    precision (modulo the SolutionArray-vs-Solution path's float
    accumulation order)."""
    gas = loaded_cycloheptane.gas
    gas.TPX = 1500.0, 20000.0, {"cC7H14": 0.04, "Kr": 0.96}

    T, rho, Y = gas.T, gas.density, gas.Y.copy()
    v = 250.0

    # Trajectory-shape: a SolutionArray of length 1 at the same state.
    states = ct.SolutionArray(
        gas, extra=["t", "t_shock", "z", "A", "vel",
                    "drhodz_tot", "drhodz", "perc_drhodz"],
    )
    states.append(
        TDY=(T, rho, Y),
        t=0.0, t_shock=0.0, z=0.0, A=0.2, vel=v,
        drhodz_tot=np.nan, drhodz=np.nan, perc_drhodz=np.nan,
    )

    traj_drhodz = drhodz(states)[0]
    point_drhodz = drhodz_at_state(gas, T, rho, Y, v, area_change_term=0.0)
    np.testing.assert_allclose(traj_drhodz, point_drhodz, rtol=1e-12, atol=0.0)

    # Per-reaction variant: evaluate both, compare elementwise.
    # ``drhodz_per_rxn`` returns shape (1, n_rxns); per-state returns (n_rxns,).
    gas.TDY = T, rho, Y
    traj_per_rxn = drhodz_per_rxn(states)[0]
    point_per_rxn = drhodz_per_rxn_at_state(gas, T, rho, Y, v, area_change_term=0.0)
    np.testing.assert_allclose(traj_per_rxn, point_per_rxn, rtol=1e-12, atol=0.0)
