"""Typed per-process pool-worker state.

A single module-level ``WorkerContext`` slot — required because
``mp.Pool.map`` task functions cannot be partial-applied with
non-picklable state.
"""
from dataclasses import dataclass
from typing import Optional

from frhodo.simulation.mechanism.mech_fcns import ChemicalMechanism


@dataclass(frozen=True)
class MechBuildPayload:
    """Inputs for ``ChemicalMechanism.set_mechanism`` in a fresh worker."""
    reset_mech: list
    thermo_coeffs: list
    coeffs: list
    coeffs_bnds: list
    rate_bnds: list


@dataclass
class WorkerContext:
    """Per-process worker state.

    ``mech`` holds a loaded mechanism (fork inheritance or post-load
    cache). ``mech_path`` holds a path string used by spawn workers to
    load their own copy at init time.
    """
    mech: Optional[ChemicalMechanism] = None
    mech_path: Optional[str] = None
