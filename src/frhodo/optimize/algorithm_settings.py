"""Typed algorithm-stage configuration for :func:`optimize_residual`.

Two stages: a global-search stage followed by a local-refinement stage.
Each is independently enabled and parameterized. Algorithm names use
the same labels as the GUI; :meth:`AlgorithmSettings.to_legacy_dict`
resolves them to the integer codes / sentinel strings the dispatcher
in :mod:`frhodo.optimize.algorithms` expects.
"""
from __future__ import annotations

from typing import Literal

import nlopt
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt


ALGORITHM_LABELS: dict[str, int | str] = {
    "DIRECT": nlopt.GN_DIRECT,
    "DIRECT-L": nlopt.GN_DIRECT_L,
    "CRS2 (Controlled Random Search)": nlopt.GN_CRS2_LM,
    "DE (Differential Evolution)": "pygmo_DE",
    "SaDE (Self-Adaptive DE)": "pygmo_SaDE",
    "PSO (Particle Swarm Optimization)": "pygmo_PSO",
    "GWO (Grey Wolf Optimizer)": "pygmo_GWO",
    "RBFOpt": "RBFOpt",
    "Nelder-Mead Simplex": nlopt.LN_NELDERMEAD,
    "Subplex": nlopt.LN_SBPLX,
    "COBYLA": nlopt.LN_COBYLA,
    "BOBYQA": nlopt.LN_BOBYQA,
    "IPOPT (Interior Point Optimizer)": "pygmo_IPOPT",
}

StopCriteria = Literal["Iteration Maximum", "Maximum Time [min]"]


class AlgorithmStage(BaseModel):
    """One stage (global or local) of the two-stage optimization."""
    algorithm: str = "Subplex"
    initial_step: PositiveFloat = 0.1
    max_eval: PositiveInt = 2500
    xtol_rel: PositiveFloat = 1e-3
    ftol_rel: PositiveFloat = 1e-3
    initial_population_multiplier: PositiveFloat = 1.0
    stop_criteria: StopCriteria = "Iteration Maximum"
    stop_value: PositiveFloat = 2500.0
    enabled: bool = True

    model_config = ConfigDict(extra="forbid", frozen=True)


class AlgorithmSettings(BaseModel):
    """Two-stage optimization settings: global search, then local refine."""
    global_stage: AlgorithmStage = Field(
        default_factory=lambda: AlgorithmStage(
            algorithm="RBFOpt", initial_step=0.5,
        )
    )
    local_stage: AlgorithmStage = Field(
        default_factory=lambda: AlgorithmStage(
            algorithm="Subplex", initial_step=0.1, xtol_rel=1e-4,
        )
    )

    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_legacy_dict(self) -> dict:
        """Return the dict shape ``frhodo.optimize.algorithms.Optimize``
        consumes (``{"global": {...}, "local": {...}}``)."""
        return {
            "global": _stage_to_legacy(self.global_stage),
            "local": _stage_to_legacy(self.local_stage),
        }


def _resolve_algorithm(label: str) -> int | str:
    if label not in ALGORITHM_LABELS:
        raise ValueError(
            f"unknown optimization algorithm: {label!r}. "
            f"valid: {sorted(ALGORITHM_LABELS)}"
        )

    return ALGORITHM_LABELS[label]


def _stage_to_legacy(stage: AlgorithmStage) -> dict:
    return {
        "algorithm": _resolve_algorithm(stage.algorithm),
        "initial_step": stage.initial_step,
        "max_eval": stage.max_eval,
        "xtol_rel": stage.xtol_rel,
        "ftol_rel": stage.ftol_rel,
        "initial_pop_multiplier": stage.initial_population_multiplier,
        "stop_criteria_type": stage.stop_criteria,
        "stop_criteria_val": stage.stop_value,
        "run": stage.enabled,
    }
