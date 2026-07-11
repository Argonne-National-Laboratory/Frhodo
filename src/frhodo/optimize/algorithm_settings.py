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
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)



ALGORITHM_LABELS: dict[str, int | str] = {
    "DIRECT": nlopt.GN_DIRECT,
    "DIRECT-L": nlopt.GN_DIRECT_L,
    "CRS2 (Controlled Random Search)": nlopt.GN_CRS2_LM,
    "DE (Differential Evolution)": "pygmo_DE",
    "SaDE (Self-Adaptive DE)": "pygmo_SaDE",
    "PSO (Particle Swarm Optimization)": "pygmo_PSO",
    "GWO (Grey Wolf Optimizer)": "pygmo_GWO",
    "RBFOpt": "RBFOpt",
    "Smurf (Sensitivity Multistart Rate Fitting)": "smurf",
    "Smurf (quick, low fidelity)": "smurf_lowfi",
    "Nelder-Mead Simplex": nlopt.LN_NELDERMEAD,
    "Subplex": nlopt.LN_SBPLX,
    "Subplex (whitened)": "whitened_sbplx",
    "Subplex (field basis)": "field_sbplx",
    "Subplex (quick, field basis, multi-fidelity)": "quick_filter",
    "BOBYQA (whitened)": "whitened_bobyqa",
    "COBYLA": nlopt.LN_COBYLA,
    "BOBYQA": nlopt.LN_BOBYQA,
    "IPOPT (Interior Point Optimizer)": "pygmo_IPOPT",
}

StopCriteria = Literal["Iteration Maximum", "Maximum Time [min]"]

MAX_ITERATION_STOP = 2**31 - 1


class AlgorithmStage(BaseModel):
    """One stage (global or local) of the two-stage optimization."""
    algorithm: str = "Subplex"
    initial_step: PositiveFloat = 0.1
    max_eval: PositiveInt = 2500
    xtol_rel: PositiveFloat = 1e-3
    ftol_rel: PositiveFloat = 1e-3
    # Population-size scale for population algorithms (CRS2/MLSL/ISRES).
    initial_population_multiplier: PositiveFloat = 1.0
    # Number of Smurf multistart descents (incumbent + Sobol starts).
    multistart_count: PositiveInt = 16
    stop_criteria: StopCriteria = "Iteration Maximum"
    stop_value: PositiveFloat = 2500.0
    enabled: bool = True
    # Seed for the stage's stochastic sampling (Smurf multistart Sobol,
    # RBFOpt initial design). 0 keeps each optimizer's own default.
    random_seed: NonNegativeInt = 0

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _iteration_stop_fits_int(self):
        """An iteration-count stop must survive the int conversion the
        nlopt/pygmo dispatchers apply."""
        if (
            self.stop_criteria == "Iteration Maximum"
            and self.stop_value > MAX_ITERATION_STOP
        ):
            raise ValueError(
                f"stop_value {self.stop_value:g} exceeds the iteration-count "
                f"limit ({MAX_ITERATION_STOP}); use stop_criteria "
                f"'Maximum Time [min]' or a smaller stop_value"
            )

        return self


class AlgorithmSettings(BaseModel):
    """Two-stage optimization settings: global search, then local refine."""
    global_stage: AlgorithmStage = Field(
        default_factory=lambda: AlgorithmStage(
            algorithm="Smurf (Sensitivity Multistart Rate Fitting)",
            initial_step=0.5, max_eval=400, stop_value=400.0,
        )
    )
    local_stage: AlgorithmStage = Field(
        default_factory=lambda: AlgorithmStage(
            algorithm="Subplex (field basis)", initial_step=0.1, xtol_rel=1e-4,
        )
    )

    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_legacy_dict(self) -> dict:
        """Return the dict shape ``frhodo.optimize.algorithms.Optimize``
        consumes (``{"global": {...}, "local": {...}}``)."""
        stages = {
            "global": _stage_to_legacy(self.global_stage),
            "local": _stage_to_legacy(self.local_stage),
        }

        return stages


def _resolve_algorithm(label: str) -> int | str:
    if label not in ALGORITHM_LABELS:
        raise ValueError(
            f"unknown optimization algorithm: {label!r}. "
            f"valid: {sorted(ALGORITHM_LABELS)}"
        )

    return ALGORITHM_LABELS[label]


def _stage_to_legacy(stage: AlgorithmStage) -> dict:
    legacy = {
        "algorithm": _resolve_algorithm(stage.algorithm),
        "initial_step": stage.initial_step,
        "max_eval": stage.max_eval,
        "xtol_rel": stage.xtol_rel,
        "ftol_rel": stage.ftol_rel,
        "initial_pop_multiplier": stage.initial_population_multiplier,
        "multistart_count": stage.multistart_count,
        "stop_criteria_type": stage.stop_criteria,
        "stop_criteria_val": stage.stop_value,
        "run": stage.enabled,
        "random_seed": stage.random_seed,
    }

    return legacy
