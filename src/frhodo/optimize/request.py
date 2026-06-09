"""Typed input bundle for :func:`frhodo.api.optimize_residual`.

Consolidates the kitchen-sink kwarg list into one ``OptimizationRequest``
pydantic model so the call site is one named argument and validation
happens up front.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveInt,
)

from frhodo.common.config import ObservableSettings
from frhodo.experiment.profiles import (
    ExperimentShock,
    WeightProfile,
)
from frhodo.optimize.algorithm_settings import AlgorithmSettings
from frhodo.optimize.cost.settings import CostSettings
from frhodo.optimize.spec import OptimizableSpec
from frhodo.simulation.shock.state import RuntimeReactorState


class OptimizationRequest(BaseModel):
    """All inputs to one :func:`frhodo.api.optimize_residual` call.

    Attributes:
        shocks: Experiment shocks to fit against.
        optimizable: Target reactions + coefficients + bounds.
        reactor_state: Per-shock reactor configuration.
        cost: Cost-function settings (residual vs Bayesian, scale).
        algorithm: Optimizer choice and stop criteria.
        observable: Observable selection (shared across shocks).
        default_weight_profile: Applied to any shock that lacks one.
            (Uncertainty is data-derived via wavelet σ — no profile needed.)
        time_uncertainty: Allowed time-shift between sim and experiment.
        random_t_uncertainty: When ``True``, each shock's time shift is
            optimized independently per cost evaluation (random
            per-shock noise). When ``False``, a single parametric model
            in ``(T, P, reactant composition)`` is fit globally across
            all shocks. Defaults to ``True`` to preserve current
            per-shock behavior.
        multiprocessing: Enable parallel cost evaluation.
        max_processors: Worker-pool size when ``multiprocessing=True``.
        display_shock_index: Index of the shock whose trace the GUI
            should plot live, or ``None`` to disable live plotting.
        save_recast_path: When set, after the run the recast (Troe)
            mechanism is written here.
    """
    shocks: list[ExperimentShock]
    optimizable: OptimizableSpec
    reactor_state: RuntimeReactorState
    cost: CostSettings
    algorithm: AlgorithmSettings = Field(default_factory=AlgorithmSettings)
    observable: ObservableSettings = Field(default_factory=ObservableSettings)
    default_weight_profile: WeightProfile = Field(default_factory=WeightProfile)
    time_uncertainty: NonNegativeFloat = 0.0
    random_t_uncertainty: bool = True
    multiprocessing: bool = False
    max_processors: PositiveInt = 1
    display_shock_index: int | None = None
    save_recast_path: Path | None = None

    model_config = ConfigDict(extra="forbid", frozen=True,
                              arbitrary_types_allowed=True)

