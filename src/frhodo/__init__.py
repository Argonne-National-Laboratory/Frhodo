"""Frhodo simulation engine — public package surface.

All names re-exported here form the supported public API. Importing
from sub-modules is supported but not part of the contract.
"""
import logging
from importlib.metadata import PackageNotFoundError, version


logging.getLogger(__name__).addHandler(logging.NullHandler())

from frhodo.api import (
    AlgorithmSettings,
    AlgorithmStage,
    ChemicalMechanism,
    CoefUncertainty,
    CostSettings,
    ExperimentShock,
    IterationUpdate,
    ObservableSettings,
    OptimizableRate,
    OptimizableSet,
    OptimizableSpec,
    OptimizableSpecBuilder,
    OptimizationCallbacks,
    OptimizationRequest,
    OptimizationResult,
    PostShockState,
    PreShockState,
    RateUncertainty,
    ShockState,
    ShockTubeConfig,
    SimulationResult,
    SolverSettings,
    StageComplete,
    StartInfo,
    WeightProfile,
    ZeroDConfig,
    apply_optimization_result,
    kJ_per_mol,
    kcal_per_mol,
    load_mechanism,
    optimize_residual,
    parse_composition,
    run_shock_tube,
    run_shock_tubes,
    run_zero_d,
    solve_shock_jump,
)


try:
    __version__ = version("frhodo")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"


__all__ = [
    "AlgorithmSettings",
    "AlgorithmStage",
    "ChemicalMechanism",
    "CoefUncertainty",
    "CostSettings",
    "ExperimentShock",
    "IterationUpdate",
    "ObservableSettings",
    "OptimizableRate",
    "OptimizableSet",
    "OptimizableSpec",
    "OptimizableSpecBuilder",
    "OptimizationCallbacks",
    "OptimizationRequest",
    "OptimizationResult",
    "PostShockState",
    "PreShockState",
    "RateUncertainty",
    "ShockState",
    "ShockTubeConfig",
    "SimulationResult",
    "SolverSettings",
    "StageComplete",
    "StartInfo",
    "WeightProfile",
    "ZeroDConfig",
    "__version__",
    "apply_optimization_result",
    "kJ_per_mol",
    "kcal_per_mol",
    "load_mechanism",
    "optimize_residual",
    "parse_composition",
    "run_shock_tube",
    "run_shock_tubes",
    "run_zero_d",
    "solve_shock_jump",
]
