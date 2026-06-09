"""Pydantic models for the Frhodo user-saved configuration (``default_config.yaml``).

The persisted config holds user *preferences*: which ODE solver, what
optimization algorithm, which display units, etc. Runtime simulation
parameters (T_reac, P_reac, composition, ...) are assembled separately
from the experiment and selected shock — see ``frhodo.api`` for those.

Loading an unrecognised ``schema_version`` raises ``SchemaVersionError``;
any other validation failure raises ``pydantic.ValidationError``.
"""
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt

from frhodo.common.errors import SchemaVersionError



class TemperatureUnits(BaseModel):
    zone_1: str = "K"
    zone_2: str = "K"
    zone_5: str = "K"


class PressureUnits(BaseModel):
    zone_1: str = "torr"
    zone_2: str = "torr"
    zone_5: str = "atm"


class ExperimentSettings(BaseModel):
    temperature_units: TemperatureUnits = Field(default_factory=TemperatureUnits)
    pressure_units: PressureUnits = Field(default_factory=PressureUnits)
    velocity_units: str = "m/s"


class SimulationEndTime(BaseModel):
    value: float = 12.0
    units: str = "μs"


class ODETolerance(BaseModel):
    relative: float = 1e-4
    absolute: float = 1e-8


class ReactorSettings(BaseModel):
    type: str = "Incident Shock Reactor"
    solve_energy: bool = True
    frozen_composition: bool = False
    simulation_end_time: SimulationEndTime = Field(default_factory=SimulationEndTime)
    ode_solver: Literal["CVODES", "BDF", "LSODA", "Radau"] = "CVODES"
    simulation_interpolation_factor: int = 1
    ode_tolerance: ODETolerance = Field(default_factory=ODETolerance)


class WeightFunction(BaseModel):
    max: float = 100.0
    min: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    time_location: list[float] = Field(default_factory=lambda: [4.5, 35.0])
    inverse_growth_rate: list[float] = Field(default_factory=lambda: [0.0, 0.7])


class OptimizationSettings(BaseModel):
    time_uncertainty: float = 0.0
    random_t_uncertainty: bool = True
    objective_function_type: Literal["Residual", "Bayesian"] = "Residual"
    objective_function_scale: Literal["Linear", "Log", "AbsoluteLog", "Bisymlog"] = "Linear"
    loss_function_alpha: str | float = "Adaptive"
    loss_function_c: float = 1.0
    bayesian_distribution_type: str = "Automatic"
    bayesian_uncertainty_sigma: float = 3.0
    multiprocessing: bool = True
    enabled: dict[str, bool] = Field(
        default_factory=lambda: {"global": True, "local": True}
    )
    algorithm: dict[str, str] = Field(
        default_factory=lambda: {"global": "RBFOpt", "local": "Subplex"}
    )
    initial_step: dict[str, float] = Field(
        default_factory=lambda: {"global": 5.0e-1, "local": 1.0e-1}
    )
    stop_criteria_type: dict[str, str] = Field(
        default_factory=lambda: {
            "global": "Iteration Maximum",
            "local": "Iteration Maximum",
        }
    )
    stop_criteria_value: dict[str, float] = Field(
        default_factory=lambda: {"global": 2500.0, "local": 2500.0}
    )
    relative_x_tolerance: dict[str, float] = Field(
        default_factory=lambda: {"global": 1.0e-3, "local": 1.0e-4}
    )
    relative_fcn_tolerance: dict[str, float] = Field(
        default_factory=lambda: {"global": 5.0e-2, "local": 1.0e-3}
    )
    initial_population_multiplier: dict[str, float] = Field(
        default_factory=lambda: {"global": 1.0}
    )
    weight_function: WeightFunction = Field(default_factory=WeightFunction)


class DirectorySettings(BaseModel):
    directory_file: str = ""


class PlotSettings(BaseModel):
    x_scale: str = "linear"
    y_scale: str = "abslog"


class SessionSettings(BaseModel):
    """Whole-session save/load behavior."""
    autosnapshot_enabled: bool = True
    snapshot_interval_s: PositiveFloat = 30.0
    last_session_file: str = ""


class FrhodoConfig(BaseModel):
    """Root model for ``default_config.yaml``.

    Unknown top-level fields are ignored on load; an unsupported
    ``schema_version`` raises :class:`SchemaVersionError`.
    """

    schema_version: Literal[2] = 2
    directory: DirectorySettings = Field(default_factory=DirectorySettings)
    experiment: ExperimentSettings = Field(default_factory=ExperimentSettings)
    reactor: ReactorSettings = Field(default_factory=ReactorSettings)
    optimization: OptimizationSettings = Field(default_factory=OptimizationSettings)
    plot: PlotSettings = Field(default_factory=PlotSettings)
    session: SessionSettings = Field(default_factory=SessionSettings)

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    @classmethod
    def from_yaml_text(cls, text: str) -> "FrhodoConfig":
        """Parse YAML config text into a validated model.

        Raises:
            SchemaVersionError: When the loaded document carries an
                unsupported ``schema_version``.
            pydantic.ValidationError: For any other validation failure.
        """
        data = yaml.safe_load(text) or {}
        version = data.get("schema_version")
        if version is not None and version != 2:
            raise SchemaVersionError(
                f"schema v{version} is not supported in Frhodo 2.0; "
                "delete `default_config.yaml` and defaults will be "
                "regenerated on next save."
            )

        return cls.model_validate(data)

    def to_yaml_text(self) -> str:
        """Serialize the config to YAML text in canonical order."""
        text = yaml.safe_dump(
            self.model_dump(mode="json"),
            sort_keys=False,
            allow_unicode=True,
        )

        return text


Composition = dict[str, float]


class PreShockState(BaseModel):
    """Measured pre-shock (zone 1) state. Triggers auto-jump in
    :func:`frhodo.api.run_shock_tube` so callers don't have to invoke
    the jump solver separately.
    """
    kind: Literal["pre_shock"] = "pre_shock"
    T1: PositiveFloat
    P1: PositiveFloat
    u1: PositiveFloat
    composition: Composition

    model_config = ConfigDict(extra="forbid", frozen=True)


class PostShockState(BaseModel):
    """Resolved post-shock reactor-inlet (zone 2 or zone 5) state.

    Use when the jump conditions have already been computed externally;
    the reactor simulation starts here directly.
    """
    kind: Literal["post_shock"] = "post_shock"
    T_reac: PositiveFloat
    P_reac: PositiveFloat
    u_incident: PositiveFloat
    rho1: PositiveFloat
    composition: Composition

    model_config = ConfigDict(extra="forbid", frozen=True)


ShockTubeInitial = PreShockState | PostShockState


class SolverSettings(BaseModel):
    """Time-integration knobs shared by shock-tube and 0-D reactors."""
    solver: Literal["CVODES", "BDF", "LSODA", "Radau"] = "CVODES"
    rtol: PositiveFloat | None = None
    atol: PositiveFloat | None = None
    sim_interp_factor: PositiveInt = 1

    model_config = ConfigDict(extra="forbid", frozen=True)


class ObservableSettings(BaseModel):
    """Which trace the reactor returns as ``SimulationResult.observable``."""
    main: str = "Density Gradient"
    sub: int | str = 0

    model_config = ConfigDict(extra="forbid", frozen=True)


class ShockTubeConfig(BaseModel):
    """Runtime parameters for one Incident-Shock-Reactor simulation.

    ``initial`` is a discriminated union: :class:`PreShockState` triggers
    an auto-jump to zone 2 before the reactor runs; :class:`PostShockState`
    starts directly from the supplied reactor-inlet conditions.
    """

    initial: ShockTubeInitial = Field(discriminator="kind")
    t_end: PositiveFloat = 1e-3
    A1: PositiveFloat = 0.2
    As: PositiveFloat = 0.2
    L: PositiveFloat = 0.1
    area_change: bool = False
    solver: SolverSettings = Field(default_factory=SolverSettings)
    observable: ObservableSettings = Field(default_factory=ObservableSettings)
    save_times: list[float] | None = None

    model_config = ConfigDict(extra="forbid", frozen=True,
                              arbitrary_types_allowed=True)


_ZERO_D_DEFAULT_SOLVER = SolverSettings(rtol=1e-4, atol=1e-7)
_ZERO_D_DEFAULT_OBSERVABLE = ObservableSettings(main="Concentration", sub=0)


class ZeroDConfig(BaseModel):
    """Runtime parameters for a 0-D constant-volume or constant-pressure reactor."""

    mode: Literal["constant_volume", "constant_pressure"]
    T_reac: PositiveFloat
    P_reac: PositiveFloat
    composition: dict[str, float]
    t_end: PositiveFloat
    solve_energy: bool = True
    frozen_comp: bool = False
    solver: SolverSettings = Field(default_factory=lambda: _ZERO_D_DEFAULT_SOLVER)
    observable: ObservableSettings = Field(default_factory=lambda: _ZERO_D_DEFAULT_OBSERVABLE)
    save_times: list[float] | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


