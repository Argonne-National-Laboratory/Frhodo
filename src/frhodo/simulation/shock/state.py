"""Runtime engine state — non-persisted models and label translators."""
from typing import Literal

from pydantic import BaseModel, ConfigDict


def zero_d_mode_from_label(label: str) -> Literal["constant_volume", "constant_pressure"]:
    """Translate the GUI's 0-D reactor display label to the engine enum.

    Raises:
        ValueError: If ``label`` is not a 0-D reactor display label.
    """
    if label == "0d Reactor - Constant Volume":
        return "constant_volume"
    if label == "0d Reactor - Constant Pressure":
        return "constant_pressure"
    raise ValueError(f"not a 0-D reactor label: {label!r}")


class RuntimeReactorState(BaseModel):
    """Per-call reactor configuration consumed by the simulation pipeline.

    Attributes:
        name: Reactor type label (e.g. ``"Incident Shock Reactor"``).
        t_end: Integration end time in seconds.
        t_unit_conv: Multiplier from input-time units to seconds (the
            GUI exposes ``µs`` by default → ``1e-6``).
        sim_interp_factor: Trajectory output interpolation factor.
        ode_solver: ``"CVODES"`` (SUNDIALS) or one of ``"BDF"``,
            ``"Radau"``, ``"LSODA"`` (scipy).
        ode_rtol: ODE relative tolerance.
        ode_atol: ODE absolute tolerance (scalar; the reactor expands
            this into a per-component vector internally).
        solve_energy: 0-D reactor only — whether to integrate the
            energy equation.
        frozen_comp: 0-D reactor only — whether to freeze composition.
    """
    name: str = "Incident Shock Reactor"
    t_end: float = 12.0e-6
    t_unit_conv: float = 1.0e-6
    sim_interp_factor: int = 1
    ode_solver: str = "CVODES"
    ode_rtol: float = 1e-4
    ode_atol: float = 1e-8
    solve_energy: bool = True
    frozen_comp: bool = False

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
