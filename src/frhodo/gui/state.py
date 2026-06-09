"""Typed widget-state models.

Each model owns a focused slice of GUI state with explicit field
semantics.
"""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LoadState(BaseModel):
    """Per-session load-completion flags.

    ``mech_loaded`` gates downstream readers (run-single, the mech
    tree's rebuilds, etc.) until ``load_mech`` finishes. The
    ``load_full_series`` flag mirrors the user's checkbox; the
    optimizer's multi-shock path requires it.
    """

    mech_loaded: bool = Field(
        default=False,
        description="True once the current mechanism has finished loading.",
    )
    load_full_series: bool = Field(
        default=False,
        description="True when 'Load Full Series Into Memory' is checked.",
    )

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RunControlState(BaseModel):
    """Single-run lifecycle and parallelism control.

    ``run_block`` is the gate that lets ``Main.__init__`` finish before
    any ``run_single`` calls go through. ``optimize_running`` /
    ``abort`` are the long-running optimization's lifecycle flags.
    """

    run_block: bool = Field(
        default=True,
        description="When True, ``run_single`` is a no-op (init not yet finished).",
    )
    optimize_running: bool = Field(
        default=False,
        description="True between ``start_threads`` and ``on_worker_done``.",
    )
    abort: bool = Field(
        default=False,
        description="Set by the abort button; the optimization worker polls this.",
    )
    multiprocessing: bool = Field(
        default=True,
        description="Multi-process the optimizer's per-shock cost evaluations.",
    )
    max_processors: int = Field(
        default=1,
        description="Worker-pool size when ``multiprocessing`` is True.",
    )

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ShockSelectionState(BaseModel):
    """Currently-displayed shock plus the previous selection.

    Step direction (``current - previous``) is read by the path-pulldown
    sync code when the chosen shock is missing.
    """

    current: int = Field(
        default=1,
        description="Currently displayed shock number (1-indexed).",
    )
    previous: int = Field(
        default=1,
        description="Last displayed shock number; ``current - previous`` gives step direction.",
    )

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TimeUncertaintyState(BaseModel):
    """Time-uncertainty and time-offset values driven by the GUI boxes.

    Stored in seconds (post unit conversion); the optimizer reads
    ``value`` to size its bounds.
    """

    value: float = Field(
        default=0.0,
        description="Time uncertainty bound in seconds; symmetric around the experiment time vector.",
    )
    offset: float = Field(
        default=0.0,
        description="Time offset in seconds applied to the experiment trace before fitting.",
    )
    random: bool = Field(
        default=True,
        description="When True, the optimizer solves each shock's time shift independently per cost call. When False, fits a single parametric model in (T, P, reactant composition) across all shocks.",
    )
    auto_fit: bool = Field(
        default=False,
        description="When True, the displayed shock's time offset is auto-refit every time the simulation updates. Toggled via the right-click menu on the time-offset box; cleared automatically when any user input changes the offset.",
    )

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SaveDialogState(BaseModel):
    """User selections in the Save dialog.

    ``species`` and ``reactions`` are indexed by row order in the source
    list widgets; the keys map back to the underlying mech indices when
    the file is written.
    """

    comment: str = Field(
        default="",
        description="Free-form note attached to the save.",
    )
    output_time: Any = Field(
        default=None,
        description="``np.ndarray`` of times (seconds) at which to sample the simulation, or ``None`` for solver steps.",
    )
    integrator_time: bool = Field(
        default=False,
        description="If True, output_time is overridden with the integrator's own step times.",
    )
    parameters: list = Field(
        default_factory=list,
        description="Reactor-state parameter names to dump (T, P, rho, ...).",
    )
    species: dict = Field(
        default_factory=dict,
        description="Species selected for output, keyed by row index in the species list widget.",
    )
    reactions: dict = Field(
        default_factory=dict,
        description="Reaction equations selected for output, keyed by row index in the reactions list widget.",
    )
    save_plot: bool = Field(
        default=True,
        description="If True, also dump a PNG snapshot of the current plot.",
    )
    output_time_offset: float = Field(
        default=0.0,
        description="Time offset (seconds) added to output_time when writing the trace.",
    )
    mech_output_dir: str = Field(
        default="",
        description="Last directory used by the Mechanism tab's save dialog.",
    )
    recast_to_arrhenius: bool = Field(
        default=False,
        description="If True, pressure-dependent rxns are flattened to Arrhenius at ``recast_pressure_pa`` before serialization.",
    )
    recast_pressure_pa: float = Field(
        default=101325.0,
        description="Pressure (Pa) at which to recast pressure-dependent rxns. Falloff variants become three-body Arrhenius with original species efficiencies preserved.",
    )

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, arbitrary_types_allowed=True,
    )
