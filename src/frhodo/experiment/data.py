"""Typed model for an experimental shock.

``ExperimentalShock`` pins the shape of the record passed between
loaders, GUI widgets, and the optimizer. Consumers index via
attribute access (``shock.T1``); for runtime-keyed lookups use
``getattr(shock, key)`` / ``setattr(shock, key, value)``.

Construction:

* ``ExperimentalShock.empty(num, path, series_name)`` — initial
  placeholder with NaNs for unread fields, used by the series builder.
* ``ExperimentalShock.from_dict(d)`` — adopt an existing dict; unknown
  keys are preserved via ``extra="allow"``.
"""
from copy import deepcopy
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


def _empty_array() -> np.ndarray:
    return np.array([])


def _nan_pair() -> list[float]:
    return [float("nan"), float("nan")]


class ExperimentalShock(BaseModel):
    """A single shock-tube experiment record.

    Fields cover the pre-shock state, the reactor zone choice, parsed
    experimental traces, GUI-set weight/uncertainty knobs, and the
    optimizer's runtime scratch space. Every field defaults to a
    safe empty/NaN value so partial construction is fine; the
    loaders and GUI widgets fill in fields as they become known.
    """

    num: int = Field(
        default=0,
        description="Shock identifier (1-indexed in the user-facing GUI).",
    )
    path: Any = Field(
        default_factory=dict,
        description="File path(s) for the .exp/.rho/.sig source files for this shock; shape varies by loader.",
    )
    include: bool = Field(
        default=False,
        description="Whether this shock participates in the optimization run.",
    )
    series_name: str = Field(
        default="",
        description="Name of the series this shock belongs to (group label in the GUI).",
    )
    run_SIM: bool = Field(
        default=True,
        description="If True, simulation is re-run when display switches to this shock.",
    )

    # Pre-shock zone (zone 1)
    T1: float = Field(default=float("nan"), description="Pre-shock gas temperature [K].")
    P1: float = Field(default=float("nan"), description="Pre-shock pressure [Pa].")
    u1: float = Field(default=float("nan"), description="Pre-shock incident velocity [m/s].")
    rho1: float = Field(default=float("nan"), description="Pre-shock density [kg/m^3].")
    P4: float = Field(default=float("nan"), description="Driver-section pressure [Pa].")

    # Driven-zone post-shock states (zone 2 = incident, zone 5 = reflected)
    T2: float = Field(default=float("nan"), description="Post-incident-shock temperature [K].")
    P2: float = Field(default=float("nan"), description="Post-incident-shock pressure [Pa].")
    u2: float = Field(default=float("nan"), description="Post-incident-shock velocity [m/s].")
    T5: float = Field(default=float("nan"), description="Post-reflected-shock temperature [K].")
    P5: float = Field(default=float("nan"), description="Post-reflected-shock pressure [Pa].")
    zone: int = Field(default=2, description="Reactor zone selection (2 = incident, 5 = reflected).")
    T_reactor: float = Field(default=float("nan"), description="Reactor temperature for the chosen zone [K].")
    P_reactor: float = Field(default=float("nan"), description="Reactor pressure for the chosen zone [Pa].")

    # Composition
    exp_mix: dict = Field(
        default_factory=dict,
        description="Experimental mole-fraction mixture (species name -> fraction).",
    )
    thermo_mix: dict = Field(
        default_factory=dict,
        description="Thermodynamic mixture (post any aliasing applied to ``exp_mix``).",
    )
    species_alias: dict = Field(
        default_factory=dict,
        description="Inherited species-name aliases for this shock's series.",
    )

    # Acquisition
    Sample_Rate: float = Field(
        default=float("nan"),
        description="Data-acquisition sample rate [samples/s] from the .exp metadata.",
    )
    observable: dict = Field(
        default_factory=lambda: {"main": "", "sub": None},
        description="Selected observable, ``{'main': str, 'sub': int|str|None}``.",
    )

    # Time / offset
    time_offset: float = Field(
        default=0.0,
        description="Time offset (seconds) applied to the experiment trace before fitting.",
    )
    opt_time_offset: float = Field(
        default=0.0,
        description="Snapshot of ``time_offset`` taken at optimization start; pinned for reproducibility.",
    )
    last_t_unc: float | None = Field(
        default=None,
        description="Cached `t_unc` from the previous cost evaluation, used to warm-start the inner ``minimize_scalar``. ``None`` means no cached solution yet (cold start).",
    )
    last_loss_alpha: float | None = Field(
        default=None,
        description="Cached per-shock ``loss_alpha`` from the previous cost evaluation, used to warm-start the per-shock adaptive-loss solve. ``None`` means cold start.",
    )

    # Weight function (GUI-controlled)
    weight_max: list[float] = Field(
        default_factory=lambda: [float("nan")],
        description="Single-element list with the weight-function plateau max.",
    )
    weight_min: list[float] = Field(
        default_factory=_nan_pair,
        description="Two-element list of weight minima at the start/end of the trace.",
    )
    weight_shift: list[float] = Field(
        default_factory=_nan_pair,
        description="Two-element list of normalized time locations for the start/end weight kinks.",
    )
    weight_k: list[float] = Field(
        default_factory=_nan_pair,
        description="Two-element list of inverse-growth-rate values controlling start/end transitions.",
    )

    # Rate optimization scratch
    rate_val: list = Field(
        default_factory=list,
        description="Per-reaction current rate values for this shock.",
    )
    rate_reset_val: list = Field(
        default_factory=list,
        description="Snapshot of ``rate_val`` taken at optimization start; restored on reset.",
    )
    rate_bnds_type: list = Field(
        default_factory=list,
        description="Per-reaction bound-type tag (factor, additive, etc.).",
    )
    rate_bnds_val: list = Field(
        default_factory=list,
        description="Per-reaction bound magnitude paired with ``rate_bnds_type``.",
    )
    rate_bnds: list = Field(
        default_factory=list,
        description="Resolved per-reaction lower/upper rate bounds.",
    )

    # Experimental data + derived traces
    raw_data: Any = Field(
        default_factory=_empty_array,
        description="Raw experimental trace, shape (N, 2) with [time, observable] columns.",
    )
    exp_data: Any = Field(
        default_factory=_empty_array,
        description="Time-aligned experimental trace ready for fitting; same shape as ``raw_data``.",
    )
    exp_data_trim: Any = Field(
        default_factory=_empty_array,
        description="Sub-slice of ``exp_data`` covering only the optimization-active window.",
    )

    # Weights / uncertainties (residual / Bayesian paths)
    weights: Any = Field(
        default_factory=_empty_array,
        description="Per-sample residual weights produced by the weight function over ``exp_data``.",
    )
    weights_trim: Any = Field(
        default_factory=_empty_array,
        description="``weights`` restricted to the optimization-active window.",
    )
    normalized_weights: Any = Field(
        default_factory=_empty_array,
        description="``weights`` rescaled to sum to 1 across the trace.",
    )
    sigma_t: Any = Field(
        default_factory=_empty_array,
        description="Per-sample wavelet-estimated noise std σ(t); length matches ``exp_data``.",
    )
    abs_uncertainties: Any = Field(
        default_factory=_empty_array,
        description="``y ± sigma_multiple·σ(t)`` lower/upper bounds; shape (N, 2).",
    )
    abs_uncertainties_trim: Any = Field(
        default_factory=_empty_array,
        description="``abs_uncertainties`` restricted to the optimization-active window.",
    )

    # Last simulation result attached for plotting / display
    SIM: Any = Field(
        default_factory=_empty_array,
        description="Most recent reactor simulation result for this shock; replaced on every re-sim.",
    )

    # Cached pre-built Bisymlog transform (Bisymlog scale only)
    bisymlog: Any = Field(
        default=None,
        description="Cached ``Bisymlog`` transform built once at optimization start; ``None`` for Linear/Log scales.",
    )

    # Error log (parser problems, missing data, etc.)
    err: list[str] = Field(
        default_factory=list,
        description="Parse / load errors recorded for this shock; checked by the optimizer to skip bad entries.",
    )

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        validate_assignment=False,
    )

    @classmethod
    def empty(cls, num: int, path: dict, series_name: str) -> "ExperimentalShock":
        """Construct a fresh shock with NaN/empty placeholders."""
        return cls(num=num, path=deepcopy(path), series_name=series_name)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentalShock":
        """Adopt an existing shock dict. Unknown keys are kept via
        ``extra='allow'``; callers that need runtime-keyed lookups use
        ``getattr(shock, key)`` / ``setattr(shock, key, value)``."""
        return cls(**d)
