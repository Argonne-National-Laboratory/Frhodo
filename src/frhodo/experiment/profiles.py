"""Typed weighting and experiment-shock models for the optimization API.

``WeightProfile`` is a parametric envelope backed by
:func:`frhodo.experiment.weight.double_sigmoid` — a config-friendly
shape that produces per-sample weights when applied to a time array.
``ExperimentShock`` bundles one shock's trace + its initial conditions
+ optional per-shock weighting override for use in
:class:`frhodo.api.OptimizationRequest`. Per-sample uncertainty is
estimated from the data itself
(:func:`frhodo.experiment.uncertainty.estimate_pointwise_sigma`) so no
parametric uncertainty profile is needed.
"""
from __future__ import annotations

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    field_validator,
    model_validator,
)

from frhodo.common.config import PostShockState, PreShockState
from frhodo.experiment.weight import double_sigmoid


class WeightProfile(BaseModel):
    """Two-sided sigmoid envelope used as per-sample cost weights.

    Defaults reproduce the GUI's stock ``WeightFunction`` shape, which
    emphasizes the post-ignition window.

    ``peak`` is the maximum weight in the rise/fall window; ``floor_pre``
    and ``floor_post`` are the weights outside it. Times are interpreted
    as percentages of trace duration when ``absolute_time`` is False; as
    absolute seconds when True. Samples outside ``[cutoff_pre,
    cutoff_post]`` (also % of trace) are dropped before the envelope is
    applied.
    """
    peak: PositiveFloat = 100.0
    floor_pre: NonNegativeFloat = 0.0
    floor_post: NonNegativeFloat = 0.0
    time_rise: PositiveFloat = 4.5
    time_fall: PositiveFloat = 35.0
    growth_rate_rise: NonNegativeFloat = 0.0
    growth_rate_fall: NonNegativeFloat = 0.7
    absolute_time: bool = False
    cutoff_pre: NonNegativeFloat = 0.0
    cutoff_post: NonNegativeFloat = 100.0

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _cutoffs_ordered(self):
        if self.cutoff_post <= self.cutoff_pre:
            raise ValueError(
                f"cutoff_post ({self.cutoff_post}) must exceed cutoff_pre "
                f"({self.cutoff_pre})"
            )

        return self

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Return per-sample weights for ``t`` (1-D ndarray of times).

        Calls ``double_sigmoid`` with the same parameter layout the
        legacy code path uses, so a ``WeightProfile`` produces a
        bit-identical envelope to the existing GUI weight pipeline for
        equivalent input values.
        """
        t = np.asarray(t, dtype=float)
        return double_sigmoid(
            t,
            A=[self.floor_pre, self.peak, self.floor_post],
            k=[self.growth_rate_rise, self.growth_rate_fall],
            x0=[self.time_rise, self.time_fall],
        )


class ExperimentShock(BaseModel):
    """One experimental shock for use in :class:`OptimizationRequest`.

    Storage is ``list[float]`` so the model round-trips through YAML and
    pickle; call :meth:`t_array` / :meth:`observable_array` to obtain
    ``np.ndarray`` views for numerical use. Weight and uncertainty
    profiles override the request-level defaults when set.
    """
    t: list[float]
    observable: list[float]
    initial: PreShockState | PostShockState = Field(discriminator="kind")
    t_end: PositiveFloat
    scalar_weight: PositiveFloat = 1.0
    weight_profile: WeightProfile | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("t", "observable", mode="before")
    @classmethod
    def _coerce_to_list(cls, v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, list):
            return v

        return list(v)

    @model_validator(mode="after")
    def _trace_lengths_match(self):
        if len(self.t) != len(self.observable):
            raise ValueError(
                f"t (len {len(self.t)}) and observable (len {len(self.observable)}) "
                "must have equal length"
            )

        return self

    def t_array(self) -> np.ndarray:
        """Return ``t`` as an ``np.ndarray`` view for numerical use."""
        return np.asarray(self.t, dtype=float)

    def observable_array(self) -> np.ndarray:
        """Return ``observable`` as an ``np.ndarray`` view for numerical use."""
        return np.asarray(self.observable, dtype=float)
