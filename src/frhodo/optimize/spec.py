"""Typed user-facing specification of "what to optimize, how far it can move".

``OptimizableSpec`` declares which rates / coefficients are optimized and
the uncertainty bounds each can move within. ``OptimizableSpec.build(mech)``
resolves the spec against a loaded mechanism and produces the immutable
:class:`frhodo.optimize.parameters.OptimizableSet` the optimizer consumes.

``OptimizableSpecBuilder`` is a mutable companion for stateful interactive
use (the GUI reaction tree). Python users typically construct
``OptimizableSpec`` directly.
"""
from __future__ import annotations

from typing import Iterable

import cantera as ct
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, model_validator

from frhodo.optimize.parameters import (
    OptimizableSet,
    OptimizableSetBuilder,
)


_PRESSURE_DEP_RATE_TYPES = (
    ct.FalloffRate,
    ct.LindemannRate,
    ct.TsangRate,
    ct.TroeRate,
    ct.SriRate,
    ct.PlogRate,
    ct.ChebyshevRate,
)


class CoefUncertainty(BaseModel):
    """Bounds for one optimization coefficient.

    Exactly one of ``factor`` / ``delta`` / ``bounds`` must be provided.

    - ``factor`` is multiplicative: the value moves within
      ``(nominal / factor, nominal * factor)``. Natural for A.
    - ``delta`` is additive: ``(nominal - delta, nominal + delta)``.
      Natural for Ea, n.
    - ``bounds`` is absolute: ``(lo, hi)``. Natural for T3/T1/T2.
    """
    factor: PositiveFloat | None = None
    delta: PositiveFloat | None = None
    bounds: tuple[float, float] | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _exactly_one(self):
        provided = sum(x is not None for x in (self.factor, self.delta, self.bounds))
        if provided != 1:
            raise ValueError(
                f"CoefUncertainty: exactly one of factor / delta / bounds must "
                f"be set; got {provided}"
            )

        return self

    def resolve(self, nominal: float, *, coef_name: str) -> tuple[float, float]:
        """Return the absolute ``(lo, hi)`` bounds for ``nominal``."""
        if self.factor is not None:
            return (nominal / self.factor, nominal * self.factor)
        if self.delta is not None:
            return (nominal - self.delta, nominal + self.delta)
        lo, hi = self.bounds
        if not (lo <= nominal <= hi):
            raise ValueError(
                f"CoefUncertainty bounds ({lo}, {hi}) do not bracket the "
                f"nominal value {nominal} for coefficient {coef_name!r}"
            )

        return (lo, hi)


class RateUncertainty(BaseModel):
    """Rate-level multiplicative uncertainty: rate ∈ (k0 / f, k0 * f)."""
    factor: PositiveFloat = 2.0

    model_config = ConfigDict(extra="forbid", frozen=True)


class OptimizableRate(BaseModel):
    """One reaction targeted for optimization.

    ``rate`` provides a rate-level uncertainty (factor of 2 by default).
    ``coefficients`` adds per-coefficient overrides. Coefficient bounds
    take precedence per coefficient; the rate-level bound is enforced
    at the (T, P) sample points the optimizer evaluates so the two are
    intersected in practice.

    ``optimize`` restricts the subset of coefficients to fit. ``None``
    means "all standard for the rate type". Pressure-dependent rates
    (Plog/Falloff/Lindemann/Sri/Troe) are recast to Troe and the full
    10-element parameter set is always fit; passing ``optimize`` or
    per-coefficient overrides for such reactions raises at build time.
    """
    rxn_idx: int = Field(ge=0)
    rate: RateUncertainty | None = None
    coefficients: dict[str, CoefUncertainty] | None = None
    optimize: list[str] | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class OptimizableSpec(BaseModel):
    """User-facing description of an optimization target set."""
    rates: list[OptimizableRate]
    default_rate: RateUncertainty = Field(default_factory=RateUncertainty)

    model_config = ConfigDict(extra="forbid", frozen=True)

    def build(self, mech) -> OptimizableSet:
        """Resolve the spec against ``mech`` and produce an
        :class:`OptimizableSet`.

        Mutates ``mech.rate_bnds`` and ``mech.coeffs_bnds`` in place so
        ``build_rxn_coef_opt`` / ``build_rxn_rate_opt`` see the user's
        uncertainty selections.
        """
        builder = self.to_builder(mech)

        return builder.build(mech)

    def to_builder(self, mech) -> OptimizableSetBuilder:
        """Resolve the spec into a populated :class:`OptimizableSetBuilder`.

        The builder is the toggle source of truth ``recast_to_troe``
        consults via ``is_coefficient_optimizable``. ``build`` wraps this
        and calls :meth:`OptimizableSetBuilder.build`; callers that need
        the builder itself (the recast step) use this directly.

        Mutates ``mech.rate_bnds`` and ``mech.coeffs_bnds`` in place so
        ``build_rxn_coef_opt`` / ``build_rxn_rate_opt`` see the user's
        uncertainty selections.

        Validation performed here:

        - ``rxn_idx`` exists in the mechanism
        - per-coefficient overrides and ``optimize`` subsets are not
          allowed for pressure-dependent reactions
        - ``CoefUncertainty.bounds`` brackets the current coefficient
          value
        """
        seen: set[int] = set()
        builder = OptimizableSetBuilder()
        n_rxns = mech.gas.n_reactions

        for entry in self.rates:
            if entry.rxn_idx >= n_rxns or entry.rxn_idx < 0:
                raise ValueError(
                    f"OptimizableRate.rxn_idx={entry.rxn_idx} is out of range "
                    f"for mechanism with {n_rxns} reactions"
                )
            if entry.rxn_idx in seen:
                raise ValueError(
                    f"OptimizableSpec: duplicate rxn_idx {entry.rxn_idx}"
                )
            seen.add(entry.rxn_idx)

            rxn = mech.gas.reaction(entry.rxn_idx)
            is_pdep = isinstance(rxn.rate, _PRESSURE_DEP_RATE_TYPES)
            if is_pdep and (entry.coefficients or entry.optimize):
                raise ValueError(
                    f"OptimizableRate(rxn_idx={entry.rxn_idx}): pressure-"
                    f"dependent reactions are recast to Troe; per-coefficient "
                    "overrides and 'optimize' subsets are not supported"
                )

            rate_unc = entry.rate or self.default_rate
            self._apply_rate_uncertainty(mech, entry.rxn_idx, rate_unc)

            builder.set_reaction_optimizable(entry.rxn_idx, True)
            self._toggle_coefficients(builder, mech, entry, rate_unc, is_pdep)

        return builder

    def _apply_rate_uncertainty(
        self, mech, rxn_idx: int, rate_unc: RateUncertainty,
    ) -> None:
        mech.rate_bnds[rxn_idx]["value"] = rate_unc.factor
        mech.rate_bnds[rxn_idx]["type"] = "F"

    def _toggle_coefficients(
        self,
        builder: OptimizableSetBuilder,
        mech,
        entry: OptimizableRate,
        rate_unc: RateUncertainty,
        is_pdep: bool,
    ) -> None:
        if is_pdep:
            return

        subset: Iterable[str] | None = entry.optimize
        overrides = entry.coefficients or {}

        for bnds_key, sub in mech.coeffs_bnds[entry.rxn_idx].items():
            for coef_name in sub:
                if not isinstance(coef_name, str):
                    continue
                if subset is not None and coef_name not in subset:
                    continue

                if coef_name in overrides:
                    self._apply_coef_uncertainty(
                        mech, entry.rxn_idx, bnds_key, coef_name,
                        overrides[coef_name],
                    )
                else:
                    self._apply_rate_factor_to_coef(
                        mech, entry.rxn_idx, bnds_key, coef_name, rate_unc.factor,
                    )

                builder.set_coefficient_optimizable(
                    entry.rxn_idx, bnds_key, coef_name, True,
                )

    def _apply_rate_factor_to_coef(
        self, mech, rxn_idx: int, bnds_key: str, coef_name: str, factor: float,
    ) -> None:
        d = mech.coeffs_bnds[rxn_idx][bnds_key][coef_name]
        d["value"] = factor
        d["type"] = "F"

    def _apply_coef_uncertainty(
        self, mech, rxn_idx: int, bnds_key: str, coef_name: str,
        cu: CoefUncertainty,
    ) -> None:
        d = mech.coeffs_bnds[rxn_idx][bnds_key][coef_name]
        if cu.factor is not None:
            d["value"] = cu.factor
            d["type"] = "F"
        elif cu.delta is not None:
            d["value"] = cu.delta
            d["type"] = "±"
        else:
            lo, hi = cu.bounds
            nominal = d.get("resetVal")
            if nominal is None:
                nominal = d.get("value")
            if nominal is not None and not (lo <= nominal <= hi):
                raise ValueError(
                    f"CoefUncertainty(bounds=({lo}, {hi})) for rxn {rxn_idx} "
                    f"coef {coef_name!r} does not bracket nominal value {nominal}"
                )

            d["limits"] = lambda lo=lo, hi=hi: [float(lo), float(hi)]
            d["value"] = 1.0
            d["type"] = "F"


class OptimizableSpecBuilder:
    """Mutable accumulator producing an immutable :class:`OptimizableSpec`.

    Designed for stateful interactive use (GUI tree widgets where the
    user toggles reaction checkboxes). Python users normally construct
    ``OptimizableSpec`` directly.
    """

    def __init__(self) -> None:
        self._rates: dict[int, OptimizableRate] = {}
        self._default_rate: RateUncertainty = RateUncertainty()

    def set_rxn(
        self,
        rxn_idx: int,
        *,
        enabled: bool,
        rate: RateUncertainty | None = None,
        coefficients: dict[str, CoefUncertainty] | None = None,
        optimize: list[str] | None = None,
    ) -> None:
        """Add or replace the entry for ``rxn_idx``.

        Passing ``enabled=False`` removes the reaction; the other
        kwargs map directly to :class:`OptimizableRate` fields.
        """
        if not enabled:
            self._rates.pop(rxn_idx, None)
            return
        self._rates[rxn_idx] = OptimizableRate(
            rxn_idx=rxn_idx,
            rate=rate,
            coefficients=coefficients,
            optimize=optimize,
        )

    def clear_rxn(self, rxn_idx: int) -> None:
        """Remove ``rxn_idx`` from the selection if present."""
        self._rates.pop(rxn_idx, None)

    def set_default_rate(self, rate: RateUncertainty) -> None:
        """Set the rate-level uncertainty used when a reaction has none."""
        self._default_rate = rate

    def build(self) -> OptimizableSpec:
        """Snapshot the current selection as an immutable :class:`OptimizableSpec`."""
        return OptimizableSpec(
            rates=list(self._rates.values()),
            default_rate=self._default_rate,
        )
