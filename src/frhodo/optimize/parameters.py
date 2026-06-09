"""Optimization parameter selection and setup.

`OptimizableSet` is the data model that says **which reaction coefficients
are being optimized** in a given run. The setup helpers
(`build_rxn_coef_opt`, `build_rxn_rate_opt`) derive the per-reaction
aggregations and rate-bound structures the optimization loop consumes.

The mechanism stays a pure engine state. The "what to optimize" decision
is a separate concern, owned by whoever drives the optimizer.
"""
import collections
from typing import Sequence

import cantera as ct
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from frhodo.simulation.mechanism.coef_helpers import (
    arrhenius_coefNames,
    rates,
    set_bnds,
)



Ru = ct.gas_constant


class OptimizableCoefficient(BaseModel):
    """One Arrhenius/Plog/falloff coefficient marked as optimizable.

    Presence in an :class:`OptimizableSet` is the opt-in flag.

    ``coef_name`` is a string for Arrhenius/rate-limb coefs
    (``"pre_exponential_factor"`` etc.) and an int 0..3 for
    ``falloff_parameters`` entries (positional index into the
    list-valued ``mech.coeffs[rxn_idx]["falloff_parameters"]``).
    """
    rxn_idx: int = Field(ge=0)
    coef_name: str | int
    coef_idx: int = Field(ge=0)
    coeffs_key: str | int = Field(description="key into mech.coeffs[rxn_idx]")
    bnds_key: str = Field(description="key into mech.coeffs_bnds[rxn_idx]")

    model_config = ConfigDict(extra="forbid", frozen=True)


class OptimizableSet(BaseModel):
    """Full selection: which reactions are optimizable + which coefficients.

    ``optimizable_reactions`` holds rxn indices marked optimizable at the
    rate level. ``coefficients`` holds per-coefficient entries. A
    coefficient appears only if both its parent reaction and the
    coefficient itself are marked.
    """
    optimizable_reactions: tuple[int, ...] = ()
    coefficients: tuple[OptimizableCoefficient, ...] = ()

    model_config = ConfigDict(extra="forbid", frozen=True)

    def is_empty(self) -> bool:
        return not self.coefficients

    def slot_index(self, rxn_idx: int, coef_name: str | int) -> int | None:
        """Index into the optimizer's ``x`` vector for one coefficient.

        Returns ``None`` when the requested ``(rxn_idx, coef_name)`` pair
        is not present in this set.
        """
        for i, c in enumerate(self.coefficients):
            if c.rxn_idx == rxn_idx and c.coef_name == coef_name:
                return i

        return None

    def rxn_slots(self, rxn_idx: int) -> tuple[int, ...]:
        """All ``x``-indices that belong to one reaction."""
        return tuple(i for i, c in enumerate(self.coefficients) if c.rxn_idx == rxn_idx)


class OptimizableSetBuilder:
    """Mutable selection of optimizable rxns + coefficients.

    GUI mech-tree widgets toggle entries here as users check / uncheck
    boxes; the orchestrator calls ``build(mech)`` at optimization start
    to get an immutable :class:`OptimizableSet`.

    The (rxnIdx, bnds_key, coef_name) toggle state is the source of
    truth for which coefficients are optimizable.
    """

    def __init__(self) -> None:
        self._rxns: dict[int, bool] = {}
        self._coefs: dict[tuple[int, str, str], bool] = {}

    def set_reaction_optimizable(self, rxnIdx: int, opt: bool) -> None:
        self._rxns[rxnIdx] = opt

    def is_reaction_optimizable(self, rxnIdx: int) -> bool:
        return self._rxns.get(rxnIdx, False)

    def set_coefficient_optimizable(
        self, rxnIdx: int, bnds_key: str, coef_name: str, opt: bool,
    ) -> None:
        self._coefs[(rxnIdx, bnds_key, coef_name)] = opt

    def is_coefficient_optimizable(
        self, rxnIdx: int, bnds_key: str, coef_name: str,
    ) -> bool:
        return self._coefs.get((rxnIdx, bnds_key, coef_name), False)

    def reset(self) -> None:
        self._rxns.clear()
        self._coefs.clear()

    def build(self, mech) -> OptimizableSet:
        """Resolve the toggle state against the current mech, producing
        an immutable :class:`OptimizableSet`.

        Skips reactions not marked optimizable; emits one
        ``OptimizableCoefficient`` per coefficient where both the
        reaction-level and coefficient-level gates are set.

        Pressure-dependent rxns (Plog, Falloff, Lindemann, Sri, Tsang,
        Troe) are recast as Troe by the orchestrator and the full Troe
        parameterization (low/high Arrhenius limbs + Fcent A/T3/T1/T2)
        is optimized.  For these rxn types the per-coefficient toggle
        is bypassed and every coefficient is emitted, so coef_opt
        aligns with the 10-element fit return slot-for-slot.  Arrhenius
        rxns retain per-coefficient gating.
        """
        pressure_dep_types = (
            ct.FalloffRate,
            ct.LindemannRate,
            ct.TsangRate,
            ct.TroeRate,
            ct.SriRate,
            ct.PlogRate,
            ct.ChebyshevRate,
        )

        opt_rxns: list[int] = []
        coefs: list[OptimizableCoefficient] = []

        for rxnIdx, rxn in enumerate(mech.gas.reactions()):
            if not self.is_reaction_optimizable(rxnIdx):
                continue
            opt_rxns.append(rxnIdx)

            is_pressure_dep = isinstance(rxn.rate, pressure_dep_types)

            for bnds_key, sub_rxn in mech.coeffs_bnds[rxnIdx].items():
                for coef_idx, (coef_name, _coef_dict) in enumerate(sub_rxn.items()):
                    if not is_pressure_dep and not self.is_coefficient_optimizable(
                        rxnIdx, bnds_key, coef_name,
                    ):
                        continue
                    coeffs_key, real_bnds_key = mech.get_coeffs_keys(
                        rxn, bnds_key, rxnIdx=rxnIdx,
                    )
                    coefs.append(OptimizableCoefficient(
                        rxn_idx=rxnIdx,
                        coef_name=coef_name,
                        coef_idx=coef_idx,
                        coeffs_key=coeffs_key,
                        bnds_key=real_bnds_key,
                    ))

        optimizable_set = OptimizableSet(
            optimizable_reactions=tuple(opt_rxns),
            coefficients=tuple(coefs),
        )

        return optimizable_set


def build_rxn_coef_opt(
    mech,
    coef_opt: list[OptimizableCoefficient],
    shocks2run: list,
    *,
    min_P_range_factor: float = 2,
    T_margin: float = 500.0,
    P_margin_factor: float = 2.0,
) -> list[dict]:
    """Aggregate ``coef_opt`` per-reaction and decorate with sampling info.

    For each optimized reaction, populates the ``T`` / ``invT`` / ``P``
    / ``X`` / ``coef_x0`` / ``coef_bnds`` / ``is_falloff_limit`` fields
    the optimizer's rate-fit step needs.

    The recast samples each reaction over its range of validity so the
    Troe fit stays faithful where the integrator visits. The T envelope
    extends ``T_margin`` K beyond the shock T extrema, clamped to the
    mechanism's thermo validity ``[gas.min_temp, gas.max_temp]``. The P
    envelope extends a ``P_margin_factor`` multiplier beyond the shock P
    extrema (widened to ``min_P_range_factor`` decades if the shocks
    don't span enough); Plog reactions additionally span their full
    rate-table pressure range.

    Returns:
        List of per-reaction sampling dicts.
    """
    rxn_coef_opt: list = []
    for coef in coef_opt:
        key_dict = {"coeffs": coef.coeffs_key, "coeffs_bnds": coef.bnds_key}
        if not rxn_coef_opt or coef.rxn_idx != rxn_coef_opt[-1]["rxnIdx"]:
            rxn_coef_opt.append({
                "rxnIdx": coef.rxn_idx,
                "key": [key_dict],
                "coefIdx": [coef.coef_idx],
                "coefName": [coef.coef_name],
            })
        else:
            rxn_coef_opt[-1]["key"].append(key_dict)
            rxn_coef_opt[-1]["coefIdx"].append(coef.coef_idx)
            rxn_coef_opt[-1]["coefName"].append(coef.coef_name)

    shock_conditions = {"T_reactor": [], "P_reactor": [], "thermo_mix": []}
    for shock in shocks2run:
        for k in shock_conditions:
            shock_conditions[k].append(getattr(shock, k))

    T_mech_min = float(mech.gas.min_temp)
    T_mech_max = float(mech.gas.max_temp)
    T_min = max(T_mech_min, np.min(shock_conditions["T_reactor"]) - T_margin)
    T_max = min(T_mech_max, np.max(shock_conditions["T_reactor"]) + T_margin)
    T_bnds = np.array([T_min, T_max])
    invT_bnds = np.divide(10000, T_bnds)

    P_min = np.min(shock_conditions["P_reactor"]) / P_margin_factor
    P_max = np.max(shock_conditions["P_reactor"]) * P_margin_factor
    P_median = np.median(shock_conditions["P_reactor"])
    P_bnds = np.array([P_min, P_max])
    if P_bnds[1] / P_bnds[0] < min_P_range_factor:
        P_f_min = min_P_range_factor**0.5
        P_bnds = np.array([P_median / P_f_min, P_median * P_f_min])

    for rxn_coef in rxn_coef_opt:
        rxnIdx = rxn_coef["rxnIdx"]
        rxn = mech.gas.reaction(rxnIdx)

        rxn_coef["coef_x0"] = []
        for key, coefName in zip(rxn_coef["key"], rxn_coef["coefName"]):
            coef_x0 = mech.coeffs_bnds[rxnIdx][key["coeffs_bnds"]][coefName]["resetVal"]
            rxn_coef["coef_x0"].append(coef_x0)

        rxn_coef["coef_bnds"] = set_bnds(
            mech, rxnIdx, rxn_coef["key"], rxn_coef["coefName"]
        )

        rxn_invT_bnds = invT_bnds
        P_mid = P_median
        if type(rxn.rate) is ct.ArrheniusRate:
            P = P_median
        elif type(rxn.rate) is ct.PlogRate:
            P_table = [PlogRxn["Pressure"] for PlogRxn in mech.coeffs[rxnIdx]]
            P_union = sorted({*P_table, float(P_bnds[0]), float(P_bnds[1])})
            if len(P_union) < 4:
                P_union = list(np.geomspace(P_union[0], P_union[-1], 4))
            P = np.asarray(P_union, dtype=float)
        elif type(rxn.rate) is ct.ChebyshevRate:
            T_lo, T_hi = rxn.rate.temperature_range
            rxn_invT_bnds = np.divide(10000, np.array([T_lo, T_hi]))
            P_lo, P_hi = rxn.rate.pressure_range
            if P_hi <= P_lo:  # single-pressure (T-only) Chebyshev
                P_lo, P_hi = float(P_bnds[0]), float(P_bnds[1])
            P = np.geomspace(P_lo, P_hi, 3)
            P_mid = float(P[1])
        elif type(rxn.rate) in [
            ct.FalloffRate, ct.LindemannRate, ct.TsangRate,
            ct.TroeRate, ct.SriRate,
        ]:
            P = np.linspace(P_bnds[0], P_bnds[1], 3)
        else:
            raise ValueError(
                f"Unsupported rate type for recast sampling: "
                f"{type(rxn.rate).__name__}"
            )

        if type(rxn.rate) is ct.ArrheniusRate:
            n_coef = len(rxn_coef["coefIdx"])
            rxn_coef["invT"] = np.linspace(*rxn_invT_bnds, n_coef)
            rxn_coef["T"] = np.divide(10000, rxn_coef["invT"])
            rxn_coef["P"] = np.ones_like(rxn_coef["T"]) * P
        elif type(rxn.rate) in [
            ct.PlogRate, ct.ChebyshevRate, ct.FalloffRate,
            ct.LindemannRate, ct.TsangRate, ct.TroeRate, ct.SriRate,
        ]:
            rxn_coef["invT"] = []
            rxn_coef["P"] = []
            for coef_type in ["low_rate", "high_rate"]:
                n_coef = sum(1 for c in rxn_coef["key"] if coef_type in c["coeffs_bnds"])
                rxn_coef["invT"].append(np.linspace(*rxn_invT_bnds, n_coef))
                if coef_type == "low_rate":
                    rxn_coef["P"].append(np.ones(n_coef) * P[0])
                elif coef_type == "high_rate":
                    rxn_coef["P"].append(np.ones(n_coef) * P[-1])
                else:
                    raise ValueError(f"unexpected coef_type {coef_type!r}")

            if type(rxn.rate) is ct.PlogRate:
                invT = np.linspace(*rxn_invT_bnds, 3)
                P, invT = np.meshgrid(P[1:-1], invT)
                rxn_coef["invT"].append(invT.T.flatten())
                rxn_coef["P"].append(P.T.flatten())
            else:
                rxn_coef["invT"].append(np.linspace(*rxn_invT_bnds, 3))
                rxn_coef["P"].append(np.ones(3) * P_mid)

            for k in ["invT", "P"]:
                rxn_coef[k] = np.concatenate(rxn_coef[k], axis=0)
            rxn_coef["T"] = np.divide(10000, rxn_coef["invT"])

        rxn_coef["X"] = shock_conditions["thermo_mix"][0]
        rxn_coef["is_falloff_limit"] = np.array([False] * len(rxn_coef["T"]))

    return rxn_coef_opt


def build_rxn_rate_opt(mech, rxn_coef_opt: list[dict]) -> dict:
    """Compute initial scaled rates and rate bounds for the optimizer.

    Temporarily mutates ``mech.coeffs`` (resets to baseline, then
    restores).

    Returns:
        ``{"x0": np.ndarray, "bnds": {"lower": np.ndarray, "upper":
        np.ndarray}}`` ready to feed
        :class:`~frhodo.optimize.algorithms.Optimize`.
    """
    rxn_rate_opt: dict = {}
    prior_coeffs = mech.reset()
    rxn_rate_opt["x0"] = rates(rxn_coef_opt, mech)

    bnds = np.array([[], []])
    for i, rxn_coef in enumerate(rxn_coef_opt):
        rxnIdx = rxn_coef["rxnIdx"]
        rxn = mech.gas.reaction(rxnIdx)
        if type(rxn.rate) in [
            ct.PlogRate, ct.ChebyshevRate, ct.FalloffRate,
            ct.LindemannRate, ct.TsangRate, ct.TroeRate, ct.SriRate,
        ]:
            key_list = np.array([x["coeffs_bnds"] for x in rxn_coef["key"]])
            key_count = collections.Counter(key_list)
            for n, T in enumerate(rxn_coef["T"]):
                if n == len(key_list):
                    break
                coef_type_key = key_list[n]
                if "rate" in coef_type_key:
                    idx_match = np.argwhere(coef_type_key == key_list)
                    if (
                        np.any(rxn_coef["coef_bnds"]["exist"][idx_match])
                        or key_count[coef_type_key] < 3
                    ):
                        rxn_coef["is_falloff_limit"][n] = True
                        if type(rxn.rate) in [ct.FalloffRate, ct.TroeRate, ct.SriRate]:
                            x = [
                                mech.coeffs_bnds[rxnIdx][coef_type_key][n]["resetVal"]
                                for n in arrhenius_coefNames
                            ]
                            rxn_rate_opt["x0"][i + n] = (
                                np.log(x[1]) + x[2] * np.log(T) - x[0] / (Ru * T)
                            )

        ln_rate = rxn_rate_opt["x0"][i : i + len(rxn_coef["T"])]
        rxn_coef_bnds = mech.rate_bnds[rxnIdx]["limits"](np.exp(ln_rate))
        rxn_coef_bnds = np.sort(np.log(rxn_coef_bnds), axis=0)
        scaled_rxn_coef_bnds = rxn_coef_bnds - ln_rate
        bnds = np.concatenate((bnds, scaled_rxn_coef_bnds), axis=1)

    rxn_rate_opt["bnds"] = {"lower": bnds[0, :], "upper": bnds[1, :]}
    mech.coeffs = prior_coeffs
    mech.modify_reactions(mech.coeffs)

    return rxn_rate_opt

