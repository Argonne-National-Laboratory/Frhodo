# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level
# directory for license and copyright information.
"""Runtime mechanism container: rate expression coefficients + mutation helpers.

:class:`ChemicalMechanism` wraps a Cantera ``Solution`` plus parallel
metadata used by the optimizer (per-coefficient bounds, reaction-type
flags, coefficient names). Loaders populate it via
:class:`~frhodo.simulation.mechanism.mechanism_loader.MechanismLoader`.
"""
import contextlib
import pathlib
import threading
from copy import deepcopy

import cantera as ct
import numpy as np

from frhodo.simulation.mechanism.coef_helpers import arrhenius_coefNames, set_bnds
from frhodo.simulation.mechanism.fit_coeffs import fit_arrhenius, fit_generic
from frhodo.simulation.mechanism.troe_kernels import ln_Troe



_FALLOFF_FAMILY = (
    ct.FalloffRate,
    ct.LindemannRate,
    ct.TsangRate,
    ct.TroeRate,
    ct.SriRate,
)

_PDEP_FAMILY = (ct.PlogRate, ct.ChebyshevRate, *_FALLOFF_FAMILY)


def _recast_one_pdep(
    eval_gas, rxn, rxnIdx, pressure_pa, composition,
    T_lo, T_hi, n_points, fit_arrhenius,
):
    """Build the recast replacement reaction (or pass through unchanged)."""
    rate = rxn.rate
    if not isinstance(rate, _PDEP_FAMILY):
        return rxn

    T_min, T_max = T_lo, T_hi
    if isinstance(rate, ct.ChebyshevRate):
        cheb_min, cheb_max = rate.temperature_range
        T_min = max(T_min, cheb_min)
        T_max = min(T_max, cheb_max)
    if T_max <= T_min:
        return rxn

    T_grid = np.linspace(T_min, T_max, n_points)
    rates = np.empty(n_points)
    for i, T in enumerate(T_grid):
        eval_gas.TPX = T, pressure_pa, composition
        rates[i] = eval_gas.forward_rate_constants[rxnIdx]

    if isinstance(rate, _FALLOFF_FAMILY):
        M_total = pressure_pa / (ct.gas_constant * T_grid)
        Ea, A, n = fit_arrhenius(rates / M_total, T_grid)
        new_rxn = ct.Reaction(
            rxn.reactants, rxn.products,
            rate=ct.ArrheniusRate(A, n, Ea),
            third_body=ct.ThirdBody(efficiencies=dict(rxn.third_body.efficiencies)),
        )
    else:
        Ea, A, n = fit_arrhenius(rates, T_grid)
        new_rxn = ct.Reaction(rxn.reactants, rxn.products, rate=ct.ArrheniusRate(A, n, Ea))

    new_rxn.duplicate = rxn.duplicate
    new_rxn.reversible = rxn.reversible
    if hasattr(new_rxn, "allow_negative_pre_exponential_factor"):
        new_rxn.allow_negative_pre_exponential_factor = True

    return new_rxn


class ChemicalMechanism:
    """Cantera ``Solution`` plus per-reaction optimization metadata.

    Construct empty; populate via :meth:`set_mechanism` (or the
    convenience loader). Cantera's ``Solution`` is not thread-safe,
    so :meth:`exclusive` returns a context manager that callers should
    hold across any sequence of state mutations.

    Attributes:
        gas: Underlying Cantera ``Solution`` (populated by loaders).
        isLoaded: ``True`` once the loader has finished wiring the gas.
        coeffs: Per-reaction rate-expression coefficients.
        coeffs_bnds: Per-coefficient bounds for the optimizer.
        rate_bnds: Per-reaction overall rate bounds.
        reset_mech: Snapshot of the reactions at load time, used by
            :meth:`reset` to revert in-place edits.
    """

    def __init__(self):
        self.isLoaded = False
        self._lock = threading.RLock()
        self._recast_originals: dict[int, ct.Reaction] = {}
        self.recast_log_rms: dict[int, float] = {}

    @contextlib.contextmanager
    def exclusive(self):
        """Serialize access to ``mech.gas`` for the call's duration.

        Cantera's ``Solution`` and Frhodo's parallel coefficient
        bookkeeping are not safe to mutate concurrently. Reactor entry
        points and any other ``mech.gas`` user holds this around their
        work so threaded callers run sequentially rather than corrupt
        shared state. Reentrant on the same thread.
        """
        with self._lock:
            yield

    def set_mechanism(self, mech_dict, species_dict=None, bnds=None):
        def get_Arrhenius_parameters(entry):
            A = entry["pre_exponential_factor"]
            b = entry["temperature_exponent"]
            Ea = entry["activation_energy"]

            return A, b, Ea

        if not species_dict:
            species = self.gas.species()
        else:
            species = []
            for n in range(len(species_dict)):
                s_dict = species_dict[n]
                s = ct.Species(
                    name=s_dict["name"],
                    composition=s_dict["composition"],
                    charge=s_dict["charge"],
                    size=s_dict["size"],
                )
                thermo = s_dict["type"](
                    s_dict["T_low"], s_dict["T_high"], s_dict["P_ref"], s_dict["coeffs"]
                )
                s.thermo = thermo

                species.append(s)

        # Set kinetics data
        rxns = []
        for rxnIdx in range(len(mech_dict)):
            if "Arrhenius Reaction" == mech_dict[rxnIdx]["rxnType"]:
                A, b, Ea = get_Arrhenius_parameters(mech_dict[rxnIdx]["rxnCoeffs"][0])
                rate = ct.ArrheniusRate(A, b, Ea)

                rxn = ct.Reaction(
                    mech_dict[rxnIdx]["reactants"], mech_dict[rxnIdx]["products"], rate
                )

            elif "Three Body Reaction" == mech_dict[rxnIdx]["rxnType"]:
                A, b, Ea = get_Arrhenius_parameters(mech_dict[rxnIdx]["rxnCoeffs"][0])
                rate = ct.ArrheniusRate(A, b, Ea)

                third_body = ct.ThirdBody(
                    efficiencies=mech_dict[rxnIdx]["rxnCoeffs"][0]["efficiencies"]
                )
                rxn = ct.Reaction(
                    mech_dict[rxnIdx]["reactants"],
                    mech_dict[rxnIdx]["products"],
                    rate=rate,
                    third_body=third_body,
                )

            elif "Plog Reaction" == mech_dict[rxnIdx]["rxnType"]:
                rates = []
                for plog in mech_dict[rxnIdx]["rxnCoeffs"]:
                    pressure = plog["Pressure"]
                    A, b, Ea = get_Arrhenius_parameters(plog)
                    rates.append((pressure, ct.ArrheniusRate(A, b, Ea)))

                rate = ct.PlogRate(rates)
                rxn = ct.Reaction(
                    mech_dict[rxnIdx]["reactants"], mech_dict[rxnIdx]["products"], rate
                )

            elif "Falloff Reaction" == mech_dict[rxnIdx]["rxnType"]:
                # high pressure limit
                A, b, Ea = get_Arrhenius_parameters(
                    mech_dict[rxnIdx]["rxnCoeffs"]["high_rate"]
                )
                high_rate = ct.ArrheniusRate(A, b, Ea)

                # low pressure limit
                A, b, Ea = get_Arrhenius_parameters(
                    mech_dict[rxnIdx]["rxnCoeffs"]["low_rate"]
                )
                low_rate = ct.ArrheniusRate(A, b, Ea)

                # falloff parameters
                falloff_type = mech_dict[rxnIdx]["rxnCoeffs"]["falloff_type"]
                falloff_coeffs = mech_dict[rxnIdx]["rxnCoeffs"]["falloff_parameters"]

                if falloff_type == "Lindemann":
                    rate = ct.LindemannRate(low_rate, high_rate, falloff_coeffs)

                elif falloff_type == "Tsang":
                    rate = ct.TsangRate(low_rate, high_rate, falloff_coeffs)

                elif falloff_type == "Troe":
                    if falloff_coeffs[-1] == 0.0:
                        falloff_coeffs = falloff_coeffs[0:-1]

                    rate = ct.TroeRate(low_rate, high_rate, falloff_coeffs)
                elif falloff_type == "SRI":
                    rate = ct.SriRate(low_rate, high_rate, falloff_coeffs)

                third_body = ct.ThirdBody(
                    efficiencies=mech_dict[rxnIdx]["rxnCoeffs"]["efficiencies"]
                )
                rxn = ct.Reaction(
                    mech_dict[rxnIdx]["reactants"],
                    mech_dict[rxnIdx]["products"],
                    rate=rate,
                    third_body=third_body,
                )

            elif "Chebyshev Reaction" == mech_dict[rxnIdx]["rxnType"]:
                cheb_data = mech_dict[rxnIdx]["rxnCoeffs"]
                rate = ct.ChebyshevRate(
                    temperature_range=[cheb_data["Tmin"], cheb_data["Tmax"]],
                    pressure_range=[cheb_data["Pmin"], cheb_data["Pmax"]],
                    data=cheb_data["coeffs"],
                )
                rxn = ct.Reaction(
                    mech_dict[rxnIdx]["reactants"],
                    mech_dict[rxnIdx]["products"],
                    rate=rate,
                )

            rxn.duplicate = mech_dict[rxnIdx]["duplicate"]
            rxn.reversible = mech_dict[rxnIdx]["reversible"]
            rxn.allow_negative_orders = True
            rxn.allow_nonreactant_orders = True

            if hasattr(rxn, "allow_negative_pre_exponential_factor"):
                rxn.allow_negative_pre_exponential_factor = True

            rxns.append(rxn)

        self.gas = ct.Solution(
            thermo="ideal-gas", kinetics="gas", species=species, reactions=rxns
        )

        self.set_rate_expression_coeffs(bnds)  # set copy of coeffs
        self.set_thermo_expression_coeffs()  # set copy of thermo coeffs

    def reaction_type(self, rxn):
        """Classify ``rxn`` for the optimizer's coefficient bookkeeping.

        Returns:
            One of ``"Arrhenius Reaction"``, ``"Three Body Reaction"``,
            ``"Plog Reaction"``, ``"Falloff Reaction"``, ``"Chebyshev
            Reaction"``, or the Cantera rate-class repr for anything
            else (signals unsupported types to callers).
        """
        if isinstance(rxn.rate, ct.ArrheniusRate):
            if rxn.reaction_type.startswith("three-body"):
                return "Three Body Reaction"

            return "Arrhenius Reaction"
        if isinstance(rxn.rate, ct.PlogRate):
            return "Plog Reaction"
        if isinstance(rxn.rate, _FALLOFF_FAMILY):
            return "Falloff Reaction"
        if isinstance(rxn.rate, ct.ChebyshevRate):
            return "Chebyshev Reaction"

        return str(type(rxn.rate))

    def set_rate_expression_coeffs(self, bnds=None):
        """Build ``coeffs`` / ``coeffs_bnds`` / ``rate_bnds`` / ``reset_mech``.

        Walks each reaction in the gas, snapshots the current rate
        expression's coefficients (Arrhenius, Plog rows, Troe falloff
        params, Chebyshev grid), and constructs the parallel bound
        records the optimizer mutates.

        Args:
            bnds: Optional ``{"rate_bnds": [...], "coeffs_bnds":
                [...]}`` from a prior session; values are copied into
                the freshly-built structure where present.
        """
        def copy_bnds(new_bnds, bnds, rxnIdx, bnds_type, keys=None):
            if not bnds:
                return new_bnds

            if bnds_type == "rate":
                for key in ("value", "type"):
                    new_bnds[rxnIdx][key] = bnds["rate_bnds"][rxnIdx][key]
            else:
                bndsKey, attrs = keys
                for coefName in attrs:
                    for key in ("value", "type"):
                        new_bnds[rxnIdx][bndsKey][coefName][key] = bnds["coeffs_bnds"][
                            rxnIdx
                        ][bndsKey][coefName][key]

            return new_bnds

        self.coeffs = coeffs = []
        self.coeffs_bnds = coeffs_bnds = []
        self.rate_bnds = rate_bnds = []
        self.reset_mech = reset_mech = []

        for rxnIdx, rxn in enumerate(self.gas.reactions()):
            rate_bnds.append(
                {
                    "value": np.nan,
                    "limits": Uncertainty("rate", rxnIdx, rate_bnds=rate_bnds),
                    "type": "F",
                }
            )
            rate_bnds = copy_bnds(rate_bnds, bnds, rxnIdx, "rate")

            rxn_type = self.reaction_type(rxn)

            if rxn_type in ["Arrhenius Reaction", "Three Body Reaction"]:
                coeffs.append(
                    [{attr: getattr(rxn.rate, attr) for attr in arrhenius_coefNames}]
                )
                if rxn_type == "Three Body Reaction":
                    coeffs[-1][0]["efficiencies"] = rxn.third_body.efficiencies

                coeffs_bnds.append(
                    {
                        "rate": {
                            attr: {
                                "resetVal": coeffs[-1][0][attr],
                                "value": np.nan,
                                "limits": Uncertainty(
                                    "coef",
                                    rxnIdx,
                                    key="rate",
                                    coef_name=attr,
                                    coeffs_bnds=coeffs_bnds,
                                ),
                                "type": "F",
                            }
                            for attr in arrhenius_coefNames
                        }
                    }
                )

                coeffs_bnds = copy_bnds(
                    coeffs_bnds, bnds, rxnIdx, "coeffs", ["rate", arrhenius_coefNames]
                )

                reset_mech.append(
                    {
                        "reactants": rxn.reactants,
                        "products": rxn.products,
                        "rxnType": rxn_type,
                        "duplicate": rxn.duplicate,
                        "reversible": rxn.reversible,
                        "orders": rxn.orders,
                        "rxnCoeffs": deepcopy(coeffs[-1]),
                    }
                )

            elif rxn_type == "Plog Reaction":
                coeffs.append([])
                coeffs_bnds.append({})
                for n, rate in enumerate(rxn.rate.rates):
                    coeffs[-1].append({"Pressure": rate[0]})
                    coeffs[-1][-1].update(
                        {attr: getattr(rate[1], attr) for attr in arrhenius_coefNames}
                    )
                    if (
                        n == 0 or n == len(rxn.rate.rates) - 1
                    ):  # only going to allow coefficient uncertainties to be placed on upper and lower pressures
                        if n == 0:
                            key = "low_rate"
                        else:
                            key = "high_rate"
                        coeffs_bnds[-1][key] = {
                            attr: {
                                "resetVal": coeffs[-1][-1][attr],
                                "value": np.nan,
                                "limits": Uncertainty(
                                    "coef",
                                    rxnIdx,
                                    key=key,
                                    coef_name=attr,
                                    coeffs_bnds=coeffs_bnds,
                                ),
                                "type": "F",
                            }
                            for attr in arrhenius_coefNames
                        }

                        coeffs_bnds = copy_bnds(
                            coeffs_bnds,
                            bnds,
                            rxnIdx,
                            "coeffs",
                            [key, arrhenius_coefNames],
                        )

                reset_mech.append(
                    {
                        "reactants": rxn.reactants,
                        "products": rxn.products,
                        "rxnType": rxn_type,
                        "duplicate": rxn.duplicate,
                        "reversible": rxn.reversible,
                        "orders": rxn.orders,
                        "rxnCoeffs": deepcopy(coeffs[-1]),
                    }
                )

            elif rxn_type == "Falloff Reaction":
                coeffs_bnds.append({})
                fallof_type = rxn.reaction_type.split("-")[1]

                coeffs.append(
                    {
                        "falloff_type": fallof_type,
                        "high_rate": [],
                        "low_rate": [],
                        "falloff_parameters": list(rxn.rate.falloff_coeffs),
                        "default_efficiency": rxn.third_body.default_efficiency,
                        "efficiencies": rxn.third_body.efficiencies,
                    }
                )
                for key in ["low_rate", "high_rate"]:
                    rate = getattr(rxn.rate, key)
                    coeffs[-1][key] = {
                        attr: getattr(rate, attr) for attr in arrhenius_coefNames
                    }

                    coeffs_bnds[-1][key] = {
                        attr: {
                            "resetVal": coeffs[-1][key][attr],
                            "value": np.nan,
                            "limits": Uncertainty(
                                "coef",
                                rxnIdx,
                                key=key,
                                coef_name=attr,
                                coeffs_bnds=coeffs_bnds,
                            ),
                            "type": "F",
                        }
                        for attr in arrhenius_coefNames
                    }

                    coeffs_bnds = copy_bnds(
                        coeffs_bnds, bnds, rxnIdx, "coeffs", [key, arrhenius_coefNames]
                    )

                key = "falloff_parameters"
                n_coef = len(rxn.rate.falloff_coeffs)
                coeffs_bnds[-1][key] = {
                    n: {
                        "resetVal": coeffs[-1][key][n],
                        "value": np.nan,
                        "limits": Uncertainty(
                            "coef",
                            rxnIdx,
                            key=key,
                            coef_name=n,
                            coeffs_bnds=coeffs_bnds,
                        ),
                        "type": "F",
                    }
                    for n in range(0, n_coef)
                }

                reset_mech.append(
                    {
                        "reactants": rxn.reactants,
                        "products": rxn.products,
                        "rxnType": rxn_type,
                        "duplicate": rxn.duplicate,
                        "reversible": rxn.reversible,
                        "orders": rxn.orders,
                        "falloffType": fallof_type,
                        "rxnCoeffs": deepcopy(coeffs[-1]),
                    }
                )

            elif rxn_type == "Chebyshev Reaction":
                coeffs.append([])
                coeffs_bnds.append({})
                Pmin, Pmax = rxn.rate.pressure_range
                for P, key in ((Pmin, "low_rate"), (Pmax, "high_rate")):
                    coeffs[-1].append({
                        "Pressure": P,
                        "activation_energy": 0.0,
                        "pre_exponential_factor": 1.0,
                        "temperature_exponent": 0.0,
                    })
                    coeffs_bnds[-1][key] = {
                        attr: {
                            "resetVal": coeffs[-1][-1][attr],
                            "value": np.nan,
                            "limits": Uncertainty(
                                "coef",
                                rxnIdx,
                                key=key,
                                coef_name=attr,
                                coeffs_bnds=coeffs_bnds,
                            ),
                            "type": "F",
                        }
                        for attr in arrhenius_coefNames
                    }
                    coeffs_bnds = copy_bnds(
                        coeffs_bnds, bnds, rxnIdx, "coeffs",
                        [key, arrhenius_coefNames],
                    )

                reset_coeffs = {
                    "Pmin": Pmin,
                    "Pmax": Pmax,
                    "Tmin": rxn.rate.temperature_range[0],
                    "Tmax": rxn.rate.temperature_range[1],
                    "coeffs": rxn.rate.data,
                }
                reset_mech.append(
                    {
                        "reactants": rxn.reactants,
                        "products": rxn.products,
                        "rxnType": rxn_type,
                        "duplicate": rxn.duplicate,
                        "reversible": rxn.reversible,
                        "orders": rxn.orders,
                        "rxnCoeffs": reset_coeffs,
                    }
                )

            else:
                coeffs.append({})
                coeffs_bnds.append({})
                reset_mech.append(
                    {
                        "reactants": rxn.reactants,
                        "products": rxn.products,
                        "rxnType": rxn_type,
                    }
                )
                msg = f"{rxn} is a {rxn_type} and is currently unsupported in Frhodo"
                raise (Exception(msg))

    def get_coeffs_keys(self, rxn, coefAbbr, rxnIdx=None):
        """Resolve ``(coef_key, bnds_key)`` for a coefficient abbreviation.

        Args:
            rxn: Cantera reaction whose rate carries the coefficient.
            coefAbbr: Coefficient abbreviation; substrings ``"high"``
                or ``"low"`` route to the high/low-pressure branches
                of Plog and falloff reactions.
            rxnIdx: Optional precomputed reaction index. Required only
                for Plog reactions when ``coefAbbr`` carries ``"high"``;
                otherwise the index is resolved by linear scan.

        Returns:
            ``(coef_key, bnds_key)`` — keys into ``self.coeffs[rxnIdx]``
            and ``self.coeffs_bnds[rxnIdx]``.
        """
        if isinstance(rxn.rate, ct.ArrheniusRate):
            bnds_key = "rate"
            coef_key = 0

        elif isinstance(rxn.rate, (ct.PlogRate, ct.ChebyshevRate)):
            if "high" in coefAbbr:
                if rxnIdx is None:
                    for rxnIdx, mechRxn in enumerate(self.gas.reactions()):
                        if rxn is mechRxn:
                            break

                bnds_key = "high_rate"
                coef_key = len(self.coeffs[rxnIdx]) - 1
            elif "low" in coefAbbr:
                bnds_key = "low_rate"
                coef_key = 0

        elif isinstance(rxn.rate, _FALLOFF_FAMILY):
            if "high" in coefAbbr:
                coef_key = bnds_key = "high_rate"
            elif "low" in coefAbbr:
                coef_key = bnds_key = "low_rate"
            else:
                coef_key = bnds_key = "falloff_parameters"

        return coef_key, bnds_key

    def set_thermo_expression_coeffs(self):
        """Snapshot per-species thermo coefficients into ``self.thermo_coeffs``.

        NASA-7 polynomials only — NASA-9 species are not supported.
        """
        self.thermo_coeffs = []
        for i in range(self.gas.n_species):
            S = self.gas.species(i)
            thermo_dict = {
                "name": S.name,
                "composition": S.composition,
                "charge": S.charge,
                "size": S.size,
                "type": type(S.thermo),
                "P_ref": S.thermo.reference_pressure,
                "T_low": S.thermo.min_temp,
                "T_high": S.thermo.max_temp,
                "coeffs": np.array(S.thermo.coeffs),
                "h_scaler": 1,
                "s_scaler": 1,
            }

            self.thermo_coeffs.append(thermo_dict)

    def modify_reactions(self, coeffs, rxnIdxs=None):
        """Push coefficient changes back into the underlying gas.

        Only changed entries trigger a ``gas.modify_reaction`` call.
        Supports Arrhenius and Troe/SRI falloff; Plog/Chebyshev are a
        no-op and unsupported types are skipped.

        Args:
            coeffs: Parallel structure to ``self.coeffs`` carrying the
                target values.
            rxnIdxs: Reaction indices to update; ``None`` means all.
                Accepts an int for a single reaction.
        """
        if rxnIdxs is None:
            indices = range(len(coeffs))
        elif isinstance(rxnIdxs, (int, np.integer)):
            indices = [int(rxnIdxs)]
        else:
            indices = list(rxnIdxs)

        for rxnIdx in indices:
            rxn = self.gas.reaction(rxnIdx)
            rxnChanged = False

            if isinstance(rxn.rate, ct.ArrheniusRate):
                for coefName in arrhenius_coefNames:
                    if coeffs[rxnIdx][0][coefName] != getattr(rxn.rate, coefName):
                        rxnChanged = True

                if rxnChanged:
                    A = coeffs[rxnIdx][0]["pre_exponential_factor"]
                    b = coeffs[rxnIdx][0]["temperature_exponent"]
                    Ea = coeffs[rxnIdx][0]["activation_energy"]
                    rxn.rate = ct.ArrheniusRate(A, b, Ea)

            elif isinstance(rxn.rate, _FALLOFF_FAMILY):
                # Default each component to the current rate; the loop
                # overrides only those that changed, so unchanged components
                # are still passed through to the new TroeRate/SriRate.
                rate_dict = {
                    "low_rate": rxn.rate.low_rate,
                    "high_rate": rxn.rate.high_rate,
                    "falloff_parameters": rxn.rate.falloff_coeffs,
                }
                for key in rate_dict:
                    if "rate" in key:
                        for coefName in arrhenius_coefNames:
                            if coeffs[rxnIdx][key][coefName] != getattr(
                                getattr(rxn.rate, key), coefName
                            ):
                                rxnChanged = True

                                A = coeffs[rxnIdx][key]["pre_exponential_factor"]
                                b = coeffs[rxnIdx][key]["temperature_exponent"]
                                Ea = coeffs[rxnIdx][key]["activation_energy"]
                                rate_dict[key] = ct.ArrheniusRate(A, b, Ea)
                                break
                    else:
                        length_different = len(coeffs[rxnIdx][key]) != len(
                            rxn.rate.falloff_coeffs
                        )
                        if (
                            length_different
                            or (coeffs[rxnIdx][key] != rxn.rate.falloff_coeffs).any()
                        ):
                            rxnChanged = True

                            if coeffs[rxnIdx]["falloff_type"] == "Troe":
                                if coeffs[rxnIdx][key][-1] == 0.0:
                                    rate_dict[key] = coeffs[rxnIdx][key][:-1]
                                else:
                                    rate_dict[key] = coeffs[rxnIdx][key]
                            else:  # could also be SRI. For optimization this would need to be cast as Troe
                                rate_dict[key] = ct.SriFalloff(coeffs[rxnIdx][key])

                if rxnChanged:
                    if coeffs[rxnIdx]["falloff_type"] == "Troe":
                        rate = ct.TroeRate(
                            rate_dict["low_rate"],
                            rate_dict["high_rate"],
                            rate_dict["falloff_parameters"],
                        )
                    else:
                        rate = ct.SriRate(
                            rate_dict["low_rate"],
                            rate_dict["high_rate"],
                            rate_dict["falloff_parameters"],
                        )

                    rxn.rate = rate

            elif isinstance(rxn.rate, ct.ChebyshevRate):
                pass
            else:
                continue

            if rxnChanged:
                self.gas.modify_reaction(rxnIdx, rxn)

    def _recast_heldout_log_rms(self, rxnIdx, T_fit, P_fit, X, coef_x0):
        """Rate-space log-RMS of the fitted Troe vs the original reaction
        on a grid denser than the fit used.

        The fit grid carries ~9 points for 10 Troe parameters, so its own
        residual understates fidelity — a narrow grid interpolates itself.
        This samples 8 temperatures × 6 pressures spanning the fit grid's
        extents and compares the fitted Troe to the reaction's true rates.
        Must run while ``rxnIdx`` still holds its original rate (before any
        structural rebuild).
        """
        T_lo, T_hi = float(np.min(T_fit)), float(np.max(T_fit))
        P_lo, P_hi = float(np.min(P_fit)), float(np.max(P_fit))
        T_axis = np.linspace(T_lo, T_hi, 8)
        if P_hi > P_lo:
            P_axis = np.geomspace(P_lo, P_hi, 6)
        else:
            P_axis = np.array([P_lo])
        P_grid, T_grid = np.meshgrid(P_axis, T_axis, indexing="ij")

        ln_k_true = np.empty(T_grid.shape)
        M = np.empty(T_grid.shape)
        for a in range(T_grid.shape[0]):
            for b in range(T_grid.shape[1]):
                self.set_TPX(T_grid[a, b], P_grid[a, b], X)
                ln_k_true[a, b] = np.log(self.gas.forward_rate_constants[rxnIdx])
                M[a, b] = self.M(rxnIdx)

        x_lna = np.asarray(coef_x0, dtype=float).copy()
        x_lna[1] = np.log(x_lna[1])
        x_lna[4] = np.log(x_lna[4])
        diff = (ln_Troe(T_grid, M, *x_lna) - ln_k_true).flatten()

        return float(np.sqrt(np.mean(diff * diff)))

    def recast_to_troe(self, rxn_coef_opt, rxn_rate_opt, optimizables):
        """Refit non-Troe pressure-dependent rxns to Troe form in place.

        Walks ``rxn_coef_opt`` and, for each non-Arrhenius reaction,
        fits the rate samples to a Troe parameterization. Mutates
        ``self.coeffs``, ``self.coeffs_bnds``, and ``self.reset_mech``;
        for structurally-different source types (Plog, Chebyshev) also
        regenerates the Cantera ``Solution`` via :meth:`set_mechanism`
        so every reaction is queryable as Troe.

        Args:
            rxn_coef_opt: Per-rxn sampling-grid records from
                :func:`build_rxn_coef_opt`. The ``coefIdx`` /
                ``coefName`` / ``key`` fields are extended in place for
                non-Falloff source types.
            rxn_rate_opt: Rate-level optimization settings from
                :func:`build_rxn_rate_opt`; ``x0`` supplies the
                per-sample ``ln(rate)`` values.
            optimizables: Object exposing
                ``is_coefficient_optimizable(rxnIdx, bnds_key,
                coef_name) -> bool``. Consulted when re-emitting
                per-coefficient bounds for the regenerated mech.

        Returns:
            ``(rxns_changed, mech_rebuilt)``. ``rxns_changed`` lists
            reaction indices whose parameters changed; ``mech_rebuilt``
            is ``True`` iff a structural recast triggered
            :meth:`set_mechanism`. Callers should rebuild
            ``rxn_coef_opt`` / ``rxn_rate_opt`` when ``mech_rebuilt``
            is true.
        """
        reset_mech = self.reset_mech
        rxns_changed: list[int] = []
        mech_rebuilt = False
        recast_log_rms: dict[int, float] = {}
        i = 0
        for rxn_coef in rxn_coef_opt:
            rxnIdx = rxn_coef["rxnIdx"]
            rxn = self.gas.reaction(rxnIdx)
            n_pts = len(rxn_coef["T"])
            if isinstance(rxn.rate, ct.ArrheniusRate):
                i += n_pts
                continue

            if isinstance(rxn.rate, ct.TroeRate):
                i += n_pts
                continue

            T, P, X = rxn_coef["T"], rxn_coef["P"], rxn_coef["X"]
            rates = np.exp(rxn_rate_opt["x0"][i : i + n_pts])
            lb = rxn_coef["coef_bnds"]["lower"]
            ub = rxn_coef["coef_bnds"]["upper"]

            if isinstance(rxn.rate, _FALLOFF_FAMILY):
                rxns_changed.append(rxnIdx)
                rxn_coef["coef_x0"] = fit_generic(
                    rates, T, P, X, rxnIdx, rxn_coef["key"], [],
                    rxn_coef["is_falloff_limit"], self, [lb, ub],
                )
                self.coeffs[rxnIdx]["falloff_type"] = "Troe"
                self.coeffs[rxnIdx]["falloff_parameters"] = rxn_coef["coef_x0"][-4:]
            else:
                rxns_changed.append(rxnIdx)
                rxn_coef["coef_x0"] = fit_generic(
                    rates, T, P, X, rxnIdx, rxn_coef["key"],
                    rxn_coef["coefName"], rxn_coef["is_falloff_limit"],
                    self, [lb, ub],
                )
                rxn_coef["coefIdx"].extend(range(0, 4))
                rxn_coef["coefName"].extend(range(0, 4))
                rxn_coef["key"].extend([
                    {"coeffs": "falloff_parameters", "coeffs_bnds": "falloff_parameters"}
                    for _ in range(0, 4)
                ])
                self.coeffs[rxnIdx] = {
                    "falloff_type": "Troe",
                    "high_rate": {},
                    "low_rate": {},
                    "falloff_parameters": rxn_coef["coef_x0"][-4:],
                    "default_efficiency": 1.0,
                    "efficiencies": {},
                }
                n = 0
                for key in ["low_rate", "high_rate"]:
                    for coefName in arrhenius_coefNames:
                        rxn_coef["key"][n]["coeffs"] = key
                        self.coeffs[rxnIdx][key][coefName] = rxn_coef["coef_x0"][n]
                        n += 1
                rxn_coef["coef_bnds"] = set_bnds(
                    self, rxnIdx, rxn_coef["key"], rxn_coef["coefName"],
                )
                mech_rebuilt = True
                reset_mech[rxnIdx]["rxnType"] = "Falloff Reaction"
                reset_mech[rxnIdx]["rxnCoeffs"] = self.coeffs[rxnIdx]

            recast_log_rms[rxnIdx] = self._recast_heldout_log_rms(
                rxnIdx, T, P, X, rxn_coef["coef_x0"],
            )
            i += n_pts

        if mech_rebuilt:
            bnds = self._freeze_bnds_after_recast(optimizables)
            self.set_mechanism(reset_mech, bnds=bnds)

        self.recast_log_rms = recast_log_rms

        return rxns_changed, mech_rebuilt

    def rebuild_after_structural_recasts(self, optimizables):
        """Apply a single :meth:`set_mechanism` call after batched recasts.

        :func:`~frhodo.simulation.mechanism.recast_single.recast_single_to_troe`
        mutates ``coeffs`` / ``reset_mech`` for Plog and Chebyshev recasts
        but leaves the live Cantera ``Solution`` unchanged. Call this
        once after one or more structural recasts to make the Solution
        catch up.

        Args:
            optimizables: Forwarded to :meth:`_freeze_bnds_after_recast`
                to decide which coef bounds survive the rebuild.
        """
        bnds = self._freeze_bnds_after_recast(optimizables)
        self.set_mechanism(self.reset_mech, bnds=bnds)

    def _freeze_bnds_after_recast(self, optimizables):
        """Bnds dict for :meth:`set_mechanism` after a structural recast.

        Each coefficient comes back as ``{"type": "F", "value": NaN}``
        unless ``optimizables.is_coefficient_optimizable`` opts it in
        and the coefficient is not a falloff-parameter slot (those
        always stay frozen — their tuning lives inside the falloff
        polish, not the per-coefficient bound machinery).
        """
        bnds: dict = {"rate_bnds": [], "coeffs_bnds": []}
        for rxnIdx in range(self.gas.n_reactions):
            coeffs_bnds: dict = {}
            for bndsKey, subRxn in self.coeffs_bnds[rxnIdx].items():
                coeffs_bnds[bndsKey] = {}
                for coefName, coefDict in subRxn.items():
                    coeffs_bnds[bndsKey][coefName] = {
                        "type": "F", "value": np.nan,
                    }
                    opted = optimizables.is_coefficient_optimizable(
                        rxnIdx, bndsKey, coefName,
                    )
                    if opted and bndsKey != "falloff_parameters":
                        coeffs_bnds[bndsKey][coefName] = {
                            "type": coefDict["type"],
                            "value": coefDict["value"],
                        }
            bnds["rate_bnds"].append({
                "value": self.rate_bnds[rxnIdx]["value"],
                "type": self.rate_bnds[rxnIdx]["type"],
            })
            bnds["coeffs_bnds"].append(coeffs_bnds)

        return bnds

    def modify_thermo(self, multipliers):
        """Scale NASA-7 polynomial coefficients per species.

        Species with non-NASA-7 thermo are skipped with a printed
        notice (NASA-9 is not supported).

        Args:
            multipliers: One scalar per species, indexed by gas species
                order. ``coeffs[1:] *= multipliers[i]`` for species ``i``.
        """
        for i in range(np.shape(self.gas.species_names)[0]):
            S_initial = self.gas.species(i)
            S = self.gas.species(i)
            if type(S.thermo) is ct.NasaPoly2:
                # Get current values
                T_low = S_initial.thermo.min_temp
                T_high = S_initial.thermo.max_temp
                P_ref = S_initial.thermo.reference_pressure
                coeffs = S_initial.thermo.coeffs

                # Update thermo properties
                coeffs[1:] *= multipliers[i]
                S.thermo = ct.NasaPoly2(T_low, T_high, P_ref, coeffs)
            else:
                print(
                    "{:.s}'s thermo is type: {:s}".format(
                        self.gas.species_names[i], type(S.thermo)
                    )
                )
                continue

            self.gas.modify_species(i, S)

    def reset(self, rxnIdxs=None, coefNames=None):
        """Revert coefficients to the snapshot taken at load time.

        Args:
            rxnIdxs: Single index, list of indices, or ``None`` for all.
            coefNames: Coefficient selector. For Arrhenius/three-body
                reactions: list of coefficient names. For Plog or
                Falloff: list of ``[limit_type, coefName]`` pairs.
                ``None`` reverts every coefficient in the selected
                reactions.

        Returns:
            Deep copy of ``self.coeffs`` taken before the revert, so
            callers can diff or undo.
        """
        if rxnIdxs is None:
            indices = range(self.gas.n_reactions)
        elif isinstance(rxnIdxs, (int, np.integer)):
            indices = [int(rxnIdxs)]
        else:
            indices = list(rxnIdxs)

        if coefNames is not None and not isinstance(coefNames, list):
            coefNames = [coefNames]

        prior_coeffs = deepcopy(self.coeffs)
        for rxnIdx in indices:
            if coefNames is None:  # resets all coefficients in rxn
                self.coeffs[rxnIdx] = self.reset_mech[rxnIdx]["rxnCoeffs"]

            elif self.reset_mech[rxnIdx]["rxnType"] in [
                "Arrhenius Reaction",
                "Three Body Reaction",
            ]:
                for coefName in coefNames:
                    self.coeffs[rxnIdx][coefName] = self.reset_mech[rxnIdx][
                        "rxnCoeffs"
                    ][coefName]

            elif "Plog Reaction" == self.reset_mech[rxnIdx]["rxnType"]:
                for [limit_type, coefName] in coefNames:
                    if limit_type == "low_rate":
                        self.coeffs[rxnIdx][0][coefName] = self.reset_mech[rxnIdx][
                            "rxnCoeffs"
                        ][0][coefName]
                    elif limit_type == "high_rate":
                        self.coeffs[rxnIdx][-1][coefName] = self.reset_mech[rxnIdx][
                            "rxnCoeffs"
                        ][-1][coefName]

            elif "Falloff Reaction" == self.reset_mech[rxnIdx]["rxnType"]:
                self.coeffs[rxnIdx]["falloff_type"] = self.reset_mech[rxnIdx][
                    "falloffType"
                ]
                for [limit_type, coefName] in coefNames:
                    self.coeffs[rxnIdx][limit_type][coefName] = self.reset_mech[rxnIdx][
                        "rxnCoeffs"
                    ][limit_type][coefName]

        self.modify_reactions(self.coeffs)

        return prior_coeffs

    def set_TPX(self, T, P, X=[]):
        """Validate inputs, then set the gas state without raising.

        Args:
            T: Temperature [K].
            P: Pressure [Pa].
            X: Mole-fraction mapping or composition string; empty
                leaves composition unchanged.

        Returns:
            ``{"success": bool, "message": list[str]}``. ``success``
            is ``False`` for invalid T/P or unknown species; the
            caller decides how to surface the message.
        """
        output = {"success": False, "message": []}
        if T <= 0 or np.isnan(T):
            output["message"].append("Error: Temperature is invalid")
            return output

        elif P <= 0 or np.isnan(P):
            output["message"].append("Error: Pressure is invalid")
            return output

        elif len(X) > 0:
            for species in X:
                if species not in self.gas.species_names:
                    output["message"].append(
                        "Species: {:s} is not in the mechanism".format(species)
                    )
                    return output

            self.gas.TPX = T, P, X

        else:
            self.gas.TP = T, P

        output["success"] = True
        return output

    def M(self, rxnIdx, TPX=[]):
        """Effective third-body concentration [kmol/m³] for reaction ``rxnIdx``.

        For three-body and falloff reactions this is Cantera's
        weighted sum ``Σ_j eff_j · conc_j``. For non-third-body
        reactions (Plog, Arrhenius, ...) Cantera returns NaN; we fall
        back to total molar density so callers using ``M`` as a
        generic normalizer still work.

        Args:
            TPX: Empty for the current gas state, or ``(T, P, X)``
                arrays for a batch evaluation. ``T`` and ``P`` must be
                array-like of equal length; ``X`` is a single
                composition shared across the batch.

        Returns:
            Scalar when ``TPX`` is empty, ``np.ndarray`` shaped like
            ``T`` otherwise.
        """
        def get_M():
            third_body_M = self.gas.third_body_concentrations[rxnIdx]
            if np.isnan(third_body_M):
                return self.gas.density_mole
            return third_body_M

        if len(TPX) == 0:
            return get_M()

        T, P, X = TPX
        M = np.empty(len(T))
        for i in range(len(T)):
            self.set_TPX(T[i], P[i], X)
            M[i] = get_M()
        return M

    def recast_pdep_at_pressure(
        self,
        pressure_pa: float,
        composition,
        *,
        n_points: int = 50,
    ) -> "ChemicalMechanism":
        """Return a new mech with every pressure-dependent rxn flattened
        to Arrhenius form valid at ``pressure_pa``.

        Falloff-family rxns (Lindemann/Troe/SRI/Tsang/FalloffRate) become
        three-body Arrhenius with the source rxn's species efficiencies
        preserved as ``(+M)`` markers — first-order linear in M near the
        recast composition. Plog and Chebyshev become pure Arrhenius
        (their pressure-dependence isn't M-mediated).

        The Arrhenius fit covers the intersection of every species'
        NASA-polynomial validity (and each rxn's intrinsic temperature
        range when defined, e.g. Chebyshev). ``self`` is not modified.

        Args:
            pressure_pa: Recast pressure in pascals.
            composition: Reference composition (dict or Cantera comma
                string). Used only when evaluating falloff rates;
                Plog/Chebyshev are M-independent.
            n_points: Number of T points sampled for the fit.
        """
        eval_gas = ct.Solution(
            thermo="ideal-gas", kinetics="gas",
            species=self.gas.species(), reactions=self.gas.reactions(),
        )
        T_lo = max(s.thermo.min_temp for s in eval_gas.species())
        T_hi = min(s.thermo.max_temp for s in eval_gas.species())

        new_reactions = []
        for rxnIdx, rxn in enumerate(eval_gas.reactions()):
            new_reactions.append(_recast_one_pdep(
                eval_gas, rxn, rxnIdx, pressure_pa, composition,
                T_lo, T_hi, n_points, fit_arrhenius,
            ))

        new_mech = ChemicalMechanism()
        new_mech.gas = ct.Solution(
            thermo="ideal-gas", kinetics="gas",
            species=self.gas.species(), reactions=new_reactions,
        )
        new_mech.isLoaded = True
        new_mech.set_rate_expression_coeffs()
        new_mech.set_thermo_expression_coeffs()

        return new_mech

    def _rebuild_gas_from_reactions(self, reactions):
        """Replace ``self.gas`` with a fresh Solution over ``reactions``.

        Mirrors :meth:`set_mechanism`'s rebuild so coefficient and thermo
        bookkeeping are regenerated to match the new reaction list.
        """
        self.gas = ct.Solution(
            thermo="ideal-gas", kinetics="gas",
            species=self.gas.species(), reactions=reactions,
        )
        self.set_rate_expression_coeffs()
        self.set_thermo_expression_coeffs()

    def recast_reaction_at_pressure(
        self,
        rxnIdx: int,
        pressure_pa: float,
        composition,
        *,
        n_points: int = 50,
    ) -> bool:
        """Recast one pressure-dependent reaction to Arrhenius form in place.

        Uses the same per-reaction fit as :meth:`recast_pdep_at_pressure`:
        falloff-family rxns become three-body Arrhenius with the source
        efficiencies preserved (rate linear in [M]); Plog and Chebyshev
        become pure Arrhenius. The original reaction is stashed so
        :meth:`revert_reaction_recast` can restore it; recasting an
        already-recast reaction keeps the first-stashed original.

        Args:
            rxnIdx: Index of the reaction to recast.
            pressure_pa: Recast pressure in pascals.
            composition: Reference composition for evaluating falloff
                rates (dict or Cantera comma string). Plog/Chebyshev are
                composition-independent.
            n_points: Number of T points sampled for the fit.

        Returns:
            ``True`` if the reaction was recast, ``False`` if it is not
            pressure-dependent or the fit range was empty (no change).
        """
        with self.exclusive():
            reactions = self.gas.reactions()
            original = reactions[rxnIdx]
            if not isinstance(original.rate, _PDEP_FAMILY):
                return False

            eval_gas = ct.Solution(
                thermo="ideal-gas", kinetics="gas",
                species=self.gas.species(), reactions=reactions,
            )
            T_lo = max(s.thermo.min_temp for s in eval_gas.species())
            T_hi = min(s.thermo.max_temp for s in eval_gas.species())
            new_rxn = _recast_one_pdep(
                eval_gas, eval_gas.reaction(rxnIdx), rxnIdx, pressure_pa,
                composition, T_lo, T_hi, n_points, fit_arrhenius,
            )
            if isinstance(new_rxn.rate, _PDEP_FAMILY):
                return False

            if rxnIdx not in self._recast_originals:
                self._recast_originals[rxnIdx] = original
            reactions[rxnIdx] = new_rxn
            self._rebuild_gas_from_reactions(reactions)

        return True

    def revert_reaction_recast(self, rxnIdx: int) -> bool:
        """Restore the pressure-dependent reaction stashed by a recast.

        Args:
            rxnIdx: Index of the reaction to restore.

        Returns:
            ``True`` if a stashed original was restored, ``False`` if the
            reaction had not been recast (no change).
        """
        with self.exclusive():
            original = self._recast_originals.pop(rxnIdx, None)
            if original is None:
                return False

            reactions = self.gas.reactions()
            reactions[rxnIdx] = original
            self._rebuild_gas_from_reactions(reactions)

        return True

    def is_reaction_recast(self, rxnIdx: int) -> bool:
        """Whether ``rxnIdx`` currently holds a recast-from-pdep reaction."""
        recast = rxnIdx in self._recast_originals

        return recast

    def to_yaml_text(self, units: dict | None = None) -> str:
        """Serialize the current mechanism to a Cantera YAML string.

        Args:
            units: Custom unit-system dict, or ``None`` for Cantera
                defaults. Custom units only work for unmodified
                mechanisms; after :meth:`modify_reactions` replaces
                an ``ArrheniusRate`` object, Cantera cannot apply
                non-default units to the detached rate
                (CanteraError on ``AnyValue::applyUnits``).

        Returns:
            YAML serialization of the gas, including any in-place
            coefficient modifications.
        """
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            tmp = pathlib.Path(f.name)
        try:
            if units is None:
                self.gas.write_yaml(str(tmp))
            else:
                self.gas.write_yaml(str(tmp), units=units)
            return tmp.read_text()
        finally:
            tmp.unlink(missing_ok=True)

    def to_chemkin(self, mech_path, sort_species: str = "molar-mass") -> None:
        """Write the current mechanism to a Chemkin ``.mech`` file.

        Wraps :func:`cantera.yaml2ck.convert`. Used by the GUI's
        "Save Mech" action and by any optimize-and-checkpoint loop.

        Args:
            mech_path: Destination path. Overwritten if it exists.
            sort_species: Species ordering hint forwarded to
                ``cantera.yaml2ck.convert``.
        """
        from cantera.yaml2ck import convert as soln2ck
        soln2ck(
            self.gas,
            mechanism_path=str(mech_path),
            sort_species=sort_species,
            overwrite=True,
        )


class Uncertainty:
    """Per-coefficient or per-rate uncertainty interval evaluator.

    Stored on every ``coeffs_bnds`` / ``rate_bnds`` record so the
    optimizer can re-read the limits without recomputing them. The
    instance is called with the reference value ``x`` to produce the
    interval; see :meth:`__call__`.

    Attributes:
        unc_type: ``"rate"`` for rate-level uncertainty, ``"coef"`` for
            coefficient-level.
        rxn_idx: Reaction index this entry refers to.
        unc_dict: Keyword arguments captured at construction; the
            evaluator routes them through to the rate or coefficient
            bound entries.
    """

    def __init__(self, unc_type, rxnIdx, **kwargs):
        self.unc_type = unc_type
        self.rxn_idx = rxnIdx
        self.unc_dict = kwargs

    def _unc_fcn(self, x, uncVal, uncType):
        """Apply one of the bound-shape formulas to ``x``.

        Supported uncertainty types:

        * ``"F"`` — geometric factor: ``[x/f, x·f]``.
        * ``"%"`` — relative: ``[x/(1+p), x·(1+p)]``.
        * ``"±"`` — additive symmetric: ``[x-δ, x+δ]``.
        * ``"+"`` / ``"-"`` — additive one-sided.

        Returns:
            Sorted ``[lower, upper]`` pair. ``[NaN, NaN]`` when
            ``uncVal`` is NaN.
        """
        if np.isnan(uncVal):
            return [np.nan, np.nan]
        if uncType == "F":
            return np.sort([x / uncVal, x * uncVal], axis=0)
        if uncType == "%":
            return np.sort([x / (1 + uncVal), x * (1 + uncVal)], axis=0)
        if uncType == "±":
            return np.sort([x - uncVal, x + uncVal], axis=0)
        if uncType == "+":
            return np.sort([x, x + uncVal], axis=0)
        if uncType == "-":
            return np.sort([x - uncVal, x], axis=0)

        raise ValueError(
            f"unknown uncertainty type {uncType!r}; "
            "expected one of 'F', '%', '±', '+', '-'"
        )

    def __call__(self, x=None):
        """Resolve the current uncertainty value and shape it into a [lo, hi] pair.

        For ``unc_type="rate"``, ``x`` is the current rate; the lookup
        pulls the live uncertainty value from
        ``rate_bnds[rxn_idx]``. For ``unc_type="coef"``, the reset
        value stored at construction is used so the bound stays
        anchored even as the coefficient drifts.

        Returns:
            ``[lower, upper]``. ``[NaN, NaN]`` for falloff parameters
            (no uncertainty bound).
        """
        if self.unc_type == "rate":
            rate_bnds = self.unc_dict["rate_bnds"]
            unc_value = rate_bnds[self.rxn_idx]["value"]
            unc_type = rate_bnds[self.rxn_idx]["type"]
            return self._unc_fcn(x, unc_value, unc_type)
        else:
            coeffs_bnds = self.unc_dict["coeffs_bnds"]
            key = self.unc_dict["key"]
            coefName = self.unc_dict["coef_name"]

            if key == "falloff_parameters":  # falloff parameters have no limits
                return [np.nan, np.nan]

            coef_dict = coeffs_bnds[self.rxn_idx][key][coefName]
            coef_val = coef_dict["resetVal"]
            unc_value = coef_dict["value"]
            unc_type = coef_dict["type"]
            return self._unc_fcn(coef_val, unc_value, unc_type)


def list2ct_mixture(mix) -> str:
    """Format a list of ``(species, mol_frac)`` pairs as a Cantera mixture string."""
    return ", ".join(
        "{!s}:{!r}".format(species, mol_frac) for (species, mol_frac) in mix
    )


def check_rxn_rates(gas) -> list[int]:
    """Flag reactions whose rate constants exceed order-of-magnitude limits.

    Used in failure-diagnostic messages to point the user at the
    likely culprits when an integrator blows up.

    Returns:
        1-indexed reaction numbers (matching Cantera's external
        numbering) whose forward or reverse rate constant exceeds
        the bimolecular / termolecular limit for its reaction order.
    """
    limit = [1e9, 1e15, 1e21]
    rxns = gas.reactions()
    fwd = gas.forward_rate_constants
    rev = gas.reverse_rate_constants
    flagged: list[int] = []
    for rxnIdx, rxn in enumerate(rxns):
        coef_sum = int(sum(rxn.reactants.values()))
        if rxn.reaction_type.startswith("three-body"):
            coef_sum += 1
        if coef_sum > 0 and coef_sum - 1 <= len(limit):
            threshold = limit[coef_sum - 1]
            if fwd[rxnIdx] > threshold or rev[rxnIdx] > threshold:
                flagged.append(rxnIdx + 1)

    return flagged
