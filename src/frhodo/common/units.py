# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level
# directory for license and copyright information.
"""Unit conversions, log-order helpers, and the bisymlog transform."""
import logging
import sys

import numpy as np
import cantera as ct
import numba


log = logging.getLogger(__name__)


_KCAL_PER_MOL_TO_J_PER_KMOL = 4.184e6
_KJ_PER_MOL_TO_J_PER_KMOL = 1.0e6


def kcal_per_mol(value: float) -> float:
    """Convert an energy expressed in kcal/mol to Cantera's internal J/kmol."""
    return value * _KCAL_PER_MOL_TO_J_PER_KMOL


def kJ_per_mol(value: float) -> float:
    """Convert an energy expressed in kJ/mol to Cantera's internal J/kmol."""
    return value * _KJ_PER_MOL_TO_J_PER_KMOL


conv2ct = {
    "torr": 101325 / 760,
    "kPa": 1e3,
    "atm": 101325,
    "bar": 100000,
    "psi": 4.4482216152605 / 0.00064516,
    "cm/s": 1e-2,
    "mm/μs": 1000,
    "ft/s": 1 / 3.28084,
    "in/s": 1 / 39.37007874,
    "mph": 1609.344 / 60**2,
    "kcal": 1 / 4184,
    "cal": 1 / 4.184,
}


PRESSURE_UNITS = ("atm", "bar", "kPa", "torr", "psi", "Pa")
pa_per_unit = {unit: conv2ct.get(unit, 1.0) for unit in PRESSURE_UNITS}


def cgs_factor(name: str, num: dict | None = None):
    """SI→CGS conversion factor for one ``ReactorOutput`` variable.

    ``num`` provides per-reaction ``reac`` / ``prod`` stoichiometric
    sums for ``eq_con``, ``rate_con``, ``rate_con_rev``. Returns
    ``1.0`` (identity) when no conversion is defined.
    """
    table = {
        "conc": 1e-3,
        "wdot": 1e-3,
        "P": 760 / 101325,
        "vel": 1e2,
        "rho": 1e-3,
        "drhodz_tot": 1e-5,
        "drhodz": 1e-5,
        "delta_h": 1e-3 / 4184,
        "h_tot": 1e-3 / 4184,
        "h": 1e-3 / 4184,
        "delta_s": 1 / 4184,
        "s_tot": 1 / 4184,
        "s": 1 / 4184,
        "net_ROP": 1e-3 / 3.8,
        "for_ROP": 1e-3 / 3.8,
        "rev_ROP": 1e-3 / 3.8,
    }
    if name in table:
        return table[name]

    if num is None:
        return 1.0

    if name == "eq_con":
        return 1e3 ** np.array(num["reac"] - num["prod"])[:, None]
    if name == "rate_con":
        return np.power(1e3, num["reac"] - 1)[:, None]
    if name == "rate_con_rev":
        return np.power(1e3, num["prod"] - 1)[:, None]

    return 1.0


@numba.jit(nopython=True, cache=True)
def OoM_numba(x, method="round"):
    """Element-wise order-of-magnitude with a rounding-mode choice.

    Args:
        x: 1-D float array. Zeros are passed through as 1.0 so the
            log doesn't blow up.
        method: ``"round"`` (default), ``"floor"``, ``"ceil"``, or
            ``"exact"`` (no rounding).

    Returns:
        Array of the same shape as ``x`` carrying ``log10(|x|)`` with
        the chosen rounding applied.
    """
    x_OoM = np.empty_like(x)
    for i, xi in enumerate(x):
        if xi == 0.0:
            x_OoM[i] = 1.0

        elif method.lower() == "floor":
            x_OoM[i] = np.floor(np.log10(np.abs(xi)))

        elif method.lower() == "ceil":
            x_OoM[i] = np.ceil(np.log10(np.abs(xi)))

        elif method.lower() == "round":
            x_OoM[i] = np.round(np.log10(np.abs(xi)))

        else:  # "exact"
            x_OoM[i] = np.log10(np.abs(xi))

    return x_OoM


def OoM(x):
    """Floor order-of-magnitude with scalar passthrough.

    Returns a scalar when ``x`` is a scalar, else an ndarray.
    """
    is_array = True
    if any([isinstance(x, _type) for _type in [int, float]]):
        is_array = False
        x = np.array([x])

    if not isinstance(x, np.ndarray):
        x = np.array(x)

    x[x == 0] = 1  # if zero, make OoM 0

    if is_array:
        return OoM_numba(x, method="floor")
    else:
        return OoM_numba(x, method="floor")[0]


def RoundToSigFigs(x, p):
    """Round each element of ``x`` to ``p`` significant figures."""
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


class Convert_Units:
    """Two-way bridge between user-facing units and Cantera-internal SI.

    Construct with a :class:`ChemicalMechanism` so the Arrhenius
    conversions can read the reaction stoichiometry needed to derive
    the molar-mass exponent on the pre-exponential factor.
    """

    def __init__(self, mech):
        self.mech = mech

    def __call__(self, value, units, unit_dir="in"):
        """Convert ``value`` in ``units`` to or from Cantera units.

        Args:
            unit_dir: ``"in"``, ``"2ct"``, ``"to_ct"``, ``"2cantera"``,
                or ``"to_cantera"`` for the user → Cantera direction;
                anything else for Cantera → user.
        """
        units = units.replace("[", "").replace("]", "")
        return self._convert_units(value, units, unit_dir)

    def _basic2Cantera(self, value, units):
        if "K" == units:
            return value
        elif "°C" == units:
            return value + 273.15
        elif "°F" == units:
            return (value - 32) * 5 / 9 + 273.15
        elif "°R" == units:
            return value * 5 / 9
        elif "Pa" == units:
            return value
        elif "m/s" == units:
            return value
        else:
            return value * conv2ct[units]

    def _basic2Display(self, value, units):
        if "K" == units:
            return value
        elif "°C" == units:
            return value - 273.15
        elif "°F" == units:
            return (value - 273.15) * 9 / 5 + 32
        elif "°R" == units:
            return value * 9 / 5
        elif "Pa" == units:
            return value
        elif "m/s" == units:
            return value
        else:
            return value / conv2ct[units]

    def _arrhenius(self, rxnIdx, coeffs, conv_type):
        rxn = self.mech.gas.reaction(rxnIdx)
        coef_sum = sum(rxn.reactants.values())
        if rxn.reaction_type.startswith("three-body"):
            coef_sum += 1

        conv_factor = {
            "Cantera2Bilbo": {
                "A": lambda x: np.log10(x * np.power(1e3, coef_sum - 1)),
                "Ea": lambda x: x / 4.184e6,
            },
            "Cantera2Chemkin": {
                "A": lambda x: x * np.power(1e3, coef_sum - 1),
                "Ea": lambda x: x / 4.184e3,
            },
            "Bilbo2Cantera": {
                "A": lambda x: np.power(10, x) / np.power(1e3, coef_sum - 1),
                "Ea": lambda x: x * 4.184e6,
            },
            "Chemkin2Cantera": {
                "A": lambda x: x / np.power(1e3, coef_sum - 1),
                "Ea": lambda x: x * 4.184e3,
            },
        }

        for coef in coeffs:  # coef of format [coef_name, coef_abbreviation, coef_value]
            if "pre_exponential_factor" in coef:
                if "Cantera2Bilbo" in conv_type:
                    coef[0] = f"log({coef[0]})"  # Corrects shorthand
                else:
                    coef[0] = (
                        coef[0].replace("log(", "").replace(")", "")
                    )  # Corrects shorthand
                with np.errstate(over="raise"):
                    try:
                        coef[2] = conv_factor[conv_type]["A"](coef[2])
                    except Exception as e:
                        coef[2] = sys.float_info.max
                        log.warning("Unit conversion overflow on A; clamped: %s", e)
            elif "activation_energy" in coef:
                if coef[2] != 0:
                    coef[2] = conv_factor[conv_type]["Ea"](coef[2])
        return coeffs

    def _convert_units(self, value, units, unit_dir):
        if unit_dir in ["in", "2ct", "to_ct", "2cantera", "to_cantera"]:
            return self._basic2Cantera(value, units)
        else:
            return self._basic2Display(value, units)


class Bisymlog:
    """Symmetric log transform with a hand-tunable linear knee.

    ``y' = sign(y) · log_base(|y / C| + 1)`` — behaves like a log far
    from zero and like a line near zero. The knee width ``C`` can be
    set explicitly or derived from data via :meth:`set_C_heuristically`.

    Attributes:
        C: Knee scale; smaller values look more log-like for a given
            data range.
        scaling_factor: ``1`` looks log-like, ``2`` looks linear-like;
            forwarded into the heuristic when ``C`` is not given.
        base: Log base for the transform.
    """

    def __init__(self, C=None, scaling_factor=2.0, base=10):
        self.C = C
        self.scaling_factor = scaling_factor
        self.base = base

    def set_C_heuristically(self, y, scaling_factor=None):
        """Pick ``C`` so the transform smoothly straddles ``y``'s range.

        Args:
            scaling_factor: Override the instance's
                ``scaling_factor``. ``1`` looks log-like; ``2`` looks
                linear-like.

        Returns:
            The chosen ``C``. Also stored on ``self.C`` as a side
            effect. Returns ``1/ln(1000)`` when ``y`` is constant
            (degenerate range).
        """
        if scaling_factor is None:
            scaling_factor = self.scaling_factor
        else:
            self.scaling_factor = scaling_factor

        min_y = y.min()
        max_y = y.max()

        if min_y == max_y:
            self.C = None
            return 1 / np.log(1000)

        elif np.sign(max_y) != np.sign(
            min_y
        ):  # if zero is within total range, find largest pos or neg range
            processed_data = [y[y >= 0], y[y <= 0]]
            C = 0
            for data in processed_data:
                range = np.abs(data.max() - data.min())
                if range > C:
                    C = range
                    max_y = data.max()

        else:
            C = np.abs(max_y - min_y)

        C *= 10 ** (OoM(max_y) + self.scaling_factor)
        C = RoundToSigFigs(C, 1)  # round to 1 significant figure

        self.C = C

        return C

    def transform(self, y):
        """Apply the bisymlog transform; NaN passes through.

        Forces ``float64`` output because matplotlib's transform
        pipeline can hand us integer pixel coords and an integer
        array can't carry NaN.
        """
        if self.C is None:
            self.C = self.set_C_heuristically(y)

        if self.C is None:
            return y

        else:
            y = np.asarray(y)
            idx = np.isfinite(y)  # only perform transformation on finite values
            # Float dtype: matplotlib's transform pipeline can hand us
            # integer pixel coords; np.nan would not fit in an int array.
            res = np.zeros_like(y, dtype=np.float64)
            res[~idx] = np.nan
            res[idx] = (
                np.sign(y[idx])
                * np.log10(np.abs(y[idx] / self.C) + 1)
                / np.log10(self.base)
            )

            return res

    def invTransform(self, y):
        """Inverse of :meth:`transform`; requires ``C`` already set.

        Raises:
            Exception: When ``C`` has not been initialized (call
                :meth:`set_C_heuristically` or :meth:`transform` first).
        """
        if self.C is None:
            raise Exception("C is unspecified in Bisymlog")

        y = np.asarray(y)
        idx = np.isfinite(y)  # only perform transformation on finite values
        res = np.zeros_like(y, dtype=np.float64)
        res[~idx] = np.nan
        res[idx] = np.sign(y[idx]) * self.C * (np.power(self.base, np.abs(y[idx])) - 1)

        return res

    def invTransform_derivative(self, y):
        """``d(invTransform)/dy`` evaluated at ``y``.

        Continuous through zero. The inverse maps to
        ``sign(y) · C · (base^|y| - 1)``, whose derivative w.r.t. ``y``
        is ``C · base^|y| · ln(base)``.

        Raises:
            Exception: When ``C`` has not been initialized.
        """
        if self.C is None:
            raise Exception("C is unspecified in Bisymlog")

        y_arr = np.asarray(y)
        result = self.C * np.power(self.base, np.abs(y_arr)) * np.log(self.base)

        return result
