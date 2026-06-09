"""Tests for the Arrhenius / falloff fitting kernels in calculate/optimize/fit_coeffs.

These hit pure-numerical code paths and run independently of any Cantera
mechanism: ``ln_arrhenius_k`` is the analytic Arrhenius form, and
``fit_arrhenius`` is a regression of (T, k) data back to (Ea, A, n).
"""
import numpy as np
import pytest

from frhodo.simulation.mechanism.coef_helpers import set_arrhenius_bnds
from frhodo.simulation.mechanism.fit_coeffs import fit_arrhenius
from frhodo.simulation.mechanism.troe_kernels import Ru, ln_arrhenius_k


class TestLnArrheniusK:
    """``ln k = ln A + n*ln T - Ea / (Ru * T)``."""

    def test_zero_activation_zero_n_recovers_lnA(self):
        T = np.array([300.0, 1000.0, 2000.0])
        ln_k = ln_arrhenius_k(T, Ea=0.0, ln_A=12.5, n=0.0)
        np.testing.assert_allclose(ln_k, 12.5, rtol=1e-12)

    def test_matches_analytic_form(self):
        T = np.array([500.0, 1000.0, 1500.0, 2000.0])
        Ea, ln_A, n = 1.0e8, 25.0, 0.5  # J/kmol, ln(cm^3...), dimensionless
        expected = ln_A + n * np.log(T) - Ea / (Ru * T)
        np.testing.assert_allclose(
            ln_arrhenius_k(T, Ea, ln_A, n),
            expected,
            rtol=1e-12,
        )

    def test_temperature_exponent_increases_with_lnT(self):
        """With Ea=0 and n=1, ln(k) - ln(A) = ln(T)."""
        T = np.array([100.0, 1000.0, 10000.0])
        ln_k = ln_arrhenius_k(T, Ea=0.0, ln_A=20.0, n=1.0)
        np.testing.assert_allclose(ln_k - 20.0, np.log(T), rtol=1e-12)


class TestFitArrhenius:
    """Round-trip: synthesize k(T) from known (Ea, A, n), fit, recover."""

    @pytest.mark.parametrize(
        "Ea,A,n",
        [
            (5.0e7, 1.0e13, 0.0),  # classic Arrhenius
            (1.5e8, 3.2e10, 0.5),  # modified Arrhenius
            (0.0, 1.0e12, 2.0),  # purely T-dependent
        ],
    )
    def test_recovers_known_arrhenius_parameters(self, Ea, A, n):
        T = np.linspace(800.0, 2500.0, 25)
        ln_k = np.log(A) + n * np.log(T) - Ea / (Ru * T)
        k = np.exp(ln_k)

        coeffs = fit_arrhenius(k, T)

        # fit_arrhenius returns coeffs ordered by default_arrhenius_coefNames =
        # ["activation_energy", "pre_exponential_factor", "temperature_exponent"]
        Ea_fit, A_fit, n_fit = coeffs
        assert Ea_fit == pytest.approx(Ea, abs=1.0e3, rel=1e-3), (
            f"Ea: expected {Ea:g}, got {Ea_fit:g}"
        )
        assert A_fit == pytest.approx(A, rel=1e-3), (
            f"A: expected {A:g}, got {A_fit:g}"
        )
        assert n_fit == pytest.approx(n, abs=1e-3), (
            f"n: expected {n}, got {n_fit}"
        )


class TestSetArrheniusBnds:
    """Bounds for the standalone Arrhenius optimizer.

    The lower/upper bound for ``activation_energy`` flips depending on
    the sign of ``x0[Ea]``: Ea is not allowed to cross zero. ``A`` and
    ``n`` get global numeric limits.
    """

    COEF_NAMES = ["activation_energy", "pre_exponential_factor", "temperature_exponent"]

    def test_positive_Ea_lower_is_zero(self):
        x0 = [5.0e7, 1.0e13, 0.0]
        lower, upper = set_arrhenius_bnds(x0, self.COEF_NAMES)
        assert lower[0] == 0.0, (
            f"positive Ea must have lower=0 (no sign change), got {lower[0]}"
        )
        assert upper[0] > 0, "positive Ea must have positive upper bound"

    def test_negative_Ea_upper_is_zero(self):
        x0 = [-5.0e7, 1.0e13, 0.0]
        lower, upper = set_arrhenius_bnds(x0, self.COEF_NAMES)
        assert upper[0] == 0.0, (
            f"negative Ea must have upper=0 (no sign change), got {upper[0]}"
        )
        assert lower[0] < 0, "negative Ea must have negative lower bound"

    def test_pre_exponential_lower_is_strictly_positive(self):
        """A must stay positive — log(A) is taken downstream."""
        x0 = [0.0, 1.0e13, 0.0]
        lower, _ = set_arrhenius_bnds(x0, self.COEF_NAMES)
        assert lower[1] > 0, f"A lower bound must be > 0, got {lower[1]}"

    def test_bounds_are_ordered(self):
        x0 = [5.0e7, 1.0e13, 0.5]
        lower, upper = set_arrhenius_bnds(x0, self.COEF_NAMES)
        for i, name in enumerate(self.COEF_NAMES):
            assert lower[i] <= upper[i], (
                f"{name}: lower={lower[i]} exceeds upper={upper[i]}"
            )
