"""Normal-shock jump solver (zones 1, 2, 5).

Solves the perfect-gas + Frosh equations to produce post-incident-shock
(zone 2) and post-reflected-shock (zone 5) conditions from any
self-consistent set of pre-shock measurements.

**Failure contract.** Pre-solve validation errors (missing ``mix``,
non-dict ``mix``, bad species names, non-positive T/P inputs) raise
``ShockJumpError(INPUT_INVALID)`` immediately from ``__init__``. These
are programmer/data errors — they do not surface as ``success=False``.
Convergence errors during the perfect-gas warm-start or the Frosh
root-find set ``self.success = False`` and ``self.error =
<ShockJumpError>`` so callers can branch on a failure-mode-aware retry.
Callers must check ``.success`` before reading ``.res``.

**Result shape.** :meth:`ShockJumpSolver.solve` returns a
:class:`ShockJumpResult` pydantic model. Direct fields (T/P/u/rho per
zone) are populated during ``solve``. Derived quantities — Mach
numbers, sound speeds, ratios of specific heats, driver pressure P4 —
are exposed as ``cached_property`` so callers pay nothing for what they
don't read.
"""
import dataclasses
import logging
from functools import cached_property
from typing import Any


log = logging.getLogger(__name__)

import cantera as ct
import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr
from scipy.optimize import root

from frhodo.common.errors import FailureReason, ShockJumpError


Ru = ct.gas_constant

ALL_VARS: tuple[str, ...] = ("T1", "P1", "u1", "T2", "P2", "T5", "P5")

# Maps shock-variable name → (zone_id, ZoneState field). Replaces the
# implicit ``int(var[1]), var[0]`` parsing — explicit, type-safe, and
# fails loudly on an unknown variable name.
_VAR_LAYOUT: dict[str, tuple[int, str]] = {
    "T1": (1, "T"), "P1": (1, "P"), "u1": (1, "u"),
    "T2": (2, "T"), "P2": (2, "P"),
    "T5": (5, "T"), "P5": (5, "P"),
}

PERFECT_GAS_MAX_ITER = 10
PERFECT_GAS_REL_TOL = 1e-2
INITIAL_PG_GUESS_PA_MPS = (1000.0, 1000.0)  # fallback [P1 (Pa), u1 (m/s)] guess
DEFAULT_FROSH_TOL = 1e-10


@dataclasses.dataclass(slots=True)
class _ZoneState:
    """Per-zone thermodynamic + flow state, mutated in place by the solver."""
    T: float = float("nan")
    P: float = float("nan")
    u: float = float("nan")
    rho: float = float("nan")
    h: float = float("nan")  # specific enthalpy [J/kg]


class ShockJumpResult(BaseModel):
    """Solved shock-tube state from one ``ShockJumpSolver.solve`` call.

    Direct fields are populated during ``solve``. Derived quantities
    (P4, Mach_i, gamma_i, sound speeds) are ``cached_property`` —
    computed on first read, never during ``solve``. Add new derived
    properties here rather than expanding the solver's eager work,
    so callers that don't need them pay nothing.

    Reading any derived property mutates the underlying gas state.
    Callers that need the gas back on driven composition must reset it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    T1: float
    P1: float
    u1: float
    rho1: float

    T2: float
    P2: float
    u2: float
    rho2: float

    T5: float
    P5: float
    u5: float
    rho5: float

    X_driven: dict[str, float]
    X_driver: dict[str, float] | None = None

    _gas: Any = PrivateAttr(default=None)
    _MW_driven: float = PrivateAttr(default=float("nan"))
    _R_specific: float = PrivateAttr(default=float("nan"))

    def _set_gas_zone(self, T: float, P: float) -> None:
        self._gas.TPX = T, P, self.X_driven

    @cached_property
    def gamma1(self) -> float:
        self._set_gas_zone(self.T1, self.P1)
        return float(self._gas.cp / self._gas.cv)

    @cached_property
    def gamma2(self) -> float:
        self._set_gas_zone(self.T2, self.P2)
        return float(self._gas.cp / self._gas.cv)

    @cached_property
    def gamma5(self) -> float:
        self._set_gas_zone(self.T5, self.P5)
        return float(self._gas.cp / self._gas.cv)

    @cached_property
    def a1(self) -> float:
        """Sound speed in zone 1 [m/s]."""
        return float(np.sqrt(self.gamma1 * self._R_specific * self.T1))

    @cached_property
    def a2(self) -> float:
        return float(np.sqrt(self.gamma2 * self._R_specific * self.T2))

    @cached_property
    def a5(self) -> float:
        return float(np.sqrt(self.gamma5 * self._R_specific * self.T5))

    @cached_property
    def Mach1(self) -> float:
        return self.u1 / self.a1

    @cached_property
    def Mach2(self) -> float:
        return self.u2 / self.a2

    @cached_property
    def Mach5(self) -> float:
        return self.u5 / self.a5

    @cached_property
    def P4(self) -> float | None:
        """Driver-section pressure (assumes T4 = T1).

        ``None`` when no driver-section composition (``mix_driver``)
        was supplied to the solver.
        """
        if self.X_driver is None:
            return None

        gas = self._gas
        gas.TPX = self.T1, self.P1, self.X_driven
        gam1 = gas.cp / gas.cv
        gas.TPX = self.T1, self.P1, self.X_driver
        MW4 = gas.mean_molecular_weight
        gam4 = gas.cp / gas.cv

        P2_P1 = self.P2 / self.P1
        a1_a4 = np.sqrt((gam1 / self._MW_driven) / (gam4 / MW4))
        P4_P1 = P2_P1 * np.power(
            1 - ((gam4 - 1) * a1_a4 * P2_P1)
            / np.sqrt(2 * gam1 * (2 * gam1 + (gam1 + 1) * (P2_P1 - 1))),
            2 * gam4 / (1 - gam4),
        )

        return float(P4_P1 * self.P1)


class ShockJumpSolver:
    """Solve normal-shock jump conditions across zones 1, 2, 5.

    See module docstring for the failure contract and result shape.

    Construction runs the solve eagerly: read :attr:`success` (and
    :attr:`error` on failure) before reading :attr:`res`.

    Attributes:
        gas: Cantera ``Solution`` used for all thermo lookups.
        X_driven: Driven-section composition supplied by the caller.
        X_driver: Driver-section composition or ``None`` if absent.
        success: ``True`` when the solve converged; ``False`` after a
            ``ShockJumpError`` was caught in ``__init__``.
        error: The caught ``ShockJumpError``, or ``None`` on success.
        res: The :class:`ShockJumpResult` model. Only valid when
            ``success`` is ``True``.
    """

    def __init__(self, gas, shock_vars):
        self.gas = gas

        shock_vars = dict(shock_vars)
        if "mix" not in shock_vars:
            raise ShockJumpError(
                FailureReason.INPUT_INVALID,
                "shock_vars must contain a 'mix' entry (driven-section composition)",
            )
        mix = shock_vars.pop("mix")
        if not isinstance(mix, dict):
            raise ShockJumpError(
                FailureReason.INPUT_INVALID,
                f"shock_vars['mix'] must be a dict[str, float] of mole fractions, "
                f"got {type(mix).__name__}",
            )
        self.X_driven: dict[str, float] = dict(mix)

        mix_driver = shock_vars.pop("mix_driver", None)
        if mix_driver is not None and not isinstance(mix_driver, dict):
            raise ShockJumpError(
                FailureReason.INPUT_INVALID,
                f"shock_vars['mix_driver'] must be a dict[str, float] of mole "
                f"fractions, got {type(mix_driver).__name__}",
            )
        self.X_driver: dict[str, float] | None = (
            dict(mix_driver) if mix_driver is not None else None
        )

        # Frozen chemistry — MW is constant across zones 1/2/5; cache once.
        # Bad species or malformed composition → INPUT_INVALID.
        try:
            self._MW = self._mean_molecular_weight(self.X_driven)
        except (ct.CanteraError, ValueError) as e:
            raise ShockJumpError(
                FailureReason.INPUT_INVALID,
                f"could not resolve composition against the Cantera mechanism: {e}",
            ) from e
        self._R_specific = Ru / self._MW

        # Frosh inner-loop cache: x signature plus algebraic intermediates.
        # T/P/u/h per zone live on ``self.zones[i]`` (always current after
        # ``_frosh_state_at`` runs); the jacobian reads them straight from
        # there, no separate state object needed.
        self._frosh_cache_x: np.ndarray | None = None
        self._frosh_a: float = float("nan")
        self._frosh_b: float = float("nan")
        self._frosh_u1s: float = float("nan")

        self.success = True
        self.error: ShockJumpError | None = None
        try:
            self.res: ShockJumpResult = self.solve(shock_vars)
        except ShockJumpError as e:
            self.success = False
            self.error = e
            log.warning("Shock jump solver failed: %s", e)

    def _mean_molecular_weight(self, X):
        """MW from composition without disturbing the gas T/P state."""
        self.gas.X = X
        return self.gas.mean_molecular_weight

    def _create_zone(self, shock_vars):
        self.zones: dict[int, _ZoneState] = {i: _ZoneState() for i in (1, 2, 5)}
        for var, val in shock_vars.items():
            zone_id, field = _VAR_LAYOUT[var]
            setattr(self.zones[zone_id], field, val)

    def _get_var(self, var: str) -> float:
        zone_id, field = _VAR_LAYOUT[var]
        return getattr(self.zones[zone_id], field)

    def _set_gas(self, T, P, X=None):
        """Set gas state. ``X=None`` skips composition renormalization.

        Use ``X=None`` only inside the Frosh / perfect-gas inner loops,
        where the composition has already been pinned to ``X_driven``
        by an earlier full TPX assignment in the same solve.
        """
        if T <= 0:
            raise ShockJumpError(
                FailureReason.TEMPERATURE_INVALID,
                f"shock solver: temperature must be positive (got {T} K)",
            )
        if P <= 0:
            raise ShockJumpError(
                FailureReason.PRESSURE_INVALID,
                f"shock solver: pressure must be positive (got {P} Pa)",
            )
        if X is None:
            self.gas.TP = T, P
        else:
            self.gas.TPX = T, P, X

    def _shock_variables(self, known_vars, unknown_vars=(), x=()):
        return {
            **{v: self._get_var(v) for v in known_vars},
            **dict(zip(unknown_vars, x)),
        }

    def _mach1(self):
        """Mach number for zone 1 from current zone state."""
        z = self.zones[1]
        self._set_gas(z.T, z.P, self.X_driven)
        gamma = self.gas.cp / self.gas.cv
        return z.u / np.sqrt(gamma * self._R_specific * z.T)

    def _perfect_gas_shock(self, rel_tol=PERFECT_GAS_REL_TOL,
                           max_iter=PERFECT_GAS_MAX_ITER):
        """Perfect-gas zone 2 + zone 5 jump iteration.

        Populates ``self.zones[2]`` and ``self.zones[5]`` (T, P) in
        place from the current zone-1 state. Used as the warm-start
        for Frosh.

        Raises:
            ShockJumpError: If the gamma-update fixed-point iteration
                does not reach ``rel_tol`` within ``max_iter`` steps.
        """
        z1, z2, z5 = self.zones[1], self.zones[2], self.zones[5]
        gas = self.gas

        T1, P1 = z1.T, z1.P
        self._set_gas(T1, P1, self.X_driven)
        gamma1 = gas.cp / gas.cv
        M1 = self._mach1()

        gamma2 = gamma5 = gamma1
        prior = None
        for _ in range(max_iter):
            gp2 = gamma2 + 1
            gp_gm2 = gp2 / (gamma2 - 1)
            gp5 = gamma5 + 1
            gp_gm5 = gp5 / (gamma5 - 1)

            # Zone 2 (incident shock)
            P2_P1 = 1 + 2 * gamma2 / gp2 * (M1 * M1 - 1)
            z2.P = P1 * P2_P1
            z2.T = T1 * (P2_P1 * (gp_gm2 + P2_P1) / (1 + gp_gm2 * P2_P1))

            # Zone 5 (reflected shock)
            P5_P2 = ((gp_gm5 + 2) * P2_P1 - 1) / (P2_P1 + gp_gm5)
            z5.P = P5_P2 * z2.P
            z5.T = z2.T * P5_P2 * (gp_gm5 + P5_P2) / (1 + gp_gm5 * P5_P2)

            current = np.array([z2.T, z2.P, z5.T, z5.P])
            if prior is not None and np.mean(np.abs((current - prior) / prior)) < rel_tol:
                return  # converged
            prior = current

            # Update gammas for next iteration (frozen chemistry, X already set)
            self._set_gas(z2.T, z2.P)
            gamma2 = (gamma1 + gas.cp / gas.cv) * 0.5
            self._set_gas(z5.T, z5.P)
            gamma5 = (gamma2 + gas.cp / gas.cv) * 0.5

        raise ShockJumpError(
            FailureReason.PERFECT_GAS_NOT_CONVERGED,
            f"perfect-gas shock iteration did not converge in {max_iter} steps "
            f"(rel_tol={rel_tol})",
        )

    def _perfect_gas_shock_zero(self, known_vars, known_vals, x):
        """Zero-finder objective: drive perfect-gas outputs to match known values.

        T1 is always a fixed input — never adjusted by the perfect-gas
        iteration — so its residual is identically zero and is omitted
        from the returned vector.
        """
        self.zones[1].P = x[0]
        self.zones[1].u = x[1]
        self._perfect_gas_shock()
        resolved = self._shock_variables(known_vars)
        return [resolved[v] - val for v, val in zip(known_vars, known_vals) if v != "T1"]

    def _frosh_state_at(self, known_vars, unknown_vars, x):
        """Resolve x → zone state, set Cantera-derived h/rho, cache algebraic intermediates.

        No-op (cache hit) when x matches the previous call, so
        ``_frosh_jacobian`` reuses the work from ``_frosh_residuals``
        within a single SciPy iteration.
        """
        if (self._frosh_cache_x is not None
                and np.array_equal(self._frosh_cache_x, x)):
            return  # cached — self.zones and self._frosh_* are still current

        resolved = self._shock_variables(known_vars, unknown_vars, x)
        z1, z2, z5 = self.zones[1], self.zones[2], self.zones[5]

        z1.u = resolved["u1"]
        z1.T = resolved["T1"]
        z1.P = resolved["P1"]
        # First call this solve: set composition; later calls use TP only.
        self._set_gas(z1.T, z1.P, self.X_driven)
        z1.h = self.gas.enthalpy_mass
        z1.rho = self.gas.density

        z2.T = resolved["T2"]
        z2.P = resolved["P2"]
        self._set_gas(z2.T, z2.P)
        z2.h = self.gas.enthalpy_mass
        z2.rho = self.gas.density
        z2.u = z1.u * z1.rho / z2.rho

        z5.T = resolved["T5"]
        z5.P = resolved["P5"]
        self._set_gas(z5.T, z5.P)
        z5.h = self.gas.enthalpy_mass
        z5.rho = self.gas.density
        z5.u = z2.u * z2.rho / z5.rho

        self._frosh_cache_x = x.copy()
        self._frosh_u1s = z1.u * z1.u
        self._frosh_a = z2.T * z1.P / (z1.T * z2.P)
        self._frosh_b = z5.T * z2.P / (z2.T * z5.P)

    def _frosh_residuals(self, known_vars, unknown_vars, x):
        self._frosh_state_at(known_vars, unknown_vars, x)
        z1, z2, z5 = self.zones[1], self.zones[2], self.zones[5]
        R = self._R_specific
        u1s, a, b = self._frosh_u1s, self._frosh_a, self._frosh_b
        return [
            (z2.P / z1.P - 1) + (u1s * (a - 1) / (R * z1.T)),
            (2 / u1s * (z2.h - z1.h)) + (a * a - 1),
            (z5.P / z2.P - 1) + (u1s / (R * z2.T) * (1 - a) ** 2 / (b - 1)),
            (2 * (z5.h - z2.h) / (u1s * (1 - a) ** 2)) + 2 / (b - 1) + 1,
        ]

    def _frosh_jacobian(self, known_vars, unknown_vars, x):
        """Analytical Jacobian; column per unknown.

        Preallocate-and-assign — concatenate is O(n²) and the per-call
        allocation traffic dominates the algebra at low n; preallocation
        also makes future jacobian additions drop in without re-introducing
        the quadratic pattern.
        """
        self._frosh_state_at(known_vars, unknown_vars, x)
        J = np.empty((4, len(unknown_vars)))
        z1, z2, z5 = self.zones[1], self.zones[2], self.zones[5]
        T1, P1, u1, h1 = z1.T, z1.P, z1.u, z1.h
        T2, P2, h2 = z2.T, z2.P, z2.h
        T5, P5, h5 = z5.T, z5.P, z5.h
        R = self._R_specific
        u1s, a, b = self._frosh_u1s, self._frosh_a, self._frosh_b

        for i, var in enumerate(unknown_vars):
            f1 = f2 = f3 = f4 = 0.0
            if var == "T1":
                f1 = u1s / (R * T1**2) * (1 - 2 * a)
                f2 = -2 / T1 * (h1 / u1s + a**2)
            elif var == "P1":
                f1 = u1s * a / (R * T1 * P1) - P2 / P1**2
                f2 = 2 * a**2 / P1
            elif var == "u1":
                f1 = 2 * u1 / (R * T1) * (a - 1)
                f2 = 4 / u1**3 * (h1 - h2)
                f3 = 2 * u1 / (R * T2) * (1 - a) ** 2 / (b - 1)
                f4 = 4 / u1**3 * (h2 - h5) / (1 - a) ** 2
            elif var == "T2":
                f1 = u1s * a / (R * T1 * T2)
                f2 = 2 / T2 * (h2 / u1s + a**2)
                f3 = u1s / (R * T2**2) * (1 - a) / (b - 1) ** 2 * (1 + a * (1 - 2 * b))
                f4 = 2 / T2 * (
                    1 / u1s * (a * (2 * h5 - h2) - h2) / (1 - a) ** 3
                    + b / (b - 1) ** 2
                )
            elif var == "P2":
                f1 = 1 / P1 - u1s * a / (R * T1 * P2)
                f2 = -2 / P2 * a**2
                f3 = (-P5 / P2**2
                      + u1s / (R * T2 * P2) * (1 - a) / (b - 1) ** 2
                      * (a * (3 * b - 2) - b))
                f4 = -1 / P2 * 2 * b / (b - 1) ** 2
            elif var == "T5":
                f3 = -u1s / (R * T2 * T5) * b * ((1 - a) / (b - 1)) ** 2
                f4 = 2 / T5 * (h5 / (u1s * (1 - a) ** 2) - b / (b**2 - 1))
            elif var == "P5":
                f3 = 1 / P2 + u1s / (R * T2 * P5) * b * ((1 - a) / (b - 1)) ** 2
                f4 = 1 / P5 * 2 * b / (b**2 - 1)
            else:
                raise ShockJumpError(
                    FailureReason.INPUT_INVALID,
                    f"unknown variable {var!r} in jacobian column assembly",
                )
            J[:, i] = (f1, f2, f3, f4)

        return J

    def solve(self, shock_vars, tol=DEFAULT_FROSH_TOL):
        """Run the perfect-gas warm-start and Frosh root-find.

        Args:
            shock_vars: Known shock variables keyed by their layout
                names (``T1``, ``P1``, ``u1``, ``T2``, ``P2``, ``T5``,
                ``P5``). Three independent values are required.
            tol: Tolerance forwarded to :func:`scipy.optimize.root`
                for both Frosh residual norm and step size.

        Returns:
            A :class:`ShockJumpResult` populated with T/P/u/rho per
            zone plus the gas handle for lazy derived properties.

        Raises:
            ShockJumpError: When the perfect-gas pre-solve or the
                Frosh root-find fail to converge.
        """
        self._create_zone(shock_vars)
        self._frosh_cache_x = None  # invalidate cache from any prior solve

        known_vars = tuple(shock_vars.keys())
        unknown_vars = tuple(v for v in ALL_VARS if v not in shock_vars)

        if set(known_vars) == {"T1", "P1", "u1"}:
            self._perfect_gas_shock()
        else:
            # Find [P1, u1] such that the perfect-gas shock outputs match the knowns.
            known_vals = tuple(self._shock_variables(known_vars).values())
            pg_result = root(
                lambda x: self._perfect_gas_shock_zero(known_vars, known_vals, x),
                INITIAL_PG_GUESS_PA_MPS,
                method="hybr",
            )
            if not pg_result.success:
                raise ShockJumpError(
                    FailureReason.PERFECT_GAS_NOT_CONVERGED,
                    f"perfect-gas pre-solve for [P1, u1] did not converge: "
                    f"{pg_result.message}",
                )

        x0 = np.array([self._get_var(v) for v in unknown_vars])

        frosh_result = root(
            lambda x: self._frosh_residuals(known_vars, unknown_vars, x),
            x0,
            method="hybr",
            jac=lambda x: self._frosh_jacobian(known_vars, unknown_vars, x),
            tol=tol,
            options={"xtol": tol},
        )
        if not frosh_result.success:
            raise ShockJumpError(
                FailureReason.FROSH_NOT_CONVERGED,
                f"Frosh shock-jump solver did not converge: {frosh_result.message} "
                f"(unknowns={unknown_vars})",
            )

        z1, z2, z5 = self.zones[1], self.zones[2], self.zones[5]
        result = ShockJumpResult(
            T1=z1.T, P1=z1.P, u1=z1.u, rho1=z1.rho,
            T2=z2.T, P2=z2.P, u2=z2.u, rho2=z2.rho,
            T5=z5.T, P5=z5.P, u5=z5.u, rho5=z5.rho,
            X_driven=dict(self.X_driven),
            X_driver=dict(self.X_driver) if self.X_driver is not None else None,
        )
        # Inject implementation state so lazy properties (Mach_i, gamma_i, P4)
        # can do per-property gas work without touching the solver.
        result._gas = self.gas
        result._MW_driven = self._MW
        result._R_specific = self._R_specific
        return result
