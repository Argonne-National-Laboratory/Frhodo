"""Reactor-output variables: registry, accessors, and observable computers.

Three layers in this file:

1. ``ReactorVar`` registry — one record per ``ReactorOutput`` attribute,
   declaring display name, sub-type variant, Cantera ``SolutionArray``
   attribute (or computed callable), and ``observable_default`` flag.
2. ``ReactorOutput`` + ``SimProperty`` — runtime accessors that consume
   the registry. ``SimProperty`` is a lazy per-attribute accessor with
   SI / CGS caches; ``ReactorOutput`` collects them under one parent.
3. ``drhodz`` / ``drhodz_per_rxn`` — density-gradient computers as
   trajectory-shape post-hoc observables; the single-state shape
   variants used by the SUNDIALS sensitivity callbacks live in
   :mod:`frhodo.simulation.shock.observables` and share the same
   formula. ``tests/engine/test_observables_consistency.py`` enforces
   that the two implementations stay in sync.

The same module also pins the ``observable_default`` flag to the set
of observables the sensitivity pipeline supports; that flag-vs-
:data:`~frhodo.simulation.shock.observables.OBSERVABLES` consistency
is also asserted by the test above.
"""
from dataclasses import dataclass
from typing import Callable, Literal

import cantera as ct
import numpy as np

from frhodo.common.units import cgs_factor


Ru = ct.gas_constant


# ────────────────────────────── observables ──────────────────────────────
def _area_change_term(states, L, As, A1):
    n = 0.5
    z = states.z
    A = states.A
    rho = states.density
    vel = states.vel
    T = states.T
    cp = states.cp_mass
    Wmix = states.mean_molecular_weight

    beta = vel ** 2 * (1.0 / (cp * T) - Wmix / (Ru * T))
    xi = np.maximum(z / L, 1e-10)
    dAdt = vel * As * n / L * xi ** (n - 1.0) / (1.0 - xi ** n) ** 2.0

    return rho * beta / A * dAdt


def drhodz(states, L=0.1, As=0.2, A1=0.2, area_change=False):
    """Total density gradient along a ``ct.SolutionArray`` trajectory.

    Trajectory-shape variant of the formula in
    :mod:`frhodo.simulation.shock.observables`; the two implementations
    are tested for agreement at a sample state in
    ``tests/engine/test_observables_consistency.py``.
    """
    vel = states.vel
    T = states.T
    cp = states.cp_mass
    Wmix = states.mean_molecular_weight
    hk = states.partial_molar_enthalpies
    wdot = states.net_production_rates

    beta = vel ** 2 * (1.0 / (cp * T) - Wmix / (Ru * T))
    species_term = np.sum(
        (hk / (cp * T)[:, None] - Wmix[:, None]) * wdot, axis=1,
    )
    area_term = _area_change_term(states, L, As, A1) if area_change else 0.0

    return (species_term - area_term) / (vel * (1.0 + beta))


def drhodz_per_rxn(states, L=0.1, As=0.2, A1=0.2, area_change=False, rxnNum=None):
    """Per-reaction density gradient along a trajectory.

    Args:
        rxnNum: Single reaction index, list of indices, or ``None`` for
            all reactions.

    Returns:
        Array of shape ``(n_steps, n_selected_rxns)``. The identity
        ``sum(axis=1) == drhodz`` holds only when ``area_change=False``.
    """
    vel = states.vel
    T = states.T
    cp = states.cp_mass
    Wmix = states.mean_molecular_weight
    nu_fwd = states.product_stoich_coeffs
    nu_rev = states.reactant_stoich_coeffs
    delta_N_full = np.sum(nu_fwd, axis=0) - np.sum(nu_rev, axis=0)

    if rxnNum is None:
        rxns = slice(None)
    elif isinstance(rxnNum, list):
        rxns = rxnNum
    else:
        rxns = [rxnNum]

    rj = states.net_rates_of_progress[:, rxns]
    hj = states.delta_enthalpy[:, rxns]
    delta_N = delta_N_full[rxns]

    beta = vel ** 2 * (1.0 / (cp * T) - Wmix / (Ru * T))
    species_term = rj * (
        hj / (cp * T)[:, None] - Wmix[:, None] * delta_N
    )
    area_term = (
        _area_change_term(states, L, As, A1)[:, None] if area_change else 0.0
    )

    return (species_term - area_term) / (vel[:, None] * (1.0 + beta[:, None]))


# ────────────────────────────── registry ──────────────────────────────
def _compute_drhodz_tot(parent):
    return drhodz(parent.states)


def _compute_drhodz_per_rxn(parent):
    return drhodz_per_rxn(parent.states)


def _compute_perc_drhodz(parent):
    drhodz_tot = parent.drhodz_tot(units="SI")[:, None]
    drhodz_per = parent.drhodz(units="SI").T
    if not np.any(drhodz_tot):
        return np.zeros_like(drhodz_per)

    return drhodz_per / np.abs(drhodz_tot) * 100


def _compute_perc_abs_drhodz(parent):
    drhodz_tot = parent.drhodz_tot(units="SI")[:, None]
    drhodz_per = parent.drhodz(units="SI").T
    if not np.any(drhodz_tot):
        return np.zeros_like(drhodz_per)

    return drhodz_per / np.abs(drhodz_per).sum(axis=1)[:, None] * 100


@dataclass(frozen=True)
class ReactorVar:
    """One record in the :data:`REACTOR_VARS` registry.

    Attributes:
        sim_name: Attribute name on :class:`ReactorOutput`; also the
            identifier used by code (``"T"``, ``"drhodz_tot"``, ...).
        display: User-facing label shown in the GUI.
        sub_type: ``"total"`` for scalar-per-step, ``"species"`` for one
            value per species per step, ``"rxn"`` for one value per
            reaction per step, or ``None`` when the distinction is
            irrelevant.
        cantera_attr: Attribute on ``ct.SolutionArray`` to read for the
            value, or ``None`` when ``compute`` is provided instead.
        compute: ``compute(parent) -> np.ndarray`` to compute the value
            from a parent :class:`ReactorOutput`. Mutually exclusive
            with ``cantera_attr``.
        observable_default: When ``True``, this variable appears in the
            GUI's observable dropdown and the sensitivity pipeline
            supports it. The set of ``True`` flags must match
            :data:`OBSERVABLES`; consistency is enforced by
            ``tests/engine/test_observables_consistency.py``.
    """
    sim_name: str
    display: str
    sub_type: Literal["total", "species", "rxn"] | None = None
    cantera_attr: str | None = None
    compute: Callable[[object], np.ndarray] | None = None
    observable_default: bool = False


REACTOR_VARS: tuple[ReactorVar, ...] = (
    ReactorVar("t_lab", "Laboratory Time", None, "t"),
    ReactorVar("t_shock", "Shockwave Time", None, "t_shock"),
    ReactorVar("z", "Position", None, "z"),
    ReactorVar("A", "Cross Section", None, "A"),
    ReactorVar("vel", "Gas Velocity", None, "vel"),
    ReactorVar("T", "Temperature", None, "T", observable_default=True),
    ReactorVar("P", "Pressure", None, "P", observable_default=True),
    ReactorVar("h_tot", "Enthalpy", "total", "enthalpy_mole"),
    ReactorVar("h", "Enthalpy", "species", "partial_molar_enthalpies"),
    ReactorVar("s_tot", "Entropy", "total", "entropy_mole"),
    ReactorVar("s", "Entropy", "species", "partial_molar_entropies"),
    ReactorVar("rho", "Density", None, "density"),
    ReactorVar("drhodz_tot", "Density Gradient", "total",
               compute=_compute_drhodz_tot, observable_default=True),
    ReactorVar("drhodz", "Density Gradient", "rxn",
               compute=_compute_drhodz_per_rxn),
    ReactorVar("perc_drhodz", "% Density Gradient", "rxn",
               compute=_compute_perc_drhodz),
    ReactorVar("perc_abs_drhodz", "± % |Density Gradient|", "rxn",
               compute=_compute_perc_abs_drhodz),
    ReactorVar("Y", "Mass Fraction", "species", "Y", observable_default=True),
    ReactorVar("X", "Mole Fraction", "species", "X", observable_default=True),
    ReactorVar("conc", "Concentration", "species", "concentrations",
               observable_default=True),
    ReactorVar("wdot", "Net Production Rate", "species", "net_production_rates"),
    ReactorVar("wdotfor", "Creation Rate", "species", "creation_rates"),
    ReactorVar("wdotrev", "Destruction Rate", "species", "destruction_rates"),
    ReactorVar("HRR_tot", "Heat Release Rate", "total", "heat_release_rate",
               observable_default=True),
    ReactorVar("HRR", "Heat Release Rate", "rxn", "heat_production_rates"),
    ReactorVar("delta_h", "Delta Enthalpy (Heat of Reaction)", "rxn",
               "delta_enthalpy"),
    ReactorVar("delta_s", "Delta Entropy", "rxn", "delta_entropy"),
    ReactorVar("eq_con", "Equilibrium Constant", "rxn", "equilibrium_constants"),
    ReactorVar("rate_con", "Forward Rate Constant", "rxn",
               "forward_rate_constants"),
    ReactorVar("rate_con_rev", "Reverse Rate Constant", "rxn",
               "reverse_rate_constants"),
    ReactorVar("net_ROP", "Net Rate of Progress", "rxn", "net_rates_of_progress"),
    ReactorVar("for_ROP", "Forward Rate of Progress", "rxn",
               "forward_rates_of_progress"),
    ReactorVar("rev_ROP", "Reverse Rate of Progress", "rxn",
               "reverse_rates_of_progress"),
)


BY_SIM_NAME: dict[str, ReactorVar] = {v.sim_name: v for v in REACTOR_VARS}
# Display-string → sim_name lookup for observables. Driven by the
# ``observable_default`` flag; ``tests/engine/test_observables_consistency.py``
# asserts this set equals :data:`observables.OBSERVABLES` so the
# GUI dropdown and the sensitivity-supported set stay in sync.
BY_DISPLAY_OBSERVABLE: dict[str, str] = {
    v.display: v.sim_name for v in REACTOR_VARS if v.observable_default
}


def _build_variants_by_display() -> dict[str, tuple[ReactorVar, ...]]:
    out: dict[str, list[ReactorVar]] = {}
    for v in REACTOR_VARS:
        out.setdefault(v.display, []).append(v)

    return {k: tuple(v) for k, v in out.items()}


VARIANTS_BY_DISPLAY: dict[str, tuple[ReactorVar, ...]] = _build_variants_by_display()


def sub_types_for_display(display: str) -> list[str] | None:
    """Sub-types registered for a display label, or ``None`` if the label
    has no sub-typed variants."""
    types = [
        v.sub_type
        for v in VARIANTS_BY_DISPLAY[display]
        if v.sub_type is not None
    ]

    return types if types else None


def base_sim_name_for_display(display: str) -> str:
    """The sim_name for a display label.

    Returns:
        The non-``total`` variant if multiple exist, else the only
        variant.
    """
    variants = VARIANTS_BY_DISPLAY[display]
    for v in variants:
        if v.sub_type != "total":
            return v.sim_name

    return variants[0].sim_name


# ────────────────────────────── accessors ──────────────────────────────
class SimProperty:
    """Lazy per-attribute accessor with SI / CGS caches.

    ``self.var`` is the ``ReactorVar`` metadata. First call to
    ``__call__`` resolves the value via either ``var.compute(parent)``
    or ``getattr(parent.states, var.cantera_attr)`` and caches it.
    """

    def __init__(self, name, parent=None):
        self.name = name
        self.var = BY_SIM_NAME[name]
        self.parent = parent
        self.conversion = None
        self.value = {"SI": np.array([]), "CGS": np.array([])}
        self.ndim = self.value["SI"].ndim

    def clear(self):
        self.value = {"SI": np.array([]), "CGS": np.array([])}
        self.ndim = self.value["SI"].ndim

    def __call__(self, idx=None, units="CGS"):
        if len(self.value["SI"]) == 0 or np.isnan(self.value["SI"]).all():
            parent = self.parent
            if self.var.compute is not None:
                self.value["SI"] = self.var.compute(parent)
            else:
                self.value["SI"] = getattr(parent.states, self.var.cantera_attr)

            if self.value["SI"].ndim > 1:
                self.value["SI"] = self.value["SI"].T

            self.ndim = self.value["SI"].ndim

        if units == "CGS" and len(self.value["CGS"]) == 0:
            if self.conversion is None:
                self.value["CGS"] = self.value["SI"]
            else:
                self.value["CGS"] = self.conversion(self.value["SI"])

        return self.value[units]


class ReactorOutput:
    """Collection of :class:`SimProperty` accessors on top of a ``ct.SolutionArray``.

    Construct with ``num=None`` for an empty placeholder; otherwise
    pass a per-reaction stoichiometry/count dict to wire CGS unit
    conversions on each property.

    Attributes:
        states: Underlying ``ct.SolutionArray`` carrying the trajectory.
        reactor_var: ``{display_label: sim_name}`` mapping for the
            variables this output exposes.
        independent_var: Set by :meth:`finalize`; convenience handle
            for the trajectory's x-axis.
        observable: Set by :meth:`finalize`; convenience handle for the
            configured observable.
        success: ``True`` after :meth:`finalize` if the underlying solve
            completed.
    """

    def __init__(self, num=None, states=None, reactor_vars=[]):
        self.states = states
        self.reactor_var = {}
        for var in reactor_vars:
            entry = BY_SIM_NAME.get(var)
            if entry is not None:
                self.reactor_var[entry.display] = var

        if num is None:
            self.reactor_var = {}
            return

        for name in reactor_vars:
            prop = SimProperty(name, parent=self)
            factor = cgs_factor(name, num)
            if not (np.isscalar(factor) and factor == 1.0):
                prop.conversion = lambda x, s=factor: x * s
            setattr(self, name, prop)

    def set_independent_var(self, ind_var, units="CGS"):
        """Cache the named variable's array as ``self.independent_var``."""
        self.independent_var = getattr(self, ind_var)(units=units)

    def set_observable(self, observable, units="CGS"):
        """Cache the selected observable's slice as ``self.observable``.

        Args:
            observable: ``{"main": display_label, "sub": index}``. For
                per-species or per-reaction observables, ``sub`` selects
                the column.

        Raises:
            ValueError: If ``observable["main"]`` is not in
                :data:`BY_DISPLAY_OBSERVABLE`.
        """
        sim_name = BY_DISPLAY_OBSERVABLE.get(observable["main"])
        if sim_name is None:
            raise ValueError(
                f"unknown observable: {observable['main']!r}; "
                f"valid: {sorted(BY_DISPLAY_OBSERVABLE)}"
            )
        self.observable = getattr(self, sim_name)(units=units)

        if self.observable.ndim > 1:
            self.observable = self.observable[observable["sub"]]

    def finalize(self, success, ind_var, observable, units="CGS"):
        """Wire ``independent_var``, ``observable``, and ``success`` after a solve."""
        self.set_independent_var(ind_var, units)
        self.set_observable(observable, units)
        self.success = success
