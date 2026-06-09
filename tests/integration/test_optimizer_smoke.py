"""Smoke test for the optimizer's hot path.

Exercises ``CostFunction`` construction + one cost-function evaluation against
a synthetic shock. Builds ``coef_opt`` / ``rxn_coef_opt`` /
``rxn_rate_opt`` via the free functions in ``frhodo.optimize.parameters``
— no ``Multithread_Optimize`` instance, no Qt.
"""
import multiprocessing as mp

import cantera as ct
import numpy as np
import pytest
import scipy.stats

from frhodo.experiment import ExperimentalShock
from frhodo.simulation.shock.state import RuntimeReactorState
from frhodo.optimize._worker_context import MechBuildPayload
from frhodo.optimize.cost.fit_fcn import CostFunction, initialize_parallel_worker
from frhodo.optimize.parameters import (
    build_rxn_coef_opt,
    build_rxn_rate_opt,
)
from frhodo.optimize.cost.settings import CostSettings


def _set_one_arrhenius_optimizable(mech):
    """Mark the first Arrhenius reaction's A/n/Ea as optimizable, with
    realistic bounds (factor-of-2 uncertainty). Default mechanism state
    has NaN bounds; the GUI sets them via the Bounds widget. The
    optimizer assumes non-NaN when opt=True. Returns cleanup info.
    """
    from frhodo.optimize.parameters import OptimizableSetBuilder

    arrh_idx = next(
        i for i, r in enumerate(mech.gas.reactions())
        if type(r.rate) is ct.ArrheniusRate
    )
    rate_orig = (mech.rate_bnds[arrh_idx]["value"],
                 mech.rate_bnds[arrh_idx]["type"])
    mech.rate_bnds[arrh_idx]["value"] = 2.0
    mech.rate_bnds[arrh_idx]["type"] = "F"

    bnds_key = next(iter(mech.coeffs_bnds[arrh_idx]))
    coefNames = list(mech.coeffs_bnds[arrh_idx][bnds_key])
    coef_orig = {}
    builder = OptimizableSetBuilder()
    builder.set_reaction_optimizable(arrh_idx, True)
    for c in coefNames:
        d = mech.coeffs_bnds[arrh_idx][bnds_key][c]
        coef_orig[c] = (d["value"], d["type"])
        d["value"] = 2.0
        d["type"] = "F"
        builder.set_coefficient_optimizable(arrh_idx, bnds_key, c, True)
    return arrh_idx, bnds_key, coefNames, rate_orig, coef_orig, builder


def _clear_opts(mech, arrh_idx, bnds_key, coefNames, rate_orig, coef_orig, _builder):
    (mech.rate_bnds[arrh_idx]["value"],
     mech.rate_bnds[arrh_idx]["type"]) = rate_orig
    for c in coefNames:
        d = mech.coeffs_bnds[arrh_idx][bnds_key][c]
        d["value"], d["type"] = coef_orig[c]


def _set_one_troe_optimizable(mech):
    """Mark Cycloheptane's Troe rxn (idx 36, C2H6 (+M) <=> 2 CH3 (+M))
    rate-coefs optimizable (low_rate + high_rate). Falloff_parameters are
    keyed by int and are not user-toggled through OptimizableSetBuilder;
    the orchestrator extends rxn_coef with them implicitly."""
    from frhodo.optimize.parameters import OptimizableSetBuilder

    troe_idx = next(
        i for i, r in enumerate(mech.gas.reactions())
        if isinstance(r.rate, ct.TroeRate)
    )

    rate_orig = (mech.rate_bnds[troe_idx]["value"],
                 mech.rate_bnds[troe_idx]["type"])
    mech.rate_bnds[troe_idx]["value"] = 2.0
    mech.rate_bnds[troe_idx]["type"] = "F"

    bnds_keys = ["low_rate", "high_rate"]
    coef_orig = {}

    builder = OptimizableSetBuilder()
    builder.set_reaction_optimizable(troe_idx, True)
    for bk in bnds_keys:
        coef_orig[bk] = {}
        for c in mech.coeffs_bnds[troe_idx][bk]:
            d = mech.coeffs_bnds[troe_idx][bk][c]
            coef_orig[bk][c] = (d["value"], d["type"])
            d["value"] = 2.0
            d["type"] = "F"
            builder.set_coefficient_optimizable(troe_idx, bk, c, True)

    return troe_idx, bnds_keys, rate_orig, coef_orig, builder


def _clear_troe_opts(mech, troe_idx, bnds_keys, rate_orig, coef_orig, _builder):
    (mech.rate_bnds[troe_idx]["value"],
     mech.rate_bnds[troe_idx]["type"]) = rate_orig
    for bk in bnds_keys:
        for c in coef_orig[bk]:
            d = mech.coeffs_bnds[troe_idx][bk][c]
            d["value"], d["type"] = coef_orig[bk][c]


def _set_specific_rxns_optimizable(mech, rxn_F_pairs):
    """Mark a specific set of rxns + all their rate coefs optimizable.

    ``rxn_F_pairs`` is ``[(rxnIdx, F_factor), ...]`` matching the user's
    reported scenario where each rxn carries its own rate-uncertainty
    factor. F=2 means k can vary within a factor-of-2 band; F=1.25 is
    a tighter ±25% constraint.
    """
    from frhodo.optimize.parameters import OptimizableSetBuilder

    rate_orig: list = []
    coef_orig: list = []
    builder = OptimizableSetBuilder()

    for rxnIdx, F in rxn_F_pairs:
        rate_orig.append((
            rxnIdx,
            mech.rate_bnds[rxnIdx]["value"],
            mech.rate_bnds[rxnIdx]["type"],
        ))
        mech.rate_bnds[rxnIdx]["value"] = F
        mech.rate_bnds[rxnIdx]["type"] = "F"

        builder.set_reaction_optimizable(rxnIdx, True)
        per_rxn: dict = {}
        for bk in mech.coeffs_bnds[rxnIdx]:
            if bk == "falloff_parameters":
                continue
            per_rxn[bk] = {}
            for c in mech.coeffs_bnds[rxnIdx][bk]:
                if not isinstance(c, str):
                    continue
                d = mech.coeffs_bnds[rxnIdx][bk][c]
                per_rxn[bk][c] = (d["value"], d["type"])
                d["value"] = F
                d["type"] = "F"
                builder.set_coefficient_optimizable(rxnIdx, bk, c, True)
        coef_orig.append((rxnIdx, per_rxn))

    return rate_orig, coef_orig, builder


def _clear_specific_opts(mech, rate_orig, coef_orig, _builder):
    for rxnIdx, val, typ in rate_orig:
        mech.rate_bnds[rxnIdx]["value"] = val
        mech.rate_bnds[rxnIdx]["type"] = typ
    for rxnIdx, per_rxn in coef_orig:
        for bk, per_bk in per_rxn.items():
            for c, (val, typ) in per_bk.items():
                d = mech.coeffs_bnds[rxnIdx][bk][c]
                d["value"], d["type"] = val, typ


@pytest.fixture
def user_scenario_optimizer_setup(loaded_cycloheptane):
    """User-reported scenario: rxns R1, R2, R3, R5, R8 (1-indexed) with
    rate-factor bounds 2, 2, 2, 2, 1.25 respectively. All Arrhenius
    coefficients optimizable. Reproduces the path that produced
    ``A=-12.7`` in the user's session."""
    from frhodo.experiment import ExperimentalShock

    mech = loaded_cycloheptane
    # User reported R1, R2, R3, R5, R8 (1-indexed Chemkin) = 0, 1, 2, 4, 7
    rxn_F_pairs = [(0, 2.0), (1, 2.0), (2, 2.0), (4, 2.0), (7, 1.25)]
    cleanup = _set_specific_rxns_optimizable(mech, rxn_F_pairs)
    builder = cleanup[-1]
    try:
        optimizable = builder.build(mech)
        coef_opt = list(optimizable.coefficients)
        assert len(coef_opt) > 0

        shocks_setup = [
            ExperimentalShock.from_dict({
                "T_reactor": 1500.0, "P_reactor": 20000.0,
                "thermo_mix": {"Kr": 0.96, "cC7H14": 0.04},
            }),
            ExperimentalShock.from_dict({
                "T_reactor": 1700.0, "P_reactor": 20000.0,
                "thermo_mix": {"Kr": 0.96, "cC7H14": 0.04},
            }),
        ]

        rxn_coef_opt = build_rxn_coef_opt(mech, coef_opt, shocks_setup)
        rxn_rate_opt = build_rxn_rate_opt(mech, rxn_coef_opt)

        yield mech, coef_opt, rxn_coef_opt, rxn_rate_opt, optimizable
    finally:
        _clear_specific_opts(mech, *cleanup)


def _set_all_rxns_optimizable(mech):
    """Mark every non-Plog rxn + every rate coef optimizable. Mirrors the
    GUI's 'select all' click. Plog rxns are skipped because the fit path
    raises NotImplementedError on them. Falloff_parameters are excluded
    (int-keyed)."""
    from frhodo.optimize.parameters import OptimizableSetBuilder

    skip = {
        i for i, r in enumerate(mech.gas.reactions())
        if isinstance(r.rate, ct.PlogRate)
    }

    rate_orig: list = []
    for rxnIdx in range(mech.gas.n_reactions):
        if rxnIdx in skip:
            continue
        rate_orig.append((
            rxnIdx,
            mech.rate_bnds[rxnIdx]["value"], mech.rate_bnds[rxnIdx]["type"],
        ))
        mech.rate_bnds[rxnIdx]["value"] = 2.0
        mech.rate_bnds[rxnIdx]["type"] = "F"

    builder = OptimizableSetBuilder()
    coef_orig: list = []
    for rxnIdx in range(mech.gas.n_reactions):
        if rxnIdx in skip:
            continue
        builder.set_reaction_optimizable(rxnIdx, True)
        per_rxn: dict = {}
        for bk in mech.coeffs_bnds[rxnIdx]:
            if bk == "falloff_parameters":
                continue
            per_rxn[bk] = {}
            for c in mech.coeffs_bnds[rxnIdx][bk]:
                if not isinstance(c, str):
                    continue
                d = mech.coeffs_bnds[rxnIdx][bk][c]
                per_rxn[bk][c] = (d["value"], d["type"])
                d["value"] = 2.0
                d["type"] = "F"
                builder.set_coefficient_optimizable(rxnIdx, bk, c, True)
        coef_orig.append((rxnIdx, per_rxn))

    return rate_orig, coef_orig, builder


def _clear_all_opts(mech, rate_orig, coef_orig, _builder):
    for rxnIdx, val, typ in rate_orig:
        mech.rate_bnds[rxnIdx]["value"] = val
        mech.rate_bnds[rxnIdx]["type"] = typ
    for rxnIdx, per_rxn in coef_orig:
        for bk, per_bk in per_rxn.items():
            for c, (cval, ctyp) in per_bk.items():
                d = mech.coeffs_bnds[rxnIdx][bk][c]
                d["value"], d["type"] = cval, ctyp


@pytest.fixture
def all_rxns_optimizer_setup(loaded_cycloheptane):
    """Every rxn + every rate coef optimizable, plus two thermodynamic
    setpoints. Mirrors the GUI 'select all' scenario the user described."""
    from frhodo.experiment import ExperimentalShock

    mech = loaded_cycloheptane
    cleanup = _set_all_rxns_optimizable(mech)
    builder = cleanup[-1]
    try:
        optimizable = builder.build(mech)
        coef_opt = list(optimizable.coefficients)
        assert len(coef_opt) > 0

        shocks_setup = [
            ExperimentalShock.from_dict({
                "T_reactor": 1500.0, "P_reactor": 20000.0,
                "thermo_mix": {"Kr": 0.96, "cC7H14": 0.04},
            }),
            ExperimentalShock.from_dict({
                "T_reactor": 1700.0, "P_reactor": 20000.0,
                "thermo_mix": {"Kr": 0.96, "cC7H14": 0.04},
            }),
        ]

        rxn_coef_opt = build_rxn_coef_opt(mech, coef_opt, shocks_setup)
        rxn_rate_opt = build_rxn_rate_opt(mech, rxn_coef_opt)

        yield mech, coef_opt, rxn_coef_opt, rxn_rate_opt, optimizable
    finally:
        _clear_all_opts(mech, *cleanup)


@pytest.fixture
def troe_optimizer_setup(loaded_cycloheptane):
    """Same shape as ``optimizer_setup`` but marks the Troe rxn optimizable."""
    from frhodo.experiment import ExperimentalShock

    mech = loaded_cycloheptane
    cleanup = _set_one_troe_optimizable(mech)
    builder = cleanup[-1]
    try:
        optimizable = builder.build(mech)
        coef_opt = list(optimizable.coefficients)
        assert len(coef_opt) > 0

        shocks_setup = [
            ExperimentalShock.from_dict({
                "T_reactor": 1500.0, "P_reactor": 20000.0,
                "thermo_mix": {"Kr": 0.96, "cC7H14": 0.04},
            }),
            ExperimentalShock.from_dict({
                "T_reactor": 1700.0, "P_reactor": 20000.0,
                "thermo_mix": {"Kr": 0.96, "cC7H14": 0.04},
            }),
        ]

        rxn_coef_opt = build_rxn_coef_opt(mech, coef_opt, shocks_setup)
        rxn_rate_opt = build_rxn_rate_opt(mech, rxn_coef_opt)

        yield mech, coef_opt, rxn_coef_opt, rxn_rate_opt, optimizable
    finally:
        _clear_troe_opts(mech, *cleanup)


@pytest.fixture
def optimizer_setup(loaded_cycloheptane):
    """Loads cycloheptane, marks one Arrhenius reaction optimizable, and
    returns ``(mech, coef_opt, rxn_coef_opt, rxn_rate_opt, optimizable)``.
    Restores opt flags on teardown."""
    mech = loaded_cycloheptane
    cleanup = _set_one_arrhenius_optimizable(mech)
    builder = cleanup[-1]
    try:
        optimizable = builder.build(mech)
        coef_opt = list(optimizable.coefficients)
        assert len(coef_opt) > 0

        from frhodo.experiment import ExperimentalShock
        shocks_setup = [
            ExperimentalShock.from_dict({
                "T_reactor": 1500.0, "P_reactor": 20000.0,
                "thermo_mix": {"Kr": 0.96, "cC7H14": 0.04},
            }),
            ExperimentalShock.from_dict({
                "T_reactor": 1700.0, "P_reactor": 20000.0,
                "thermo_mix": {"Kr": 0.96, "cC7H14": 0.04},
            }),
        ]

        rxn_coef_opt = build_rxn_coef_opt(mech, coef_opt, shocks_setup)
        rxn_rate_opt = build_rxn_rate_opt(mech, rxn_coef_opt)

        yield mech, coef_opt, rxn_coef_opt, rxn_rate_opt, optimizable
    finally:
        _clear_opts(mech, *cleanup)


def _synthetic_shock(T_reactor=1500.0):
    """Minimal shock that survives ``calculate_residuals``."""
    from frhodo.experiment import ExperimentalShock

    t = np.linspace(1e-7, 5e-5, 50)
    obs = np.zeros_like(t)
    return ExperimentalShock.from_dict({
        "T_reactor": T_reactor,
        "P_reactor": 20000.0,
        "thermo_mix": {"Kr": 0.96, "cC7H14": 0.04},
        "u2": 181.85,
        "rho1": 0.0230433,
        "observable": {"main": "Density Gradient", "sub": 0},
        "exp_data": np.column_stack([t, obs]),
        "weights": np.ones_like(t),
        "normalized_weights": np.ones_like(t),
        "weights_trim": np.ones_like(t),
        "exp_data_trim": np.column_stack([t, obs]),
        "opt_time_offset": 0.0,
    })


def _make_fit_fun(
    mech, coef_opt, rxn_coef_opt, rxn_rate_opt,
    *, shocks2run=None, multiprocessing=False, pool=None,
    display_shock_provider=None, progress_callback=None,
    time_unc=0.0, random_t_uncertainty=True,
):
    from frhodo.optimize.residual import OptimizeRunInputs

    if shocks2run is None:
        shocks2run = [_synthetic_shock()]
    inputs = OptimizeRunInputs(
        mech=mech,
        shocks2run=shocks2run,
        coef_opt=coef_opt,
        rxn_coef_opt=rxn_coef_opt,
        rxn_rate_opt=rxn_rate_opt,
        initial_scalers=np.zeros(rxn_rate_opt["x0"].size),
        reactor_state=RuntimeReactorState(
            name="Incident Shock Reactor",
            t_end=5e-5, t_unit_conv=1e-6,
            sim_interp_factor=1, ode_solver="BDF",
            ode_rtol=1e-4, ode_atol=1e-7,
        ),
        time_unc=time_unc,
        cost_settings=CostSettings(
            obj_fcn_type="Residual",
            scale="Linear",
            bisymlog_scaling_factor=1.0,
            loss_alpha=2.0,
            loss_c=1.0,
            bayes_dist_type="Automatic",
            bayes_unc_sigma=2.0,
        ),
        opt_settings_optimize={},
        dist=scipy.stats.norm,
        multiprocessing=multiprocessing,
        max_processors=1,
        random_t_uncertainty=random_t_uncertainty,
    )

    return CostFunction(
        inputs,
        pool=pool,
        display_shock_provider=display_shock_provider,
        progress_callback=progress_callback,
    )


class TestOptimizerSetupPipeline:
    def test_coef_opt_has_one_entry_per_optimized_coefficient(self, optimizer_setup):
        _, coef_opt, _, _, optimizable = optimizer_setup
        assert len(coef_opt) == len(optimizable.coefficients)

    def test_rxn_coef_opt_groups_coefs_by_reaction(self, optimizer_setup):
        _, coef_opt, rxn_coef_opt, _, _ = optimizer_setup
        # All Arrhenius coeffs (A, n, Ea) for one rxn → one rxn_coef entry
        assert len(rxn_coef_opt) == 1
        rxn = rxn_coef_opt[0]
        assert isinstance(rxn["coefName"], list)
        assert len(rxn["coefName"]) == len(coef_opt)
        for key in ("invT", "T", "P", "coef_x0", "coef_bnds"):
            assert key in rxn

    def test_rxn_rate_opt_has_x0_and_bnds(self, optimizer_setup):
        _, _, _, rxn_rate_opt, _ = optimizer_setup
        assert "x0" in rxn_rate_opt
        assert "bnds" in rxn_rate_opt
        assert "lower" in rxn_rate_opt["bnds"]
        assert "upper" in rxn_rate_opt["bnds"]
        assert rxn_rate_opt["x0"].size > 0


class TestFitFunConstruction:
    def test_constructs_with_explicit_kwargs(self, optimizer_setup):
        mech, coef_opt, rxn_coef_opt, rxn_rate_opt, _ = optimizer_setup
        fit_fun = _make_fit_fun(mech, coef_opt, rxn_coef_opt, rxn_rate_opt)
        assert fit_fun.mech is mech
        assert fit_fun.coef_opt is coef_opt
        assert fit_fun.t_unc == (0.0, 0.0)
        assert fit_fun.dist is scipy.stats.norm

    def test_no_parent_attribute(self, optimizer_setup):
        mech, coef_opt, rxn_coef_opt, rxn_rate_opt, _ = optimizer_setup
        fit_fun = _make_fit_fun(mech, coef_opt, rxn_coef_opt, rxn_rate_opt)
        assert not hasattr(fit_fun, "parent")

    def test_no_signals_attribute(self, optimizer_setup):
        mech, coef_opt, rxn_coef_opt, rxn_rate_opt, _ = optimizer_setup
        fit_fun = _make_fit_fun(mech, coef_opt, rxn_coef_opt, rxn_rate_opt)
        assert not hasattr(fit_fun, "signals")


class TestFitFunCallEndToEnd:
    """The hot path: one cost-function evaluation with a synthetic shock.
    Catches regressions in the consumers of ``coef_opt`` (which is what
    phase-5b restructuring will touch)."""

    def test_returns_finite_objective(self, optimizer_setup):
        mech, coef_opt, rxn_coef_opt, rxn_rate_opt, _ = optimizer_setup
        fit_fun = _make_fit_fun(mech, coef_opt, rxn_coef_opt, rxn_rate_opt)
        s = np.zeros(rxn_rate_opt["x0"].size)
        result = fit_fun(s, optimizing=True)
        assert np.isfinite(result), f"obj_fcn was not finite: {result}"

    def test_zero_perturbation_runs_without_error(self, optimizer_setup):
        """s=0 means 'use the unperturbed mechanism'. Should always succeed."""
        mech, coef_opt, rxn_coef_opt, rxn_rate_opt, _ = optimizer_setup
        fit_fun = _make_fit_fun(mech, coef_opt, rxn_coef_opt, rxn_rate_opt)
        s = np.zeros(rxn_rate_opt["x0"].size)
        # Just shouldn't throw
        fit_fun(s, optimizing=True)

    def test_parametric_t_uncertainty_runs_and_fits_model(self, optimizer_setup):
        """random_t_uncertainty=False with t_unc>0 fits the global elastic-net
        shift model across shocks and still returns a finite objective."""
        mech, coef_opt, rxn_coef_opt, rxn_rate_opt, _ = optimizer_setup
        shocks = [_synthetic_shock(T) for T in (1700.0, 1900.0, 2100.0, 2300.0)]
        fit_fun = _make_fit_fun(
            mech, coef_opt, rxn_coef_opt, rxn_rate_opt,
            shocks2run=shocks, time_unc=1.0e-6, random_t_uncertainty=False,
        )
        s = np.zeros(rxn_rate_opt["x0"].size)
        result = fit_fun(s, optimizing=True)
        assert np.isfinite(result), f"parametric obj_fcn not finite: {result}"
        assert fit_fun._shift_model_info is not None
        assert fit_fun._shift_model_info.get("model") is not None

    @pytest.mark.parametrize("scaler", [-10.0, -5.0, -3.0, -1.0, -0.5, 0.5, 1.0, 3.0, 5.0, 10.0])
    def test_all_rxns_perturbation_sweep_keeps_pre_exponential_positive(
        self, all_rxns_optimizer_setup, scaler
    ):
        """User-reported scenario: 'everything optimizable, rate constrained'
        produced ``A=-12.7`` reaching Cantera's ArrheniusBase::check. With
        all 66 cycloheptane rxns + every rate coef optimizable, this sweep
        exercises whatever path leaks an unconverted ln(A)."""
        mech, coef_opt, rxn_coef_opt, rxn_rate_opt, _ = all_rxns_optimizer_setup
        fit_fun = _make_fit_fun(mech, coef_opt, rxn_coef_opt, rxn_rate_opt)
        s = np.full(rxn_rate_opt["x0"].size, scaler)

        fit_fun(s, optimizing=True)

        for c in coef_opt:
            if c.coef_name == "pre_exponential_factor":
                A = mech.coeffs[c.rxn_idx][c.coeffs_key][c.coef_name]
                assert A > 0, (
                    f"non-positive A={A} written for rxn {c.rxn_idx} "
                    f"({c.coeffs_key}) at scaler={scaler}"
                )

    def test_fit_generic_falloff_pressure_types_routed(self):
        """All pressure-dependent rate types Frhodo recognizes as
        'Falloff Reaction' must enter the Troe-fit branch in fit_generic.
        This pins the type list so a future Cantera adds-a-rate-class
        regresses loud rather than silently."""
        from frhodo.simulation.mechanism.fit_coeffs import fit_generic
        import inspect

        src = inspect.getsource(fit_generic)
        for cls in (ct.PlogRate, ct.FalloffRate, ct.LindemannRate,
                    ct.TsangRate, ct.TroeRate, ct.SriRate):
            assert f"ct.{cls.__name__}" in src, (
                f"{cls.__name__} should be in fit_generic's "
                f"pressure-dependent branch"
            )

    @pytest.mark.parametrize("seed", range(8))
    def test_arrhenius_only_user_scenario_keeps_pre_exponential_positive(
        self, loaded_cycloheptane, seed
    ):
        """The Arrhenius-only subset of the user's scenario (R3, R5 =
        rxns 2, 4 in 0-indexed). All Arrhenius coeffs optimizable, F=2.
        Sweep with non-uniform random scalers across the rate-bound range.
        No A should ever land negative."""
        from frhodo.experiment import ExperimentalShock
        from frhodo.optimize.parameters import OptimizableSetBuilder

        mech = loaded_cycloheptane
        rxn_F_pairs = [(2, 2.0), (4, 2.0)]
        cleanup = _set_specific_rxns_optimizable(mech, rxn_F_pairs)
        builder = cleanup[-1]
        try:
            optimizable = builder.build(mech)
            coef_opt = list(optimizable.coefficients)
            shocks_setup = [
                ExperimentalShock.from_dict({
                    "T_reactor": 1500.0, "P_reactor": 20000.0,
                    "thermo_mix": {"Kr": 0.96, "cC7H14": 0.04},
                }),
                ExperimentalShock.from_dict({
                    "T_reactor": 1700.0, "P_reactor": 20000.0,
                    "thermo_mix": {"Kr": 0.96, "cC7H14": 0.04},
                }),
            ]
            rxn_coef_opt = build_rxn_coef_opt(mech, coef_opt, shocks_setup)
            rxn_rate_opt = build_rxn_rate_opt(mech, rxn_coef_opt)
            fit_fun = _make_fit_fun(mech, coef_opt, rxn_coef_opt, rxn_rate_opt)

            rng = np.random.default_rng(seed)
            lb = rxn_rate_opt["bnds"]["lower"]
            ub = rxn_rate_opt["bnds"]["upper"]
            s = rng.uniform(lb, ub)

            fit_fun(s, optimizing=True)

            for c in coef_opt:
                if c.coef_name == "pre_exponential_factor":
                    A = mech.coeffs[c.rxn_idx][c.coeffs_key][c.coef_name]
                    assert A > 0, (
                        f"non-positive A={A} written for rxn {c.rxn_idx} "
                        f"({c.coeffs_key}) at seed={seed}"
                    )
        finally:
            _clear_specific_opts(mech, *cleanup)

    @pytest.mark.parametrize("seed", range(8))
    def test_all_rxns_random_sweep_keeps_pre_exponential_positive(
        self, all_rxns_optimizer_setup, seed
    ):
        """Same scenario as the uniform sweep, but with a non-uniform
        ``s`` vector mimicking the chaotic exploration of rbfopt/DE.
        Different rxns get different perturbation magnitudes, which is
        more likely to hit numerical edge cases in fit_arrhenius."""
        mech, coef_opt, rxn_coef_opt, rxn_rate_opt, _ = all_rxns_optimizer_setup
        fit_fun = _make_fit_fun(mech, coef_opt, rxn_coef_opt, rxn_rate_opt)

        rng = np.random.default_rng(seed)
        s = rng.uniform(-5.0, 5.0, size=rxn_rate_opt["x0"].size)

        fit_fun(s, optimizing=True)

        for c in coef_opt:
            if c.coef_name == "pre_exponential_factor":
                A = mech.coeffs[c.rxn_idx][c.coeffs_key][c.coef_name]
                assert A > 0, (
                    f"non-positive A={A} written for rxn {c.rxn_idx} "
                    f"({c.coeffs_key}) at seed={seed}"
                )

    @pytest.mark.parametrize("scaler", [-3.0, -1.0, -0.5, 0.5, 1.0, 3.0])
    def test_troe_perturbation_sweep_keeps_pre_exponential_positive(
        self, troe_optimizer_setup, scaler
    ):
        """Troe-rxn equivalent of the Arrhenius sweep. The Troe.fit() path
        is the more likely culprit for a missing exp; this exercises it
        across a range of rate-space perturbations."""
        mech, coef_opt, rxn_coef_opt, rxn_rate_opt, _ = troe_optimizer_setup
        fit_fun = _make_fit_fun(mech, coef_opt, rxn_coef_opt, rxn_rate_opt)
        s = np.full(rxn_rate_opt["x0"].size, scaler)

        fit_fun(s, optimizing=True)

        for c in coef_opt:
            if c.coef_name == "pre_exponential_factor":
                A = mech.coeffs[c.rxn_idx][c.coeffs_key][c.coef_name]
                assert A > 0, (
                    f"non-positive A={A} written for rxn {c.rxn_idx} "
                    f"({c.coeffs_key}) at scaler={scaler}"
                )

    @pytest.mark.parametrize("scaler", [-3.0, -1.0, -0.5, 0.5, 1.0, 3.0])
    def test_perturbation_sweep_keeps_pre_exponential_positive(
        self, optimizer_setup, scaler
    ):
        """fit_arrhenius and Troe.fit must always exp ln(A) back to linear A
        before update_mech_coef_opt writes it. A perturbation sweep across
        the scaler axis exercises the fit kernels at points away from the
        x0 minimum; any path that forgets the exp will trip the assertion
        in update_mech_coef_opt naming the offending rxn.
        """
        mech, coef_opt, rxn_coef_opt, rxn_rate_opt, _ = optimizer_setup
        fit_fun = _make_fit_fun(mech, coef_opt, rxn_coef_opt, rxn_rate_opt)
        s = np.full(rxn_rate_opt["x0"].size, scaler)

        # Should not raise: the assertion in update_mech_coef_opt fires
        # if any A drops to <=0.
        fit_fun(s, optimizing=True)

        for c in coef_opt:
            if c.coef_name == "pre_exponential_factor":
                A = mech.coeffs[c.rxn_idx][c.coeffs_key][c.coef_name]
                assert A > 0, (
                    f"non-positive A={A} written for rxn {c.rxn_idx} "
                    f"({c.coeffs_key}) at scaler={scaler}"
                )

    @pytest.mark.parametrize("scaler", [-3.0, -1.0, -0.5, 0.5, 1.0, 3.0])
    def test_orchestrator_update_gas_keeps_pre_exponential_positive(
        self, loaded_cycloheptane, scaler, tmp_path
    ):
        """End-to-end via the orchestrator's ``_update_gas`` upgrade path.

        User-reported scenario: R1/R2/R8 are PlogRate, R3/R5 are Arrhenius;
        bounds 2/2/2/2/1.25.  The orchestrator upgrades all PlogRates to Troe
        and re-initializes coef_opt; ``OptimizableSetBuilder.build`` must emit
        all 10 Troe coefficients (low + high Arrhenius + 4 falloff_parameters)
        for the upgraded rxns so coef_opt aligns slot-for-slot with the
        10-element fit return.  Without that, the surplus values cascade into
        later rxns' slots and a T-parameter ends up in an A slot — triggering
        ``A=-12.7``-style failures.
        """
        from frhodo.optimize.parameters import build_rxn_coef_opt, build_rxn_rate_opt

        mech = loaded_cycloheptane
        rxn_F_pairs = [(0, 2.0), (1, 2.0), (2, 2.0), (4, 2.0), (7, 1.25)]
        cleanup = _set_specific_rxns_optimizable(mech, rxn_F_pairs)
        builder = cleanup[-1]
        try:
            shocks2run = [
                _synthetic_shock(T_reactor=1500.0),
                _synthetic_shock(T_reactor=1700.0),
            ]

            def _prep(mech):
                optimizable = builder.build(mech)
                coef_opt = [c for c in optimizable.coefficients if c.coef_name != 4]
                rxn_coef_opt = build_rxn_coef_opt(mech, coef_opt, shocks2run)
                rxn_rate_opt = build_rxn_rate_opt(mech, rxn_coef_opt)

                return coef_opt, rxn_coef_opt, rxn_rate_opt

            coef_opt, rxn_coef_opt, rxn_rate_opt = _prep(mech)

            rxns_changed, _ = mech.recast_to_troe(rxn_coef_opt, rxn_rate_opt, builder)
            if rxns_changed:
                coef_opt, rxn_coef_opt, rxn_rate_opt = _prep(mech)

            entries_per_rxn: dict = {}
            for c in coef_opt:
                entries_per_rxn.setdefault(c.rxn_idx, []).append(c)
            for upgraded_idx in (0, 1, 7):
                rxn = mech.gas.reaction(upgraded_idx)
                assert isinstance(rxn.rate, ct.TroeRate), (
                    f"rxn {upgraded_idx} should be Troe after upgrade, "
                    f"got {type(rxn.rate).__name__}"
                )
                entries = entries_per_rxn[upgraded_idx]
                assert len(entries) == 10, (
                    f"upgraded rxn {upgraded_idx} should have 10 coef_opt "
                    f"entries (6 rate + 4 falloff_parameters), got {len(entries)}"
                )
            for arrh_idx in (2, 4):
                rxn = mech.gas.reaction(arrh_idx)
                assert isinstance(rxn.rate, ct.ArrheniusRate)
                entries = entries_per_rxn[arrh_idx]
                assert len(entries) == 3, (
                    f"Arrhenius rxn {arrh_idx} should have 3 coef_opt entries, "
                    f"got {len(entries)}"
                )

            fit_fun = _make_fit_fun(
                mech, coef_opt, rxn_coef_opt, rxn_rate_opt, shocks2run=shocks2run,
            )
            s = np.full(rxn_rate_opt["x0"].size, scaler)
            fit_fun(s, optimizing=True)

            for c in coef_opt:
                if c.coef_name == "pre_exponential_factor":
                    A = mech.coeffs[c.rxn_idx][c.coeffs_key][c.coef_name]
                    assert A > 0, (
                        f"non-positive A={A} written for R{c.rxn_idx + 1} "
                        f"({c.coeffs_key}) at scaler={scaler}"
                    )
        finally:
            _clear_specific_opts(mech, *cleanup)

    @pytest.mark.slow
    def test_fit_all_coeffs_pool_matches_sequential(self, loaded_cycloheptane):
        """Dispatching per-rxn fits across a worker pool must produce the
        same coefficient vector as the sequential master-side path. A
        mismatch means parallel introduces non-determinism (RNG seeding,
        JIT order, or worker-mech state leakage). Uses a mixed
        Arrhenius+Troe set so the slow falloff path is exercised
        alongside the cheap Arrhenius path.
        """
        mech = loaded_cycloheptane
        rxn_F_pairs = [(2, 2.0), (4, 2.0), (36, 2.0)]
        cleanup = _set_specific_rxns_optimizable(mech, rxn_F_pairs)
        builder = cleanup[-1]
        try:
            optimizable = builder.build(mech)
            coef_opt = list(optimizable.coefficients)
            shocks_setup = [
                ExperimentalShock.from_dict({
                    "T_reactor": 1500.0, "P_reactor": 20000.0,
                    "thermo_mix": {"Kr": 0.96, "cC7H14": 0.04},
                }),
                ExperimentalShock.from_dict({
                    "T_reactor": 1700.0, "P_reactor": 20000.0,
                    "thermo_mix": {"Kr": 0.96, "cC7H14": 0.04},
                }),
            ]
            rxn_coef_opt = build_rxn_coef_opt(mech, coef_opt, shocks_setup)
            rxn_rate_opt = build_rxn_rate_opt(mech, rxn_coef_opt)
            rates = np.exp(rxn_rate_opt["x0"])

            fit_fun_seq = _make_fit_fun(mech, coef_opt, rxn_coef_opt, rxn_rate_opt)
            coeffs_seq = fit_fun_seq.fit_all_coeffs(rates)

            payload = MechBuildPayload(
                reset_mech=mech.reset_mech,
                thermo_coeffs=mech.thermo_coeffs,
                coeffs=mech.coeffs,
                coeffs_bnds=mech.coeffs_bnds,
                rate_bnds=mech.rate_bnds,
            )
            pool = mp.Pool(
                processes=2,
                initializer=initialize_parallel_worker,
                initargs=(payload,),
            )
            try:
                fit_fun_par = _make_fit_fun(
                    mech, coef_opt, rxn_coef_opt, rxn_rate_opt,
                    multiprocessing=True, pool=pool,
                )
                coeffs_par = fit_fun_par.fit_all_coeffs(rates)
            finally:
                pool.close()
                pool.join()

            assert coeffs_seq is not None, "sequential fit returned None"
            assert coeffs_par is not None, "parallel fit returned None"
            assert coeffs_seq.size > 0
            np.testing.assert_allclose(coeffs_par, coeffs_seq, rtol=1e-10, atol=0)
        finally:
            _clear_specific_opts(mech, *cleanup)

    def test_display_shock_provider_polled_each_evaluation(self, optimizer_setup):
        """Each cost evaluation must call the provider — GUI mid-run shock
        switches only propagate if the provider is polled, not snapshot
        at CostFunction construction."""
        mech, coef_opt, rxn_coef_opt, rxn_rate_opt, _ = optimizer_setup
        call_count = {"n": 0}

        def _provider():
            call_count["n"] += 1
            return None

        fit_fun = _make_fit_fun(
            mech, coef_opt, rxn_coef_opt, rxn_rate_opt,
            display_shock_provider=_provider,
        )
        s = np.zeros(rxn_rate_opt["x0"].size)
        fit_fun(s, optimizing=True)
        fit_fun(s, optimizing=True)

        assert call_count["n"] == 2

    def test_display_shock_provider_switch_changes_progress_update(self, optimizer_setup):
        """Switching the shock the provider returns between calls makes
        the next progress update track the new shock's simulation trace."""
        mech, coef_opt, rxn_coef_opt, rxn_rate_opt, _ = optimizer_setup
        shock_a = _synthetic_shock(T_reactor=1500.0)
        shock_b = _synthetic_shock(T_reactor=1700.0)
        current = {"shock": shock_a}
        updates: list = []
        fit_fun = _make_fit_fun(
            mech, coef_opt, rxn_coef_opt, rxn_rate_opt,
            shocks2run=[shock_a, shock_b],
            display_shock_provider=lambda: current["shock"],
            progress_callback=updates.append,
        )
        s = np.zeros(rxn_rate_opt["x0"].size)

        fit_fun(s, optimizing=True)
        first_obs = updates[-1]["observable"]
        current["shock"] = shock_b
        fit_fun(s, optimizing=True)
        second_obs = updates[-1]["observable"]

        assert first_obs is not None and second_obs is not None
        assert not np.array_equal(first_obs, second_obs), (
            "display_shock provider switch did not propagate; same shock "
            "trace served for both iterations"
        )

    def test_log_callback_receives_no_messages_on_success(self, optimizer_setup):
        mech, coef_opt, rxn_coef_opt, rxn_rate_opt, _ = optimizer_setup
        messages: list[str] = []
        fit_fun = _make_fit_fun(
            mech, coef_opt, rxn_coef_opt, rxn_rate_opt,
            shocks2run=[_synthetic_shock()],
        )
        fit_fun._log = messages.append
        s = np.zeros(rxn_rate_opt["x0"].size)
        fit_fun(s, optimizing=True)
        assert messages == []
