"""Three-backend optimizer integration test.

Each parametrized run exercises ``optimize_residual`` with a different
backend (nlopt, pygmo, rbfopt) on a tiny synthetic shock so the test
stays under a few seconds.
"""
import importlib

import cantera as ct
import numpy as np
import pytest

from frhodo.api import (
    AlgorithmSettings,
    AlgorithmStage,
    CostSettings,
    ExperimentShock,
    ObservableSettings,
    OptimizableRate,
    OptimizableSpec,
    OptimizationRequest,
    OptimizationResult,
    PostShockState,
    RateUncertainty,
    _to_internal_shock,
    optimize_residual,
)
from frhodo.gui.views import build_view_context, compute_arrhenius_lines
from frhodo.optimize.parameters import build_rxn_coef_opt, build_rxn_rate_opt
from frhodo.simulation.mechanism.coef_helpers import rates
from frhodo.simulation.shock.state import RuntimeReactorState



pygmo_available = importlib.util.find_spec("pygmo") is not None


def _synthetic_shock():
    t = np.linspace(1e-7, 5e-5, 50)
    return ExperimentShock(
        t=t, observable=np.zeros_like(t),
        initial=PostShockState(
            T_reac=1500.0, P_reac=20000.0,
            u_incident=181.85, rho1=0.0230433,
            composition={"Kr": 0.96, "cC7H14": 0.04},
        ),
        t_end=5e-5,
    )


def _build_request(loaded_cycloheptane, algorithm_label: str, max_iters: int):
    arrh_idx = next(
        i for i, r in enumerate(loaded_cycloheptane.gas.reactions())
        if type(r.rate) is ct.ArrheniusRate
    )

    return OptimizationRequest(
        shocks=[_synthetic_shock()],
        optimizable=OptimizableSpec(rates=[
            OptimizableRate(rxn_idx=arrh_idx, rate=RateUncertainty(factor=2.0)),
        ]),
        reactor_state=RuntimeReactorState(
            name="Incident Shock Reactor", t_end=5e-5, t_unit_conv=1e-6,
            sim_interp_factor=1, ode_solver="BDF", ode_rtol=1e-4, ode_atol=1e-7,
        ),
        cost=CostSettings(
            scale="Linear",
            bisymlog_scaling_factor=1.0, loss_alpha=2.0, loss_c=1.0,
        ),
        algorithm=AlgorithmSettings(
            global_stage=AlgorithmStage(
                algorithm="RBFOpt", enabled=False, stop_value=1.0,
            ),
            local_stage=AlgorithmStage(
                algorithm=algorithm_label, enabled=True,
                max_eval=max_iters, stop_value=float(max_iters),
            ),
        ),
        observable=ObservableSettings(),
    )


@pytest.mark.slow
class TestNloptBackend:
    @pytest.mark.parametrize("algorithm_label", [
        "Nelder-Mead Simplex", "Subplex", "COBYLA",
    ])
    def test_returns_optimization_result(self, loaded_cycloheptane, algorithm_label):
        request = _build_request(loaded_cycloheptane, algorithm_label, max_iters=2)
        result = optimize_residual(loaded_cycloheptane, request)
        assert isinstance(result, OptimizationResult)
        assert np.isfinite(result.fval), (
            f"{algorithm_label}: fval was {result.fval}"
        )


@pytest.mark.slow
@pytest.mark.skipif(not pygmo_available, reason="pygmo not installed")
class TestPygmoBackend:
    @pytest.mark.parametrize("algorithm_label", [
        "DE (Differential Evolution)",
        "PSO (Particle Swarm Optimization)",
    ])
    def test_returns_optimization_result(self, loaded_cycloheptane, algorithm_label):
        request = _build_request(loaded_cycloheptane, algorithm_label, max_iters=2)
        result = optimize_residual(loaded_cycloheptane, request)
        assert isinstance(result, OptimizationResult)
        assert np.isfinite(result.fval)


class TestViewContext:
    """The static view snapshot built at run start from real structures."""

    def test_context_from_real_mechanism(self, loaded_cycloheptane):
        request = _build_request(loaded_cycloheptane, "Subplex", max_iters=2)
        optimizable_set = request.optimizable.build(loaded_cycloheptane)
        coef_opt = list(optimizable_set.coefficients)
        shocks = [_to_internal_shock(s, request, loaded_cycloheptane)
                  for s in request.shocks]
        rxn_coef_opt = build_rxn_coef_opt(loaded_cycloheptane, coef_opt, shocks)
        rxn_rate_opt = build_rxn_rate_opt(loaded_cycloheptane, rxn_coef_opt)

        context = build_view_context(
            loaded_cycloheptane, rxn_coef_opt, rxn_rate_opt, shocks,
        )
        assert len(context.rxn_indices) == 1
        assert len(context.param_labels) == len(context.lower_bounds)
        ln_k0 = np.asarray(context.ln_k_initial)
        assert ln_k0.shape == (1, len(context.P_grid), len(context.T_grid))
        assert np.all(np.isfinite(ln_k0)), "initial ln k must be finite"
        assert context.rxn_band_halfwidth[0] == pytest.approx(
            np.log(2.0), rel=1e-6,
        ), "factor-2 uncertainty must give a ln(2) band"
        assert context.P_grid[0] < context.P_reference < context.P_grid[-1]

        lines = compute_arrhenius_lines(loaded_cycloheptane, context, shocks)
        ln_k = np.asarray(lines["ln_k"])
        np.testing.assert_allclose(
            ln_k, ln_k0, rtol=1e-12,
            err_msg="at the start point the incumbent must equal initial",
        )


class TestRateOptAnchorAlignment:
    """x0 overrides and bounds slices must address each reaction's own
    anchor span. The falloff limit-anchor override fires only when
    per-coefficient bounds exist (the GUI's coefficient uncertainty
    boxes); with the second optimized reaction pressure-dependent it
    used to write into the first reaction's baselines."""

    def test_falloff_override_lands_in_its_own_span(self, loaded_cycloheptane):
        mech = loaded_cycloheptane
        arrh_idx = next(i for i, r in enumerate(mech.gas.reactions())
                        if type(r.rate) is ct.ArrheniusRate)
        troe_idx = next(i for i, r in enumerate(mech.gas.reactions())
                        if type(r.rate) is ct.TroeRate and i > arrh_idx)
        request = OptimizationRequest(
            shocks=[_synthetic_shock()],
            optimizable=OptimizableSpec(rates=[
                OptimizableRate(rxn_idx=arrh_idx,
                                rate=RateUncertainty(factor=2.0)),
                OptimizableRate(rxn_idx=troe_idx,
                                rate=RateUncertainty(factor=2.0)),
            ]),
            reactor_state=RuntimeReactorState(
                name="Incident Shock Reactor", t_end=5e-5, t_unit_conv=1e-6,
                sim_interp_factor=1, ode_solver="BDF", ode_rtol=1e-4,
                ode_atol=1e-7,
            ),
            cost=CostSettings(
                scale="Linear",
                bisymlog_scaling_factor=1.0, loss_alpha=2.0, loss_c=1.0,
            ),
            algorithm=AlgorithmSettings(
                global_stage=AlgorithmStage(algorithm="RBFOpt", enabled=False),
                local_stage=AlgorithmStage(algorithm="Subplex", enabled=True,
                                           max_eval=2, stop_value=2.0),
            ),
            observable=ObservableSettings(),
        )
        optimizable_set = request.optimizable.build(mech)

        # Mimic the GUI's per-coefficient uncertainty boxes on the Troe
        # target: set values make set_bnds report exist=True, firing the
        # limit-anchor override.
        touched = []
        for sub in mech.coeffs_bnds[troe_idx].values():
            for coef_name, d in sub.items():
                if isinstance(coef_name, str):
                    touched.append((d, d["value"], d["type"]))
                    d["value"] = 2.0
                    d["type"] = "F"
        try:
            coef_opt = list(optimizable_set.coefficients)
            shocks = [_to_internal_shock(s, request, mech)
                      for s in request.shocks]
            rxn_coef_opt = build_rxn_coef_opt(mech, coef_opt, shocks)
            baseline = rates(rxn_coef_opt, mech)
            rxn_rate_opt = build_rxn_rate_opt(mech, rxn_coef_opt)
        finally:
            for d, value, unc_type in touched:
                d["value"] = value
                d["type"] = unc_type

        lens = [len(rc["T"]) for rc in rxn_coef_opt]
        troe_pos = [rc["rxnIdx"] for rc in rxn_coef_opt].index(troe_idx)
        start = sum(lens[:troe_pos])
        end = start + lens[troe_pos]
        overridden = np.where(
            np.abs(rxn_rate_opt["x0"] - baseline) > 1e-9
        )[0]
        assert overridden.size > 0, (
            "coefficient bounds on the Troe target must fire its "
            "limit-anchor x0 overrides"
        )
        assert overridden.min() >= start and overridden.max() < end, (
            f"limit-anchor overrides wrote x0 positions "
            f"{overridden.tolist()} outside the Troe reaction's span "
            f"[{start}, {end})"
        )
        np.testing.assert_allclose(
            rxn_rate_opt["x0"][:start], baseline[:start], rtol=1e-12,
            err_msg="the first reaction's baselines must be untouched",
        )


@pytest.mark.slow
class TestRBFOptBackend:
    def test_returns_optimization_result(self, loaded_cycloheptane):
        # RBFOpt runs in the global stage; flip the stages so we exercise it.
        arrh_idx = next(
            i for i, r in enumerate(loaded_cycloheptane.gas.reactions())
            if type(r.rate) is ct.ArrheniusRate
        )
        request = OptimizationRequest(
            shocks=[_synthetic_shock()],
            optimizable=OptimizableSpec(rates=[
                OptimizableRate(rxn_idx=arrh_idx, rate=RateUncertainty(factor=2.0)),
            ]),
            reactor_state=RuntimeReactorState(
                name="Incident Shock Reactor", t_end=5e-5, t_unit_conv=1e-6,
                sim_interp_factor=1, ode_solver="BDF", ode_rtol=1e-4, ode_atol=1e-7,
            ),
            cost=CostSettings(
                scale="Linear",
                bisymlog_scaling_factor=1.0, loss_alpha=2.0, loss_c=1.0,
            ),
            algorithm=AlgorithmSettings(
                global_stage=AlgorithmStage(
                    algorithm="RBFOpt", enabled=True, max_eval=3, stop_value=3.0,
                ),
                local_stage=AlgorithmStage(algorithm="Subplex", enabled=False),
            ),
        )
        result = optimize_residual(loaded_cycloheptane, request)
        assert isinstance(result, OptimizationResult)
        assert np.isfinite(result.fval)
