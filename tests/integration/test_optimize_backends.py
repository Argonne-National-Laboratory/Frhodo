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
    optimize_residual,
)
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
            obj_fcn_type="Residual", scale="Linear",
            bisymlog_scaling_factor=1.0, loss_alpha=2.0, loss_c=1.0,
            bayes_dist_type="Automatic", bayes_unc_sigma=2.0,
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
                obj_fcn_type="Residual", scale="Linear",
                bisymlog_scaling_factor=1.0, loss_alpha=2.0, loss_c=1.0,
                bayes_dist_type="Automatic", bayes_unc_sigma=2.0,
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
