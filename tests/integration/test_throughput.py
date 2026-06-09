"""Throughput regression bound for the ML inner loop.

Runs a tiny inner loop — mutate one Arrhenius coefficient, run N
shocks, sum residuals — and asserts the per-iteration wall time
stays below a generous bound. The bound is permissive (200 ms /
iter at 4 shocks on cycloheptane) so it catches large regressions
without tracking micro-perf.
"""
import time

import cantera as ct
import numpy as np
import pytest

from frhodo.api import (
    PostShockState,
    PreShockState,
    ShockTubeConfig,
    run_shock_tube,
    solve_shock_jump,
)


T1_K = 294.15
P1_PA = 5.01 * 133.322368421
U1_MPS = 120e-3 / 116.557292e-6
MIX = {"Kr": 0.96, "cC7H14": 0.04}
T_END = 5.0e-5

N_ITERS = 5
N_SHOCKS_PER_ITER = 4

MAX_SECONDS_PER_ITER = 0.200


@pytest.mark.slow
def test_ml_inner_loop_throughput_within_bound(loaded_cycloheptane):
    mech = loaded_cycloheptane
    ic = PreShockState(T1=T1_K, P1=P1_PA, u1=U1_MPS, composition=dict(MIX))
    ss = solve_shock_jump(ic, mech)
    assert ss.success

    cfg = ShockTubeConfig(
        initial=PostShockState(
            T_reac=ss.T2, P_reac=ss.P2,
            u_incident=ss.u2, rho1=ss.rho1,
            composition=dict(MIX),
        ),
        t_end=T_END,
    )

    arrh_idxs = [
        i for i, r in enumerate(mech.gas.reactions())
        if type(r.rate) is ct.ArrheniusRate
    ]
    target = arrh_idxs[0]
    base_A = mech.coeffs[target][0]["pre_exponential_factor"]
    rng = np.random.default_rng(42)

    start = time.perf_counter()
    for _ in range(N_ITERS):
        mech.coeffs[target][0]["pre_exponential_factor"] = base_A * (
            1.0 + 0.05 * rng.normal()
        )
        mech.modify_reactions(mech.coeffs, rxnIdxs=[target])
        for _ in range(N_SHOCKS_PER_ITER):
            r = run_shock_tube(mech, cfg)
            assert r.success
    elapsed = time.perf_counter() - start

    seconds_per_iter = elapsed / N_ITERS
    assert seconds_per_iter < MAX_SECONDS_PER_ITER, (
        f"ML inner-loop throughput regressed: "
        f"{seconds_per_iter*1000:.0f} ms/iter > {MAX_SECONDS_PER_ITER*1000:.0f} ms/iter"
    )
