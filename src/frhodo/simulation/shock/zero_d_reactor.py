"""0-D ideal-gas reactor runner — constant-volume or constant-pressure.

Drives Cantera's built-in ``IdealGasReactor`` /
``IdealGasConstPressureReactor`` via a ``ReactorNet``; collects the
trajectory into a ``ReactorOutput`` mirroring the incident-shock path.
"""
import sys

import cantera as ct
import numpy as np

from frhodo.common.errors import FailureReason
from frhodo.simulation.mechanism.mech_fcns import check_rxn_rates, list2ct_mixture
from frhodo.simulation.shock.reactor_output import ReactorOutput


def _zero_d_ideal_gas_reactor(gas, reactor, details, t_end, **kwargs):
    var = {
        "observable": {"main": "Concentration", "sub": 0},
        "t_lab_save": None,
        "rtol": 1e-4,
        "atol": 1e-7,
    }
    var.update(kwargs)

    reactor.energy_enabled = var["solve_energy"]
    reactor.chemistry_enabled = not var["frozen_comp"]

    sim = ct.ReactorNet([reactor])
    sim.atol = var["atol"]
    sim.rtol = var["rtol"]

    ind_var = "t_lab"
    if var["t_lab_save"] is None:
        t_all = [float(t_end)]
    else:
        t_all = np.sort(np.unique(np.concatenate(([t_end], var["t_lab_save"]))))

    details["success"] = True
    states = ct.SolutionArray(gas, extra=["t"])
    states.append(reactor.thermo.state, t=0.0)

    for t in t_all:
        if t <= sim.time:
            continue
        try:
            sim.advance(t)
        except Exception:
            details["success"] = False
            details["failure_reason"] = FailureReason.SOLVER_FAILURE
            explanation = "\nCheck for: Fast rates or bad thermo data"
            flagged = check_rxn_rates(gas)
            if len(flagged) > 0:
                explanation += "\nSuggested Reactions: " + ", ".join(
                    str(x) for x in flagged
                )
            details["message"] = "\nODE Error: {:s}\n{:s}\n".format(
                str(sys.exc_info()[1]), explanation
            )
            break
        states.append(reactor.thermo.state, t=sim.time)

    reactor_vars = [
        "t_lab", "T", "P",
        "h_tot", "h", "s_tot", "s",
        "rho", "Y", "X", "conc",
        "wdot", "wdotfor", "wdotrev",
        "HRR_tot", "HRR",
        "delta_h", "delta_s",
        "eq_con", "rate_con", "rate_con_rev",
        "net_ROP", "for_ROP", "rev_ROP",
    ]

    num = {
        "reac": np.sum(gas.reactant_stoich_coeffs, axis=0),
        "prod": np.sum(gas.product_stoich_coeffs, axis=0),
        "rxns": gas.n_reactions,
    }

    SIM = ReactorOutput(num, states, reactor_vars)
    SIM.finalize(details["success"], ind_var, var["observable"], units="CGS")
    return SIM, details


def run_zero_d(mech, mode, t_end, T_reac, P_reac, mix, **kwargs):
    """Run a 0-D ideal-gas reactor.

    Holds ``mech.exclusive()`` for the call's duration.

    Args:
        mode: ``"constant_volume"`` or ``"constant_pressure"``.

    Returns:
        ``(ReactorOutput | None, details_dict)``. The output is ``None``
        if ``mech.set_TPX`` rejects the initial state; ``details_dict``
        always carries at least ``success`` and ``message``.

    Raises:
        ValueError: If ``mode`` is not one of the supported labels.
    """
    if isinstance(mix, list):
        mix = list2ct_mixture(mix)

    with mech.exclusive():
        mech_out = mech.set_TPX(T_reac, P_reac, mix)
        if not mech_out["success"]:
            return None, mech_out

        if mode == "constant_volume":
            reactor = ct.IdealGasMoleReactor(mech.gas)
        elif mode == "constant_pressure":
            reactor = ct.IdealGasConstPressureMoleReactor(mech.gas)
        else:
            raise ValueError(f"unknown 0-D reactor mode: {mode!r}")

        return _zero_d_ideal_gas_reactor(
            mech.gas, reactor, {"success": False, "message": []}, t_end, **kwargs
        )
