"""Simple functions for common tasks in Frhodo.

The goal is to eventually distribute alongside the GUI components"""
from typing import Dict, Tuple, List, Union

import numpy as np

from frhodo.calculate.mech_fcns import Chemical_Mechanism

CoefIndex = Tuple[int, Union[str, int], str]
"""How to specify the index of a reaction parameter: reaction index, pressure index, name of the parameter"""

RxnConditions = Tuple[float, float, dict]
"""Specification for the temperature, pressure and concentration for a reaction"""


def set_coefficients(mech: Chemical_Mechanism, new_values: Dict[CoefIndex, float]):
    """Get the coefficients of a chemical mechanism object

    Args:
        mech: Mechanism to modify
        new_values: New values for different coefficients
    """
    for key, new_value in new_values.items():
        rxn_id, prs_id, coef_name = key
        rxn_model = mech.coeffs[rxn_id][prs_id]
        assert coef_name in rxn_model, f'Key {coef_name} not present for reaction #{rxn_id} at pressure #{prs_id}'
        rxn_model[coef_name] = new_value

    # Update the reaction rates for the current shock
    mech.modify_reactions(mech.coeffs)


def run_simulation(mech: Chemical_Mechanism, rxn_conditions: RxnConditions, sim_kwargs: dict) -> np.ndarray:
    """Run a simulation for a single reaction condition

    Args:
        mech: Mechanism describing the chemical kinetics
        rxn_conditions: Conditions of the reaction (temperature, pressure, composition)
        sim_kwargs: Keywords to the simulator
    Returns:
        Array with time as first column and observable as the second
    """
    mech.set_TPX(*rxn_conditions)
    # Run the simulation
    #  TODO (wardlt): Do not hardcode the runtime or reactor conditions
    sim, _ = mech.run('Incident Shock Reactor', 1.2e-05, *rxn_conditions, **sim_kwargs)
    assert sim.success, "Simulation failed"
    return np.stack([
        sim.independent_var,
        sim.observable
    ], axis=1)


def get_coefficients(mech: Chemical_Mechanism, indices: List[CoefIndex]) -> List[float]:
    """Get specific coefficients from the reaction model

    Args:
        mech: Mechanism to interrogate
        indices: List of coefficients to retrieve
    Returns:
        Desired coefficents
    """
    output = []
    for rxn_id, prs_id, coef_name in indices:
        rxn_model = mech.coeffs[rxn_id][prs_id]
        assert coef_name in rxn_model, f'Key {coef_name} not present for reaction #{rxn_id} at pressure #{prs_id}'
        output.append(rxn_model[coef_name])
    return output


def compute_kinetic_coefficients(mech: Chemical_Mechanism, rxn_conditions: RxnConditions) -> np.ndarray:
    """Get the kinetic coefficients at a certain conditions

    Args:
        mech: Mechanism object to use
        rxn_conditions: Reaction conditions (T, P, X)
    """

    mech.set_TPX(*rxn_conditions)
    return np.array([
        mech.gas.forward_rate_constants[i] for i in range(mech.gas.n_reactions)
    ])
