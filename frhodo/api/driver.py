"""Low-level driver for controlling a Frhodo GUI session"""

from pathlib import Path
from typing import Optional, Dict, List, Union, Any, Tuple

import cantera as ct
import numpy as np
from PyQt5.QtWidgets import QApplication

from frhodo.calculate.reactors import Simulation_Result
from frhodo.main import Main, launch_gui

CoefIndex = Tuple[int, Union[str, int], str]
"""How to specify the index of a reaction parameter: reaction index, pressure index, name of the parameter"""


class FrhodoDriver:
    """Driver the Frhodo GUI application

    Implements simple interfaces for common tasks, such as loading different datasets
    or evaluating changes to different mechanisms

    **Limitations**

    The driver only supports a subset of features of Frhodo

    - We can only run one driver per process. You will even need to "spawn" new processes rather than fork on Linux
    - Only supports experiments that use experimental data from a single series
    - Does not support changing parameters to reactions which are pressure-dependent
    """

    def __init__(self, window: Main, app: QApplication):
        """Create the driver given connection to a running instance of Frhodo

        Args:
            window: Link to the main Frhodo window
            app: Link to the Qt backend
        """
        self.window = window
        self.app = app

    def load_files(self, experiment_path: Path,
                   mechanism_path: Path,
                   output_path: Path,
                   aliases: Optional[Dict[str, str]] = None):
        """Load the input files for Frhodo

        Args:
            experiment_path: Path to the experimental data
            mechanism_path: Path to the mechanism files
            output_path: Path to which simulation results should be stored
            aliases: List of aliases of chemical species as mapping of the name
                of a chemical in the experiment data to name of the same chemical in the mechanism.
        """

        # Set the files in the text boxes
        self.window.exp_main_box.setPlainText(str(experiment_path.resolve().absolute()))
        self.window.mech_main_box.setPlainText(str(mechanism_path.resolve().absolute()))
        self.window.sim_main_box.setPlainText(str(output_path.resolve().absolute()))

        # Trigger Frhodo to process these files
        self.app.processEvents()

        # Add in the aliases
        if aliases is not None:
            aliases = aliases.copy()  # Ensure that later changes to aliases don't propagate here
            self.window.series.species_alias[0] = aliases
            for shock in self.window.series.shock[0]:
                shock['species_alias'] = aliases

    @property
    def n_shocks(self):
        """Number of shock experiments loaded"""
        n_series = len(self.window.series.shock)
        assert n_series <= 1, "We only support one series"
        if n_series == 0:
            return 0
        return len(self.window.series.shock[0])

    @property
    def rxn_coeffs(self) -> List[Union[List[Dict[str, float]], Dict[str, Any]]]:
        """Reaction rate coefficients"""
        return self.window.mech.coeffs

    def _select_shock(self, n: int):
        """Change which shock experiment is being displayed and simulated

        Args:
            n: Which shock experiment to evaluate
        """

        # Check if it is in the list
        assert self.n_shocks > 0, 'No shocks are loaded'
        assert n in self.window.series.current['shock_num'], f'Shock number {n} is not in the current series'

        # Trigger an update of the displayed system
        self.window.shock_choice_box.setValue(n + 1)
        self.app.processEvents()

    def get_simulator_inputs(self) -> Tuple[List[dict], List[Tuple[float, float, Dict[str, float]]]]:
        """Get the list of variables that serve as input for the simulator for each shock that runs

        Returns:
            - Keyword arguments passed to the simulator
            - Temperature, pressure and concentration of different guasses
        """
        var_outputs = []
        rxn_outputs = []

        for shock in self.window.series.shock[0]:
            self._select_shock(shock['num'])
            var_outputs.append(self.window.get_simulation_kwargs(shock, np.array([0])))
            rxn_outputs.append((shock['T_reactor'], shock['P_reactor'], shock['thermo_mix']))

        return var_outputs, rxn_outputs

    def get_observables(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get the observable data and associated weights from each shock experiment

        Returns:
            - List of experimental data arrays where each is a 2D
             array with the first column is the time and second
             is the observable
            - 1D array of the weights for each point
        """

        if self.n_shocks == 0:
            return [], []

        # Loop over each shock
        output = []
        output_weights = []
        for shock in self.window.series.shock[0]:
            # Changing the index in the GUI forces data to be loaded
            self._select_shock(shock['num'])

            # Get the data from the display shock
            output.append(self.window.display_shock['exp_data'])
            output_weights.append(self.window.display_shock['normalized_weights'])
        return output, output_weights

    def run_simulations(self) -> List[np.ndarray]:
        """Run the simulation for each of the shocks

        Returns:
            List of simulated data arrays where each is a 2D
             array with the first column is the time and second
             is the simulated observable
        """

        # Loop over all shocks
        output = []
        for shock in self.window.series.shock[0]:
            # We force the simulation by changing the shock index in the GUI
            #  That could be an issue if the behavior of the GUI changes
            self._select_shock(shock['num'])
            assert self.window.SIM.success, 'Simulation failed'
            output.append(np.stack([
                self.window.SIM.independent_var,
                self.window.SIM.observable
            ], axis=1))
        return output

    def run_simulation_from_kwargs(self, sim_kwargs, rxn_conditions) -> np.ndarray:
        """Perform a simulation for an arbitrary reactor and reaction conditions.

        Args:
            sim_kwargs: Keywords defining the reactor
            rxn_conditions: Conditions at which the reaction occurs

        Returns:
            Simulated data arrays where each is a 2D
             array with the first column is the time and second
             is the simulated observable
        """

        # Update the mechanism for the specified reaction conditions
        self.window.mech.set_TPX(*rxn_conditions)

        # Run the simulation
        #  TODO (wardlt): Do not hardcode the runtime or reactor conditions
        sim, _ = self.window.mech.run('Incident Shock Reactor', 1.2e-05, *rxn_conditions, **sim_kwargs)

        assert sim.success, "Simulation failed"
        return np.stack([
            sim.independent_var,
            sim.observable
        ], axis=1)

    def get_reaction_rates(self) -> np.ndarray:
        """Get the reaction rates for each shock experiment

        Returns:
            Array where each row is a different reaction rate and each column is
             a different shock tube experiment
        """

        # Run them directly through the `series` interface rather than changing the display
        #  to avoid re-running the
        output = []
        for shock in self.window.series.shock[0]:
            output.append(self.window.series.rates(shock))

        # Call with the current shock to ensure `mech` hasn't been changed
        self.window.series.rates(self.window.display_shock)
        return np.stack(output, axis=1)

    def get_fittable_parameters(self, chosen_reactions: Optional[List[int]] = None) -> List[CoefIndex]:
        """Get a list of all parameters than can be fit during optimization

        Args:
            chosen_reactions: Which reactions we're allowed to optimize
        Returns:
            Index of each parameter that could be modified. Indices are defined by:
                - Reaction number
                - Index of the pressure level
                - Name of the coefficient
        """
        _not_parameter = ['efficiencies', 'Pressure']

        output = []
        # Loop over all reactions, getting both the coefficients (stored in Fhrodo)
        #   and reaction models (stored in a Cantera reaction object)
        for rxn_id, (coeffs, rxn_obj) in enumerate(zip(self.rxn_coeffs, self.window.mech.gas.reactions())):
            # Determine whether to skip this reaction
            if chosen_reactions is not None and rxn_id not in chosen_reactions:
                continue

            # We get the high and low pressure for the falloff reactions
            if type(rxn_obj) is ct.FalloffReaction:
                for p_id in ['low_rate', 'high_rate']:
                    for coeff in coeffs[p_id].keys():
                        if coeff not in _not_parameter:
                            output.append((rxn_id, p_id, coeff))
            else:
                # Loop over pressure levels
                for p_id, p_coeffs in enumerate(coeffs):
                    for coeff in p_coeffs.keys():
                        if coeff not in _not_parameter:
                            output.append((rxn_id, p_id, coeff))
        return output

    def get_coefficients(self, indices: List[CoefIndex]) -> List[float]:
        """Get the values of specific coefficients

        Args:
            indices: List of coefficients to extract
        Returns:
            List of their current values
        """

        output = []
        for rxn_id, prs_id, coef_name in indices:
            rxn_model = self.window.mech.coeffs[rxn_id][prs_id]
            assert coef_name in rxn_model, f'Key {coef_name} not present for reaction #{rxn_id} at pressure #{prs_id}'
            output.append(rxn_model[coef_name])
        return output

    def set_coefficients(self, new_values: Dict[CoefIndex, float]):
        """Update the parameters of a reaction parameter

        Args:
            new_values: A dictionary where the key is a tuple defining which
                coefficient is being altered: (<rxn index: int>, <pressure index: int>, <coeff. name: str>)
                The value is the new value for that coefficient
        """

        # Update the value in the Chemical_Mechanism dictionary
        for key, new_value in new_values.items():
            rxn_id, prs_id, coef_name = key
            rxn_model = self.window.mech.coeffs[rxn_id][prs_id]
            assert coef_name in rxn_model, f'Key {coef_name} not present for reaction #{rxn_id} at pressure #{prs_id}'
            rxn_model[coef_name] = new_value

        # Update the reaction rates for the current shock
        self.window.mech.modify_reactions(self.window.mech.coeffs)
        self.window.tree.update_rates()
        self.app.processEvents()

    @classmethod
    def create_driver(cls, headless: bool = True, fresh: bool = True) -> 'FrhodoDriver':
        """Create a Frhodo driver

        Args:
            headless: Whether to suppress the display
            fresh: Whether to skip loading in previous configuration
        Returns:
            Driver connected to a new Frhodo instance
        """
        qt_args = ['frhodo', '-platform', 'offscreen'] if headless else ['frhodo']
        app, window = launch_gui(qt_args, fresh_gui=fresh)
        return cls(window, app)
