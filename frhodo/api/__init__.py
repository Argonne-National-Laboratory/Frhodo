"""Functions related to using Frhodo from a Python interface

Many of these functions require launching the Frhodo GUI first.
using :meth:`~fhrodo.main.launch_gui`
"""

from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
from PyQt5.QtWidgets import QApplication

from ..main import Main


class FrhodoDriver:
    """Driver the Frhodo GUI application

    Implements simple interfaces for common tasks, such as loading different datasets
    or evaluating changes to different mechanisms

    **Limitations**

    The driver only supports a subset of features of Frhodo

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
    def rxn_coeffs(self) -> List[List[Dict[str, float]]]:
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

    def get_observables(self) -> List[np.ndarray]:
        """Get the observable data from each shock experiment

        Returns:
            List of experimental data arrays where each is a 2D
             array with the first column is the time and second
             is the observable
        """

        if self.n_shocks == 0:
            return []

        # Loop over each shock
        output = []
        for shock in self.window.series.shock[0]:
            output.append(shock['exp_data'])
        return output

    def run_simulations(self) -> List[np.ndarray]:
        """Run the simulation for each of the observed

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

    def change_coefficient(self, new_values: Dict[Tuple[int, int, str], float]):
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
