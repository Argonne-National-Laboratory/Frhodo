"""Functions related to using Frhodo from a Python interface

Many of these functions require launching the Frhodo GUI first.
using :meth:`~fhrodo.main.launch_gui`
"""

from pathlib import Path
from typing import List

import numpy as np
from PyQt5.QtWidgets import QApplication

from ..main import Main


class FrhodoDriver:
    """Driver the Frhodo GUI application

    Implements simple interfaces for common tasks, such as loading differnet datasets
    or evaluating changes to different mechanisms
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
                   output_path: Path):
        """Load the input files for Frhodo

        Args:
            experiment_path: Path to the experimental data
            mechanism_path: Path to the mechanism files
            output_path: Path to which simulation results should be stored
        """

        self.window.exp_main_box.setPlainText(str(experiment_path.resolve().absolute()))
        self.window.mech_main_box.setPlainText(str(mechanism_path.resolve().absolute()))
        self.window.sim_main_box.setPlainText(str(output_path.resolve().absolute()))

        # Trigger Frhodo to process these files
        self.app.processEvents()

    @property
    def n_shocks(self):
        """Number of shock experiments loaded"""
        n_series = len(self.window.series.shock)
        assert n_series <= 1, "We only support one series"
        if n_series == 0:
            return 0
        return len(self.window.series.shock[0])

    def _select_shock(self, n: int):
        """Change which shock experiment is being displayed and simulated

        Args:
            n: Which shock experiment to evaluate
        """

        # Check if it is in the list
        assert self.n_shocks > 0
        assert any(n == x['num'] for x in self.window.series.shock[0])
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
