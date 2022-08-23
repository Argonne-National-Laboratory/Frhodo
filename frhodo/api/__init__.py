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
        """Change which shock experiment is running

        Args:
            n: Which shock experiment to evaluate
        """

        self.window.shock_choice_box.setValue(n + 1)
        self.app.processEvents()

    def get_observables(self) -> List[np.ndarray]:
        """Get the observable data from each shock experiment

        Returns:
            List of experimental data arrays where each is a 2D
             array with the first column is the time and second
             is the observable
        """
        # Loop over each shock
        output = []
        for i in range(self.n_shocks):
            self._select_shock(i)
            output.append(self.window.display_shock['exp_data'])
        return output
