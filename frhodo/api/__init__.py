"""Functions related to using Frhodo from a Python interface

Many of these functions require launching the Frhodo GUI first.
using :meth:`~fhrodo.main.launch_gui`
"""

from pathlib import Path

from PyQt5.QtWidgets import QApplication

from ..main import Main


def load_files(window: Main,
               app: QApplication,
               experiment_path: Path,
               mechanism_path: Path,
               output_path: Path):
    """Load the input files for Frhodo

    Args:
        window: Link to the main window
        app: Link to the Qt application
        experiment_path: Path to the experimental data
        mechanism_path: Path to the mechanism files
        output_path: Path to which simulation results should be stored
    """

    window.exp_main_box.setPlainText(str(experiment_path.resolve().absolute()))
    window.mech_main_box.setPlainText(str(mechanism_path.resolve().absolute()))
    window.sim_main_box.setPlainText(str(output_path.resolve().absolute()))

    # Trigger Frhodo to process these files
    app.processEvents()

    return
