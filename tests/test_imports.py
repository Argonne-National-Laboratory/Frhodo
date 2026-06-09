"""Every src/frhodo module imports cleanly."""
import importlib

import pytest

ENGINE_MODULES = [
    "frhodo",
    "frhodo.api",
    "frhodo.common",
    "frhodo.common.config",
    "frhodo.common.errors",
    "frhodo.common.units",
    "frhodo.simulation",
    "frhodo.simulation.mechanism",
    "frhodo.simulation.mechanism.mech_fcns",
    "frhodo.simulation.mechanism.mechanism_loader",
    "frhodo.simulation.mechanism.fit_coeffs",
    "frhodo.simulation.shock",
    "frhodo.simulation.shock.state",
    "frhodo.simulation.shock.reactor_output",
    "frhodo.simulation.shock.zero_d_reactor",
    "frhodo.simulation.shock.shock_solver",
    "frhodo.simulation.shock.incident_shock_reactor",
    "frhodo.experiment",
    "frhodo.experiment.data",
    "frhodo.experiment.parsers",
    "frhodo.experiment.uncertainty",
    "frhodo.experiment.weight",
    "frhodo.optimize",
    "frhodo.optimize.algorithms",
    "frhodo.simulation.mechanism.coef_helpers",
    "frhodo.optimize.cost",
    "frhodo._vendor.opendsm.adaptive_loss",
    "frhodo._vendor.opendsm.adaptive_loss_Z",
    "frhodo._vendor.opendsm.outliers",
    "frhodo._vendor.opendsm.stats_basic",
    "frhodo._vendor.opendsm.utils",
    "frhodo.optimize.cost.bayesian",
    "frhodo.optimize.cost.fit_fcn",
    "frhodo.optimize.cost.settings",
]

GUI_MODULES = [
    "frhodo.app",
    "frhodo.gui",
    "frhodo.gui.optimize_orchestrator",
    "frhodo.gui.workers.optimize_worker",
    "frhodo.gui.widgets.colors",
    "frhodo.gui.widgets.config_io",
    "frhodo.gui.widgets.error_window",
    "frhodo.gui.widgets.help_menu",
    "frhodo.gui.widgets.mech_widget",
    "frhodo.gui.widgets.misc_widget",
    "frhodo.gui.widgets.options_panel_widgets",
    "frhodo.gui.widgets.save_output",
    "frhodo.gui.widgets.save_widget",
    "frhodo.gui.widgets.series_viewer_widget",
    "frhodo.gui.widgets.settings",
    "frhodo.gui.widgets.sim_explorer_widget",
    "frhodo.gui.widgets.thermo_widget",
    "frhodo.gui.plots.base_plot",
    "frhodo.gui.plots.custom_mpl_ticker_formatter",
    "frhodo.gui.plots.custom_mplscale",
    "frhodo.gui.plots.draggable",
    "frhodo.gui.plots.optimization_plot",
    "frhodo.gui.plots.plot_main",
    "frhodo.gui.plots.plot_widget",
    "frhodo.gui.plots.raw_signal_plot",
    "frhodo.gui.plots.signal_plot",
    "frhodo.gui.plots.sim_explorer_plot",
]


@pytest.mark.parametrize("module_name", ENGINE_MODULES)
def test_engine_module_imports(module_name):
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", GUI_MODULES)
def test_gui_module_imports(module_name):
    importlib.import_module(module_name)
