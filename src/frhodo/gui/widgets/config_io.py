# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level
# directory for license and copyright information.
"""GUI bridge between Qt widgets and ``FrhodoConfig``.

``GUI_settings`` reads the persisted ``default_config.yaml`` into a
typed :class:`frhodo.common.config.FrhodoConfig`, then drives the Qt boxes
from its attributes (load); on save it pulls box values back onto the
model and dumps via ``to_yaml_text()``. There is no intermediate
dict-shaped view.
"""
import io

from qtpy import QtWidgets

from frhodo.common.errors import SchemaVersionError
from frhodo.common.config import FrhodoConfig



def _set_box(box, val):
    """Best-effort set of a Qt input box; missing setters are no-ops."""
    try:
        if isinstance(box, (QtWidgets.QDoubleSpinBox, QtWidgets.QSpinBox)):
            box.setValue(val)
        elif isinstance(box, QtWidgets.QComboBox):
            box.setCurrentText(val)
        elif isinstance(box, QtWidgets.QCheckBox):
            box.setChecked(val)
        elif isinstance(box, QtWidgets.QTextEdit):
            box.setPlainText(val)
    except Exception:
        pass


class GUI_settings:
    def __init__(self, parent):
        self.parent = parent
        self.config = FrhodoConfig()

    def load(self, *_args, **_kwargs):
        parent = self.parent

        cfg_path = parent.path["default_config"]
        if cfg_path.exists():
            text = cfg_path.read_text(encoding="utf-8")
            try:
                self.config = FrhodoConfig.from_yaml_text(text)
            except SchemaVersionError as exc:
                if hasattr(parent, "log"):
                    parent.log.append(str(exc), alert=True)
                self.config = FrhodoConfig()
            except Exception as exc:
                if hasattr(parent, "log"):
                    parent.log.append(
                        f"Config validation failed (delete {cfg_path} "
                        f"to regenerate from defaults): {exc}",
                        alert=True,
                    )
                self.config = FrhodoConfig()
        else:
            self.config = FrhodoConfig()

        self.apply_config_to_boxes()

    def apply_config_to_boxes(self):
        """Drive the Qt input boxes from ``self.config`` (no disk read)."""
        parent = self.parent
        cfg = self.config

        _set_box(parent.T1_units_box, f"[{cfg.experiment.temperature_units.zone_1}]")
        _set_box(parent.T2_units_box, f"[{cfg.experiment.temperature_units.zone_2}]")
        _set_box(parent.T5_units_box, f"[{cfg.experiment.temperature_units.zone_5}]")

        _set_box(parent.P1_units_box, f"[{cfg.experiment.pressure_units.zone_1}]")
        _set_box(parent.P2_units_box, f"[{cfg.experiment.pressure_units.zone_2}]")
        _set_box(parent.P5_units_box, f"[{cfg.experiment.pressure_units.zone_5}]")

        _set_box(parent.u1_units_box, f"[{cfg.experiment.velocity_units}]")

        _set_box(parent.reactor_select_box, cfg.reactor.type)
        _set_box(parent.solve_energy_box, cfg.reactor.solve_energy)
        _set_box(parent.frozen_comp_box, cfg.reactor.frozen_composition)
        _set_box(parent.end_time_value_box, cfg.reactor.simulation_end_time.value)
        _set_box(
            parent.end_time_units_box,
            f"[{cfg.reactor.simulation_end_time.units}]",
        )
        _set_box(parent.ODE_solver_box, cfg.reactor.ode_solver)
        _set_box(
            parent.sim_interp_factor_box,
            cfg.reactor.simulation_interpolation_factor,
        )

        opt = cfg.optimization
        _set_box(parent.time_unc_box, opt.time_uncertainty)
        _set_box(parent.random_t_unc_box, opt.random_t_uncertainty)
        _set_box(parent.obj_fcn_type_box, opt.objective_function_type)
        _set_box(parent.obj_fcn_scale_box, opt.objective_function_scale)
        _set_box(parent.loss_alpha_box, str(opt.loss_function_alpha))
        _set_box(parent.loss_c_box, opt.loss_function_c)
        _set_box(parent.bayes_dist_type_box, opt.bayesian_distribution_type)
        _set_box(parent.bayes_unc_sigma_box, opt.bayesian_uncertainty_sigma)
        _set_box(parent.multiprocessing_box, opt.multiprocessing)

        for opt_type in ("global", "local"):
            widget = parent.optimization_settings.widgets[opt_type]

            if opt_type == "global":
                _set_box(parent.global_opt_enable_box, opt.enabled[opt_type])
                _set_box(parent.global_opt_choice_box, opt.algorithm[opt_type])
                _set_box(
                    widget["initial_pop_multiplier"],
                    opt.initial_population_multiplier[opt_type],
                )
            else:
                _set_box(parent.local_opt_enable_box, opt.enabled[opt_type])
                _set_box(parent.local_opt_choice_box, opt.algorithm[opt_type])

            _set_box(widget["initial_step"], opt.initial_step[opt_type])
            _set_box(widget["stop_criteria_type"], opt.stop_criteria_type[opt_type])
            _set_box(widget["stop_criteria_val"], opt.stop_criteria_value[opt_type])
            _set_box(widget["xtol_rel"], opt.relative_x_tolerance[opt_type])
            _set_box(widget["ftol_rel"], opt.relative_fcn_tolerance[opt_type])

        shock = parent.display_shock
        wf = opt.weight_function
        shock.weight_max = [wf.max]
        shock.weight_min = list(wf.min)
        shock.weight_shift = list(wf.time_location)
        shock.weight_k = list(wf.inverse_growth_rate)
        parent.weight.set_boxes()

        parent.plot.signal._set_scale(
            "x", cfg.plot.x_scale, parent.plot.signal.ax[1], True,
        )
        parent.plot.signal._set_scale(
            "y", cfg.plot.y_scale, parent.plot.signal.ax[1], True,
        )

        parent.shock_choice_box.setValue(1)
        parent.path_file_box.setPlainText(cfg.directory.directory_file)

    def save(self, save_all: bool = False):
        self.pull_config_from_boxes()

        with io.open(
            self.parent.path["default_config"], "w", encoding="utf-8",
        ) as f:
            f.write(self.config.to_yaml_text())

    def pull_config_from_boxes(self):
        """Read the Qt input boxes back onto ``self.config`` (no disk write)."""
        parent = self.parent
        cfg = self.config

        cfg.directory.directory_file = str(parent.path["path_file"])

        for i in (1, 2, 5):
            T_unit = (
                getattr(parent, f"T{i}_units_box").currentText().lstrip("[").rstrip("]")
            )
            P_unit = (
                getattr(parent, f"P{i}_units_box").currentText().lstrip("[").rstrip("]")
            )
            setattr(cfg.experiment.temperature_units, f"zone_{i}", T_unit)
            setattr(cfg.experiment.pressure_units, f"zone_{i}", P_unit)

        cfg.experiment.velocity_units = (
            parent.u1_units_box.currentText().lstrip("[").rstrip("]")
        )

        cfg.reactor.type = parent.reactor_select_box.currentText()
        cfg.reactor.solve_energy = parent.solve_energy_box.isChecked()
        cfg.reactor.frozen_composition = parent.frozen_comp_box.isChecked()
        cfg.reactor.simulation_end_time.value = parent.end_time_value_box.value()
        cfg.reactor.simulation_end_time.units = (
            parent.end_time_units_box.currentText().lstrip("[").rstrip("]")
        )
        cfg.reactor.ode_solver = parent.ODE_solver_box.currentText()
        cfg.reactor.simulation_interpolation_factor = (
            parent.sim_interp_factor_box.value()
        )

        opt = cfg.optimization
        opt.time_uncertainty = parent.time_unc_box.value()
        opt.random_t_uncertainty = parent.random_t_unc_box.isChecked()
        opt.objective_function_type = parent.obj_fcn_type_box.currentText()
        opt.objective_function_scale = parent.obj_fcn_scale_box.currentText()
        loss_alpha_text = parent.loss_alpha_box.currentText()
        try:
            opt.loss_function_alpha = float(loss_alpha_text)
        except (TypeError, ValueError):
            opt.loss_function_alpha = loss_alpha_text
        opt.loss_function_c = parent.loss_c_box.value()
        opt.bayesian_distribution_type = parent.bayes_dist_type_box.currentText()
        opt.bayesian_uncertainty_sigma = parent.bayes_unc_sigma_box.value()
        opt.multiprocessing = parent.multiprocessing_box.isChecked()

        for opt_type in ("global", "local"):
            widget = parent.optimization_settings.widgets[opt_type]

            if opt_type == "global":
                opt.enabled[opt_type] = parent.global_opt_enable_box.isChecked()
                opt.algorithm[opt_type] = parent.global_opt_choice_box.currentText()
                opt.initial_population_multiplier[opt_type] = widget[
                    "initial_pop_multiplier"
                ].value()
            else:
                opt.enabled[opt_type] = parent.local_opt_enable_box.isChecked()
                opt.algorithm[opt_type] = parent.local_opt_choice_box.currentText()

            opt.initial_step[opt_type] = widget["initial_step"].value()
            opt.stop_criteria_type[opt_type] = widget["stop_criteria_type"].currentText()
            opt.stop_criteria_value[opt_type] = widget["stop_criteria_val"].value()
            opt.relative_x_tolerance[opt_type] = widget["xtol_rel"].value()
            opt.relative_fcn_tolerance[opt_type] = widget["ftol_rel"].value()

        shock = parent.display_shock
        opt.weight_function.max = shock.weight_max[0]
        opt.weight_function.min = list(shock.weight_min)
        opt.weight_function.time_location = list(shock.weight_shift)
        opt.weight_function.inverse_growth_rate = list(shock.weight_k)

        cfg.plot.x_scale = parent.plot.signal.ax[1].get_xscale()
        cfg.plot.y_scale = parent.plot.signal.ax[1].get_yscale()

        cfg.window.maximized = parent.isMaximized()
        if not parent.isMaximized():
            cfg.window.width = parent.width()
            cfg.window.height = parent.height()
        sizes = parent.splitter.sizes()
        if sizes:
            cfg.window.option_panel_width = sizes[0]
