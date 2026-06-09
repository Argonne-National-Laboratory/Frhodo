# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level
# directory for license and copyright information.

from tabulate import tabulate

import matplotlib as mpl
import numpy as np
from scipy import stats

from frhodo.common.units import OoM
from frhodo.experiment.uncertainty import smooth_centerline
from frhodo.gui.plots.base_plot import Base_Plot
from frhodo.gui.plots.draggable import Draggable



def shape_data(x, y):
    return np.transpose(np.vstack((x, y)))


# The optimization "Scale" combo and the plot y-axis name the same four
# scales differently (e.g. "AbsoluteLog" vs "abslog").
_OBJ_TO_PLOT_SCALE = {
    "Linear": "linear",
    "Log": "log",
    "AbsoluteLog": "abslog",
    "Bisymlog": "bisymlog",
}
_PLOT_TO_OBJ_SCALE = {plot: obj for obj, plot in _OBJ_TO_PLOT_SCALE.items()}


class Plot(Base_Plot):
    def __init__(self, parent, widget, mpl_layout):
        # reentrancy guard for the obj-fcn-scale ↔ y-axis-scale sync; True
        # during init so create_canvas()'s default-scale call doesn't try to
        # reach the not-yet-built obj-fcn box / series.
        self._scale_syncing = True
        super().__init__(parent, widget, mpl_layout)
        self._scale_syncing = False

        # Connect Signals
        self.canvas.mpl_connect("resize_event", self._resize_event)
        parent.num_sim_lines_box.valueChanged.connect(self.set_history_lines)
        parent.plot_tab_widget.currentChanged.connect(self.tab_changed)

    def tab_changed(self, idx):  # Run simulation is tab changed to Sim Explorer
        if self.parent.plot_tab_widget.tabText(idx) == "Signal/Sim":
            self._draw_event()

    def _draw_items_artist(self):  # only draw if tab is open
        idx = self.parent.plot_tab_widget.currentIndex()
        if self.parent.plot_tab_widget.tabText(idx) == "Signal/Sim":
            super()._draw_items_artist()

    def _draw_event(self, event=None):  # only draw if tab is open
        idx = self.parent.plot_tab_widget.currentIndex()
        if self.parent.plot_tab_widget.tabText(idx) == "Signal/Sim":
            super()._draw_event(event)

    def info_table_text(self):
        parent = self.parent
        # TODO: Fix variables when implementing zone 2 and 5 option
        shock_zone = parent.display_shock.zone
        if shock_zone == 2:
            display_vars = ["T2", "P2"]
        elif shock_zone == 5:
            display_vars = ["T5", "P5"]
        else:
            raise ValueError(f"unexpected shock zone {shock_zone}")

        table = [["Shock {:d}".format(parent.shock_selection.current), ""]]

        # This sets the info table to have the units selected in the shock properties window
        if not np.isnan([getattr(parent.display_shock, key) for key in display_vars]).all():
            T_unit = eval("str(parent." + display_vars[0] + "_units_box.currentText())")
            P_unit = eval("str(parent." + display_vars[1] + "_units_box.currentText())")
            T_value = parent.convert_units(
                getattr(parent.display_shock, display_vars[0]), T_unit, "out",
            )
            P_value = parent.convert_units(
                getattr(parent.display_shock, display_vars[1]), P_unit, "out",
            )
            table.append(
                ["T{:.0f} {:s}".format(shock_zone, T_unit), "{:.2f}".format(T_value)]
            )
            table.append(
                ["P{:.0f} {:s}".format(shock_zone, P_unit), "{:.2f}".format(P_value)]
            )

        for species, mol_frac in parent.display_shock.thermo_mix.items():
            table.append(["{:s}".format(species), "{:g}".format(mol_frac)])

        table = tabulate(table).split("\n")[1:-1]  # removes header and footer

        table_left_justified = []
        max_len = len(max(table, key=len))
        for line in table:
            table_left_justified.append("{:<{max_len}}".format(line, max_len=max_len))
        text = "\n".join(table_left_justified)

        return text

    def create_canvas(self):
        self.ax = []

        ## Set upper plots ##
        self.ax.append(self.fig.add_subplot(4, 1, 1))
        self.ax[0].item = {}
        self.ax[0].item["weight_unc_fcn"] = self.ax[0].add_line(
            mpl.lines.Line2D([], [], c="#800000", zorder=1)
        )
        markers = {
            "weight_shift": {"marker": "o", "markersize": 7},
            "weight_k": {"marker": "$" + "\u2194" + "$", "markersize": 12},
            "weight_extrema": {"marker": "$" + "\u2195" + "$", "markersize": 12},
        }

        for name, attr in markers.items():
            self.ax[0].item[name] = self.ax[0].add_line(
                mpl.lines.Line2D(
                    [],
                    [],
                    marker=attr["marker"],
                    markersize=attr["markersize"],
                    markerfacecolor="#BF0000",
                    markeredgecolor="None",
                    linestyle="None",
                    zorder=2,
                )
            )

        self.ax[0].item["sim_info_text"] = self.ax[0].text(
            0.98,
            0.92,
            "",
            fontsize=10,
            fontname="DejaVu Sans Mono",
            horizontalalignment="right",
            verticalalignment="top",
            transform=self.ax[0].transAxes,
        )

        self.ax[0].set_ylim(-0.1, 1.1)
        self.ax[0].tick_params(labelbottom=False)

        self.ax[0].item["title"] = self.ax[0].text(
            0.5,
            0.95,
            "Weighting",
            fontsize="large",
            horizontalalignment="center",
            verticalalignment="top",
            transform=self.ax[0].transAxes,
        )

        self.fig.subplots_adjust(
            left=0.06, bottom=0.065, right=0.98, top=0.98, hspace=0, wspace=0.12
        )

        ## Set lower plots ##
        self.ax.append(self.fig.add_subplot(4, 1, (2, 4), sharex=self.ax[0]))
        self.ax[1].item = {}
        init_array = [0, 1]
        self.ax[1].item["unc_shading"] = self.ax[1].fill_between(
            init_array,
            init_array,
            init_array,
            color="#0C94FC",
            alpha=0.2,
            linewidth=0,
            zorder=0,
        )
        self.ax[1].item["exp_data"] = self.ax[1].scatter(
            [], [], color="0", facecolors="0", linewidth=0.5, alpha=0.85, zorder=2
        )
        self.ax[1].item["sim_data"] = self.ax[1].add_line(
            mpl.lines.Line2D([], [], c="#0C94FC", zorder=4)
        )
        self.ax[1].item["history_data"] = []
        self.lastRxnNum = None

        self.ax[1].text(
            0.5,
            0.98,
            "Observable",
            fontsize="large",
            horizontalalignment="center",
            verticalalignment="top",
            transform=self.ax[1].transAxes,
        )

        self.parent.rxn_change_history = []
        self.set_history_lines()

        # Create colorbar legend
        self.cbax = self.fig.add_axes([0.90, 0.575, 0.02, 0.15], zorder=3)
        self.cb = mpl.colorbar.ColorbarBase(
            self.cbax, cmap=mpl.cm.gray, ticks=[0, 0.5, 1], orientation="vertical"
        )
        self.cbax.invert_yaxis()
        self.cbax.set_yticklabels(["1", "0.5", "0"])  # horizontal colorbar
        self.cb.set_label("Weighting")

        # Create canvas from Base
        super().create_canvas()
        self._set_scale("y", "abslog", self.ax[1])  # set Signal/SIM y axis to abslog
        self.ax[
            0
        ].animateAxisLabels = True  # set weight/unc plot to have animated axis labels

        # Add draggable lines
        draggable_items = [
            [0, "weight_shift"],
            [0, "weight_k"],
            [0, "weight_extrema"],
            [1, "sim_data"],
        ]
        for pair in draggable_items:
            n, name = pair  # n is the axis number, name is the item key
            items = self.ax[n].item[name]
            if not isinstance(items, list):  # check if the type is a list
                items = [self.ax[n].item[name]]
            for item in items:
                update_fcn = lambda x, y, item=item: self.draggable_update_fcn(
                    item, x, y
                )
                press_fcn = lambda x, y, item=item: self.draggable_press_fcn(item, x, y)
                release_fcn = lambda item=item: self.draggable_release_fcn(item)
                item.draggable = Draggable(
                    self, item, update_fcn, press_fcn, release_fcn
                )

    def set_history_lines(self):
        old_num_hist_lines = len(self.ax[1].item["history_data"])
        num_hist_lines = self.parent.num_sim_lines_box.value() - 1
        numDiff = np.abs(old_num_hist_lines - num_hist_lines)

        if old_num_hist_lines > num_hist_lines:
            for entry in self.ax[1].item["history_data"][0:numDiff]:
                entry["line"].remove()
            del self.ax[1].item["history_data"][0:numDiff]
        elif old_num_hist_lines < num_hist_lines:
            for n in range(old_num_hist_lines, old_num_hist_lines + numDiff):
                line = mpl.lines.Line2D([], [], zorder=3)
                self.ax[1].item["history_data"].append(
                    {"line": self.ax[1].add_line(line), "rxnNum": None}
                )

        color = mpl.cm.nipy_spectral(np.linspace(0.05, 0.95, num_hist_lines)[::-1])
        for n, item in enumerate(self.ax[1].item["history_data"]):
            item["line"].set_color(color[n])

        if hasattr(self, "canvas"):  # this can be deleted after testing color changes
            self._draw_items_artist()

    def draggable_press_fcn(self, item, x, y):
        x0, xpress, xnew, xpressnew = x["0"], x["press"], x["new"], x["press_new"]
        y0, ypress, ynew, ypressnew = y["0"], y["press"], y["new"], y["press_new"]
        xy_data = item.get_xydata()

        xy_press = np.array([xpress, ypress])
        xy_OoM = 10 ** OoM(xy_press)

        # calculate distance from press and points, don't need sqrt for comparison, divide by OoM for large differences in x/y OoM
        distance_cmp = np.sum(
            np.subtract(xy_data / xy_OoM, xy_press / xy_OoM) ** 2, axis=1
        )
        item.draggable.nearest_index = np.nanargmin(
            distance_cmp
        )  # choose closest point to press

    def draggable_release_fcn(self, item):
        item.draggable.nearest_index = 0  # reset nearest_index

        if item is self.ax[1].item["sim_data"]:
            parent = self.parent
            parent.tree._copy_expanded_tab_rates()
            self.update_sim(parent.SIM.independent_var, parent.SIM.observable)

    def draggable_update_fcn(self, item, x, y):
        parent = self.parent

        x = {
            key: np.array(val) / parent.reactor_state.t_unit_conv
            for key, val in x.items()
        }  # scale with unit choice
        x0, xpress, xnew, xpressnew = x["0"], x["press"], x["new"], x["press_new"]
        y0, ypress, ynew, ypressnew = y["0"], y["press"], y["new"], y["press_new"]
        exp_data = parent.display_shock.exp_data

        if item is self.ax[1].item["sim_data"]:
            time_offset = np.round(xnew[0] / 0.01) * 0.01
            for box in parent.time_offset_box.twin:
                box.blockSignals(True)
                box.setValue(time_offset)
                box.blockSignals(False)

            t_unit_conv = parent.reactor_state.t_unit_conv
            parent.time_uncertainty.offset = time_offset * t_unit_conv
            parent.display_shock.time_offset = time_offset * t_unit_conv
            parent.time_uncertainty.auto_fit = False  # user drag overrides

            sim_data = self.ax[1].item["sim_data"]
            sim_data.set_xdata(parent.SIM.independent_var + parent.display_shock.time_offset)

        elif item is self.ax[0].item["weight_shift"]:
            t_conv = parent.reactor_state.t_unit_conv
            n = item.draggable.nearest_index

            xnew = (
                (xnew[n] * t_conv - exp_data[0, 0])
                / (exp_data[-1, 0] - exp_data[0, 0])
                * 100
            )
            shift_arr = parent.display_shock.weight_shift
            if n == 0:
                xnew = max(0.0, min(xnew, shift_arr[1]))
            elif n == 1:
                xnew = max(shift_arr[0], min(xnew, 100.0))
            else:
                raise ValueError(f"unexpected weight-shift index {n}")

            parent.weight.boxes["weight_shift"][n].setValue(xnew)

        elif item is self.ax[0].item["weight_k"]:
            n = item.draggable.nearest_index
            i = n // 2

            shift = parent.display_shock.weight_shift[i]
            shift = shift / 100 * (exp_data[-1, 0] - exp_data[0, 0]) + exp_data[0, 0]
            shift /= parent.reactor_state.t_unit_conv

            sigma_new = -((-1) ** n) * (xnew[n] - shift)
            sigma_new = max(0.0, sigma_new)

            parent.weight.boxes["weight_k"][i].setValue(sigma_new)

        elif item is self.ax[0].item["weight_extrema"]:
            xy_data = item.get_xydata()
            n = item.draggable.nearest_index
            if n != 1:
                weight_type = "weight_min"
                i = n // 2
            else:
                weight_type = "weight_max"
                i = 0

            box = parent.weight.boxes[weight_type][i]
            GUI_max = getattr(parent.display_shock, weight_type)[i]
            extrema_new = ynew[n] + GUI_max - xy_data[n][1]
            box.setValue(extrema_new)

        # Update plot if data exists
        if exp_data.size > 0:
            parent.update_user_settings()
            self.update()

    def _resize_event(self, event=None):
        canvas_width = self.canvas.size().width()
        left = (
            -7.6e-08 * canvas_width**2 + 2.2e-04 * canvas_width + 7.55e-01
        )  # Might be better to adjust by pixels
        self.cbax.set_position([left, 0.575, 0.02, 0.15])

    def update(self, update_lim=False):
        parent = self.parent
        if parent.display_shock.exp_data.size == 0:
            return

        t = parent.display_shock.exp_data[:, 0]
        data = parent.display_shock.exp_data[:, 1]

        self.update_weight_plot()
        self.update_uncertainty_shading()

        # Update lower plot
        weights = parent.display_shock.weights
        self.ax[1].item["exp_data"].set_offsets(shape_data(t, data))
        self.ax[1].item["exp_data"].set_facecolor(np.char.mod("%f", 1 - weights))

        self.update_info_text()

        if update_lim:
            self.update_xylim(self.ax[1])

    def update_weight_plot(self):
        parent = self.parent
        if parent.display_shock.exp_data.size == 0:
            return

        t = parent.display_shock.exp_data[:, 0]

        shift = (
            np.array(parent.display_shock.weight_shift) / 100 * (t[-1] - t[0]) + t[0]
        )
        inv_growth_rate = (
            np.array(parent.display_shock.weight_k)
            * self.parent.reactor_state.t_unit_conv
        )

        weight_fcn = parent.series.weights
        weights = parent.display_shock.weights = weight_fcn(t)

        self.ax[0].item["weight_unc_fcn"].set_xdata(t)
        self.ax[0].item["weight_unc_fcn"].set_ydata(weights)

        # calculate mu markers
        mu = shift
        f_mu = weight_fcn(mu, calcIntegral=False)

        # calculate extrema markers
        t_range = np.max(t) - np.min(t)
        t_extrema = (
            np.array([np.min(t), np.mean(mu), np.max(t)])
            + np.array([0.0125, 0, -0.025]) * t_range
        )  # put arrow at 95% of x data

        # calculate sigma markers
        ones_shape = (np.shape(f_mu)[0], 2)
        sigma = (
            np.ones(ones_shape) * mu
            + (np.ones(ones_shape) * np.array([-1, 1])).T * inv_growth_rate
        )
        sigma = sigma.T  # sort may be unnecessary
        f_sigma = np.reshape(
            weight_fcn(sigma.flatten(), calcIntegral=False), ones_shape
        )

        for i in np.argwhere(inv_growth_rate == 0.0):
            f = weight_fcn(
                np.array([(1.0 - 1e-3), (1.0 + 1e-3)]) * mu[i], calcIntegral=False
            )
            f_mu[i] = np.mean(f)
            perc = 0.1824
            f_sigma[i] = [
                (1 - perc) * f[0] + perc * f[1],
                perc * f[0] + (1 - perc) * f[1],
            ]

        sigma = sigma.flatten()
        f_sigma = f_sigma.flatten()
        if (
            sigma[1] >= 0.80 * t_extrema[1] + 0.20 * mu[0]
        ):  # hide sigma symbols if too close to center extrema
            sigma[1] = np.nan
        if sigma[2] <= 0.75 * t_extrema[1] + 0.25 * mu[1]:
            sigma[2] = np.nan

        # Set markers
        self.ax[0].item["weight_shift"].set_xdata(mu)
        self.ax[0].item["weight_shift"].set_ydata(f_mu)

        self.ax[0].item["weight_k"].set_xdata(sigma.flatten())
        self.ax[0].item["weight_k"].set_ydata(f_sigma.flatten())

        self.ax[0].item["weight_extrema"].set_xdata(t_extrema)
        self.ax[0].item["weight_extrema"].set_ydata(
            weight_fcn(t_extrema, calcIntegral=False)
        )

    def update_uncertainty_shading(self):
        """Shade the 95% credible interval around the experimental trace.

        Draws ``μ̂(t) ± 1.96·σ̂(t)`` in the optimizer's residual scale,
        where μ̂ is a σ-weighted smoothing spline through the data and
        σ̂ comes from the 4th-order Hall-Müller + Lepski estimator,
        post-smoothed for visual continuity. Hidden in Residual mode.
        """
        parent = self.parent
        shock = parent.display_shock
        if parent.obj_fcn_type_box.currentText() != "Bayesian":
            self.ax[1].item["unc_shading"].set_visible(False)

            return
        if shock.exp_data.size == 0:
            self.ax[1].item["unc_shading"].set_visible(False)

            return

        scale = parent.series.scale_for(shock)
        sigma_t = getattr(shock, "sigma_t", None)
        if sigma_t is None or sigma_t.size != shock.exp_data.shape[0]:
            parent.series.uncertainties(shock)
            sigma_t = shock.sigma_t

        t = shock.exp_data[:, 0]
        y = shock.exp_data[:, 1]
        mu_scaled = smooth_centerline(y, scale=scale)
        k = 1.96
        lower = scale.inverse(mu_scaled - k * sigma_t)
        upper = scale.inverse(mu_scaled + k * sigma_t)

        finite = np.isfinite(t) & np.isfinite(lower) & np.isfinite(upper)
        if not finite.any():
            self.ax[1].item["unc_shading"].set_visible(False)

            return

        self.ax[1].item["unc_shading"].remove()
        self.ax[1].item["unc_shading"] = self.ax[1].fill_between(
            t[finite], lower[finite], upper[finite],
            color="#0C94FC", alpha=0.3, linewidth=0, zorder=3,
        )

    def on_obj_fcn_type_changed(self):
        """Show the data-uncertainty band only in Bayesian mode."""
        self.update_uncertainty_shading()
        self.update()
        self._draw_items_artist()

    def _set_scale(self, coord, type, event, update_xylim=False):
        """Set an axis scale; mirror the observable y-scale to the obj-fcn box."""
        super()._set_scale(coord, type, event, update_xylim)
        if self._scale_syncing:
            return
        axes = self._find_calling_axes(event)
        obj_scale = _PLOT_TO_OBJ_SCALE.get(type)
        box = getattr(self.parent, "obj_fcn_scale_box", None)
        if coord != "y" or axes is not self.ax[1] or obj_scale is None or box is None:
            return
        if box.currentText() == obj_scale:
            return
        self._scale_syncing = True
        try:
            box.setCurrentText(obj_scale)
        finally:
            self._scale_syncing = False

    def _sync_y_scale_to_obj_fcn(self):
        """Set the observable y-axis scale to match the obj-fcn scale box."""
        box = getattr(self.parent, "obj_fcn_scale_box", None)
        if self._scale_syncing or box is None:
            return
        plot_scale = _OBJ_TO_PLOT_SCALE.get(box.currentText())
        if plot_scale is None or self.ax[1].scale["y"].get(plot_scale, False):
            return
        self._scale_syncing = True
        try:
            self._set_scale("y", plot_scale, self.ax[1], update_xylim=True)
        finally:
            self._scale_syncing = False

    def on_obj_fcn_scale_changed(self):
        """Sync the y-axis to the cost scale, then recompute σ for the shock."""
        parent = self.parent
        self._sync_y_scale_to_obj_fcn()
        for shock_group in getattr(parent.series, "shock", None) or []:
            for shock in shock_group:
                if hasattr(shock, "_scale"):
                    del shock._scale
        shock = getattr(parent, "display_shock", None)
        if shock is not None and getattr(shock, "exp_data", np.array([])).size > 0:
            parent.series.uncertainties(shock)
        self.update_uncertainty_shading()
        self.update()
        self._draw_items_artist()

    def update_info_text(self, redraw=False):
        self.ax[0].item["sim_info_text"].set_text(self.info_table_text())
        if redraw:
            self._draw_items_artist()

    def clear_sim(self):
        self.ax[1].item["sim_data"].raw_data = np.array([])
        self.ax[1].item["sim_data"].set_xdata([])
        self.ax[1].item["sim_data"].set_ydata([])

    def update_sim(self, t, observable, rxnChanged=False):
        time_offset = self.parent.display_shock.time_offset
        exp_data = self.parent.display_shock.exp_data

        self.ax[0].item["sim_info_text"].set_text(self.info_table_text())

        if len(self.ax[1].item["history_data"]) > 0:
            self.update_history()

        # logic to update lim
        self.sim_update_lim = False
        if hasattr(self.ax[1].item["sim_data"], "raw_data"):
            old_data = self.ax[1].item["sim_data"].raw_data
            if old_data.size == 0 or old_data.ndim != 2 or old_data[-1, 0] != t[-1]:
                self.sim_update_lim = True
        else:
            self.sim_update_lim = True

        self.ax[1].item["sim_data"].raw_data = np.array([t, observable]).T
        self.ax[1].item["sim_data"].set_xdata(t + time_offset)
        self.ax[1].item["sim_data"].set_ydata(observable)

        if (
            exp_data.size == 0 and not np.isnan(t).any()
        ):  # if exp data doesn't exist rescale
            self.set_xlim(self.ax[1], [t[0], t[-1]])
            if (
                np.count_nonzero(observable) > 0
            ):  # only update ylim if not all values are zero
                self.set_ylim(self.ax[1], observable)
                self._draw_event()
        else:
            if self.sim_update_lim:
                self.update_xylim(self.ax[1])
            else:
                self._draw_items_artist()

    def update_history(self):
        def reset_history_lines(line):
            for n in range(0, len(line)):
                line[n]["line"].set_xdata([])
                line[n]["line"].set_ydata([])
                line[n]["rxnNum"] = None

        numHist = self.parent.num_sim_lines_box.value()
        rxnHist = self.parent.rxn_change_history

        if len(rxnHist) > 0:
            if self.lastRxnNum != rxnHist[-1]:  # only update if the rxnNum changed
                self.lastRxnNum = rxnHist[-1]
            else:
                if self.lastRxnNum is None:  # don't update from original mech
                    self.lastRxnNum = rxnHist[-1]

                return
        else:
            self.lastRxnNum = None
            reset_history_lines(self.ax[1].item["history_data"])

            return

        histRxnNum = [item["rxnNum"] for item in self.ax[1].item["history_data"]]

        if rxnHist[-1] in histRxnNum:  # if matching rxnNum, replace that
            n = histRxnNum.index(rxnHist[-1])
        else:
            firstNone = next((n for n, x in enumerate(histRxnNum) if x is None), None)

            if firstNone is not None:
                n = firstNone
            else:  # if no matching rxnNums, replace differing rxnNum
                s = set(histRxnNum)
                n = [n for n, x in enumerate(rxnHist[:-numHist:-1]) if x not in s][0]

        hist = self.ax[1].item["history_data"][n]
        hist["rxnNum"] = rxnHist[-1]
        hist["line"].set_xdata(self.ax[1].item["sim_data"].get_xdata())
        hist["line"].set_ydata(self.ax[1].item["sim_data"].get_ydata())
