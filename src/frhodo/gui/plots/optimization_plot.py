# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level
# directory for license and copyright information.

import numpy as np
from qtpy import QtCore

from frhodo.common.units import pa_per_unit
from frhodo.gui.plots.base_plot import Base_Plot
from frhodo.gui.views import IterationEvent, ViewRouter
from frhodo.gui.widgets import misc_widget



VIEW_LABELS = {
    "Objective Trace": "objective_trace",
    "Arrhenius with Bounds": "arrhenius",
    "Arrhenius Ratios (k / k₀)": "arrhenius_ratio",
    "Misfit Map": "misfit",
    "Improvement (start vs now)": "improvement",
    "Time Offsets": "time_offsets",
}

PRESSURE_VIEW_KEYS = ("arrhenius", "arrhenius_ratio")


class Plot(Base_Plot):
    """Optimization-outcome views host.

    The :class:`ViewRouter` owns the figure and all axes; the base
    class supplies the canvas, toolbar, and key handling. The base
    blit machinery is inert here — there are no per-axes animated
    artists to track.
    """

    def __init__(self, parent, widget, mpl_layout):
        super().__init__(parent, widget, mpl_layout)

        self.router = ViewRouter(self.fig)
        # Per-view axis-limit memory: view name -> {"home": (xlim, ylim),
        # "lims": (xlim, ylim), "user_x"/"user_y": bool}. An axis is
        # frozen only if the USER moved it off its autoscale home; the
        # other axis keeps tracking the live autoscale (a y-only zoom
        # must not stop x from extending mid-run). "home" is the autoscaled
        # state; "lims" is a user zoom/pan reapplied across switches and
        # cadence refreshes.
        self._view_state = {}
        self._lim_guard = False
        box = parent.opt_view_box
        box.blockSignals(True)
        box.clear()
        box.addItems(list(VIEW_LABELS))
        box.blockSignals(False)
        box.currentTextChanged.connect(self._view_changed)
        parent.opt_view_detail_box.currentIndexChanged.connect(
            self._detail_changed
        )
        self.ratio_box = misc_widget.CheckableSearchComboBox(parent)
        self.ratio_box.setVisible(False)
        parent.gridLayout_43.addWidget(self.ratio_box, 0, 3)
        self._ratio_updating = False
        self.ratio_box.model().itemChanged.connect(
            self._ratio_selection_changed
        )
        parent.arrhenius_P_value_box.valueChanged.connect(
            self._arrhenius_pressure_changed
        )
        parent.arrhenius_P_units_box.currentTextChanged.connect(
            self._arrhenius_unit_changed
        )
        parent.show_opt_overlay_box.stateChanged.connect(self._overlay_toggled)

        # Home means "live autoscale", not the last stack snapshot:
        # clear the per-axis freezes and recompute the view.
        self._toolbar_home = self.toolbar.home
        self.toolbar.home = self._nav_home

        parent.plot_tab_widget.currentChanged.connect(self.tab_changed)
        self._view_changed(box.currentText())

    def _overlay_toggled(self, _state=None) -> None:
        checked = self.parent.show_opt_overlay_box.isChecked()
        self.parent.plot.signal.set_opt_overlay_visible(checked)

    def tab_changed(self, idx):
        if self.parent.plot_tab_widget.tabText(idx) == "Optimization":
            self.canvas.draw_idle()

    def _decorate_axes(self):
        # The router builds and owns all axes; nothing for blit to track.
        self.ax = []

    def _draw_event(self, event=None):
        # Router views render through the standard matplotlib draw.
        pass

    def attach_run_context(self, context) -> None:
        """New optimization run: rebuild the views for its targets and
        reset the recorded history."""
        self._view_state = {}
        self.router.set_context(context)
        self.router.start_run()
        detail = self.parent.opt_view_detail_box
        detail.blockSignals(True)
        detail.clear()
        if context is not None:
            detail.addItems(list(context.rxn_equations))
        detail.blockSignals(False)
        # Populate with model signals live (the popup view and the combo's
        # current text rely on them); the guard keeps the per-item
        # itemChanged storm from re-rendering during population.
        ratio = self.ratio_box
        self._ratio_updating = True
        ratio.blockSignals(True)
        ratio.clear()
        if context is not None:
            ratio.addItems(list(context.rxn_equations))
            for i in range(ratio.count()):
                ratio.model().item(i, 0).setCheckState(QtCore.Qt.Checked)
        ratio.blockSignals(False)
        self._ratio_updating = False
        if context is not None:
            self._ratio_selection_changed()
            value_box = self.parent.arrhenius_P_value_box
            value_box.blockSignals(True)
            value_box.setValue(context.P_reference / self._pa_per_display_unit())
            value_box.blockSignals(False)
            self._arrhenius_pressure_changed()
        # Fresh run: drop the previous overlay stash and auto-enable the
        # start/best/current overlay on the signal plot.
        self.parent.plot.signal.reset_opt_overlay()
        overlay_box = self.parent.show_opt_overlay_box
        overlay_box.blockSignals(True)
        overlay_box.setChecked(True)
        overlay_box.blockSignals(False)
        self.parent.plot.signal.set_opt_overlay_visible(True)
        # set_context rebuilt the active view's axes; rewire the limit
        # callbacks and reset the toolbar's Home to the fresh view.
        self._view_changed(self.parent.opt_view_box.currentText())

    def _sync_detail_visibility(self) -> None:
        parent = self.parent
        arrhenius = (
            self.router.active_name == "arrhenius"
            and parent.opt_view_detail_box.count() > 0
        )
        ratio = (
            self.router.active_name == "arrhenius_ratio"
            and self.ratio_box.count() > 0
        )
        parent.opt_view_detail_box.setVisible(arrhenius)
        self.ratio_box.setVisible(ratio)
        parent.arrhenius_P_value_box.setVisible(arrhenius or ratio)
        parent.arrhenius_P_units_box.setVisible(arrhenius or ratio)

    def _detail_changed(self, pos: int) -> None:
        view = self.router.views.get("arrhenius")
        if view is None or pos < 0:
            return
        view.set_reaction(pos)
        if self.router.active_name == "arrhenius" and self._refresh_active():
            self.canvas.draw_idle()

    def _ratio_selection_changed(self, _item=None) -> None:
        if self._ratio_updating:
            return
        view = self.router.views.get("arrhenius_ratio")
        if view is None:
            return
        checked = [i for i in range(self.ratio_box.count())
                   if self.ratio_box.itemChecked(i)]
        view.set_selection(checked)
        if (self.router.active_name == "arrhenius_ratio"
                and self._refresh_active()):
            self.canvas.draw_idle()

    def _pa_per_display_unit(self) -> float:
        unit = self.parent.arrhenius_P_units_box.currentText().strip("[]")

        return pa_per_unit.get(unit, 1.0)

    def _pressure_views(self) -> list:
        views = []
        for key in PRESSURE_VIEW_KEYS:
            view = self.router.views.get(key)
            if view is not None:
                views.append(view)

        return views

    def _arrhenius_pressure_changed(self, _value=None) -> None:
        views = self._pressure_views()
        if not views:
            return
        value = self.parent.arrhenius_P_value_box.value()
        for view in views:
            view.set_pressure(value * self._pa_per_display_unit())
        if (self.router.active_name in PRESSURE_VIEW_KEYS
                and self._refresh_active()):
            self.canvas.draw_idle()

    def _arrhenius_unit_changed(self, _text=None) -> None:
        """Unit switch re-expresses the same physical pressure."""
        views = self._pressure_views()
        if not views:
            return
        value_box = self.parent.arrhenius_P_value_box
        value_box.blockSignals(True)
        value_box.setValue(views[0].pressure_pa / self._pa_per_display_unit())
        value_box.blockSignals(False)
        self._arrhenius_pressure_changed()

    def record_iteration(self, update: dict, is_best: bool) -> None:
        """Accumulate every iteration into the view history (cheap;
        redraws stay behind the plot cadence)."""
        self.router.record(IterationEvent.from_update(update, is_best))

    def _active_ax(self):
        view = self.router.views.get(self.router.active_name)

        return getattr(view, "ax", None)

    def _on_user_lim_change(self, ax) -> None:
        """Record a zoom/pan the user made on the active view. Returning
        to the stored autoscale limits (the toolbar Home) hands control
        back to autoscaling."""
        if self._lim_guard:
            return
        name = self.router.active_name
        if name is None or ax is not self._active_ax():
            return
        state = self._view_state.setdefault(name, {})
        lims = (ax.get_xlim(), ax.get_ylim())
        home = state.get("home")
        if home is None:
            state["user_x"] = state["user_y"] = True
            state["lims"] = lims

            return
        x_at_home = np.allclose(lims[0], home[0], rtol=1e-9)
        y_at_home = np.allclose(lims[1], home[1], rtol=1e-9)
        state["user_x"] = not x_at_home
        state["user_y"] = not y_at_home
        if x_at_home and y_at_home:
            state.pop("lims", None)
            # Re-autoscale immediately: after a run ends there is no
            # cadence tick to bring Home/Back up to the live limits.
            self.refresh()
        else:
            state["lims"] = lims

    def _wire_limit_callbacks(self, ax) -> None:
        ax.callbacks.connect("xlim_changed", self._on_user_lim_change)
        ax.callbacks.connect("ylim_changed", self._on_user_lim_change)

    def _refresh_active(self) -> bool:
        """Guarded router refresh: hold the user's zoom on frozen axes,
        let free axes track the live autoscale as Home."""
        self._lim_guard = True
        try:
            ax = self._active_ax()
            state = None
            if ax is not None:
                state = self._view_state.setdefault(
                    self.router.active_name, {},
                )
                # A zoom gesture disables autoscale on both axes;
                # restore it on the free axes so they keep tracking.
                ax.set_autoscalex_on(not state.get("user_x"))
                ax.set_autoscaley_on(not state.get("user_y"))
            changed = self.router.refresh()
            if ax is not None:
                frozen = state.get("user_x") or state.get("user_y")
                if frozen:
                    # Reapply the frozen axes (views that clear their
                    # axes reset limits); free axes keep the fresh
                    # autoscale.
                    if state.get("user_x"):
                        ax.set_xlim(state["lims"][0])
                    if state.get("user_y"):
                        ax.set_ylim(state["lims"][1])
                    state["lims"] = (ax.get_xlim(), ax.get_ylim())
                lims = (ax.get_xlim(), ax.get_ylim())
                if not frozen and state.get("home") != lims:
                    # Track the moving autoscale as Home in the
                    # toolbar's nav stack too, so pressing Home
                    # mid-run restores the live autoscale instead
                    # of a stale snapshot (which would read as a
                    # user zoom and freeze the view). While an axis
                    # is frozen the stack is left alone so Back can
                    # still return from the zoom.
                    state["home"] = lims
                    self.toolbar.update()
                    self.toolbar.push_current()
                if frozen:
                    # Only the free-axis Home components may move;
                    # frozen components keep their freeze-time values
                    # so panning back to them still reads as "at home".
                    home = state.get("home", lims)
                    x_home, y_home = lims
                    if state.get("user_x"):
                        x_home = home[0]
                    if state.get("user_y"):
                        y_home = home[1]
                    state["home"] = (x_home, y_home)
        finally:
            self._lim_guard = False

        return changed

    def refresh(self) -> None:
        """Plot-cadence update of the active view."""
        if self._refresh_active():
            self.canvas.draw_idle()

    def _nav_home(self, *args) -> None:
        """Toolbar Home: return every axis to the live autoscale."""
        name = self.router.active_name
        if name is None or self._active_ax() is None:
            self._toolbar_home(*args)

            return
        state = self._view_state.setdefault(name, {})
        state["user_x"] = state["user_y"] = False
        state.pop("lims", None)
        self._refresh_active()
        self.canvas.draw_idle()

    def _view_changed(self, label: str) -> None:
        self._lim_guard = True
        try:
            self.router.set_view(VIEW_LABELS.get(label))
            ax = self._active_ax()
            if ax is not None:
                state = self._view_state.setdefault(
                    self.router.active_name, {},
                )
                state["home"] = (ax.get_xlim(), ax.get_ylim())
                self.toolbar.update()
                self.toolbar.push_current()
                if state.get("user_x") or state.get("user_y"):
                    if state.get("user_x"):
                        ax.set_xlim(state["lims"][0])
                    if state.get("user_y"):
                        ax.set_ylim(state["lims"][1])
                    self.toolbar.push_current()
                self._wire_limit_callbacks(ax)
        finally:
            self._lim_guard = False
        self.canvas.draw_idle()
        self._sync_detail_visibility()
