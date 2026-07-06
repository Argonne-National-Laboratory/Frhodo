"""Smoke tests for ``Main`` window construction.

Booting ``Main`` exercises every import + widget construction path the
GUI uses. This is the catch-net for issues like the matplotlib
``set_xdata(scalar)`` break or qtpy import errors that would otherwise
only surface when a user clicks a button.
"""
import numpy as np
import pytest
from qtpy import QtCore

from frhodo.gui.plots.optimization_plot import VIEW_LABELS
from frhodo.gui.screening_runner import (
    format_screening_summary,
    gather_screening_shocks,
)
from frhodo.gui.views.events import ViewContext
from frhodo.optimize.screening import ScreeningResult
from frhodo.simulation.mechanism.mech_fcns import ChemicalMechanism



pytestmark = pytest.mark.gui


class TestMainWindowBoot:
    def test_constructs_without_exception(self, main_window):
        assert main_window is not None

    def test_has_expected_top_level_attributes(self, main_window):
        """A handful of attributes that callers across the codebase rely on."""
        for attr in ("plot", "mech", "convert_units", "user_settings", "series"):
            assert hasattr(main_window, attr), (
                f"Main missing attribute '{attr}' after init"
            )

    def test_mechanism_object_initialized(self, main_window):
        assert isinstance(main_window.mech, ChemicalMechanism)

    def test_settings_path_added_to_path_dict(self, main_window, isolated_path):
        """``settings.Path`` populates derived path entries from ``appdata``."""
        for key in ("default_config", "Cantera_Mech", "graphics"):
            assert key in isolated_path, (
                f"settings.Path did not register '{key}' on the path dict"
            )

    def test_plot_subwidgets_constructed(self, main_window):
        for attr in ("raw_sig", "signal", "opt"):
            assert hasattr(main_window.plot, attr), (
                f"main.plot is missing sub-plot '{attr}'"
            )
        assert callable(main_window.plot.raw_sig.update)


class TestMechWidgetPopulation:
    """Loading Cycloheptane populates ``mech_tree``."""

    def test_tree_data_has_one_entry_per_reaction(self, main_with_loaded_mech):
        n_rxn = main_with_loaded_mech.mech.gas.n_reactions
        assert len(main_with_loaded_mech.tree.mech_tree_data) == n_rxn, (
            f"mech_tree_data should have {n_rxn} entries, "
            f"got {len(main_with_loaded_mech.tree.mech_tree_data)}"
        )

    def test_qmodel_row_count_matches_reaction_count(self, main_with_loaded_mech):
        """The Qt model is what the reaction table view binds to."""
        assert main_with_loaded_mech.tree.model.rowCount() == 66, (
            "Qt model row count should equal mech.n_reactions for Cycloheptane"
        )

    def test_first_tree_row_holds_reaction_zero(self, main_with_loaded_mech):
        first = main_with_loaded_mech.tree.mech_tree_data[0]
        assert first["num"] == 0
        assert first["eqn"] == "cC7H14 <=> 1C7H14", (
            f"first reaction equation drift: got {first['eqn']!r}"
        )


class TestShockPathsDiscovery:
    """``Path.shock_paths`` finds Shock<N>.<ext> files under exp_main."""

    @pytest.fixture
    def main_with_synthetic_exp_dir(self, main_window, tmp_path):
        """Plant fake shock files at varying depths to exercise the
        depth-limit and de-duplication logic without touching the bundled
        example tree."""
        exp_dir = tmp_path / "exp"
        (exp_dir / "set_a").mkdir(parents=True)
        (exp_dir / "set_a" / "deep").mkdir()

        (exp_dir / "Shock1.exp").write_text("")
        (exp_dir / "Shock2.exp").write_text("")
        (exp_dir / "set_a" / "Shock3.exp").write_text("")
        # max_depth=2 → this 3-deep file should NOT show up:
        (exp_dir / "set_a" / "deep" / "Shock99.exp").write_text("")

        main_window.path["exp_main"] = exp_dir

        return main_window

    def test_returns_one_row_per_shock(self, main_with_synthetic_exp_dir):
        result = main_with_synthetic_exp_dir.path_set.shock_paths(
            prefix="Shock", ext="exp", max_depth=2,
        )
        assert len(result) == 3, (
            f"expected 3 shocks (1, 2, 3) within max_depth=2, "
            f"got {len(result)}"
        )

    def test_excludes_files_beyond_max_depth(self, main_with_synthetic_exp_dir):
        result = main_with_synthetic_exp_dir.path_set.shock_paths(
            prefix="Shock", ext="exp", max_depth=2,
        )
        shock_nums = [int(row[0]) for row in result]
        assert 99 not in shock_nums, (
            f"Shock99 is at depth 3 and should be excluded; got {shock_nums}"
        )

    def test_results_sorted_by_shock_number(self, main_with_synthetic_exp_dir):
        result = main_with_synthetic_exp_dir.path_set.shock_paths(
            prefix="Shock", ext="exp", max_depth=2,
        )
        shock_nums = [int(row[0]) for row in result]
        assert shock_nums == sorted(shock_nums), (
            f"shock_paths output must be ascending by shock number; got {shock_nums}"
        )

    def test_empty_directory_returns_empty_list(self, main_window, tmp_path):
        empty_exp = tmp_path / "empty"
        empty_exp.mkdir()
        main_window.path["exp_main"] = empty_exp

        result = main_window.path_set.shock_paths(prefix="Shock", ext="exp")

        assert result == [] or len(result) == 0, (
            f"empty exp dir should yield empty result; got {result}"
        )


class TestAddSeriesToTable:
    """Adding a series via the GUI populates ``Series_Viewer.data_table``.

    This exercises the full chain ``series.add_series`` ->
    ``Series_Viewer._add_series_table`` -> ``DataSetsTable._update`` ->
    ``series.thermo_mix(shock=...)``, which depends on attribute access
    rather than dict subscripting on ``ExperimentalShock``.
    """

    @pytest.fixture
    def main_with_exp_main(self, main_with_loaded_mech, repo_root):
        main = main_with_loaded_mech
        main.path["exp_main"] = repo_root / "example" / "experiment"

        return main

    def test_add_series_populates_data_table(self, main_with_exp_main):
        main = main_with_exp_main
        main.series.add_series()
        main.series_viewer._add_series_table(None)

        assert len(main.series_viewer.data_table) == 1, (
            f"expected one data_table entry after add_series + "
            f"_add_series_table; got {len(main.series_viewer.data_table)}"
        )

    def test_add_series_records_one_shock_row(self, main_with_exp_main):
        """Bundled example/experiment has Shock1.exp; the table should
        have one row keyed on shock number 1."""
        main = main_with_exp_main
        main.series.add_series()
        main.series_viewer._add_series_table(None)

        table = main.series_viewer.data_table[0]
        assert table.all_shocks == [1], (
            f"expected one shock at number 1; got {table.all_shocks}"
        )


class TestPathShockStep:
    """``Path.shock`` resolves a target shock index from a list of
    available shock numbers and the ``ShockSelectionState`` step.

    Regression: ``np.where(prev == shock_num)`` errors with "Calling
    nonzero on 0d arrays" when ``shock_num`` is a Python list because
    ``int == list`` returns a scalar bool, not an array.
    """

    def test_step_forward_with_list_input(self, main_window):
        main_window.shock_selection.previous = 1
        main_window.shock_selection.current = 2
        idx = main_window.path_set.shock([1, 2, 3])
        assert idx == 1, f"step from shock 1 to shock 2 should land at idx 1; got {idx}"

    def test_step_backward_with_list_input(self, main_window):
        main_window.shock_selection.previous = 3
        main_window.shock_selection.current = 2
        idx = main_window.path_set.shock([1, 2, 3])
        assert idx == 1, f"step from shock 3 to shock 2 should land at idx 1; got {idx}"

    def test_previous_not_in_list_falls_back_to_nearest(self, main_window):
        """When previous shock isn't available, snap to nearest current."""
        main_window.shock_selection.previous = 5
        main_window.shock_selection.current = 4
        idx = main_window.path_set.shock([1, 2, 3])
        assert idx == 2, f"nearest fallback for shock 4 in [1,2,3] should be idx 2; got {idx}"

    def test_jump_of_more_than_one_uses_nearest(self, main_window):
        """When |current - previous| > 1, the find_nearest branch runs."""
        main_window.shock_selection.previous = 1
        main_window.shock_selection.current = 5
        idx = main_window.path_set.shock([1, 2, 3])
        assert idx == 2, f"jump to shock 5 in [1,2,3] should clamp to idx 2; got {idx}"


class TestDirectoryBoxSizing:
    """The Files-tab text boxes hold a fixed number of text lines,
    derived from font metrics so display scaling carries through."""

    @pytest.mark.parametrize("box_name, n_lines", [
        ("exp_main_box", 4),
        ("mech_main_box", 4),
        ("sim_main_box", 4),
        ("path_file_box", 5),
    ])
    def test_box_fits_its_line_count(self, main_window, box_name, n_lines):
        box = getattr(main_window, box_name)
        assert box.minimumHeight() == box.maximumHeight(), (
            f"{box_name} height must be fixed"
        )
        chrome = 2 * (box.frameWidth() + int(box.document().documentMargin()))
        content = box.maximumHeight() - chrome
        line = box.fontMetrics().lineSpacing()
        assert n_lines * line <= content < (n_lines + 1) * line, (
            f"{box_name}: content height {content}px should fit exactly "
            f"{n_lines} lines of {line}px"
        )


class TestOptOverlay:
    """The start/best/current sim overlay on the signal plot: stash,
    toggle, time offset, and per-shock selection."""

    def _traces(self, num, t_offset=0.0):
        t = np.linspace(0.0, 1e-4, 20)
        traces = {
            "start": [{"num": num, "t": t, "obs": np.full(20, 1.0),
                       "t_offset": t_offset}],
            "best": [{"num": num, "t": t, "obs": np.full(20, 2.0),
                      "t_offset": t_offset}],
            "current": [{"num": num, "t": t, "obs": np.full(20, 3.0),
                         "t_offset": t_offset}],
        }

        return traces

    def test_overlay_hidden_until_enabled(self, main_window):
        sig = main_window.plot.signal
        num = main_window.display_shock.num
        sig.reset_opt_overlay()
        sig.ingest_opt_traces(self._traces(num))
        sig.refresh_opt_overlay()
        for line in sig._opt_overlay_lines().values():
            assert not line.get_visible()

    def test_enabled_overlay_shows_displayed_shock(self, main_window):
        sig = main_window.plot.signal
        num = main_window.display_shock.num
        sig.reset_opt_overlay()
        sig.ingest_opt_traces(self._traces(num))
        sig.set_opt_overlay_visible(True)
        lines = sig._opt_overlay_lines()
        for line in lines.values():
            assert line.get_visible()
        np.testing.assert_allclose(lines["best"].get_ydata(), 2.0)
        np.testing.assert_allclose(lines["current"].get_ydata(), 3.0)
        assert not sig.ax[1].item["sim_data"].get_visible(), (
            "live sim trace must hide while the overlay draws"
        )

    def test_disable_hides_but_keeps_stash(self, main_window):
        sig = main_window.plot.signal
        num = main_window.display_shock.num
        sig.reset_opt_overlay()
        sig.ingest_opt_traces(self._traces(num))
        sig.set_opt_overlay_visible(True)
        sig.set_opt_overlay_visible(False)
        for line in sig._opt_overlay_lines().values():
            assert not line.get_visible()
        assert sig.ax[1].item["sim_data"].get_visible(), (
            "live sim trace must return when the overlay is disabled"
        )
        assert int(num) in sig._opt_overlay, "stash must survive disabling"

    def test_blank_for_shock_without_stash(self, main_window):
        sig = main_window.plot.signal
        num = main_window.display_shock.num
        sig.reset_opt_overlay()
        sig.ingest_opt_traces(self._traces(int(num) + 1000))
        sig.set_opt_overlay_visible(True)
        for line in sig._opt_overlay_lines().values():
            assert not line.get_visible()
        assert sig.ax[1].item["sim_data"].get_visible(), (
            "live sim trace must stay visible when the overlay has no "
            "trace for the displayed shock"
        )

    def test_time_offset_shifts_xdata(self, main_window):
        sig = main_window.plot.signal
        num = main_window.display_shock.num
        t = np.linspace(0.0, 1e-4, 20)
        sig.reset_opt_overlay()
        sig.ingest_opt_traces(self._traces(num, t_offset=5e-6))
        sig.set_opt_overlay_visible(True)
        np.testing.assert_allclose(
            sig._opt_overlay_lines()["start"].get_xdata(), t + 5e-6,
        )

    def test_best_delta_merges_without_dropping_start(self, main_window):
        """A 'best'-only delta merges into the stash; kinds absent from
        the delta keep their cached start/current entries."""
        sig = main_window.plot.signal
        num = main_window.display_shock.num
        sig.reset_opt_overlay()
        sig.ingest_opt_traces(self._traces(num))
        t = np.linspace(0.0, 1e-4, 20)
        sig.ingest_opt_traces(
            {"best": [{"num": num, "t": t, "obs": np.full(20, 9.0),
                       "t_offset": 0.0}]}
        )
        slot = sig._opt_overlay[int(num)]
        assert set(slot) == {"start", "best", "current"}
        np.testing.assert_allclose(slot["best"][1], 9.0)
        np.testing.assert_allclose(slot["start"][1], 1.0)


class TestSimLegend:
    """The signal plot shows a legend only while multiple simulations
    are drawn (the optimization overlay); a lone live trace has none."""

    def _traces(self, num):
        t = np.linspace(0.0, 1e-4, 20)
        traces = {
            kind: [{"num": num, "t": t, "obs": np.full(20, y),
                    "t_offset": 0.0}]
            for kind, y in (("start", 1.0), ("best", 2.0), ("current", 3.0))
        }

        return traces

    def test_legend_appears_with_overlay(self, main_window):
        sig = main_window.plot.signal
        num = main_window.display_shock.num
        sig.reset_opt_overlay()
        sig.ingest_opt_traces(self._traces(num))
        sig.set_opt_overlay_visible(True)
        legend = sig.ax[1].get_legend()
        assert legend is not None, "overlay must bring up the legend"
        labels = [t.get_text() for t in legend.get_texts()]
        assert labels == ["Start", "Best", "Current"]

    def test_legend_hides_without_overlay(self, main_window):
        sig = main_window.plot.signal
        num = main_window.display_shock.num
        sig.reset_opt_overlay()
        sig.ingest_opt_traces(self._traces(num))
        sig.set_opt_overlay_visible(True)
        sig.set_opt_overlay_visible(False)
        assert sig.ax[1].get_legend() is None, (
            "a lone live trace must not carry a legend"
        )


class TestViewZoomPersistence:
    """Zoom/pan on an optimization view survives switching views; Home
    returns to autoscale."""

    def _seed_history(self, main_window):
        diag = {
            "T": [1500.0], "P": [8000.0],
            "loss_raw": [0.02], "loss_raw_start": [0.05],
            "sigma_bar": [0.01], "z": [0.1], "irls_weights": [1.0],
            "coverage": [1.0], "user": [1.0], "trim_weights": [1.0],
            "t_unc": [2e-7], "t_unc_star": [2e-7],
            "t_unc_mode": "independent", "t_unc_bounds": [-1e-6, 1e-6],
            "t_offset_base": [0.0],
        }
        update = {
            "i": 1, "type": "local", "obj_fcn": 2.0, "s": [0.0],
            "stat_plot": {"per_shock": diag, "shocks2run": [{"num": 1}]},
        }
        main_window.plot.opt.record_iteration(update, is_best=True)

    def test_zoom_survives_view_switch(self, main_window):
        opt = main_window.plot.opt
        self._seed_history(main_window)
        opt._view_changed("Objective Trace")
        ax = opt._active_ax()
        ax.set_xlim(0.25, 0.75)
        ax.set_ylim(1.5, 2.5)

        opt._view_changed("Misfit Map")
        opt._view_changed("Objective Trace")
        ax = opt._active_ax()
        assert ax.get_xlim() == pytest.approx((0.25, 0.75)), (
            "user xlim must survive the round trip"
        )
        assert ax.get_ylim() == pytest.approx((1.5, 2.5))

    def test_home_restores_autoscale_and_reenables_it(self, main_window):
        opt = main_window.plot.opt
        self._seed_history(main_window)
        opt._view_changed("Objective Trace")
        ax = opt._active_ax()
        home_xlim = ax.get_xlim()
        ax.set_xlim(0.25, 0.75)
        opt.toolbar.home()
        assert ax.get_xlim() == pytest.approx(home_xlim), (
            "Home must restore the autoscaled limits"
        )
        state = opt._view_state["objective_trace"]
        assert not state.get("user_x") and not state.get("user_y"), (
            "returning to Home must hand control back to autoscale"
        )

    def test_home_mid_run_tracks_live_autoscale(self, main_window):
        """Home pressed after the autoscale has moved must restore the
        CURRENT autoscale, not the view-switch snapshot — a stale
        restore reads as a user zoom and freezes the trace for the
        rest of the run (seen at the global -> local transition)."""
        opt = main_window.plot.opt
        opt.attach_run_context(None)
        opt._view_changed("Objective Trace")
        for j in range(5):
            self._seed_history(main_window)
            opt.refresh()
        ax = opt._active_ax()
        live_xlim = ax.get_xlim()
        opt.toolbar.home()
        assert ax.get_xlim() == pytest.approx(live_xlim), (
            f"Home must land on the live autoscale, got {ax.get_xlim()}"
        )
        state = opt._view_state["objective_trace"]
        assert not state.get("user_x") and not state.get("user_y")

        for j in range(15):
            self._seed_history(main_window)
            opt.refresh()
        assert ax.get_xlim()[1] > live_xlim[1], (
            f"limits must keep tracking after Home, stuck at {ax.get_xlim()}"
        )

    def test_home_and_back_after_run_end_show_full_extent(self, main_window):
        """With no refresh cadence left (run over), Home and Back must
        still land on the full-data autoscale, not a stale window."""
        opt = main_window.plot.opt
        opt.attach_run_context(None)
        opt._view_changed("Objective Trace")
        for j in range(5):
            self._seed_history(main_window)
            opt.refresh()
        ax = opt._active_ax()
        # More data arrives, then the run ends without a final refresh
        # while the user is zoomed in.
        ax.set_xlim(0.25, 0.75)
        opt.toolbar.push_current()
        for j in range(15):
            self._seed_history(main_window)

        opt.toolbar.back()
        assert ax.get_xlim()[1] >= 20, (
            f"Back must land on the live full extent, got {ax.get_xlim()}"
        )

        ax.set_xlim(0.25, 0.75)
        opt.toolbar.push_current()
        opt.toolbar.home()
        assert ax.get_xlim()[1] >= 20, (
            f"Home must land on the live full extent, got {ax.get_xlim()}"
        )

    def test_y_zoom_keeps_x_extending_mid_run(self, main_window):
        opt = main_window.plot.opt
        opt.attach_run_context(None)
        opt._view_changed("Objective Trace")
        for j in range(5):
            self._seed_history(main_window)
            opt.refresh()
        ax = opt._active_ax()
        x_before = ax.get_xlim()[1]
        ax.set_ylim(1.5, 2.5)
        for j in range(15):
            self._seed_history(main_window)
            opt.refresh()
        assert ax.get_xlim()[1] > x_before, (
            f"x must keep extending under a y-only zoom, got {ax.get_xlim()}"
        )
        assert ax.get_ylim() == pytest.approx((1.5, 2.5)), (
            "the y zoom must hold across refreshes"
        )

    def test_back_returns_from_zoom_mid_run(self, main_window):
        """Cadence refreshes while zoomed must not wipe the nav stack:
        Back still has the zoom entry to return from."""
        opt = main_window.plot.opt
        opt.attach_run_context(None)
        opt._view_changed("Objective Trace")
        for j in range(5):
            self._seed_history(main_window)
            opt.refresh()
        ax = opt._active_ax()
        ax.set_xlim(0.25, 0.75)
        ax.set_ylim(1.5, 2.5)
        opt.toolbar.push_current()
        for j in range(5):
            self._seed_history(main_window)
            opt.refresh()

        opt.toolbar.back()
        assert ax.get_xlim()[1] > 0.75, (
            f"Back must leave the zoom window, got {ax.get_xlim()}"
        )

    def test_home_clears_y_freeze_mid_run(self, main_window):
        """Home while a y-only zoom is active must return y to the live
        autoscale and keep it tracking afterwards."""
        opt = main_window.plot.opt
        opt.attach_run_context(None)
        opt._view_changed("Objective Trace")
        for j in range(5):
            self._seed_history(main_window)
            opt.refresh()
        ax = opt._active_ax()
        ax.set_ylim(1.5, 2.5)
        for j in range(5):
            self._seed_history(main_window)
            opt.refresh()

        opt.toolbar.home()
        assert ax.get_ylim() != pytest.approx((1.5, 2.5)), (
            "Home must release the y zoom"
        )
        state = opt._view_state["objective_trace"]
        assert not state.get("user_x") and not state.get("user_y")
        assert ax.get_autoscaley_on(), (
            "y autoscale must be re-enabled after Home"
        )
        x_at_home = ax.get_xlim()[1]
        for j in range(5):
            self._seed_history(main_window)
            opt.refresh()
        assert ax.get_xlim()[1] > x_at_home, (
            f"x must keep tracking after Home, got {ax.get_xlim()}"
        )

    def test_unzoomed_view_keeps_autoscaling(self, main_window):
        opt = main_window.plot.opt
        self._seed_history(main_window)
        opt._view_changed("Objective Trace")
        opt._view_changed("Misfit Map")
        opt._view_changed("Objective Trace")
        state = opt._view_state["objective_trace"]
        assert not state.get("user_x") and not state.get("user_y"), (
            "switching alone must not freeze the view's limits"
        )


class TestScreeningSort:
    """Mech-tree sort dropdown + screening-result plumbing."""

    def _fake_result(self, n):
        leverage = [float(n - i) for i in range(n)]
        leverage[2] = float(n + 5)  # rxn 2 leads
        result = ScreeningResult(
            rxn_indices=list(range(n)),
            importance=[1.0] * n,
            importance_max=[1.0] * n,
            leverage=leverage,
            suggested=[False] * n,
            spectral_gap_rank=None,
            loss_start=1.0,
            singular_values=[1.0],
            effective_rank=1.0,
            skipped_shocks=[],
            shock_nums=[1],
            slopes=[[1.0] * n],
            footprints=[[1.0] * n],
        )

        return result

    def test_dropdown_defaults_and_options(self, main_window):
        box = main_window.mech_tree_sort_box
        labels = [box.itemText(i) for i in range(box.count())]
        assert labels == ["Reaction #", "Leverage", "Importance"]
        assert box.currentText() == "Reaction #"

    def test_score_sort_reorders_and_number_restores(
        self, main_with_loaded_mech,
    ):
        main = main_with_loaded_mech
        tree = main.tree
        n = main.mech.gas.n_reactions
        main.screening_result = self._fake_result(n)

        main.mech_tree_sort_box.setCurrentText("Leverage")
        top = tree.proxy_model.index(0, 0)
        item = tree.model.itemFromIndex(tree.proxy_model.mapToSource(top))
        assert item.info["rxnNum"] == 2, (
            "highest-leverage reaction must lead the sorted tree"
        )

        main.mech_tree_sort_box.setCurrentText("Reaction #")
        top = tree.proxy_model.index(0, 0)
        item = tree.model.itemFromIndex(tree.proxy_model.mapToSource(top))
        assert item.info["rxnNum"] == 0

    def test_score_sort_without_result_schedules_then_reverts(
        self, main_with_loaded_mech,
    ):
        """Selecting a score sort with no result keeps the choice and
        schedules a background run; with no experiment data the launch
        reverts the combo."""
        main = main_with_loaded_mech
        main.screening_result = None
        main.mech_tree_sort_box.setCurrentText("Leverage")
        assert main.mech_tree_sort_box.currentText() == "Leverage"
        assert main.tree._screen_timer.isActive()

        main.tree._screen_timer.stop()
        main.tree._launch_screening()
        assert main.mech_tree_sort_box.currentText() == "Reaction #", (
            "no-data launch must revert the score sort"
        )

    def test_all_zero_score_sort_option_greys_out(
        self, main_with_loaded_mech,
    ):
        main = main_with_loaded_mech
        n = main.mech.gas.n_reactions
        result = self._fake_result(n)
        zeroed = result.model_copy(update={"leverage": [0.0] * n})
        main.tree._sync_sort_options(zeroed)
        box = main.mech_tree_sort_box
        assert not box.model().item(box.findText("Leverage")).isEnabled()
        assert box.model().item(box.findText("Importance")).isEnabled()

        main.tree._sync_sort_options(result)
        assert box.model().item(box.findText("Leverage")).isEnabled()

    def test_stale_mark_debounces_a_background_run(
        self, main_with_loaded_mech,
    ):
        main = main_with_loaded_mech
        main.tree._screen_timer.stop()
        main.tree._screening_dirty = False
        main.tree._mark_screening_stale()
        assert main.tree._screening_dirty
        assert main.tree._screen_timer.isActive()
        main.tree._screen_timer.stop()


class TestScreeningRunnerHelpers:
    def test_gather_falls_back_to_display_shock(self, main_with_loaded_mech):
        main = main_with_loaded_mech
        shocks = gather_screening_shocks(main)
        if main.display_shock.exp_data.size == 0:
            assert shocks == []
        else:
            assert shocks == [main.display_shock]

    def test_screening_summary_is_one_line(self, main_with_loaded_mech):
        main = main_with_loaded_mech
        n = main.mech.gas.n_reactions
        result = TestScreeningSort()._fake_result(n)
        textout = format_screening_summary(result)
        assert "\n" not in textout, "the summary must be a single line"
        assert "effective rank 1.0" in textout
        assert "no clear kink" in textout


class TestOptimizationViewSwitching:
    """The Optimization-tab view selector routes every entry through
    the ViewRouter; the base blit machinery is inert (no legacy axes)."""

    def _select_optimization_tab(self, main_window):
        tabs = main_window.plot_tab_widget
        for i in range(tabs.count()):
            if tabs.tabText(i) == "Optimization":
                tabs.setCurrentIndex(i)

                return

    def test_boot_default_is_objective_trace(self, main_window):
        box = main_window.opt_view_box
        assert box.currentText() == "Objective Trace"
        assert main_window.plot.opt.router.active_name == "objective_trace"

    def test_blit_machinery_is_inert(self, main_window):
        """The base draw path must never touch router-owned axes."""
        opt = main_window.plot.opt
        assert opt.ax == [], "the optimization plot owns no blit axes"
        opt._draw_event()  # no-op by contract

    def test_full_dropdown_cycle(self, main_window):
        self._select_optimization_tab(main_window)
        box = main_window.opt_view_box
        labels = [box.itemText(i) for i in range(box.count())]
        assert labels == list(VIEW_LABELS), (
            f"view dropdown must mirror VIEW_LABELS, got {labels}"
        )
        contextual = {"arrhenius", "arrhenius_ratio", "band_utilization"}
        for label in labels + [labels[0]]:
            box.setCurrentText(label)
            key = VIEW_LABELS[label]
            active = main_window.plot.opt.router.active_name
            if key in contextual:
                # These views exist only once a run context arrives.
                assert active is None
            else:
                assert active == key

    def test_attach_run_context_populates_selectors_and_renders(
        self, main_window,
    ):
        """The run-start path: reaction selectors fill (ratio box all
        checked), the pressure box takes the campaign reference, and the
        new views render from a recorded event."""
        self._select_optimization_tab(main_window)
        opt = main_window.plot.opt
        context = ViewContext(
            param_labels=["R1 @ 1500 K", "R1 @ 1800 K"],
            param_rxn=[0, 0],
            lower_bounds=[-0.7, -0.7],
            upper_bounds=[0.7, 0.7],
            rxn_indices=[0],
            rxn_equations=["A <=> B"],
            rxn_is_pressure_dependent=[False],
            rxn_band_halfwidth=[0.7],
            T_grid=np.linspace(1400.0, 2000.0, 5).tolist(),
            P_grid=[4000.0, 8000.0, 16000.0],
            P_reference=8000.0,
            ln_k_initial=[[[10.0] * 5] * 3],
        )
        opt.attach_run_context(context)
        assert main_window.opt_view_detail_box.count() == 1
        assert opt.ratio_box.count() == 1
        assert opt.ratio_box.itemChecked(0) is True, (
            "ratio box must start with every reaction checked"
        )

        diag = {
            "T": [1600.0], "P": [8000.0],
            "loss_raw": [0.02], "loss_raw_start": [0.05],
            "sigma_bar": [0.01], "z": [0.1],
            "irls_weights": [1.0], "coverage": [1.0],
            "user": [1.0], "trim_weights": [1.0],
            "t_unc": [2e-7], "t_unc_star": [3e-7],
            "t_unc_bounds": [-1e-6, 1e-6], "t_offset_base": [5e-7],
        }
        update = {
            "i": 1, "type": "local", "obj_fcn": 1.0, "s": [0.0, 0.1],
            "stat_plot": {"per_shock": diag, "shocks2run": [{"num": 1}]},
            "views": {"ln_k": [[[10.1] * 5] * 3]},
        }
        opt.record_iteration(update, is_best=True)

        box = main_window.opt_view_box
        for label in ("Time Offsets", "Improvement (start vs now)",
                      "Arrhenius Ratios (k / k₀)"):
            box.setCurrentText(label)
            assert len(opt.fig.axes) >= 1, f"{label} drew no axes"
        assert not opt.ratio_box.isHidden(), (
            "ratio box must show while the ratio view is active"
        )

        # Unchecking through the model drives the selection signal.
        opt.ratio_box.model().item(0, 0).setCheckState(QtCore.Qt.Unchecked)
        texts = " ".join(
            t.get_text() for ax in opt.fig.axes for t in ax.texts
        )
        assert "no reactions selected" in texts

        box.setCurrentText("Objective Trace")
        assert opt.ratio_box.isHidden(), (
            "ratio box must hide when leaving the ratio view"
        )
