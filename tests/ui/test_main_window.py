"""Smoke tests for ``Main`` window construction.

Booting ``Main`` exercises every import + widget construction path the
GUI uses. This is the catch-net for issues like the matplotlib
``set_xdata(scalar)`` break or qtpy import errors that would otherwise
only surface when a user clicks a button.
"""
import pytest

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
        from frhodo.simulation.mechanism.mech_fcns import ChemicalMechanism

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
