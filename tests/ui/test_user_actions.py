"""User-action regression tests for GUI widgets.

Each test boots ``Main`` (or a fixture variant), triggers a single user
action programmatically, and asserts a minimum observable outcome. The
goal is a catch-net for regressions in interaction sequences that have
historically been silent until a user clicked the affected control.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest


pytestmark = pytest.mark.gui


class TestSimDataDragSplit:
    """Dragging the sim line: motion stays cheap, release commits the
    heavy state propagation. Mirrors the contract introduced when
    ``_copy_expanded_tab_rates`` and full ``update_sim`` moved out of
    ``draggable_update_fcn`` and into ``draggable_release_fcn``.
    """

    @pytest.fixture
    def main_with_sim_stub(self, main_with_loaded_mech):
        """Inject a minimal SIM so update_sim can run without a real shock."""
        main = main_with_loaded_mech
        main.SIM.independent_var = np.array([0.0, 1e-5, 2e-5])
        main.SIM.observable = np.array([0.0, 0.5, 1.0])

        sim_data = main.plot.signal.ax[1].item["sim_data"]
        sim_data.raw_data = np.array(
            [main.SIM.independent_var, main.SIM.observable]
        ).T

        return main

    def test_motion_updates_time_offset_and_skips_heavy_work(
        self, main_with_sim_stub
    ):
        main = main_with_sim_stub
        sim_data = main.plot.signal.ax[1].item["sim_data"]
        main.tree._copy_expanded_tab_rates = MagicMock()
        offset_before = main.display_shock.time_offset

        # x is in matplotlib axis units (SI seconds, since sim_data plots
        # ``t + time_offset`` directly in SI). Picking ``new[0] = 0.5e-6``
        # rounds to 0.5 μs after the handler's divide-by-t_unit_conv.
        t0 = main.SIM.independent_var
        new = t0 + 0.5e-6
        x = {"0": t0, "press": 0.0, "new": new, "press_new": new[0]}
        y = {
            "0": np.zeros_like(t0),
            "press": 0.0,
            "new": np.zeros_like(t0),
            "press_new": 0.0,
        }

        main.plot.signal.draggable_update_fcn(sim_data, x, y)

        assert main.tree._copy_expanded_tab_rates.call_count == 0, (
            "_copy_expanded_tab_rates must NOT run on every motion event"
        )
        assert main.display_shock.time_offset != offset_before, (
            f"time_offset should change during sim_data drag; "
            f"stayed at {offset_before}"
        )

    def test_release_runs_deferred_heavy_work(self, main_with_sim_stub):
        main = main_with_sim_stub
        sim_data = main.plot.signal.ax[1].item["sim_data"]
        sim_data.draggable.nearest_index = 0
        main.tree._copy_expanded_tab_rates = MagicMock()

        main.plot.signal.draggable_release_fcn(sim_data)

        assert main.tree._copy_expanded_tab_rates.call_count == 1, (
            "release must commit the deferred clipboard rate copy"
        )


class TestPlotScaleSwitch:
    """Switching axis scale + format_coord with integer args is the
    matplotlib mouse-pixel pipeline. Bisymlog used to crash on int
    input because ``np.zeros_like(y)`` inherited the int dtype.
    """

    def test_bisymlog_format_coord_with_int_args(self, main_window):
        import matplotlib.pyplot as plt

        ax = main_window.plot.signal.ax[1]
        ax.plot([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0])
        ax.set_xscale("bisymlog", C=1.0)
        ax.set_yscale("bisymlog", C=1.0)
        ax.figure.canvas.draw()

        # int args mimic the path used by mouseMoveEvent → format_coord.
        msg = ax.format_coord(0, 0)
        assert isinstance(msg, str) and "x" in msg, (
            f"format_coord(0, 0) on bisymlog axes should return a coord "
            f"string; got {msg!r}"
        )
        plt.close(ax.figure)


class TestMechTreeCheckboxToggle:
    """Toggling reaction-optimizable checkboxes flows into the
    ``OptimizableSetBuilder`` on Main."""

    def test_toggle_reaction_optimizable_updates_builder(
        self, main_with_loaded_mech
    ):
        main = main_with_loaded_mech
        builder = main.optimizables

        builder.set_reaction_optimizable(0, True)
        try:
            opt_set = builder.build(main.mech)
            assert 0 in opt_set.optimizable_reactions, (
                "rxn 0 should be marked optimizable after builder toggle; "
                f"got optimizable_reactions={opt_set.optimizable_reactions}"
            )
        finally:
            builder.reset()

    def test_toggle_coefficient_optimizable_updates_builder(
        self, main_with_loaded_mech
    ):
        main = main_with_loaded_mech
        builder = main.optimizables
        bnds_key = next(iter(main.mech.coeffs_bnds[0]))
        coef_name = next(iter(main.mech.coeffs_bnds[0][bnds_key]))

        builder.set_reaction_optimizable(0, True)
        builder.set_coefficient_optimizable(0, bnds_key, coef_name, True)
        try:
            opt_set = builder.build(main.mech)
            assert any(
                c.rxn_idx == 0 and c.coef_name == coef_name
                for c in opt_set.coefficients
            ), (
                f"coef ({coef_name}) on rxn 0 not present in built "
                f"OptimizableSet; got {[(c.rxn_idx, c.coef_name) for c in opt_set.coefficients]}"
            )
        finally:
            builder.reset()


class TestThermoMixShockArg:
    """``series.thermo_mix(shock=<ExperimentalShock>)`` is called from
    Series_Viewer when populating the data table. Pre-fix the function
    used ``len(shock)`` as a no-arg sentinel, which crashes on a
    pydantic model. This test exercises the path with an explicit shock.
    """

    @pytest.fixture
    def main_with_series(self, main_with_loaded_mech, repo_root):
        main = main_with_loaded_mech
        main.path["exp_main"] = repo_root / "example" / "experiment"
        main.series.add_series()
        return main

    def test_thermo_mix_with_shock_arg(self, main_with_series):
        main = main_with_series
        shock = main.series.shock[main.series.idx][main.series.shock_idx]

        main.series.thermo_mix(shock=shock)

        assert isinstance(shock.thermo_mix, dict), (
            f"thermo_mix should populate shock.thermo_mix dict; "
            f"got {type(shock.thermo_mix).__name__}"
        )

    def test_thermo_mix_default_uses_display_shock(self, main_with_series):
        """No-arg call should target ``parent.display_shock`` -- this is
        the path the spinbox-driven ``update_user_settings`` uses."""
        main = main_with_series

        main.series.thermo_mix()

        assert isinstance(main.display_shock.thermo_mix, dict)


class TestSaveChemkinRoundTrip:
    """Saving a mech to Chemkin format and reading it back via the
    loader must yield the same reaction count. Catches Cantera-API
    regressions in the export path that escape mech-load tests."""

    def test_chemkin_export_reloads_with_same_reaction_count(
        self, main_with_loaded_mech, tmp_path
    ):
        main = main_with_loaded_mech
        n_rxn_before = main.mech.gas.n_reactions

        out = tmp_path / "roundtrip.mech"
        main.save.chemkin_format(main.mech.gas, out)
        assert out.exists(), f"Chemkin export did not write file: {out}"

        from frhodo.simulation.mechanism.mechanism_loader import MechanismLoader
        reloaded = MechanismLoader().load({
            "mech": out,
            "thermo": None,
            "Cantera_Mech": tmp_path / "roundtrip.yaml",
        })
        assert reloaded.gas.n_reactions == n_rxn_before, (
            f"reaction count drifted on Chemkin round-trip: "
            f"{n_rxn_before} -> {reloaded.gas.n_reactions}"
        )
