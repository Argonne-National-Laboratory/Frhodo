"""End-to-end overlay navigation: a real (tiny) optimization run through
the orchestrator with two shocks, then shock navigation, asserting each
shock's overlay lines draw at its own stashed solved offsets.
"""
import shutil

import numpy as np
import pytest
from qtpy import QtCore



pytestmark = pytest.mark.gui


@pytest.fixture
def main_two_shocks(main_with_loaded_mech, example_dir, tmp_path):
    """Main with a 2-shock series (example shock duplicated)."""
    main = main_with_loaded_mech
    main.convert_units.mech = main.mech
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    src = example_dir / "experiment"
    for num in (1, 2):
        shutil.copy(src / "shock1.exp", exp_dir / f"Shock{num}.exp")
        shutil.copy(src / "shock1.rho", exp_dir / f"Shock{num}.rho")
        shutil.copy(src / "shock1raw1.sig", exp_dir / f"Shock{num}raw1.sig")

    main.path["exp_main"] = exp_dir
    main.load_full_series_box.blockSignals(True)
    main.load_full_series_box.setChecked(True)
    main.load_full_series_box.blockSignals(False)
    main.load_state.load_full_series = True
    main.directory.update_icons()
    main.series.add_series()
    main.series.set("exp_data")
    main.series_viewer._add_series_table(None)
    for shock in main.series.shock[main.series.idx]:
        shock.include = True
    assert len(main.series.shock[main.series.idx]) == 2

    # The run saves PreOpt/Opt mechanisms into mech_main; redirect it
    # after the directory checks so test artifacts stay out of the
    # example library.
    mech_out = tmp_path / "mech_out"
    mech_out.mkdir()
    main.path["mech_main"] = mech_out

    return main


def _run_tiny_optimization(main, qapp, max_eval=8, multiprocessing=False):
    shipped = []
    sig = main.plot.signal
    orig_ingest = sig.ingest_opt_traces

    def recording_ingest(sim_traces):
        if sim_traces:
            shipped.append({k: [dict(tr) for tr in v]
                            for k, v in sim_traces.items() if v is not None})
        orig_ingest(sim_traces)

    sig.ingest_opt_traces = recording_ingest
    main.multiprocessing_box.setChecked(multiprocessing)
    main.global_opt_enable_box.setChecked(False)
    main.local_opt_enable_box.setChecked(True)
    widgets = main.optimization_settings.widgets
    widgets["local"]["stop_criteria_val"].setValue(max_eval)
    main.time_uncertainty.value = 5e-7

    main.mech.rate_bnds[0]["value"] = 3.0
    main.mech.rate_bnds[0]["type"] = "F"
    main.optimizables.set_reaction_optimizable(0, True)
    bnds_key = next(iter(main.mech.coeffs_bnds[0]))
    coef_name = next(iter(main.mech.coeffs_bnds[0][bnds_key]))
    main.optimizables.set_coefficient_optimizable(0, bnds_key, coef_name, True)

    print("INVALID >>>", main.directory.invalid, {k: main.path.get(k) for k in ("exp_main", "mech_main", "sim_main")})
    main.optimize.start_threads()
    deadline = QtCore.QDeadlineTimer(180_000)
    while main.run_control.optimize_running and not deadline.hasExpired():
        qapp.processEvents()
        QtCore.QThread.msleep(20)
    assert not main.run_control.optimize_running, "run did not finish"
    sig.ingest_opt_traces = orig_ingest

    return shipped


class TestOverlayNavigation:
    @pytest.mark.parametrize("multiprocessing", [False, True])
    def test_each_shock_draws_its_own_solved_offsets(
        self, main_two_shocks, qapp, multiprocessing,
    ):
        main = main_two_shocks
        sig = main.plot.signal
        shipped = _run_tiny_optimization(
            main, qapp, multiprocessing=multiprocessing,
        )

        stash = sig._opt_overlay

        # The stash must hold exactly what the engine shipped: start from
        # the first delta, best from the last best delta — nothing may
        # rewrite them after the run.
        first_start = next(d["start"] for d in shipped if "start" in d)
        last_best = [d["best"] for d in shipped if "best" in d][-1]
        for tr in first_start:
            assert stash[int(tr["num"])]["start"][2] == tr["t_offset"], (
                f"shock {tr['num']} start offset drifted from the engine's"
            )
        for tr in last_best:
            assert stash[int(tr["num"])]["best"][2] == tr["t_offset"], (
                f"shock {tr['num']} best offset drifted from the engine's"
            )
        assert set(stash) == {1, 2}, f"stash keys: {sorted(stash)}"
        for num in (1, 2):
            assert set(stash[num]) == {"start", "best", "current"}, (
                f"shock {num} kinds: {sorted(stash[num])}"
            )

        main.show_opt_overlay_box.setChecked(True)
        for num in (2, 1, 2):
            main.shock_choice_box.setValue(num)
            qapp.processEvents()
            lines = sig._opt_overlay_lines()
            for kind, line in lines.items():
                t, obs, t_offset = stash[num][kind]
                assert line.get_visible(), (
                    f"shock {num} {kind} line hidden after navigation"
                )
                np.testing.assert_allclose(
                    line.get_xdata(), t + t_offset,
                    err_msg=(
                        f"shock {num} {kind} drawn at the wrong offset "
                        f"after navigation"
                    ),
                )
