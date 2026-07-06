"""Optimization-outcome views: event adapter and router logic."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from frhodo.gui.plots.optimization_plot import VIEW_LABELS
from frhodo.gui.views.events import IterationEvent, ViewContext
from frhodo.gui.views.view_router import ViewRouter



P_GRID = [4000.0, 8000.0, 16000.0]


def _context(n_params=4, n_rxns=2, pressure_dependent=None):
    per_rxn = n_params // n_rxns
    if pressure_dependent is None:
        pressure_dependent = [False] * n_rxns
    context = ViewContext(
        param_labels=[f"R{r + 1} @ {1500 + 100 * j:.0f} K"
                      for r in range(n_rxns) for j in range(per_rxn)],
        param_rxn=[r for r in range(n_rxns) for _ in range(per_rxn)],
        lower_bounds=[-0.693] * n_params,
        upper_bounds=[0.693] * n_params,
        rxn_indices=list(range(n_rxns)),
        rxn_equations=[f"A{r} <=> B{r}" for r in range(n_rxns)],
        rxn_is_pressure_dependent=pressure_dependent,
        rxn_band_halfwidth=[0.693] * n_rxns,
        T_grid=np.linspace(1400.0, 2000.0, 5).tolist(),
        P_grid=P_GRID,
        P_reference=8000.0,
        ln_k_initial=[
            [[10.0 + r + p] * 5 for p in range(len(P_GRID))]
            for r in range(n_rxns)
        ],
    )

    return context


def _update(i=1, obj=2.0, stage="global", n=3, with_views=True):
    diag = {
        "T": [1500.0, 1700.0, 1900.0][:n],
        "P": [8000.0, 4000.0, 6000.0][:n],
        "loss_raw": [0.02, 0.04, 0.03][:n],
        "loss_raw_start": [0.05, 0.06, 0.07][:n],
        "sigma_bar": [0.01, 0.02, 0.01][:n],
        "z": [0.1, -0.2, 1.5][:n],
        "irls_weights": [1.0, 1.0, 0.6][:n],
        "coverage": [1.1, 0.9, 1.0][:n],
        "user": [1.0, 1.0, 1.0][:n],
        "trim_weights": [1.0, 0.5, 1.0][:n],
        "t_unc": [2e-7, 5e-7, 9.9e-7][:n],
        "t_unc_star": [3e-7, 4e-7, 9.9e-7][:n],
        "t_unc_mode": "parametric",
        "t_unc_bounds": [-1e-6, 1e-6],
        "t_offset_base": [5e-7, 5e-7, 5e-7][:n],
    }
    update = {
        "i": i,
        "type": stage,
        "obj_fcn": obj,
        "s": [0.0, 0.1, -0.69299, 0.2],
        "stat_plot": {"per_shock": diag, "shocks2run": [{"num": 7},
                                                        {"num": 8},
                                                        {"num": 9}][:n]},
    }
    if with_views:
        update["views"] = {"ln_k": [
            [[10.1 + p] * 5 for p in range(len(P_GRID))],
            [[11.2 + p] * 5 for p in range(len(P_GRID))],
        ]}

    return update


class TestIterationEventAdapter:
    def test_builds_typed_event_from_update(self):
        event = IterationEvent.from_update(_update(), is_best=True)
        assert event is not None
        assert event.shock_num == [7, 8, 9]
        assert event.loss_obj == pytest.approx([0.02, 0.04, 0.03])
        assert event.stage == "global"
        assert event.is_best

    def test_update_without_per_shock_diag_returns_none(self):
        update = _update()
        update["stat_plot"] = {}
        assert IterationEvent.from_update(update, is_best=False) is None

    def test_missing_views_payload_gives_empty_ln_k(self):
        event = IterationEvent.from_update(_update(with_views=False),
                                           is_best=False)
        assert event.ln_k == []

    def test_time_offset_and_weight_fields_parsed(self):
        event = IterationEvent.from_update(_update(), is_best=False)
        assert event.t_unc == pytest.approx([2e-7, 5e-7, 9.9e-7])
        assert event.t_unc_star == pytest.approx([3e-7, 4e-7, 9.9e-7])
        assert event.t_unc_bounds == pytest.approx([-1e-6, 1e-6])
        assert event.t_offset_base == pytest.approx([5e-7, 5e-7, 5e-7])
        assert event.t_unc_mode == "parametric"
        assert event.trim_weights == pytest.approx([1.0, 0.5, 1.0])
        assert event.user_weights == pytest.approx([1.0, 1.0, 1.0])

    def test_missing_offset_diag_defaults_benign(self):
        update = _update()
        for key in ("t_unc", "t_unc_star", "t_unc_mode", "t_unc_bounds",
                    "t_offset_base", "user", "trim_weights"):
            update["stat_plot"]["per_shock"].pop(key)
        event = IterationEvent.from_update(update, is_best=False)
        assert event.t_unc == [0.0, 0.0, 0.0]
        assert event.t_unc_star == [0.0, 0.0, 0.0]
        assert event.t_unc_mode == "fixed"
        assert event.t_unc_bounds == [0.0, 0.0]
        assert event.t_offset_base == [0.0, 0.0, 0.0]
        assert event.user_weights == [1.0, 1.0, 1.0]
        assert event.trim_weights == [1.0, 1.0, 1.0]


class TestRunHistory:
    def _router(self):
        fig = plt.figure()

        return ViewRouter(fig, _context())

    def test_first_event_is_initial_and_best(self):
        router = self._router()
        router.record(IterationEvent.from_update(_update(obj=5.0),
                                                 is_best=True))
        assert router.history.initial_event is router.history.best_event

    def test_best_so_far_is_monotone(self):
        router = self._router()
        for j, obj in enumerate([5.0, 3.0, 4.0, 2.0]):
            router.record(IterationEvent.from_update(
                _update(i=j, obj=obj), is_best=obj in (3.0, 2.0)))
        assert router.history.best_objectives == [5.0, 3.0, 3.0, 2.0]

    def test_stage_boundary_found(self):
        router = self._router()
        for stage in ["global", "global", "local", "local"]:
            router.record(IterationEvent.from_update(
                _update(stage=stage), is_best=False))
        assert router.history.stage_boundary() == 2

    def test_none_events_ignored(self):
        router = self._router()
        router.record(None)
        assert router.history.initial_event is None


class TestViewRouterRendering:
    @pytest.mark.parametrize("view_key", [
        "misfit", "arrhenius", "objective_trace",
        "arrhenius_ratio", "improvement", "time_offsets",
    ])
    def test_each_view_builds_and_updates(self, view_key):
        fig = plt.figure()
        router = ViewRouter(fig, _context())
        for j in range(3):
            router.record(IterationEvent.from_update(
                _update(i=j, obj=3.0 - j), is_best=True))
        router.set_view(view_key)
        assert router.refresh() is True
        assert len(fig.axes) > 0, f"{view_key} drew no axes"
        plt.close(fig)

    def test_inactive_router_refresh_is_false(self):
        fig = plt.figure()
        router = ViewRouter(fig, _context())
        router.record(IterationEvent.from_update(_update(), is_best=True))
        assert router.refresh() is False
        plt.close(fig)

    def test_switching_back_and_forth_replays_history(self):
        fig = plt.figure()
        router = ViewRouter(fig, _context())
        for j in range(4):
            router.record(IterationEvent.from_update(
                _update(i=j, obj=4.0 - j), is_best=True))
        router.set_view("objective_trace")
        router.set_view("misfit")
        router.set_view("objective_trace")
        line = fig.axes[0].lines[-1]
        assert len(line.get_xdata()) == 4, (
            "trace must replay the full recorded history after switching"
        )
        plt.close(fig)

    def test_arrhenius_reaction_selector_switches_curves(self):
        fig = plt.figure()
        router = ViewRouter(fig, _context())
        router.record(IterationEvent.from_update(_update(), is_best=True))
        router.set_view("arrhenius")
        view = router.views["arrhenius"]
        view.set_reaction(1)
        router.refresh()
        ax = fig.axes[0]
        incumbent = [ln for ln in ax.lines if ln.get_label() == "incumbent"][0]
        # box pressure defaults to P_reference = grid point 1 -> +1 row
        np.testing.assert_allclose(incumbent.get_ydata(), [12.2] * 5)
        plt.close(fig)

    def test_arrhenius_interpolates_between_pressure_grid_points(self):
        """The surface rows are 10.1/11.1/12.1 at 4/8/16 kPa; halfway in
        ln P between the last two rows the curve must read ~11.6."""
        fig = plt.figure()
        router = ViewRouter(fig, _context())
        router.record(IterationEvent.from_update(_update(), is_best=True))
        router.set_view("arrhenius")
        view = router.views["arrhenius"]
        view.set_pressure(np.sqrt(8000.0 * 16000.0))
        router.refresh()
        ax = fig.axes[0]
        incumbent = [ln for ln in ax.lines if ln.get_label() == "incumbent"][0]
        np.testing.assert_allclose(incumbent.get_ydata(), [11.6] * 5,
                                   rtol=1e-9)
        plt.close(fig)

    def test_arrhenius_pressure_clamps_to_grid_edges(self):
        fig = plt.figure()
        router = ViewRouter(fig, _context())
        router.record(IterationEvent.from_update(_update(), is_best=True))
        router.set_view("arrhenius")
        view = router.views["arrhenius"]
        view.set_pressure(1.0)
        router.refresh()
        incumbent = [ln for ln in fig.axes[0].lines
                     if ln.get_label() == "incumbent"][0]
        np.testing.assert_allclose(incumbent.get_ydata(), [10.1] * 5)
        plt.close(fig)

    def test_arrhenius_flags_at_bound_anchor(self):
        """The scaler at -0.693 sits at reaction 0's lower bound; the
        annotation must say so on that reaction's subplot."""
        fig = plt.figure()
        router = ViewRouter(fig, _context())
        router.record(IterationEvent.from_update(_update(), is_best=True))
        router.set_view("arrhenius")
        router.views["arrhenius"].set_reaction(1)
        router.refresh()
        texts = " ".join(t.get_text() for t in fig.axes[0].texts)
        assert "AT BOUND" in texts, (
            f"reaction with a pinned anchor must be flagged, got {texts!r}"
        )
        plt.close(fig)

    def test_arrhenius_marks_pressure_dependent_band_as_advisory(self):
        fig = plt.figure()
        router = ViewRouter(fig, _context(pressure_dependent=[True, False]))
        router.record(IterationEvent.from_update(_update(), is_best=True))
        router.set_view("arrhenius")
        texts = " ".join(t.get_text() for t in fig.axes[0].texts)
        assert "limit-rate anchors" in texts, (
            "pressure-dependent reactions must mark the band as advisory"
        )
        plt.close(fig)


class TestMisfitEncodings:
    def _router_with_update(self, update):
        fig = plt.figure()
        router = ViewRouter(fig, _context())
        router.record(IterationEvent.from_update(update, is_best=True))
        router.set_view("misfit")

        return fig, router

    def test_marker_alpha_follows_effective_weights(self):
        """w = user × coverage × trim = [1.1, 0.45, 1.0]; opacity is
        floored at 0.15 and scaled to the max weight."""
        fig, router = self._router_with_update(_update())
        incumbent = fig.axes[0].collections[1]
        np.testing.assert_allclose(
            incumbent.get_alpha(), [1.0, 0.497727, 0.922727], atol=1e-4,
            err_msg="marker opacity must track user × coverage × trim",
        )
        plt.close(fig)

    def test_outlier_ring_on_high_z(self):
        """The ring is a separate full-opacity overlay, so trimming
        (low marker alpha) can never fade the flag."""
        update = _update()
        update["stat_plot"]["per_shock"]["z"] = [0.1, -0.2, 3.0]
        fig, router = self._router_with_update(update)
        collections = fig.axes[0].collections
        assert len(collections) == 3, (
            f"expected start + incumbent + ring overlay, got "
            f"{len(collections)}"
        )
        ring = collections[2].get_offsets()
        assert ring.shape == (1, 2), "exactly one shock should be flagged"
        np.testing.assert_allclose(ring[0], [1000.0 / 1900.0, 0.03],
                                   rtol=1e-9)
        plt.close(fig)

    def test_no_ring_overlay_when_no_outliers(self):
        fig, router = self._router_with_update(_update())
        assert len(fig.axes[0].collections) == 2, (
            "no |z| flag → only the start and incumbent scatters"
        )
        plt.close(fig)


class TestRateRatioView:
    def _router(self, update=None, **context_kwargs):
        fig = plt.figure()
        router = ViewRouter(fig, _context(**context_kwargs))
        router.record(IterationEvent.from_update(update or _update(),
                                                 is_best=True))
        router.set_view("arrhenius_ratio")

        return fig, router

    def _ratio_curves(self, fig):
        return [ln for ln in fig.axes[0].lines if "<=>" in ln.get_label()]

    def test_default_selection_plots_every_reaction(self):
        """Surfaces put the incumbent 0.1 (rxn 0) and 0.2 (rxn 1) above
        the initial ln k, so the ratio curves are flat at e^0.1, e^0.2."""
        fig, router = self._router()
        curves = {ln.get_label(): ln for ln in self._ratio_curves(fig)}
        assert len(curves) == 2, f"expected both reactions, got {curves}"
        by_prefix = {label.split("  [")[0]: ln for label, ln in curves.items()}
        np.testing.assert_allclose(by_prefix["A0 <=> B0"].get_ydata(),
                                   np.exp(0.1) * np.ones(5), rtol=1e-9)
        np.testing.assert_allclose(by_prefix["A1 <=> B1"].get_ydata(),
                                   np.exp(0.2) * np.ones(5), rtol=1e-9)
        plt.close(fig)

    def test_selection_subsets_curves(self):
        fig, router = self._router()
        router.views["arrhenius_ratio"].set_selection([1])
        router.refresh()
        curves = self._ratio_curves(fig)
        assert len(curves) == 1
        assert curves[0].get_label().startswith("A1 <=> B1")
        plt.close(fig)

    def test_band_lines_sit_at_uncertainty_factor(self):
        """halfwidth ln-space 0.693 → dotted lines at ×2 and ÷2."""
        fig, router = self._router()
        levels = sorted({
            round(float(ln.get_ydata()[0]), 6)
            for ln in fig.axes[0].lines
            if len(ln.get_ydata()) == 2
            and ln.get_ydata()[0] == ln.get_ydata()[1]
        })
        assert pytest.approx(np.exp(-0.693), rel=1e-6) == levels[0]
        assert pytest.approx(np.exp(0.693), rel=1e-6) == levels[-1]
        plt.close(fig)

    def test_at_bound_reaction_labeled(self):
        """Scaler index 2 (-0.69299) sits at reaction 1's lower bound."""
        fig, router = self._router()
        labels = [ln.get_label() for ln in self._ratio_curves(fig)]
        flagged = [lab for lab in labels if "AT BOUND" in lab]
        assert flagged and flagged[0].startswith("A1 <=> B1"), (
            f"reaction 1 must carry the AT BOUND tag; labels: {labels}"
        )
        plt.close(fig)

    def test_empty_selection_says_so(self):
        fig, router = self._router()
        router.views["arrhenius_ratio"].set_selection([])
        router.refresh()
        texts = " ".join(t.get_text() for t in fig.axes[0].texts)
        assert "no reactions selected" in texts
        plt.close(fig)


class TestImprovementView:
    def test_counts_improved_and_worse(self):
        """Start [0.05, 0.06, 0.07] vs now [0.02, 0.04, 0.03]: all 3
        improved."""
        fig = plt.figure()
        router = ViewRouter(fig, _context())
        router.record(IterationEvent.from_update(_update(), is_best=True))
        router.set_view("improvement")
        texts = " ".join(t.get_text() for t in fig.axes[0].texts)
        assert "3 improved · 0 worse" in texts, f"got {texts!r}"
        plt.close(fig)

    def test_without_start_losses_says_so(self):
        update = _update()
        update["stat_plot"]["per_shock"]["loss_raw_start"] = [np.nan] * 3
        fig = plt.figure()
        router = ViewRouter(fig, _context())
        router.record(IterationEvent.from_update(update, is_best=True))
        router.set_view("improvement")
        texts = " ".join(t.get_text() for t in fig.axes[0].texts)
        assert "start losses unavailable" in texts
        plt.close(fig)


class TestTimeOffsetView:
    def _router(self, update=None):
        fig = plt.figure()
        router = ViewRouter(fig, _context())
        router.record(IterationEvent.from_update(update or _update(),
                                                 is_best=True))
        router.set_view("time_offsets")

        return fig, router

    def _by_label(self, ax, label):
        for coll in ax.collections:
            if coll.get_label() == label:
                return coll
        labels = [c.get_label() for c in ax.collections]

        raise AssertionError(f"no collection labeled {label!r}: {labels}")

    def test_parametric_mode_shows_free_optimum_and_pull(self):
        """star differs from applied → hollow free-optimum markers plus
        the filled applied markers."""
        fig, router = self._router()
        ax = fig.axes[0]
        texts = " ".join(t.get_text() for t in ax.texts)
        assert "parametric-model offset" in texts
        assert self._by_label(ax, "free per-shock optimum") is not None
        assert self._by_label(ax, "applied") is not None
        plt.close(fig)

    def test_offset_at_uncertainty_limit_is_flagged(self):
        """base 0.5 + τ 0.99 µs sits within 2% of the +1.5 µs edge."""
        fig, router = self._router()
        texts = " ".join(t.get_text() for t in fig.axes[0].texts)
        assert "at the ± time-uncertainty limit" in texts
        plt.close(fig)

    def test_independent_mode_shows_applied_only(self):
        update = _update()
        update["stat_plot"]["per_shock"]["t_unc_mode"] = "independent"
        fig, router = self._router(update)
        ax = fig.axes[0]
        texts = " ".join(t.get_text() for t in ax.texts)
        assert "random mode" in texts
        labels = [c.get_label() for c in ax.collections]
        assert "free per-shock optimum" not in labels, (
            "independent mode must not draw the hollow free-optimum layer"
        )
        plt.close(fig)

    def test_fixed_mode_notes_no_solving(self):
        update = _update()
        diag = update["stat_plot"]["per_shock"]
        diag["t_unc_mode"] = "fixed"
        diag["t_unc"] = [0.0, 0.0, 0.0]
        diag["t_unc_star"] = [0.0, 0.0, 0.0]
        fig, router = self._router(update)
        texts = " ".join(t.get_text() for t in fig.axes[0].texts)
        assert "time offsets fixed" in texts
        plt.close(fig)

    def test_applied_offsets_are_absolute_microseconds(self):
        """Display = set offset (0.5 µs) + solved τ, not τ alone."""
        fig, router = self._router()
        applied = self._by_label(fig.axes[0], "applied")
        y = applied.get_offsets()[:, 1]
        np.testing.assert_allclose(y, [0.7, 1.0, 1.49], rtol=1e-9,
                                   atol=1e-12)
        plt.close(fig)

    def test_no_window_artists_drawn(self):
        """The allowed-window segments and set-offset dashes are gone."""
        fig, router = self._router()
        labels = [c.get_label() for c in fig.axes[0].collections]
        assert "allowed window" not in labels
        assert "set offset" not in labels
        plt.close(fig)


class TestPlotIntegrationContract:
    def test_view_labels_match_router_keys(self):
        fig = plt.figure()
        router = ViewRouter(fig, _context())
        label_keys = {key for key in VIEW_LABELS.values() if key is not None}
        assert label_keys == set(router.views), (
            f"dropdown keys {sorted(label_keys)} must match router views "
            f"{sorted(router.views)}"
        )
        plt.close(fig)
