"""Optimization-outcome views: matplotlib renderers behind one router.

The router owns which view occupies the optimization figure and feeds
every view from :class:`IterationEvent` alone. The residuals (QQ +
density) canvas is not a router view — the plot widget keeps it as the
default mode and hands the figure to the router only while one of
these views is selected.
"""
import matplotlib as mpl
import numpy as np

from frhodo.gui.views.events import IterationEvent, ViewContext



AT_BOUND_FRACTION = 0.01
Z_FLAG = 2.5
ALPHA_FLOOR = 0.15
EDGE_PIN_FRACTION = 0.02


def _interp_ln_p(surface, P_grid, pressure_pa) -> np.ndarray:
    """Interpolate one reaction's (P, T) surface at a display pressure
    (linear in ln P, clamped to the grid edges)."""
    arr = np.asarray(surface, dtype=float)
    P_grid = np.asarray(P_grid, dtype=float)
    if arr.shape[0] == 1 or P_grid.size == 1:
        return arr[0]
    ln_p = np.log(np.clip(pressure_pa, P_grid[0], P_grid[-1]))
    ln_grid = np.log(P_grid)
    curve = np.array([
        np.interp(ln_p, ln_grid, arr[:, j]) for j in range(arr.shape[1])
    ])

    return curve


def _at_bound_count(context: ViewContext, event: IterationEvent,
                    rxn_pos: int) -> int:
    """Rate anchors of one reaction sitting at their scaler bounds."""
    s = np.asarray(event.scalers, dtype=float)
    lb = np.asarray(context.lower_bounds)
    ub = np.asarray(context.upper_bounds)
    if s.size != lb.size:
        return 0
    rxn_idx = context.rxn_indices[rxn_pos]
    mine = np.asarray(context.param_rxn) == rxn_idx
    span = np.where(ub > lb, ub - lb, 1.0)
    position = (s - lb) / span
    at_bound = (position <= AT_BOUND_FRACTION) | (
        position >= 1.0 - AT_BOUND_FRACTION
    )

    return int(np.sum(at_bound & mine))


def _effective_weights(event: IterationEvent, n: int) -> np.ndarray:
    """Each shock's weight in the objective: user × coverage × adaptive
    trimming weight at the solved aggregate shape."""
    user = np.asarray(event.user_weights, dtype=float)
    cov = np.asarray(event.coverage, dtype=float)
    trim = np.asarray(event.trim_weights, dtype=float)
    if user.size != n or cov.size != n or trim.size != n:
        return np.ones(n)

    return user * cov * trim


def _weight_alpha(event: IterationEvent, n: int) -> np.ndarray:
    """Marker opacity from the effective objective weights, floored so
    fully trimmed shocks stay visible."""
    w = _effective_weights(event, n)
    if w.size:
        w_max = w.max()
    else:
        w_max = 0.0
    if not np.isfinite(w_max) or w_max <= 0:
        return np.ones(n)

    return ALPHA_FLOOR + (1.0 - ALPHA_FLOOR) * w / w_max


def _z_flagged(event: IterationEvent, n: int) -> np.ndarray:
    """Shocks anomalous relative to the campaign-typical misfit level."""
    z = np.asarray(event.z, dtype=float)
    if z.size != n:
        return np.zeros(n, dtype=bool)

    return np.abs(z) > Z_FLAG


def _ring_overlay(ax, x, y, flagged) -> None:
    """Full-opacity crimson rings; a separate artist so per-point marker
    alpha never fades the flag."""
    if np.any(flagged):
        ax.scatter(np.asarray(x)[flagged], np.asarray(y)[flagged], s=95,
                   facecolors="none", edgecolors="crimson", linewidths=1.4,
                   zorder=3)


def _condition_norm(view, color: np.ndarray):
    """One shared color normalization per view instance, fixed at the
    first update so scatter colors and the colorbar agree across the run."""
    if view._norm is None:
        view._norm = mpl.colors.Normalize(vmin=float(np.min(color)),
                                          vmax=float(np.max(color)))

    return view._norm


def _condition_colorbar(view, ax) -> None:
    if view._colorbar is None:
        sm = mpl.cm.ScalarMappable(norm=view._norm, cmap="viridis")
        view._colorbar = ax.figure.colorbar(sm, ax=ax, pad=0.01)
        view._colorbar.set_label("log10 P")


class _RunHistory:
    """Objective trace + incumbent/initial bookkeeping across one run."""

    def __init__(self):
        self.iterations = []
        self.objectives = []
        self.stages = []
        self.best_objectives = []
        self.best_event = None
        self.initial_event = None

    def add(self, event: IterationEvent) -> None:
        if self.initial_event is None:
            self.initial_event = event
        if event.is_best or self.best_event is None:
            self.best_event = event
        self.iterations.append(len(self.iterations) + 1)
        self.objectives.append(event.obj_fcn)
        self.stages.append(event.stage)
        if self.best_objectives:
            best = min(self.best_objectives[-1], event.obj_fcn)
        else:
            best = event.obj_fcn
        self.best_objectives.append(best)

    def stage_boundary(self):
        """Index of the first local-stage event, or None."""
        for j, stage in enumerate(self.stages):
            if stage == "local":
                if j > 0:
                    return j

                return None

        return None

    def clear(self) -> None:
        self.__init__()


class MisfitMapView:
    """Per-shock loss vs 1000/T, colored by log10 P; start vs best.

    Marker opacity encodes each shock's effective weight in the
    objective; a crimson ring flags shocks with |z| beyond the
    campaign-typical misfit level.
    """

    title = "Misfit Map"

    def __init__(self):
        self.ax = None
        self._colorbar = None
        self._norm = None

    def build(self, fig) -> None:
        self.ax = fig.add_subplot(1, 1, 1)
        self.ax.set_xlabel("1000 / T [1/K]")
        self.ax.set_ylabel("Per-shock loss (objective basis)")
        self.ax.set_yscale("log")
        self.ax.text(0.02, 0.02,
                     f"opacity ∝ weight in objective · "
                     f"red ring: |z| > {Z_FLAG:g}",
                     transform=self.ax.transAxes, fontsize="x-small",
                     color="0.45", verticalalignment="bottom")
        self._colorbar = None
        self._norm = None

    def update(self, event: IterationEvent, history: _RunHistory) -> None:
        best = history.best_event or event
        n = len(best.T)
        x = 1000.0 / np.asarray(best.T)
        color = np.log10(np.asarray(best.P))
        y_best = np.asarray(best.loss_obj)
        y_start = np.asarray(best.loss_obj_start)

        ax = self.ax
        for artist in list(ax.collections) + list(ax.lines):
            artist.remove()
        if np.any(np.isfinite(y_start)):
            ax.scatter(x, y_start, s=42, facecolors="none", edgecolors="0.55",
                       label="start")
        norm = _condition_norm(self, color)
        ax.scatter(x, y_best, s=42, c=color, cmap="viridis", norm=norm,
                   alpha=_weight_alpha(best, n), label="incumbent best")
        _ring_overlay(ax, x, y_best, _z_flagged(best, n))
        for xi, y0, y1 in zip(x, y_start, y_best):
            if np.isfinite(y0):
                ax.plot([xi, xi], [y0, y1], color="0.8", lw=0.8, zorder=0)
        _condition_colorbar(self, ax)
        legend = ax.get_legend()
        if legend is None:
            ax.legend(loc="best", fontsize="small")
        ax.relim()
        ax.autoscale_view()


class ArrheniusView:
    """ln k(T) for one selected reaction: initial, incumbent, bound band.

    Curves are interpolated in ln P from the worker-computed (P, T)
    surfaces, so the pressure box acts instantly and display-side. The
    band is the initial curve ± the rate-uncertainty half-width — a
    hard constraint for plain-Arrhenius multiplier targets. For
    pressure-dependent targets the optimizer constrains the limit-rate
    anchors instead, and the refit blend at the display pressure can
    leave this band mid-run; the subplot says so.
    """

    title = "Arrhenius with Bounds"

    def __init__(self, context: ViewContext):
        self.context = context
        self.reaction_pos = 0
        self.pressure_pa = float(context.P_reference)
        self.ax = None

    def set_reaction(self, pos: int) -> None:
        self.reaction_pos = int(np.clip(pos, 0,
                                        len(self.context.rxn_indices) - 1))

    def set_pressure(self, pressure_pa: float) -> None:
        self.pressure_pa = float(pressure_pa)

    def build(self, fig) -> None:
        self.ax = fig.add_subplot(1, 1, 1)
        self.ax.set_xlabel("1000 / T [1/K]")
        self.ax.set_ylabel("ln k")

    def update(self, event: IterationEvent, history: _RunHistory) -> None:
        best = history.best_event or event
        if not best.ln_k:
            return
        r = self.reaction_pos
        ax = self.ax
        x = 1000.0 / np.asarray(self.context.T_grid)
        for artist in list(ax.lines) + list(ax.collections) + list(ax.texts):
            artist.remove()
        P_grid = self.context.P_grid
        ln_k0 = _interp_ln_p(self.context.ln_k_initial[r], P_grid,
                             self.pressure_pa)
        half = self.context.rxn_band_halfwidth[r]
        ax.fill_between(x, ln_k0 - half, ln_k0 + half, color="0.9", zorder=0)
        ax.plot(x, ln_k0, color="0.45", ls="--", lw=1.2, label="initial")
        ax.plot(x, _interp_ln_p(best.ln_k[r], P_grid, self.pressure_pa),
                color="C0", lw=1.6, label="incumbent")
        notes = []
        if self.context.rxn_is_pressure_dependent[r]:
            notes.append("pressure-dependent: band constrains the limit-rate "
                         "anchors, not this curve pointwise")
        n_pinned = _at_bound_count(self.context, best, r)
        if n_pinned:
            notes.append(f"{n_pinned} rate anchor(s) AT BOUND")
        if notes:
            ax.text(0.02, 0.02, "\n".join(notes), transform=ax.transAxes,
                    fontsize="x-small", color="crimson",
                    verticalalignment="bottom")
        if ax.get_legend() is None:
            ax.legend(loc="best", fontsize="x-small")
        ax.relim()
        ax.autoscale_view()


class RateRatioView:
    """k/k₀ vs T for the checked reactions, at the display pressure.

    Ratios normalize every reaction onto one axes: a curve at 1 is
    unchanged, the dotted lines are each reaction's ×f / ÷f rate
    uncertainty, and a curve hugging its line is at its bound. For
    plain-Arrhenius multiplier targets the dotted lines are the exact
    feasible range; for pressure-dependent targets they constrain the
    limit-rate anchors only.
    """

    title = "Arrhenius Ratios"

    def __init__(self, context: ViewContext):
        self.context = context
        self.selected = list(range(len(context.rxn_indices)))
        self.pressure_pa = float(context.P_reference)
        self.ax = None

    def set_selection(self, positions) -> None:
        n = len(self.context.rxn_indices)
        self.selected = [int(p) for p in positions if 0 <= int(p) < n]

    def set_pressure(self, pressure_pa: float) -> None:
        self.pressure_pa = float(pressure_pa)

    def build(self, fig) -> None:
        self.ax = fig.add_subplot(1, 1, 1)
        self.ax.set_xlabel("1000 / T [1/K]")
        self.ax.set_ylabel("k / k₀")
        self.ax.set_yscale("log")

    def update(self, event: IterationEvent, history: _RunHistory) -> None:
        best = history.best_event or event
        if not best.ln_k:
            return
        ax = self.ax
        for artist in list(ax.lines) + list(ax.collections) + list(ax.texts):
            artist.remove()
        if not self.selected:
            ax.text(0.5, 0.5, "no reactions selected",
                    transform=ax.transAxes, ha="center", color="0.5")

            return
        ax.axhline(1.0, color="0.3", lw=0.8, zorder=1)
        x = 1000.0 / np.asarray(self.context.T_grid)
        P_grid = self.context.P_grid
        any_pressure_dependent = False
        for j, r in enumerate(self.selected):
            color = f"C{j % 10}"
            ln_k0 = _interp_ln_p(self.context.ln_k_initial[r], P_grid,
                                 self.pressure_pa)
            ln_kb = _interp_ln_p(best.ln_k[r], P_grid, self.pressure_pa)
            label = self.context.rxn_equations[r]
            n_pinned = _at_bound_count(self.context, best, r)
            if n_pinned:
                label += f"  [{n_pinned} AT BOUND]"
            ax.plot(x, np.exp(ln_kb - ln_k0), color=color, lw=1.6,
                    label=label)
            half = self.context.rxn_band_halfwidth[r]
            ax.axhline(np.exp(half), color=color, ls=":", lw=0.9, zorder=1)
            ax.axhline(np.exp(-half), color=color, ls=":", lw=0.9, zorder=1)
            if self.context.rxn_is_pressure_dependent[r]:
                any_pressure_dependent = True
        if any_pressure_dependent:
            ax.text(0.02, 0.02,
                    "pressure-dependent reactions: dotted lines constrain "
                    "the limit-rate anchors, not this curve pointwise",
                    transform=ax.transAxes, fontsize="x-small",
                    color="crimson", verticalalignment="bottom")
        ax.legend(loc="best", fontsize="x-small")
        ax.relim()
        ax.autoscale_view()


class ImprovementView:
    """Per-shock loss at start vs at the incumbent, with the diagonal.

    Points below the diagonal improved; above it were sacrificed for
    the aggregate. Opacity and rings carry the same weight / |z|
    encodings as the misfit maps.
    """

    title = "Improvement"

    def __init__(self):
        self.ax = None
        self._colorbar = None
        self._norm = None

    def build(self, fig) -> None:
        self.ax = fig.add_subplot(1, 1, 1)
        self.ax.set_xlabel("Per-shock loss at start (objective basis)")
        self.ax.set_ylabel("Per-shock loss at incumbent")
        self.ax.set_xscale("log")
        self.ax.set_yscale("log")
        self._colorbar = None
        self._norm = None

    def update(self, event: IterationEvent, history: _RunHistory) -> None:
        best = history.best_event or event
        n = len(best.T)
        ax = self.ax
        for artist in (list(ax.collections) + list(ax.lines)
                       + list(ax.texts)):
            artist.remove()
        x = np.asarray(best.loss_obj_start, dtype=float)
        y = np.asarray(best.loss_obj, dtype=float)
        finite = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if not np.any(finite):
            ax.text(0.5, 0.5, "start losses unavailable for this run",
                    transform=ax.transAxes, ha="center", color="0.5")

            return
        lo = min(x[finite].min(), y[finite].min())
        hi = max(x[finite].max(), y[finite].max())
        ax.plot([lo, hi], [lo, hi], color="0.4", ls="--", lw=1.0, zorder=1)
        color = np.log10(np.asarray(best.P))
        norm = _condition_norm(self, color)
        ax.scatter(x, y, s=42, c=color, cmap="viridis", norm=norm,
                   alpha=_weight_alpha(best, n))
        _ring_overlay(ax, x, y, _z_flagged(best, n))
        _condition_colorbar(self, ax)
        n_better = int(np.sum(y[finite] < x[finite]))
        n_worse = int(np.sum(y[finite] > x[finite]))
        ax.text(0.02, 0.98, f"{n_better} improved · {n_worse} worse",
                transform=ax.transAxes, fontsize="small",
                verticalalignment="top")
        ax.text(0.98, 0.02,
                f"below the diagonal = improved · opacity ∝ weight · "
                f"red ring: |z| > {Z_FLAG:g}",
                transform=ax.transAxes, fontsize="x-small", color="0.45",
                horizontalalignment="right", verticalalignment="bottom")
        ax.relim()
        ax.autoscale_view()


class TimeOffsetView:
    """Applied per-shock time offsets vs 1000/T, in absolute terms
    (set time offset + the solved adjustment).

    Shows only what the optimizer actually did, keyed on its shift
    mode: in parametric mode, hollow circles add each shock's free
    per-shock optimum and the drop line is the model's pull; in
    independent (random) mode just the applied offsets appear. A
    crimson ring marks offsets at the ± time-uncertainty limit —
    alignment wanting more shift than the setting allows.
    """

    title = "Time Offsets"

    def __init__(self):
        self.ax = None
        self._colorbar = None
        self._norm = None

    def build(self, fig) -> None:
        self.ax = fig.add_subplot(1, 1, 1)
        self.ax.set_xlabel("1000 / T [1/K]")
        self.ax.set_ylabel("applied time offset [µs]")
        self._colorbar = None
        self._norm = None

    def update(self, event: IterationEvent, history: _RunHistory) -> None:
        best = history.best_event or event
        n = len(best.T)
        ax = self.ax
        for artist in (list(ax.collections) + list(ax.lines)
                       + list(ax.texts) + list(ax.patches)):
            artist.remove()
        x = 1000.0 / np.asarray(best.T)
        base = np.asarray(best.t_offset_base, dtype=float) * 1e6
        if base.size != n:
            base = np.zeros(n)
        applied = base + np.asarray(best.t_unc, dtype=float) * 1e6
        star = base + np.asarray(best.t_unc_star, dtype=float) * 1e6
        if star.size != n:
            star = applied

        if best.t_unc_mode == "parametric":
            ax.scatter(x, star, s=42, facecolors="none", edgecolors="0.55",
                       label="free per-shock optimum")
            for xi, y0, y1 in zip(x, star, applied):
                ax.plot([xi, xi], [y0, y1], color="0.8", lw=0.8, zorder=1)

        bounds = np.asarray(best.t_unc_bounds, dtype=float) * 1e6
        if bounds.size == 2 and bounds[1] > bounds[0]:
            window = float(bounds[1] - bounds[0])
            win_lo = base + float(bounds[0])
            win_hi = base + float(bounds[1])
            pinned = ((applied - win_lo < EDGE_PIN_FRACTION * window)
                      | (win_hi - applied < EDGE_PIN_FRACTION * window))
        else:
            pinned = np.zeros(n, dtype=bool)
        color = np.log10(np.asarray(best.P))
        norm = _condition_norm(self, color)
        ax.scatter(x, applied, s=42, c=color, cmap="viridis", norm=norm,
                   label="applied")
        _ring_overlay(ax, x, applied, pinned)
        _condition_colorbar(self, ax)

        if best.t_unc_mode == "parametric":
            note = ("filled: parametric-model offset · hollow: free "
                    "per-shock optimum")
        elif best.t_unc_mode == "independent":
            note = "offsets solved per shock (random mode)"
        else:
            note = "time offsets fixed (time uncertainty = 0)"
        n_pinned = int(np.sum(pinned))
        if n_pinned:
            ax.text(0.02, 0.98,
                    f"{n_pinned} offset(s) at the ± time-uncertainty limit",
                    transform=ax.transAxes, fontsize="x-small",
                    color="crimson", verticalalignment="top")
        ax.text(0.02, 0.02, note, transform=ax.transAxes,
                fontsize="x-small", color="0.45",
                verticalalignment="bottom")
        if ax.get_legend() is None and best.t_unc_mode == "parametric":
            ax.legend(loc="best", fontsize="x-small")
        ax.relim()
        ax.autoscale_view()


class ObjectiveTraceView:
    """Objective vs evaluation with best-so-far and the stage boundary."""

    title = "Objective Trace"

    def __init__(self):
        self.ax = None

    def build(self, fig) -> None:
        self.ax = fig.add_subplot(1, 1, 1)
        self.ax.set_xlabel("evaluation")
        self.ax.set_ylabel("objective")
        self.ax.set_yscale("log")

    def update(self, event: IterationEvent, history: _RunHistory) -> None:
        ax = self.ax
        for artist in list(ax.lines):
            artist.remove()
        i = np.asarray(history.iterations)
        obj = np.asarray(history.objectives, dtype=float)
        finite = np.isfinite(obj) & (obj > 0)
        ax.plot(i[finite], obj[finite], ".", color="0.6", ms=3,
                label="evaluations")
        ax.plot(i, np.asarray(history.best_objectives), color="C0", lw=1.6,
                label="best so far")
        boundary = history.stage_boundary()
        if boundary is not None:
            ax.axvline(boundary + 0.5, color="crimson", ls=":", lw=1.2)
        if ax.get_legend() is None:
            ax.legend(loc="best", fontsize="small")
        ax.relim()
        ax.autoscale_view()


class ViewRouter:
    """Routes iteration events to whichever view owns the figure."""

    def __init__(self, fig, context: ViewContext | None = None):
        self.fig = fig
        self.history = _RunHistory()
        self.views = {}
        self.active_name = None
        self.set_context(context)

    def set_context(self, context: ViewContext | None) -> None:
        """(Re)create the views for a run's static context."""
        self.context = context
        self.views = {
            "misfit": MisfitMapView(),
            "objective_trace": ObjectiveTraceView(),
            "improvement": ImprovementView(),
            "time_offsets": TimeOffsetView(),
        }
        if context is not None:
            self.views["arrhenius"] = ArrheniusView(context)
            self.views["arrhenius_ratio"] = RateRatioView(context)
        if self.active_name in self.views:
            self.set_view(self.active_name)

    def start_run(self) -> None:
        self.history.clear()

    def set_view(self, name: str | None) -> None:
        """Activate a view (clearing the figure), or None to release the
        figure back to the residuals canvas."""
        if name in self.views:
            self.active_name = name
        else:
            self.active_name = None
        if self.active_name is None:
            return
        self.fig.clear()
        # The figure inherits the residual canvas's tight margins; give
        # router views room for their y labels. Switching back re-applies
        # the legacy margins when the residual axes rebuild.
        self.fig.subplots_adjust(left=0.1, bottom=0.09, right=0.97, top=0.97)
        self.views[self.active_name].build(self.fig)
        self.refresh()

    def record(self, event: IterationEvent | None) -> None:
        """Accumulate run history; cheap, called for every iteration so
        a view selected mid-run has the full trace."""
        if event is None:
            return
        self.history.add(event)

    def refresh(self) -> bool:
        """Redraw the active view from history. Returns True when the
        figure changed and needs a canvas redraw."""
        if self.active_name is None:
            return False
        event = self.history.best_event
        if event is None:
            return False
        self.views[self.active_name].update(event, self.history)

        return True
