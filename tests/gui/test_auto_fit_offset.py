"""Regression test for ``apply_auto_fit_time_offset``.

Auto-fit routes through the optimizer's ``_solve_t_unc``; the
stationarity rework dropped that solver's ``warm_start_t_unc`` argument,
and the auto-fit caller kept passing it — a ``TypeError`` the moment the
user toggled Auto-fit on with a simulation present. The faked parent
exercises the real solve path without booting Qt or running a sim.
"""
import types

import numpy as np
import pytest

from frhodo.gui.widgets.options_panel_widgets import apply_auto_fit_time_offset



class _OffsetBox:
    def __init__(self):
        self._value = 0.0
        self.twin = [self]

    def blockSignals(self, _flag):
        pass

    def setValue(self, value):
        self._value = value

    def value(self):
        return self._value


def _bump(t, center, width):
    return np.exp(-(((t - center) / width) ** 2))


def _fake_parent(delta):
    span = 1.0e-4
    t_exp = np.linspace(0.0, span, 200)
    center = 0.5 * span
    width = 0.1 * span
    obs_exp = _bump(t_exp, center, width)
    # SIM bump sits earlier by delta, so aligning needs a +delta shift —
    # an interior optimum inside the (0, 0.5*span) search bound.
    obs_sim = _bump(t_exp, center - delta, width)

    shock = types.SimpleNamespace(
        exp_data=np.column_stack([t_exp, obs_exp]),
        weight_shift=[5.0, 50.0],
        time_offset=None,
        opt_time_offset=0.0,
    )
    # A signal axis with no "sim_data" item, so the live-plot tail is
    # exercised but takes its no-op branch.
    signal_ax = types.SimpleNamespace(item={})
    plot = types.SimpleNamespace(
        signal=types.SimpleNamespace(ax=[None, signal_ax]),
    )
    parent = types.SimpleNamespace(
        time_uncertainty=types.SimpleNamespace(auto_fit=True, offset=None),
        display_shock=shock,
        SIM=types.SimpleNamespace(independent_var=t_exp, observable=obs_sim),
        run_control=types.SimpleNamespace(optimize_running=False),
        reactor_state=types.SimpleNamespace(t_unit_conv=1.0e-6),
        series=types.SimpleNamespace(
            weights=lambda t, shock=None, calcIntegral=True: np.ones_like(t),
        ),
        time_offset_box=_OffsetBox(),
        plot=plot,
    )

    return parent, shock, span


def test_auto_fit_solves_without_stale_kwarg():
    parent, shock, _span = _fake_parent(delta=1.0e-5)
    apply_auto_fit_time_offset(parent)
    assert shock.time_offset is not None, (
        "auto-fit must set a time offset through the solver"
    )
    assert np.isfinite(shock.time_offset)
    assert parent.time_uncertainty.offset == shock.time_offset


def test_auto_fit_recovers_known_shift():
    delta = 1.0e-5
    parent, shock, _span = _fake_parent(delta=delta)
    apply_auto_fit_time_offset(parent)
    assert shock.time_offset == pytest.approx(delta, abs=2.0e-6), (
        f"auto-fit should recover the {delta:g} s lag, got {shock.time_offset}"
    )


def test_auto_fit_noop_when_disabled():
    parent, shock, _span = _fake_parent(delta=1.0e-5)
    parent.time_uncertainty.auto_fit = False
    apply_auto_fit_time_offset(parent)
    assert shock.time_offset is None
