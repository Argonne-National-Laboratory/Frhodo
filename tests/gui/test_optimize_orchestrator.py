"""Lifecycle wiring on :class:`Multithread_Optimize`.

Covers auto-fit save/restore around an optimization run: the run forces
``time_uncertainty.auto_fit`` on while it owns the per-iteration t_unc
solve, then restores the user's prior setting on completion or abort
without disturbing the current time offset.
"""
from unittest.mock import MagicMock

from frhodo.gui.optimize_orchestrator import Multithread_Optimize
from frhodo.gui.state import RunControlState, TimeUncertaintyState


class _StubParent:
    """Minimum surface for ``Multithread_Optimize.__init__``."""

    def __init__(self):
        self.run_control = RunControlState(multiprocessing=False)
        self.time_uncertainty = TimeUncertaintyState()
        self.action_Run = MagicMock()
        self.action_Abort = MagicMock()
        self.multiprocessing_box = MagicMock()
        self.multiprocessing_box.isChecked.return_value = False
        self.threadpool = MagicMock()


class TestAutoFitSaveRestore:
    """Optimization run forces ``auto_fit`` on and restores the prior
    state when the run finishes or is aborted. The current time offset
    is left untouched on restore."""

    def test_abort_restores_prior_auto_fit_off(self):
        parent = _StubParent()
        parent.time_uncertainty.auto_fit = False
        adapter = Multithread_Optimize(parent)
        adapter._saved_auto_fit = False
        parent.time_uncertainty.auto_fit = True  # simulate "while running"
        parent.time_uncertainty.offset = 0.42

        adapter.abort_workers()

        assert parent.time_uncertainty.auto_fit is False
        assert parent.time_uncertainty.offset == 0.42
        assert adapter._saved_auto_fit is None

    def test_abort_restores_prior_auto_fit_on(self):
        parent = _StubParent()
        adapter = Multithread_Optimize(parent)
        adapter._saved_auto_fit = True
        parent.time_uncertainty.auto_fit = True

        adapter.abort_workers()

        assert parent.time_uncertainty.auto_fit is True

    def test_restore_is_noop_when_never_saved(self):
        """``_restore_auto_fit`` without a prior save (e.g. _prepare_run
        rejected before saving) must not flip the flag."""
        parent = _StubParent()
        parent.time_uncertainty.auto_fit = True
        adapter = Multithread_Optimize(parent)

        adapter._restore_auto_fit()

        assert parent.time_uncertainty.auto_fit is True
