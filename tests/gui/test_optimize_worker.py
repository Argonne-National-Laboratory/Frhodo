"""Tests for ``frhodo.gui.workers.optimize_worker``.

The ``Worker.run()`` method is the boundary between Qt's threadpool
and the long-running optimization. It used to swallow every exception
silently (bare ``except:`` discarding the captured exc_info); the fix
narrows to ``except Exception`` and emits ``signals.error`` so the GUI
hears about failures. These tests pin both halves of that contract.
"""
import sys

import pytest
from qtpy.QtCore import QRunnable
from unittest.mock import MagicMock

from frhodo.gui.workers.optimize_worker import Worker, WorkerSignals


@pytest.fixture
def worker(qapp):
    """Build a Worker without the heavy mech/coef_opt construction
    ``__init__`` would otherwise demand. Individual tests then patch
    ``optimize_coeffs`` to control success vs failure."""
    instance = Worker.__new__(Worker)
    QRunnable.__init__(instance)
    instance.signals = WorkerSignals()
    instance._abort = False

    return instance


class TestWorkerErrorSignal:
    """Bare-except fix: ``run()`` must surface failures by emitting
    ``signals.error`` instead of swallowing them."""

    def test_error_signal_emitted_on_exception(self, qtbot, worker):
        worker.optimize_coeffs = MagicMock(
            side_effect=RuntimeError("synthetic failure"),
        )

        with qtbot.waitSignal(worker.signals.error, timeout=2000) as blocker:
            worker.run()

        assert blocker.args is not None, (
            "signals.error should have fired before timeout"
        )
        payload = blocker.args[0]
        assert isinstance(payload, tuple), (
            f"error payload should be a 2-tuple; got {type(payload).__name__}"
        )
        assert len(payload) == 2
        exc_info, tb_text = payload
        assert "synthetic failure" in tb_text, (
            f"traceback should mention the raised message; got {tb_text!r}"
        )

    def test_error_signal_contains_exc_info_pair(self, qtbot, worker):
        """First half of the payload is ``sys.exc_info()[:2]``: the
        exception type and instance."""
        worker.optimize_coeffs = MagicMock(side_effect=ValueError("bad input"))

        with qtbot.waitSignal(worker.signals.error, timeout=2000) as blocker:
            worker.run()

        exc_info, _ = blocker.args[0]
        exc_type, exc_value = exc_info
        assert exc_type is ValueError
        assert isinstance(exc_value, ValueError)
        assert "bad input" in str(exc_value)

    def test_result_signal_not_emitted_on_error(self, qtbot, worker):
        """A failed run must never emit ``signals.result`` — the GUI's
        result handler does completion bookkeeping it shouldn't run for
        a half-done optimization."""
        worker.optimize_coeffs = MagicMock(side_effect=RuntimeError("boom"))
        result_emissions = []
        worker.signals.result.connect(result_emissions.append)

        worker.run()
        qtbot.wait(50)  # drain queued signal deliveries

        assert result_emissions == [], (
            f"result should not fire on error; got {result_emissions!r}"
        )


class TestWorkerSuccessSignal:
    """The other half: a normal completion still emits ``signals.result``
    with the optimizer's return value."""

    def test_result_signal_emitted_on_success(self, qtbot, worker):
        sentinel = {"x": [1.0, 2.0], "fval": 0.001}
        worker.optimize_coeffs = MagicMock(return_value=sentinel)

        with qtbot.waitSignal(worker.signals.result, timeout=2000) as blocker:
            worker.run()

        assert blocker.args == [sentinel]

    def test_error_signal_not_emitted_on_success(self, qtbot, worker):
        worker.optimize_coeffs = MagicMock(return_value="ok")
        error_emissions = []
        worker.signals.error.connect(error_emissions.append)

        worker.run()
        qtbot.wait(50)

        assert error_emissions == []


class TestWorkerCriticalExceptions:
    """The narrowed ``except Exception`` clause must still let
    ``KeyboardInterrupt`` and ``SystemExit`` propagate — they're not
    application-level errors and should reach the runner."""

    def test_keyboard_interrupt_propagates(self, worker):
        worker.optimize_coeffs = MagicMock(side_effect=KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            worker.run()

    def test_system_exit_propagates(self, worker):
        worker.optimize_coeffs = MagicMock(side_effect=SystemExit(2))
        with pytest.raises(SystemExit):
            worker.run()


class TestWorkerSignalsSurface:
    """``WorkerSignals`` is the contract between Worker and the GUI;
    pin the signal set so a future removal would fail loudly."""

    @pytest.mark.parametrize("name", [
        "finished", "error", "result", "update", "progress", "log", "abort",
    ])
    def test_signal_attribute_exists(self, name):
        signals = WorkerSignals()
        assert hasattr(signals, name), (
            f"WorkerSignals should expose {name!r} (consumers connect to it)"
        )


