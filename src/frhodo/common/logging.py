"""Logging utilities for Frhodo.

Core modules emit through ``logging.getLogger(__name__)``. The GUI
installs ``GuiLogHandler`` at startup so records flow to the log
widget; CLI and library users see ``NullHandler`` by default and can
configure their own handlers.

``WARNING`` and above carry the GUI's blinking-tab alert.
"""
import logging


class GuiLogHandler(logging.Handler):
    """Pipes ``logging`` records into the GUI log widget.

    The widget update is dispatched onto the Qt main thread via a
    queued signal so handler callers can emit from worker threads
    safely. Records at ``WARNING`` level or above trigger the
    blinking-tab alert.
    """

    def __init__(self, log_widget):
        super().__init__()
        from PyQt5.QtCore import QObject, Qt, pyqtSignal

        self._log_widget = log_widget

        class _Dispatcher(QObject):
            message = pyqtSignal(str, bool)

        self._dispatcher = _Dispatcher()
        self._dispatcher.message.connect(
            self._append_to_widget, type=Qt.QueuedConnection
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record)
        except Exception:
            self.handleError(record)
            return
        alert = record.levelno >= logging.WARNING
        self._dispatcher.message.emit(text, alert)

    def _append_to_widget(self, text: str, alert: bool) -> None:
        self._log_widget.append(text, alert=alert)
