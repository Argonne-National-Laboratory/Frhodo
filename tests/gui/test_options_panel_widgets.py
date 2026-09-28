"""Tests for ``frhodo.gui.widgets.options_panel_widgets``.

``Log`` blinks its tab while an alert is unread. A blinking log must stay
freeable by reference counting: a timer slot that captures the log puts it
in a cycle only the garbage collector can free, and a collection on a
worker thread then leaves the timer calling a cleared function on the GUI
thread.
"""
import gc
import weakref

import pytest
from qtpy.QtWidgets import QPushButton, QTabWidget, QTextEdit, QWidget

from frhodo.gui.widgets.options_panel_widgets import Log



@pytest.fixture
def tab_widget(qtbot):
    tabs = QTabWidget()
    qtbot.addWidget(tabs)
    log_tab = QWidget()
    log_tab.setObjectName("log_tab")
    tabs.addTab(QWidget(), "Options")
    tabs.addTab(log_tab, "Log")

    return tabs


def _make_log(tab_widget):
    return Log(tab_widget, QTextEdit(), QPushButton(), QPushButton())


class TestBlinkTimer:
    def test_alert_on_hidden_log_tab_starts_blinking(self, tab_widget):
        log = _make_log(tab_widget)

        log.append("message")

        assert log.timer.isActive(), (
            "an alert while the log tab is hidden should start the blink timer"
        )

    def test_showing_log_tab_stops_blinking(self, tab_widget):
        log = _make_log(tab_widget)
        log.append("message")

        tab_widget.setCurrentIndex(log.log_tab_idx)

        assert not log.timer.isActive(), "showing the log tab should stop the blink timer"

    def test_blinking_log_is_freed_by_reference_counting(self, tab_widget):
        log = _make_log(tab_widget)
        log.append("message")
        ref = weakref.ref(log)

        gc.disable()
        try:
            del log
            freed = ref() is None
        finally:
            gc.enable()

        assert freed, (
            "a blinking Log outlived its last reference, so only the garbage "
            "collector could free it"
        )
