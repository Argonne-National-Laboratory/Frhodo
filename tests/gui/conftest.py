"""Shared fixtures for GUI tests.

Forces the offscreen QPA platform so tests run on a headless host (CI,
WSL without an X server, etc.). ``pytest-qt`` provides ``qapp`` and
``qtbot`` automatically once Qt is importable.
"""
import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _isolate_qt_capture_env(monkeypatch):
    """Keep the capture-hook env var off for every GUI test by default —
    individual tests opt in via ``monkeypatch.setenv``."""
    monkeypatch.delenv("FRHODO_TROE_CAPTURE", raising=False)
