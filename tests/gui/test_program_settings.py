"""Program settings dialog: config round-trip and live cache control."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from frhodo.common.config import FrhodoConfig
from frhodo.gui.widgets.program_settings import ProgramSettingsDialog
from frhodo.optimize.sensitivity_cache import shared_cache



@pytest.fixture()
def stub_parent(tmp_path):
    parent = SimpleNamespace(
        path={"appdata": tmp_path},
        user_settings=SimpleNamespace(
            config=FrhodoConfig(), save=MagicMock(),
        ),
    )

    return parent


@pytest.fixture()
def dialog(qtbot, stub_parent):
    dlg = ProgramSettingsDialog(None)
    dlg.parent = stub_parent
    qtbot.addWidget(dlg)

    return dlg


class TestLoadFromConfig:
    def test_defaults_show_auto_workers_and_cache_cap(self, dialog):
        dialog._load_from_config()
        assert dialog.auto_workers_box.isChecked()
        assert not dialog.worker_count_box.isEnabled()
        assert dialog.cache_mb_box.value() == 250
        assert dialog.prespawn_box.isChecked()

    def test_override_populates_and_enables_spinbox(self, dialog):
        opt = dialog.parent.user_settings.config.optimization
        opt.worker_count = 8
        dialog._load_from_config()
        assert not dialog.auto_workers_box.isChecked()
        assert dialog.worker_count_box.isEnabled()
        assert dialog.worker_count_box.value() == 8


class TestExecuteApply:
    def test_accept_writes_config_and_saves(self, dialog, monkeypatch):
        def user_edits_then_ok():
            dialog.auto_workers_box.setChecked(False)
            dialog.worker_count_box.setValue(12)
            dialog.prespawn_box.setChecked(False)
            dialog.cache_mb_box.setValue(100)

            return 1

        monkeypatch.setattr(dialog, "exec_", user_edits_then_ok)

        dialog.execute()

        opt = dialog.parent.user_settings.config.optimization
        assert opt.worker_count == 12
        assert opt.pool_prespawn is False
        assert opt.sensitivity_cache_mb == 100
        dialog.parent.user_settings.save.assert_called_once()

    def test_cancel_leaves_config_untouched(self, dialog, monkeypatch):
        def user_edits_then_cancel():
            dialog.cache_mb_box.setValue(100)

            return 0

        monkeypatch.setattr(dialog, "exec_", user_edits_then_cancel)

        dialog.execute()

        opt = dialog.parent.user_settings.config.optimization
        assert opt.sensitivity_cache_mb == 250
        dialog.parent.user_settings.save.assert_not_called()

    def test_auto_checkbox_clears_worker_override(self, dialog, monkeypatch):
        opt = dialog.parent.user_settings.config.optimization
        opt.worker_count = 8

        def user_checks_auto_then_ok():
            dialog.auto_workers_box.setChecked(True)

            return 1

        monkeypatch.setattr(dialog, "exec_", user_checks_auto_then_ok)

        dialog.execute()

        assert opt.worker_count is None


class TestCacheControls:
    def test_clear_button_empties_shared_cache(self, dialog):
        shared_cache.put(("t",), (np.zeros(16),))
        dialog._load_from_config()
        dialog._clear_cache()
        assert len(shared_cache) == 0
        assert dialog.cache_usage_label.text().startswith("In use: 0.0 MB")

    def test_working_dir_shows_appdata(self, dialog, stub_parent):
        dialog._load_from_config()
        assert Path(dialog.working_dir_box.text()) == stub_parent.path["appdata"]
