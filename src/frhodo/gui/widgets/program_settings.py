# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level
# directory for license and copyright information.
"""Program-level settings dialog (File > Settings).

Surfaces machine/installation preferences that don't belong on the
per-run options panel: worker-pool sizing, pool pre-spawn, the shared
sensitivity-cache cap, and the working (appdata) directory. Values
persist through ``FrhodoConfig`` on OK.
"""
from qtpy.QtCore import QUrl
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from frhodo.optimize.pool import default_worker_count
from frhodo.optimize.sensitivity_cache import shared_cache



class ProgramSettingsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Settings")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.addWidget(self._performance_group())
        layout.addWidget(self._cache_group())
        layout.addWidget(self._paths_group())

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _performance_group(self):
        group = QGroupBox("Performance")
        grid = QGridLayout(group)

        self.auto_workers_box = QCheckBox(
            f"Auto ({default_worker_count()} for this machine)"
        )
        self.auto_workers_box.setToolTip(
            "Size the optimization worker pool from the CPU"
        )
        self.worker_count_box = QSpinBox()
        self.worker_count_box.setRange(1, 128)
        self.auto_workers_box.toggled.connect(
            lambda checked: self.worker_count_box.setEnabled(not checked)
        )
        grid.addWidget(QLabel("Worker processes:"), 0, 0)
        grid.addWidget(self.auto_workers_box, 0, 1)
        grid.addWidget(self.worker_count_box, 0, 2)

        self.prespawn_box = QCheckBox(
            "Start worker pool in the background on mechanism load"
        )
        self.prespawn_box.setToolTip(
            "Hides worker startup behind run setup; takes effect on the "
            "next mechanism load"
        )
        grid.addWidget(self.prespawn_box, 1, 0, 1, 3)

        return group

    def _cache_group(self):
        group = QGroupBox("Sensitivity cache")
        grid = QGridLayout(group)

        self.cache_mb_box = QSpinBox()
        self.cache_mb_box.setRange(10, 10_000)
        self.cache_mb_box.setSuffix(" MB")
        self.cache_mb_box.setToolTip(
            "Memory cap on cached start-mechanism solves and "
            "sensitivities (screening, weighting, Sim Explorer)"
        )
        grid.addWidget(QLabel("Size limit:"), 0, 0)
        grid.addWidget(self.cache_mb_box, 0, 1)

        self.cache_usage_label = QLabel()
        clear_button = QPushButton("Clear cache")
        clear_button.clicked.connect(self._clear_cache)
        grid.addWidget(self.cache_usage_label, 1, 0, 1, 2)
        grid.addWidget(clear_button, 1, 2)

        return group

    def _paths_group(self):
        group = QGroupBox("Paths")
        row = QHBoxLayout(group)

        self.working_dir_box = QLineEdit()
        self.working_dir_box.setReadOnly(True)
        self.working_dir_box.setToolTip(
            "Config, generated mechanisms, logs, and autosnapshots "
            "live here"
        )
        open_button = QPushButton("Open")
        open_button.clicked.connect(self._open_working_dir)
        row.addWidget(QLabel("Working directory:"))
        row.addWidget(self.working_dir_box, stretch=1)
        row.addWidget(open_button)

        return group

    def _load_from_config(self):
        self.working_dir_box.setText(str(self.parent.path["appdata"]))

        opt = self.parent.user_settings.config.optimization
        self.auto_workers_box.setChecked(opt.worker_count is None)
        self.worker_count_box.setEnabled(opt.worker_count is not None)
        self.worker_count_box.setValue(
            opt.worker_count or default_worker_count()
        )
        self.prespawn_box.setChecked(opt.pool_prespawn)
        self.cache_mb_box.setValue(opt.sensitivity_cache_mb)
        self._refresh_cache_usage()

    def _refresh_cache_usage(self):
        used_mb = shared_cache.nbytes / 2**20
        self.cache_usage_label.setText(
            f"In use: {used_mb:.1f} MB ({len(shared_cache)} entries)"
        )

    def _clear_cache(self):
        shared_cache.clear()
        self._refresh_cache_usage()

    def _open_working_dir(self):
        QDesktopServices.openUrl(
            QUrl.fromLocalFile(self.working_dir_box.text())
        )

    def execute(self):
        self._load_from_config()
        if not self.exec_():
            return

        opt = self.parent.user_settings.config.optimization
        if self.auto_workers_box.isChecked():
            opt.worker_count = None
        else:
            opt.worker_count = int(self.worker_count_box.value())
        opt.pool_prespawn = self.prespawn_box.isChecked()
        opt.sensitivity_cache_mb = int(self.cache_mb_box.value())
        shared_cache.set_max_bytes(opt.sensitivity_cache_mb * 2**20)
        self.parent.user_settings.save()
