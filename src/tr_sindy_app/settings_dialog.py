"""Settings dialog for TR-SINDy.

Exposes user-configurable preferences (mmap directory, default backend,
theme, recent-file count, FFmpeg path) backed by ``QSettings`` so they
persist across sessions.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

DEFAULTS = {
    "mmap_dir": "./velocity_mmaps",
    "default_backend": "farneback",
    "theme": "dark",
    "recent_count": 10,
    "ffmpeg_path": "",
    "auto_save": True,
}


def load_settings() -> dict:
    """Load settings from QSettings, falling back to defaults."""
    qs = QtCore.QSettings("TurbulenceRealm", "TR-SINDy")
    return {key: qs.value(key, default, type=str if isinstance(default, str)
                          else type(default))
            for key, default in DEFAULTS.items()}


def save_settings(settings: dict) -> None:
    """Persist settings via QSettings."""
    qs = QtCore.QSettings("TurbulenceRealm", "TR-SINDy")
    for key, val in settings.items():
        qs.setValue(key, val)


class SettingsDialog(QtWidgets.QDialog):
    """Modal dialog for editing application settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(420)
        self._settings = load_settings()
        layout = QtWidgets.QFormLayout(self)

        # mmap directory
        mmap_row = QtWidgets.QHBoxLayout()
        self.mmap_edit = QtWidgets.QLineEdit(self._settings["mmap_dir"])
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_mmap)
        mmap_row.addWidget(self.mmap_edit)
        mmap_row.addWidget(browse_btn)
        layout.addRow("Mmap directory:", _wrap(mmap_row))

        # default backend
        from .optical_flow import available_backends
        self.backend_combo = QtWidgets.QComboBox()
        for b in available_backends():
            self.backend_combo.addItem(b)
        self.backend_combo.setCurrentText(self._settings["default_backend"])
        layout.addRow("Default backend:", self.backend_combo)

        # theme
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(["dark", "light"])
        self.theme_combo.setCurrentText(self._settings["theme"])
        layout.addRow("Theme:", self.theme_combo)

        # recent count
        self.recent_spin = QtWidgets.QSpinBox()
        self.recent_spin.setRange(1, 50)
        self.recent_spin.setValue(int(self._settings["recent_count"]))
        layout.addRow("Recent files count:", self.recent_spin)

        # ffmpeg path
        ffmpeg_row = QtWidgets.QHBoxLayout()
        self.ffmpeg_edit = QtWidgets.QLineEdit(self._settings["ffmpeg_path"])
        self.ffmpeg_edit.setPlaceholderText("auto-detect (leave empty)")
        ff_browse = QtWidgets.QPushButton("Browse…")
        ff_browse.clicked.connect(self._browse_ffmpeg)
        ffmpeg_row.addWidget(self.ffmpeg_edit)
        ffmpeg_row.addWidget(ff_browse)
        layout.addRow("FFmpeg path:", _wrap(ffmpeg_row))

        # auto-save
        self.autosave_chk = QtWidgets.QCheckBox("Enable auto-save / session restore")
        self.autosave_chk.setChecked(bool(self._settings["auto_save"]))
        layout.addRow("", self.autosave_chk)

        # buttons
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addRow(bb)

    def _browse_mmap(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Mmap directory")
        if d:
            self.mmap_edit.setText(d)

    def _browse_ffmpeg(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "FFmpeg executable")
        if f:
            self.ffmpeg_edit.setText(f)

    def get_settings(self) -> dict:
        return {
            "mmap_dir": self.mmap_edit.text(),
            "default_backend": self.backend_combo.currentText(),
            "theme": self.theme_combo.currentText(),
            "recent_count": self.recent_spin.value(),
            "ffmpeg_path": self.ffmpeg_edit.text(),
            "auto_save": self.autosave_chk.isChecked(),
        }


def _wrap(layout) -> QtWidgets.QWidget:
    w = QtWidgets.QWidget()
    w.setLayout(layout)
    return w
