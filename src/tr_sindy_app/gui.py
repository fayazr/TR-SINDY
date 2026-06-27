"""Main application window for Turbulence Realm - SINDy (v2.1 layout).

Layout redesign: a left navigation rail switches between functional pages;
each page is a horizontal split with a scrollable control panel on the left
and a content/preview area on the right. A visual pipeline stepper in the
setup page shows workflow progress at all times.

Integrates the modular backends (optical_flow, sindy_core, ml_models,
analysis, quality, export, project).
"""

from __future__ import annotations

import json
import os
import types

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from scipy.ndimage import gaussian_filter

# Light modules needed during construction — imported eagerly.
from . import optical_flow, project
from ._logging import get_logger
from .roi_dialog import ROICalibDialog
from .theme import Theme, apply_matplotlib_theme, stylesheet

log = get_logger(__name__)

# Heavy modules (pysindy ~1.7s, pandas ~1s) — imported lazily on first use
# to keep application startup fast.  Accessed via self._sindy_core etc.
_sindy_core = None
_analysis = None
_quality = None
_export = None
_ml_models = None


def _get_sindy_core():
    global _sindy_core
    if _sindy_core is None:
        from . import sindy_core as _m
        _sindy_core = _m
    return _sindy_core


def _get_analysis():
    global _analysis
    if _analysis is None:
        from . import analysis as _m
        _analysis = _m
    return _analysis


def _get_quality():
    global _quality
    if _quality is None:
        from . import quality as _m
        _quality = _m
    return _quality


def _get_export():
    global _export
    if _export is None:
        from . import export as _m
        _export = _m
    return _export


def _get_ml_models():
    global _ml_models
    if _ml_models is None:
        from . import ml_models as _m
        _ml_models = _m
    return _ml_models

COLORMAPS = ["turbo", "viridis", "plasma", "inferno", "magma", "cividis",
             "RdBu_r", "coolwarm", "jet", "gray"]

NAV_PAGES = ["Setup", "Visualize", "ML Models", "Export"]


def _info_dialog(parent, title, text):
    QtWidgets.QMessageBox.information(parent, title, text)


def _warn(parent, title, text):
    QtWidgets.QMessageBox.warning(parent, title, text)


def _err(parent, title, text):
    QtWidgets.QMessageBox.critical(parent, title, text)


# ---------------------------------------------------------------------
#  Pipeline stepper widget
# ---------------------------------------------------------------------
class PipelineStepper(QtWidgets.QWidget):
    """Vertical stepper showing the 5 pipeline stages with state dots."""

    STEPS = [
        ("01", "Open Video"),
        ("02", "ROI & Calibrate"),
        ("03", "Optical Flow"),
        ("04", "SINDy Model"),
        ("05", "Predict & Analyse"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("stepper")
        self._states = ["pending"] * len(self.STEPS)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(0)
        title = QtWidgets.QLabel("PIPELINE")
        title.setObjectName("stepperTitle")
        v.addWidget(title)
        v.addSpacing(12)
        self._dots = []
        self._labels = []
        self._bars = []
        for i, (num, name) in enumerate(self.STEPS):
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(12)
            dot = QtWidgets.QLabel("○")
            dot.setFixedSize(22, 22)
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dot.setObjectName("stepperDot")
            lbl = QtWidgets.QLabel(f"{num}  {name}")
            lbl.setObjectName("stepperLabel")
            row.addWidget(dot)
            row.addWidget(lbl, 1)
            v.addLayout(row)
            self._dots.append(dot)
            self._labels.append(lbl)
            if i < len(self.STEPS) - 1:
                bar = QtWidgets.QLabel("│")
                bar.setFixedWidth(22)
                bar.setAlignment(Qt.AlignmentFlag.AlignHCenter)
                bar.setObjectName("stepperBar")
                v.addWidget(bar)
                self._bars.append(bar)
        v.addStretch(1)
        self._refresh()

    def set_state(self, index: int, state: str):
        """state: pending | active | done"""
        if 0 <= index < len(self._states):
            self._states[index] = state
            self._refresh()

    def _refresh(self):
        colors = {"pending": Theme.TEXT_FAINT, "active": Theme.ACCENT, "done": Theme.GOOD}
        glyphs = {"pending": "○", "active": "◐", "done": "●"}
        for i, st in enumerate(self._states):
            c = colors.get(st, Theme.TEXT_FAINT)
            self._dots[i].setText(glyphs.get(st, "○"))
            # Glowing dot for active/done states
            if st == "active":
                self._dots[i].setStyleSheet(
                    f"color: {c}; font-size: 16pt; background: transparent;"
                    f" font-weight: bold;")
            elif st == "done":
                self._dots[i].setStyleSheet(
                    f"color: {c}; font-size: 16pt; background: transparent;"
                    f" font-weight: bold;")
            else:
                self._dots[i].setStyleSheet(
                    f"color: {c}; font-size: 14pt; background: transparent;")
            label_color = c if st != "pending" else Theme.TEXT_MUTED
            weight = "700" if st in ("active", "done") else "500"
            self._labels[i].setStyleSheet(
                f"color: {label_color}; font-size: 9pt; background: transparent;"
                f" font-weight: {weight};")
        # Color the connector bars based on completion
        for i, bar in enumerate(self._bars):
            if i < len(self._states) - 1:
                if self._states[i] == "done" and self._states[i + 1] != "pending":
                    bar.setStyleSheet(
                        f"color: {Theme.GOOD}; background: transparent; font-weight: bold;")
                elif self._states[i] == "done":
                    bar.setStyleSheet(
                        f"color: {Theme.ACCENT}; background: transparent;")
                else:
                    bar.setStyleSheet(
                        f"color: {Theme.HAIRLINE}; background: transparent;")


# ---------------------------------------------------------------------
#  Main window
# ---------------------------------------------------------------------
class FluidGui(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Turbulence Realm — SINDy")
        self.setMinimumSize(1440, 900)
        self.resize(1560, 940)
        if os.path.exists("logo.png"):
            self.setWindowIcon(QtGui.QIcon("logo.png"))

        apply_matplotlib_theme()
        self._theme_name = "dark"

        # state
        self.result = {}
        self.of_result = {}
        self.sindy_result = {}
        self.finished = False
        self.X_pred_optical = None
        self.history = project.ProcessingHistory()
        self._mmap_dir = "./velocity_mmaps"

        # ---- central: glass background + nav rail + stacked pages ----
        central = QtWidgets.QWidget()
        central.setObjectName("contentRoot")
        self.setCentralWidget(central)
        # We need a container that holds the glass background behind everything
        outer = QtWidgets.QGridLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Glass background (bottom layer — paints gradient + glowing orbs)
        from .glass_background import GlassBackground
        self._glass_bg = GlassBackground(central)
        outer.addWidget(self._glass_bg, 0, 0, 1, 2)

        # Content container (top layer — transparent so background shows through)
        content_container = QtWidgets.QWidget()
        content_container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        content_layout = QtWidgets.QHBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        content_layout.addWidget(self._build_nav_rail())

        self.pages = QtWidgets.QStackedWidget()
        content_layout.addWidget(self.pages, 1)

        outer.addWidget(content_container, 0, 0, 1, 2)

        self._build_setup_page()
        self._build_visualize_page()
        self._build_ml_page()
        self._build_export_page()

        self.status_bar = QtWidgets.QStatusBar()
        self.status_bar.showMessage("Ready")
        self.setStatusBar(self.status_bar)

        self._build_menubar()
        self.setStyleSheet(stylesheet())
        project.install_builtin_presets()
        self._refresh_recent_menu()
        self._select_nav(0)

        # Offer to restore the previous session (non-blocking).
        QtCore.QTimer.singleShot(500, self._restore_session)

    # =================================================================
    #  Navigation rail
    # =================================================================
    def _build_nav_rail(self):
        rail = QtWidgets.QFrame()
        rail.setObjectName("navRail")
        rail.setFixedWidth(220)
        v = QtWidgets.QVBoxLayout(rail)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # brand block
        brand = QtWidgets.QFrame()
        brand.setObjectName("navBrand")
        bv = QtWidgets.QVBoxLayout(brand)
        bv.setContentsMargins(20, 22, 20, 20)
        bv.setSpacing(3)
        mark = QtWidgets.QLabel("⬡")
        mark.setStyleSheet(
            f"color: {Theme.ACCENT}; font-size: 26pt; background: transparent;"
            f" font-weight: bold;")
        bv.addWidget(mark)
        title = QtWidgets.QLabel(
            f"Turbulence <span style='color:{Theme.ACCENT}'>Realm</span>")
        title.setObjectName("navTitle")
        title.setTextFormat(Qt.TextFormat.RichText)
        bv.addWidget(title)
        sub = QtWidgets.QLabel("· SINDy")
        sub.setObjectName("navSubtitle")
        bv.addWidget(sub)
        v.addWidget(brand)

        # nav buttons
        self.nav_buttons = []
        for i, name in enumerate(NAV_PAGES):
            b = QtWidgets.QPushButton(name)
            b.setObjectName("navButton")
            b.setCheckable(True)
            b.setCursor(QtGui.QCursor(Qt.CursorShape.PointingHandCursor))
            b.clicked.connect(lambda _=False, idx=i: self._select_nav(idx))
            v.addWidget(b)
            self.nav_buttons.append(b)

        v.addStretch(1)

        # bottom controls
        bot = QtWidgets.QVBoxLayout()
        bot.setContentsMargins(14, 8, 14, 16)
        bot.setSpacing(8)
        self.theme_btn = QtWidgets.QPushButton("☾  Light Theme")
        self.theme_btn.setObjectName("navAux")
        self.theme_btn.setToolTip("Toggle dark / light theme (Ctrl+T)")
        self.theme_btn.clicked.connect(self._toggle_theme)
        bot.addWidget(self.theme_btn)
        self.about_btn = QtWidgets.QPushButton("ⓘ  About")
        self.about_btn.setObjectName("navAux")
        self.about_btn.clicked.connect(self.show_about_dialog)
        bot.addWidget(self.about_btn)
        v.addLayout(bot)
        return rail

    def _select_nav(self, idx):
        self.pages.setCurrentIndex(idx)
        for i, b in enumerate(self.nav_buttons):
            b.setChecked(i == idx)

    def _set_nav_badge(self, page_idx: int, text: str):
        """Append a status marker to a nav button label."""
        base = NAV_PAGES[page_idx]
        self.nav_buttons[page_idx].setText(f"{base}  {text}" if text else base)

    # =================================================================
    #  Header pill (status indicator shown in status bar area)
    # =================================================================
    def _set_pill(self, text, state="idle"):
        colors = {
            "idle":  (Theme.TEXT_MUTED, "rgba(139,161,192,0.10)", "rgba(139,161,192,0.30)"),
            "ready": (Theme.ACCENT,     Theme.GLOW_ACCENT,        "rgba(34,211,238,0.40)"),
            "busy":  (Theme.AMBER,      Theme.GLOW_AMBER,         "rgba(251,191,36,0.40)"),
            "done":  (Theme.GOOD,       Theme.GLOW_GOOD,          "rgba(52,211,153,0.40)"),
        }
        c, bg, br = colors.get(state, colors["idle"])
        pill = QtWidgets.QLabel(text)
        pill.setStyleSheet(
            f"color: {c}; background-color: {bg}; border: 1px solid {br};"
            f" border-radius: 11px; padding: 3px 12px; font-family: '{Theme.MONO_FONT}';"
            f" font-size: 8pt; font-weight: bold; letter-spacing: 1px;")
        self.status_bar.clearMessage()
        self.status_bar.addWidget(pill)
        self._current_pill = pill
        QtWidgets.QApplication.processEvents()

    def _toggle_theme(self):
        self._theme_name = "light" if self._theme_name == "dark" else "dark"
        from .theme import apply_theme
        apply_theme(self._theme_name)
        apply_matplotlib_theme()
        self.setStyleSheet(stylesheet())
        self.theme_btn.setText("☀  Dark Theme" if self._theme_name == "light" else "☾  Light Theme")
        self.stepper._refresh()
        self._refresh_plots_theme()

    def _refresh_plots_theme(self):
        for fig in getattr(self, "_figs", []):
            try:
                fig.patch.set_facecolor(Theme.SURFACE)
                for ax in fig.axes:
                    ax.set_facecolor(Theme.BG_BASE)
                fig.canvas.draw_idle()
            except Exception:
                pass

    # =================================================================
    #  Menubar + shortcuts
    # =================================================================
    def _build_menubar(self):
        mb = self.menuBar()
        file_menu = mb.addMenu("&File")
        a_open = QAction("Open Video…", self)
        a_open.setShortcut(QKeySequence("Ctrl+O"))
        a_open.triggered.connect(self.pick_file)
        file_menu.addAction(a_open)
        self.recent_menu = file_menu.addMenu("Recent Projects")
        a_save = QAction("Save Project…", self)
        a_save.setShortcut(QKeySequence("Ctrl+S"))
        a_save.triggered.connect(self.save_project)
        file_menu.addAction(a_save)
        a_load = QAction("Load Project…", self)
        a_load.setShortcut(QKeySequence("Ctrl+L"))
        a_load.triggered.connect(self.load_project)
        file_menu.addAction(a_load)
        file_menu.addSeparator()
        a_quit = QAction("Quit", self)
        a_quit.setShortcut(QKeySequence("Ctrl+Q"))
        a_quit.triggered.connect(self.close)
        file_menu.addAction(a_quit)

        preset_menu = mb.addMenu("&Presets")
        a_list = QAction("List / Apply Preset…", self)
        a_list.triggered.connect(self.apply_preset)
        preset_menu.addAction(a_list)
        a_save_p = QAction("Save Current as Preset…", self)
        a_save_p.triggered.connect(self.save_preset)
        preset_menu.addAction(a_save_p)

        view_menu = mb.addMenu("&View")
        for i, name in enumerate(NAV_PAGES):
            a = QAction(name, self)
            a.triggered.connect(lambda _=False, idx=i: self._select_nav(idx))
            view_menu.addAction(a)
        view_menu.addSeparator()
        a_theme = QAction("Toggle Theme", self)
        a_theme.setShortcut(QKeySequence("Ctrl+T"))
        a_theme.triggered.connect(self._toggle_theme)
        view_menu.addAction(a_theme)

        edit_menu = mb.addMenu("&Edit")
        a_settings = QAction("Settings…", self)
        a_settings.triggered.connect(self.show_settings)
        edit_menu.addAction(a_settings)

        help_menu = mb.addMenu("&Help")
        a_help = QAction("Workflow Guide", self)
        a_help.setShortcut(QKeySequence("F1"))
        a_help.triggered.connect(self.show_help)
        help_menu.addAction(a_help)
        a_about = QAction("About", self)
        a_about.triggered.connect(self.show_about_dialog)
        help_menu.addAction(a_about)

    def _refresh_recent_menu(self):
        self.recent_menu.clear()
        for p in project.load_recent():
            act = QAction(os.path.basename(p), self)
            act.setToolTip(p)
            act.triggered.connect(lambda _=False, path=p: self.load_project(path))
            self.recent_menu.addAction(act)

    # =================================================================
    #  Page builders — each is a horizontal split: controls | content
    # =================================================================
    def _make_split_page(self, controls_widget, content_widget,
                         controls_min=480, controls_max=720):
        """Wrap a page as a splitter with scrollable controls on the left."""
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls_widget)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setMinimumWidth(controls_min)
        scroll.setMaximumWidth(controls_max)
        # Make the scroll area + viewport transparent so glass background shows through
        scroll.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        scroll.viewport().setAutoFillBackground(False)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        split = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(scroll)
        split.addWidget(content_widget)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setHandleWidth(2)
        # Make splitter transparent so glass background shows through
        split.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # Give the controls column a comfortable default width
        split.setSizes([640, 800])
        return split

    def _controls_panel(self):
        """A scrollable-friendly container widget for a control column."""
        w = QtWidgets.QWidget()
        w.setObjectName("controlsPanel")
        return w

    def _apply_glass_effects(self, widget):
        """Apply drop shadows to all groupboxes and cards in a widget tree.

        This creates the depth that makes glassmorphism visible — without
        shadows, translucent surfaces blend into the background.
        """
        from .glass_background import apply_drop_shadow
        for child in widget.findChildren(QtWidgets.QGroupBox):
            apply_drop_shadow(child, blur_radius=24, y_offset=6, alpha=100)
        for child in widget.findChildren(QtWidgets.QFrame):
            if child.objectName() == "card":
                apply_drop_shadow(child, blur_radius=20, y_offset=4, alpha=80)

    # -----------------------------------------------------------------
    #  Setup page
    # -----------------------------------------------------------------
    def _build_setup_page(self):
        controls = self._controls_panel()
        cl = QtWidgets.QVBoxLayout(controls)
        cl.setContentsMargins(12, 14, 12, 14)
        cl.setSpacing(12)

        # Video source
        fg = QtWidgets.QGroupBox("VIDEO SOURCE")
        fl = QtWidgets.QHBoxLayout(fg)
        fl.setContentsMargins(8, 8, 8, 8)
        fl.setSpacing(8)
        self.file_edit = QtWidgets.QLineEdit()
        self.file_edit.setPlaceholderText("No video file selected…")
        self.file_btn = QtWidgets.QPushButton("Browse…")
        self.file_btn.setFixedWidth(100)
        self.file_btn.clicked.connect(self.pick_file)
        fl.addWidget(self.file_edit, 1); fl.addWidget(self.file_btn)
        cl.addWidget(fg)

        # ROI
        rg = QtWidgets.QGroupBox("ROI & CALIBRATION")
        rl = QtWidgets.QVBoxLayout(rg)
        self.run_btn = QtWidgets.QPushButton("Select ROI & Calibrate")
        self.run_btn.setProperty("variant", "primary")
        self.run_btn.setToolTip("Draw a rectangle ROI, then a calibration line of known length.")
        self.run_btn.clicked.connect(self.start_roi)
        self.mpp_label = QtWidgets.QLabel("Calibration:  — px = — m    ·    1 px = — m")
        self.mpp_label.setObjectName("readout")
        self.mpp_label.setWordWrap(True)
        rl.addWidget(self.run_btn); rl.addWidget(self.mpp_label)
        cl.addWidget(rg)

        # Optical flow config
        ofg = QtWidgets.QGroupBox("OPTICAL FLOW")
        ofgl = QtWidgets.QGridLayout(ofg)
        ofgl.setSpacing(8)
        ofgl.addWidget(QtWidgets.QLabel("Backend:"), 0, 0)
        self.backend_combo = QtWidgets.QComboBox()
        for b in optical_flow.available_backends():
            self.backend_combo.addItem(b)
        self.backend_combo.setToolTip(
            "farneback: classic dense flow (default)\n"
            "lucas_kanade: sparse feature tracking, interpolated\n"
            "tvl1: total-variation regularised (needs opencv-contrib)\n"
            "raft / pwcnet: deep learning (needs torch+torchvision)")
        ofgl.addWidget(self.backend_combo, 0, 1)
        ofgl.addWidget(QtWidgets.QLabel("Smoothing:"), 1, 0)
        self.ts_combo = QtWidgets.QComboBox()
        self.ts_combo.addItems(["none", "ema", "moving"])
        self.ts_combo.setToolTip("Reduces flicker between frames.")
        ofgl.addWidget(self.ts_combo, 1, 1)
        ofgl.addWidget(QtWidgets.QLabel("EMA α:"), 1, 2)
        self.alpha_spin = QtWidgets.QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0); self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(0.6)
        ofgl.addWidget(self.alpha_spin, 1, 3)
        self.multiscale_chk = QtWidgets.QCheckBox("Multi-scale pyramid")
        self.multiscale_chk.setToolTip("Coarse-to-fine estimation for large motions.")
        ofgl.addWidget(self.multiscale_chk, 2, 0, 1, 2)
        self.gauss_chk = QtWidgets.QCheckBox("Gaussian denoise")
        self.nlm_chk = QtWidgets.QCheckBox("NLM denoise")
        ofgl.addWidget(self.gauss_chk, 2, 2)
        ofgl.addWidget(self.nlm_chk, 2, 3)
        self.quality_chk = QtWidgets.QCheckBox("Compute quality metrics")
        self.quality_chk.setToolTip("Forward-backward consistency error per frame.")
        ofgl.addWidget(self.quality_chk, 3, 0, 1, 2)
        cl.addWidget(ofg)

        # SINDy config
        sg = QtWidgets.QGroupBox("SINDy MODEL")
        sgl = QtWidgets.QGridLayout(sg)
        sgl.setSpacing(8)
        sgl.addWidget(QtWidgets.QLabel("Library:"), 0, 0)
        self.lib_combo = QtWidgets.QComboBox()
        self.lib_combo.addItems(["polynomial", "fourier", "combined", "custom", "trig"])
        self.lib_combo.setToolTip(
            "polynomial: PolynomialLibrary(degree)\n"
            "fourier/trig: FourierLibrary (periodic flows)\n"
            "combined: polynomial × Fourier tensor product\n"
            "custom: user-defined nonlinear terms")
        sgl.addWidget(self.lib_combo, 0, 1)
        sgl.addWidget(QtWidgets.QLabel("Degree:"), 0, 2)
        self.degree_spin = QtWidgets.QSpinBox()
        self.degree_spin.setRange(1, 5); self.degree_spin.setValue(3)
        sgl.addWidget(self.degree_spin, 0, 3)
        sgl.addWidget(QtWidgets.QLabel("Optimizer:"), 1, 0)
        self.opt_combo = QtWidgets.QComboBox()
        self.opt_combo.addItems(["stlsq", "sr3", "frols", "constrained_sr3"])
        sgl.addWidget(self.opt_combo, 1, 1)
        sgl.addWidget(QtWidgets.QLabel("Threshold:"), 1, 2)
        self.thresh_spin = QtWidgets.QDoubleSpinBox()
        self.thresh_spin.setRange(0.001, 1.0); self.thresh_spin.setSingleStep(0.01)
        self.thresh_spin.setValue(0.07); self.thresh_spin.setDecimals(3)
        sgl.addWidget(self.thresh_spin, 1, 3)
        self.divfree_chk = QtWidgets.QCheckBox("Divergence-free constraint")
        self.divfree_chk.setToolTip("Enforce incompressibility (∇·u = 0).")
        sgl.addWidget(self.divfree_chk, 2, 0, 1, 4)
        cl.addWidget(sg)

        # Pipeline actions
        pg = QtWidgets.QGroupBox("PIPELINE")
        pgl = QtWidgets.QVBoxLayout(pg)
        pgl.setSpacing(8)
        self.process_btn = QtWidgets.QPushButton("① Process Optical Flow")
        self.process_btn.setProperty("variant", "primary")
        self.process_btn.clicked.connect(self.run_optical_flow_gui)
        self.process_btn.setEnabled(False)
        pgl.addWidget(self.process_btn)
        self.sindy_btn = QtWidgets.QPushButton("② Run SINDy Modeling")
        self.sindy_btn.setProperty("variant", "primary")
        self.sindy_btn.clicked.connect(self.run_sindy_gui)
        self.sindy_btn.setEnabled(False)
        pgl.addWidget(self.sindy_btn)
        self.pred_btn = QtWidgets.QPushButton("③ Run SINDy Prediction")
        self.pred_btn.setProperty("variant", "primary")
        self.pred_btn.clicked.connect(self.run_prediction_gui)
        self.pred_btn.setEnabled(False)
        pgl.addWidget(self.pred_btn)
        extras = QtWidgets.QHBoxLayout()
        self.sindy_eq_btn = QtWidgets.QPushButton("Equation")
        self.sindy_eq_btn.setProperty("variant", "violet")
        self.sindy_eq_btn.clicked.connect(self.show_sindy_equation_gui)
        self.sindy_eq_btn.setEnabled(False)
        extras.addWidget(self.sindy_eq_btn)
        self.cv_btn = QtWidgets.QPushButton("Cross-validate")
        self.cv_btn.setToolTip("k-fold cross-validation of the SINDy model.")
        self.cv_btn.clicked.connect(self.run_cross_validation)
        self.cv_btn.setEnabled(False)
        extras.addWidget(self.cv_btn)
        self.compare_btn = QtWidgets.QPushButton("Compare")
        self.compare_btn.setToolTip("Fit several libraries and compare RMSE / complexity.")
        self.compare_btn.clicked.connect(self.run_model_comparison)
        self.compare_btn.setEnabled(False)
        extras.addWidget(self.compare_btn)
        pgl.addLayout(extras)
        cl.addWidget(pg)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setValue(0); self.progress.setMaximum(100)
        self.progress.hide()
        cl.addWidget(self.progress)
        cl.addStretch(1)

        # ---- content (right) ----
        content = QtWidgets.QWidget()
        ct = QtWidgets.QVBoxLayout(content)
        ct.setContentsMargins(18, 18, 18, 18)
        ct.setSpacing(14)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(14)
        self.stepper = PipelineStepper()
        top.addWidget(self.stepper)
        guide_wrap = QtWidgets.QFrame()
        guide_wrap.setObjectName("card")
        gl = QtWidgets.QVBoxLayout(guide_wrap)
        gl.setContentsMargins(16, 16, 16, 16)
        self.status = QtWidgets.QLabel(
            "01  ·  Choose a video file\n"
            "02  ·  Select ROI & calibrate scale\n"
            "03  ·  Configure & process dense optical flow\n"
            "04  ·  Fit the SINDy model (optionally cross-validate)\n"
            "05  ·  Predict, analyse, visualise & export")
        self.status.setObjectName("stepGuide")
        self.status.setWordWrap(True)
        gl.addWidget(self.status)
        top.addWidget(guide_wrap, 1)
        ct.addLayout(top)

        preview_card = QtWidgets.QGroupBox("LIVE OPTICAL-FLOW PREVIEW")
        pv = QtWidgets.QVBoxLayout(preview_card)
        self.of_preview = QtWidgets.QLabel("Run optical flow to see the live HSV + quiver field here.")
        self.of_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.of_preview.setMinimumHeight(320)
        self.of_preview.setStyleSheet(
            f"background-color: {Theme.BG_BASE}; color: {Theme.TEXT_FAINT};"
            f" border: 1px solid {Theme.HAIRLINE}; border-radius: 12px;"
            f" font-family: '{Theme.MONO_FONT}'; font-size: 9pt; padding: 20px;")
        self.of_preview.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                      QtWidgets.QSizePolicy.Policy.Expanding)
        pv.addWidget(self.of_preview)
        ct.addWidget(preview_card, 1)

        page = self._make_split_page(controls, content); self._apply_glass_effects(page); self.pages.addWidget(page)

    # -----------------------------------------------------------------
    #  Visualize page
    # -----------------------------------------------------------------
    def _build_visualize_page(self):
        controls = self._controls_panel()
        cl = QtWidgets.QVBoxLayout(controls)
        cl.setContentsMargins(18, 18, 18, 18)
        cl.setSpacing(12)

        ag = QtWidgets.QGroupBox("ANALYSIS")
        agl = QtWidgets.QGridLayout(ag)
        agl.setSpacing(8)
        self.quiver_btn = self._mk_btn("Quiver (split)", agl, 0, 0, self.show_quiver_section)
        self.contour_btn = self._mk_btn("Contour", agl, 0, 1, self.plot_contour_gui)
        self.stream_btn = self._mk_btn("Streamlines", agl, 0, 2, self.plot_stream_gui)
        self.error_btn = self._mk_btn("Error Analysis", agl, 1, 0, self.run_error_analysis_gui)
        self.vort_btn = self._mk_btn("Vorticity", agl, 1, 1, self.plot_vorticity_gui)
        self.strain_btn = self._mk_btn("Strain Rate", agl, 1, 2, self.plot_strain_gui)
        self.pod_btn = self._mk_btn("POD Modes", agl, 2, 0, self.plot_pod_gui)
        self.dmd_btn = self._mk_btn("DMD Modes", agl, 2, 1, self.plot_dmd_gui)
        self.spec_btn = self._mk_btn("Spectrum", agl, 2, 2, self.plot_spectral_gui)
        self.heatmap_btn = self._mk_btn("Animated Heatmap", agl, 3, 0, self.plot_animated_heatmap)
        self.turb_btn = self._mk_btn("Turbulence Stats", agl, 3, 1, self.show_turbulence_stats)
        self.datatable_btn = self._mk_btn("Data Table", agl, 3, 2, self.show_data_table)
        cl.addWidget(ag)

        cm = QtWidgets.QGroupBox("DISPLAY")
        cml = QtWidgets.QHBoxLayout(cm)
        cml.addWidget(QtWidgets.QLabel("Colormap:"))
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        cml.addWidget(self.cmap_combo, 1)
        cl.addWidget(cm)

        # embedded quiver controls
        qctl = QtWidgets.QGroupBox("FRAME SCRUBBER")
        ql = QtWidgets.QVBoxLayout(qctl)
        sl = QtWidgets.QHBoxLayout()
        sl.addWidget(QtWidgets.QLabel("Frame"))
        self.quiver_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.quiver_slider.setMinimum(1); self.quiver_slider.setMaximum(1)
        self.quiver_frame_label = QtWidgets.QLabel("1 / 1")
        self.quiver_frame_label.setObjectName("readout")
        self.quiver_frame_label.setFixedWidth(110)
        self.quiver_frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sl.addWidget(self.quiver_slider, 1); sl.addWidget(self.quiver_frame_label)
        ql.addLayout(sl)
        cl.addWidget(qctl)
        cl.addStretch(1)

        # ---- content (right): embedded quiver always visible ----
        content = QtWidgets.QWidget()
        ct = QtWidgets.QVBoxLayout(content)
        ct.setContentsMargins(0, 0, 0, 0)
        self.quiver_section = QtWidgets.QWidget()
        qsl = QtWidgets.QVBoxLayout(self.quiver_section)
        qsl.setContentsMargins(14, 14, 14, 14)
        qsl.setSpacing(8)
        head = QtWidgets.QLabel("LIVE QUIVER  ·  ACTUAL vs SINDy")
        head.setObjectName("contentHeader")
        qsl.addWidget(head)
        self.quiver_fig = plt.Figure(figsize=(10, 5), tight_layout=True)
        self.quiver_fig.patch.set_facecolor(Theme.SURFACE)
        self.quiver_ax1 = self.quiver_fig.add_subplot(121)
        self.quiver_ax2 = self.quiver_fig.add_subplot(122)
        self.quiver_canvas = FigureCanvas(self.quiver_fig)
        qsl.addWidget(self.quiver_canvas, 1)
        self.quiver_toolbar = NavigationToolbar(self.quiver_canvas, self.quiver_section)
        qsl.addWidget(self.quiver_toolbar)
        ct.addWidget(self.quiver_section, 1)
        self._figs = [self.quiver_fig]

        page = self._make_split_page(controls, content); self._apply_glass_effects(page); self.pages.addWidget(page)

    def _mk_btn(self, label, grid, r, c, slot):
        b = QtWidgets.QPushButton(label)
        b.clicked.connect(slot)
        b.setEnabled(False)
        grid.addWidget(b, r, c)
        return b

    def _enable_visualization(self):
        for b in (self.quiver_btn, self.contour_btn, self.stream_btn, self.error_btn,
                  self.vort_btn, self.strain_btn, self.pod_btn, self.dmd_btn,
                  self.spec_btn, self.heatmap_btn, self.turb_btn, self.datatable_btn):
            b.setEnabled(True)
        self._enable_export()
        self._set_nav_badge(1, "●")
        self.stepper.set_state(4, "done")

    def _enable_export(self):
        for b in (self.export_csv_btn, self.export_hdf5_btn, self.export_netcdf_btn,
                  self.export_parquet_btn, self.export_json_btn, self.export_pdf_btn,
                  self.export_imgs_btn, self.anim_btn):
            b.setEnabled(True)
        self._set_nav_badge(3, "●")

    # -----------------------------------------------------------------
    #  ML page
    # -----------------------------------------------------------------
    def _build_ml_page(self):
        controls = self._controls_panel()
        cl = QtWidgets.QVBoxLayout(controls)
        cl.setContentsMargins(18, 18, 18, 18)
        cl.setSpacing(12)

        info = QtWidgets.QLabel(
            "Select a model, pick a device/seed, adjust its hyper-parameters, "
            "then click Run. Predictions appear in the visualization tab on the "
            "right. After training, export the model to .pt / TorchScript / ONNX.")
        info.setObjectName("stepGuide")
        info.setWordWrap(True)
        cl.addWidget(info)

        # ---- model selector ----
        sg = QtWidgets.QGroupBox("MODEL")
        sgl = QtWidgets.QVBoxLayout(sg)
        sgl.setSpacing(8)
        self.ml_model_combo = QtWidgets.QComboBox()
        self.ml_models_list = [
            "PINN (Navier-Stokes)",
            "Autoencoder-SINDy",
            "Fourier Neural Operator",
            "DeepONet",
            "ConvLSTM Forecast",
            "VAE / beta-VAE",
            "Ensemble Uncertainty",
            "GAN Synthesis",
            "Causal / Granger",
        ]
        self.ml_model_combo.addItems(self.ml_models_list)
        self.ml_model_combo.currentIndexChanged.connect(self._ml_switch_params)
        sgl.addWidget(self.ml_model_combo)
        cl.addWidget(sg)

        # ---- shared compute controls (device + seed) ----
        cg = QtWidgets.QGroupBox("COMPUTE")
        cgl = QtWidgets.QGridLayout(cg)
        cgl.setSpacing(8)
        cgl.addWidget(QtWidgets.QLabel("Device:"), 0, 0)
        self.ml_device_combo = QtWidgets.QComboBox()
        self.ml_device_combo.addItems(["Auto", "CPU", "CUDA"])
        self.ml_device_combo.setToolTip(
            "Auto picks CUDA when available, else CPU.")
        cgl.addWidget(self.ml_device_combo, 0, 1)
        cgl.addWidget(QtWidgets.QLabel("Random seed:"), 1, 0)
        self.ml_seed_spin = QtWidgets.QSpinBox()
        self.ml_seed_spin.setRange(0, 2_147_483_647)
        self.ml_seed_spin.setValue(0)
        self.ml_seed_spin.setToolTip("Seeds Python / NumPy / PyTorch before each run.")
        cgl.addWidget(self.ml_seed_spin, 1, 1)
        self.ml_seed_chk = QtWidgets.QCheckBox("Use fixed seed (reproducible)")
        self.ml_seed_chk.setChecked(True)
        cgl.addWidget(self.ml_seed_chk, 2, 0, 1, 2)
        cl.addWidget(cg)

        # ---- dynamic parameter panel (stacked) ----
        self.ml_param_stack = QtWidgets.QStackedWidget()
        self.ml_param_widgets = {}
        # build a parameter page per model
        self._build_pinn_params()
        self._build_ae_sindy_params()
        self._build_fno_params()
        self._build_deeponet_params()
        self._build_convlstm_params()
        self._build_vae_params()
        self._build_ensemble_params()
        self._build_gan_params()
        self._build_causal_params()
        cl.addWidget(self.ml_param_stack)

        # ---- run button ----
        self.ml_run_btn = QtWidgets.QPushButton("▶  Train & Predict")
        self.ml_run_btn.setProperty("variant", "primary")
        self.ml_run_btn.setToolTip("Train the selected model with the parameters above.")
        self.ml_run_btn.clicked.connect(self._ml_run_selected)
        cl.addWidget(self.ml_run_btn)

        self.ml_progress = QtWidgets.QProgressBar()
        self.ml_progress.setMaximum(100)
        self.ml_progress.hide()
        cl.addWidget(self.ml_progress)

        # ---- export trained model ----
        xg = QtWidgets.QGroupBox("EXPORT TRAINED MODEL")
        xgl = QtWidgets.QVBoxLayout(xg)
        xgl.setSpacing(6)
        fmt_row = QtWidgets.QHBoxLayout()
        self.ml_fmt_pt = QtWidgets.QCheckBox("PyTorch .pt")
        self.ml_fmt_pt.setChecked(True)
        self.ml_fmt_ts = QtWidgets.QCheckBox("TorchScript")
        self.ml_fmt_onnx = QtWidgets.QCheckBox("ONNX")
        for c in (self.ml_fmt_pt, self.ml_fmt_ts, self.ml_fmt_onnx):
            fmt_row.addWidget(c)
        xgl.addLayout(fmt_row)
        self.ml_export_btn = QtWidgets.QPushButton("⏏  Export Model")
        self.ml_export_btn.setProperty("variant", "violet")
        self.ml_export_btn.setEnabled(False)
        self.ml_export_btn.setToolTip("Train a model first, then export its weights.")
        self.ml_export_btn.clicked.connect(self._ml_export_model)
        xgl.addWidget(self.ml_export_btn)
        cl.addWidget(xg)

        cl.addStretch(1)

        # ---- content (right): tabbed log + visualization ----
        content = QtWidgets.QWidget()
        ct = QtWidgets.QVBoxLayout(content)
        ct.setContentsMargins(0, 0, 0, 0)
        ct.setSpacing(0)
        self.ml_tabs = QtWidgets.QTabWidget()

        # tab 1: log
        log_tab = QtWidgets.QWidget()
        lt = QtWidgets.QVBoxLayout(log_tab)
        lt.setContentsMargins(14, 14, 14, 14)
        self.ml_log = QtWidgets.QPlainTextEdit()
        self.ml_log.setReadOnly(True)
        self.ml_log.setPlaceholderText("Training log will appear here…")
        lt.addWidget(self.ml_log)
        self.ml_tabs.addTab(log_tab, "Log")
        # Bridge the package logger into the in-app log widget so that
        # log.info()/warning()/error() from any module shows up here.
        import logging as _logging

        from ._logging import QtLogHandler
        self._log_handler = QtLogHandler(self.ml_log)
        self._log_handler.setLevel(_logging.INFO)
        _logging.getLogger("tr_sindy_app").addHandler(self._log_handler)
        _logging.getLogger("tr_sindy_app").setLevel(_logging.INFO)

        # tab 2: prediction visualization
        viz_tab = QtWidgets.QWidget()
        vt = QtWidgets.QVBoxLayout(viz_tab)
        vt.setContentsMargins(14, 14, 14, 14)
        vt.setSpacing(8)
        # controls row
        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.addWidget(QtWidgets.QLabel("Frame:"))
        self.ml_viz_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.ml_viz_slider.setMinimum(1); self.ml_viz_slider.setMaximum(1)
        self.ml_viz_frame_label = QtWidgets.QLabel("1 / 1")
        self.ml_viz_frame_label.setObjectName("readout")
        self.ml_viz_frame_label.setFixedWidth(100)
        self.ml_viz_frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ctrl_row.addWidget(self.ml_viz_slider, 1)
        ctrl_row.addWidget(self.ml_viz_frame_label)
        self.ml_viz_type_combo = QtWidgets.QComboBox()
        self.ml_viz_type_combo.addItems(["Quiver (Actual vs Pred)", "Magnitude Heatmap",
                                         "Error Map", "Loss Curve"])
        self.ml_viz_type_combo.currentIndexChanged.connect(self._ml_update_viz)
        ctrl_row.addWidget(QtWidgets.QLabel("View:"))
        ctrl_row.addWidget(self.ml_viz_type_combo)
        vt.addLayout(ctrl_row)
        # canvas
        self.ml_viz_fig = plt.Figure(figsize=(10, 6), tight_layout=True)
        self.ml_viz_fig.patch.set_facecolor(Theme.SURFACE)
        self.ml_viz_canvas = FigureCanvas(self.ml_viz_fig)
        vt.addWidget(self.ml_viz_canvas, 1)
        self.ml_viz_toolbar = NavigationToolbar(self.ml_viz_canvas, viz_tab)
        vt.addWidget(self.ml_viz_toolbar)
        self.ml_tabs.addTab(viz_tab, "Visualization")
        self._figs.append(self.ml_viz_fig)

        ct.addWidget(self.ml_tabs)
        page = self._make_split_page(controls, content); self._apply_glass_effects(page); self.pages.addWidget(page)

        # state for ML predictions
        self.ml_pred = None  # dict: {u_pred, v_pred, p_pred, loss_history, model_name, actual_u, actual_v, n_frames, h, w}
        self.ml_model_obj = None  # trained wrapper instance, for export
        self.ml_model_name = None

        # connect slider
        self.ml_viz_slider.valueChanged.connect(self._ml_update_viz)

    def _ml_log(self, msg):
        self.ml_log.appendPlainText(msg)
        QtWidgets.QApplication.processEvents()

    # ----- parameter page builders -----
    def _add_param_page(self, key, title):
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)
        grp = QtWidgets.QGroupBox(title.upper())
        gl = QtWidgets.QGridLayout(grp)
        gl.setSpacing(8)
        v.addWidget(grp)
        v.addStretch(1)
        self.ml_param_stack.addWidget(page)
        self.ml_param_widgets[key] = {}
        self.ml_param_widgets[key]['_page'] = page
        self.ml_param_widgets[key]['_grid'] = gl
        self.ml_param_widgets[key]['_row'] = 0
        return gl

    def _add_param(self, key, label, widget):
        gl = self.ml_param_widgets[key]['_grid']
        r = self.ml_param_widgets[key]['_row']
        gl.addWidget(QtWidgets.QLabel(label), r, 0)
        gl.addWidget(widget, r, 1)
        self.ml_param_widgets[key]['_row'] = r + 1
        return widget

    def _build_pinn_params(self):
        k = 'pinn'
        self._add_param_page(k, "PINN Parameters")
        self.ml_param_widgets[k]['steps'] = self._add_param(k, "Training steps:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['steps'].setRange(10, 10000); self.ml_param_widgets[k]['steps'].setValue(200)
        self.ml_param_widgets[k]['hidden'] = self._add_param(k, "Hidden units:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['hidden'].setRange(4, 512); self.ml_param_widgets[k]['hidden'].setValue(32)
        self.ml_param_widgets[k]['layers'] = self._add_param(k, "Layers:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['layers'].setRange(1, 20); self.ml_param_widgets[k]['layers'].setValue(4)
        self.ml_param_widgets[k]['nu'] = self._add_param(k, "Viscosity ν:", QtWidgets.QDoubleSpinBox())
        self.ml_param_widgets[k]['nu'].setRange(1e-6, 1.0); self.ml_param_widgets[k]['nu'].setValue(1e-3)
        self.ml_param_widgets[k]['nu'].setDecimals(6); self.ml_param_widgets[k]['nu'].setSingleStep(1e-4)
        self.ml_param_widgets[k]['lr'] = self._add_param(k, "Learning rate:", QtWidgets.QDoubleSpinBox())
        self.ml_param_widgets[k]['lr'].setRange(1e-6, 1.0); self.ml_param_widgets[k]['lr'].setValue(1e-3)
        self.ml_param_widgets[k]['lr'].setDecimals(6); self.ml_param_widgets[k]['lr'].setSingleStep(1e-4)
        self.ml_param_widgets[k]['lambda_data'] = self._add_param(k, "Data loss weight:", QtWidgets.QDoubleSpinBox())
        self.ml_param_widgets[k]['lambda_data'].setRange(0.0, 100.0); self.ml_param_widgets[k]['lambda_data'].setValue(1.0)
        self.ml_param_widgets[k]['lambda_pde'] = self._add_param(k, "PDE loss weight:", QtWidgets.QDoubleSpinBox())
        self.ml_param_widgets[k]['lambda_pde'].setRange(0.0, 100.0); self.ml_param_widgets[k]['lambda_pde'].setValue(0.1)
        self.ml_param_widgets[k]['frame'] = self._add_param(k, "Train on frame:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['frame'].setRange(1, 9999); self.ml_param_widgets[k]['frame'].setValue(1)

    def _build_ae_sindy_params(self):
        k = 'ae_sindy'
        self._add_param_page(k, "Autoencoder-SINDy Parameters")
        self.ml_param_widgets[k]['latent_dim'] = self._add_param(k, "Latent dim:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['latent_dim'].setRange(1, 64); self.ml_param_widgets[k]['latent_dim'].setValue(4)
        self.ml_param_widgets[k]['epochs'] = self._add_param(k, "Epochs:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['epochs'].setRange(1, 10000); self.ml_param_widgets[k]['epochs'].setValue(50)
        self.ml_param_widgets[k]['lr'] = self._add_param(k, "Learning rate:", QtWidgets.QDoubleSpinBox())
        self.ml_param_widgets[k]['lr'].setRange(1e-6, 1.0); self.ml_param_widgets[k]['lr'].setValue(1e-3)
        self.ml_param_widgets[k]['lr'].setDecimals(6); self.ml_param_widgets[k]['lr'].setSingleStep(1e-4)
        self.ml_param_widgets[k]['threshold'] = self._add_param(k, "SINDy threshold:", QtWidgets.QDoubleSpinBox())
        self.ml_param_widgets[k]['threshold'].setRange(0.0, 10.0); self.ml_param_widgets[k]['threshold'].setValue(0.05)
        self.ml_param_widgets[k]['threshold'].setDecimals(4); self.ml_param_widgets[k]['threshold'].setSingleStep(0.01)

    def _build_fno_params(self):
        k = 'fno'
        self._add_param_page(k, "FNO Parameters")
        self.ml_param_widgets[k]['modes'] = self._add_param(k, "Fourier modes:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['modes'].setRange(1, 64); self.ml_param_widgets[k]['modes'].setValue(12)
        self.ml_param_widgets[k]['width'] = self._add_param(k, "Width:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['width'].setRange(1, 256); self.ml_param_widgets[k]['width'].setValue(32)
        self.ml_param_widgets[k]['n_layers'] = self._add_param(k, "Layers:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['n_layers'].setRange(1, 16); self.ml_param_widgets[k]['n_layers'].setValue(4)
        self.ml_param_widgets[k]['epochs'] = self._add_param(k, "Epochs:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['epochs'].setRange(1, 10000); self.ml_param_widgets[k]['epochs'].setValue(20)
        self.ml_param_widgets[k]['lr'] = self._add_param(k, "Learning rate:", QtWidgets.QDoubleSpinBox())
        self.ml_param_widgets[k]['lr'].setRange(1e-6, 1.0); self.ml_param_widgets[k]['lr'].setValue(1e-3)
        self.ml_param_widgets[k]['lr'].setDecimals(6); self.ml_param_widgets[k]['lr'].setSingleStep(1e-4)

    def _build_deeponet_params(self):
        k = 'deeponet'
        self._add_param_page(k, "DeepONet Parameters")
        self.ml_param_widgets[k]['n_sensors'] = self._add_param(k, "Sensors:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['n_sensors'].setRange(1, 10000); self.ml_param_widgets[k]['n_sensors'].setValue(100)
        self.ml_param_widgets[k]['hidden'] = self._add_param(k, "Hidden units:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['hidden'].setRange(4, 512); self.ml_param_widgets[k]['hidden'].setValue(64)
        self.ml_param_widgets[k]['epochs'] = self._add_param(k, "Epochs:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['epochs'].setRange(1, 10000); self.ml_param_widgets[k]['epochs'].setValue(50)
        self.ml_param_widgets[k]['lr'] = self._add_param(k, "Learning rate:", QtWidgets.QDoubleSpinBox())
        self.ml_param_widgets[k]['lr'].setRange(1e-6, 1.0); self.ml_param_widgets[k]['lr'].setValue(1e-3)
        self.ml_param_widgets[k]['lr'].setDecimals(6); self.ml_param_widgets[k]['lr'].setSingleStep(1e-4)

    def _build_convlstm_params(self):
        k = 'convlstm'
        self._add_param_page(k, "ConvLSTM Parameters")
        self.ml_param_widgets[k]['hidden'] = self._add_param(k, "Hidden channels:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['hidden'].setRange(1, 128); self.ml_param_widgets[k]['hidden'].setValue(16)
        self.ml_param_widgets[k]['kernel'] = self._add_param(k, "Kernel size:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['kernel'].setRange(1, 11); self.ml_param_widgets[k]['kernel'].setValue(5)
        self.ml_param_widgets[k]['kernel'].setSingleStep(2)
        self.ml_param_widgets[k]['epochs'] = self._add_param(k, "Epochs:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['epochs'].setRange(1, 10000); self.ml_param_widgets[k]['epochs'].setValue(20)
        self.ml_param_widgets[k]['horizon'] = self._add_param(k, "Forecast horizon:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['horizon'].setRange(1, 100); self.ml_param_widgets[k]['horizon'].setValue(1)
        self.ml_param_widgets[k]['lr'] = self._add_param(k, "Learning rate:", QtWidgets.QDoubleSpinBox())
        self.ml_param_widgets[k]['lr'].setRange(1e-6, 1.0); self.ml_param_widgets[k]['lr'].setValue(1e-3)
        self.ml_param_widgets[k]['lr'].setDecimals(6); self.ml_param_widgets[k]['lr'].setSingleStep(1e-4)

    def _build_vae_params(self):
        k = 'vae'
        self._add_param_page(k, "VAE Parameters")
        self.ml_param_widgets[k]['latent_dim'] = self._add_param(k, "Latent dim:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['latent_dim'].setRange(1, 128); self.ml_param_widgets[k]['latent_dim'].setValue(8)
        self.ml_param_widgets[k]['hidden'] = self._add_param(k, "Hidden units:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['hidden'].setRange(4, 1024); self.ml_param_widgets[k]['hidden'].setValue(128)
        self.ml_param_widgets[k]['beta'] = self._add_param(k, "β (KL weight):", QtWidgets.QDoubleSpinBox())
        self.ml_param_widgets[k]['beta'].setRange(0.0, 100.0); self.ml_param_widgets[k]['beta'].setValue(1.0)
        self.ml_param_widgets[k]['epochs'] = self._add_param(k, "Epochs:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['epochs'].setRange(1, 10000); self.ml_param_widgets[k]['epochs'].setValue(50)
        self.ml_param_widgets[k]['lr'] = self._add_param(k, "Learning rate:", QtWidgets.QDoubleSpinBox())
        self.ml_param_widgets[k]['lr'].setRange(1e-6, 1.0); self.ml_param_widgets[k]['lr'].setValue(1e-3)
        self.ml_param_widgets[k]['lr'].setDecimals(6); self.ml_param_widgets[k]['lr'].setSingleStep(1e-4)

    def _build_ensemble_params(self):
        k = 'ensemble'
        self._add_param_page(k, "Ensemble Parameters")
        self.ml_param_widgets[k]['n_models'] = self._add_param(k, "Number of models:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['n_models'].setRange(2, 50); self.ml_param_widgets[k]['n_models'].setValue(5)
        self.ml_param_widgets[k]['subsample'] = self._add_param(k, "Subsample size:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['subsample'].setRange(1000, 500000); self.ml_param_widgets[k]['subsample'].setValue(50000)
        self.ml_param_widgets[k]['subsample'].setSingleStep(10000)

    def _build_gan_params(self):
        k = 'gan'
        self._add_param_page(k, "GAN Parameters")
        self.ml_param_widgets[k]['noise_dim'] = self._add_param(k, "Noise dim:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['noise_dim'].setRange(4, 512); self.ml_param_widgets[k]['noise_dim'].setValue(32)
        self.ml_param_widgets[k]['hidden'] = self._add_param(k, "Hidden units:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['hidden'].setRange(4, 1024); self.ml_param_widgets[k]['hidden'].setValue(128)
        self.ml_param_widgets[k]['epochs'] = self._add_param(k, "Epochs:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['epochs'].setRange(1, 100000); self.ml_param_widgets[k]['epochs'].setValue(100)
        self.ml_param_widgets[k]['n_samples'] = self._add_param(k, "Samples to generate:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['n_samples'].setRange(1, 10000); self.ml_param_widgets[k]['n_samples'].setValue(16)
        self.ml_param_widgets[k]['lr'] = self._add_param(k, "Learning rate:", QtWidgets.QDoubleSpinBox())
        self.ml_param_widgets[k]['lr'].setRange(1e-6, 1.0); self.ml_param_widgets[k]['lr'].setValue(2e-4)
        self.ml_param_widgets[k]['lr'].setDecimals(6); self.ml_param_widgets[k]['lr'].setSingleStep(1e-4)

    def _build_causal_params(self):
        k = 'causal'
        self._add_param_page(k, "Causal Analysis Parameters")
        self.ml_param_widgets[k]['max_lag'] = self._add_param(k, "Max lag:", QtWidgets.QSpinBox())
        self.ml_param_widgets[k]['max_lag'].setRange(1, 50); self.ml_param_widgets[k]['max_lag'].setValue(5)

    def _ml_switch_params(self, idx):
        self.ml_param_stack.setCurrentIndex(idx)

    def _ml_param(self, key, name):
        return self.ml_param_widgets[key][name].value()

    def _ml_run_selected(self):
        idx = self.ml_model_combo.currentIndex()
        handlers = [self.run_pinn, self.run_ae_sindy, self.run_fno, self.run_deeponet,
                    self.run_convlstm, self.run_vae, self.run_ensemble, self.run_gan,
                    self.run_causal]
        if 0 <= idx < len(handlers):
            handlers[idx]()

    # -----------------------------------------------------------------
    #  Export page
    # -----------------------------------------------------------------
    def _build_export_page(self):
        controls = self._controls_panel()
        cl = QtWidgets.QVBoxLayout(controls)
        cl.setContentsMargins(18, 18, 18, 18)
        cl.setSpacing(12)

        eg = QtWidgets.QGroupBox("EXPORT")
        egl = QtWidgets.QGridLayout(eg)
        egl.setSpacing(8)
        self.export_csv_btn = self._mk_btn("CSV", egl, 0, 0, self.export_csv_gui)
        self.export_hdf5_btn = self._mk_btn("HDF5", egl, 0, 1, self.export_hdf5_gui)
        self.export_netcdf_btn = self._mk_btn("NetCDF", egl, 0, 2, self.export_netcdf_gui)
        self.export_parquet_btn = self._mk_btn("Parquet", egl, 1, 0, self.export_parquet_gui)
        self.export_json_btn = self._mk_btn("Metadata JSON", egl, 1, 1, self.export_metadata_gui)
        self.export_pdf_btn = self._mk_btn("PDF Report", egl, 1, 2, self.export_pdf_gui)
        self.export_imgs_btn = self._mk_btn("Image Sequence", egl, 2, 0, self.export_images_gui)
        self.anim_btn = self._mk_btn("Quiver Animation", egl, 2, 1, self.export_animation_gui)
        self.anim_btn.setProperty("variant", "violet")
        cl.addWidget(eg)
        cl.addStretch(1)

        # ---- content (right): summary ----
        content = QtWidgets.QWidget()
        ct = QtWidgets.QVBoxLayout(content)
        ct.setContentsMargins(14, 14, 14, 14)
        ct.setSpacing(8)
        head = QtWidgets.QLabel("EXPORT FORMATS")
        head.setObjectName("contentHeader")
        ct.addWidget(head)
        hint = QtWidgets.QLabel(
            "CSV  ·  per-frame u/v actual & predicted\n\n"
            "HDF5  ·  scientific hierarchy (needs h5py)\n"
            "NetCDF  ·  climate/data standard (needs scipy)\n"
            "Parquet  ·  columnar table (needs pyarrow)\n\n"
            "Metadata JSON  ·  all processing parameters\n"
            "PDF Report  ·  metrics + figures, auto-generated\n\n"
            "Image Sequence  ·  PNG/JPG per frame\n"
            "Quiver Animation  ·  MP4 (requires FFmpeg)")
        hint.setObjectName("stepGuide")
        hint.setWordWrap(True)
        ct.addWidget(hint)
        ct.addStretch(1)
        page = self._make_split_page(controls, content); self._apply_glass_effects(page); self.pages.addWidget(page)

    # =================================================================
    #  Pipeline: file / ROI
    # =================================================================
    def pick_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select video file", ".", "Videos (*.mp4 *.avi *.mov);;All (*)")
        if fname:
            self.file_edit.setText(fname)
            self._set_pill("● VIDEO LOADED", "ready")
            self.status_bar.showMessage(f"Loaded: {os.path.basename(fname)}")
            self.stepper.set_state(0, "done"); self.stepper.set_state(1, "active")

    def start_roi(self):
        path = self.file_edit.text()
        if not os.path.exists(path):
            _warn(self, "No file", "Please select a valid video file first.")
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            _warn(self, "Open error", "Cannot open selected video.")
            return
        ret, first_frame = cap.read()
        cap.release()
        if not ret:
            _warn(self, "Read error", "Cannot read first frame.")
            return
        self.status.setText("Draw the ROI, then a calibration line of known length.")
        dlg = ROICalibDialog(first_frame, self)
        res = dlg.get()
        if res is None:
            self.status.setText("ROI / calibration cancelled.")
            return
        xa, ya, xb, yb = res['roi']
        roi_w, roi_h = xb - xa, yb - ya
        px_len = res['px_len']
        use_meters, ok = QtWidgets.QInputDialog.getDouble(
            self, "Real length", "Enter real-world length (meters) of calibration line:",
            value=0.1, min=1e-6, decimals=6)
        if not ok or use_meters <= 0:
            self.status.setText("Calibration cancelled (invalid input).")
            return
        meters_per_pixel = use_meters / px_len
        self.mpp_label.setText(
            f"Calibration:  {px_len:.2f} px = {use_meters:.5f} m    ·    1 px = {meters_per_pixel:.8f} m")
        self.result = {
            'video_file': path, 'roi_box': (xa, ya, xb, yb),
            'calibration_px': px_len, 'calibration_m': use_meters,
            'meters_per_pixel': meters_per_pixel,
            'first_frame_roi': first_frame[ya:yb, xa:xb].copy(),
        }
        self.finished = True
        self.process_btn.setEnabled(True)
        self.history.log("roi_calibration", {"roi": [xa, ya, xb, yb], "m_per_px": meters_per_pixel})
        self.status.setText("ROI / calibration complete.  Ready for optical flow.")
        self._set_pill("● CALIBRATED", "ready")
        self.status_bar.showMessage(f"ROI {roi_w}×{roi_h} px · scale {meters_per_pixel:.6e} m/px")
        self.stepper.set_state(1, "done"); self.stepper.set_state(2, "active")

    # =================================================================
    #  Pipeline: optical flow
    # =================================================================
    def _show_of_preview(self, bgr, frame_idx: int = 0):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        tw = max(self.of_preview.width() - 8, 200)
        th = max(self.of_preview.height() - 8, 200)
        self.of_preview.setPixmap(pix.scaled(
            tw, th, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        QtWidgets.QApplication.processEvents()

    def _flow_config(self) -> optical_flow.FlowConfig:
        return optical_flow.FlowConfig(
            backend=self.backend_combo.currentText(),
            multiscale=self.multiscale_chk.isChecked(),
            temporal_smoothing=self.ts_combo.currentText(),
            temporal_alpha=float(self.alpha_spin.value()),
            enable_gauss=self.gauss_chk.isChecked(),
            enable_nlm=self.nlm_chk.isChecked(),
            compute_quality=self.quality_chk.isChecked(),
        )

    def run_optical_flow_gui(self):
        if not getattr(self, 'finished', False):
            _warn(self, "No ROI/Calibration", "Finish ROI/Calibration first.")
            return
        cfg = self._flow_config()
        self._set_pill("● OPTICAL FLOW", "busy")
        self.progress.show(); self.progress.setValue(0)
        QtWidgets.QApplication.processEvents()
        box = self.result['roi_box']
        mpp = self.result['meters_per_pixel']
        os.makedirs(self._mmap_dir, exist_ok=True)
        mmap_paths = {
            "u": os.path.abspath(os.path.join(self._mmap_dir, "u.dat")),
            "v": os.path.abspath(os.path.join(self._mmap_dir, "v.dat")),
            "frames": os.path.abspath(os.path.join(self._mmap_dir, "frames.dat")),
        }
        for p in mmap_paths.values():
            if os.path.exists(p):
                try: os.remove(p)
                except Exception: pass

        def pcb(i, n, stage):
            self.progress.setValue(int(i / n * 100))
            if i > 0 and n > 0:
                import time as _t
                if not hasattr(self, '_of_start_t'):
                    self._of_start_t = _t.time()
                elapsed = _t.time() - self._of_start_t
                fps = i / elapsed if elapsed > 0 else 0
                remaining = (n - i) / fps if fps > 0 else 0
                eta_m, eta_s = divmod(int(remaining), 60)
                self.status_bar.showMessage(
                    f"{stage}: {i}/{n}  ~{eta_m}m{eta_s:02d}s remaining")
            QtWidgets.QApplication.processEvents()

        try:
            meta = optical_flow.process_video(
                self.result['video_file'], box, mpp, cfg, mmap_paths,
                progress_cb=pcb, preview_cb=self._show_of_preview)
        except Exception as e:
            self.progress.hide()
            self._set_pill("● ERROR", "idle")
            _err(self, "Optical flow error", str(e))
            return
        self.progress.setValue(100); self.progress.hide()
        self.of_result = {
            "u_mmap_path": mmap_paths["u"], "v_mmap_path": mmap_paths["v"],
            "frame_mmap_path": mmap_paths["frames"], **meta,
            "mmap_paths": mmap_paths,
        }
        self.history.log("optical_flow", vars(cfg))
        self.status.setText(f"Optical flow done [{meta['frames']}, {meta['roi_h']}×{meta['roi_w']}].")
        self._set_pill("● FLOW READY", "done")
        self.sindy_btn.setEnabled(True)
        self.stepper.set_state(2, "done"); self.stepper.set_state(3, "active")
        _info_dialog(self, "Done", f"Optical flow complete ({meta['backend']}).")

    # =================================================================
    #  Pipeline: SINDy
    # =================================================================
    def _sindy_config(self):
        return _get_sindy_core().SINDyConfig(
            library=self.lib_combo.currentText(),
            degree=int(self.degree_spin.value()),
            optimizer=self.opt_combo.currentText(),
            threshold=float(self.thresh_spin.value()),
            divergence_free=self.divfree_chk.isChecked(),
        )

    def run_sindy_gui(self):
        if not self.of_result:
            _warn(self, "No flow data", "Run optical flow first.")
            return
        cfg = self._sindy_config()
        self._set_pill("● SINDy FIT", "busy")
        n = self.of_result['frames']; h = self.of_result['roi_h']; w = self.of_result['roi_w']
        u_mmap = np.memmap(self.of_result['u_mmap_path'], np.float32, mode='r', shape=(n, h, w))
        v_mmap = np.memmap(self.of_result['v_mmap_path'], np.float32, mode='r', shape=(n, h, w))
        dt = 1.0 / self.of_result['FPS']
        self.status.setText("Building SINDy dataset (derivatives)…")
        self.progress.show(); self.progress.setValue(5)
        QtWidgets.QApplication.processEvents()
        X, Xdot, names = _get_sindy_core().build_sindy_dataset(u_mmap, v_mmap, dt)
        self.progress.setValue(40)

        def pcb(i, nb, stage):
            self.progress.setValue(40 + int(i / nb * 55))
            QtWidgets.QApplication.processEvents()

        try:
            fit = _get_sindy_core().fit_sindy(X, Xdot, dt, cfg, feature_names=names, progress_cb=pcb)
        except Exception as e:
            self.progress.hide(); self._set_pill("● ERROR", "idle")
            _err(self, "SINDy error", str(e)); return
        self.progress.setValue(100); self.progress.hide()
        mmap_dir = os.path.dirname(self.of_result['u_mmap_path'])
        X_path = os.path.join(mmap_dir, "X_optical.dat")
        Xdot_path = os.path.join(mmap_dir, "X_dot_optical.dat")
        Xm = np.memmap(X_path, np.float32, mode='w+', shape=X.shape)
        Xdm = np.memmap(Xdot_path, np.float32, mode='w+', shape=Xdot.shape)
        Xm[:] = X; Xdm[:] = Xdot; Xm.flush(); Xdm.flush()
        self.sindy_result = {**fit, "X_optical_path": X_path, "X_dot_optical_path": Xdot_path,
                             "n_frames": n, "roi_h": h, "roi_w": w, "DT": dt,
                             "total_points": X.shape[0], "batch_size": max(100_000, h * w)}
        self.history.log("sindy_fit", vars(cfg))
        self.pred_btn.setEnabled(True); self.sindy_eq_btn.setEnabled(True)
        self.cv_btn.setEnabled(True); self.compare_btn.setEnabled(True)
        self._set_pill("● MODEL FIT", "done")
        self.stepper.set_state(3, "done"); self.stepper.set_state(4, "active")
        _info_dialog(self, "Done", "SINDy model fit successful!")

    def run_prediction_gui(self):
        try:
            model = self.sindy_result['model']
            X_path = self.sindy_result['X_optical_path']
            total = self.sindy_result['total_points']
            n = self.sindy_result['n_frames']
            h = self.sindy_result['roi_h']; w = self.sindy_result['roi_w']
            bs = self.sindy_result['batch_size']
        except KeyError:
            _warn(self, "No model", "Fit a SINDy model first."); return
        self._set_pill("● PREDICTING", "busy")
        self.progress.show(); self.progress.setValue(0)
        pred_path = os.path.join(os.path.dirname(X_path), "X_pred_optical.dat")
        if os.path.exists(pred_path): os.remove(pred_path)
        pred_mmap = np.memmap(pred_path, np.float32, mode='w+', shape=(total, 2))
        n_batches = int(np.ceil(total / bs))
        X_full = np.memmap(X_path, np.float32, mode='r', shape=(total, 8))
        for b in range(n_batches):
            b0, b1 = b * bs, min((b + 1) * bs, total)
            pred_mmap[b0:b1] = _get_sindy_core().predict_sindy(model, X_full[b0:b1], flip=True)
            self.progress.setValue(int((b + 1) * 80 / n_batches))
            QtWidgets.QApplication.processEvents()
        pred_mmap.flush()
        X_pred = np.zeros((n, h, w, 2), np.float32)
        for f in range(n):
            seg = pred_mmap[f * h * w:(f + 1) * h * w]
            X_pred[f, ..., 0] = gaussian_filter(seg[:, 0].reshape(h, w), 0.8)
            X_pred[f, ..., 1] = gaussian_filter(seg[:, 1].reshape(h, w), 0.8)
            if f % max(n // 10, 1) == 0:
                self.progress.setValue(80 + int(f / n * 20))
                QtWidgets.QApplication.processEvents()
        self.progress.setValue(100); self.progress.hide()
        self.X_pred_optical = X_pred
        self.history.log("prediction", {"n_frames": n})
        self._enable_visualization()
        self._set_pill("● PREDICTION READY", "done")
        _info_dialog(self, "Ready", "SINDy prediction and reconstruction ready.")

    # =================================================================
    #  SINDy extras: equation, CV, compare
    # =================================================================
    def show_sindy_equation_gui(self):
        model = self.sindy_result.get('model')
        if model is None:
            _warn(self, "No SINDy Model", "Run SINDy modeling first."); return
        eq = model.equations(precision=5)
        if isinstance(eq, list):
            eq = "\n\n".join(eq)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("SINDy Fitted Equation"); dlg.resize(640, 420)
        v = QtWidgets.QVBoxLayout(dlg)
        t = QtWidgets.QPlainTextEdit(); t.setPlainText(eq); t.setReadOnly(True)
        v.addWidget(t)
        b = QtWidgets.QPushButton("Copy to Clipboard"); b.setProperty("variant", "primary")
        b.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(eq))
        v.addWidget(b); dlg.exec()

    def run_cross_validation(self):
        if 'X_optical_path' not in self.sindy_result:
            _warn(self, "No data", "Fit a model first."); return
        k, ok = QtWidgets.QInputDialog.getInt(self, "Cross-validation", "k folds:", 5, 2, 10)
        if not ok: return
        total = self.sindy_result['total_points']
        X = np.memmap(self.sindy_result['X_optical_path'], np.float32, mode='r', shape=(total, 8))
        Xd = np.memmap(self.sindy_result['X_dot_optical_path'], np.float32, mode='r', shape=(total, 2))
        n_sub = min(total, 200_000)
        idx = np.random.default_rng(0).choice(total, n_sub, replace=False)
        cfg = self._sindy_config()
        self._set_pill("● CROSS-VALIDATING", "busy")
        self.progress.show(); self.progress.setValue(0)
        QtWidgets.QApplication.processEvents()
        res = _get_sindy_core().kfold_cross_validate(X[idx], Xd[idx], self.sindy_result['DT'], cfg, k,
            progress_cb=lambda i, n, s: (self.progress.setValue(int(i / n * 100)),
                                         QtWidgets.QApplication.processEvents()))
        self.progress.hide(); self._set_pill("● MODEL FIT", "done")
        msg = (f"k={res['k']}\nRMSE: {res['rmse_mean']:.5g} ± {res['rmse_std']:.5g}\n"
               f"MSE: {res['mse_mean']:.5g}\nPer-fold RMSE: {res['rmse_per_fold']}")
        if res.get("stability"):
            msg += f"\nCoef std mean: {res['stability']['coef_std_mean']:.5g}"
        _info_dialog(self, "Cross-validation", msg)

    def run_model_comparison(self):
        if 'X_optical_path' not in self.sindy_result:
            _warn(self, "No data", "Fit a model first."); return
        total = self.sindy_result['total_points']
        X = np.memmap(self.sindy_result['X_optical_path'], np.float32, mode='r', shape=(total, 8))
        Xd = np.memmap(self.sindy_result['X_dot_optical_path'], np.float32, mode='r', shape=(total, 2))
        n_sub = min(total, 200_000)
        idx = np.random.default_rng(0).choice(total, n_sub, replace=False)
        configs = [
            _get_sindy_core().SINDyConfig(library="polynomial", degree=2, threshold=0.07),
            _get_sindy_core().SINDyConfig(library="polynomial", degree=3, threshold=0.07),
            _get_sindy_core().SINDyConfig(library="fourier", n_freq=1, threshold=0.07),
            _get_sindy_core().SINDyConfig(library="combined", degree=2, threshold=0.07),
        ]
        self._set_pill("● COMPARING", "busy")
        res = _get_sindy_core().compare_models(X[idx], Xd[idx], self.sindy_result['DT'], configs)
        self._set_pill("● MODEL FIT", "done")
        lines = ["library          optimizer      rmse      n_terms"]
        for r in res:
            lines.append(f"{r['library']:16s} {r['optimizer']:14s} {r['rmse']:.4g}  {r['n_terms']}")
        _info_dialog(self, "Model comparison", "\n".join(lines))

    # =================================================================
    #  Visualization handlers
    # =================================================================
    def _load_frame(self, idx):
        n = self.sindy_result['n_frames']; h = self.sindy_result['roi_h']; w = self.sindy_result['roi_w']
        u = np.asarray(np.memmap(self.of_result['u_mmap_path'], np.float32, mode='r',
                                 shape=(n, h, w))[idx])
        v = np.asarray(np.memmap(self.of_result['v_mmap_path'], np.float32, mode='r',
                                 shape=(n, h, w))[idx])
        fr = np.asarray(np.memmap(self.of_result['frame_mmap_path'], np.uint8, mode='r',
                                  shape=(n, h, w))[idx])
        return u, v, fr

    def _pick_frame(self):
        n = self.sindy_result['n_frames']
        idx, ok = QtWidgets.QInputDialog.getInt(self, "Frame", f"Frame (1-{n}):", 1, 1, n)
        return (idx - 1) if ok else None

    def show_quiver_section(self):
        scale, ok = QtWidgets.QInputDialog.getDouble(self, "Quiver Scale",
            "Scale (higher = smaller arrows):", 1, 0.1, 150, 2)
        if not ok: return
        self.quiver_scale = scale
        n = self.sindy_result['n_frames']
        self.quiver_slider.setMinimum(1); self.quiver_slider.setMaximum(n)
        self.quiver_slider.setValue(1)
        self.quiver_frame_label.setText(f"1 / {n}")
        if not hasattr(self, '_qsc_connected'):
            self.quiver_slider.valueChanged.connect(self.update_quiver_plot)
            self._qsc_connected = True
        self.quiver_section.setVisible(True)
        self.update_quiver_plot(1)

    def update_quiver_plot(self, frame_idx):
        try:
            idx = frame_idx - 1
            n = self.sindy_result['n_frames']
            if idx < 0 or idx >= n: return
            self.quiver_frame_label.setText(f"{frame_idx} / {n}")
            mpp = self.result['meters_per_pixel']
            u, v, bg = self._load_frame(idx)
            pu = self.X_pred_optical[idx, ..., 0]
            pv = self.X_pred_optical[idx, ..., 1]
            h, w = u.shape
            step = max(h // 25, 2)
            ys = np.arange(0, h, step); xs = np.arange(0, w, step)
            xg, yg = np.meshgrid(xs, ys)
            ext = [0, w * mpp, h * mpp, 0]
            for ax, field, title, color in (
                (self.quiver_ax1, (u, v), "Actual", Theme.ACCENT),
                (self.quiver_ax2, (pu, pv), "SINDy", Theme.MAGENTA)):
                ax.clear(); ax.set_facecolor(Theme.BG_BASE)
                ax.imshow(bg, cmap='gray', origin='upper', extent=ext)
                ax.quiver(xg * mpp, yg * mpp,
                          field[0][ys[:, None], xs[None, :]],
                          field[1][ys[:, None], xs[None, :]],
                          color=color, angles='xy', scale=self.quiver_scale,
                          width=0.006)
                ax.set_title(f"{title} · Frame {frame_idx}")
                ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
                ax.set_xlim(0, w * mpp); ax.set_ylim(h * mpp, 0)
            self.quiver_fig.tight_layout()
            self.quiver_canvas.draw_idle()
            QtWidgets.QApplication.processEvents()
        except Exception as e:
            _err(self, "Plot Error", f"Failed to update quiver plot:\n{e}")

    def _pop_plot(self, fig):
        fig.patch.set_facecolor(Theme.SURFACE)
        for ax in fig.axes:
            ax.set_facecolor(Theme.BG_BASE)
        plt.show()

    def plot_contour_gui(self):
        idx = self._pick_frame()
        if idx is None: return
        which, ok = QtWidgets.QInputDialog.getItem(self, "Field", "Contour for:",
            ("Actual", "SINDy Prediction", "Error"), 0, False)
        if not ok: return
        mpp = self.result['meters_per_pixel']
        u, v, bg = self._load_frame(idx)
        pu = self.X_pred_optical[idx, ..., 0]; pv = self.X_pred_optical[idx, ..., 1]
        if which == "Actual": mag = np.sqrt(u ** 2 + v ** 2); title = f"Frame {idx+1} Actual"
        elif which == "SINDy Prediction": mag = np.sqrt(pu ** 2 + pv ** 2); title = f"Frame {idx+1} SINDy"
        else: mag = np.sqrt((u - pu) ** 2 + (v - pv) ** 2); title = f"Frame {idx+1} Error"
        ext = [0, u.shape[1] * mpp, u.shape[0] * mpp, 0]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(bg, cmap='gray', extent=ext, alpha=0.7)
        lv = np.linspace(np.nanmin(mag), np.nanmax(mag), 35)
        ax.contourf(np.linspace(0, u.shape[1] * mpp, u.shape[1]),
                    np.linspace(0, u.shape[0] * mpp, u.shape[0]),
                    np.nan_to_num(mag), levels=lv, cmap=self.cmap_combo.currentText(),
                    alpha=0.6, extend='both')
        ax.set_title(title + " magnitude"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        plt.colorbar(ax.collections[-1], label="|V|")
        plt.tight_layout(); self._pop_plot(fig)

    def plot_stream_gui(self):
        idx = self._pick_frame()
        if idx is None: return
        which, ok = QtWidgets.QInputDialog.getItem(self, "Field", "Streamlines for:",
            ("Actual", "SINDy Prediction", "Error"), 0, False)
        if not ok: return
        mpp = self.result['meters_per_pixel']
        u, v, bg = self._load_frame(idx)
        pu = self.X_pred_optical[idx, ..., 0]; pv = self.X_pred_optical[idx, ..., 1]
        if which == "Actual": pu_, pv_ = u, v; title = "Actual"
        elif which == "SINDy Prediction": pu_, pv_ = pu, pv; title = "SINDy"
        else: pu_, pv_ = u - pu, v - pv; title = "Error"
        ext = [0, u.shape[1] * mpp, u.shape[0] * mpp, 0]
        xv = np.linspace(0, u.shape[1] * mpp, u.shape[1])
        yv = np.linspace(0, u.shape[0] * mpp, u.shape[0])
        Xg, Yg = np.meshgrid(xv, yv)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(bg, cmap='gray', extent=ext, alpha=0.7)
        ax.streamplot(Xg, Yg, pu_, pv_, color=Theme.ACCENT, density=1.2, linewidth=1, arrowsize=1.2)
        ax.set_title(f"Frame {idx+1} {title} streamlines")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        plt.tight_layout(); self._pop_plot(fig)

    def plot_vorticity_gui(self):
        idx = self._pick_frame()
        if idx is None: return
        mpp = self.result['meters_per_pixel']
        u, v, _ = self._load_frame(idx)
        w = _get_analysis().vorticity(u, v, mpp, mpp)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(w, cmap="RdBu_r", origin='upper',
                       extent=[0, u.shape[1]*mpp, u.shape[0]*mpp, 0])
        ax.set_title(f"Vorticity ω_z · Frame {idx+1}")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        plt.colorbar(im, label="ω_z (1/s)")
        plt.tight_layout(); self._pop_plot(fig)

    def plot_strain_gui(self):
        idx = self._pick_frame()
        if idx is None: return
        mpp = self.result['meters_per_pixel']
        u, v, _ = self._load_frame(idx)
        s = _get_analysis().strain_rate(u, v, mpp, mpp)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(s["magnitude"], cmap=self.cmap_combo.currentText(), origin='upper',
                       extent=[0, u.shape[1]*mpp, u.shape[0]*mpp, 0])
        ax.set_title(f"Strain-rate magnitude · Frame {idx+1}")
        plt.colorbar(im, label="|S|")
        plt.tight_layout(); self._pop_plot(fig)

    def plot_pod_gui(self):
        n = self.sindy_result['n_frames']; h = self.sindy_result['roi_h']; w = self.sindy_result['roi_w']
        u = np.asarray(np.memmap(self.of_result['u_mmap_path'], np.float32, mode='r', shape=(n, h, w)))
        res = _get_analysis().pod_decompose(u, n_modes=6)
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for i, ax in enumerate(axes.ravel()):
            if i < len(res["modes"]):
                ax.imshow(res["modes"][i], cmap=self.cmap_combo.currentText())
                ax.set_title(f"POD mode {i+1}  ({res['energies'][i]:.3g})")
            ax.axis('off')
        fig.suptitle("POD modes (u-component)")
        plt.tight_layout(); self._pop_plot(fig)

    def plot_dmd_gui(self):
        n = self.sindy_result['n_frames']; h = self.sindy_result['roi_h']; w = self.sindy_result['roi_w']
        u = np.asarray(np.memmap(self.of_result['u_mmap_path'], np.float32, mode='r', shape=(n, h, w)))
        res = _get_analysis().dmd_decompose(u, n_modes=6)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(res["eigenvalues"].real, res["eigenvalues"].imag, c=Theme.ACCENT)
        axes[0].add_patch(plt.Circle((0, 0), 1, fill=False, ls='--', color=Theme.TEXT_MUTED))
        axes[0].set_aspect('equal'); axes[0].set_title("DMD eigenvalues")
        axes[0].set_xlabel("Re(λ)"); axes[0].set_ylabel("Im(λ)")
        mags = np.abs(res["modes"][:, :6]).mean(axis=1).reshape(h, w)
        im = axes[1].imshow(mags, cmap=self.cmap_combo.currentText())
        axes[1].set_title("Mean |DMD mode| magnitude")
        plt.colorbar(im, ax=axes[1])
        plt.tight_layout(); self._pop_plot(fig)

    def plot_spectral_gui(self):
        idx = self._pick_frame()
        if idx is None: return
        u, v, _ = self._load_frame(idx)
        su = _get_analysis().spatial_spectrum(u)
        sv = _get_analysis().spatial_spectrum(v)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.loglog(su["k"][1:], su["E"][1:], label="u", color=Theme.ACCENT)
        ax.loglog(sv["k"][1:], sv["E"][1:], label="v", color=Theme.MAGENTA)
        ax.set_title(f"Spatial energy spectrum · Frame {idx+1}")
        ax.set_xlabel("wavenumber k"); ax.set_ylabel("E(k)"); ax.legend()
        plt.tight_layout(); self._pop_plot(fig)

    def plot_animated_heatmap(self):
        n = self.sindy_result['n_frames']; h = self.sindy_result['roi_h']; w = self.sindy_result['roi_w']
        u = np.asarray(np.memmap(self.of_result['u_mmap_path'], np.float32, mode='r', shape=(n, h, w)))
        v = np.asarray(np.memmap(self.of_result['v_mmap_path'], np.float32, mode='r', shape=(n, h, w)))
        mag = np.sqrt(u ** 2 + v ** 2)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(mag[0], cmap=self.cmap_combo.currentText(), animated=True,
                       vmin=float(mag.min()), vmax=float(mag.max()))
        fig.colorbar(im, ax=ax, label="|V|")
        ax.set_title("Velocity magnitude evolution")
        def update(f):
            im.set_array(mag[f])
            ax.set_title(f"Velocity magnitude · Frame {f+1}/{n}")
            return [im]
        import matplotlib.animation as anim
        a = anim.FuncAnimation(fig, update, frames=n, interval=80, blit=False)
        # Keep a reference alive: FuncAnimation is driven by a timer that stops
        # the moment the object is garbage-collected, which happens immediately
        # when this method returns under the non-blocking Qt backend.
        self._heatmap_anim = a
        fig.canvas.mpl_connect(
            "close_event", lambda _e: setattr(self, "_heatmap_anim", None))
        plt.show()

    def show_turbulence_stats(self):
        idx = self._pick_frame()
        if idx is None: return
        mpp = self.result['meters_per_pixel']
        u, v, _ = self._load_frame(idx)
        stats = _get_analysis().turbulence_statistics(u, v, length_scale=min(u.shape) * mpp)
        lines = [f"{k}: {v:.5g}" for k, v in stats.items()]
        _info_dialog(self, f"Turbulence stats · Frame {idx+1}", "\n".join(lines))

    def show_data_table(self):
        idx = self._pick_frame()
        if idx is None: return
        u, v, _ = self._load_frame(idx)
        pu = self.X_pred_optical[idx, ..., 0]; pv = self.X_pred_optical[idx, ..., 1]
        h, w = u.shape
        yy, xx = np.mgrid[0:h, 0:w]
        df = pd.DataFrame({
            "y": yy.ravel(), "x": xx.ravel(),
            "actual_u": u.ravel(), "actual_v": v.ravel(),
            "pred_u": pu.ravel(), "pred_v": pv.ravel(),
        })
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Velocity data · Frame {idx+1}")
        dlg.resize(720, 520)
        v = QtWidgets.QVBoxLayout(dlg)
        table = QtWidgets.QTableWidget()
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setRowCount(min(len(df), 5000)); table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels(list(df.columns))
        for r in range(table.rowCount()):
            for c, col in enumerate(df.columns):
                table.setItem(r, c, QtWidgets.QTableWidgetItem(str(df[col].iloc[r])))
        v.addWidget(table)
        dlg.exec()

    def run_error_analysis_gui(self):
        n = self.sindy_result['n_frames']; h = self.sindy_result['roi_h']; w = self.sindy_result['roi_w']
        u = np.asarray(np.memmap(self.of_result['u_mmap_path'], np.float32, mode='r', shape=(n, h, w)))
        v = np.asarray(np.memmap(self.of_result['v_mmap_path'], np.float32, mode='r', shape=(n, h, w)))
        errs = _get_analysis().per_frame_errors(u, v, self.X_pred_optical)
        fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(10, 9))
        a1.plot(errs["rmse"], color=Theme.ACCENT); a1.set_ylabel("RMSE"); a1.set_title("Per-frame errors")
        a2.plot(errs["mse"], color=Theme.MAGENTA); a2.set_ylabel("MSE")
        a3.plot(errs["mae"], color=Theme.AMBER); a3.set_ylabel("MAE"); a3.set_xlabel("frame")
        plt.tight_layout(); self._pop_plot(fig)
        idx = self._pick_frame()
        if idx is None: return
        err = np.sqrt((u[idx] - self.X_pred_optical[idx, ..., 0]) ** 2 +
                      (v[idx] - self.X_pred_optical[idx, ..., 1]) ** 2)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(err, cmap='inferno', origin='upper')
        ax.set_title(f"Spatial error map · Frame {idx+1}")
        plt.colorbar(im, label="per-pixel RMSE")
        plt.tight_layout(); self._pop_plot(fig)

    # =================================================================
    #  ML handlers (lazy torch)
    # =================================================================
    def _require_flow(self):
        if not self.of_result:
            _warn(self, "No flow", "Run optical flow first."); return None
        n = self.of_result['frames']; h = self.of_result['roi_h']; w = self.of_result['roi_w']
        u = np.asarray(np.memmap(self.of_result['u_mmap_path'], np.float32, mode='r', shape=(n, h, w)))
        v = np.asarray(np.memmap(self.of_result['v_mmap_path'], np.float32, mode='r', shape=(n, h, w)))
        return u, v, n, h, w

    def _ml_device(self):
        """Map the Device combo to a torch device string ('' / 'cpu' / 'cuda')."""
        choice = self.ml_device_combo.currentText()
        if choice == "CPU":
            return "cpu"
        if choice == "CUDA":
            return "cuda"
        return None  # Auto

    def _ml_start_train(self, model_name):
        """Common preamble: seed RNGs, clear log, show progress, switch to log tab."""
        if self.ml_seed_chk.isChecked():
            _get_ml_models().set_seed(int(self.ml_seed_spin.value()))
        self.ml_log.clear()
        self._ml_log(f"=== {model_name} ===")
        dev = self._ml_device() or "auto"
        self._ml_log(f"  device={dev}"
                     + (f"  seed={self.ml_seed_spin.value()}" if self.ml_seed_chk.isChecked() else "  seed=off"))
        self.ml_progress.show(); self.ml_progress.setValue(0)
        self.ml_tabs.setCurrentIndex(0)  # log tab
        QtWidgets.QApplication.processEvents()

    def _ml_finish(self, model_name, pred_dict, model_obj=None):
        """Store predictions (and the trained model for export), update viz."""
        self.ml_progress.setValue(100); self.ml_progress.hide()
        pred_dict['model_name'] = model_name
        self.ml_pred = pred_dict
        self.ml_model_obj = model_obj
        self.ml_model_name = model_name
        self._refresh_ml_export_btn()
        if pred_dict.get('n_frames', 1) > 1:
            self.ml_viz_slider.setMinimum(1)
            self.ml_viz_slider.setMaximum(pred_dict['n_frames'])
            self.ml_viz_slider.setValue(1)
        else:
            self.ml_viz_slider.setMinimum(1); self.ml_viz_slider.setMaximum(1)
        self.ml_tabs.setCurrentIndex(1)  # viz tab
        self._ml_update_viz()
        self._ml_log("Done. Switch to Visualization tab to see predictions.")

    def _refresh_ml_export_btn(self):
        has_model = self.ml_model_obj is not None
        self.ml_export_btn.setEnabled(has_model)
        if has_model:
            self.ml_export_btn.setToolTip(
                f"Export the trained {self.ml_model_name} model.")
        else:
            self.ml_export_btn.setToolTip("Train a model first, then export its weights.")

    def _ml_export_model(self):
        if self.ml_model_obj is None:
            _warn(self, "No trained model", "Train a model before exporting."); return
        formats = []
        if self.ml_fmt_pt.isChecked():
            formats.append("pt")
        if self.ml_fmt_ts.isChecked():
            formats.append("torchscript")
        if self.ml_fmt_onnx.isChecked():
            formats.append("onnx")
        if not formats:
            _warn(self, "No format selected", "Tick at least one export format."); return
        default = f"{(self.ml_model_name or 'model').replace(' ', '_').replace('/', '-').lower()}.pt"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export trained model", default, "PyTorch checkpoint (*.pt);;All (*)")
        if not path:
            return
        # best-effort metadata from the processing pipeline
        meta = None
        try:
            meta = self._meta()
        except Exception:
            meta = None
        meta = {"ml_model": self.ml_model_name, "pipeline": meta} if meta else \
               {"ml_model": self.ml_model_name}
        self.status_bar.showMessage(f"Exporting {self.ml_model_name}…")
        QtWidgets.QApplication.processEvents()
        try:
            report = _get_ml_models().export_model(
                self.ml_model_obj, path, formats=tuple(formats), metadata=meta)
        except Exception as e:
            _err(self, "Export error", str(e)); return
        self._ml_log("=== Model export ===")
        self._ml_log(self._format_export_report(report))
        _info_dialog(self, "Model exported",
                     f"Saved to:\n{path}\n\n{self._format_export_report(report)}")
        self.status_bar.showMessage(f"Model exported: {os.path.basename(path)}")

    @staticmethod
    def _format_export_report(report: dict) -> str:
        lines = []
        for fmt, val in report.items():
            if isinstance(val, dict) and "status" in val:
                lines.append(f"{fmt}: {val['status']}")
            elif isinstance(val, dict):
                lines.append(f"{fmt}:")
                for mod, st in val.items():
                    status = st.get("status", st) if isinstance(st, dict) else st
                    lines.append(f"    {mod}: {status}")
            else:
                lines.append(f"{fmt}: {val}")
        return "\n".join(lines)

    def run_pinn(self):
        ml = _get_ml_models()
        if not ml.torch_available():
            _warn(self, "PyTorch missing", "Install torch to use PINN."); return
        data = self._require_flow()
        if data is None: return
        u, v, n, h, w = data
        # read params
        steps = self._ml_param('pinn', 'steps')
        hidden = self._ml_param('pinn', 'hidden')
        layers = self._ml_param('pinn', 'layers')
        nu = self._ml_param('pinn', 'nu')
        lr = self._ml_param('pinn', 'lr')
        lam_data = self._ml_param('pinn', 'lambda_data')
        lam_pde = self._ml_param('pinn', 'lambda_pde')
        frame_idx = min(self._ml_param('pinn', 'frame') - 1, n - 1)
        self._ml_start_train(f"PINN (Navier-Stokes, {steps} steps, hidden={hidden}, layers={layers})")
        self._ml_log(f"  ν={nu:.2e}  lr={lr:.2e}  λ_data={lam_data}  λ_pde={lam_pde}  frame={frame_idx+1}")
        try:
            import torch
            pinn = ml.PINNFlowNet(hidden=hidden, layers=layers, nu=nu, lr=lr,
                                  device=self._ml_device())
            dev = pinn.device
            xs, ys = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
            ts = np.zeros_like(xs)
            x = torch.tensor(xs.ravel(), dtype=torch.float32, device=dev)
            y = torch.tensor(ys.ravel(), dtype=torch.float32, device=dev)
            t = torch.tensor(ts.ravel(), dtype=torch.float32, device=dev)
            ud = torch.tensor(u[frame_idx].ravel(), dtype=torch.float32, device=dev)
            vd = torch.tensor(v[frame_idx].ravel(), dtype=torch.float32, device=dev)
            log_every = max(1, steps // 20)
            def cb(step, total, loss):
                if step % log_every == 0 or step == total - 1:
                    self._ml_log(f"  step {step}/{total}  loss={loss:.4g}")
                self.ml_progress.setValue(int((step + 1) / total * 100))
                QtWidgets.QApplication.processEvents()
            res = pinn.fit(x, y, t, ud, vd, steps=steps,
                           lambda_data=lam_data, lambda_pde=lam_pde,
                           log_every=1, callback=cb)
            up, vp, pp = pinn.predict(x, y, t)
            up = up.reshape(h, w); vp = vp.reshape(h, w); pp = pp.reshape(h, w)
            self._ml_log(f"  final loss: {res['final_loss']:.4g}")
            self._ml_finish("PINN", {
                'u_pred': up, 'v_pred': vp, 'p_pred': pp,
                'loss_history': res['loss_history'],
                'actual_u': u[frame_idx], 'actual_v': v[frame_idx],
                'n_frames': 1, 'h': h, 'w': w,
            }, model_obj=pinn)
        except Exception as e:
            self._ml_log(f"PINN error: {e}"); self.ml_progress.hide()

    def run_ae_sindy(self):
        ml = _get_ml_models()
        if not ml.torch_available():
            _warn(self, "PyTorch missing", "Install torch for AE-SINDy."); return
        data = self._require_flow()
        if data is None: return
        u, v, n, h, w = data
        latent_dim = self._ml_param('ae_sindy', 'latent_dim')
        epochs = self._ml_param('ae_sindy', 'epochs')
        lr = self._ml_param('ae_sindy', 'lr')
        threshold = self._ml_param('ae_sindy', 'threshold')
        self._ml_start_train(f"Autoencoder-SINDy (latent={latent_dim}, epochs={epochs})")
        snap = np.stack([u, v], axis=-1)
        try:
            ae = ml.AutoencoderSINDy(latent_dim=latent_dim, epochs=epochs, lr=lr,
                                     threshold=threshold, device=self._ml_device())
            log_every = max(1, epochs // 20)
            def cb(ep, total, loss):
                if ep % log_every == 0 or ep == total - 1:
                    self._ml_log(f"  epoch {ep}/{total}  loss={loss:.4g}")
                self.ml_progress.setValue(int((ep + 1) / total * 100))
                QtWidgets.QApplication.processEvents()
            res = ae.fit(snap, dt=1.0 / self.of_result['FPS'], callback=cb)
            self._ml_log(f"  recon loss: {res['recon_loss']:.4g}")
            # reconstruct first frame as prediction
            traj = ae.predict_trajectory(1)
            if traj.ndim == 3:  # (1, dim)
                recon = traj[0][:h*w*2].reshape(h, w, 2) if traj[0].size == h*w*2 else np.zeros((h,w,2))
            else:
                recon = traj[0, :h, :w, :] if traj.ndim == 4 else np.zeros((h, w, 2))
            self._ml_finish("Autoencoder-SINDy", {
                'u_pred': recon[..., 0], 'v_pred': recon[..., 1], 'p_pred': None,
                'loss_history': res['loss_history'],
                'actual_u': u[0], 'actual_v': v[0],
                'n_frames': 1, 'h': h, 'w': w,
            }, model_obj=ae)
        except Exception as e:
            self._ml_log(f"AE-SINDy error: {e}"); self.ml_progress.hide()

    def run_fno(self):
        ml = _get_ml_models()
        if not ml.torch_available():
            _warn(self, "PyTorch missing", "Install torch for FNO."); return
        data = self._require_flow()
        if data is None: return
        u, v, n, h, w = data
        modes = self._ml_param('fno', 'modes')
        width = self._ml_param('fno', 'width')
        n_layers = self._ml_param('fno', 'n_layers')
        epochs = self._ml_param('fno', 'epochs')
        lr = self._ml_param('fno', 'lr')
        self._ml_start_train(f"FNO2D (modes={modes}, width={width}, layers={n_layers}, epochs={epochs})")
        try:
            X = np.stack([u[:-1], v[:-1]], axis=1)
            Y = np.stack([u[1:], v[1:]], axis=1)
            fno = ml.FNO2D(modes=modes, width=width, n_layers=n_layers, lr=lr,
                           device=self._ml_device())
            log_every = max(1, epochs // 20)
            def cb(ep, total, loss):
                if ep % log_every == 0 or ep == total - 1:
                    self._ml_log(f"  epoch {ep}/{total}  loss={loss:.4g}")
                self.ml_progress.setValue(int((ep + 1) / total * 100))
                QtWidgets.QApplication.processEvents()
            res = fno.fit(X, Y, epochs=epochs, callback=cb)
            pred = fno.predict(X[:1])  # predict next frame from first
            pred_2d = pred[0]  # (2, H, W)
            self._ml_log(f"  final loss: {res['loss']:.4g}")
            self._ml_finish("FNO", {
                'u_pred': pred_2d[0], 'v_pred': pred_2d[1], 'p_pred': None,
                'loss_history': res['loss_history'],
                'actual_u': u[1], 'actual_v': v[1],
                'n_frames': 1, 'h': h, 'w': w,
            }, model_obj=fno)
        except Exception as e:
            self._ml_log(f"FNO error: {e}"); self.ml_progress.hide()

    def run_deeponet(self):
        ml = _get_ml_models()
        if not ml.torch_available():
            _warn(self, "PyTorch missing", "Install torch for DeepONet."); return
        data = self._require_flow()
        if data is None: return
        u, v, n, h, w = data
        n_sensors = self._ml_param('deeponet', 'n_sensors')
        hidden = self._ml_param('deeponet', 'hidden')
        epochs = self._ml_param('deeponet', 'epochs')
        lr = self._ml_param('deeponet', 'lr')
        self._ml_start_train(f"DeepONet (sensors={n_sensors}, hidden={hidden}, epochs={epochs})")
        try:
            n_sens = min(n_sensors, h * w)
            sensors = u.reshape(n, -1)[:, :n_sens]
            yy, xx = np.mgrid[0:h, 0:w]
            Y = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32) / max(h, w)
            S = u[0].ravel()[:, None]
            don = ml.DeepONet(n_sensors=n_sens, hidden=hidden, lr=lr,
                              device=self._ml_device())
            log_every = max(1, epochs // 20)
            def cb(ep, total, loss):
                if ep % log_every == 0 or ep == total - 1:
                    self._ml_log(f"  epoch {ep}/{total}  loss={loss:.4g}")
                self.ml_progress.setValue(int((ep + 1) / total * 100))
                QtWidgets.QApplication.processEvents()
            res = don.fit(np.tile(sensors[0], (Y.shape[0], 1)), Y, S, epochs=epochs, callback=cb)
            pred = don.predict(np.tile(sensors[0], (Y.shape[0], 1)), Y).ravel().reshape(h, w)
            self._ml_log(f"  final loss: {res['loss']:.4g}")
            self._ml_finish("DeepONet", {
                'u_pred': pred, 'v_pred': v[0], 'p_pred': None,
                'loss_history': res['loss_history'],
                'actual_u': u[0], 'actual_v': v[0],
                'n_frames': 1, 'h': h, 'w': w,
            }, model_obj=don)
        except Exception as e:
            self._ml_log(f"DeepONet error: {e}"); self.ml_progress.hide()

    def run_convlstm(self):
        ml = _get_ml_models()
        if not ml.torch_available():
            _warn(self, "PyTorch missing", "Install torch for ConvLSTM."); return
        data = self._require_flow()
        if data is None: return
        u, v, n, h, w = data
        hidden = self._ml_param('convlstm', 'hidden')
        kernel = self._ml_param('convlstm', 'kernel')
        epochs = self._ml_param('convlstm', 'epochs')
        horizon = self._ml_param('convlstm', 'horizon')
        lr = self._ml_param('convlstm', 'lr')
        self._ml_start_train(f"ConvLSTM (hidden={hidden}, kernel={kernel}, epochs={epochs}, horizon={horizon})")
        seq = np.stack([u, v], axis=-1)
        try:
            m = ml.ConvLSTMSeq(hidden=hidden, kernel=kernel, epochs=epochs, lr=lr,
                               device=self._ml_device())
            log_every = max(1, epochs // 20)
            def cb(ep, total, loss):
                if ep % log_every == 0 or ep == total - 1:
                    self._ml_log(f"  epoch {ep}/{total}  loss={loss:.4g}")
                self.ml_progress.setValue(int((ep + 1) / total * 100))
                QtWidgets.QApplication.processEvents()
            res = m.fit(seq, horizon=horizon, callback=cb)
            self._ml_log(f"  final loss: {res['loss']:.4g}")
            self._ml_finish("ConvLSTM", {
                'u_pred': u[0], 'v_pred': v[0], 'p_pred': None,
                'loss_history': res['loss_history'],
                'actual_u': u[0], 'actual_v': v[0],
                'n_frames': 1, 'h': h, 'w': w,
            }, model_obj=m)
        except Exception as e:
            self._ml_log(f"ConvLSTM error: {e}"); self.ml_progress.hide()

    def run_vae(self):
        ml = _get_ml_models()
        if not ml.torch_available():
            _warn(self, "PyTorch missing", "Install torch for VAE."); return
        data = self._require_flow()
        if data is None: return
        u, v, n, h, w = data
        latent_dim = self._ml_param('vae', 'latent_dim')
        hidden = self._ml_param('vae', 'hidden')
        beta = self._ml_param('vae', 'beta')
        epochs = self._ml_param('vae', 'epochs')
        lr = self._ml_param('vae', 'lr')
        self._ml_start_train(f"VAE (latent={latent_dim}, hidden={hidden}, β={beta}, epochs={epochs})")
        snap = np.stack([u, v], axis=-1).reshape(n, -1)
        try:
            vae = ml.FlowVAE(latent_dim=latent_dim, hidden=hidden, beta=beta, epochs=epochs,
                             lr=lr, device=self._ml_device())
            log_every = max(1, epochs // 20)
            def cb(ep, total, loss):
                if ep % log_every == 0 or ep == total - 1:
                    self._ml_log(f"  epoch {ep}/{total}  loss={loss:.4g}")
                self.ml_progress.setValue(int((ep + 1) / total * 100))
                QtWidgets.QApplication.processEvents()
            res = vae.fit(snap, callback=cb)
            self._ml_log(f"  recon={res['recon']:.4g}  kld={res['kld']:.4g}")
            self._ml_finish("VAE", {
                'u_pred': u[0], 'v_pred': v[0], 'p_pred': None,
                'loss_history': res['loss_history'],
                'actual_u': u[0], 'actual_v': v[0],
                'n_frames': 1, 'h': h, 'w': w,
            }, model_obj=vae)
        except Exception as e:
            self._ml_log(f"VAE error: {e}"); self.ml_progress.hide()

    def run_ensemble(self):
        if 'model' not in self.sindy_result:
            _warn(self, "No model", "Fit SINDy first."); return
        n_models = self._ml_param('ensemble', 'n_models')
        subsample = self._ml_param('ensemble', 'subsample')
        total = self.sindy_result['total_points']
        X = np.memmap(self.sindy_result['X_optical_path'], np.float32, mode='r', shape=(total, 8))
        cfg = self._sindy_config()
        h = self.sindy_result['roi_h']; w = self.sindy_result['roi_w']
        n = self.sindy_result['n_frames']
        self._ml_start_train(f"Ensemble ({n_models} models, subsample={subsample})")
        preds = []
        members = []
        for k in range(n_models):
            idx = np.random.default_rng(k).choice(total, min(total, subsample), replace=True)
            fit = _get_sindy_core().fit_sindy(X[idx], np.memmap(self.sindy_result['X_dot_optical_path'],
                np.float32, mode='r', shape=(total, 2))[idx], self.sindy_result['DT'], cfg)
            members.append(fit["model"])
            preds.append(lambda X, m=fit["model"]: _get_sindy_core().predict_sindy(m, X, flip=False))
            self._ml_log(f"  model {k+1}/{n_models} fitted")
            self.ml_progress.setValue(int((k + 1) / n_models * 100))
            QtWidgets.QApplication.processEvents()
        n_eval = min(total, subsample)
        mean, std = _get_ml_models().ensemble_uncertainty(preds, X[:n_eval], n_models=n_models)
        rmse = np.sqrt(np.mean(mean ** 2))
        self._ml_log(f"  ensemble RMSE={rmse:.4g}  mean std={np.mean(std):.4g}")
        # reshape to field for visualization
        mean_field = mean[:h*w].reshape(h, w, 2) if mean.ndim == 2 else mean[:h*w].reshape(h, w)
        std_field = std[:h*w].reshape(h, w, 2) if std.ndim == 2 else std[:h*w].reshape(h, w)
        self._ml_finish("Ensemble", {
            'u_pred': mean_field[..., 0] if mean_field.ndim == 3 else mean_field,
            'v_pred': mean_field[..., 1] if mean_field.ndim == 3 else (
                np.memmap(self.of_result['v_mmap_path'], np.float32, mode='r',
                          shape=(n, h, w))[0] if self.of_result else np.zeros((h, w))),
            'p_pred': std_field[..., 0] if std_field.ndim == 3 else std_field,
            'loss_history': [],
            'actual_u': np.memmap(self.of_result['u_mmap_path'], np.float32, mode='r',
                                  shape=(n, h, w))[0] if self.of_result else np.zeros((h, w)),
            'actual_v': np.memmap(self.of_result['v_mmap_path'], np.float32, mode='r',
                                  shape=(n, h, w))[0] if self.of_result else np.zeros((h, w)),
            'n_frames': 1, 'h': h, 'w': w,
        }, model_obj=types.SimpleNamespace(
            model_type="SINDyEnsemble", n_models=n_models,
            subsample=subsample, members=members))

    def run_gan(self):
        ml = _get_ml_models()
        if not ml.torch_available():
            _warn(self, "PyTorch missing", "Install torch for GAN."); return
        data = self._require_flow()
        if data is None: return
        u, v, n, h, w = data
        noise_dim = self._ml_param('gan', 'noise_dim')
        hidden = self._ml_param('gan', 'hidden')
        epochs = self._ml_param('gan', 'epochs')
        lr = self._ml_param('gan', 'lr')
        self._ml_start_train(f"GAN (noise={noise_dim}, hidden={hidden}, epochs={epochs})")
        samples = np.stack([u, v], axis=-1).reshape(n, -1)[:, :hidden]
        try:
            gan = ml.FlowGAN(noise_dim=noise_dim, hidden=hidden, lr=lr,
                             device=self._ml_device())
            log_every = max(1, epochs // 20)
            def cb(ep, total, loss):
                if ep % log_every == 0 or ep == total - 1:
                    self._ml_log(f"  epoch {ep}/{total}  lossG={loss:.4g}")
                self.ml_progress.setValue(int((ep + 1) / total * 100))
                QtWidgets.QApplication.processEvents()
            res = gan.fit(samples, epochs=epochs, callback=cb)
            self._ml_log(f"  lossD={res['lossD']:.4g}  lossG={res['lossG']:.4g}")
            self._ml_finish("GAN", {
                'u_pred': u[0], 'v_pred': v[0], 'p_pred': None,
                'loss_history': res['loss_history'],
                'actual_u': u[0], 'actual_v': v[0],
                'n_frames': 1, 'h': h, 'w': w,
            }, model_obj=gan)
        except Exception as e:
            self._ml_log(f"GAN error: {e}"); self.ml_progress.hide()

    def run_causal(self):
        data = self._require_flow()
        if data is None: return
        u, v, n, h, w = data
        max_lag = self._ml_param('causal', 'max_lag')
        self._ml_start_train(f"Causal / Granger (max_lag={max_lag})")
        cy, cx = h // 2, w // 2
        f = _get_ml_models().granger_causality(u[:, cy, cx], v[:, cy, cx], max_lag=max_lag)
        self._ml_log(f"Granger causality u->v at centre: F={f:.4g}")
        if 'model' in self.sindy_result:
            try:
                coef = np.asarray(self.sindy_result['model'].coefficients())
                names = self.sindy_result['model'].get_feature_names()
                summ = _get_ml_models().causal_sindy_summary(coef, names)
                self._ml_log("Causal SINDy drivers: " + str(summ))
            except Exception as e:
                self._ml_log(f"Causal summary error: {e}")
        self.ml_progress.hide()
        # no field prediction to visualize for causal; show a placeholder
        self._ml_finish("Causal", {
            'u_pred': u[0], 'v_pred': v[0], 'p_pred': None,
            'loss_history': [],
            'actual_u': u[0], 'actual_v': v[0],
            'n_frames': 1, 'h': h, 'w': w,
        })

    # ----- ML visualization -----
    def _ml_update_viz(self, *_):
        if self.ml_pred is None:
            self.ml_viz_fig.clear()
            ax = self.ml_viz_fig.add_subplot(111)
            ax.set_facecolor(Theme.BG_BASE)
            ax.text(0.5, 0.5, "Run a model to see predictions",
                    transform=ax.transAxes, ha='center', va='center',
                    color=Theme.TEXT_FAINT, fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
            self.ml_viz_canvas.draw_idle()
            return
        p = self.ml_pred
        frame_idx = max(0, self.ml_viz_slider.value() - 1)
        n = p.get('n_frames', 1)
        self.ml_viz_frame_label.setText(f"{frame_idx+1} / {n}")
        viz_type = self.ml_viz_type_combo.currentIndex()
        self.ml_viz_fig.clear()
        h, w = p['h'], p['w']

        if viz_type == 3:  # loss curve
            ax = self.ml_viz_fig.add_subplot(111)
            ax.set_facecolor(Theme.BG_BASE)
            lh = p.get('loss_history', [])
            if lh:
                ax.plot(lh, color=Theme.ACCENT, linewidth=1.5)
                ax.set_xlabel("epoch / step"); ax.set_ylabel("loss")
                ax.set_title(f"Training Loss — {p['model_name']}")
                ax.set_yscale('log')
            else:
                ax.text(0.5, 0.5, "No loss history available",
                        transform=ax.transAxes, ha='center', va='center',
                        color=Theme.TEXT_FAINT, fontsize=12)
                ax.set_xticks([]); ax.set_yticks([])
        elif viz_type == 0:  # quiver actual vs pred
            ax1 = self.ml_viz_fig.add_subplot(121)
            ax2 = self.ml_viz_fig.add_subplot(122)
            step = max(h // 25, 2)
            ys = np.arange(0, h, step); xs = np.arange(0, w, step)
            xg, yg = np.meshgrid(xs, ys)
            au = p['actual_u']; av = p['actual_v']
            pu = p['u_pred']; pv = p['v_pred']
            for ax, field, title, color in (
                (ax1, (au, av), "Actual", Theme.ACCENT),
                (ax2, (pu, pv), f"{p['model_name']} Pred", Theme.MAGENTA)):
                ax.set_facecolor(Theme.BG_BASE)
                ax.quiver(xg, yg,
                          field[0][ys[:, None], xs[None, :]],
                          field[1][ys[:, None], xs[None, :]],
                          color=color, angles='xy', scale=1.0, width=0.006)
                ax.set_title(title); ax.set_aspect('equal')
            self.ml_viz_fig.suptitle(f"Quiver — {p['model_name']}", color=Theme.TEXT)
        elif viz_type == 1:  # magnitude heatmap
            ax1 = self.ml_viz_fig.add_subplot(121)
            ax2 = self.ml_viz_fig.add_subplot(122)
            au = p['actual_u']; av = p['actual_v']
            pu = p['u_pred']; pv = p['v_pred']
            mag_a = np.sqrt(au**2 + av**2)
            mag_p = np.sqrt(pu**2 + pv**2)
            for ax, mag, title in (
                (ax1, mag_a, "Actual |V|"),
                (ax2, mag_p, f"{p['model_name']} |V|")):
                ax.set_facecolor(Theme.BG_BASE)
                im = ax.imshow(mag, cmap=self.cmap_combo.currentText(), origin='upper')
                ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
                self.ml_viz_fig.colorbar(im, ax=ax, fraction=0.046)
            self.ml_viz_fig.suptitle(f"Magnitude — {p['model_name']}", color=Theme.TEXT)
        elif viz_type == 2:  # error map
            ax = self.ml_viz_fig.add_subplot(111)
            ax.set_facecolor(Theme.BG_BASE)
            au = p['actual_u']; av = p['actual_v']
            pu = p['u_pred']; pv = p['v_pred']
            err = np.sqrt((au - pu)**2 + (av - pv)**2)
            im = ax.imshow(err, cmap='inferno', origin='upper')
            ax.set_title(f"Error Map — {p['model_name']}")
            ax.set_xticks([]); ax.set_yticks([])
            self.ml_viz_fig.colorbar(im, ax=ax, fraction=0.046)
        self.ml_viz_fig.patch.set_facecolor(Theme.SURFACE)
        self.ml_viz_fig.tight_layout()
        self.ml_viz_canvas.draw_idle()
        QtWidgets.QApplication.processEvents()

    # =================================================================
    #  Export handlers
    # =================================================================
    def _mmaps(self):
        n = self.sindy_result['n_frames']; h = self.sindy_result['roi_h']; w = self.sindy_result['roi_w']
        u = np.memmap(self.of_result['u_mmap_path'], np.float32, mode='r', shape=(n, h, w))
        v = np.memmap(self.of_result['v_mmap_path'], np.float32, mode='r', shape=(n, h, w))
        fr = np.memmap(self.of_result['frame_mmap_path'], np.uint8, mode='r', shape=(n, h, w))
        return u, v, fr, n, h, w

    def _meta(self):
        return _get_export().build_metadata(self.result, self.of_result, self.sindy_result)

    def export_csv_gui(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Output dir", "./output")
        if not d: return
        u, v, fr, n, h, w = self._mmaps()
        _get_export().export_csv(d, u, v, self.X_pred_optical,
            progress_cb=lambda i, n, s: self.status_bar.showMessage(f"CSV {i}/{n}"))
        _info_dialog(self, "Exported", f"CSV files saved to:\n{d}")

    def export_hdf5_gui(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save HDF5", "result.h5", "HDF5 (*.h5)")
        if not p: return
        u, v, fr, n, h, w = self._mmaps()
        try:
            _get_export().export_hdf5(p, u, v, fr, self.X_pred_optical, self._meta())
            _info_dialog(self, "Exported", p)
        except Exception as e: _err(self, "HDF5 error", str(e))

    def export_netcdf_gui(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save NetCDF", "result.nc", "NetCDF (*.nc)")
        if not p: return
        u, v, fr, n, h, w = self._mmaps()
        try:
            _get_export().export_netcdf(p, u, v, self.X_pred_optical, self._meta())
            _info_dialog(self, "Exported", p)
        except Exception as e: _err(self, "NetCDF error", str(e))

    def export_parquet_gui(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Parquet", "result.parquet", "Parquet (*.parquet)")
        if not p: return
        u, v, fr, n, h, w = self._mmaps()
        try:
            _get_export().export_parquet(p, u, v, self.X_pred_optical)
            _info_dialog(self, "Exported", p)
        except Exception as e: _err(self, "Parquet error", str(e))

    def export_metadata_gui(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save metadata", "metadata.json", "JSON (*.json)")
        if not p: return
        _get_export().export_metadata(p, self._meta())
        _info_dialog(self, "Exported", p)

    def export_pdf_gui(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save report", "report.pdf", "PDF (*.pdf)")
        if not p: return
        u, v, fr, n, h, w = self._mmaps()
        try:
            _get_export().export_pdf_report(p, u, v, self.X_pred_optical, self._meta())
            _info_dialog(self, "Exported", p)
        except Exception as e: _err(self, "PDF error", str(e))

    def export_images_gui(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Image output dir", "./frames")
        if not d: return
        u, v, fr, n, h, w = self._mmaps()
        _get_export().export_image_sequence(d, fr, progress_cb=lambda i, n, s: self.status_bar.showMessage(f"img {i}/{n}"))
        _info_dialog(self, "Exported", f"Frames saved to:\n{d}")

    def export_animation_gui(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save animation", "quiver.mp4", "MP4 (*.mp4)")
        if not p: return
        scale, ok = QtWidgets.QInputDialog.getDouble(self, "Quiver scale", "scale:", 1, 0.1, 150, 2)
        if not ok: return
        u, v, fr, n, h, w = self._mmaps()
        self._set_pill("● RENDERING", "busy")
        try:
            _get_export().export_animation(p, u, v, fr, self.X_pred_optical,
                self.result['meters_per_pixel'], scale=scale,
                progress_cb=lambda i, n, s: self.status_bar.showMessage(f"render {i}/{n}"))
            _info_dialog(self, "Done", f"Animation saved:\n{p}")
        except Exception as e:
            _err(self, "Animation error", str(e))
        self._set_pill("● PREDICTION READY", "done")

    # =================================================================
    #  Project / presets
    # =================================================================
    def save_project(self):
        if not self.result:
            _warn(self, "Nothing to save", "Set up a video + ROI first."); return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save project", "project.trsindy",
                                                     "TR-SINDy project (*.trsindy)")
        if not p: return
        state = {
            "video_file": self.result.get("video_file"),
            "roi_box": list(self.result.get("roi_box", [])),
            "calibration_px": self.result.get("calibration_px"),
            "calibration_m": self.result.get("calibration_m"),
            "meters_per_pixel": self.result.get("meters_per_pixel"),
            "optical_flow": self._flow_config().__dict__,
            "sindy": self._sindy_config().__dict__,
            "history": self.history.to_list(),
        }
        mmaps = self.of_result.get("mmap_paths")
        project.save_project(p, state, mmaps, bundle_mmaps=True)
        self._refresh_recent_menu()
        _info_dialog(self, "Saved", f"Project saved:\n{p}")

    def load_project(self, path: str = None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load project", ".",
                                                            "TR-SINDy project (*.trsindy)")
        if not path: return
        state = project.load_project(path)
        self.result = {
            "video_file": state.get("video_file"),
            "roi_box": tuple(state.get("roi_box", [])),
            "calibration_px": state.get("calibration_px"),
            "calibration_m": state.get("calibration_m"),
            "meters_per_pixel": state.get("meters_per_pixel"),
        }
        self.finished = True
        if self.result.get("video_file"):
            self.file_edit.setText(self.result["video_file"])
        mpp = self.result.get("meters_per_pixel")
        if mpp:
            self.mpp_label.setText(f"Calibration: 1 px = {mpp:.8f} m (loaded from project)")
        self.process_btn.setEnabled(True)
        self.history = project.ProcessingHistory.from_list(state.get("history", []))
        bundled = state.get("bundled_mmaps", {})
        if bundled:
            self.of_result = {"mmap_paths": bundled,
                              "u_mmap_path": bundled.get("u"),
                              "v_mmap_path": bundled.get("v"),
                              "frame_mmap_path": bundled.get("frames")}
            self.status_bar.showMessage("Project loaded (memmaps bundled). Re-run SINDy to model.")
        self._refresh_recent_menu()
        _info_dialog(self, "Loaded", f"Project loaded:\n{path}")

    def apply_preset(self):
        names = project.list_presets()
        if not names:
            project.install_builtin_presets(); names = project.list_presets()
        name, ok = QtWidgets.QInputDialog.getItem(self, "Apply preset", "Preset:", names, 0, False)
        if not ok: return
        try:
            p = project.load_preset(name)
        except Exception as e:
            _err(self, "Preset error", str(e)); return
        of = p.optical_flow
        if of:
            if "backend" in of and self.backend_combo.findText(of["backend"]) >= 0:
                self.backend_combo.setCurrentText(of["backend"])
            for k, w in (("temporal_smoothing", self.ts_combo),):
                if k in of: w.setCurrentText(of[k])
            if "temporal_alpha" in of: self.alpha_spin.setValue(of["temporal_alpha"])
            if "multiscale" in of: self.multiscale_chk.setChecked(of["multiscale"])
        sd = p.sindy
        if sd:
            if "library" in sd: self.lib_combo.setCurrentText(sd["library"])
            if "degree" in sd: self.degree_spin.setValue(sd["degree"])
            if "threshold" in sd: self.thresh_spin.setValue(sd["threshold"])
        _info_dialog(self, "Preset applied", f"Applied preset '{name}'.")

    def save_preset(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Preset name", "Name:")
        if not ok or not name: return
        p = project.Preset(name, "user preset",
                           optical_flow=self._flow_config().__dict__,
                           sindy=self._sindy_config().__dict__)
        project.save_preset(p)
        _info_dialog(self, "Saved", f"Preset '{name}' saved.")

    # =================================================================
    #  Help / about
    # =================================================================
    def show_help(self):
        _info_dialog(self, "Workflow Guide",
            "1. File → Open Video (Ctrl+O)\n"
            "2. Select ROI & Calibrate\n"
            "3. Configure optical flow backend & SINDy library\n"
            "4. ① Process Optical Flow\n"
            "5. ② Run SINDy Modeling (optionally cross-validate / compare)\n"
            "6. ③ Run SINDy Prediction\n"
            "7. Visualize tab: quiver, vorticity, POD, DMD, spectra…\n"
            "8. ML Models tab: PINN, Autoencoder-SINDy, FNO, ConvLSTM, VAE…\n"
            "   · export trained models to .pt / TorchScript / ONNX\n"
            "9. Export tab: CSV / HDF5 / NetCDF / Parquet / PDF / animation\n"
            "Ctrl+S save project · Ctrl+L load · Ctrl+T toggle theme")

    def show_about_dialog(self):
        from . import __version__ as _v
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("About — Turbulence Realm SINDy")
        dlg.setFixedSize(460, 480)
        v = QtWidgets.QVBoxLayout(dlg)
        v.setContentsMargins(24, 24, 24, 24)
        v.setSpacing(8)
        mark = QtWidgets.QLabel("⬡")
        mark.setStyleSheet(
            f"color: {Theme.ACCENT}; font-size: 64pt; font-weight: bold;")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(mark)
        t = QtWidgets.QLabel("Turbulence Realm — SINDy")
        t.setStyleSheet(
            f"font-size: 18pt; font-weight: 800; color: {Theme.TEXT};")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter); v.addWidget(t)
        info = QtWidgets.QLabel(
            f"<p style='text-align:center; color:{Theme.TEXT_MUTED}; font-size:10pt'>"
            f"Version {_v}<br><em>Video-based Fluid Flow Analysis Platform</em></p>"
            f"<p style='text-align:center; font-size:9pt; color:{Theme.TEXT_FAINT}'>"
            "Optical Flow · SINDy · PINN · Neural Operators · POD/DMD<br>"
            "Developed by Fayaz Rasheed<br>"
            f"<a style='color:{Theme.ACCENT}' href='http://www.turbulencerealm.com'>"
            "www.turbulencerealm.com</a></p>")
        info.setOpenExternalLinks(True)
        info.setAlignment(Qt.AlignmentFlag.AlignCenter); v.addWidget(info)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        bb.accepted.connect(dlg.accept); v.addWidget(bb, alignment=Qt.AlignmentFlag.AlignCenter)
        dlg.exec()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(ev)

    # =================================================================
    #  Settings dialog
    # =================================================================
    def show_settings(self):
        from .settings_dialog import SettingsDialog, save_settings
        dlg = SettingsDialog(self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            new = dlg.get_settings()
            save_settings(new)
            self._mmap_dir = new["mmap_dir"]
            if new["theme"] != self._theme_name:
                self._toggle_theme()
            self.status_bar.showMessage("Settings saved.")

    # =================================================================
    #  Session restore / auto-save
    # =================================================================
    _SESSION_FILE = os.path.join(os.path.expanduser("~"), ".tr_sindy", "session.json")

    def _save_session(self):
        """Save current session state for restore on next launch."""
        try:
            state = {
                "video_file": self.result.get("video_file"),
                "roi_box": self.result.get("roi_box"),
                "calibration_px": self.result.get("calibration_px"),
                "calibration_m": self.result.get("calibration_m"),
                "meters_per_pixel": self.result.get("meters_per_pixel"),
                "finished": getattr(self, "finished", False),
                "current_page": self.pages.currentIndex() if hasattr(self, "pages") else 0,
            }
            os.makedirs(os.path.dirname(self._SESSION_FILE), exist_ok=True)
            with open(self._SESSION_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log.debug("could not save session: %s", e)

    def _restore_session(self):
        """Offer to restore the previous session on startup."""
        if not os.path.exists(self._SESSION_FILE):
            return
        try:
            with open(self._SESSION_FILE) as f:
                state = json.load(f)
        except Exception:
            return
        if not state.get("video_file"):
            return
        vf = state["video_file"]
        if not os.path.exists(vf):
            return
        reply = QtWidgets.QMessageBox.question(
            self, "Restore session",
            f"Restore last session?\n\nVideo: {os.path.basename(vf)}",
            QtWidgets.QMessageBox.StandardButton.Yes |
            QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes)
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.result = dict(state)
            self.finished = state.get("finished", False)
            if self.finished:
                self.process_btn.setEnabled(True)
            if hasattr(self, "pages") and "current_page" in state:
                self._select_nav(state["current_page"])
            self.status_bar.showMessage(f"Session restored: {os.path.basename(vf)}")

    def closeEvent(self, ev):
        """Auto-save session on close."""
        self._save_session()
        super().closeEvent(ev)

    # =================================================================
    #  Cancel + ETA for long-running jobs
    # =================================================================
    def _make_progress_cb(self, stage="processing"):
        """Return a progress callback with ETA display and cancel check."""
        import time as _time
        start = _time.time()
        last_fps_update = [start]

        def pcb(i, n, s):
            elapsed = _time.time() - start
            self.progress.setValue(int(i / n * 100))
            if i > 0 and elapsed > 0:
                fps = i / elapsed
                remaining = (n - i) / fps if fps > 0 else 0
                if _time.time() - last_fps_update[0] > 0.5:
                    eta_m, eta_s = divmod(int(remaining), 60)
                    self.status_bar.showMessage(
                        f"{s}: {i}/{n}  ~{eta_m}m{eta_s:02d}s remaining")
                    last_fps_update[0] = _time.time()
            QtWidgets.QApplication.processEvents()
        return pcb
