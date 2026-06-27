"""
Turbulence Realm - SINDy
========================
Video-based fluid-flow analysis: dense optical flow + SINDy (Sparse
Identification of Nonlinear Dynamics) modelling, prediction, visualisation
and export.

This is a UI modernisation of the original TR-SINDY-Final.py. The full
processing pipeline (ROI/calibration -> Farneback optical flow -> SINDy fit
-> batch prediction -> quiver/contour/streamline/error analysis -> CSV &
animation export) is preserved byte-for-byte in behaviour; only the
presentation layer has been reskinned into a futuristic dark console.
"""

import math
import os
import sys
import time

import cv2
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysindy as ps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplashScreen,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from pysindy.feature_library import PolynomialLibrary
from pysindy.optimizers import STLSQ
from scipy.ndimage import gaussian_filter


# =====================================================================
#  Design tokens — "Flux Reactor" futuristic console
#  Deep space-black surfaces, an electric violet→cyan plasma accent,
#  and a hot-magenta signal reserved for the SINDy / prediction layer.
# =====================================================================
class Theme:
    BG_BASE      = "#070B14"   # deepest void
    BG_GRAD_TOP  = "#0C1422"   # header / gradient lift
    SURFACE      = "#0F1828"   # panels / cards
    SURFACE_HI   = "#16223A"   # inputs, elevated chrome
    SURFACE_GLOW = "#1B2C49"   # hover fills
    HAIRLINE     = "#22324E"   # subtle dividers
    BORDER_HOVER = "#3A557F"   # focus / hover borders

    TEXT         = "#EAF2FF"   # primary text
    TEXT_MUTED   = "#8FA6C4"   # secondary text
    TEXT_FAINT   = "#566783"   # captions / disabled

    ACCENT       = "#46E5FF"   # cyan plasma — primary signal
    ACCENT_2     = "#7B5CFF"   # violet — gradient partner
    ACCENT_DEEP  = "#1C7E9E"   # pressed / shadow
    MAGENTA      = "#FF4FD8"   # SINDy / prediction signal
    AMBER        = "#FFC24B"   # calibration signal
    DANGER       = "#FF5D6C"   # destructive / error
    GOOD         = "#43F5A8"   # success / ready

    UI_FONT   = "Inter"
    MONO_FONT = "JetBrains Mono"

    # OpenCV overlay colours (BGR), matched to the palette above.
    BGR_ACCENT = (255, 229, 70)    # cyan  #46E5FF
    BGR_AMBER  = (75, 194, 255)    # amber #FFC24B
    BGR_MAGENTA = (216, 79, 255)   # magenta #FF4FD8


def _stylesheet():
    t = Theme
    return f"""
    QWidget {{
        font-family: '{t.UI_FONT}', 'Segoe UI', sans-serif;
        color: {t.TEXT};
        font-size: 10pt;
    }}
    QWidget#contentRoot, QMainWindow {{ background-color: {t.BG_BASE}; }}

    /* ---- Header instrument bar ---- */
    QFrame#headerBar {{
        background-color: {t.BG_GRAD_TOP};
        border: none;
        border-bottom: 1px solid {t.HAIRLINE};
    }}
    QLabel#appTitle {{
        font-size: 18pt; font-weight: 800; color: {t.TEXT};
        letter-spacing: -0.5px;
    }}
    QLabel#appSubtitle {{
        font-family: '{t.MONO_FONT}', monospace;
        font-size: 8pt; color: {t.TEXT_MUTED}; letter-spacing: 3px;
    }}
    QLabel#statusPill {{
        font-family: '{t.MONO_FONT}', monospace;
        font-size: 8pt; font-weight: bold; color: {t.ACCENT};
        background-color: rgba(70, 229, 255, 0.10);
        border: 1px solid rgba(70, 229, 255, 0.35);
        border-radius: 11px; padding: 4px 14px; letter-spacing: 1px;
    }}

    /* ---- Tabs ---- */
    QTabWidget::pane {{
        border: 1px solid {t.HAIRLINE};
        border-radius: 12px;
        background: {t.SURFACE};
        top: -1px;
    }}
    QTabBar::tab {{
        background: transparent;
        color: {t.TEXT_MUTED};
        border: 1px solid transparent;
        border-top-left-radius: 9px; border-top-right-radius: 9px;
        min-width: 150px; padding: 10px 22px; margin-right: 4px;
        font-weight: 600; font-size: 9.5pt;
    }}
    QTabBar::tab:selected {{
        color: {t.ACCENT};
        background: {t.SURFACE};
        border: 1px solid {t.HAIRLINE};
        border-bottom-color: {t.SURFACE};
    }}
    QTabBar::tab:hover:!selected {{ color: {t.TEXT}; }}

    /* ---- Cards ---- */
    QGroupBox {{
        background-color: {t.SURFACE};
        border: 1px solid {t.HAIRLINE};
        border-radius: 12px;
        margin-top: 16px; padding: 16px 14px 14px 14px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin; subcontrol-position: top left;
        left: 14px; top: 3px; padding: 0 6px;
        color: {t.ACCENT};
        font-family: '{t.MONO_FONT}', monospace;
        font-size: 8pt; font-weight: bold; letter-spacing: 1px;
    }}

    /* ---- Buttons: quiet ghost default ---- */
    QPushButton {{
        background-color: {t.SURFACE_HI};
        color: {t.TEXT};
        border: 1px solid {t.HAIRLINE};
        border-radius: 9px;
        padding: 11px 16px; min-height: 18px;
        font-size: 10pt; font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {t.SURFACE_GLOW};
        border: 1px solid {t.BORDER_HOVER};
    }}
    QPushButton:pressed {{ background-color: #111B2E; }}
    QPushButton:disabled {{
        background-color: {t.SURFACE};
        color: {t.TEXT_FAINT};
        border: 1px solid {t.HAIRLINE};
    }}

    /* ---- Primary action: plasma cyan ---- */
    QPushButton[variant="primary"] {{
        background-color: {t.ACCENT};
        color: #04222B; border: 1px solid {t.ACCENT};
        font-weight: 800;
    }}
    QPushButton[variant="primary"]:hover {{
        background-color: #6FECFF; border: 1px solid #6FECFF;
    }}
    QPushButton[variant="primary"]:pressed {{
        background-color: {t.ACCENT_DEEP}; color: {t.TEXT};
    }}
    QPushButton[variant="primary"]:disabled {{
        background-color: {t.SURFACE}; color: {t.TEXT_FAINT};
        border: 1px solid {t.HAIRLINE};
    }}

    /* ---- Secondary action: violet ---- */
    QPushButton[variant="violet"] {{
        background-color: rgba(123, 92, 255, 0.16);
        color: #C6B8FF; border: 1px solid rgba(123, 92, 255, 0.45);
        font-weight: 700;
    }}
    QPushButton[variant="violet"]:hover {{
        background-color: rgba(123, 92, 255, 0.26);
    }}
    QPushButton[variant="violet"]:disabled {{
        background-color: {t.SURFACE}; color: {t.TEXT_FAINT};
        border: 1px solid {t.HAIRLINE};
    }}

    QLabel {{ color: {t.TEXT}; font-size: 10pt; background: transparent; }}
    QLabel#fieldLabel {{ color: {t.TEXT_MUTED}; font-size: 9pt; }}
    QLabel#readout {{
        font-family: '{t.MONO_FONT}', monospace; font-size: 9pt;
        color: {t.ACCENT}; background-color: {t.BG_BASE};
        border: 1px solid {t.HAIRLINE}; border-radius: 7px; padding: 8px 12px;
    }}
    QLabel#stepGuide {{
        font-family: '{t.MONO_FONT}', monospace; font-size: 9pt;
        color: {t.TEXT_MUTED}; background-color: {t.BG_BASE};
        border: 1px solid {t.HAIRLINE}; border-radius: 9px;
        padding: 14px 16px; line-height: 150%;
    }}

    QLineEdit {{
        border: 1px solid {t.HAIRLINE}; border-radius: 8px;
        padding: 9px 12px; font-family: '{t.MONO_FONT}', monospace;
        font-size: 10pt; color: {t.TEXT}; background-color: {t.BG_BASE};
        selection-background-color: {t.ACCENT}; selection-color: #04222B;
    }}
    QLineEdit:focus {{ border: 1px solid {t.ACCENT}; }}
    QLineEdit:disabled {{ color: {t.TEXT_FAINT}; background-color: {t.SURFACE}; }}

    QPlainTextEdit, QTextEdit {{
        background-color: {t.BG_BASE}; color: {t.TEXT};
        border: 1px solid {t.HAIRLINE}; border-radius: 8px;
        font-family: '{t.MONO_FONT}', monospace; font-size: 10pt;
        selection-background-color: {t.ACCENT}; selection-color: #04222B;
    }}

    /* ---- Progress: plasma chunk ---- */
    QProgressBar {{
        border: 1px solid {t.HAIRLINE}; border-radius: 8px;
        text-align: center; background-color: {t.BG_BASE};
        color: {t.TEXT_MUTED}; font-family: '{t.MONO_FONT}', monospace;
        font-size: 8.5pt; height: 18px;
    }}
    QProgressBar::chunk {{
        border-radius: 7px;
        background-color: {t.ACCENT};
    }}

    /* ---- Slider ---- */
    QSlider::groove:horizontal {{
        height: 5px; background: {t.HAIRLINE}; border-radius: 2px;
    }}
    QSlider::sub-page:horizontal {{
        background: {t.ACCENT}; border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {t.ACCENT}; width: 16px; height: 16px;
        margin: -6px 0; border-radius: 8px; border: 2px solid {t.BG_BASE};
    }}
    QSlider::handle:horizontal:hover {{ background: #6FECFF; }}

    QStatusBar {{
        border-top: 1px solid {t.HAIRLINE};
        font-family: '{t.MONO_FONT}', monospace; font-size: 9pt;
        background-color: {t.BG_GRAD_TOP}; color: {t.TEXT_MUTED};
    }}
    QStatusBar::item {{ border: none; }}

    QMenuBar {{
        background-color: {t.BG_GRAD_TOP}; color: {t.TEXT};
        border-bottom: 1px solid {t.HAIRLINE}; padding: 2px 6px;
    }}
    QMenuBar::item {{ background: transparent; padding: 6px 12px; border-radius: 6px; }}
    QMenuBar::item:selected {{ background-color: {t.SURFACE_HI}; color: {t.ACCENT}; }}
    QMenu {{
        background-color: {t.SURFACE_HI}; border: 1px solid {t.HAIRLINE};
        border-radius: 8px; padding: 6px;
    }}
    QMenu::item {{ padding: 7px 26px; border-radius: 6px; }}
    QMenu::item:selected {{ background-color: {t.ACCENT}; color: #04222B; }}

    QMessageBox, QInputDialog, QDialog {{ background-color: {t.SURFACE}; }}
    QMessageBox QLabel, QInputDialog QLabel {{ color: {t.TEXT}; }}

    QScrollBar:vertical {{ background: {t.BG_BASE}; width: 11px; margin: 0; border-radius: 5px; }}
    QScrollBar::handle:vertical {{ background: {t.HAIRLINE}; border-radius: 5px; min-height: 28px; }}
    QScrollBar::handle:vertical:hover {{ background: {t.BORDER_HOVER}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}

    /* matplotlib navigation toolbar */
    QToolBar {{ background: {t.SURFACE}; border: none; spacing: 2px; }}
    QToolButton {{ background: transparent; border-radius: 6px; padding: 3px; }}
    QToolButton:hover {{ background: {t.SURFACE_HI}; }}
    """


def apply_matplotlib_theme():
    """Dark 'instrument' matplotlib theme so embedded and pop-out plots read
    as native console traces."""
    from matplotlib import font_manager
    fam = "DejaVu Sans"
    try:
        avail = {f.name for f in font_manager.fontManager.ttflist}
        if "Inter" in avail:
            fam = "Inter"
    except Exception:
        pass
    matplotlib.rcParams.update({
        "figure.facecolor":  Theme.SURFACE,
        "axes.facecolor":    Theme.BG_BASE,
        "savefig.facecolor": Theme.SURFACE,
        "axes.edgecolor":    Theme.HAIRLINE,
        "axes.labelcolor":   Theme.TEXT_MUTED,
        "axes.titlecolor":   Theme.TEXT,
        "text.color":        Theme.TEXT_MUTED,
        "xtick.color":       Theme.TEXT_FAINT,
        "ytick.color":       Theme.TEXT_FAINT,
        "xtick.labelcolor":  Theme.TEXT_MUTED,
        "ytick.labelcolor":  Theme.TEXT_MUTED,
        "grid.color":        Theme.HAIRLINE,
        "grid.linewidth":    0.7,
        "grid.alpha":        0.5,
        "axes.grid":         True,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       fam,
        "font.size":         8.5,
        "axes.titlesize":    10.5,
        "axes.titleweight":  "bold",
        "legend.framealpha": 0.0,
        "legend.labelcolor": Theme.TEXT_MUTED,
        "figure.autolayout": False,
    })


def _cv2_has_gui():
    """True only if this OpenCV build ships highgui (namedWindow/imshow).
    Headless wheels (opencv-python-headless) do not, so the live preview
    windows must be skipped."""
    try:
        info = cv2.getBuildInformation()
    except Exception:
        return False
    for key in ("GTK+", "QT", "WIN32UI", "Cocoa", "GTK"):
        for line in info.splitlines():
            if key in line and "YES" in line.upper():
                return True
    return False


CV2_HAS_GUI = _cv2_has_gui()


# ---------------------------------------------------------------------
#  In-app ROI + calibration selector (Qt, no OpenCV highgui needed)
# ---------------------------------------------------------------------
class ROICalibDialog(QtWidgets.QDialog):
    """Two-step selector on the first video frame:
        1. drag a rectangle  -> region of interest
        2. drag a line       -> calibration reference
    Returns frame-pixel coordinates, identical in meaning to the original
    OpenCV selectors, but rendered entirely inside Qt."""

    def __init__(self, frame_bgr, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select ROI & Calibration")
        if parent is not None:
            self.setStyleSheet(parent.styleSheet())

        h, w = frame_bgr.shape[:2]
        self._rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).copy()
        qimg = QtGui.QImage(self._rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)

        maxw, maxh = 1120, 680
        self.scale = min(maxw / w, maxh / h, 1.0)
        self.disp_w, self.disp_h = int(w * self.scale), int(h * self.scale)
        self.frame_w, self.frame_h = w, h
        self.base_pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.disp_w, self.disp_h, Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation)

        self.stage = 'roi'
        self.roi_disp = None        # (x0, y0, x1, y1) in display coords
        self.p1 = self.p2 = None    # calibration endpoints, display coords
        self._start = self._cur = None
        self._dragging = False

        v = QVBoxLayout(self)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)

        self.info = QLabel()
        self.info.setObjectName("readout")
        v.addWidget(self.info)

        self.canvas = QLabel()
        self.canvas.setFixedSize(self.disp_w, self.disp_h)
        self.canvas.setMouseTracking(True)
        self.canvas.mousePressEvent = self._press
        self.canvas.mouseMoveEvent = self._move
        self.canvas.mouseReleaseEvent = self._release
        v.addWidget(self.canvas, alignment=Qt.AlignmentFlag.AlignCenter)

        row = QHBoxLayout()
        row.setSpacing(10)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self._reset_stage)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.primary_btn = QPushButton("Next: Calibration  ▶")
        self.primary_btn.setProperty("variant", "primary")
        self.primary_btn.clicked.connect(self._advance)
        row.addWidget(self.reset_btn)
        row.addWidget(self.cancel_btn)
        row.addStretch(1)
        row.addWidget(self.primary_btn)
        v.addLayout(row)

        self._refresh()

    # ----- coordinate mapping -----
    def _to_frame(self, x, y):
        fx = int(round(min(max(x, 0), self.disp_w) / self.scale))
        fy = int(round(min(max(y, 0), self.disp_h) / self.scale))
        fx = min(max(fx, 0), self.frame_w - 1)
        fy = min(max(fy, 0), self.frame_h - 1)
        return fx, fy

    # ----- mouse -----
    def _press(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._start = (ev.position().x(), ev.position().y())
            self._cur = self._start
            self._refresh()

    def _move(self, ev):
        if self._dragging:
            self._cur = (ev.position().x(), ev.position().y())
            self._refresh()

    def _release(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self._cur = (ev.position().x(), ev.position().y())
            x0, y0 = self._start
            x1, y1 = self._cur
            if self.stage == 'roi':
                self.roi_disp = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            else:
                self.p1, self.p2 = (x0, y0), (x1, y1)
            self._refresh()

    # ----- stage control -----
    def _reset_stage(self):
        if self.stage == 'roi':
            self.roi_disp = None
        else:
            self.p1 = self.p2 = None
        self._start = self._cur = None
        self._refresh()

    def _advance(self):
        if self.stage == 'roi':
            if not self.roi_disp:
                QtWidgets.QMessageBox.warning(self, "No ROI", "Draw a region of interest first.")
                return
            x0, y0, x1, y1 = self.roi_disp
            if abs(x1 - x0) < 6 or abs(y1 - y0) < 6:
                QtWidgets.QMessageBox.warning(self, "ROI too small", "Draw a larger region of interest.")
                return
            self.stage = 'calib'
            self._start = self._cur = None
            self.primary_btn.setText("Confirm  ✓")
            self._refresh()
        else:
            if not (self.p1 and self.p2):
                QtWidgets.QMessageBox.warning(self, "No line", "Draw a calibration line first.")
                return
            if math.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]) < 3:
                QtWidgets.QMessageBox.warning(self, "Line too short", "Draw a longer calibration line.")
                return
            self.accept()

    # ----- render -----
    def _refresh(self):
        if self.stage == 'roi':
            self.info.setText("STEP 1/2  ·  Drag to draw the REGION OF INTEREST, then press Next.")
        else:
            self.info.setText("STEP 2/2  ·  Drag a line of known real-world length, then Confirm.")

        pix = self.base_pix.copy()
        pr = QPainter(pix)
        pr.setRenderHint(QPainter.RenderHint.Antialiasing)

        # committed ROI
        if self.roi_disp:
            x0, y0, x1, y1 = self.roi_disp
            pr.setPen(QPen(QColor(Theme.ACCENT), 2))
            pr.setBrush(QColor(70, 229, 255, 28))
            pr.drawRect(int(x0), int(y0), int(x1 - x0), int(y1 - y0))

        # committed calibration line
        if self.p1 and self.p2:
            pr.setPen(QPen(QColor(Theme.AMBER), 2))
            pr.drawLine(int(self.p1[0]), int(self.p1[1]), int(self.p2[0]), int(self.p2[1]))
            for p in (self.p1, self.p2):
                pr.setBrush(QColor(Theme.AMBER))
                pr.drawEllipse(int(p[0]) - 4, int(p[1]) - 4, 8, 8)

        # live drag preview
        if self._dragging and self._start and self._cur:
            x0, y0 = self._start
            x1, y1 = self._cur
            if self.stage == 'roi':
                pr.setPen(QPen(QColor(Theme.ACCENT), 2, Qt.PenStyle.DashLine))
                pr.setBrush(Qt.BrushStyle.NoBrush)
                pr.drawRect(int(min(x0, x1)), int(min(y0, y1)), int(abs(x1 - x0)), int(abs(y1 - y0)))
            else:
                pr.setPen(QPen(QColor(Theme.AMBER), 2, Qt.PenStyle.DashLine))
                pr.drawLine(int(x0), int(y0), int(x1), int(y1))
        pr.end()
        self.canvas.setPixmap(pix)

    # ----- result -----
    def get(self):
        if self.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        x0, y0, x1, y1 = self.roi_disp
        xa, ya = self._to_frame(x0, y0)
        xb, yb = self._to_frame(x1, y1)
        xa, xb = sorted((xa, xb))
        ya, yb = sorted((ya, yb))
        p1 = self._to_frame(*self.p1)
        p2 = self._to_frame(*self.p2)
        px_len = float(np.linalg.norm(np.array(p1) - np.array(p2)))
        return {'roi': (xa, ya, xb, yb), 'px_len': px_len, 'calib': (p1, p2)}


# ---------------------------------------------------------------------
#  Main application window
# ---------------------------------------------------------------------
class FluidGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Turbulence Realm — SINDy")
        self.setMinimumSize(1080, 860)

        logo_path = "logo.png"
        if os.path.exists(logo_path):
            self.setWindowIcon(QtGui.QIcon(logo_path))

        apply_matplotlib_theme()

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Header instrument bar ---
        main_layout.addWidget(self._build_header())

        # --- Body ---
        content_root = QWidget()
        content_root.setObjectName("contentRoot")
        body = QVBoxLayout(content_root)
        body.setContentsMargins(16, 14, 16, 10)
        body.setSpacing(0)
        main_layout.addWidget(content_root, 1)

        self.tabs = QTabWidget()
        body.addWidget(self.tabs)

        self._build_setup_tab()
        self._build_visualize_tab()
        self._build_export_tab()

        # --- Footer (Status Bar) ---
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Ready")
        main_layout.addWidget(self.status_bar)

        # Button enablement is now driven by *actual stage completion*
        # (see the end of each pipeline handler) rather than click order, so a
        # cancelled or failed step never prematurely unlocks later ones.

        # Initial data structures
        self.result = {}
        self.of_result = {}
        self.sindy_result = {}
        self.finished = False
        self.frames_to_display = 8
        self.quiver_step = None

        self.setStyleSheet(_stylesheet())

    # ----- chrome builders -------------------------------------------
    def _build_header(self):
        header = QFrame()
        header.setObjectName("headerBar")
        header.setFixedHeight(76)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(24, 0, 22, 0)
        hl.setSpacing(16)

        mark = QLabel("⬡")
        mark.setStyleSheet(f"color: {Theme.ACCENT}; font-size: 26pt; background: transparent;")
        hl.addWidget(mark)

        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title = QLabel(
            "Turbulence <span style='color:%s'>Realm</span> "
            "<span style='color:%s'>·</span> SINDy" % (Theme.ACCENT, Theme.MAGENTA))
        title.setObjectName("appTitle")
        title.setTextFormat(Qt.TextFormat.RichText)
        subtitle = QLabel("OPTICAL FLOW  ·  SPARSE DYNAMICS  ·  FLOW RECONSTRUCTION")
        subtitle.setObjectName("appSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        hl.addLayout(title_box)

        hl.addStretch()

        self.about_btn = QPushButton("About")
        self.about_btn.setFixedWidth(96)
        self.about_btn.clicked.connect(self.show_about_dialog)
        hl.addWidget(self.about_btn)

        self.status_pill = QLabel("● IDLE")
        self.status_pill.setObjectName("statusPill")
        hl.addWidget(self.status_pill)

        return header

    def _set_pill(self, text, state="idle"):
        colors = {
            "idle":  (Theme.TEXT_MUTED, "rgba(143,166,196,0.10)", "rgba(143,166,196,0.30)"),
            "ready": (Theme.ACCENT,     "rgba(70,229,255,0.10)",  "rgba(70,229,255,0.38)"),
            "busy":  (Theme.AMBER,      "rgba(255,194,75,0.12)",  "rgba(255,194,75,0.40)"),
            "done":  (Theme.GOOD,       "rgba(67,245,168,0.12)",  "rgba(67,245,168,0.40)"),
        }
        c, bg, br = colors.get(state, colors["idle"])
        self.status_pill.setText(text)
        self.status_pill.setStyleSheet(
            f"QLabel#statusPill {{ color: {c}; background-color: {bg};"
            f" border: 1px solid {br}; border-radius: 11px; padding: 4px 14px;"
            f" font-family: '{Theme.MONO_FONT}'; font-size: 8pt; font-weight: bold;"
            " letter-spacing: 1px; }")
        QtWidgets.QApplication.processEvents()

    def _show_of_preview(self, bgr):
        """Render an optical-flow preview frame (BGR) into the in-app label.
        Independent of OpenCV highgui, so it works on headless builds."""
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

    def _enable_visualization(self):
        """Unlock visualisation/export once a SINDy prediction exists."""
        for b in (self.quiver_btn, self.contour_btn, self.stream_btn,
                  self.error_btn, self.export_btn, self.anim_btn):
            b.setEnabled(True)
        self.tabs.setTabText(1, "Visualization & Analysis  ●")

    # ----- tabs ------------------------------------------------------
    def _build_setup_tab(self):
        self.setup_tab = QWidget()
        layout = QVBoxLayout(self.setup_tab)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(14)

        # File Selection
        file_group = QtWidgets.QGroupBox("VIDEO SOURCE")
        fg = QHBoxLayout(file_group)
        fg.setSpacing(10)
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("No video file selected…")
        self.file_btn = QPushButton("Browse Video")
        self.file_btn.setFixedWidth(150)
        self.file_btn.clicked.connect(self.pick_file)
        fg.addWidget(self.file_edit)
        fg.addWidget(self.file_btn)
        layout.addWidget(file_group)

        # ROI & Calibration
        roi_group = QtWidgets.QGroupBox("REGION OF INTEREST & CALIBRATION")
        rg = QVBoxLayout(roi_group)
        rg.setSpacing(10)
        self.run_btn = QPushButton("Select ROI & Calibrate")
        self.run_btn.setProperty("variant", "primary")
        self.run_btn.clicked.connect(self.start_roi)
        self.mpp_label = QLabel("Calibration:  — px = — m    ·    1 px = — m")
        self.mpp_label.setObjectName("readout")
        rg.addWidget(self.run_btn)
        rg.addWidget(self.mpp_label)
        layout.addWidget(roi_group)

        # Processing pipeline
        proc_group = QtWidgets.QGroupBox("PROCESSING PIPELINE")
        pg = QGridLayout(proc_group)
        pg.setSpacing(10)

        self.process_btn = QPushButton("① Process Optical Flow")
        self.process_btn.setProperty("variant", "primary")
        self.process_btn.clicked.connect(self.run_optical_flow_gui)
        self.process_btn.setEnabled(False)
        pg.addWidget(self.process_btn, 0, 0)

        self.sindy_btn = QPushButton("② Run SINDy Modeling")
        self.sindy_btn.setProperty("variant", "primary")
        self.sindy_btn.clicked.connect(self.run_sindy_gui)
        self.sindy_btn.setEnabled(False)
        pg.addWidget(self.sindy_btn, 0, 1)

        self.pred_btn = QPushButton("③ Run SINDy Prediction")
        self.pred_btn.setProperty("variant", "primary")
        self.pred_btn.clicked.connect(self.run_prediction_gui)
        self.pred_btn.setEnabled(False)
        pg.addWidget(self.pred_btn, 1, 0)

        self.sindy_eq_btn = QPushButton("Show SINDy Equation")
        self.sindy_eq_btn.setProperty("variant", "violet")
        self.sindy_eq_btn.clicked.connect(self.show_sindy_equation_gui)
        self.sindy_eq_btn.setEnabled(False)
        pg.addWidget(self.sindy_eq_btn, 1, 1)
        layout.addWidget(proc_group)

        # Progress
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setMaximum(100)
        self.progress.hide()
        layout.addWidget(self.progress)

        # Live in-app optical-flow preview (works without OpenCV highgui)
        self.preview_group = QtWidgets.QGroupBox("LIVE OPTICAL-FLOW PREVIEW")
        pv = QVBoxLayout(self.preview_group)
        self.of_preview = QLabel("Run optical flow to see the live HSV + quiver field here.")
        self.of_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.of_preview.setMinimumHeight(300)
        self.of_preview.setStyleSheet(
            f"background-color: {Theme.BG_BASE}; color: {Theme.TEXT_FAINT};"
            f" border: 1px solid {Theme.HAIRLINE}; border-radius: 8px;"
            f" font-family: '{Theme.MONO_FONT}'; font-size: 9pt;")
        self.of_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        pv.addWidget(self.of_preview)
        layout.addWidget(self.preview_group, 1)

        # Step guide
        self.status = QLabel(
            "01  ·  Choose a video file\n"
            "02  ·  Select ROI & calibrate scale\n"
            "03  ·  Process dense optical flow\n"
            "04  ·  Fit the SINDy model\n"
            "05  ·  Predict, analyse & export")
        self.status.setObjectName("stepGuide")
        layout.addWidget(self.status)

        self.tabs.addTab(self.setup_tab, "Setup & Processing")

    def _build_visualize_tab(self):
        self.visualize_tab = QWidget()
        layout = QVBoxLayout(self.visualize_tab)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(14)

        viz_group = QtWidgets.QGroupBox("VISUALISATION & ANALYSIS")
        vg = QGridLayout(viz_group)
        vg.setSpacing(10)

        self.quiver_btn = QPushButton("Quiver Comparison")
        self.quiver_btn.clicked.connect(self.show_quiver_section)
        self.quiver_btn.setEnabled(False)
        vg.addWidget(self.quiver_btn, 0, 0)

        self.contour_btn = QPushButton("Contour Plot")
        self.contour_btn.clicked.connect(self.plot_contour_gui)
        self.contour_btn.setEnabled(False)
        vg.addWidget(self.contour_btn, 0, 1)

        self.stream_btn = QPushButton("Streamline Plot")
        self.stream_btn.clicked.connect(self.plot_stream_gui)
        self.stream_btn.setEnabled(False)
        vg.addWidget(self.stream_btn, 1, 0)

        self.error_btn = QPushButton("Error Analysis / Maps")
        self.error_btn.clicked.connect(self.run_error_analysis_gui)
        self.error_btn.setEnabled(False)
        vg.addWidget(self.error_btn, 1, 1)
        layout.addWidget(viz_group)

        # Embedded quiver comparison
        self.quiver_section = QtWidgets.QGroupBox("LIVE QUIVER  ·  ACTUAL vs SINDy")
        quiver_layout = QVBoxLayout(self.quiver_section)
        quiver_layout.setSpacing(10)

        slider_layout = QHBoxLayout()
        flbl = QLabel("Frame")
        flbl.setObjectName("fieldLabel")
        self.quiver_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.quiver_slider.setMinimum(1)
        self.quiver_slider.setMaximum(1)
        self.quiver_frame_label = QLabel("1 / 1")
        self.quiver_frame_label.setObjectName("readout")
        self.quiver_frame_label.setFixedWidth(110)
        self.quiver_frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slider_layout.addWidget(flbl)
        slider_layout.addWidget(self.quiver_slider, 1)
        slider_layout.addWidget(self.quiver_frame_label)
        quiver_layout.addLayout(slider_layout)

        self.quiver_container = QWidget()
        container_layout = QVBoxLayout(self.quiver_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        self.quiver_fig = plt.Figure(figsize=(8, 6), tight_layout=True)
        self.quiver_fig.patch.set_facecolor(Theme.SURFACE)
        self.quiver_ax = self.quiver_fig.add_subplot(111)
        self.quiver_canvas = FigureCanvas(self.quiver_fig)
        container_layout.addWidget(self.quiver_canvas)

        self.quiver_toolbar = NavigationToolbar(self.quiver_canvas, self.quiver_section)
        quiver_layout.addWidget(self.quiver_container, 1)
        quiver_layout.addWidget(self.quiver_toolbar)

        self.quiver_section.setVisible(False)
        layout.addWidget(self.quiver_section, 1)
        layout.addStretch(0)
        self.tabs.addTab(self.visualize_tab, "Visualization & Analysis")

    def _build_export_tab(self):
        self.export_tab = QWidget()
        layout = QVBoxLayout(self.export_tab)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(14)

        export_group = QtWidgets.QGroupBox("EXPORT DATA & ANIMATIONS")
        eg = QGridLayout(export_group)
        eg.setSpacing(10)

        self.export_btn = QPushButton("Export All Frame CSVs")
        self.export_btn.clicked.connect(self.export_csv_gui)
        self.export_btn.setEnabled(False)
        eg.addWidget(self.export_btn, 0, 0)

        self.anim_btn = QPushButton("Export Quiver Animation")
        self.anim_btn.setProperty("variant", "violet")
        self.anim_btn.clicked.connect(self.export_animation_gui)
        self.anim_btn.setEnabled(False)
        eg.addWidget(self.anim_btn, 0, 1)
        layout.addWidget(export_group)

        hint = QLabel(
            "CSV export writes one file per frame (actual_u/v + predicted_u/v).\n"
            "Animation export renders the actual-vs-SINDy quiver overlay to MP4 "
            "(requires FFmpeg).")
        hint.setObjectName("stepGuide")
        layout.addWidget(hint)

        layout.addStretch(1)
        self.tabs.addTab(self.export_tab, "Export Options")

    # =================================================================
    #  PIPELINE  (logic preserved from TR-SINDY-Final.py)
    # =================================================================
    def show_quiver_section(self):
        scale, ok = QtWidgets.QInputDialog.getDouble(
            self, "Quiver Scale", "Set quiver scale (higher = smaller arrows):",
            1, 0.1, 150, 2)
        if not ok:
            return
        width, ok = QtWidgets.QInputDialog.getDouble(
            self, "Quiver Width", "Set arrow width:", 0.006, 0.0001, 0.1, 4)
        if not ok:
            return

        self.quiver_scale = scale
        self.quiver_width = width

        nfr = self.sindy_result['n_frames']
        self.quiver_slider.setMinimum(1)
        self.quiver_slider.setMaximum(nfr)
        self.quiver_slider.setValue(1)
        self.quiver_frame_label.setText(f"1 / {nfr}")

        if not hasattr(self, '_quiver_slider_connected'):
            self.quiver_slider.valueChanged.connect(self.update_quiver_plot)
            self._quiver_slider_connected = True

        self.quiver_section.setVisible(True)
        self.update_quiver_plot(1)

    def update_quiver_plot(self, frame_idx):
        if 'frame_mmap_path' not in self.of_result:
            QtWidgets.QMessageBox.critical(self, "Data Error", "Missing frame data - rerun optical flow processing")
            return

        idx = frame_idx - 1
        nfr = self.sindy_result['n_frames']
        if idx < 0 or idx >= nfr:
            return
        self.quiver_frame_label.setText(f"{frame_idx} / {nfr}")

        try:
            meters_per_pixel = self.result['meters_per_pixel']
            u_path = self.of_result['u_mmap_path']
            v_path = self.of_result['v_mmap_path']
            frame_path = self.of_result['frame_mmap_path']

            shape = (nfr, self.sindy_result['roi_h'], self.sindy_result['roi_w'])
            u_mmap = np.memmap(u_path, dtype=np.float32, mode='r', shape=shape)
            v_mmap = np.memmap(v_path, dtype=np.float32, mode='r', shape=shape)
            frame_mmap = np.memmap(frame_path, dtype=np.uint8, mode='r', shape=shape)

            roi_height, roi_width = u_mmap.shape[1], u_mmap.shape[2]
            bg_img = np.asarray(frame_mmap[idx])
            actual_u = np.asarray(u_mmap[idx])
            actual_v = np.asarray(v_mmap[idx])
            pred_u = self.X_pred_optical[idx, ..., 0]
            pred_v = self.X_pred_optical[idx, ..., 1]

            quiver_step = max(roi_height // 25, 2)
            ys = np.arange(0, roi_height, quiver_step)
            xs = np.arange(0, roi_width, quiver_step)
            xg, yg = np.meshgrid(xs, ys)
            xg_real = xg * meters_per_pixel
            yg_real = yg * meters_per_pixel

            self.quiver_ax.clear()
            self.quiver_ax.set_facecolor(Theme.BG_BASE)
            self.quiver_ax.imshow(bg_img, cmap='gray', origin='upper',
                                  extent=[0, roi_width * meters_per_pixel, roi_height * meters_per_pixel, 0])

            self.quiver_ax.quiver(
                xg_real, yg_real,
                actual_u[ys[:, None], xs[None, :]],
                actual_v[ys[:, None], xs[None, :]],
                color=Theme.ACCENT, angles='xy',
                scale=self.quiver_scale, width=self.quiver_width, label='Actual')
            self.quiver_ax.quiver(
                xg_real, yg_real,
                pred_u[ys[:, None], xs[None, :]],
                pred_v[ys[:, None], xs[None, :]],
                color=Theme.MAGENTA, angles='xy',
                scale=self.quiver_scale, width=self.quiver_width, label='SINDy')

            self.quiver_ax.set_title(f"Frame {frame_idx}: Actual vs SINDy Prediction")
            self.quiver_ax.set_xlabel("x (m)")
            self.quiver_ax.set_ylabel("y (m)")
            self.quiver_ax.legend(labelcolor=Theme.TEXT_MUTED)
            self.quiver_ax.set_xlim(0, roi_width * meters_per_pixel)
            self.quiver_ax.set_ylim(roi_height * meters_per_pixel, 0)

            self.quiver_fig.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.9)
            self.quiver_fig.tight_layout()
            self.quiver_canvas.draw_idle()
            self.quiver_container.updateGeometry()
            QtWidgets.QApplication.processEvents()
            QtWidgets.QApplication.processEvents()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Plot Error", f"Failed to update quiver plot:\n{str(e)}")

    def show_about_dialog(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Turbulence Realm — SINDy")
        dlg.setFixedSize(420, 420)
        dlg.setStyleSheet(f"QDialog {{ background-color: {Theme.SURFACE}; }}")
        vbox = QtWidgets.QVBoxLayout(dlg)

        logo_path = "logo.png"
        if os.path.exists(logo_path):
            logo_label = QtWidgets.QLabel()
            pixmap = QtGui.QPixmap(logo_path).scaled(
                110, 110, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(pixmap)
            logo_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            vbox.addWidget(logo_label)
        else:
            mark = QtWidgets.QLabel("⬡")
            mark.setStyleSheet(f"color: {Theme.ACCENT}; font-size: 56pt;")
            mark.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            vbox.addWidget(mark)

        title = QtWidgets.QLabel("Turbulence Realm — SINDy")
        title.setStyleSheet(f"font-size: 17px; font-weight: bold; color: {Theme.TEXT}; padding: 8px;")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(title)

        info = QtWidgets.QLabel(f"""<p style='text-align:center; color:{Theme.TEXT_MUTED}'>
            Version 1.0<br><em>Video-based Fluid Flow Analysis Tool</em></p>
            <p style='text-align:center; font-size:11px; color:{Theme.TEXT_FAINT}'>
            Optical Flow &amp; SINDy modelling (pysindy)<br>
            Developed by Fayaz Rasheed<br>
            <a style='color:{Theme.ACCENT}' href='http://www.turbulencerealm.com'>www.turbulencerealm.com</a></p>""")
        info.setOpenExternalLinks(True)
        info.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(info)

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        btn_box.accepted.connect(dlg.accept)
        vbox.addWidget(btn_box, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        dlg.exec()

    def pick_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select video file", ".", "Videos (*.mp4 *.avi *.mov);;All (*)")
        if fname:
            self.file_edit.setText(fname)
            self._set_pill("● VIDEO LOADED", "ready")
            self.status_bar.showMessage(f"Loaded: {os.path.basename(fname)}")

    def start_roi(self):
        path = self.file_edit.text()
        if not os.path.exists(path):
            QtWidgets.QMessageBox.warning(self, "No file", "Please select a valid video file first.")
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "Open error", "Cannot open selected video.")
            return
        ret, first_frame = cap.read()
        cap.release()
        if not ret:
            QtWidgets.QMessageBox.warning(self, "Read error", "Cannot read first frame.")
            return
        self.status.setText("Draw the ROI, then a calibration line of known length.")
        dlg = ROICalibDialog(first_frame, self)
        res = dlg.get()
        if res is None:
            self.status.setText("ROI / calibration cancelled.")
            return
        xa, ya, xb, yb = res['roi']
        roi_w, roi_h = xb - xa, yb - ya
        roi_img = first_frame[ya:yb, xa:xb]
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
            'video_file': path,
            'roi_box': (xa, ya, xb, yb),
            'calibration_px': px_len,
            'calibration_m': use_meters,
            'meters_per_pixel': meters_per_pixel,
            'first_frame_roi': roi_img.copy()
        }
        self.finished = True
        self.process_btn.setEnabled(True)
        self.status.setText("ROI / calibration complete.  Ready for optical flow.")
        self._set_pill("● CALIBRATED", "ready")
        self.status_bar.showMessage(f"ROI {roi_w}×{roi_h} px · scale {meters_per_pixel:.6e} m/px")

    def run_optical_flow_gui(self):
        if not getattr(self, 'finished', False):
            QtWidgets.QMessageBox.warning(self, "No ROI/Calibration", "Finish ROI/Calibration first.")
            return
        filters, ok = QtWidgets.QInputDialog.getItem(
            self, "Filtering", "Select filter for flow (will preview updates):",
            ("None", "Gaussian Blur", "Nonlocal Means", "Both"), 0, False)
        if not ok:
            return
        gaussian_params = {}
        nlm_params = {}
        enable_gauss = enable_nlm = False
        if filters in ("Gaussian Blur", "Both"):
            enable_gauss = True
            k, ok = QtWidgets.QInputDialog.getInt(self, "Gaussian kernel", "Kernel size (odd)", 5, 3, 25, 2)
            if not ok:
                return
            s, ok = QtWidgets.QInputDialog.getDouble(self, "Gaussian sigma", "Sigma", 1.3, 0.01, 10.0, 2)
            if not ok:
                return
            gaussian_params = {"ksize": (k, k), "sigmaX": s}
        if filters in ("Nonlocal Means", "Both"):
            enable_nlm = True
            h, ok = QtWidgets.QInputDialog.getDouble(self, "NLM h", "h (denoise strength)", 10, 1, 30, 1)
            if not ok:
                return
            tpl, ok = QtWidgets.QInputDialog.getInt(self, "Template Size", "templateWindowSize (odd)", 7, 3, 25, 2)
            if not ok:
                return
            sw, ok = QtWidgets.QInputDialog.getInt(self, "Search Size", "searchWindowSize (odd)", 21, 5, 41, 2)
            if not ok:
                return
            nlm_params = {"h": h, "templateWindowSize": tpl, "searchWindowSize": sw}
        self.status.setText("Optical flow processing started.")
        self._set_pill("● OPTICAL FLOW", "busy")
        self.progress.show()
        self.progress.setValue(0)
        QtWidgets.QApplication.processEvents()
        box = self.result['roi_box']
        meters_per_pixel = self.result['meters_per_pixel']
        video_file = self.result['video_file']
        MMAP_DIR = "./velocity_mmaps"
        os.makedirs(MMAP_DIR, exist_ok=True)
        u_path = os.path.abspath(os.path.join(MMAP_DIR, "u.dat"))
        v_path = os.path.abspath(os.path.join(MMAP_DIR, "v.dat"))
        for p in [u_path, v_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

        frame_path = os.path.join(MMAP_DIR, "frames.dat")
        if os.path.exists(frame_path):
            os.remove(frame_path)

        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        roi_w = box[2] - box[0]
        roi_h = box[3] - box[1]
        n_frames = total_frames - 1
        u_mmap = np.memmap(u_path, dtype=np.float32, mode='w+', shape=(n_frames, roi_h, roi_w))
        v_mmap = np.memmap(v_path, dtype=np.float32, mode='w+', shape=(n_frames, roi_h, roi_w))
        frame_mmap = np.memmap(frame_path, dtype=np.uint8, mode='w+', shape=(n_frames, roi_h, roi_w))
        self.of_result["frame_mmap_path"] = frame_path

        cap = cv2.VideoCapture(video_file)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame[box[1]:box[3], box[0]:box[2]], cv2.COLOR_BGR2GRAY)
        if enable_gauss:
            prev_gray = cv2.GaussianBlur(prev_gray, **gaussian_params)
        if enable_nlm:
            prev_gray = cv2.fastNlMeansDenoising(prev_gray, None, **nlm_params)
        FPS = cap.get(cv2.CAP_PROP_FPS)
        step_show = max(1, n_frames // 40)
        grid_step = max(roi_h // 25, 2)
        ygrid = np.arange(0, roi_h, grid_step)
        xgrid = np.arange(0, roi_w, grid_step)
        xgrid2d, ygrid2d = np.meshgrid(xgrid, ygrid)
        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            curr_gray = cv2.cvtColor(frame[box[1]:box[3], box[0]:box[2]], cv2.COLOR_BGR2GRAY)
            if enable_gauss:
                curr_gray = cv2.GaussianBlur(curr_gray, **gaussian_params)
            if enable_nlm:
                curr_gray = cv2.fastNlMeansDenoising(curr_gray, None, **nlm_params)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, pyr_scale=0.3, levels=7, winsize=21, iterations=7,
                poly_n=7, poly_sigma=1.1, flags=0)
            u_mmap[i] = flow[..., 0] * meters_per_pixel * FPS
            v_mmap[i] = flow[..., 1] * meters_per_pixel * FPS
            frame_mmap[i] = curr_gray
            frame_mmap.flush()

            prev_gray = curr_gray.copy()
            # Live HSV + quiver preview, rendered into the in-app label (no
            # OpenCV highgui needed, so it works on headless builds too).
            if i % step_show == 0 or i == n_frames - 1:
                rgb_disp = cv2.cvtColor(curr_gray, cv2.COLOR_GRAY2BGR)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv = np.zeros_like(rgb_disp)
                hsv[..., 1] = 255
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                preview = cv2.addWeighted(rgb_disp, 0.5, bgr, 0.5, 0).astype(np.uint8)
                uf = flow[..., 0][ygrid2d, xgrid2d]
                vf = flow[..., 1][ygrid2d, xgrid2d]
                for (x, y, du, dv) in zip(xgrid2d.flatten(), ygrid2d.flatten(), uf.flatten(), vf.flatten()):
                    tip = (int(round(x + du * 2)), int(round(y + dv * 2)))
                    cv2.arrowedLine(preview, (int(x), int(y)), tip, Theme.BGR_MAGENTA, 1, tipLength=0.28)
                self._show_of_preview(preview)
                # Optional native window when this OpenCV build supports it.
                if CV2_HAS_GUI:
                    try:
                        cv2.imshow("Optical Flow Progress", preview)
                        cv2.waitKey(1)
                    except cv2.error:
                        pass
            if i % 3 == 0:
                self.progress.setValue(int((i + 1) / n_frames * 100))
                QtWidgets.QApplication.processEvents()
        u_mmap.flush()
        v_mmap.flush()
        cap.release()
        self.progress.setValue(100)
        self.progress.hide()
        try:
            cv2.destroyWindow("Optical Flow Progress")
        except Exception:
            pass
        self.status.setText(f"Optical flow done [{n_frames}, {roi_h}×{roi_w}].  Run SINDy model next.")
        self._set_pill("● FLOW READY", "done")
        QtWidgets.QMessageBox.information(self, "Done", "Optical flow complete!")
        self.of_result = {
            "u_mmap_path": u_path, "v_mmap_path": v_path, "frame_mmap_path": frame_path,
            "frames": n_frames, "roi_h": roi_h, "roi_w": roi_w, "FPS": FPS,
            "filters": filters, "enable_gauss": enable_gauss, "gauss": gaussian_params,
            "enable_nlm": enable_nlm, "nlm": nlm_params}
        self.sindy_btn.setEnabled(True)

    def run_sindy_gui(self):
        degree, ok = QtWidgets.QInputDialog.getInt(
            self, "SINDy Feature Library", "Polynomial degree (1-5)", 3, 1, 5, 1)
        if not ok:
            return
        threshold, ok = QtWidgets.QInputDialog.getDouble(
            self, "STLSQ Threshold", "STLSQ threshold", 0.07, 0.01, 1.0, 3)
        if not ok:
            return
        self._set_pill("● SINDy FIT", "busy")
        u_path = self.of_result['u_mmap_path']
        v_path = self.of_result['v_mmap_path']
        n_frames = self.of_result['frames']
        roi_h = self.of_result['roi_h']
        roi_w = self.of_result['roi_w']
        MMAP_DIR = os.path.dirname(u_path)
        u_mmap = np.memmap(u_path, dtype=np.float32, mode='r', shape=(n_frames, roi_h, roi_w))
        v_mmap = np.memmap(v_path, dtype=np.float32, mode='r', shape=u_mmap.shape)
        total_points = n_frames * roi_h * roi_w

        def mmap_gradient(arr_mmap, axis=0):
            g = np.zeros_like(arr_mmap)
            if axis == 0:
                g[1:-1] = (arr_mmap[2:] - arr_mmap[:-2]) / 2.0
                g[0] = arr_mmap[1] - arr_mmap[0]
                g[-1] = arr_mmap[-1] - arr_mmap[-2]
            elif axis == 1:
                g[:, 1:-1, :] = (arr_mmap[:, 2:, :] - arr_mmap[:, :-2, :]) / 2.0
                g[:, 0, :] = arr_mmap[:, 1, :] - arr_mmap[:, 0, :]
                g[:, -1, :] = arr_mmap[:, -1, :] - arr_mmap[:, -2, :]
            elif axis == 2:
                g[:, :, 1:-1] = (arr_mmap[:, :, 2:] - arr_mmap[:, :, :-2]) / 2.0
                g[:, :, 0] = arr_mmap[:, :, 1] - arr_mmap[:, :, 0]
                g[:, :, -1] = arr_mmap[:, :, -1] - arr_mmap[:, :, -2]
            return g

        self.status.setText("Computing derivatives…")
        QtWidgets.QApplication.processEvents()
        u_x = mmap_gradient(u_mmap, axis=2)
        u_y = mmap_gradient(u_mmap, axis=1)
        u_t = mmap_gradient(u_mmap, axis=0)
        v_x = mmap_gradient(v_mmap, axis=2)
        v_y = mmap_gradient(v_mmap, axis=1)
        v_t = mmap_gradient(v_mmap, axis=0)
        u_xy = mmap_gradient(u_y, axis=2)
        v_xy = mmap_gradient(v_y, axis=2)

        def mmap_stack_X_data(filename, dtype, shape):
            if os.path.exists(filename):
                os.remove(filename)
            return np.memmap(filename, dtype=dtype, mode='w+', shape=shape)

        X_optical_path = os.path.join(MMAP_DIR, "X_optical.dat")
        X_dot_optical_path = os.path.join(MMAP_DIR, "X_dot_optical.dat")
        X_optical_mmap = mmap_stack_X_data(X_optical_path, np.float32, (total_points, 8))
        X_dot_optical_mmap = mmap_stack_X_data(X_dot_optical_path, np.float32, (total_points, 2))
        self.status.setText("Preprocessing SINDy input data…")
        self.progress.show()
        QtWidgets.QApplication.processEvents()
        idx = 0
        for f in range(n_frames):
            Xf = np.stack([
                u_mmap[f], v_mmap[f], u_x[f], u_y[f], v_x[f], v_y[f], u_xy[f], v_xy[f]
            ], axis=-1)
            Xdf = np.stack([u_t[f], v_t[f]], axis=-1)
            nrow = Xf.shape[0] * Xf.shape[1]
            X_optical_mmap[idx:idx + nrow] = Xf.reshape(-1, 8)
            X_dot_optical_mmap[idx:idx + nrow] = Xdf.reshape(-1, 2)
            idx += nrow
            if f % 3 == 0:
                self.progress.setValue(int((f + 1) / n_frames * 50))
                QtWidgets.QApplication.processEvents()
        X_optical_mmap.flush()
        X_dot_optical_mmap.flush()
        self.status.setText("Fitting SINDy model…")
        QtWidgets.QApplication.processEvents()
        batch_size = max(100_000, roi_w * roi_h)
        n_batches = math.ceil(total_points / batch_size)
        DT = 1.0 / self.of_result['FPS']
        library = PolynomialLibrary(degree=degree)
        optimizer = STLSQ(threshold=threshold)
        sindy_model = ps.SINDy(feature_library=library, optimizer=optimizer)
        for b in range(n_batches):
            b0 = b * batch_size
            b1 = min(b0 + batch_size, total_points)
            if b == 0:
                sindy_model.fit(X_optical_mmap[b0:b1], t=DT, x_dot=X_dot_optical_mmap[b0:b1])
            self.progress.setValue(50 + int((b + 1) / n_batches * 45))
            QtWidgets.QApplication.processEvents()
        self.progress.setValue(95)
        self.status.setText("SINDy fit complete.")
        self.progress.hide()
        self._set_pill("● MODEL FIT", "done")
        QtWidgets.QApplication.processEvents()
        self.sindy_result = {
            'model': sindy_model, 'X_optical_path': X_optical_path, 'X_dot_optical_path': X_dot_optical_path,
            'total_points': total_points, 'n_frames': n_frames, 'roi_h': roi_h, 'roi_w': roi_w, 'batch_size': batch_size,
            'DT': DT, 'library_degree': degree, 'stlsq_thresh': threshold}
        self.pred_btn.setEnabled(True)
        self.sindy_eq_btn.setEnabled(True)
        QtWidgets.QMessageBox.information(self, "Done", "SINDy model fit successful!")

    def run_prediction_gui(self):
        sindy_model = self.sindy_result['model']
        X_optical_path = self.sindy_result['X_optical_path']
        total_points = self.sindy_result['total_points']
        n_frames = self.sindy_result['n_frames']
        roi_h = self.sindy_result['roi_h']
        roi_w = self.sindy_result['roi_w']
        batch_size = self.sindy_result['batch_size']
        self._set_pill("● PREDICTING", "busy")
        pred_result_path = os.path.join(os.path.dirname(X_optical_path), "X_pred_optical.dat")
        if os.path.exists(pred_result_path):
            os.remove(pred_result_path)
        X_pred_optical_mmap = np.memmap(pred_result_path, np.float32, mode='w+', shape=(total_points, 2))
        n_batches = int(np.ceil(total_points / batch_size))
        self.progress.setValue(0)
        self.progress.show()
        self.status.setText("Batch prediction (SINDy)…")
        QtWidgets.QApplication.processEvents()
        for b in range(n_batches):
            b0 = b * batch_size
            b1 = min((b + 1) * batch_size, total_points)
            chunk = np.memmap(X_optical_path, np.float32, mode='r', shape=(total_points, 8))[b0:b1]
            pred = sindy_model.predict(chunk)
            pred *= -1.0
            X_pred_optical_mmap[b0:b1] = pred
            self.progress.setValue(int((b + 1) * 80 / n_batches))
            QtWidgets.QApplication.processEvents()
        X_pred_optical_mmap.flush()
        self.status.setText("Prediction done. Postprocessing for plots.")
        QtWidgets.QApplication.processEvents()
        X_pred_optical = np.zeros((n_frames, roi_h, roi_w, 2), np.float32)
        for f in range(n_frames):
            out = X_pred_optical_mmap[f * roi_h * roi_w:(f + 1) * roi_h * roi_w]
            X_pred_optical[f, :, :, 0] = gaussian_filter(out[:, 0].reshape((roi_h, roi_w)), sigma=0.8)
            X_pred_optical[f, :, :, 1] = gaussian_filter(out[:, 1].reshape((roi_h, roi_w)), sigma=0.8)
            if f % max(n_frames // 10, 1) == 0:
                self.progress.setValue(80 + int(f / n_frames * 20))
                QtWidgets.QApplication.processEvents()
        self.status.setText("Prediction ready.  You may now plot / export.")
        self.progress.setValue(100)
        self.progress.hide()
        self._set_pill("● PREDICTION READY", "done")
        QtWidgets.QApplication.processEvents()
        self.X_pred_optical = X_pred_optical
        self._enable_visualization()
        QtWidgets.QMessageBox.information(self, "Ready", "SINDy prediction and reconstruction ready.")

    def plot_quiver_gui(self):
        nfr = self.sindy_result['n_frames']
        idx, ok = QtWidgets.QInputDialog.getInt(self, "Frame", f"Frame index (1 - {nfr}):", 1, 1, nfr, 1)
        if not ok:
            return
        idx = idx - 1
        scale, ok = QtWidgets.QInputDialog.getDouble(
            self, "Quiver scale", "Set quiver scale (increase for shorter arrows):", 1, 0.1, 150, 2)
        if not ok:
            return
        meters_per_pixel = self.result['meters_per_pixel']
        u_path, v_path = self.of_result['u_mmap_path'], self.of_result['v_mmap_path']
        u_mmap = np.memmap(u_path, np.float32, mode='r', shape=(nfr, self.sindy_result['roi_h'], self.sindy_result['roi_w']))
        v_mmap = np.memmap(v_path, np.float32, mode='r', shape=u_mmap.shape)
        u, v = np.asarray(u_mmap[idx]), np.asarray(v_mmap[idx])
        xg, yg = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))
        frame_path = self.of_result['frame_mmap_path']
        frame_mmap = np.memmap(frame_path, dtype=np.uint8, mode='r',
                               shape=(self.sindy_result['n_frames'], self.sindy_result['roi_h'], self.sindy_result['roi_w']))
        bg_img = np.asarray(frame_mmap[idx])

        plt.figure(figsize=(7, 7))
        plt.imshow(bg_img, cmap='gray', origin='upper', extent=[0, u.shape[1] * meters_per_pixel, u.shape[0] * meters_per_pixel, 0])
        plt.quiver(xg * meters_per_pixel, yg * meters_per_pixel, u, v, color=Theme.ACCENT, scale=scale, width=0.006, label='Actual')
        u_pred = self.X_pred_optical[idx, :, :, 0]
        v_pred = self.X_pred_optical[idx, :, :, 1]
        plt.quiver(xg * meters_per_pixel, yg * meters_per_pixel, u_pred, v_pred, color=Theme.MAGENTA, scale=scale, width=0.006, label='SINDy')
        plt.title(f"Frame {idx + 1}: Quiver Overlay (Cyan=Actual, Magenta=SINDy)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_contour_gui(self):
        nfr = self.sindy_result['n_frames']
        idx, ok = QtWidgets.QInputDialog.getInt(self, "Frame", f"Frame index (1 - {nfr}):", 1, 1, nfr, 1)
        if not ok:
            return
        idx = idx - 1
        which, ok = QtWidgets.QInputDialog.getItem(
            self, "Plot Field", "Display contour for:", ("Actual", "SINDy Prediction", "Error"), 0, False)
        if not ok:
            return
        meters_per_pixel = self.result['meters_per_pixel']
        u_path, v_path = self.of_result['u_mmap_path'], self.of_result['v_mmap_path']
        u_mmap = np.memmap(u_path, np.float32, mode='r', shape=(nfr, self.sindy_result['roi_h'], self.sindy_result['roi_w']))
        v_mmap = np.memmap(v_path, np.float32, mode='r', shape=u_mmap.shape)
        u = np.asarray(u_mmap[idx])
        v = np.asarray(v_mmap[idx])
        u_pred = self.X_pred_optical[idx, :, :, 0]
        v_pred = self.X_pred_optical[idx, :, :, 1]
        if which == "Actual":
            mag = np.sqrt(u ** 2 + v ** 2)
            title = f"Frame {idx + 1} Actual Velocity Magnitude"
        elif which == "SINDy Prediction":
            mag = np.sqrt(u_pred ** 2 + v_pred ** 2)
            title = f"Frame {idx + 1} SINDy Prediction Magnitude"
        else:
            mag = np.sqrt((u - u_pred) ** 2 + (v - v_pred) ** 2)
            title = f"Frame {idx + 1} Prediction Error Magnitude"
        frame_path = self.of_result['frame_mmap_path']
        frame_mmap = np.memmap(frame_path, dtype=np.uint8, mode='r',
                               shape=(self.sindy_result['n_frames'], self.sindy_result['roi_h'], self.sindy_result['roi_w']))
        bg_img = np.asarray(frame_mmap[idx])

        extent = [0, u.shape[1] * meters_per_pixel, u.shape[0] * meters_per_pixel, 0]
        plt.figure(figsize=(8, 6))
        plt.imshow(bg_img, cmap='gray', extent=extent, alpha=0.7)
        levels = np.linspace(np.nanmin(mag), np.nanmax(mag), 35)
        to_plot = np.nan_to_num(mag)
        plt.contourf(np.linspace(0, u.shape[1] * meters_per_pixel, u.shape[1]),
                     np.linspace(0, u.shape[0] * meters_per_pixel, u.shape[0]), to_plot,
                     levels=levels, cmap='turbo', alpha=0.6, extend='both')
        plt.colorbar(label="Velocity magnitude")
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_stream_gui(self):
        nfr = self.sindy_result['n_frames']
        idx, ok = QtWidgets.QInputDialog.getInt(self, "Frame", f"Frame index (1 - {nfr}):", 1, 1, nfr, 1)
        if not ok:
            return
        idx = idx - 1
        which, ok = QtWidgets.QInputDialog.getItem(
            self, "Plot Field", "Display streamlines for:", ("Actual", "SINDy Prediction", "Error (diff)"), 0, False)
        if not ok:
            return
        meters_per_pixel = self.result['meters_per_pixel']
        u_path, v_path = self.of_result['u_mmap_path'], self.of_result['v_mmap_path']
        u_mmap = np.memmap(u_path, np.float32, mode='r', shape=(nfr, self.sindy_result['roi_h'], self.sindy_result['roi_w']))
        v_mmap = np.memmap(v_path, np.float32, mode='r', shape=u_mmap.shape)
        u = np.asarray(u_mmap[idx])
        v = np.asarray(v_mmap[idx])
        u_pred = self.X_pred_optical[idx, :, :, 0]
        v_pred = self.X_pred_optical[idx, :, :, 1]
        if which == "Actual":
            plotu, plotv = u, v
            title = f"Frame {idx + 1} Actual Velocity Streamlines"
        elif which == "SINDy Prediction":
            plotu, plotv = u_pred, v_pred
            title = f"Frame {idx + 1} SINDy Prediction Streamlines"
        else:
            plotu, plotv = u - u_pred, v - v_pred
            title = f"Frame {idx + 1} Error (Actual - SINDy) Streamlines"
        frame_path = self.of_result['frame_mmap_path']
        frame_mmap = np.memmap(frame_path, dtype=np.uint8, mode='r',
                               shape=(self.sindy_result['n_frames'], self.sindy_result['roi_h'], self.sindy_result['roi_w']))
        bg_img = np.asarray(frame_mmap[idx])

        extent = [0, u.shape[1] * meters_per_pixel, u.shape[0] * meters_per_pixel, 0]
        x_vals = np.linspace(0, u.shape[1] * meters_per_pixel, u.shape[1])
        y_vals = np.linspace(0, u.shape[0] * meters_per_pixel, u.shape[0])
        Xg, Yg = np.meshgrid(x_vals, y_vals)
        plt.figure(figsize=(8, 6))
        plt.imshow(bg_img, cmap='gray', extent=extent, alpha=0.7)
        plt.streamplot(Xg, Yg, plotu, plotv, color=Theme.ACCENT, density=1.2, linewidth=1, arrowsize=1.2)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def show_sindy_equation_gui(self):
        sindy_model = self.sindy_result.get('model', None)
        if sindy_model is None:
            QtWidgets.QMessageBox.warning(self, "No SINDy Model", "Please run SINDy modeling first.")
            return
        eq_str = sindy_model.equations(precision=5)
        if isinstance(eq_str, list):
            eq_str = "\n\n".join(eq_str)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("SINDy Fitted Equation")
        dlg.resize(640, 420)
        dlg.setStyleSheet(f"QDialog {{ background-color: {Theme.SURFACE}; }}")
        vbox = QtWidgets.QVBoxLayout(dlg)
        text = QtWidgets.QPlainTextEdit()
        text.setPlainText(eq_str)
        text.setReadOnly(True)
        vbox.addWidget(text)
        copy_btn = QtWidgets.QPushButton("Copy to Clipboard")
        copy_btn.setProperty("variant", "primary")
        vbox.addWidget(copy_btn)

        def copy_to_clipboard():
            QtWidgets.QApplication.clipboard().setText(eq_str)
        copy_btn.clicked.connect(copy_to_clipboard)
        dlg.exec()

    def export_csv_gui(self):
        output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory", "./output")
        if not output_dir:
            return
        u_path = self.of_result['u_mmap_path']
        v_path = self.of_result['v_mmap_path']
        nfr = self.sindy_result['n_frames']
        u_mmap = np.memmap(u_path, np.float32, mode='r', shape=(nfr, self.sindy_result['roi_h'], self.sindy_result['roi_w']))
        v_mmap = np.memmap(v_path, np.float32, mode='r', shape=u_mmap.shape)
        self.status_bar.showMessage("Exporting CSV files...", 0)
        for frame_idx in range(u_mmap.shape[0]):
            frame_data = {
                'actual_u': u_mmap[frame_idx].flatten(),
                'actual_v': v_mmap[frame_idx].flatten(),
                'predicted_u': self.X_pred_optical[frame_idx, :, :, 0].flatten(),
                'predicted_v': self.X_pred_optical[frame_idx, :, :, 1].flatten()}
            df = pd.DataFrame(frame_data)
            df.to_csv(os.path.join(output_dir, f'frame_{frame_idx + 1}_values.csv'), index=False)
            self.status_bar.showMessage(f"Exporting CSV: Frame {frame_idx + 1}/{u_mmap.shape[0]}", 0)
            QtWidgets.QApplication.processEvents()
        self.status_bar.showMessage(f'CSV files for frames saved to: {output_dir}', 5000)
        QtWidgets.QMessageBox.information(self, "Exported", f'CSV files for frames saved to:\n{output_dir}')

    def run_error_analysis_gui(self):
        u_path = self.of_result['u_mmap_path']
        v_path = self.of_result['v_mmap_path']
        nfr = self.sindy_result['n_frames']
        u_mmap = np.memmap(u_path, np.float32, mode='r', shape=(nfr, self.sindy_result['roi_h'], self.sindy_result['roi_w']))
        v_mmap = np.memmap(v_path, np.float32, mode='r', shape=u_mmap.shape)
        pred = self.X_pred_optical
        n_frames, h, w = u_mmap.shape
        rmse, mse = [], []
        for f in range(n_frames):
            err = (u_mmap[f] - pred[f, :, :, 0]) ** 2 + (v_mmap[f] - pred[f, :, :, 1]) ** 2
            mse.append(np.mean(err))
            rmse.append(np.sqrt(np.mean(err)))
        rmse, mse = np.array(rmse), np.array(mse)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(np.arange(1, n_frames + 1), rmse, color=Theme.ACCENT, lw=1.8)
        ax1.set_ylabel("RMSE")
        ax1.set_xlabel("Frame")
        ax1.set_title("RMSE per frame (Actual vs SINDy)")
        ax2.plot(np.arange(1, n_frames + 1), mse, color=Theme.MAGENTA, lw=1.8)
        ax2.set_ylabel("MSE")
        ax2.set_xlabel("Frame")
        ax2.set_title("MSE per frame")
        plt.tight_layout()
        plt.show()
        idx, ok = QtWidgets.QInputDialog.getInt(self, "Frame for Spatial Error Map", f"Frame index (1-{n_frames})", 1, 1, n_frames, 1)
        if not ok:
            return
        idx -= 1
        err = np.sqrt((u_mmap[idx] - pred[idx, :, :, 0]) ** 2 + (v_mmap[idx] - pred[idx, :, :, 1]) ** 2)
        plt.figure(figsize=(8, 6))
        plt.imshow(err, cmap='inferno', origin='upper')
        plt.colorbar(label="Per-pixel RMSE")
        plt.title(f"Spatial Error Map: Frame {idx + 1}")
        plt.tight_layout()
        plt.show()

    def export_animation_gui(self):
        if getattr(sys, 'frozen', False):
            bundled_ffmpeg_path = os.path.join(os.path.dirname(sys.argv[0]), "ffmpeg.exe")
            plt.rcParams['animation.ffmpeg_path'] = bundled_ffmpeg_path
            print(f"Matplotlib FFmpeg path set to: {bundled_ffmpeg_path}")
        else:
            print("Running in development mode. Matplotlib will use system FFmpeg if available.")
        u_path = self.of_result['u_mmap_path']
        v_path = self.of_result['v_mmap_path']
        nfr = self.sindy_result['n_frames']
        u_mmap = np.memmap(u_path, np.float32, mode='r', shape=(nfr, self.sindy_result['roi_h'], self.sindy_result['roi_w']))
        v_mmap = np.memmap(v_path, np.float32, mode='r', shape=u_mmap.shape)
        pred = self.X_pred_optical
        n_frames, h, w = u_mmap.shape
        step = max(h // 30, 2)
        ys = np.arange(0, h, step)
        xs = np.arange(0, w, step)
        xg, yg = np.meshgrid(xs, ys)
        meters_per_pixel = self.result['meters_per_pixel']

        scale, ok = QtWidgets.QInputDialog.getDouble(
            self, "Quiver scale", "Set quiver scale (increase for shorter arrows):", 1, 0.1, 150, 2)
        if not ok:
            return
        width, ok = QtWidgets.QInputDialog.getDouble(
            self, "Quiver width", "Set quiver arrow width (e.g. 0.006):", 0.006, 0.0001, 0.1, 4)
        if not ok:
            return

        out_file, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Animation",
            os.path.join(os.path.expanduser("~"), "quiver_animation.mp4"), "MP4 files (*.mp4);;All files (*)")
        if not out_file:
            return
        self.status.setText("Rendering quiver animation. May take several minutes…")
        self._set_pill("● RENDERING", "busy")
        self.progress.show()
        self.status_bar.showMessage("Rendering quiver animation...", 0)
        fig, ax = plt.subplots(figsize=(8, 8))
        frame_path = self.of_result['frame_mmap_path']
        frame_mmap = np.memmap(frame_path, dtype=np.uint8, mode='r',
                               shape=(self.sindy_result['n_frames'], self.sindy_result['roi_h'], self.sindy_result['roi_w']))
        extent = [0, w * meters_per_pixel, h * meters_per_pixel, 0]

        def animate(fidx):
            bg_img = frame_mmap[fidx]
            ax.clear()
            ax.imshow(bg_img, cmap='gray', origin='upper', extent=extent)
            ax.quiver(
                xg * meters_per_pixel, yg * meters_per_pixel,
                u_mmap[fidx][ys[:, None], xs[None, :]],
                v_mmap[fidx][ys[:, None], xs[None, :]],
                color=Theme.ACCENT, angles='xy', scale=scale, width=width, label='Actual')
            ax.quiver(
                xg * meters_per_pixel, yg * meters_per_pixel,
                pred[fidx, ys[:, None], xs[None, :], 0],
                pred[fidx, ys[:, None], xs[None, :], 1],
                color=Theme.MAGENTA, angles='xy', scale=scale, width=width, label='SINDy')
            ax.set_title(f"Frame {fidx + 1}: Quiver Overlay (Cyan=Actual, Magenta=SINDy)")
            ax.set_xlabel('x (meters)')
            ax.set_ylabel('y (meters)')
            ax.legend()
            self.status_bar.showMessage(f"Rendering quiver animation: Frame {fidx + 1}/{n_frames}", 0)
            QtWidgets.QApplication.processEvents()

        ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200)
        from matplotlib.animation import FFMpegWriter
        ani.save(out_file, writer=FFMpegWriter(fps=8))
        plt.close(fig)
        self.status.setText(f"Animation export complete: {out_file}")
        self.status_bar.showMessage(f"Animation export complete: {out_file}", 5000)
        self._set_pill("● PREDICTION READY", "done")
        QtWidgets.QMessageBox.information(self, "Done", f"Animation saved:\n{out_file}")
        self.progress.hide()


# ---------------------------------------------------------------------
#  Dark Fusion palette + designed splash + main()
# ---------------------------------------------------------------------
def _apply_dark_palette(app):
    p = QPalette()
    base = QColor(Theme.BG_BASE)
    surf = QColor(Theme.SURFACE)
    surfhi = QColor(Theme.SURFACE_HI)
    text = QColor(Theme.TEXT)
    muted = QColor(Theme.TEXT_MUTED)
    accent = QColor(Theme.ACCENT)
    p.setColor(QPalette.ColorRole.Window, surf)
    p.setColor(QPalette.ColorRole.WindowText, text)
    p.setColor(QPalette.ColorRole.Base, base)
    p.setColor(QPalette.ColorRole.AlternateBase, surfhi)
    p.setColor(QPalette.ColorRole.ToolTipBase, surfhi)
    p.setColor(QPalette.ColorRole.ToolTipText, text)
    p.setColor(QPalette.ColorRole.Text, text)
    p.setColor(QPalette.ColorRole.Button, surfhi)
    p.setColor(QPalette.ColorRole.ButtonText, text)
    p.setColor(QPalette.ColorRole.Highlight, accent)
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#04222B"))
    p.setColor(QPalette.ColorRole.PlaceholderText, muted)
    p.setColor(QPalette.ColorRole.Link, accent)
    for grp in (QPalette.ColorGroup.Disabled,):
        p.setColor(grp, QPalette.ColorRole.Text, QColor(Theme.TEXT_FAINT))
        p.setColor(grp, QPalette.ColorRole.ButtonText, QColor(Theme.TEXT_FAINT))
        p.setColor(grp, QPalette.ColorRole.WindowText, QColor(Theme.TEXT_FAINT))
    app.setPalette(p)


def _build_splash():
    """Designed splash: void-black gradient with a glowing plasma hex sigil."""
    W, H = 700, 390
    pm = QPixmap(W, H)
    pm.fill(QColor(Theme.BG_BASE))
    pr = QPainter(pm)
    pr.setRenderHint(QPainter.RenderHint.Antialiasing)
    pr.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    grad = QLinearGradient(0, 0, W, H)
    grad.setColorAt(0.0, QColor("#0C1422"))
    grad.setColorAt(1.0, QColor(Theme.BG_BASE))
    pr.fillRect(0, 0, W, H, QBrush(grad))

    # plasma glow
    cx, cy, r = 150, H // 2, 70
    glow = QRadialGradient(cx, cy, r * 2.1)
    glow.setColorAt(0.0, QColor(70, 229, 255, 150))
    glow.setColorAt(0.5, QColor(123, 92, 255, 45))
    glow.setColorAt(1.0, QColor(123, 92, 255, 0))
    pr.setPen(Qt.PenStyle.NoPen)
    pr.setBrush(QBrush(glow))
    pr.drawEllipse(int(cx - r * 2.1), int(cy - r * 2.1), int(r * 4.2), int(r * 4.2))

    # hexagon sigil
    import math as _m
    pts = []
    for i in range(6):
        ang = _m.pi / 6 + i * _m.pi / 3
        pts.append(QtCore.QPointF(cx + r * _m.cos(ang), cy + r * _m.sin(ang)))
    poly = QtGui.QPolygonF(pts)
    pr.setPen(QPen(QColor(Theme.ACCENT), 2.6))
    pr.setBrush(Qt.BrushStyle.NoBrush)
    pr.drawPolygon(poly)
    pr.setPen(QPen(QColor(Theme.MAGENTA), 1.4))
    inner = QtGui.QPolygonF([QtCore.QPointF(cx + (r * 0.5) * _m.cos(_m.pi / 6 + i * _m.pi / 3),
                                            cy + (r * 0.5) * _m.sin(_m.pi / 6 + i * _m.pi / 3)) for i in range(6)])
    pr.drawPolygon(inner)

    # wordmark
    tx = 270
    pr.setPen(QColor(Theme.TEXT))
    f1 = QFont(Theme.UI_FONT, 27)
    f1.setWeight(QFont.Weight.ExtraBold)
    pr.setFont(f1)
    pr.drawText(tx, cy - 16, "Turbulence Realm")
    pr.setPen(QColor(Theme.ACCENT))
    pr.drawText(tx, cy + 24, "· SINDy")
    pr.setPen(QColor(Theme.TEXT_MUTED))
    f2 = QFont(Theme.MONO_FONT, 9)
    pr.setFont(f2)
    pr.drawText(tx + 2, cy + 58, "OPTICAL FLOW · SPARSE DYNAMICS · v1.0")
    pr.end()
    return pm


def main():
    app = QApplication(sys.argv)
    try:
        app.setStyle('Fusion')
    except Exception as e:
        print(f"Could not set Fusion style: {e}.")
    _apply_dark_palette(app)
    app.setFont(QFont(Theme.UI_FONT, 10))

    splash = QSplashScreen(_build_splash())
    splash.setFont(QFont(Theme.MONO_FONT, 9))
    splash.showMessage("  Initializing reactor…",
                       Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
                       QColor(Theme.TEXT_MUTED))
    splash.show()
    QApplication.processEvents()

    loading_steps = ["Loading core modules", "Building interface", "Arming solvers", "Ready"]
    for i, step in enumerate(loading_steps):
        splash.showMessage(f"  [{i + 1}/{len(loading_steps)}]  {step}…",
                           Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
                           QColor(Theme.ACCENT))
        QApplication.processEvents()
        time.sleep(0.4)

    win = FluidGui()
    win.show()
    splash.finish(win)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
