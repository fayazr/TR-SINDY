"""In-app ROI + calibration selector rendered entirely in Qt.

Two-step selector on the first video frame:
    1. drag a rectangle  -> region of interest
    2. drag a line        -> calibration reference

Adds multi-level undo/redo for the ROI and calibration steps so users can
revert accidental selections without restarting the dialog.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
)

from .theme import Theme


class ROICalibDialog(QtWidgets.QDialog):
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
        self.roi_disp = None
        self.p1 = self.p2 = None
        self._start = self._cur = None
        self._dragging = False

        # Undo/redo stacks keyed by stage.
        self._undo = {'roi': [], 'calib': []}
        self._redo = {'roi': [], 'calib': []}

        v = QVBoxLayout(self)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)

        self.info = QtWidgets.QLabel()
        self.info.setObjectName("readout")
        v.addWidget(self.info)

        self.canvas = QtWidgets.QLabel()
        self.canvas.setFixedSize(self.disp_w, self.disp_h)
        self.canvas.setMouseTracking(True)
        self.canvas.mousePressEvent = self._press
        self.canvas.mouseMoveEvent = self._move
        self.canvas.mouseReleaseEvent = self._release
        v.addWidget(self.canvas, alignment=Qt.AlignmentFlag.AlignCenter)

        row = QHBoxLayout()
        row.setSpacing(10)
        self.undo_btn = QtWidgets.QPushButton("Undo")
        self.undo_btn.setToolTip("Revert the last ROI / calibration change (Ctrl+Z)")
        self.undo_btn.clicked.connect(self._undo_step)
        self.redo_btn = QtWidgets.QPushButton("Redo")
        self.redo_btn.setToolTip("Re-apply a reverted change (Ctrl+Y)")
        self.redo_btn.clicked.connect(self._redo_step)
        self.reset_btn = QtWidgets.QPushButton("Reset")
        self.reset_btn.clicked.connect(self._reset_stage)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.primary_btn = QtWidgets.QPushButton("Next: Calibration  ▶")
        self.primary_btn.setProperty("variant", "primary")
        self.primary_btn.clicked.connect(self._advance)
        row.addWidget(self.undo_btn)
        row.addWidget(self.redo_btn)
        row.addWidget(self.reset_btn)
        row.addWidget(self.cancel_btn)
        row.addStretch(1)
        row.addWidget(self.primary_btn)
        v.addLayout(row)

        # Keyboard shortcuts.
        self.undo_btn.setShortcut(QtGui.QKeySequence("Ctrl+Z"))
        self.redo_btn.setShortcut(QtGui.QKeySequence("Ctrl+Y"))

        self._refresh()

    # ----- coordinate mapping -----
    def _to_frame(self, x, y):
        fx = int(round(min(max(x, 0), self.disp_w) / self.scale))
        fy = int(round(min(max(y, 0), self.disp_h) / self.scale))
        fx = min(max(fx, 0), self.frame_w - 1)
        fy = min(max(fy, 0), self.frame_h - 1)
        return fx, fy

    # ----- history helpers -----
    def _snapshot(self):
        if self.stage == 'roi':
            return self.roi_disp
        return (self.p1, self.p2)

    def _restore(self, snap):
        if self.stage == 'roi':
            self.roi_disp = snap
        else:
            self.p1, self.p2 = snap

    def _push_undo(self):
        self._undo[self.stage].append(self._snapshot())
        self._redo[self.stage].clear()
        self._update_history_buttons()

    def _undo_step(self):
        st = self.stage
        if self._undo[st]:
            self._redo[st].append(self._snapshot())
            self._restore(self._undo[st].pop())
            self._refresh()
        self._update_history_buttons()

    def _redo_step(self):
        st = self.stage
        if self._redo[st]:
            self._undo[st].append(self._snapshot())
            self._restore(self._redo[st].pop())
            self._refresh()
        self._update_history_buttons()

    def _update_history_buttons(self):
        st = self.stage
        self.undo_btn.setEnabled(bool(self._undo[st]))
        self.redo_btn.setEnabled(bool(self._redo[st]))

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
            self._push_undo()
            if self.stage == 'roi':
                self.roi_disp = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            else:
                self.p1, self.p2 = (x0, y0), (x1, y1)
            self._refresh()

    # ----- stage control -----
    def _reset_stage(self):
        self._push_undo()
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
            self._update_history_buttons()
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

        if self.roi_disp:
            x0, y0, x1, y1 = self.roi_disp
            pr.setPen(QPen(QColor(Theme.ACCENT), 2))
            pr.setBrush(QColor(70, 229, 255, 28))
            pr.drawRect(int(x0), int(y0), int(x1 - x0), int(y1 - y0))

        if self.p1 and self.p2:
            pr.setPen(QPen(QColor(Theme.AMBER), 2))
            pr.drawLine(int(self.p1[0]), int(self.p1[1]), int(self.p2[0]), int(self.p2[1]))
            for p in (self.p1, self.p2):
                pr.setBrush(QColor(Theme.AMBER))
                pr.drawEllipse(int(p[0]) - 4, int(p[1]) - 4, 8, 8)

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
        self._update_history_buttons()

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
