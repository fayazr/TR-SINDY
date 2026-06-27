"""Application entry point: palette, splash screen and main window launch."""

from __future__ import annotations

import sys

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

from ._logging import get_logger
from .theme import Theme as _T

log = get_logger(__name__)


def _apply_dark_palette(app):
    p = QPalette()
    base = QColor(_T.BG_BASE); surf = QColor(_T.SURFACE)
    surfhi = QColor(_T.SURFACE_HI); text = QColor(_T.TEXT)
    muted = QColor(_T.TEXT_MUTED); accent = QColor(_T.ACCENT)
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
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#04142B"))
    p.setColor(QPalette.ColorRole.PlaceholderText, muted)
    p.setColor(QPalette.ColorRole.Link, accent)
    for grp in (QPalette.ColorGroup.Disabled,):
        p.setColor(grp, QPalette.ColorRole.Text, QColor(_T.TEXT_FAINT))
        p.setColor(grp, QPalette.ColorRole.ButtonText, QColor(_T.TEXT_FAINT))
        p.setColor(grp, QPalette.ColorRole.WindowText, QColor(_T.TEXT_FAINT))
    app.setPalette(p)


def _build_splash():
    import math as _m
    W, H = 720, 400
    pm = QPixmap(W, H)
    pm.fill(QColor(_T.BG_BASE))
    pr = QPainter(pm)
    pr.setRenderHint(QPainter.RenderHint.Antialiasing)
    pr.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    # Diagonal gradient background
    grad = QLinearGradient(0, 0, W, H)
    grad.setColorAt(0.0, QColor(_T.BG_GRAD_TOP))
    grad.setColorAt(0.6, QColor(_T.BG_BASE))
    grad.setColorAt(1.0, QColor("#040608"))
    pr.fillRect(0, 0, W, H, QBrush(grad))
    cx, cy, r = 160, H // 2, 72
    # Glow halo behind hexagon
    glow = QRadialGradient(cx, cy, r * 2.3)
    glow.setColorAt(0.0, QColor(34, 211, 238, 160))
    glow.setColorAt(0.4, QColor(129, 140, 248, 60))
    glow.setColorAt(1.0, QColor(129, 140, 248, 0))
    pr.setPen(Qt.PenStyle.NoPen); pr.setBrush(QBrush(glow))
    pr.drawEllipse(int(cx - r * 2.3), int(cy - r * 2.3), int(r * 4.6), int(r * 4.6))
    # Outer hexagon
    pts = [QtCore.QPointF(cx + r * _m.cos(_m.pi / 6 + i * _m.pi / 3),
                          cy + r * _m.sin(_m.pi / 6 + i * _m.pi / 3)) for i in range(6)]
    pr.setPen(QPen(QColor(_T.ACCENT), 2.8)); pr.setBrush(Qt.BrushStyle.NoBrush)
    pr.drawPolygon(QtGui.QPolygonF(pts))
    # Inner hexagon
    pr.setPen(QPen(QColor(_T.ACCENT_2), 1.6))
    inner = QtGui.QPolygonF([QtCore.QPointF(
        cx + (r * 0.5) * _m.cos(_m.pi / 6 + i * _m.pi / 3),
        cy + (r * 0.5) * _m.sin(_m.pi / 6 + i * _m.pi / 3)) for i in range(6)])
    pr.drawPolygon(inner)
    # Title text
    tx = 285
    pr.setPen(QColor(_T.TEXT))
    f1 = QFont(_T.UI_FONT, 28); f1.setWeight(QFont.Weight.ExtraBold)
    pr.setFont(f1); pr.drawText(tx, cy - 14, "Turbulence Realm")
    pr.setPen(QColor(_T.ACCENT))
    f2 = QFont(_T.UI_FONT, 22); f2.setWeight(QFont.Weight.Bold)
    pr.setFont(f2); pr.drawText(tx, cy + 22, "· SINDy")
    pr.setPen(QColor(_T.TEXT_MUTED))
    pr.setFont(QFont(_T.MONO_FONT, 9))
    from . import __version__ as _v
    pr.drawText(tx + 2, cy + 56, f"OPTICAL FLOW · SPARSE DYNAMICS · v{_v}")
    # Bottom accent line
    pr.setPen(QPen(QColor(_T.ACCENT), 2))
    pr.drawLine(0, H - 3, W, H - 3)
    pr.end()
    return pm


def main():
    from .gui import FluidGui
    from .theme import apply_matplotlib_theme
    app = QtWidgets.QApplication(sys.argv)
    try:
        app.setStyle('Fusion')
    except Exception as e:
        log.warning("Could not set Fusion style: %s", e)
    _apply_dark_palette(app)
    app.setFont(QFont(_T.UI_FONT, 10))

    splash = QtWidgets.QSplashScreen(_build_splash())
    splash.setFont(QFont(_T.MONO_FONT, 9))
    splash.showMessage("  Loading…",
                       Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
                       QColor(_T.TEXT_MUTED))
    splash.show()
    QtWidgets.QApplication.processEvents()

    apply_matplotlib_theme()
    win = FluidGui()
    win.show()
    splash.finish(win)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
