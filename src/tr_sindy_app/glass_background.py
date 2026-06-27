"""Glassmorphism background widget.

Paints a rich gradient background with soft glowing orbs so that
translucent surfaces (QGroupBox, QFrame#card) placed on top actually
show visible depth — the core of the glassmorphism aesthetic.

Also provides helper functions to apply drop-shadow and blur effects
to individual widgets for the frosted-glass look.
"""

from __future__ import annotations

import math

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QLinearGradient,
    QPainter,
    QRadialGradient,
)
from PyQt6.QtWidgets import QGraphicsBlurEffect, QGraphicsDropShadowEffect

from .theme import Theme


class GlassBackground(QtWidgets.QWidget):
    """Widget that paints a gradient + glowing orbs background.

    Place this as the bottom-most widget in a layout, then put translucent
    surfaces on top of it. The orbs and gradient will show through.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._orbs = self._generate_orbs()
        self._anim_phase = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)

    def _generate_orbs(self):
        """Generate static orb positions/sizes/colors."""
        import random
        rng = random.Random(42)  # deterministic
        orbs = []
        colors = [
            (Theme.ACCENT, 0.35),
            (Theme.ACCENT_2, 0.30),
            (Theme.MAGENTA, 0.20),
            (Theme.ACCENT, 0.25),
            (Theme.ACCENT_2, 0.18),
            (Theme.GOOD, 0.12),
        ]
        for i in range(6):
            orbs.append({
                "x": rng.uniform(0.05, 0.95),
                "y": rng.uniform(0.05, 0.95),
                "r": rng.uniform(180, 400),
                "color": colors[i % len(colors)],
                "phase": rng.uniform(0, 2 * math.pi),
                "speed": rng.uniform(0.2, 0.5),
            })
        return orbs

    def _tick(self):
        self._anim_phase += 0.02
        self.update()

    def paintEvent(self, event):
        pr = QPainter(self)
        pr.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Base gradient (top-left → bottom-right)
        grad = QLinearGradient(0, 0, w, h)
        grad.setColorAt(0.0, QColor("#0E1A2E"))
        grad.setColorAt(0.5, QColor(Theme.BG_BASE))
        grad.setColorAt(1.0, QColor("#02040A"))
        pr.fillRect(0, 0, w, h, QBrush(grad))

        # Glowing orbs (slowly drifting)
        for orb in self._orbs:
            cx = orb["x"] * w + math.sin(self._anim_phase * orb["speed"]
                                         + orb["phase"]) * 30
            cy = orb["y"] * h + math.cos(self._anim_phase * orb["speed"]
                                         + orb["phase"] * 1.3) * 20
            r = orb["r"]
            color_hex, alpha = orb["color"]
            # Parse hex color
            c = QColor(color_hex)
            glow = QRadialGradient(cx, cy, r)
            glow.setColorAt(0.0, QColor(c.red(), c.green(), c.blue(),
                                        int(255 * alpha)))
            glow.setColorAt(0.5, QColor(c.red(), c.green(), c.blue(),
                                        int(255 * alpha * 0.3)))
            glow.setColorAt(1.0, QColor(c.red(), c.green(), c.blue(), 0))
            pr.setPen(Qt.PenStyle.NoPen)
            pr.setBrush(QBrush(glow))
            pr.drawEllipse(int(cx - r), int(cy - r), int(r * 2), int(r * 2))

        pr.end()


def apply_drop_shadow(widget, blur_radius=20, y_offset=4, alpha=80):
    """Apply a soft drop shadow to a widget for depth.

    This is what makes glass surfaces appear to "float" above the background.
    """
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur_radius)
    effect.setOffset(0, y_offset)
    color = QColor(0, 0, 0, alpha)
    effect.setColor(color)
    widget.setGraphicsEffect(effect)
    return effect


def apply_glow_border(widget, color=None, blur_radius=16, alpha=60):
    """Apply a colored glow around a widget (like a neon border).

    Uses a drop shadow with the accent color instead of black.
    """
    if color is None:
        color = Theme.ACCENT
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur_radius)
    effect.setOffset(0, 0)
    c = QColor(color) if isinstance(color, str) else QColor(color)
    c.setAlpha(alpha)
    effect.setColor(c)
    widget.setGraphicsEffect(effect)
    return effect


def apply_blur(widget, blur_radius=15):
    """Apply a blur effect to a widget (for frosted-glass backgrounds).

    Note: this blurs the widget's own content, not what's behind it.
    For true backdrop blur, place a GlassBackground behind and blur it.
    """
    effect = QGraphicsBlurEffect(widget)
    effect.setBlurRadius(blur_radius)
    widget.setGraphicsEffect(effect)
    return effect
