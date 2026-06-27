"""Theme tokens, Qt stylesheet and matplotlib theming.

Supports a dark (default "Flux Reactor") and a light theme, switchable at
runtime via :func:`apply_theme`.

The v2.2 redesign introduces a modern aesthetic with:
  * Deeper, richer dark palette with blue-violet gradient accents
  * Glassmorphism-style surfaces with subtle translucency
  * Glowing focus states and hover transitions
  * Animated gradient progress bars
  * Modernized navigation rail with active-state glow
  * Refined typography with tighter letter-spacing
"""

from __future__ import annotations

import matplotlib


# ---------------------------------------------------------------------
#  Palettes
# ---------------------------------------------------------------------
class _Dark:
    BG_BASE      = "#060912"
    BG_GRAD_TOP  = "#0A1020"
    SURFACE      = "#0D1424"
    SURFACE_HI   = "#141E33"
    SURFACE_GLOW = "#1A2740"
    HAIRLINE     = "#1E2A42"
    BORDER_HOVER = "#345880"
    TEXT         = "#EDF4FF"
    TEXT_MUTED   = "#8BA1C0"
    TEXT_FAINT   = "#4A5C78"
    ACCENT       = "#22D3EE"
    ACCENT_2     = "#818CF8"
    ACCENT_DEEP  = "#0E7490"
    MAGENTA      = "#F472B6"
    AMBER        = "#FBBF24"
    DANGER       = "#F87171"
    GOOD         = "#34D399"
    # Gradient endpoints for buttons / accents
    GRAD_ACCENT_1 = "#22D3EE"
    GRAD_ACCENT_2 = "#818CF8"
    GRAD_VIOLET_1 = "#818CF8"
    GRAD_VIOLET_2 = "#C084FC"
    # Glow colors (rgba)
    GLOW_ACCENT   = "rgba(34, 211, 238, 0.35)"
    GLOW_VIOLET   = "rgba(129, 140, 248, 0.30)"
    GLOW_GOOD     = "rgba(52, 211, 153, 0.30)"
    GLOW_AMBER    = "rgba(251, 191, 36, 0.30)"
    GLOW_DANGER   = "rgba(248, 113, 113, 0.30)"
    UI_FONT   = "Inter"
    MONO_FONT = "JetBrains Mono"
    BGR_ACCENT  = (255, 229, 70)
    BGR_AMBER   = (75, 194, 255)
    BGR_MAGENTA = (216, 79, 255)


class _Light:
    BG_BASE      = "#F0F4FA"
    BG_GRAD_TOP  = "#E4EAF3"
    SURFACE      = "#FFFFFF"
    SURFACE_HI   = "#EDF2F8"
    SURFACE_GLOW = "#E0E7F0"
    HAIRLINE     = "#C8D3E4"
    BORDER_HOVER = "#8FA8C8"
    TEXT         = "#0B1526"
    TEXT_MUTED   = "#3D5273"
    TEXT_FAINT   = "#7A8DA8"
    ACCENT       = "#0891B2"
    ACCENT_2     = "#6366F1"
    ACCENT_DEEP  = "#0E7490"
    MAGENTA      = "#DB2777"
    AMBER        = "#D97706"
    DANGER       = "#DC2626"
    GOOD         = "#059669"
    GRAD_ACCENT_1 = "#0891B2"
    GRAD_ACCENT_2 = "#6366F1"
    GRAD_VIOLET_1 = "#6366F1"
    GRAD_VIOLET_2 = "#A855F7"
    GLOW_ACCENT   = "rgba(8, 145, 178, 0.20)"
    GLOW_VIOLET   = "rgba(99, 102, 241, 0.18)"
    GLOW_GOOD     = "rgba(5, 150, 105, 0.18)"
    GLOW_AMBER    = "rgba(217, 119, 6, 0.18)"
    GLOW_DANGER   = "rgba(220, 38, 38, 0.18)"
    UI_FONT   = "Inter"
    MONO_FONT = "JetBrains Mono"
    BGR_ACCENT  = (184, 143, 10)
    BGR_AMBER   = (26, 131, 201)
    BGR_MAGENTA = (174, 31, 200)


# Active theme singleton; swap via apply_theme().
class Theme:
    pass


def _install(theme: str):
    src = _Dark if theme == "dark" else _Light
    for k in dir(src):
        if k.isupper():
            setattr(Theme, k, getattr(src, k))
    Theme.NAME = theme


_install("dark")


def apply_theme(name: str):
    """Switch the active palette to 'dark' or 'light'."""
    if name not in ("dark", "light"):
        raise ValueError(name)
    _install(name)


# ---------------------------------------------------------------------
#  Qt stylesheet
# ---------------------------------------------------------------------
def stylesheet() -> str:
    t = Theme
    return f"""
    /* ================================================================
       Global
       ================================================================ */
    QWidget {{
        font-family: '{t.UI_FONT}', 'Segoe UI', sans-serif;
        color: {t.TEXT};
        font-size: 10pt;
    }}
    QWidget#contentRoot, QMainWindow {{
        background-color: transparent;
    }}

    /* ================================================================
       Header bar
       ================================================================ */
    QFrame#headerBar {{
        background-color: {t.BG_GRAD_TOP};
        border: none; border-bottom: 1px solid {t.HAIRLINE};
    }}
    QLabel#appTitle {{
        font-size: 18pt; font-weight: 800; color: {t.TEXT}; letter-spacing: -0.5px;
    }}
    QLabel#appSubtitle {{
        font-family: '{t.MONO_FONT}', monospace; font-size: 8pt;
        color: {t.TEXT_MUTED}; letter-spacing: 3px;
    }}
    QLabel#statusPill {{
        font-family: '{t.MONO_FONT}', monospace; font-size: 8pt; font-weight: bold;
        color: {t.ACCENT}; background-color: {t.GLOW_ACCENT};
        border: 1px solid rgba(34, 211, 238, 0.40); border-radius: 11px;
        padding: 4px 14px; letter-spacing: 1px;
    }}

    /* ================================================================
       Tabs — modern pill-style with gradient underline
       ================================================================ */
    QTabWidget::pane {{
        border: 1px solid {t.HAIRLINE}; border-radius: 14px;
        background: {t.SURFACE}; top: -1px;
    }}
    QTabBar::tab {{
        background: transparent; color: {t.TEXT_MUTED}; border: none;
        border-top-left-radius: 10px; border-top-right-radius: 10px;
        min-width: 140px; padding: 10px 22px; margin-right: 2px;
        font-weight: 600; font-size: 9.5pt;
    }}
    QTabBar::tab:selected {{
        color: {t.ACCENT}; background: {t.SURFACE};
        border-bottom: 2px solid {t.ACCENT};
    }}
    QTabBar::tab:hover:!selected {{
        color: {t.TEXT}; border-bottom: 2px solid {t.HAIRLINE};
    }}

    /* ================================================================
       Group boxes — glassmorphism cards (translucent so background
       gradient + orbs show through; drop shadows applied in code)
       ================================================================ */
    QGroupBox {{
        background-color: rgba(13, 20, 36, 0.45);
        border: 1px solid rgba(34, 211, 238, 0.25);
        border-radius: 14px; margin-top: 14px; padding: 12px 10px 10px 10px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin; subcontrol-position: top left;
        left: 14px; top: 3px; padding: 0 6px; color: {t.ACCENT};
        font-family: '{t.MONO_FONT}', monospace; font-size: 8pt;
        font-weight: bold; letter-spacing: 1.5px;
    }}

    /* ================================================================
       Buttons — gradient fills, glow on hover
       ================================================================ */
    QPushButton {{
        background-color: {t.SURFACE_HI}; color: {t.TEXT};
        border: 1px solid {t.HAIRLINE}; border-radius: 10px;
        padding: 11px 18px; min-height: 20px; font-size: 10pt; font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {t.SURFACE_GLOW};
        border: 1px solid {t.BORDER_HOVER};
    }}
    QPushButton:pressed {{ background-color: {t.SURFACE}; }}
    QPushButton:disabled {{
        background-color: {t.SURFACE}; color: {t.TEXT_FAINT};
        border: 1px solid {t.HAIRLINE};
    }}

    /* Primary — gradient cyan→indigo with glow */
    QPushButton[variant="primary"] {{
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {t.GRAD_ACCENT_1}, stop:1 {t.GRAD_ACCENT_2});
        color: #04142B; border: 1px solid transparent; font-weight: 800;
        border-radius: 10px;
    }}
    QPushButton[variant="primary"]:hover {{
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {t.GRAD_ACCENT_2}, stop:1 {t.GRAD_VIOLET_2});
        border: 1px solid transparent;
    }}
    QPushButton[variant="primary"]:pressed {{
        background-color: {t.ACCENT_DEEP}; color: {t.TEXT};
    }}
    QPushButton[variant="primary"]:disabled {{
        background-color: {t.SURFACE}; color: {t.TEXT_FAINT};
        border: 1px solid {t.HAIRLINE};
    }}

    /* Violet variant — translucent with glow border */
    QPushButton[variant="violet"] {{
        background-color: rgba(129, 140, 248, 0.14); color: #C6B8FF;
        border: 1px solid rgba(129, 140, 248, 0.45); font-weight: 700;
        border-radius: 10px;
    }}
    QPushButton[variant="violet"]:hover {{
        background-color: rgba(129, 140, 248, 0.24);
        border: 1px solid rgba(129, 140, 248, 0.65);
    }}
    QPushButton[variant="violet"]:disabled {{
        background-color: {t.SURFACE}; color: {t.TEXT_FAINT};
        border: 1px solid {t.HAIRLINE};
    }}

    /* ================================================================
       Labels
       ================================================================ */
    QLabel {{ color: {t.TEXT}; font-size: 10pt; background: transparent; }}
    QLabel#fieldLabel {{ color: {t.TEXT_MUTED}; font-size: 9pt; }}
    QLabel#readout {{
        font-family: '{t.MONO_FONT}', monospace; font-size: 9pt; color: {t.ACCENT};
        background-color: {t.BG_BASE}; border: 1px solid {t.HAIRLINE};
        border-radius: 8px; padding: 8px 12px;
    }}
    QLabel#stepGuide {{
        font-family: '{t.MONO_FONT}', monospace; font-size: 9pt; color: {t.TEXT_MUTED};
        background-color: rgba(6, 9, 18, 0.6); border: 1px solid {t.HAIRLINE};
        border-radius: 12px; padding: 16px 18px;
    }}

    /* ================================================================
       Line edits — glowing focus
       ================================================================ */
    QLineEdit {{
        border: 1px solid {t.HAIRLINE}; border-radius: 8px; padding: 9px 12px;
        font-family: '{t.MONO_FONT}', monospace; font-size: 10pt; color: {t.TEXT};
        background-color: {t.BG_BASE}; selection-background-color: {t.ACCENT};
        selection-color: #04142B;
    }}
    QLineEdit:focus {{
        border: 1px solid {t.ACCENT};
    }}
    QLineEdit:disabled {{ color: {t.TEXT_FAINT}; background-color: {t.SURFACE}; }}

    QPlainTextEdit, QTextEdit, QTableWidget, QTableView {{
        background-color: {t.BG_BASE}; color: {t.TEXT}; border: 1px solid {t.HAIRLINE};
        border-radius: 10px; font-family: '{t.MONO_FONT}', monospace; font-size: 10pt;
        selection-background-color: {t.ACCENT}; selection-color: #04142B;
    }}
    QTableWidget::item {{ padding: 4px 8px; }}
    QHeaderView::section {{
        background-color: {t.SURFACE_HI}; color: {t.TEXT_MUTED};
        border: none; border-right: 1px solid {t.HAIRLINE}; padding: 8px 10px;
        font-weight: 600; font-size: 9pt;
    }}

    /* ================================================================
       Combo boxes — modern dropdown
       ================================================================ */
    QComboBox {{
        background-color: {t.SURFACE_HI}; color: {t.TEXT};
        border: 1px solid {t.HAIRLINE}; border-radius: 8px; padding: 8px 12px;
        min-height: 20px; font-weight: 500;
    }}
    QComboBox:hover {{ border: 1px solid {t.BORDER_HOVER}; }}
    QComboBox:focus {{ border: 1px solid {t.ACCENT}; }}
    QComboBox::drop-down {{
        border: none; width: 28px;
        background: transparent;
    }}
    QComboBox::down-arrow {{
        image: none; width: 0; height: 0;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 6px solid {t.TEXT_MUTED};
        margin-right: 10px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {t.SURFACE_HI}; color: {t.TEXT};
        selection-background-color: {t.ACCENT}; selection-color: #04142B;
        border: 1px solid {t.HAIRLINE}; border-radius: 8px;
        padding: 4px; outline: none;
    }}

    /* ================================================================
       Checkboxes / Radio — custom indicators with glow
       ================================================================ */
    QCheckBox, QRadioButton {{ color: {t.TEXT}; spacing: 8px; font-size: 10pt; }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 18px; height: 18px; border: 1.5px solid {t.HAIRLINE};
        border-radius: 5px; background: {t.BG_BASE};
    }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
        border: 1.5px solid {t.BORDER_HOVER};
    }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 {t.GRAD_ACCENT_1}, stop:1 {t.GRAD_ACCENT_2});
        border: 1.5px solid {t.ACCENT};
    }}
    QRadioButton::indicator {{ border-radius: 9px; }}
    QRadioButton::indicator:checked {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 {t.GRAD_ACCENT_1}, stop:1 {t.GRAD_ACCENT_2});
        border: 1.5px solid {t.ACCENT};
    }}

    /* ================================================================
       Progress bar — animated gradient chunk
       ================================================================ */
    QProgressBar {{
        border: 1px solid {t.HAIRLINE}; border-radius: 9px; text-align: center;
        background-color: {t.BG_BASE}; color: {t.TEXT_MUTED};
        font-family: '{t.MONO_FONT}', monospace; font-size: 8.5pt; height: 22px;
    }}
    QProgressBar::chunk {{
        border-radius: 8px;
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {t.GRAD_ACCENT_1}, stop:0.5 {t.ACCENT}, stop:1 {t.GRAD_ACCENT_2});
    }}

    /* ================================================================
       Sliders — modern track + glowing handle
       ================================================================ */
    QSlider::groove:horizontal {{
        height: 6px; background: {t.HAIRLINE}; border-radius: 3px;
    }}
    QSlider::sub-page:horizontal {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {t.GRAD_ACCENT_1}, stop:1 {t.GRAD_ACCENT_2});
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {t.ACCENT}; width: 18px; height: 18px; margin: -7px 0;
        border-radius: 9px; border: 2px solid {t.BG_BASE};
    }}
    QSlider::handle:horizontal:hover {{
        background: {t.ACCENT_2}; border: 2px solid {t.BG_BASE};
    }}

    /* ================================================================
       Spin boxes
       ================================================================ */
    QSpinBox, QDoubleSpinBox {{
        background-color: {t.BG_BASE}; color: {t.TEXT};
        border: 1px solid {t.HAIRLINE}; border-radius: 8px;
        padding: 7px 10px; font-family: '{t.MONO_FONT}', monospace; font-size: 10pt;
    }}
    QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {t.ACCENT}; }}
    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        background: transparent; border: none; width: 20px;
    }}

    /* ================================================================
       Status bar
       ================================================================ */
    QStatusBar {{
        border-top: 1px solid {t.HAIRLINE}; font-family: '{t.MONO_FONT}', monospace;
        font-size: 9pt; background-color: {t.BG_GRAD_TOP}; color: {t.TEXT_MUTED};
    }}
    QStatusBar::item {{ border: none; }}

    /* ================================================================
       Menu bar + menus
       ================================================================ */
    QMenuBar {{
        background-color: {t.BG_GRAD_TOP}; color: {t.TEXT};
        border-bottom: 1px solid {t.HAIRLINE}; padding: 3px 8px;
        font-size: 10pt;
    }}
    QMenuBar::item {{
        background: transparent; padding: 7px 14px; border-radius: 7px;
        font-weight: 500;
    }}
    QMenuBar::item:selected {{
        background-color: {t.SURFACE_HI}; color: {t.ACCENT};
    }}
    QMenu {{
        background-color: {t.SURFACE_HI}; border: 1px solid {t.HAIRLINE};
        border-radius: 10px; padding: 6px;
    }}
    QMenu::item {{ padding: 8px 28px; border-radius: 7px; }}
    QMenu::item:selected {{
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {t.GRAD_ACCENT_1}, stop:1 {t.GRAD_ACCENT_2});
        color: #04142B; font-weight: 600;
    }}
    QMenu::separator {{ height: 1px; background: {t.HAIRLINE}; margin: 4px 10px; }}

    /* ================================================================
       Dialogs
       ================================================================ */
    QMessageBox, QInputDialog, QDialog {{
        background-color: {t.SURFACE};
    }}
    QMessageBox QLabel, QInputDialog QLabel {{ color: {t.TEXT}; font-size: 10pt; }}

    /* ================================================================
       Scrollbars — slim, modern, rounded
       ================================================================ */
    QScrollBar:vertical {{
        background: transparent; width: 10px; margin: 2px; border-radius: 5px;
    }}
    QScrollBar::handle:vertical {{
        background: {t.SURFACE_GLOW}; border-radius: 5px; min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {t.BORDER_HOVER}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; border: none; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QScrollBar:horizontal {{
        background: transparent; height: 10px; margin: 2px; border-radius: 5px;
    }}
    QScrollBar::handle:horizontal {{
        background: {t.SURFACE_GLOW}; border-radius: 5px; min-width: 30px;
    }}
    QScrollBar::handle:horizontal:hover {{ background: {t.BORDER_HOVER}; }}

    /* ================================================================
       Toolbars / tooltips
       ================================================================ */
    QToolBar {{ background: {t.SURFACE}; border: none; spacing: 2px; }}
    QToolButton {{
        background: transparent; border-radius: 7px; padding: 4px;
    }}
    QToolButton:hover {{ background: {t.SURFACE_HI}; }}
    QToolTip {{
        background-color: {t.SURFACE_HI}; color: {t.TEXT};
        border: 1px solid {t.ACCENT}; border-radius: 8px; padding: 8px 10px;
        font-size: 9pt;
    }}

    /* ================================================================
       Navigation rail — gradient background + active glow
       ================================================================ */
    QFrame#navRail {{
        background-color: rgba(10, 16, 32, 0.60);
        border: none; border-right: 1px solid {t.HAIRLINE};
    }}
    QFrame#navBrand {{
        background: transparent; border: none;
        border-bottom: 1px solid {t.HAIRLINE};
    }}
    QLabel#navTitle {{
        font-size: 14pt; font-weight: 800; color: {t.TEXT}; background: transparent;
        letter-spacing: -0.3px;
    }}
    QLabel#navSubtitle {{
        font-family: '{t.MONO_FONT}', monospace; font-size: 9pt;
        color: {t.ACCENT_2}; background: transparent; letter-spacing: 1.5px;
        font-weight: 600;
    }}
    QPushButton#navButton {{
        background: transparent; color: {t.TEXT_MUTED}; border: none;
        border-left: 3px solid transparent; border-radius: 0;
        text-align: left; padding: 14px 20px; font-size: 10.5pt; font-weight: 600;
    }}
    QPushButton#navButton:hover {{
        background-color: {t.SURFACE_HI}; color: {t.TEXT};
        border-left: 3px solid {t.BORDER_HOVER};
    }}
    QPushButton#navButton:checked {{
        background-color: {t.SURFACE_HI}; color: {t.ACCENT};
        border-left: 3px solid {t.ACCENT}; font-weight: 800;
    }}
    QPushButton#navAux {{
        background: transparent; color: {t.TEXT_MUTED};
        border: 1px solid {t.HAIRLINE}; border-radius: 10px;
        padding: 10px 14px; font-size: 9pt; text-align: left; font-weight: 500;
    }}
    QPushButton#navAux:hover {{
        background-color: {t.SURFACE_HI}; color: {t.TEXT};
        border: 1px solid {t.BORDER_HOVER};
    }}

    /* ================================================================
       Controls panel — transparent (glass background shows through)
       ================================================================ */
    QWidget#controlsPanel {{
        background-color: transparent;
        border-right: 1px solid {t.HAIRLINE};
    }}

    /* ================================================================
       Content area headers
       ================================================================ */
    QLabel#contentHeader {{
        font-family: '{t.MONO_FONT}', monospace; font-size: 9pt; font-weight: bold;
        color: {t.ACCENT}; letter-spacing: 2px; background: transparent;
        padding: 2px 0 8px 0;
    }}

    /* ================================================================
       Pipeline stepper — glowing dots
       ================================================================ */
    QWidget#stepper {{
        background-color: rgba(13, 20, 36, 0.50);
        border: 1px solid rgba(34, 211, 238, 0.15);
        border-radius: 14px;
    }}
    QLabel#stepperTitle {{
        font-family: '{t.MONO_FONT}', monospace; font-size: 8pt; font-weight: bold;
        color: {t.ACCENT}; letter-spacing: 2px; background: transparent;
    }}
    QLabel#stepperBar {{ color: {t.HAIRLINE}; background: transparent; }}

    /* ================================================================
       Card frame — glassmorphism (translucent)
       ================================================================ */
    QFrame#card {{
        background-color: rgba(13, 20, 36, 0.50);
        border: 1px solid rgba(34, 211, 238, 0.15);
        border-radius: 14px;
    }}

    /* ================================================================
       Splitter handle
       ================================================================ */
    QSplitter::handle {{ background-color: {t.HAIRLINE}; }}
    QSplitter::handle:horizontal {{ width: 1px; }}
    QSplitter::handle:vertical {{ height: 1px; }}

    /* ================================================================
       Scroll area — transparent so glass background shows through
       ================================================================ */
    QScrollArea {{
        background-color: transparent; border: none;
    }}
    QScrollArea > QWidget > QWidget {{
        background-color: transparent;
    }}

    /* ================================================================
       Frame polish
       ================================================================ */
    QFrame[variant="glow"] {{
        border: 1px solid {t.ACCENT};
        border-radius: 14px;
    }}
    """


def apply_matplotlib_theme():
    """Style matplotlib to match the active console theme."""
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


def cv2_has_gui() -> bool:
    """True if the installed OpenCV build ships highgui (namedWindow/imshow)."""
    import cv2
    try:
        info = cv2.getBuildInformation()
    except Exception:
        return False
    for key in ("GTK+", "QT", "WIN32UI", "Cocoa", "GTK"):
        for line in info.splitlines():
            if key in line and "YES" in line.upper():
                return True
    return False
