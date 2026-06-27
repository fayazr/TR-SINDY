"""Turbulence Realm - SINDy application package.

A modular fluid-flow analysis platform combining optical flow, SINDy,
machine-learning methods, advanced analysis, rich visualisation and
flexible export.

Public entry point: :func:`tr_sindy_app.app.main`.
"""

from __future__ import annotations

try:  # Prefer version from installed package metadata.
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        __version__ = _pkg_version("tr-sindy")
    except PackageNotFoundError:
        __version__ = "2.1.0"
except Exception:  # pragma: no cover - very old Python or no metadata
    __version__ = "2.1.0"

from .app import main

__all__ = ["main", "__version__"]
