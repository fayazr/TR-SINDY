"""Pytest configuration and shared fixtures for TR-SINDy tests."""

from __future__ import annotations

import os
import sys

# Make the src/ layout importable without installing the package.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
