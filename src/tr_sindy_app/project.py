"""Project save/load, parameter presets, processing history and recent
files.

A project file (.trsindy) is a JSON archive that stores all parameters,
ROI/calibration, optical-flow config, SINDy config, references to the
velocity memmaps (which live alongside the project file), and a full
processing-history log. Results memmaps can optionally be copied into the
project directory for portability.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from ._logging import get_logger

log = get_logger(__name__)


PROJECT_EXT = ".trsindy"
PRESETS_DIR = os.path.join(os.path.expanduser("~"), ".tr_sindy", "presets")
RECENT_FILE = os.path.join(os.path.expanduser("~"), ".tr_sindy", "recent.json")


# ---------------------------------------------------------------------
#  Parameter presets
# ---------------------------------------------------------------------
@dataclass
class Preset:
    name: str
    description: str = ""
    optical_flow: dict = field(default_factory=dict)
    sindy: dict = field(default_factory=dict)
    visualization: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Preset:
        return cls(**{k: d.get(k) for k in
                      ("name", "description", "optical_flow", "sindy", "visualization")})


def list_presets() -> list[str]:
    os.makedirs(PRESETS_DIR, exist_ok=True)
    return sorted(f[:-len(PROJECT_EXT)] for f in os.listdir(PRESETS_DIR)
                  if f.endswith(PROJECT_EXT))


def load_preset(name: str) -> Preset:
    path = os.path.join(PRESETS_DIR, name + PROJECT_EXT)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No preset named {name!r}")
    with open(path) as f:
        return Preset.from_dict(json.load(f))


def save_preset(preset: Preset) -> str:
    os.makedirs(PRESETS_DIR, exist_ok=True)
    path = os.path.join(PRESETS_DIR, preset.name + PROJECT_EXT)
    with open(path, "w") as f:
        json.dump(preset.to_dict(), f, indent=2)
    return path


def delete_preset(name: str) -> None:
    path = os.path.join(PRESETS_DIR, name + PROJECT_EXT)
    if os.path.exists(path):
        os.remove(path)


# A few built-in presets for common scenarios.
BUILTIN_PRESETS = [
    Preset("farneback-default", "Classic Farneback + polynomial SINDy",
           optical_flow={"backend": "farneback", "multiscale": False,
                         "temporal_smoothing": "none"},
           sindy={"library": "polynomial", "degree": 3, "threshold": 0.07}),
    Preset("tvl1-turbulent", "TV-L1 flow for turbulent, high-gradient video",
           optical_flow={"backend": "tvl1", "multiscale": True,
                         "temporal_smoothing": "ema", "temporal_alpha": 0.6},
           sindy={"library": "combined", "degree": 3, "threshold": 0.05}),
    Preset("raft-precise", "RAFT deep-learning flow (needs torch) + Fourier SINDy",
           optical_flow={"backend": "raft", "multiscale": False,
                         "temporal_smoothing": "moving", "temporal_window": 3},
           sindy={"library": "fourier", "n_freq": 2, "threshold": 0.05}),
    Preset("lk-fast", "Lucas-Kanade sparse flow for quick previews",
           optical_flow={"backend": "lucas_kanade", "multiscale": False},
           sindy={"library": "polynomial", "degree": 2, "threshold": 0.1}),
]


def install_builtin_presets() -> None:
    for p in BUILTIN_PRESETS:
        try:
            save_preset(p)
        except Exception as e:
            log.debug("could not install preset %s: %s", p.name, e)


# ---------------------------------------------------------------------
#  Processing history
# ---------------------------------------------------------------------
@dataclass
class HistoryEntry:
    timestamp: float
    step: str
    params: dict
    status: str = "ok"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ProcessingHistory:
    def __init__(self):
        self.entries: list[HistoryEntry] = []

    def log(self, step: str, params: dict, status: str = "ok",
            notes: str = "") -> None:
        self.entries.append(HistoryEntry(time.time(), step, params, status, notes))

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self.entries]

    @classmethod
    def from_list(cls, lst: list[dict]) -> ProcessingHistory:
        h = cls()
        for d in lst:
            h.entries.append(HistoryEntry(**d))
        return h

    def summary(self) -> str:
        return "\n".join(f"{time.strftime('%H:%M:%S', time.localtime(e.timestamp))}  "
                         f"{e.step:20s} {e.status:6s} {e.notes}" for e in self.entries)


# ---------------------------------------------------------------------
#  Project save/load
# ---------------------------------------------------------------------
def save_project(path: str, project_state: dict,
                 mmap_paths: Optional[dict] = None,
                 bundle_mmaps: bool = True) -> str:
    """Write a .trsindy project file.

    project_state should contain: video_file, roi_box, calibration_*,
    meters_per_pixel, optical_flow config, sindy config, history, metadata.
    If bundle_mmaps is True, the velocity/frame memmaps are copied next to
    the project file so it is self-contained.
    """
    if not path.endswith(PROJECT_EXT):
        path += PROJECT_EXT
    project_dir = os.path.dirname(os.path.abspath(path))
    state = dict(project_state)
    bundled = {}
    if bundle_mmaps and mmap_paths:
        os.makedirs(project_dir, exist_ok=True)
        for key, src in mmap_paths.items():
            if src and os.path.exists(src):
                dst = os.path.join(project_dir, os.path.basename(src))
                shutil.copy2(src, dst)
                bundled[key] = dst
    state["bundled_mmaps"] = bundled
    state["saved_at"] = time.time()
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=_json_default)
    register_recent(path)
    return path


def load_project(path: str) -> dict:
    with open(path) as f:
        state = json.load(f)
    # resolve bundled mmap paths relative to the project file
    project_dir = os.path.dirname(os.path.abspath(path))
    bundled = state.get("bundled_mmaps", {})
    for key, p in list(bundled.items()):
        if not os.path.isabs(p):
            bundled[key] = os.path.join(project_dir, p)
    state["bundled_mmaps"] = bundled
    register_recent(path)
    return state


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------
#  Recent files
# ---------------------------------------------------------------------
def load_recent(n: int = 10) -> list[str]:
    if not os.path.exists(RECENT_FILE):
        return []
    try:
        with open(RECENT_FILE) as f:
            return json.load(f)[:n]
    except Exception as e:
        log.warning("could not read recent files: %s", e)
        return []


def register_recent(path: str, n: int = 10) -> None:
    os.makedirs(os.path.dirname(RECENT_FILE), exist_ok=True)
    recent = load_recent(n * 2)
    if path in recent:
        recent.remove(path)
    recent.insert(0, path)
    recent = recent[:n]
    try:
        with open(RECENT_FILE, "w") as f:
            json.dump(recent, f, indent=2)
    except Exception as e:
        log.warning("could not save recent files: %s", e)
