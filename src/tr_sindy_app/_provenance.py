"""Provenance collection for reproducible scientific exports.

Captures the full execution context — software versions, git commit,
random seed, input file hash, and configuration — so that any exported
result can be traced back to the exact code and environment that produced
it.

Usage::

    from tr_sindy_app._provenance import collect_provenance
    provenance = collect_provenance(
        input_file="video.mp4",
        seed=42,
        config={"backend": "farneback", "library": "polynomial"},
    )
    metadata["provenance"] = provenance
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys


def _git_commit() -> str:
    """Return the current git commit hash, or 'unknown' if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _git_dirty() -> bool:
    """Return True if the git working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


def _package_versions() -> dict:
    """Return versions of key scientific packages."""
    versions = {}
    packages = [
        "numpy", "scipy", "pandas", "matplotlib", "cv2",
        "pysindy", "h5py", "pyarrow", "torch", "torchvision",
        "onnx", "PyQt6",
    ]
    from importlib.metadata import PackageNotFoundError, version
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
        except Exception:
            pass
    return versions


def _file_sha256(path: str, chunk_size: int = 65536) -> str | None:
    """Compute SHA-256 hash of a file, or None if the file doesn't exist."""
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def collect_provenance(
    input_file: str | None = None,
    seed: int | None = None,
    config: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """Collect full provenance information for an export.

    Parameters
    ----------
    input_file : path to the input video/data file (hashed for integrity).
    seed : random seed used for this run (if any).
    config : the full configuration dict (FlowConfig, SINDyConfig, ML hyperparams).
    extra : any additional metadata to include.

    Returns a dict suitable for embedding in export metadata.
    """
    from . import __version__ as _app_version

    provenance = {
        "app_version": _app_version,
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "packages": _package_versions(),
        "timestamp": _iso_timestamp(),
    }
    if seed is not None:
        provenance["random_seed"] = seed
    if input_file:
        provenance["input_file"] = os.path.basename(input_file)
        provenance["input_sha256"] = _file_sha256(input_file)
    if config:
        provenance["config"] = config
    if extra:
        provenance["extra"] = extra
    return provenance


def _iso_timestamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
