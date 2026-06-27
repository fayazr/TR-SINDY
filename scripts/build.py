#!/usr/bin/env python3
"""Build script for Turbulence Realm — SINDy executable.

Creates a standalone executable using PyInstaller (default) or Nuitka.

Usage::

    # Default: one-folder build without ML
    python scripts/build.py

    # Include ML models (torch, torchvision, onnx) — larger bundle
    python scripts/build.py --with-ml

    # One-file build (single executable, slower startup)
    python scripts/build.py --onefile

    # Use Nuitka instead of PyInstaller
    python scripts/build.py --nuitka

    # Clean previous builds first
    python scripts/build.py --clean
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(ROOT, "build")
DIST_DIR = os.path.join(ROOT, "dist")
SPEC_FILE = os.path.join(ROOT, "TurbulenceRealmSINDy.spec")

# Modules to exclude from the bundle (they bloat the build significantly)
EXCLUDES_BASE = [
    "PyQt5", "PySide6", "PySide2",
    "tensorflow", "tensorflow_intel", "keras", "tensorboard",
    "cupy", "jax", "jaxlib",
    "grpc", "google", "google.api_core", "google.protobuf",
    "IPython", "ipykernel", "jupyter", "jupyter_client", "notebook",
    "nbconvert", "nbformat", "qtconsole", "jedi", "numba", "llvmlite",
    "sympy", "numexpr", "tables", "bokeh", "plotly", "dask",
    "sphinx", "pytest", "wx", "tkinter", "test",
]

# Hidden imports that PyInstaller can't auto-detect
HIDDEN_IMPORTS = [
    "sklearn.linear_model",
    "sklearn.utils._typedefs",
    "scipy.ndimage",
    "matplotlib.backends.backend_qtagg",
    "tr_sindy_app._logging",
    "tr_sindy_app._provenance",
    "tr_sindy_app.settings_dialog",
]

# ML hidden imports (only included with --with-ml)
HIDDEN_IMPORTS_ML = [
    "torch", "torchvision", "torch.nn", "torch.optim",
    "onnx", "onnxruntime",
]


def clean():
    """Remove build/ and dist/ directories."""
    for d in (BUILD_DIR, DIST_DIR):
        if os.path.exists(d):
            print(f"Cleaning {d} ...")
            shutil.rmtree(d)


def collect_data_files():
    """Collect data files to bundle (logo, config, ffmpeg)."""
    datas = []
    for fname in ("logo.png", "logo.ico", "config.json"):
        fpath = os.path.join(ROOT, fname)
        if os.path.exists(fpath):
            datas.append((fpath, "."))
    # FFmpeg
    for fname in ("ffmpeg.exe", "ffmpeg"):
        fpath = os.path.join(ROOT, fname)
        if os.path.exists(fpath):
            datas.append((fpath, "."))
            break
    return datas


def build_pyinstaller(onefile: bool = False, with_ml: bool = False):
    """Build with PyInstaller."""
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    excludes = list(EXCLUDES_BASE)
    if not with_ml:
        excludes.extend(["torch", "torchvision", "torchaudio", "onnx", "onnxruntime"])

    hiddenimports = list(HIDDEN_IMPORTS)
    if with_ml:
        hiddenimports.extend(HIDDEN_IMPORTS_ML)

    datas = collect_data_files()

    # Build the spec content
    spec_content = _generate_spec(onefile, excludes, hiddenimports, datas)
    with open(SPEC_FILE, "w") as f:
        f.write(spec_content)
    print(f"Generated {SPEC_FILE}")

    cmd = [sys.executable, "-m", "PyInstaller", SPEC_FILE, "--noconfirm"]
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT)

    output_dir = os.path.join(DIST_DIR, "TurbulenceRealmSINDy")
    print(f"\nBuild complete! Output: {output_dir}")
    if onefile:
        exe = os.path.join(DIST_DIR, "TurbulenceRealmSINDy.exe")
        if os.path.exists(exe):
            print(f"Executable: {exe}")
    else:
        exe = os.path.join(output_dir, "TurbulenceRealmSINDy.exe")
        if os.path.exists(exe):
            print(f"Executable: {exe}")


def build_nuitka(with_ml: bool = False):
    """Build with Nuitka."""
    try:
        import nuitka  # noqa: F401
    except ImportError:
        print("Nuitka not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nuitka"])

    cmd = [
        sys.executable, "-m", "nuitka",
        "--onefile",
        "--windows-console-mode=disable",
        f"--output-dir={BUILD_DIR}",
        "--enable-plugin=pyqt6",
        "--include-package=cv2",
        "--include-package=pysindy",
        "--include-package=pandas",
        "--include-package=matplotlib",
        "--include-package=scipy",
        "--include-package=sklearn",
        os.path.join(ROOT, "run.py"),
    ]

    # Data files
    for fname in ("logo.png", "logo.ico", "config.json"):
        fpath = os.path.join(ROOT, fname)
        if os.path.exists(fpath):
            cmd.append(f"--include-data-file={fpath}={fname}")

    if with_ml:
        cmd.extend([
            "--include-package=torch",
            "--include-package=torchvision",
            "--include-package=onnx",
        ])
    else:
        cmd.extend([
            "--nofollow-import-to=torch",
            "--nofollow-import-to=torchvision",
            "--nofollow-import-to=onnx",
        ])

    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT)
    print(f"\nBuild complete! Output in {BUILD_DIR}/")


def _generate_spec(onefile, excludes, hiddenimports, datas):
    """Generate PyInstaller .spec content."""
    excludes_str = ", ".join(f"'{e}'" for e in excludes)
    hidden_str = ", ".join(f"'{h}'" for h in hiddenimports)
    datas_str = ", ".join(f"('{src}', '{dst}')" for src, dst in datas)

    if onefile:
        return f"""# -*- mode: python ; coding: utf-8 -*-
# Auto-generated by scripts/build.py — do not edit manually.
# One-file build: {'with ML' if 'torch' not in excludes else 'without ML'}

block_cipher = None

a = Analysis(
    ['run.py'],
    pathex=['src'],
    binaries=[],
    datas=[{datas_str}],
    hiddenimports=[{hidden_str}],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[{excludes_str}],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='TurbulenceRealmSINDy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
"""
    else:
        return f"""# -*- mode: python ; coding: utf-8 -*-
# Auto-generated by scripts/build.py — do not edit manually.
# One-folder build: {'with ML' if 'torch' not in excludes else 'without ML'}

block_cipher = None

a = Analysis(
    ['run.py'],
    pathex=['src'],
    binaries=[],
    datas=[{datas_str}],
    hiddenimports=[{hidden_str}],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[{excludes_str}],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TurbulenceRealmSINDy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='TurbulenceRealmSINDy',
)
"""


def main():
    parser = argparse.ArgumentParser(
        description="Build TR-SINDY executable")
    parser.add_argument("--with-ml", action="store_true",
                        help="Include ML models (torch, torchvision, onnx)")
    parser.add_argument("--onefile", action="store_true",
                        help="Single-file executable (slower startup)")
    parser.add_argument("--nuitka", action="store_true",
                        help="Use Nuitka instead of PyInstaller")
    parser.add_argument("--clean", action="store_true",
                        help="Clean build/ and dist/ before building")
    args = parser.parse_args()

    if args.clean:
        clean()

    if args.nuitka:
        build_nuitka(with_ml=args.with_ml)
    else:
        build_pyinstaller(onefile=args.onefile, with_ml=args.with_ml)


if __name__ == "__main__":
    main()
