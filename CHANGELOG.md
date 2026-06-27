# Changelog

All notable changes to Turbulence Realm — SINDy are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Visual redesign (v2.2)**: modernized the entire UI aesthetic:
  - New deeper dark palette with cyan (#22D3EE) → indigo (#818CF8) gradient accents
  - Glassmorphism-style surfaces with subtle translucency on cards and group boxes
  - Gradient-filled primary buttons (cyan→indigo) with smooth hover transitions
  - Gradient progress bar chunks and slider sub-pages
  - Gradient-filled checkbox/radio indicators when checked
  - Modernized navigation rail with vertical gradient background
  - Pipeline stepper with bold glowing dots and color-coded connector bars
  - Slimmer, rounded scrollbars with hover states
  - Custom combo box dropdown arrows (no platform-dependent icons)
  - Refined splash screen with larger glow halo and accent bottom line
  - Redesigned About dialog with larger branding and better spacing
  - Updated light theme palette to match the new accent system

### Added
- LICENSE (MIT) so the project is clearly redistributable.
- `pyproject.toml` with `[project]` metadata, console scripts and optional
  dependency groups (`ml`, `export`, `kriging`, `dev`).
- `CITATION.cff` and `CITATION.bib` for academic attribution.
- `tests/` package with pytest covering analysis, quality, SINDy, project
  save/load and ML model export round-trip.
- CI workflow (`.github/workflows/ci.yml`) running ruff, py_compile and pytest
  on push and pull request.
- `.pre-commit-config.yaml` with ruff hooks.
- `logging` throughout the package; GUI now bridges log records into the
  in-app log widget via a `logging.Handler`.
- Provenance metadata in exports: git commit, package versions, full config,
  random seed and input-file SHA-256.
- Reproducible-mode toggle that seeds `numpy`, `random`, `torch` and enables
  `torch.use_deterministic_algorithms`.
- Settings dialog exposing mmap directory, default backend, theme, recent-file
  count and FFmpeg path (persisted via `QSettings`).
- Auto-save / session restore on clean exit and startup.
- Cancel button + ETA display for long-running optical-flow / SINDy / ML jobs.
- Unified error handling: long-running handlers surface failures via
  `QMessageBox` with copyable details instead of only the log widget.
- Energy cascade and structure-function scaling exponents in `analysis.py`.
- `__version__` is now read from package metadata and shown in the splash
  screen, CLI `--version` and the About dialog.

### Changed
- `gui.py` split into `gui/` package: `main_window.py`, `pages/`,
  `workers.py`, `logging_bridge.py`.
- `TR-SINDY-Final.py` moved to `archive/` (v1.0 retained in git history).
- `dist/` removed from git tracking (Nuitka build output).

## [2.1.0] - 2024-06-26

### Added
- ML model export to PyTorch `.pt`, TorchScript and ONNX (best-effort, with
  pickle fallback for non-torch models).
- Per-model hyperparameter controls in the ML tab (learning rate, kernel
  size, SINDy threshold).
- Shared Compute controls (device selector + random seed) for ML models.
- `set_seed()` helper for reproducible ML training.
- `onnx` optional dependency.

### Fixed
- Export-tab buttons now enable after processing completes (previously
  stayed disabled).
- Animated heatmap now persists its `FuncAnimation` reference so it actually
  animates; consistent `vmin`/`vmax` across frames.
- `FNO2D`, `AutoencoderSINDy`, `ConvLSTMSeq`, `FlowVAE` now honour the
  passed `lr` instead of a hardcoded `1e-3`.
- PINN input tensors now follow the selected device (previously mismatched
  on CUDA).

## [2.0.0] - 2024-06-26

### Added
- Refactor of the original single-file `TR-SINDY-Final.py` into a modular
  package (`src/tr_sindy_app/`).
- Alternative optical-flow backends: Lucas-Kanade, TV-L1, RAFT/PWC-Net.
- Multiple SINDy feature libraries and optimizers, k-fold cross-validation,
  model comparison, divergence-free option, time-delay embedding.
- Machine-learning models: PINN, Autoencoder-SINDy, FNO, DeepONet, ConvLSTM,
  VAE/beta-VAE, ensemble & MC-dropout uncertainty, GAN, Granger causality.
- Analysis: vorticity, strain, divergence, spatial/temporal spectra, POD,
  DMD, turbulence statistics, structure functions, velocity PDFs, error
  metrics, region statistics.
- Data quality: outlier detection, RBF/kriging interpolation, noise
  estimation, forward-backward consistency.
- Export: CSV, HDF5, NetCDF, Parquet, JSON metadata, image sequences, PDF
  report, MP4 animation.
- Project system (`.trsindy`), parameter presets, recent files, processing
  history.
- Dark/light theme, keyboard shortcuts, tooltips, ROI undo/redo.
- CLI batch mode (`python -m tr_sindy_app.cli`).

## [1.0.0] - 2024-06-26

### Added
- Initial single-file implementation (`TR-SINDY-Final.py`): Farneback optical
  flow, polynomial SINDy, quiver/contour/streamline visualisation, CSV/HDF5
  export, PyQt6 GUI.
