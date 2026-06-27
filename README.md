# Turbulence Realm — SINDy

<p align="center">
  <strong>Video-based fluid-flow analysis with Sparse Identification of Nonlinear Dynamics</strong>
</p>

<p align="center">
  <a href="https://github.com/fayazr/TR-SINDY/actions"><img src="https://github.com/fayazr/TR-SINDY/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.turbulencerealm.com"><img src="https://img.shields.io/badge/www-turbulencerealm.com-22D3EE.svg" alt="Website"></a>
</p>

---

A futuristic desktop application for **video-based fluid-flow analysis**. It
extracts a dense velocity field from a video using optical flow, fits a
**SINDy** (Sparse Identification of Nonlinear Dynamics) model, predicts the
reconstructed field, and provides rich visualisation, advanced analysis,
machine-learning models and flexible export.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Using the GUI](#using-the-gui)
- [Using the CLI](#using-the-cli)
- [Building an Executable](#building-an-executable)
- [Project Structure](#project-structure)
- [Testing & Development](#testing--development)
- [Citation](#citation)
- [Author](#author)

## Features

- **Optical Flow**: Farneback (default), Lucas-Kanade, TV-L1, RAFT/PWC-Net
  (deep learning), multi-scale pyramid, temporal smoothing, quality metrics.
- **SINDy Modeling**: polynomial / Fourier / combined / custom libraries;
  STLSQ / SR3 / FROLS / constrained optimizers; k-fold cross-validation;
  model comparison; divergence-free option; time-delay embedding.
- **Prediction**: batched SINDy prediction and smoothed field reconstruction.
- **Analysis**: vorticity, strain rate, divergence, spatial/temporal spectra,
  POD, DMD, turbulence statistics (TKE, Reynolds number), structure functions
  with scaling exponents, isotropic energy spectrum with Kolmogorov fit,
  velocity PDFs, comprehensive error metrics, region statistics.
- **Machine Learning** (optional, requires torch): PINN for Navier-Stokes,
  Autoencoder-SINDy, Fourier Neural Operator, DeepONet, ConvLSTM, VAE/beta-VAE,
  ensemble & MC-dropout uncertainty, GAN synthesis, Granger causality.
- **Data Quality**: outlier detection (z-score / modified z-score / IQR),
  RBF/kriging interpolation, noise estimation, forward-backward consistency.
- **Visualisation**: side-by-side actual vs SINDy quiver, contour, streamlines,
  vorticity/strain fields, POD/DMD modes, spectra, animated heatmaps, custom
  colormaps, data table view.
- **Export**: CSV, HDF5, NetCDF, Parquet, JSON metadata (with full provenance),
  image sequences, PDF report, MP4 quiver animation.
- **Reproducibility**: provenance metadata (git commit, package versions,
  config, seed, input SHA-256) embedded in every export; reproducible mode
  seeds numpy/random/torch with deterministic algorithms.
- **Project System**: save/load `.trsindy` projects (bundles memmaps),
  parameter presets, recent files, processing history, auto-save/session
  restore.
- **UX**: futuristic dark/light theme with gradient accents, glassmorphism
  surfaces, settings dialog, progress ETA, keyboard shortcuts, tooltips,
  ROI undo/redo, in-app log widget.

## Installation

### From source (recommended)

```bash
git clone https://github.com/fayazr/TR-SINDY.git
cd TR-SINDY
pip install -e ".[ml,export]"
```

### Minimal install (no ML / advanced export)

```bash
pip install -e .
```

### Optional dependency groups

| Group | Packages | Purpose |
|-------|----------|---------|
| `ml` | torch, torchvision, onnx | PINN, FNO, DeepONet, ConvLSTM, VAE, GAN, RAFT |
| `export` | h5py, pyarrow | HDF5 and Parquet export |
| `tvl1` | opencv-contrib-python | TV-L1 optical flow backend |
| `kriging` | pykrige | Kriging interpolation |
| `dev` | pytest, ruff, pre-commit | Development and testing |

> All optional features degrade gracefully: the app runs without them and
> shows a clear message when a feature needs an optional dependency.

### System requirements

- Python 3.9+
- FFmpeg (only required for MP4 animation export)
- A display (GUI mode) or headless (CLI mode)

## Quick Start

```bash
# Launch the GUI
tr-sindy
# or
python run.py

# Process a video from the command line
tr-sindy-cli process video.mp4 \
    --roi 600,300,1300,750 --calib-px 200 --calib-m 0.1 \
    --backend farneback --library polynomial --degree 3 --threshold 0.07 \
    --export-dir ./out --formats csv,hdf5,pdf
```

## Using the GUI

The GUI has four main pages, accessible from the navigation rail on the left:

### 1. Setup Page

This is where you load a video, calibrate the scale, configure parameters, and
run the processing pipeline.

**Step-by-step workflow:**

1. **Open a video** — Click **Browse…** in the *Video Source* card or use
   **File → Open Video** (Ctrl+O). Supported formats: MP4, AVI, MOV, MKV
   (anything OpenCV can decode).

2. **Select ROI & Calibrate** — Click **Select ROI & Calibrate**. A dialog
   opens showing the first frame:
   - **Draw a rectangle** over the region of interest (the area where you want
     to measure flow).
   - **Draw a calibration line** of known physical length. Enter the real-world
     length in meters.
   - Use **Ctrl+Z / Ctrl+Y** to undo/redo if you make a mistake.
   - Click **OK** when done. The calibration readout updates to show
     meters-per-pixel.

3. **Configure Optical Flow** — Choose a backend and options:
   - **Backend**: `farneback` (default, robust), `lucas_kanade` (sparse,
     fast), `tvl1` (needs opencv-contrib), `raft` / `pwcnet` (deep learning,
     needs torch).
   - **Smoothing**: `none`, `ema` (exponential moving average), `moving`
     (moving average) — reduces flicker between frames.
   - **EMA α**: Smoothing factor (0–1). Higher = more responsive, lower =
     smoother.
   - **Multi-scale pyramid**: Coarse-to-fine estimation for large motions.
   - **Gaussian / NLM denoise**: Pre-denoise frames for cleaner flow.
   - **Quality metrics**: Compute forward-backward consistency error per frame.

4. **Configure SINDy** — Choose library and optimizer:
   - **Library**: `polynomial`, `fourier`, `combined` (poly × Fourier),
     `custom`, `trig`.
   - **Degree**: Polynomial degree (1–5).
   - **Optimizer**: `stlsq` (default), `sr3`, `frols`, `constrained_sr3`.
   - **Threshold**: Sparsity threshold (0.001–1.0). Higher = sparser model.
   - **Divergence-free**: Enforce incompressibility (∇·u = 0).

5. **Run the pipeline** — Click the buttons in order:
   - **① Process Optical Flow** — Extracts the velocity field. A live HSV +
     quiver preview appears. Progress and ETA are shown in the status bar.
   - **② Run SINDy Modeling** — Fits the sparse dynamics model. Optionally
     use **Equation** to view the discovered equation, **Cross-validate** for
     k-fold CV, or **Compare** to try multiple libraries.
   - **③ Run SINDy Prediction** — Reconstructs the predicted velocity field.

6. **Apply a preset** (optional) — Use **Presets → List / Apply Preset…** to
   load saved parameter sets, or **Save Current as Preset…** to save your
   current configuration.

### 2. Visualization Page

After running the pipeline, visualize the results:

- **Quiver plot** — Side-by-side actual vs SINDy-predicted velocity arrows.
- **Contour / streamlines** — Velocity magnitude contours or streamline plots.
- **Vorticity / strain** — Color-mapped vorticity and strain-rate fields.
- **Animated heatmap** — Play through frames as an animated velocity heatmap.
- **Custom colormap** — Choose from matplotlib colormaps.
- **Data table** — Tabular view of the velocity data.

### 3. ML Models Page

Train machine-learning models (requires `pip install -e ".[ml]"`):

- **PINN** — Physics-Informed Neural Network with Navier-Stokes residuals.
- **Autoencoder-SINDy** — Compressed latent-space SINDy.
- **FNO** — Fourier Neural Operator for flow prediction.
- **DeepONet** — Deep Operator Network.
- **ConvLSTM** — Recurrent convolutional model for temporal sequences.
- **VAE / beta-VAE** — Variational Autoencoder for flow generation.
- **Ensemble** — SINDy ensemble with uncertainty quantification.
- **GAN** — Generative Adversarial Network for flow synthesis.
- **Causal** — Granger causality analysis.

Select a model, adjust hyperparameters in the parameter panel, and click
**Train**. Training logs appear in the in-app log widget. Trained models can
be exported to TorchScript / ONNX.

### 4. Export Page

Export results in multiple formats:

- **CSV** — Velocity field as comma-separated values.
- **HDF5** — Hierarchical data format (needs `h5py`).
- **NetCDF** — Network Common Data Form.
- **Parquet** — Columnar storage (needs `pyarrow`).
- **JSON metadata** — Full metadata with provenance (git commit, versions,
  config, seed, SHA-256).
- **Image sequence** — PNG frames of the visualization.
- **PDF report** — Comprehensive PDF with metrics and figures.
- **MP4 animation** — Quiver animation video (needs FFmpeg).

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+O | Open video |
| Ctrl+S | Save project |
| Ctrl+L | Load project |
| Ctrl+T | Toggle dark/light theme |
| Ctrl+Q | Quit |
| F1 | Workflow guide |
| Esc | Close window |

### Settings

**Edit → Settings…** opens the settings dialog:

- **Mmap directory** — Where velocity memmaps are stored.
- **Default backend** — Default optical-flow backend.
- **Theme** — Dark or light.
- **Recent files count** — Number of recent projects to remember.
- **FFmpeg path** — Custom FFmpeg executable path (leave empty for auto-detect).
- **Auto-save** — Enable session restore on next launch.

Settings are persisted via QSettings and survive restarts.

## Using the CLI

The CLI supports batch processing and preset management:

### Process a single video

```bash
tr-sindy-cli process video.mp4 \
    --roi 600,300,1300,750 \
    --calib-px 200 --calib-m 0.1 \
    --backend farneback \
    --library polynomial --degree 3 --threshold 0.07 \
    --export-dir ./out \
    --formats csv,hdf5,pdf
```

**Arguments:**

| Flag | Description | Default |
|------|-------------|---------|
| `--roi` | ROI as `x0,y0,x1,y1` (pixels) | required |
| `--calib-px` | Calibration line length in pixels | required |
| `--calib-m` | Calibration line length in meters | required |
| `--backend` | Optical-flow backend | `farneback` |
| `--library` | SINDy library | `polynomial` |
| `--degree` | Polynomial degree | `3` |
| `--threshold` | Sparsity threshold | `0.07` |
| `--optimizer` | SINDy optimizer | `stlsq` |
| `--export-dir` | Output directory | `./out` |
| `--formats` | Export formats (comma-separated) | `csv,json` |
| `--seed` | Random seed for reproducibility | none |
| `--verbose` | Enable verbose logging | off |

### Batch processing

Create a JSON file with multiple jobs:

```json
[
  {
    "video": "experiment1.mp4",
    "roi": [600, 300, 1300, 750],
    "calib_px": 200,
    "calib_m": 0.1,
    "backend": "farneback",
    "library": "polynomial",
    "degree": 3,
    "threshold": 0.07,
    "export_dir": "./out/exp1",
    "formats": ["csv", "hdf5", "pdf"]
  },
  {
    "video": "experiment2.mp4",
    "roi": [100, 50, 800, 600],
    "calib_px": 150,
    "calib_m": 0.05,
    "backend": "lucas_kanade",
    "library": "fourier",
    "degree": 2,
    "threshold": 0.05,
    "export_dir": "./out/exp2",
    "formats": ["csv", "json"]
  }
]
```

Run with:

```bash
tr-sindy-cli batch jobs.json
```

### Preset management

```bash
# List available presets
tr-sindy-cli presets list

# Apply a preset to a video
tr-sindy-cli presets apply farneback-default video.mp4 \
    --roi 600,300,1300,750 --calib-px 200 --calib-m 0.1 \
    --export-dir ./out
```

### Other CLI commands

```bash
# Show version
tr-sindy-cli --version

# Show help
tr-sindy-cli --help
```

## Building an Executable

TR-SINDY can be packaged as a standalone executable using PyInstaller. This
creates a distributable that doesn't require Python to be installed.

### Prerequisites

```bash
pip install pyinstaller
pip install -e ".[ml,export]"  # install all optional deps you want bundled
```

### Build with PyInstaller

```bash
# One-folder build (faster startup, easier to debug)
pyinstaller TurbulenceRealmSINDy.spec

# Or use the build script
python scripts/build.py
```

The executable will be in `dist/TurbulenceRealmSINDy/`.

### Build script

A convenience build script is provided:

```bash
# Default build (one-folder, no ML)
python scripts/build.py

# Include ML models (larger bundle, ~2GB+)
python scripts/build.py --with-ml

# One-file build (single .exe, slower startup)
python scripts/build.py --onefile
```

### Custom icon

Place `logo.ico` (Windows) or `logo.png` in the project root before building
to set the executable icon.

### Bundling FFmpeg

To bundle FFmpeg for animation export, place `ffmpeg.exe` (Windows) or
`ffmpeg` (Linux/macOS) in the project root. The build script will include it
automatically.

### Nuitka alternative

For a potentially faster binary, Nuitka is also supported:

```bash
pip install nuitka

python scripts/build.py --nuitka
```

## Project Structure

```
TR-SINDY/
├── src/
│   └── tr_sindy_app/           # v2.2 modular package
│       ├── __init__.py         # version, public API
│       ├── app.py              # entry point + splash screen
│       ├── gui.py              # main window + pages
│       ├── theme.py            # dark/light theme, stylesheet
│       ├── roi_dialog.py       # ROI/calibration with undo/redo
│       ├── optical_flow.py     # Farneback/LK/TV-L1/RAFT + multiscale
│       ├── sindy_core.py       # libraries, optimizers, CV, comparison
│       ├── ml_models.py        # PINN, AE-SINDy, FNO, DeepONet, ConvLSTM...
│       ├── analysis.py         # vorticity, POD, DMD, spectra, turbulence
│       ├── quality.py          # outliers, interpolation, noise
│       ├── export.py           # CSV/HDF5/NetCDF/Parquet/PDF/animation
│       ├── project.py          # save/load, presets, history, recent
│       ├── cli.py              # command-line batch mode
│       ├── settings_dialog.py  # settings dialog (QSettings)
│       ├── _logging.py         # logging config + Qt log handler
│       └── _provenance.py      # reproducibility metadata collection
├── tests/                      # pytest test suite (72 tests)
├── scripts/
│   └── build.py                # executable build script
├── archive/
│   └── TR-SINDY-Final.py       # original v1.0 reference
├── .github/workflows/ci.yml    # CI: ruff + pytest + GUI smoke test
├── run.py                      # launcher script
├── pyproject.toml              # package metadata + tool config
├── TurbulenceRealmSINDy.spec   # PyInstaller spec
├── requirements.txt
├── ROADMAP.md                  # full feature status matrix
├── CHANGELOG.md                # version history
├── CITATION.cff                # citation metadata
├── CITATION.bib                # BibTeX citation
└── LICENSE                     # MIT
```

## Testing & Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/

# Format check
ruff format --check src/ tests/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

CI runs automatically on push and pull requests via GitHub Actions
(`.github/workflows/ci.yml`), covering ruff linting, pytest on Python 3.10/3.11/3.12,
and a GUI smoke test.

## Citation

If you use TR-SINDY in your research, please cite it:

```bibtex
@software{rasheed_trsindy,
  author = {Rasheed, Fayaz},
  title = {{Turbulence Realm — SINDy}: Video-based Fluid Flow Analysis},
  year = {2024},
  url = {https://github.com/fayazr/TR-SINDY},
  license = {MIT}
}
```

See `CITATION.cff` and `CITATION.bib` for more details.

## Author

Developed by **Fayaz Rasheed** — [www.turbulencerealm.com](https://www.turbulencerealm.com)

## License

[MIT](LICENSE) — Copyright (c) 2024 Fayaz Rasheed
