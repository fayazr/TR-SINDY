# Turbulence Realm — SINDy

<p align="center">
  <img src="logo.png" width="80" height="80" alt="Turbulence Realm logo">
</p>

<p align="center">
  <strong>Video-based fluid-flow analysis with Sparse Identification of Nonlinear Dynamics</strong>
</p>

<p align="center">
  <a href="https://github.com/fayazr/TR-SINDY/actions"><img src="https://github.com/fayazr/TR-SINDY/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.turbulencerealm.com"><img src="https://img.shields.io/badge/www-turbulencerealm.com-a67c2a.svg" alt="Website"></a>
  <a href="https://github.com/fayazr/TR-SINDY/releases"><img src="https://img.shields.io/badge/platform-Linux%20%7C%20Windows-a67c2a.svg" alt="Platforms"></a>
</p>

---

A desktop application for **video-based fluid-flow analysis**. It
extracts a dense velocity field from a video using optical flow, fits a
**SINDy** (Sparse Identification of Nonlinear Dynamics) model, predicts the
reconstructed field, and provides rich visualisation, advanced analysis,
machine-learning models and flexible export.

The app features a gold/cream Turbulence Realm brand identity matching the
[product website](https://www.turbulencerealm.com).

## Table of Contents

- [Features](#features)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Using the GUI](#using-the-desktop-gui)
  - [Setup Page](#1-setup-page)
  - [Visualization Page](#2-visualization-page)
  - [ML Models Page](#3-ml-models-page)
  - [Export Page](#4-export-page)
  - [Analysis Tools](#analysis-tools)
  - [Keyboard Shortcuts](#keyboard-shortcuts)
  - [Settings](#settings)
- [Using the CLI](#using-the-cli)
- [Building from Source](#building-from-source)
  - [Desktop Executable](#building-a-desktop-executable)
- [Project Structure](#project-structure)
- [Testing & Development](#testing--development)
- [Citation](#citation)
- [Author](#author)

## Features

### Optical Flow
- **Backends**: Farneback (default, robust dense), Lucas-Kanade (sparse, fast),
  TV-L1 (needs opencv-contrib), RAFT / PWC-Net (deep learning, needs torch).
- **Multi-scale pyramid**: Coarse-to-fine estimation for large motions.
- **Temporal smoothing**: EMA (exponential moving average) or moving average
  to reduce flicker between frames.
- **Denoising**: Gaussian or Non-Local Means pre-denoise for cleaner flow.
- **Quality metrics**: Forward-backward consistency error computed per frame.

### SINDy Modeling
- **Libraries**: polynomial, Fourier, combined (poly × Fourier), custom, trig.
- **Optimizers**: STLSQ (default), SR3, FROLS, constrained SR3.
- **Degree**: Polynomial degree 1–5.
- **Threshold**: Sparsity threshold (0.001–1.0) — higher = sparser model.
- **Divergence-free**: Enforce incompressibility (∇·u = 0).
- **Cross-validation**: k-fold CV with error reporting.
- **Model comparison**: Try multiple libraries/optimizers side-by-side.
- **Time-delay embedding**: For systems with hidden variables.

### Prediction
- Batched SINDy prediction and smoothed field reconstruction.
- Side-by-side actual vs predicted velocity comparison.

### Analysis
- **Vorticity**: Color-mapped vorticity field (ω = ∂v/∂x − ∂u/∂y).
- **Strain rate**: Symmetric and antisymmetric strain components.
- **Divergence**: ∇·u field for compressibility assessment.
- **Spectra**: Spatial and temporal energy spectra.
- **POD**: Proper Orthogonal Decomposition modes and energies.
- **DMD**: Dynamic Mode Decomposition modes and growth rates.
- **Turbulence statistics**: TKE, Reynolds number, structure functions with
  scaling exponents, isotropic energy spectrum with Kolmogorov fit.
- **Velocity PDFs**: Probability density functions of velocity components.
- **Error metrics**: RMSE, MAE, R², relative error between actual and predicted.
- **Region statistics**: Spatially-averaged flow quantities.

### Machine Learning (optional, requires torch)
- **PINN** — Physics-Informed Neural Network with Navier-Stokes residuals.
- **Autoencoder-SINDy** — Compressed latent-space SINDy.
- **FNO** — Fourier Neural Operator for flow prediction.
- **DeepONet** — Deep Operator Network.
- **ConvLSTM** — Recurrent convolutional model for temporal sequences.
- **VAE / beta-VAE** — Variational Autoencoder for flow generation.
- **Ensemble** — SINDy ensemble with uncertainty quantification.
- **GAN** — Generative Adversarial Network for flow synthesis.
- **Granger causality** — Causal analysis between flow variables.

### Data Quality
- **Outlier detection**: z-score, modified z-score, IQR methods.
- **Interpolation**: RBF (radial basis function), kriging (needs pykrige).
- **Noise estimation**: Estimated noise level in the velocity field.
- **Forward-backward consistency**: Detects unreliable flow regions.

### Visualisation
- Side-by-side actual vs SINDy quiver plots.
- Velocity magnitude contours and streamline plots.
- Vorticity and strain-rate color-mapped fields.
- POD/DMD mode visualizations.
- Energy spectra plots.
- Animated heatmaps (play through frames).
- Custom matplotlib colormaps.
- Data table view (tabular velocity data).

### Export
- **CSV** — Velocity field as comma-separated values.
- **HDF5** — Hierarchical data format (needs `h5py`).
- **NetCDF** — Network Common Data Form.
- **Parquet** — Columnar storage (needs `pyarrow`).
- **JSON metadata** — Full metadata with provenance (git commit, versions,
  config, seed, SHA-256).
- **Image sequence** — PNG frames of the visualization.
- **PDF report** — Comprehensive PDF with metrics and figures.
- **MP4 animation** — Quiver animation video (needs FFmpeg).

### Reproducibility
- Provenance metadata (git commit, package versions, config, seed, input
  SHA-256) embedded in every export.
- Reproducible mode seeds numpy/random/torch with deterministic algorithms.

### Project System
- Save/load `.trsindy` projects (bundles memmaps).
- Parameter presets (built-in + user-saved).
- Recent files list.
- Processing history.
- Auto-save and session restore on next launch.

### UX
- Gold/cream Turbulence Realm brand theme (dark and light variants).
- Glassmorphism surfaces with subtle translucency.
- Gradient accents and glowing focus states.
- Settings dialog with persistent preferences.
- Progress ETA in the status bar.
- Keyboard shortcuts and tooltips throughout.
- ROI undo/redo.
- In-app log widget.

## Screenshots

The app features a navigation rail on the left with four main pages
(Setup, Visualize, ML Models, Export), a glassmorphism background with drifting
glow orbs, and gold/cream branding matching the [product website](https://www.turbulencerealm.com).

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

The GUI has four main pages, accessible from the navigation rail on the left.
The brand logo appears at the top of the rail, followed by the app title
"Turbulence Realm" and subtitle "· SINDy".

### 1. Setup Page

This is where you load a video, calibrate the scale, configure parameters, and
run the processing pipeline.

#### Step-by-step workflow

**Step 1 — Open a video**

Click **Browse…** in the *Video Source* card or use **File → Open Video**
(Ctrl+O). Supported formats: MP4, AVI, MOV, MKV (anything OpenCV can decode).

**Step 2 — Select ROI & Calibrate**

Click **Select ROI & Calibrate**. A dialog opens showing the first frame:

- **Draw a rectangle** over the region of interest (the area where you want
  to measure flow).
- **Draw a calibration line** of known physical length. Enter the real-world
  length in meters.
- Use **Ctrl+Z / Ctrl+Y** to undo/redo if you make a mistake.
- Click **OK** when done. The calibration readout updates to show
  meters-per-pixel.

**Step 3 — Configure Optical Flow**

Choose a backend and options in the *Optical Flow Configuration* card:

| Option | Values | Description |
|--------|--------|-------------|
| Backend | `farneback`, `lucas_kanade`, `tvl1`, `raft`, `pwcnet` | Flow algorithm |
| Smoothing | `none`, `ema`, `moving` | Temporal smoothing |
| EMA α | 0–1 | Smoothing factor (higher = more responsive) |
| Multi-scale pyramid | on/off | Coarse-to-fine for large motions |
| Gaussian denoise | on/off | Pre-denoise with Gaussian filter |
| NLM denoise | on/off | Pre-denoise with Non-Local Means |
| Quality metrics | on/off | Forward-backward consistency error |

**Step 4 — Configure SINDy**

Choose library and optimizer in the *SINDy Configuration* card:

| Option | Values | Description |
|--------|--------|-------------|
| Library | `polynomial`, `fourier`, `combined`, `custom`, `trig` | Candidate function library |
| Degree | 1–5 | Polynomial degree |
| Optimizer | `stlsq`, `sr3`, `frols`, `constrained_sr3` | Sparse regression optimizer |
| Threshold | 0.001–1.0 | Sparsity threshold (higher = sparser) |
| Divergence-free | on/off | Enforce ∇·u = 0 |

**Step 5 — Run the pipeline**

Click the buttons in order:

1. **① Process Optical Flow** — Extracts the velocity field. A live HSV +
   quiver preview appears. Progress and ETA are shown in the status bar.
2. **② Run SINDy Modeling** — Fits the sparse dynamics model. The discovered
   equation is displayed. Optionally use **Cross-validate** for k-fold CV, or
   **Compare** to try multiple libraries.
3. **③ Run SINDy Prediction** — Reconstructs the predicted velocity field.

**Step 6 — Apply a preset (optional)**

Use **Presets → List / Apply Preset…** to load saved parameter sets, or
**Save Current as Preset…** to save your current configuration.

### 2. Visualization Page

After running the pipeline, visualize the results. The page has a control
panel on the left and a matplotlib canvas on the right.

- **Quiver plot** — Side-by-side actual vs SINDy-predicted velocity arrows.
  Use the frame slider to scrub through time.
- **Contour / streamlines** — Velocity magnitude contours or streamline plots.
- **Vorticity / strain** — Color-mapped vorticity and strain-rate fields.
- **Animated heatmap** — Play through frames as an animated velocity heatmap.
  Press the play button to animate.
- **Custom colormap** — Choose from matplotlib colormaps (viridis, plasma,
  magma, inferno, cividis, etc.).
- **Data table** — Tabular view of the velocity data (u, v, magnitude per
  grid point).

### 3. ML Models Page

Train machine-learning models (requires `pip install -e ".[ml]"`).

Select a model from the dropdown, adjust hyperparameters in the parameter
panel, and click **Train**. Training logs appear in the in-app log widget.

| Model | Description | Key Hyperparameters |
|-------|-------------|---------------------|
| PINN | Physics-Informed NN with Navier-Stokes residuals | layers, epochs, lr |
| Autoencoder-SINDy | Compressed latent-space SINDy | latent_dim, epochs |
| FNO | Fourier Neural Operator | modes, width, epochs |
| DeepONet | Deep Operator Network | depth, width, epochs |
| ConvLSTM | Recurrent conv model for temporal sequences | hidden_dim, layers |
| VAE / beta-VAE | Variational Autoencoder for flow generation | latent_dim, beta |
| Ensemble | SINDy ensemble with uncertainty quantification | n_models |
| GAN | Generative Adversarial Network for flow synthesis | epochs, lr |
| Causal | Granger causality analysis | max_lag |

Trained models can be exported to TorchScript / ONNX from the Export page.

### 4. Export Page

Export results in multiple formats. Check the formats you want and click
**Export**:

| Format | Needs | Description |
|--------|------|-------------|
| CSV | — | Velocity field as comma-separated values |
| HDF5 | `h5py` | Hierarchical data format |
| NetCDF | — | Network Common Data Form |
| Parquet | `pyarrow` | Columnar storage |
| JSON metadata | — | Full metadata with provenance |
| Image sequence | — | PNG frames of the visualization |
| PDF report | — | Comprehensive PDF with metrics and figures |
| MP4 animation | FFmpeg | Quiver animation video |

All exports include provenance metadata (git commit, package versions, config,
seed, input SHA-256) for full reproducibility.

### Analysis Tools

Accessible from the Visualization page, these tools provide deeper insight:

- **POD (Proper Orthogonal Decomposition)** — Extract dominant coherent
  structures. View mode shapes and energy spectrum.
- **DMD (Dynamic Mode Decomposition)** — Identify spatiotemporal patterns
  with growth rates and frequencies.
- **Spectra** — Spatial and temporal energy spectra with Kolmogorov fit.
- **Turbulence statistics** — TKE, Reynolds number, structure functions,
  scaling exponents.
- **Velocity PDFs** — Probability density functions of velocity components.
- **Error metrics** — RMSE, MAE, R² between actual and predicted fields.

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
- **Theme** — Dark or light (both use the gold/cream brand palette).
- **Recent files count** — Number of recent projects to remember.
- **FFmpeg path** — Custom FFmpeg executable path (leave empty for auto-detect).
- **Auto-save** — Enable session restore on next launch.

Settings are persisted via QSettings and survive restarts.

### About Dialog

**Help → About** shows the Turbulence Realm brand logo, app version, a short
description, and a link to [www.turbulencerealm.com](https://www.turbulencerealm.com).

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

## Building from Source

### Building a Desktop Executable

TR-SINDY can be packaged as a standalone executable using PyInstaller. This
creates a distributable that doesn't require Python to be installed.

#### Prerequisites

```bash
pip install pyinstaller
pip install -e ".[ml,export]"  # install all optional deps you want bundled
```

#### Build with PyInstaller

```bash
# One-folder build (faster startup, easier to debug)
pyinstaller TurbulenceRealmSINDy.spec

# Or use the build script
python scripts/build.py
```

The executable will be in `build/dist/TurbulenceRealmSINDy/`.

#### Build script options

```bash
# Default build (one-folder, no ML)
python scripts/build.py

# Include ML models (larger bundle, ~2GB+)
python scripts/build.py --with-ml

# One-file build (single .exe, slower startup)
python scripts/build.py --onefile
```

#### Custom logo / icon

The `logo.png` file (a gold rounded square with white flowing curve paths —
the Turbulence Realm brand mark) is automatically bundled and used as:
- The window icon (title bar / taskbar)
- The navigation rail brand mark (top-left corner)
- The About dialog logo

To use a custom icon, replace `logo.png` in the project root before building.

#### Bundling FFmpeg

To bundle FFmpeg for animation export, place `ffmpeg` in the project root.
The build script will include it automatically.

#### Nuitka alternative

For a potentially faster binary, Nuitka is also supported:

```bash
pip install nuitka
python scripts/build.py --nuitka
```

### Building a Windows Installer (Inno Setup)

To create a professional Windows setup wizard (with disclaimer, install
location selection, Start Menu shortcuts, and uninstaller):

#### Prerequisites

- Windows (or use the GitHub Actions CI workflow which does this automatically)
- [Inno Setup](https://jrsoftware.org/isdl.php) installed
- A PyInstaller build already in `build/dist/TurbulenceRealmSINDy/`

#### Build the installer

```bash
# First, build the executable with PyInstaller
pyinstaller TurbulenceRealmSINDy.spec

# Then, compile the installer
iscc TurbulenceRealmSINDy.iss
```

The installer will be in `installer/TurbulenceRealmSINDy-2.2.0-Setup.exe`.

#### Installer features

- **Disclaimer page**: Shows the no-liability disclaimer before installation
- **Install location**: User-selectable (defaults to Program Files)
- **Start Menu shortcut**: Creates a program group with app + uninstall links
- **Optional desktop icon**: User can opt in during install
- **Add/Remove Programs**: Registered in Windows Programs and Features
- **Uninstaller**: Fully removes the app and all files
- **Launch on finish**: Option to start the app after installation

The installer configuration is in `TurbulenceRealmSINDy.iss` and the
disclaimer text is in `DISCLAIMER.txt`.

### Building a Linux Installer (Makeself)

To create a self-extracting interactive installer for Linux (with disclaimer,
install location selection, desktop/menu shortcuts, and uninstaller):

#### Prerequisites

- Linux (or use the GitHub Actions CI workflow which does this automatically)
- `makeself` installed (`sudo apt install makeself`)
- A PyInstaller build already in `build/dist/TurbulenceRealmSINDy/`

#### Build the installer

```bash
# First, build the executable with PyInstaller
pyinstaller TurbulenceRealmSINDy.spec

# Then, create the self-extracting installer
./scripts/make_linux_installer.sh
```

The installer will be in `installer/TurbulenceRealmSINDy-2.2.0-Linux-Installer.run`.

#### Installer features

- **Disclaimer prompt**: Shows the no-liability disclaimer (requires acceptance)
- **Install location**: User-selectable (defaults to `/opt/TurbulenceRealmSINDy`)
- **Application menu entry**: Creates a `.desktop` file for the app menu
- **Optional desktop shortcut**: User can opt in during install
- **Uninstaller**: Run `/opt/TurbulenceRealmSINDy/install.sh --uninstall`
- **Launch on finish**: Option to start the app after installation

The install script is in `linux/install.sh` and the Makeself wrapper is in
`scripts/make_linux_installer.sh`.

### Building a .deb Package (Debian/Ubuntu)

To create a `.deb` package for installation via `dpkg`/`apt`:

#### Prerequisites

- Debian/Ubuntu (or use the GitHub Actions CI workflow)
- `dpkg-deb` installed (standard on Debian/Ubuntu)
- A PyInstaller build already in `build/dist/TurbulenceRealmSINDy/`

#### Build the package

```bash
# First, build the executable with PyInstaller
pyinstaller TurbulenceRealmSINDy.spec

# Then, build the .deb
./scripts/make_deb.sh
```

The package will be in `installer/TurbulenceRealmSINDy-2.2.0-amd64.deb`.

#### Install / uninstall

```bash
# Install
sudo dpkg -i installer/TurbulenceRealmSINDy-2.2.0-amd64.deb

# Uninstall
sudo dpkg -r turbulencerealm-sindy
```

The `.deb` installs to `/opt/TurbulenceRealmSINDy/` and creates an application
menu entry automatically. The package metadata is in `linux/deb/DEBIAN/control`.

## Project Structure

```
TR-SINDY/
├── src/
│   └── tr_sindy_app/           # Desktop app package
│       ├── __init__.py         # version, public API
│       ├── app.py              # entry point + splash screen
│       ├── gui.py              # main window + pages
│       ├── theme.py            # gold/cream dark/light theme, stylesheet
│       ├── glass_background.py # animated glassmorphism background
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
├── logo.png                    # Turbulence Realm brand mark
├── logo.ico                    # Windows icon (for installer)
├── DISCLAIMER.txt              # No-liability disclaimer (shown in installers)
├── tests/                      # pytest test suite
├── scripts/
│   ├── build.py                # executable build script
│   ├── make_linux_installer.sh # Makeself .run installer builder
│   └── make_deb.sh             # .deb package builder
├── linux/
│   ├── install.sh              # Linux post-install script (shortcuts, uninstall)
│   └── deb/                    # .deb package template
│       ├── DEBIAN/control      # dpkg package metadata
│       └── usr/share/          # .desktop entry + icon
├── archive/
│   └── TR-SINDY-Final.py       # original v1.0 reference
├── .github/workflows/
│   ├── ci.yml                  # CI: ruff + pytest + GUI smoke test
│   └── build-release.yml       # Build Windows + Linux installers
├── run.py                      # launcher script
├── pyproject.toml              # package metadata + tool config
├── TurbulenceRealmSINDy.spec   # PyInstaller spec
├── TurbulenceRealmSINDy.iss    # Inno Setup script (Windows installer)
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

If you use TR-SINDy in your research, please cite it:

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
