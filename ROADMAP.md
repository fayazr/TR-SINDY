# TR-SINDy Development Roadmap

This document tracks what has been implemented in v2.0 and what remains as
future work. The v2.0 release refactored the original monolithic
`TR-SINDY-Final.py` into a modular package (`src/tr_sindy_app/`) and added a
large set of scientific, ML, analysis, export and UX features.

Status legend: ✅ implemented & smoke-tested · 🟡 scaffold/stub (API present,
needs hardening) · ⬜ planned (not yet implemented).

---

## A. Scientific & Methodological

### Optical flow
- ✅ Farneback (dense, default)
- ✅ Lucas-Kanade (sparse, RBF-interpolated to dense)
- ✅ TV-L1 (DualTVL1, requires opencv-contrib)
- ✅ RAFT / PWC-Net (torchvision, lazy)
- ✅ Multi-scale pyramid (coarse-to-fine)
- ✅ Temporal smoothing (EMA / moving average)
- ✅ Forward-backward consistency error & quality metrics
- ⬜ EpicFlow, FlowNet2.0

### SINDy
- ✅ Multiple feature libraries (polynomial, Fourier, combined, custom, trig)
- ✅ Multiple optimizers (STLSQ, SR3, FROLS, constrained SR3 w/ fallback)
- ✅ k-fold cross-validation with coefficient stability
- ✅ Model comparison (multi-config RMSE / complexity)
- ✅ Divergence-free constraint option (placeholder matrix; user can supply)
- ✅ Time-delay embedding helper
- ⬜ SINDy with explicit control inputs (API hook present, not wired in UI)
- ⬜ Trigonometric / piecewise-polynomial libraries as first-class UI options

### Analysis
- ✅ Vorticity, strain-rate tensor, divergence
- ✅ Spatial FFT spectrum, temporal PSD (Welch)
- ✅ POD (snapshot method), DMD (standard)
- ✅ Turbulence statistics (TKE, Reynolds number, Reynolds stress, intensity)
- ✅ Structure function, velocity PDF
- ✅ Comprehensive error metrics (RMSE/MSE/MAE/max/median/p95/correlation/NRMSE)
- ✅ Region-based statistics
- ✅ Energy cascade analysis, structure-function scaling exponents
- ✅ Isotropic kinetic energy spectrum with Kolmogorov exponent fit

### Data quality
- ✅ Outlier detection (z-score, modified z-score, IQR)
- ✅ Outlier replacement via interpolation
- ✅ RBF & kriging (pykrige optional) interpolation
- ✅ Noise estimation (Laplacian MAD)
- ✅ Flow quality mask (forward-backward + magnitude)

---

## B. Machine Learning (torch, lazy)

- ✅ PINN for Navier-Stokes (autograd residual + data loss)
- ✅ Autoencoder-SINDy (latent sparse identification)
- ✅ Fourier Neural Operator (FNO2D, spectral conv)
- ✅ DeepONet (branch + trunk)
- ✅ ConvLSTM spatiotemporal forecaster
- ✅ VAE / beta-VAE
- ✅ Ensemble uncertainty
- ✅ Monte-Carlo dropout uncertainty
- ✅ GAN flow synthesis (scaffold, trainable)
- ✅ Granger causality + causal SINDy summary
- 🟡 Diffusion model (stub — integrate `diffusers`)
- 🟡 Graph Neural Network (stub — integrate `torch_geometric`/`dgl`)
- ⬜ Reinforcement learning for flow control
- ⬜ Hybrid PINN-SINDy with adaptive basis functions

---

## C. Data Export & Reporting

- ✅ CSV (per-frame)
- ✅ HDF5 (h5py)
- ✅ NetCDF (scipy.io.netcdf)
- ✅ Parquet (pyarrow)
- ✅ JSON metadata
- ✅ Image sequence (PNG/JPG)
- ✅ PDF report (metrics + figures)
- ✅ MP4 quiver animation (FFmpeg)
- ✅ Configuration/metadata included in exports

---

## D. Interface & UX

- ✅ Project save/load (.trsindy, bundles memmaps)
- ✅ Parameter presets (built-in + user, persistent)
- ✅ Recent files menu
- ✅ Processing history log
- ✅ Dark/light theme toggle (Ctrl+T)
- ✅ Keyboard shortcuts (Ctrl+O/S/L/Q/T, F1)
- ✅ Tooltips on all key parameters
- ✅ ROI/calibration undo/redo (Ctrl+Z / Ctrl+Y)
- ✅ Side-by-side actual vs predicted quiver
- ✅ Vorticity / strain / POD / DMD / spectral plots
- ✅ Animated heatmap, custom colormap selection
- ✅ Data table view
- ✅ ML models tab with live log
- ✅ CLI batch mode (`python -m tr_sindy_app.cli`)
- ✅ Auto-save / session restore
- ✅ Settings dialog (mmap dir, backend, theme, FFmpeg path via QSettings)
- ✅ Progress ETA display for long-running jobs
- ✅ Logging throughout (console + in-app log widget bridge)
- ⬜ 3D flow visualization (mayavi/pyvista)
- ⬜ Interactive tutorials / guided tours
- ⬜ Multi-language (i18n)
- ⬜ Plugin system
- ⬜ REST API / Jupyter notebook export
- ⬜ Customizable drag-and-drop panel layout
- ⬜ Cloud / cluster (SLURM/PBS) submission

---

## E. Performance

- ✅ Memmap-based streaming for large videos
- ✅ Batched SINDy fit/predict
- ✅ GPU auto-detection for torch models
- ⬜ CUDA optical flow (Farneback is CPU; RAFT uses GPU when available)
- ⬜ Chunked processing options in UI
- ⬜ Intelligent caching of intermediate results
- ✅ Progress time estimates (ETA in status bar)

---

## F. Engineering & Reproducibility

- ✅ LICENSE (MIT)
- ✅ `pyproject.toml` with project metadata, console scripts, optional deps
- ✅ `CHANGELOG.md` (Keep a Changelog format)
- ✅ `CITATION.cff` + `CITATION.bib` for academic attribution
- ✅ Test suite (pytest, 72 tests covering analysis, quality, SINDy, project, ML export, provenance)
- ✅ CI workflow (GitHub Actions: ruff + pytest + GUI smoke test)
- ✅ Ruff linter + pre-commit hooks
- ✅ Logging throughout (replaces print statements; GUI log bridge)
- ✅ Provenance metadata in exports (git commit, package versions, config, seed, SHA-256)
- ✅ Reproducible mode (`set_reproducible()` seeds numpy/random/torch + deterministic algorithms)
- ✅ Dynamic `__version__` from package metadata (splash, CLI --version, About dialog)
- ✅ `TR-SINDY-Final.py` archived to `archive/`
- ⬜ Split `gui.py` into `gui/` package (pages/, workers/, logging_bridge/)
- ⬜ Type hints on public APIs (mypy --strict on core modules)
- ⬜ mypy integration in CI

---

## Notes on stubs

Items marked 🟡 expose a stable API and raise clear, actionable errors
directing the user to install the required integration library
(`diffusers`, `torch_geometric`, `dgl`). They are ready to be fleshed out
without changing call sites.
