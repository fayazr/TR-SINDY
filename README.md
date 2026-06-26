# Turbulence Realm — SINDy

A futuristic desktop console for **video-based fluid-flow analysis**. It extracts
a dense velocity field from a video using Farneback optical flow, then fits a
**SINDy** (Sparse Identification of Nonlinear Dynamics) model to the flow,
predicts the reconstructed field, and provides rich visualisation and export.

This is a modern UI reskin of the original `TR-SINDY-Final.py`. The full
processing pipeline is preserved; only the interface has been redesigned into a
dark, neon "Flux Reactor" console (cyan/violet plasma accents, hot-magenta for
the SINDy/prediction layer).

## Features

- **Optical Flow**: dense Farneback flow over a calibrated region of interest,
  with optional Gaussian / Non-Local-Means denoising and a live preview.
- **SINDy Modeling**: polynomial feature library + STLSQ optimiser, fit on the
  flow field and its spatial/temporal derivatives.
- **Prediction**: batched SINDy prediction and smoothed field reconstruction.
- **Visualisation**: live actual-vs-SINDy quiver comparison (frame slider),
  contour magnitude plots, streamlines, and RMSE/MSE error analysis + spatial
  error maps.
- **Export**: per-frame CSV (actual & predicted u/v) and an MP4 quiver overlay
  animation (requires FFmpeg).

## Installation

```bash
cd TR-SINDY
pip install -r requirements.txt
```

> `pysindy` pulls in `scikit-learn`. FFmpeg is only required for animation
> export.

## Run

```bash
python run.py
# or
python src/tr_sindy.py
```

## Workflow

1. **Video Source** — Browse and select a video (`.mp4 / .avi / .mov`).
2. **ROI & Calibration** — Draw a region of interest, then a calibration line
   of known real-world length (meters) to set the scale.
3. **Process Optical Flow** — Choose a filter, then compute the dense flow
   field (a live HSV + quiver preview is shown).
4. **Run SINDy Modeling** — Pick the polynomial degree and STLSQ threshold.
5. **Run SINDy Prediction** — Reconstruct the predicted field.
6. **Visualization & Analysis** — Quiver comparison, contour, streamlines,
   error maps.
7. **Export Options** — Frame CSVs and quiver animation.

The status pill (top-right) and the bottom status bar report pipeline state
throughout.

## Project Structure

```
TR-SINDY/
├── src/
│   └── tr_sindy.py          # Main application (modern UI)
├── run.py                   # Launcher
├── requirements.txt
├── config.json
├── TR-SINDY-Final.py        # Original reference implementation
└── README.md
```

## Building a Windows executable

See `Nuitka for windows` for the original Nuitka one-file command (uses
`logo.png`, `logo.ico`, and a bundled `ffmpeg.exe`).

## Author

Developed by Fayaz Rasheed — [www.turbulencerealm.com](https://www.turbulencerealm.com)
