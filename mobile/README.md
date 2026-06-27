# Turbulence Realm SINDy — Mobile Edition

A Kivy-based mobile app for video-based fluid-flow analysis that runs
**on-device** (Android phone/tablet). No server required.

## Features

- **Video loading** — Pick a video from device storage
- **Touch ROI selection** — Draw a rectangle on the first frame
- **Optical flow** — Farneback or Lucas-Kanade (via OpenCV)
- **SINDy modeling** — Pure-numpy STLSQ optimizer (no scipy needed)
- **Visualization** — Quiver, magnitude, and vorticity fields on Kivy canvas
- **Equation display** — Shows the discovered sparse dynamics equations

## Limitations vs Desktop

| Feature | Desktop | Mobile |
|---------|---------|--------|
| Optical flow | Farneback, LK, TV-L1, RAFT, PWC-Net | Farneback, LK |
| SINDy libraries | poly, Fourier, combined, custom, trig | Polynomial only |
| Optimizers | STLSQ, SR3, FROLS, constrained | STLSQ |
| ML models | PINN, FNO, DeepONet, ConvLSTM, VAE, GAN | Not available |
| Analysis | POD, DMD, spectra, structure functions | Basic (vorticity) |
| Export | CSV, HDF5, NetCDF, Parquet, PDF, MP4 | Not available |
| Max frames | Unlimited | 60 (performance) |

The mobile version is designed for **on-site quick analysis**. For full
processing, use the desktop version.

## Installation

### Desktop testing (Linux/macOS/Windows)

```bash
cd mobile
pip install kivy numpy opencv-python
python main.py
```

### Android build

```bash
# Install buildozer
pip install buildozer

# Build APK (requires Java, Android SDK, NDK — buildozer auto-downloads)
cd mobile
buildozer -v android debug

# Output: bin/trsindy-2.2.0-debug.apk
```

Install on your device:
```bash
adb install bin/trsindy-2.2.0-debug.apk
```

### Prerequisites for Android build

- Linux or WSL (buildozer doesn't work on native Windows)
- Java JDK 11+
- ~10GB free disk space for SDK/NDK
- Python 3.8+

## Usage

1. **Setup tab** → Tap "Browse Video…" and select an MP4/AVI file
2. **Flow tab** → Touch and drag to draw a rectangle over the region of interest
3. **Setup tab** → Enter calibration values (pixels and meters)
4. **Setup tab** → Tap "① Process Optical Flow"
5. **Setup tab** → Tap "② Run SINDy"
6. **SINDy tab** → View the discovered equations
7. **Viz tab** → Switch between quiver, magnitude, and vorticity views

## Architecture

```
mobile/
├── main.py                    # entry point
├── buildozer.spec             # Android build config
├── tr_sindy_mobile/
│   ├── __init__.py
│   ├── main.py                # Kivy app + UI (KV language)
│   ├── sindy_lite.py          # pure-numpy SINDy (STLSQ + polynomial library)
│   └── flow_lite.py           # optical flow (cv2 Farneback/LK + numpy fallback)
└── README.md
```

### Why pure-numpy SINDy?

The desktop version uses `pysindy` which depends on `scipy` and `scikit-learn`.
Neither library has Android builds. `sindy_lite.py` implements the STLSQ
optimizer and polynomial library builder using only numpy, which works on
mobile via python-for-android.

### Why Kivy canvas instead of matplotlib?

Matplotlib doesn't work with Kivy's rendering pipeline on mobile. The
`FlowCanvasWidget` draws velocity fields directly using Kivy's GPU-accelerated
canvas instructions.
