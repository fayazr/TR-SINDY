"""Data export and reporting.

Formats:
    * CSV (per-frame, original behaviour)
    * HDF5 (h5py)
    * NetCDF (scipy.io.netcdf or xarray/netCDF4)
    * Parquet (pyarrow)
    * JSON metadata / configuration
    * Image sequences (PNG/JPG per frame)
    * PDF report (matplotlib PdfPages)
    * MP4 quiver animation (matplotlib + FFmpeg)

Heavy/optional writers degrade gracefully: if the backend library is
missing, a clear error is raised listing the install command.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter

from ._logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------
#  Metadata helpers
# ---------------------------------------------------------------------
def build_metadata(project: dict, of_result: dict, sindy_result: dict,
                   extra: Optional[dict] = None,
                   seed: Optional[int] = None) -> dict:
    """Build export metadata including provenance for reproducibility.

    Parameters
    ----------
    project : project state dict (video_file, roi, calibration, configs).
    of_result : optical-flow result dict.
    sindy_result : SINDy fit result dict.
    extra : additional metadata fields to merge.
    seed : random seed used for this run (recorded in provenance).
    """
    from . import __version__ as _v
    from ._provenance import collect_provenance

    config = {
        "optical_flow": project.get("optical_flow", {}),
        "sindy": project.get("sindy", {}),
    }
    meta = {
        "app": "Turbulence Realm - SINDy",
        "version": _v,
        "video_file": project.get("video_file"),
        "roi_box": list(project.get("roi_box", [])),
        "calibration_px": project.get("calibration_px"),
        "calibration_m": project.get("calibration_m"),
        "meters_per_pixel": project.get("meters_per_pixel"),
        "optical_flow": {k: v for k, v in of_result.items()
                         if k not in ("u_mmap_path", "v_mmap_path", "frame_mmap_path")},
        "sindy": {k: v for k, v in sindy_result.items()
                  if k not in ("model", "X_optical_path", "X_dot_optical_path")},
        "provenance": collect_provenance(
            input_file=project.get("video_file"),
            seed=seed,
            config=config,
        ),
    }
    if extra:
        meta.update(extra)
    return meta


def export_metadata(path: str, metadata: dict) -> None:
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------
#  CSV
# ---------------------------------------------------------------------
def export_csv(output_dir: str, u_mmap, v_mmap, pred: np.ndarray,
               progress_cb=None) -> int:
    os.makedirs(output_dir, exist_ok=True)
    n = u_mmap.shape[0]
    for f in range(n):
        df = pd.DataFrame({
            "actual_u": u_mmap[f].ravel(),
            "actual_v": v_mmap[f].ravel(),
            "predicted_u": pred[f, ..., 0].ravel(),
            "predicted_v": pred[f, ..., 1].ravel(),
        })
        df.to_csv(os.path.join(output_dir, f"frame_{f + 1}_values.csv"), index=False)
        if progress_cb:
            progress_cb(f + 1, n, "csv")
    return n


# ---------------------------------------------------------------------
#  HDF5
# ---------------------------------------------------------------------
def export_hdf5(path: str, u_mmap, v_mmap, frame_mmap, pred: np.ndarray,
                metadata: Optional[dict] = None) -> None:
    try:
        import h5py
    except Exception as exc:
        raise RuntimeError("HDF5 export requires h5py: pip install h5py") from exc
    with h5py.File(path, "w") as f:
        f.create_dataset("u", data=np.asarray(u_mmap))
        f.create_dataset("v", data=np.asarray(v_mmap))
        f.create_dataset("frames", data=np.asarray(frame_mmap))
        f.create_dataset("predicted_u", data=pred[..., 0])
        f.create_dataset("predicted_v", data=pred[..., 1])
        if metadata:
            f.attrs["metadata"] = json.dumps(metadata, default=_json_default)


# ---------------------------------------------------------------------
#  NetCDF
# ---------------------------------------------------------------------
def export_netcdf(path: str, u_mmap, v_mmap, pred: np.ndarray,
                  metadata: Optional[dict] = None) -> None:
    try:
        from scipy.io import netcdf_file
    except Exception as exc:
        raise RuntimeError("NetCDF export requires scipy.io.netcdf") from exc
    n, h, w = u_mmap.shape
    f = netcdf_file(path, "w")
    f.createDimension("time", n)
    f.createDimension("y", h)
    f.createDimension("x", w)
    vu = f.createVariable("u", "f4", ("time", "y", "x"))
    vv = f.createVariable("v", "f4", ("time", "y", "x"))
    vpu = f.createVariable("predicted_u", "f4", ("time", "y", "x"))
    vpv = f.createVariable("predicted_v", "f4", ("time", "y", "x"))
    vu[:] = np.asarray(u_mmap)
    vv[:] = np.asarray(v_mmap)
    vpu[:] = pred[..., 0]
    vpv[:] = pred[..., 1]
    if metadata:
        for k, val in metadata.items():
            try:
                f.history = f"{k}: {val}"
            except Exception as e:
                log.debug("netcdf: could not store metadata %s: %s", k, e)
    f.close()


# ---------------------------------------------------------------------
#  Parquet
# ---------------------------------------------------------------------
def export_parquet(path: str, u_mmap, v_mmap, pred: np.ndarray,
                   progress_cb=None) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("Parquet export requires pyarrow: pip install pyarrow") from exc
    n, h, w = u_mmap.shape
    frames = np.repeat(np.arange(1, n + 1), h * w)
    yy = np.tile(np.repeat(np.arange(h), w), n)
    xx = np.tile(np.tile(np.arange(w), h), n)
    table = pa.table({
        "frame": frames,
        "y": yy,
        "x": xx,
        "actual_u": np.asarray(u_mmap).ravel(),
        "actual_v": np.asarray(v_mmap).ravel(),
        "predicted_u": pred[..., 0].ravel(),
        "predicted_v": pred[..., 1].ravel(),
    })
    pq.write_table(table, path)


# ---------------------------------------------------------------------
#  Image sequences
# ---------------------------------------------------------------------
def export_image_sequence(output_dir: str, frame_mmap, ext: str = "png",
                          progress_cb=None) -> int:
    import cv2
    os.makedirs(output_dir, exist_ok=True)
    n = frame_mmap.shape[0]
    for f in range(n):
        cv2.imwrite(os.path.join(output_dir, f"frame_{f + 1:05d}.{ext}"),
                    np.asarray(frame_mmap[f]))
        if progress_cb:
            progress_cb(f + 1, n, "images")
    return n


# ---------------------------------------------------------------------
#  PDF report
# ---------------------------------------------------------------------
def export_pdf_report(path: str, u_mmap, v_mmap, pred: np.ndarray,
                      metadata: Optional[dict] = None,
                      analysis: Optional[dict] = None) -> None:
    from matplotlib.backends.backend_pdf import PdfPages

    from .analysis import error_metrics, per_frame_errors, vorticity
    n = u_mmap.shape[0]
    errs = per_frame_errors(u_mmap, v_mmap, pred)
    overall = error_metrics(np.asarray(u_mmap), np.asarray(v_mmap),
                            pred[..., 0], pred[..., 1])
    with PdfPages(path) as pdf:
        # page 1: summary
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle("Turbulence Realm - SINDy Analysis Report", fontsize=16)
        txt = "Overall error metrics\n"
        for k, v in overall.items():
            txt += f"  {k}: {v:.5g}\n"
        if metadata:
            txt += "\nMetadata\n"
            for k in ("video_file", "meters_per_pixel", "optical_flow", "sindy"):
                if k in metadata:
                    txt += f"  {k}: {metadata[k]}\n"
        if analysis:
            txt += "\nAnalysis\n"
            for k, v in analysis.items():
                txt += f"  {k}: {v}\n"
        fig.text(0.08, 0.85, txt, family="monospace", fontsize=9)
        pdf.savefig(fig); plt.close(fig)
        # page 2: per-frame errors
        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.plot(errs["rmse"], label="RMSE")
        ax.plot(errs["mae"], label="MAE")
        ax.set_xlabel("frame"); ax.set_ylabel("error"); ax.legend()
        ax.set_title("Per-frame errors")
        pdf.savefig(fig); plt.close(fig)
        # page 3: vorticity of middle frame
        mid = n // 2
        w = vorticity(np.asarray(u_mmap[mid]), np.asarray(v_mmap[mid]))
        fig, ax = plt.subplots(figsize=(8.5, 6))
        im = ax.imshow(w, cmap="RdBu_r"); ax.set_title(f"Vorticity (frame {mid + 1})")
        fig.colorbar(im, ax=ax)
        pdf.savefig(fig); plt.close(fig)
        d = pdf.infodict()
        d["Title"] = "TR-SINDy Analysis Report"
        d["Producer"] = "Turbulence Realm - SINDy"


# ---------------------------------------------------------------------
#  Animation
# ---------------------------------------------------------------------
def export_animation(path: str, u_mmap, v_mmap, frame_mmap, pred: np.ndarray,
                     meters_per_pixel: float, scale: float = 1.0,
                     width: float = 0.006, fps: int = 8,
                     ffmpeg_path: Optional[str] = None,
                     progress_cb=None) -> None:
    import matplotlib.animation as animation
    if ffmpeg_path:
        plt.rcParams["animation.ffmpeg_path"] = ffmpeg_path
    n, h, w = u_mmap.shape
    step = max(h // 30, 2)
    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)
    xg, yg = np.meshgrid(xs, ys)
    extent = [0, w * meters_per_pixel, h * meters_per_pixel, 0]
    from .theme import Theme
    fig, ax = plt.subplots(figsize=(8, 8))

    def animate(fidx):
        ax.clear()
        ax.imshow(np.asarray(frame_mmap[fidx]), cmap="gray", origin="upper", extent=extent)
        ax.quiver(xg * meters_per_pixel, yg * meters_per_pixel,
                  np.asarray(u_mmap[fidx])[ys[:, None], xs[None, :]],
                  np.asarray(v_mmap[fidx])[ys[:, None], xs[None, :]],
                  color=Theme.ACCENT, angles="xy", scale=scale, width=width, label="Actual")
        ax.quiver(xg * meters_per_pixel, yg * meters_per_pixel,
                  pred[fidx, ys[:, None], xs[None, :], 0],
                  pred[fidx, ys[:, None], xs[None, :], 1],
                  color=Theme.MAGENTA, angles="xy", scale=scale, width=width, label="SINDy")
        ax.set_title(f"Frame {fidx + 1}: Quiver Overlay")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.legend()
        if progress_cb:
            progress_cb(fidx + 1, n, "animation")

    ani = animation.FuncAnimation(fig, animate, frames=n, interval=200)
    ani.save(path, writer=FFMpegWriter(fps=fps))
    plt.close(fig)
