"""Command-line batch processing interface for TR-SINDy.

Examples:
    python -m tr_sindy_app.cli process video.mp4 --roi 100,50,400,300 \\
        --calib-px 120 --calib-m 0.1 --backend farneback \\
        --library polynomial --degree 3 --threshold 0.07 \\
        --export-dir ./out --formats csv,hdf5,pdf

    python -m tr_sindy_app.cli batch jobs.json

    python -m tr_sindy_app.cli presets list
    python -m tr_sindy_app.cli presets apply farneback-default ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from . import analysis, export, optical_flow, project, sindy_core
from ._logging import configure_cli_logging, get_logger

log = get_logger(__name__)


def _parse_roi(s: str):
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--roi must be x0,y0,x1,y1")
    return tuple(parts)


def _process_one(video_file: str, roi, calib_px: float, calib_m: float,
                 flow_cfg: optical_flow.FlowConfig,
                 sindy_cfg: sindy_core.SINDyConfig,
                 export_dir: str, formats: list[str],
                 mmap_dir: str, verbose: bool = True) -> dict:
    meters_per_pixel = calib_m / calib_px
    os.makedirs(mmap_dir, exist_ok=True)
    mmap_paths = {
        "u": os.path.abspath(os.path.join(mmap_dir, "u.dat")),
        "v": os.path.abspath(os.path.join(mmap_dir, "v.dat")),
        "frames": os.path.abspath(os.path.join(mmap_dir, "frames.dat")),
    }
    for p in mmap_paths.values():
        if os.path.exists(p):
            os.remove(p)

    def pcb(i, n, stage):
        if verbose and i % max(1, n // 20) == 0:
            log.info("[%s] %d/%d", stage, i, n)

    of_meta = optical_flow.process_video(video_file, roi, meters_per_pixel,
                                         flow_cfg, mmap_paths, progress_cb=pcb)
    n, h, w = of_meta["frames"], of_meta["roi_h"], of_meta["roi_w"]
    u_mmap = np.memmap(mmap_paths["u"], np.float32, mode="r", shape=(n, h, w))
    v_mmap = np.memmap(mmap_paths["v"], np.float32, mode="r", shape=(n, h, w))
    frame_mmap = np.memmap(mmap_paths["frames"], np.uint8, mode="r", shape=(n, h, w))

    X, Xdot, names = sindy_core.build_sindy_dataset(u_mmap, v_mmap, 1.0 / of_meta["FPS"])
    fit = sindy_core.fit_sindy(X, Xdot, 1.0 / of_meta["FPS"], sindy_cfg,
                               feature_names=names, progress_cb=pcb)
    pred = sindy_core.predict_sindy(fit["model"], X, progress_cb=pcb)
    pred_field = np.zeros((n, h, w, 2), np.float32)
    for f in range(n):
        seg = pred[f * h * w:(f + 1) * h * w]
        pred_field[f, ..., 0] = gaussian_filter(seg[:, 0].reshape(h, w), 0.8)
        pred_field[f, ..., 1] = gaussian_filter(seg[:, 1].reshape(h, w), 0.8)

    os.makedirs(export_dir, exist_ok=True)
    project_state = {
        "video_file": video_file, "roi_box": list(roi),
        "calibration_px": calib_px, "calibration_m": calib_m,
        "meters_per_pixel": meters_per_pixel,
        "optical_flow": vars(flow_cfg),
        "sindy": vars(sindy_cfg),
    }
    metadata = export.build_metadata(project_state, of_meta, fit)

    written = []
    for fmt in formats:
        fmt = fmt.lower()
        try:
            if fmt == "csv":
                export.export_csv(os.path.join(export_dir, "csv"), u_mmap, v_mmap,
                                  pred_field, progress_cb=pcb)
                written.append("csv")
            elif fmt == "hdf5":
                export.export_hdf5(os.path.join(export_dir, "result.h5"),
                                   u_mmap, v_mmap, frame_mmap, pred_field, metadata)
                written.append("hdf5")
            elif fmt == "netcdf":
                export.export_netcdf(os.path.join(export_dir, "result.nc"),
                                     u_mmap, v_mmap, pred_field, metadata)
                written.append("netcdf")
            elif fmt == "parquet":
                export.export_parquet(os.path.join(export_dir, "result.parquet"),
                                      u_mmap, v_mmap, pred_field, progress_cb=pcb)
                written.append("parquet")
            elif fmt == "json":
                export.export_metadata(os.path.join(export_dir, "metadata.json"),
                                       metadata)
                written.append("json")
            elif fmt == "pdf":
                export.export_pdf_report(os.path.join(export_dir, "report.pdf"),
                                         u_mmap, v_mmap, pred_field, metadata)
                written.append("pdf")
            elif fmt == "images":
                export.export_image_sequence(os.path.join(export_dir, "frames"),
                                             frame_mmap, progress_cb=pcb)
                written.append("images")
            else:
                log.warning("unknown format: %s", fmt)
        except Exception as e:
            log.error("%s export failed: %s", fmt, e)

    # analysis summary
    metrics = analysis.error_metrics(np.asarray(u_mmap), np.asarray(v_mmap),
                                     pred_field[..., 0], pred_field[..., 1])
    log.info("Error metrics:")
    for k, v in metrics.items():
        log.info("  %s: %.5g", k, v)

    proj_path = os.path.join(export_dir, "project" + project.PROJECT_EXT)
    project.save_project(proj_path, project_state, mmap_paths, bundle_mmaps=True)
    log.info("Project saved: %s", proj_path)
    log.info("Exports: %s", ", ".join(written))
    return {"metrics": metrics, "exports": written, "project": proj_path}


def _build_flow_cfg(args) -> optical_flow.FlowConfig:
    return optical_flow.FlowConfig(
        backend=args.backend,
        multiscale=args.multiscale,
        multiscale_levels=args.multiscale_levels,
        temporal_smoothing=args.temporal_smoothing,
        temporal_alpha=args.temporal_alpha,
        temporal_window=args.temporal_window,
        enable_gauss=args.gauss,
        gauss_ksize=args.gauss_ksize,
        gauss_sigma=args.gauss_sigma,
        enable_nlm=args.nlm,
        nlm_h=args.nlm_h,
        compute_quality=args.quality,
    )


def _build_sindy_cfg(args) -> sindy_core.SINDyConfig:
    return sindy_core.SINDyConfig(
        library=args.library, degree=args.degree, n_freq=args.n_freq,
        optimizer=args.optimizer, threshold=args.threshold,
        divergence_free=args.divergence_free,
    )


def cmd_process(args):
    flow_cfg = _build_flow_cfg(args)
    sindy_cfg = _build_sindy_cfg(args)
    _process_one(args.video, args.roi, args.calib_px, args.calib_m,
                 flow_cfg, sindy_cfg, args.export_dir,
                 [f.strip() for f in args.formats.split(",")],
                 args.mmap_dir)


def cmd_batch(args):
    with open(args.jobs) as f:
        jobs = json.load(f)
    results = []
    for i, job in enumerate(jobs):
        log.info("=== Job %d/%d: %s ===", i + 1, len(jobs), job.get("video"))
        flow_cfg = optical_flow.FlowConfig(**job.get("optical_flow", {}))
        sindy_cfg = sindy_core.SINDyConfig(**job.get("sindy", {}))
        res = _process_one(job["video"], tuple(job["roi"]), job["calib_px"],
                           job["calib_m"], flow_cfg, sindy_cfg,
                           job.get("export_dir", f"./out/job_{i + 1}"),
                           job.get("formats", "csv,json").split(","),
                           job.get("mmap_dir", "./velocity_mmaps"))
        results.append(res)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)


def cmd_presets(args):
    project.install_builtin_presets()
    if args.action == "list":
        for name in project.list_presets():
            print(name)
    elif args.action == "show":
        p = project.load_preset(args.name)
        print(json.dumps(p.to_dict(), indent=2))
    elif args.action == "delete":
        project.delete_preset(args.name)
        log.info("deleted %s", args.name)


def build_parser() -> argparse.ArgumentParser:
    from . import __version__ as _v
    p = argparse.ArgumentParser(prog="tr-sindy", description="TR-SINDy batch CLI")
    p.add_argument("--version", action="version", version=f"tr-sindy {_v}")
    p.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("process", help="process a single video")
    pp.add_argument("video")
    pp.add_argument("--roi", type=_parse_roi, required=True)
    pp.add_argument("--calib-px", type=float, required=True)
    pp.add_argument("--calib-m", type=float, required=True)
    pp.add_argument("--backend", default="farneback",
                    choices=optical_flow.available_backends())
    pp.add_argument("--multiscale", action="store_true")
    pp.add_argument("--multiscale-levels", type=int, default=3)
    pp.add_argument("--temporal-smoothing", default="none",
                    choices=["none", "ema", "moving"])
    pp.add_argument("--temporal-alpha", type=float, default=0.6)
    pp.add_argument("--temporal-window", type=int, default=3)
    pp.add_argument("--gauss", action="store_true")
    pp.add_argument("--gauss-ksize", type=int, default=5)
    pp.add_argument("--gauss-sigma", type=float, default=1.3)
    pp.add_argument("--nlm", action="store_true")
    pp.add_argument("--nlm-h", type=float, default=10.0)
    pp.add_argument("--quality", action="store_true")
    pp.add_argument("--library", default="polynomial",
                    choices=["polynomial", "fourier", "combined", "custom", "trig"])
    pp.add_argument("--degree", type=int, default=3)
    pp.add_argument("--n-freq", type=int, default=1)
    pp.add_argument("--optimizer", default="stlsq",
                    choices=["stlsq", "sr3", "frols", "constrained_sr3"])
    pp.add_argument("--threshold", type=float, default=0.07)
    pp.add_argument("--divergence-free", action="store_true")
    pp.add_argument("--export-dir", default="./output")
    pp.add_argument("--formats", default="csv,json,pdf")
    pp.add_argument("--mmap-dir", default="./velocity_mmaps")
    pp.set_defaults(func=cmd_process)

    bp = sub.add_parser("batch", help="process multiple jobs from a JSON file")
    bp.add_argument("jobs")
    bp.add_argument("--output", help="optional results JSON")
    bp.set_defaults(func=cmd_batch)

    pr = sub.add_parser("presets", help="manage parameter presets")
    pr.add_argument("action", choices=["list", "show", "delete"])
    pr.add_argument("name", nargs="?")
    pr.set_defaults(func=cmd_presets)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    configure_cli_logging(verbose=getattr(args, "verbose", False))
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
