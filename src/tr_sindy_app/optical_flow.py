"""Optical-flow computation backends and helpers.

Backends:
    * Farneback (dense, OpenCV)            - the original default
    * Lucas-Kanade (sparse feature track) - upsampled to a dense grid
    * TV-L1 (DualTVL1, OpenCV optflow)    - total-variation regularised
    * RAFT / PWC-Net (torchvision)        - deep learning, lazy import

Additional capabilities:
    * Temporal smoothing (exponential / moving-average) of the flow field
    * Multi-scale pyramid processing (coarse-to-fine)
    * Forward-backward consistency error & per-pixel quality metrics
    * Outlier masking & interpolation (delegated to quality.py)

All backends return a flow array of shape (H, W, 2) in (x, y) pixel units.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from ._logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------
#  Backend registry
# ---------------------------------------------------------------------
_AVAILABLE_BACKENDS: list[str] | None = None


def available_backends() -> list[str]:
    """Return the list of optical-flow backends usable in this environment.

    Uses ``importlib.util.find_spec`` to probe for torch/torchvision without
    actually importing them (importing torch takes ~10s on some systems).
    The result is cached after the first call.
    """
    global _AVAILABLE_BACKENDS
    if _AVAILABLE_BACKENDS is not None:
        return _AVAILABLE_BACKENDS
    import importlib.util
    backends = ["farneback", "lucas_kanade", "tvl1"]
    if importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"):
        backends.append("raft")
        # PWC-Net: not probed here because find_spec on the submodule still
        # triggers a torch import (~4s).  The backend will raise a clear
        # error at runtime if selected and unavailable.
    _AVAILABLE_BACKENDS = backends
    return backends


def has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------
#  Single-pair flow estimators
# ---------------------------------------------------------------------
def farneback(prev_gray, curr_gray, params: Optional[dict] = None) -> np.ndarray:
    p = dict(pyr_scale=0.3, levels=7, winsize=21, iterations=7,
             poly_n=7, poly_sigma=1.1, flags=0)
    if params:
        p.update(params)
    return cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **p)


def lucas_kanade(prev_gray, curr_gray, params: Optional[dict] = None,
                 grid_step: int = 8) -> np.ndarray:
    """Sparse Lucas-Kanade feature tracking interpolated onto a dense grid."""
    p = dict(winSize=(21, 21), maxLevel=5,
             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if params:
        p.update(params)
    h, w = prev_gray.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    pts0 = np.ascontiguousarray(
        np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)[::grid_step])
    pts1, st, _err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts0, None, **p)
    if pts1 is None:
        return np.zeros((h, w, 2), np.float32)
    st = st.ravel().astype(bool)
    src = pts0[st]
    dst = pts1[st] - pts0[st]
    if src.shape[0] < 4:
        return np.zeros((h, w, 2), np.float32)
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    flow = np.zeros((h, w, 2), np.float32)
    for ch in (0, 1):
        vals = dst[:, ch]
        grid = None
        if hasattr(cv2, "gridData"):
            try:
                grid = cv2.gridData((src.astype(np.float32),), vals.astype(np.float32),
                                    (gx.astype(np.float32), gy.astype(np.float32)),
                                    interpolation=cv2.INTER_CUBIC)
            except Exception:
                grid = None
        if grid is None:
            from .quality import rbf_interpolate
            grid = rbf_interpolate(src, vals, gx, gy)
        flow[..., ch] = grid
    return flow


def tvl1(prev_gray, curr_gray, params: Optional[dict] = None) -> np.ndarray:
    """Dual TV-L1 optical flow (requires opencv-contrib)."""
    try:
        tvl1_ = cv2.optflow.DualTVL1OpticalFlow_create()
    except Exception as exc:
        raise RuntimeError(
            "TV-L1 optical flow requires opencv-contrib-python "
            "(cv2.optflow). Install it with: pip install opencv-contrib-python"
        ) from exc
    if params:
        for k, v in params.items():
            try:
                getattr(tvl1_, "set" + k[0].upper() + k[1:])(v)
            except Exception:
                pass
    return tvl1_.calc(prev_gray, curr_gray, None)


# ---------------------------------------------------------------------
#  Deep-learning backends (lazy torch import)
# ---------------------------------------------------------------------
@dataclass
class _DLCache:
    model: object = None
    name: str = ""
    device: str = "cpu"


_DL_CACHE = _DLCache()


def _load_dl_model(name: str):
    import torch
    from torchvision.models.optical_flow import raft_large, raft_small
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "raft":
        model = raft_large(weights="DEFAULT", progress=False).to(device).eval()
    elif name == "raft_small":
        model = raft_small(weights="DEFAULT", progress=False).to(device).eval()
    else:
        raise ValueError(f"Unknown DL flow backend: {name}")
    _DL_CACHE.model = model
    _DL_CACHE.name = name
    _DL_CACHE.device = device
    return model, device


def _raft_like(prev_gray, curr_gray, name: str = "raft") -> np.ndarray:
    """Run a torchvision RAFT-family model on a grayscale image pair.

    RAFT expects 3-channel images and dimensions divisible by 8. We pad
    with edge replication, replicate the grayscale to 3 channels, run
    inference, then crop the flow back to the original size.
    """
    import torch
    if _DL_CACHE.model is None or _DL_CACHE.name != name:
        _load_dl_model(name)
    model, device = _DL_CACHE.model, _DL_CACHE.device
    h, w = prev_gray.shape[:2]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    a = np.pad(prev_gray, ((0, pad_h), (0, pad_w)), mode="edge")
    b = np.pad(curr_gray, ((0, pad_h), (0, pad_w)), mode="edge")

    def _to_rgb_tensor(g):
        return torch.from_numpy(np.stack([g, g, g], 0)[None].astype(np.float32) / 255.0).to(device)

    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        flow = model(_to_rgb_tensor(a), _to_rgb_tensor(b))[-1]
    flow = flow[0].cpu().numpy()[:h, :w]
    return flow.astype(np.float32)


def raft(prev_gray, curr_gray, params: Optional[dict] = None) -> np.ndarray:
    return _raft_like(prev_gray, curr_gray, "raft")


def pwcnet(prev_gray, curr_gray, params: Optional[dict] = None) -> np.ndarray:
    # PWC-Net availability varies; fall back to RAFT-small if missing.
    try:
        from torchvision.models.optical_flow import pwcrnet  # noqa: F401
    except Exception:
        return _raft_like(prev_gray, curr_gray, "raft_small")
    return _raft_like(prev_gray, curr_gray, "pwcnet")


_BACKENDS: dict[str, Callable] = {
    "farneback": farneback,
    "lucas_kanade": lucas_kanade,
    "tvl1": tvl1,
    "raft": raft,
    "pwcnet": pwcnet,
}


def compute_pair(prev_gray, curr_gray, backend: str = "farneback",
                 params: Optional[dict] = None) -> np.ndarray:
    fn = _BACKENDS.get(backend)
    if fn is None:
        raise ValueError(f"Unknown optical-flow backend: {backend!r}. "
                         f"Available: {available_backends()}")
    return fn(prev_gray, curr_gray, params)


# ---------------------------------------------------------------------
#  Multi-scale pyramid wrapper
# ---------------------------------------------------------------------
def compute_pair_multiscale(prev_gray, curr_gray, backend: str = "farneback",
                            params: Optional[dict] = None,
                            levels: int = 3) -> np.ndarray:
    """Coarse-to-fine pyramid flow estimation.

    Computes flow at the coarsest level, warps the second image towards the
    first, then refines at the next finer level. Falls back to a single
    pass if the backend is already multi-scale (Farneback/RAFT).
    """
    if backend in ("farneback", "raft", "pwcnet") or levels <= 1:
        return compute_pair(prev_gray, curr_gray, backend, params)
    h, w = prev_gray.shape[:2]
    min_dim = min(h, w)
    if min_dim < 64 * (2 ** (levels - 1)):
        return compute_pair(prev_gray, curr_gray, backend, params)
    pyr_prev = [prev_gray]
    pyr_curr = [curr_gray]
    for _ in range(levels - 1):
        pyr_prev.append(cv2.pyrDown(pyr_prev[-1]))
        pyr_curr.append(cv2.pyrDown(pyr_curr[-1]))
    flow = compute_pair(pyr_prev[-1], pyr_curr[-1], backend, params)
    for lvl in range(levels - 2, -1, -1):
        h_l, w_l = pyr_prev[lvl].shape[:2]
        flow = cv2.resize(flow, (w_l, h_l), interpolation=cv2.INTER_CUBIC) * 2.0
        warped = cv2.remap(pyr_curr[lvl],
                           (np.arange(w_l)[None, :] + flow[..., 0]).astype(np.float32),
                           (np.arange(h_l)[:, None] + flow[..., 1]).astype(np.float32),
                           cv2.INTER_LINEAR)
        fine = compute_pair(pyr_prev[lvl], warped, backend, params)
        flow = flow + fine
    return flow


# ---------------------------------------------------------------------
#  Temporal smoothing
# ---------------------------------------------------------------------
def temporal_smooth(stack: np.ndarray, method: str = "ema",
                    alpha: float = 0.6, window: int = 3) -> np.ndarray:
    """Smooth a flow stack (T, H, W, 2) across the time axis.

    methods:
        ema    - exponential moving average (alpha = weight of new frame)
        moving - centered moving average over `window` frames
        none   - passthrough
    """
    if method == "none" or stack.shape[0] < 2:
        return stack
    out = np.empty_like(stack)
    if method == "ema":
        out[0] = stack[0]
        for i in range(1, stack.shape[0]):
            out[i] = alpha * stack[i] + (1 - alpha) * out[i - 1]
    elif method == "moving":
        k = max(1, window)
        for c in range(2):
            out[..., c] = _moving_avg(stack[..., c], k)
    else:
        raise ValueError(f"Unknown temporal smoothing method: {method}")
    return out


def _moving_avg(x: np.ndarray, k: int) -> np.ndarray:
    """Moving average along axis 0 with edge replication."""
    t = x.shape[0]
    out = np.empty_like(x)
    pad = k // 2
    xp = np.concatenate([np.repeat(x[:1], pad, axis=0), x,
                         np.repeat(x[-1:], pad, axis=0)], axis=0)
    for i in range(t):
        out[i] = xp[i:i + k].mean(axis=0)
    return out


# ---------------------------------------------------------------------
#  Quality metrics
# ---------------------------------------------------------------------
def forward_backward_error(prev_gray, curr_gray, flow: np.ndarray) -> np.ndarray:
    """Per-pixel forward-backward consistency error.

    Computes flow from curr->prev and measures the endpoint mismatch.
    Returns an (H, W) array of endpoint errors in pixels.
    """
    fb = cv2.calcOpticalFlowFarneback(curr_gray, prev_gray, None,
                                      pyr_scale=0.3, levels=5, winsize=15,
                                      iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
    h, w = flow.shape[:2]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    fwd = flow
    bx = fb[..., 0]
    by = fb[..., 1]
    err = np.sqrt((fwd[..., 0] + bx) ** 2 + (fwd[..., 1] + by) ** 2)
    return err.astype(np.float32)


def flow_quality_metrics(prev_gray, curr_gray, flow: np.ndarray) -> dict:
    """Return a dict of scalar quality metrics for a flow field."""
    fb_err = forward_backward_error(prev_gray, curr_gray, flow)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return {
        "fb_mean": float(np.mean(fb_err)),
        "fb_median": float(np.median(fb_err)),
        "fb_p95": float(np.percentile(fb_err, 95)),
        "fb_max": float(np.max(fb_err)),
        "mag_mean": float(np.mean(mag)),
        "mag_std": float(np.std(mag)),
        "valid_fraction": float(np.mean(fb_err < 1.0)),
    }


# ---------------------------------------------------------------------
#  Full video processing
# ---------------------------------------------------------------------
@dataclass
class FlowConfig:
    backend: str = "farneback"
    backend_params: dict = field(default_factory=dict)
    multiscale: bool = False
    multiscale_levels: int = 3
    temporal_smoothing: str = "none"      # none|ema|moving
    temporal_alpha: float = 0.6
    temporal_window: int = 3
    enable_gauss: bool = False
    gauss_ksize: int = 5
    gauss_sigma: float = 1.3
    enable_nlm: bool = False
    nlm_h: float = 10.0
    nlm_template: int = 7
    nlm_search: int = 21
    compute_quality: bool = False


def _denoise(gray, cfg: FlowConfig):
    if cfg.enable_gauss:
        gray = cv2.GaussianBlur(gray, (cfg.gauss_ksize, cfg.gauss_ksize), cfg.gauss_sigma)
    if cfg.enable_nlm:
        gray = cv2.fastNlMeansDenoising(gray, None, h=cfg.nlm_h,
                                        templateWindowSize=cfg.nlm_template,
                                        searchWindowSize=cfg.nlm_search)
    return gray


def process_video(video_file: str, roi_box, meters_per_pixel: float,
                  cfg: FlowConfig, mmap_paths: dict,
                  progress_cb: Optional[Callable[[int, int, str], None]] = None,
                  preview_cb: Optional[Callable[[np.ndarray, int], None]] = None
                  ) -> dict:
    """Run dense optical flow over a video ROI and stream results to memmaps.

    mmap_paths must contain 'u', 'v', 'frames' absolute .dat paths.
    Returns a metadata dict (also stored into mmap_paths by caller).
    """
    xa, ya, xb, yb = roi_box
    roi_w, roi_h = xb - xa, yb - ya
    cap = cv2.VideoCapture(video_file)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = max(total - 1, 1)

    u_mmap = np.memmap(mmap_paths["u"], dtype=np.float32, mode="w+",
                       shape=(n_frames, roi_h, roi_w))
    v_mmap = np.memmap(mmap_paths["v"], dtype=np.float32, mode="w+",
                       shape=(n_frames, roi_h, roi_w))
    frame_mmap = np.memmap(mmap_paths["frames"], dtype=np.uint8, mode="w+",
                           shape=(n_frames, roi_h, roi_w))

    ret, prev_full = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read first frame from video.")
    prev_gray = _denoise(cv2.cvtColor(prev_full[ya:yb, xa:xb], cv2.COLOR_BGR2GRAY), cfg)

    quality_log = []
    raw_stack = []  # only kept if temporal smoothing requested
    do_smooth = cfg.temporal_smoothing != "none"

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            n_frames = i
            break
        curr_gray = _denoise(cv2.cvtColor(frame[ya:yb, xa:xb], cv2.COLOR_BGR2GRAY), cfg)
        if cfg.multiscale:
            flow = compute_pair_multiscale(prev_gray, curr_gray, cfg.backend,
                                           cfg.backend_params, cfg.multiscale_levels)
        else:
            flow = compute_pair(prev_gray, curr_gray, cfg.backend, cfg.backend_params)
        if do_smooth:
            raw_stack.append(flow.copy())
        else:
            u_mmap[i] = flow[..., 0] * meters_per_pixel * fps
            v_mmap[i] = flow[..., 1] * meters_per_pixel * fps
        frame_mmap[i] = curr_gray
        frame_mmap.flush()
        prev_gray = curr_gray.copy()

        if cfg.compute_quality and (i % max(1, n_frames // 30) == 0):
            quality_log.append(flow_quality_metrics(prev_gray, curr_gray, flow))

        if preview_cb and (i % max(1, n_frames // 40) == 0 or i == n_frames - 1):
            preview_cb(_make_preview(curr_gray, flow), i)

        if progress_cb and (i % 3 == 0 or i == n_frames - 1):
            progress_cb(i + 1, n_frames, "optical flow")

    cap.release()

    if do_smooth and raw_stack:
        stack = np.stack(raw_stack, axis=0)
        sm = temporal_smooth(stack, cfg.temporal_smoothing,
                             cfg.temporal_alpha, cfg.temporal_window)
        for i in range(sm.shape[0]):
            u_mmap[i] = sm[i, ..., 0] * meters_per_pixel * fps
            v_mmap[i] = sm[i, ..., 1] * meters_per_pixel * fps

    u_mmap.flush()
    v_mmap.flush()
    frame_mmap.flush()
    return {
        "frames": n_frames, "roi_h": roi_h, "roi_w": roi_w, "FPS": fps,
        "backend": cfg.backend, "multiscale": cfg.multiscale,
        "temporal_smoothing": cfg.temporal_smoothing,
        "quality_log": quality_log,
    }


def _make_preview(gray: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Compose an HSV + quiver preview frame (BGR) for live display."""
    from .theme import Theme
    rgb_disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(rgb_disp)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    preview = cv2.addWeighted(rgb_disp, 0.5, bgr, 0.5, 0).astype(np.uint8)
    h, w = gray.shape[:2]
    grid_step = max(h // 25, 2)
    ygrid, xgrid = np.mgrid[0:h:grid_step, 0:w:grid_step]
    uf = flow[..., 0][ygrid, xgrid]
    vf = flow[..., 1][ygrid, xgrid]
    for (x, y, du, dv) in zip(xgrid.ravel(), ygrid.ravel(), uf.ravel(), vf.ravel()):
        tip = (int(round(x + du * 2)), int(round(y + dv * 2)))
        cv2.arrowedLine(preview, (int(x), int(y)), tip, Theme.BGR_MAGENTA, 1, tipLength=0.28)
    return preview
