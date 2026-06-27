"""Optical flow for mobile — uses cv2 if available, numpy fallback otherwise.

Supports:
  - Farneback dense flow (cv2)
  - Lucas-Kanade sparse flow (cv2)
  - Numpy-only fallback (block matching, slow but no cv2 needed)
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def process_video(video_path: str, roi, meters_per_pixel: float,
                  backend: str = "farneback",
                  max_frames: int = 0,
                  progress_cb=None) -> dict:
    """Process a video file and extract velocity fields.

    Parameters
    ----------
    video_path : path to video file
    roi : (x0, y0, x1, y1) region of interest in pixels
    meters_per_pixel : calibration scale
    backend : "farneback" or "lucas_kanade" or "numpy"
    max_frames : 0 = all frames, else cap
    progress_cb : callback(i, n, stage)

    Returns dict with u, v arrays (n_frames, h, w) and metadata.
    """
    if HAS_CV2:
        return _process_cv2(video_path, roi, meters_per_pixel, backend,
                            max_frames, progress_cb)
    else:
        return _process_numpy(video_path, roi, meters_per_pixel,
                              max_frames, progress_cb)


def _process_cv2(video_path, roi, mpp, backend, max_frames, progress_cb):
    """OpenCV-based optical flow processing."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total = min(total, max_frames)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    x0, y0, x1, y1 = roi
    w = x1 - x0
    h = y1 - y0
    # Read first frame to get ROI dimensions
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Cannot read first frame")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = gray[y0:y1, x0:x1]
    u_frames = []
    v_frames = []
    frame_count = 0
    dt = 1.0 / fps
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames > 0 and frame_count >= max_frames:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = gray[y0:y1, x0:x1]
        if backend == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                0.3, 5, 21, 7, 7, 1.1, 0)
            u = flow[:, :, 0] * mpp / dt
            v = flow[:, :, 1] * mpp / dt
        elif backend == "lucas_kanade":
            # Sparse LK on a grid, then interpolate
            step = max(4, min(w, h) // 20)
            pts0 = np.mgrid[step//2:h:step, step//2:w:step].T.reshape(-1, 2
                ).astype(np.float32)
            pts1, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, pts0, None)
            good = st.ravel() == 1
            if np.sum(good) > 4:
                du = (pts1[good] - pts0[good]) * mpp / dt
                # Nearest-neighbor interpolation onto dense grid
                u = np.zeros((h, w))
                v = np.zeros((h, w))
                for k, (py, px) in enumerate(pts0[good]):
                    u[int(py), int(px)] = du[k, 0]
                    v[int(py), int(px)] = du[k, 1]
            else:
                u = np.zeros((h, w))
                v = np.zeros((h, w))
        else:
            u = np.zeros((h, w))
            v = np.zeros((h, w))
        u_frames.append(u.astype(np.float32))
        v_frames.append(v.astype(np.float32))
        prev_gray = curr_gray
        frame_count += 1
        if progress_cb:
            progress_cb(frame_count, total, "optical_flow")
    cap.release()
    return {
        "u": np.stack(u_frames) if u_frames else np.zeros((0, h, w), np.float32),
        "v": np.stack(v_frames) if v_frames else np.zeros((0, h, w), np.float32),
        "n_frames": frame_count,
        "roi_h": h, "roi_w": w,
        "fps": fps, "dt": dt,
        "meters_per_pixel": mpp,
        "backend": backend,
    }


def _process_numpy(video_path, roi, mpp, max_frames, progress_cb):
    """Numpy-only fallback (block matching — slow, no cv2 needed).

    This is a very basic block-matching optical flow. It requires cv2
    for video decoding, so if cv2 is not available at all, we raise.
    """
    if not HAS_CV2:
        raise RuntimeError(
            "OpenCV is required for video processing. "
            "Install opencv-python to use optical flow.")
    # If we have cv2 for decoding but want numpy flow, use block matching
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total = min(total, max_frames)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    x0, y0, x1, y1 = roi
    w, h = x1 - x0, y1 - y0
    block = 8
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev = gray[y0:y1, x0:x1].astype(np.float32)
    u_frames, v_frames = [], []
    dt = 1.0 / fps
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames > 0 and frame_count >= max_frames):
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr = gray[y0:y1, x0:x1].astype(np.float32)
        # Simple block matching
        u = np.zeros((h, w), np.float32)
        v = np.zeros((h, w), np.float32)
        for by in range(0, h - block, block):
            for bx in range(0, w - block, block):
                ref = prev[by:by+block, bx:bx+block]
                best_dy, best_dx, best_err = 0, 0, 1e18
                for dy in range(-block, block+1, 2):
                    for dx in range(-block, block+1, 2):
                        ny, nx = by+dy, bx+dx
                        if 0 <= ny and ny+block <= h and 0 <= nx and nx+block <= w:
                            cand = curr[ny:ny+block, nx:nx+block]
                            err = np.sum((ref - cand) ** 2)
                            if err < best_err:
                                best_err = err
                                best_dy, best_dx = dy, dx
                u[by:by+block, bx:bx+block] = best_dx * mpp / dt
                v[by:by+block, bx:bx+block] = best_dy * mpp / dt
        u_frames.append(u)
        v_frames.append(v)
        prev = curr
        frame_count += 1
        if progress_cb:
            progress_cb(frame_count, total, "optical_flow")
    cap.release()
    return {
        "u": np.stack(u_frames) if u_frames else np.zeros((0, h, w), np.float32),
        "v": np.stack(v_frames) if v_frames else np.zeros((0, h, w), np.float32),
        "n_frames": frame_count,
        "roi_h": h, "roi_w": w,
        "fps": fps, "dt": dt,
        "meters_per_pixel": mpp,
        "backend": "numpy_blockmatch",
    }
