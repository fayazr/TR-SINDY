"""Data quality control: outlier detection, optical-flow consistency
checks, and interpolation of missing/invalid values.

Methods:
    * Statistical outlier flagging (z-score, modified z-score, IQR)
    * Forward-backward consistency masking
    * Interpolation: RBF, kriging (via scipy/pykrige if available), nearest
    * Noise estimation
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage as _ndi
from scipy.interpolate import Rbf

from ._logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------
#  Outlier detection
# ---------------------------------------------------------------------
def zscore_outliers(field: np.ndarray, thresh: float = 4.0) -> np.ndarray:
    """Boolean mask of z-score outliers (|z| > thresh)."""
    mu = np.nanmean(field)
    sigma = np.nanstd(field)
    if sigma < 1e-30:
        return np.zeros_like(field, dtype=bool)
    z = np.abs((field - mu) / sigma)
    return z > thresh


def modified_zscore_outliers(field: np.ndarray, thresh: float = 3.5) -> np.ndarray:
    """Modified z-score using the median absolute deviation (robust)."""
    med = np.nanmedian(field)
    mad = np.nanmedian(np.abs(field - med))
    if mad < 1e-30:
        return np.zeros_like(field, dtype=bool)
    modz = 0.6745 * (field - med) / mad
    return np.abs(modz) > thresh


def iqr_outliers(field: np.ndarray, k: float = 1.5) -> np.ndarray:
    """IQR-based outlier mask."""
    q1, q3 = np.nanpercentile(field, 25), np.nanpercentile(field, 75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (field < lo) | (field > hi)


def detect_outliers(field: np.ndarray, method: str = "modz",
                    thresh: float = 3.5) -> np.ndarray:
    if method == "zscore":
        return zscore_outliers(field, thresh)
    if method == "modz":
        return modified_zscore_outliers(field, thresh)
    if method == "iqr":
        return iqr_outliers(field, thresh)
    raise ValueError(f"Unknown outlier method: {method!r}")


def replace_outliers(field: np.ndarray, mask: np.ndarray,
                     method: str = "linear") -> np.ndarray:
    """Replace masked values via 2-D interpolation."""
    out = field.copy().astype(np.float64)
    good = ~mask
    if not np.any(good):
        return out
    if method in ("linear", "cubic", "nearest"):
        idx = _ndi.distance_transform_edt(mask, return_distances=False, return_indices=True)
        out[mask] = field[tuple(idx[:, mask])]
    elif method == "rbf":
        ys, xs = np.mgrid[0:field.shape[0], 0:field.shape[1]]
        gy, gx = ys[good], xs[good]
        rbf = Rbf(gx, gy, field[good], function="thin_plate")
        out[mask] = rbf(xs[mask], ys[mask])
    else:
        raise ValueError(f"Unknown interpolation method: {method!r}")
    return out


# ---------------------------------------------------------------------
#  Interpolation helpers
# ---------------------------------------------------------------------
def rbf_interpolate(points: np.ndarray, values: np.ndarray,
                    gx: np.ndarray, gy: np.ndarray,
                    function: str = "thin_plate") -> np.ndarray:
    """Radial-basis-function interpolation from scattered points to a grid."""
    rbf = Rbf(points[:, 0], points[:, 1], values, function=function)
    return rbf(gx, gy).astype(np.float32)


def kriging_interpolate(points: np.ndarray, values: np.ndarray,
                        gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Ordinary kriging via pykrige if available, else RBF fallback."""
    try:
        from pykrige.ok import OrdinaryKriging
        ok = OrdinaryKriging(points[:, 0], points[:, 1], values,
                             variogram_model="linear")
        z, _ = ok.execute("grid", gx[0, :], gy[:, 0])
        return z.astype(np.float32)
    except Exception as e:
        log.debug("kriging failed, falling back to RBF: %s", e)
        return rbf_interpolate(points, values, gx, gy)


def fill_nans(field: np.ndarray, method: str = "linear") -> np.ndarray:
    """Fill NaNs in a 2-D field via interpolation."""
    mask = np.isnan(field)
    if not np.any(mask):
        return field
    return replace_outliers(field, mask, method)


# ---------------------------------------------------------------------
#  Noise estimation
# ---------------------------------------------------------------------
def estimate_noise(field: np.ndarray) -> dict:
    """Estimate noise level via the median absolute deviation of the
    high-pass-filtered field (Laplacian)."""
    lap = _ndi.laplace(field.astype(np.float64))
    sigma = np.median(np.abs(lap)) / 0.6745
    return {"noise_sigma": float(sigma),
            "snr_db": float(10 * np.log10(np.var(field) / (sigma ** 2 + 1e-30)))}


# ---------------------------------------------------------------------
#  Flow-field quality mask
# ---------------------------------------------------------------------
def flow_quality_mask(flow: np.ndarray, fb_error: np.ndarray,
                      fb_thresh: float = 1.0,
                      mag_thresh: Optional[float] = None) -> np.ndarray:
    """Boolean mask of *valid* flow pixels (low fb error, finite, optional
    magnitude floor)."""
    valid = np.isfinite(flow).all(axis=-1) & (fb_error < fb_thresh)
    if mag_thresh is not None:
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        valid &= mag > mag_thresh
    return valid
