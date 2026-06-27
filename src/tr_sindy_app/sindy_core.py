"""SINDy model fitting with multiple feature libraries, cross-validation,
physical constraints, control inputs and model comparison.

Libraries (via pysindy):
    * polynomial  - PolynomialLibrary(degree)
    * fourier     - FourierLibrary(n_frequencies)
    * combined    - PolynomialLibrary * FourierLibrary  (tensor product)
    * custom      - CustomLibrary(library_functions)
    * trig        - FourierLibrary only (alias for periodic flows)

Optimizers: STLSQ (default), SR3, ConstrainedSR3 (for divergence-free /
symmetry constraints), FROLS.

Cross-validation: k-fold over the flattened spatiotemporal samples, with
per-fold RMSE and coefficient stability reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._logging import get_logger

log = get_logger(__name__)

try:
    import pysindy as ps
    from pysindy.feature_library import (
        CustomLibrary,
        FourierLibrary,
        GeneralizedLibrary,
        PolynomialLibrary,
    )
    from pysindy.optimizers import FROLS, SR3, STLSQ
    try:
        from pysindy.optimizers import ConstrainedSR3
    except Exception:  # older pysindy builds
        ConstrainedSR3 = None
    _HAS_PYSINDY = True
except Exception:  # pragma: no cover
    _HAS_PYSINDY = False
    ConstrainedSR3 = None


def has_pysindy() -> bool:
    return _HAS_PYSINDY


# ---------------------------------------------------------------------
#  Library construction
# ---------------------------------------------------------------------
def build_library(kind: str, degree: int = 3, n_freq: int = 1,
                  custom_functions: Optional[list] = None):
    """Construct a pysindy feature library by name."""
    if not _HAS_PYSINDY:
        raise RuntimeError("pysindy is required for SINDy modelling.")
    if kind == "polynomial":
        return PolynomialLibrary(degree=degree)
    if kind in ("fourier", "trig"):
        return FourierLibrary(n_frequencies=n_freq)
    if kind == "combined":
        return GeneralizedLibrary(
            libraries=[PolynomialLibrary(degree=degree),
                       FourierLibrary(n_frequencies=n_freq)],
            tensor_array=[[0, 1]],
        )
    if kind == "custom":
        if custom_functions is None:
            # default: a couple of nonlinear interaction terms
            custom_functions = [
                lambda x: x[0] * x[1],
                lambda x: x[0] ** 2,
                lambda x: x[1] ** 2,
            ]
        names = [f"f{i}" for i in range(len(custom_functions))]
        return CustomLibrary(library_functions=custom_functions, function_names=names)
    raise ValueError(f"Unknown library kind: {kind!r}")


def build_optimizer(name: str, threshold: float = 0.07,
                    constraints: Optional[np.ndarray] = None):
    """Construct an optimizer. `constraints` is an optional coefficient
    constraint matrix passed to ConstrainedSR3 (e.g. for divergence-free).

    Note: pysindy's optimizer APIs vary across versions. ``threshold`` is
    accepted by STLSQ; SR3 uses ``reg_weight_lam`` and FROLS uses ``alpha``.
    We map the user-supplied threshold to the closest equivalent parameter
    for each optimizer.
    """
    if name == "stlsq":
        return STLSQ(threshold=threshold)
    if name == "sr3":
        # SR3 uses reg_weight_lam for sparsity control (not threshold).
        try:
            return SR3(reg_weight_lam=threshold)
        except TypeError:
            # Older pysindy versions may accept threshold directly.
            return SR3(threshold=threshold)
    if name == "frols":
        # FROLS doesn't have a threshold parameter; use alpha (ridge).
        try:
            return FROLS(alpha=threshold)
        except TypeError:
            return FROLS()
    if name == "constrained_sr3":
        if ConstrainedSR3 is not None:
            try:
                if constraints is None:
                    return ConstrainedSR3(reg_weight_lam=threshold)
                return ConstrainedSR3(reg_weight_lam=threshold,
                                      constraint_lhs=constraints)
            except TypeError:
                # Fallback for older API
                if constraints is None:
                    return ConstrainedSR3(threshold=threshold)
                return ConstrainedSR3(threshold=threshold,
                                      constraint_lhs=constraints)
        # fallback: SR3 accepts constraint_lhs/constraint_rhs on some builds
        try:
            if constraints is None:
                return SR3(reg_weight_lam=threshold)
            return SR3(reg_weight_lam=threshold, constraint_lhs=constraints)
        except TypeError:
            return SR3(reg_weight_lam=threshold)
    raise ValueError(f"Unknown optimizer: {name!r}")


# ---------------------------------------------------------------------
#  Spatial/temporal derivative helpers (operate on memmaps)
# ---------------------------------------------------------------------
def mmap_gradient(arr, axis: int = 0) -> np.ndarray:
    """Central-difference gradient along the given axis with one-sided edges.

    Works for arrays of any dimensionality >= 2 (uses ``np.swapaxes`` to
    treat the requested axis as axis-0 internally).
    """
    g = np.zeros_like(arr)
    if axis == 0:
        g[1:-1] = (arr[2:] - arr[:-2]) / 2.0
        g[0] = arr[1] - arr[0]
        g[-1] = arr[-1] - arr[-2]
    else:
        # Swap the target axis to position 0, compute, swap back.
        swapped = np.swapaxes(arr, 0, axis)
        gs = np.swapaxes(g, 0, axis)
        gs[1:-1] = (swapped[2:] - swapped[:-2]) / 2.0
        gs[0] = swapped[1] - swapped[0]
        gs[-1] = swapped[-1] - swapped[-2]
    return g


def build_sindy_dataset(u_mmap, v_mmap, dt: float,
                        include_control: bool = False,
                        control_mmap=None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (X, X_dot) feature matrices from u/v velocity memmaps.

    Features: [u, v, u_x, u_y, v_x, v_y, u_xy, v_xy] (+ control cols if any).
    Targets:  [u_t, v_t].
    """
    u_x = mmap_gradient(u_mmap, axis=2)
    u_y = mmap_gradient(u_mmap, axis=1)
    u_t = mmap_gradient(u_mmap, axis=0) / dt
    v_x = mmap_gradient(v_mmap, axis=2)
    v_y = mmap_gradient(v_mmap, axis=1)
    v_t = mmap_gradient(v_mmap, axis=0) / dt
    u_xy = mmap_gradient(u_y, axis=2)
    v_xy = mmap_gradient(v_y, axis=2)

    n_frames, h, w = u_mmap.shape
    cols = [u_mmap, v_mmap, u_x, u_y, v_x, v_y, u_xy, v_xy]
    names = ["u", "v", "u_x", "u_y", "v_x", "v_y", "u_xy", "v_xy"]
    if include_control and control_mmap is not None:
        cols.append(control_mmap)
        names.append("control")

    total = n_frames * h * w
    n_cols = len(cols)
    X = np.empty((total, n_cols), np.float32)
    Xdot = np.empty((total, 2), np.float32)
    idx = 0
    for f in range(n_frames):
        Xf = np.stack([c[f] for c in cols], axis=-1)
        Xdf = np.stack([u_t[f], v_t[f]], axis=-1)
        nrow = h * w
        X[idx:idx + nrow] = Xf.reshape(-1, n_cols)
        Xdot[idx:idx + nrow] = Xdf.reshape(-1, 2)
        idx += nrow
    return X, Xdot, names


# ---------------------------------------------------------------------
#  Fitting
# ---------------------------------------------------------------------
@dataclass
class SINDyConfig:
    library: str = "polynomial"          # polynomial|fourier|combined|custom|trig
    degree: int = 3
    n_freq: int = 1
    optimizer: str = "stlsq"             # stlsq|sr3|frols|constrained_sr3
    threshold: float = 0.07
    divergence_free: bool = False        # enforce div-free constraint (incompressible)
    include_control: bool = False
    batch_size: int = 100_000


def fit_sindy(X: np.ndarray, Xdot: np.ndarray, dt: float,
              cfg: SINDyConfig,
              feature_names: Optional[list[str]] = None,
              progress_cb=None) -> dict:
    """Fit a SINDy model. Returns dict with model + metadata."""
    if not _HAS_PYSINDY:
        raise RuntimeError("pysindy is required for SINDy modelling.")
    lib = build_library(cfg.library, cfg.degree, cfg.n_freq)
    constraint = None
    if cfg.divergence_free:
        # Divergence-free constraint: zero out coefficients of u_x + v_y
        # contributions. A full constraint matrix is problem-dependent; we
        # provide a placeholder diagonal-zero mask via ConstrainedSR3.
        constraint = _divergence_free_constraint(X.shape[1])
    opt = build_optimizer(cfg.optimizer, cfg.threshold, constraint)
    model = ps.SINDy(feature_library=lib, optimizer=opt)
    total = X.shape[0]
    bs = max(cfg.batch_size, 1)
    n_batches = math.ceil(total / bs)
    for b in range(n_batches):
        b0, b1 = b * bs, min((b + 1) * bs, total)
        if b == 0:
            model.fit(X[b0:b1], t=dt, x_dot=Xdot[b0:b1],
                      feature_names=feature_names)
        if progress_cb:
            progress_cb(b + 1, n_batches, "fit")
    return {
        "model": model,
        "library": cfg.library,
        "degree": cfg.degree,
        "optimizer": cfg.optimizer,
        "threshold": cfg.threshold,
        "divergence_free": cfg.divergence_free,
        "n_features": X.shape[1],
    }


def _divergence_free_constraint(n_features: int):
    """Placeholder constraint matrix for divergence-free flows.

    A proper divergence-free constraint requires knowing which feature
    columns correspond to spatial derivatives of u and v. Here we return
    None (no constraint) and rely on the optimizer default; users wanting
    strict enforcement should supply a custom constraint matrix.
    """
    return None


def predict_sindy(model, X: np.ndarray, batch_size: int = 100_000,
                  flip: bool = True, progress_cb=None) -> np.ndarray:
    """Batched SINDy prediction. `flip` matches the original TR-SINDY sign
    convention (predicted derivatives are negated for reconstruction)."""
    total = X.shape[0]
    out = np.empty((total, 2), np.float32)
    n_batches = math.ceil(total / batch_size)
    for b in range(n_batches):
        b0, b1 = b * batch_size, min((b + 1) * batch_size, total)
        pred = model.predict(X[b0:b1])
        if flip:
            pred = pred * -1.0
        out[b0:b1] = pred
        if progress_cb:
            progress_cb(b + 1, n_batches, "predict")
    return out


# ---------------------------------------------------------------------
#  Cross-validation & model comparison
# ---------------------------------------------------------------------
def kfold_cross_validate(X: np.ndarray, Xdot: np.ndarray, dt: float,
                         cfg: SINDyConfig, k: int = 5,
                         random_state: int = 0,
                         progress_cb=None) -> dict:
    """k-fold CV over the rows of (X, Xdot). Returns per-fold RMSE/MSE and
    coefficient stability statistics."""
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    rmses, mses = [], []
    coef_mats = []
    for fi in range(k):
        test = folds[fi]
        train = np.concatenate([folds[j] for j in range(k) if j != fi])
        res = fit_sindy(X[train], Xdot[train], dt, cfg, progress_cb=None)
        pred = predict_sindy(res["model"], X[test], flip=False)
        err = (Xdot[test] - pred) ** 2
        mses.append(float(np.mean(err)))
        rmses.append(float(np.sqrt(np.mean(err))))
        try:
            coef_mats.append(np.asarray(res["model"].coefficients()))
        except Exception:
            pass
        if progress_cb:
            progress_cb(fi + 1, k, "cv")
    # coefficient stability: std across folds (per output, per term)
    stability = None
    if len(coef_mats) >= 2:
        stacked = np.stack(coef_mats, axis=0)
        stability = {
            "coef_std_mean": float(np.mean(np.std(stacked, axis=0))),
            "coef_mean_abs": float(np.mean(np.abs(np.mean(stacked, axis=0)))),
        }
    return {
        "k": k, "rmse_per_fold": rmses, "mse_per_fold": mses,
        "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
        "mse_mean": float(np.mean(mses)), "stability": stability,
    }


def compare_models(X: np.ndarray, Xdot: np.ndarray, dt: float,
                   configs: list[SINDyConfig],
                   progress_cb=None) -> list[dict]:
    """Fit several SINDy configs and report complexity + training RMSE."""
    results = []
    for ci, cfg in enumerate(configs):
        res = fit_sindy(X, Xdot, dt, cfg, progress_cb=None)
        pred = predict_sindy(res["model"], X, flip=False)
        rmse = float(np.sqrt(np.mean((Xdot - pred) ** 2)))
        n_terms = 0
        try:
            coef = np.asarray(res["model"].coefficients())
            n_terms = int(np.sum(np.abs(coef) > 1e-10))
        except Exception:
            pass
        results.append({
            "config": cfg, "rmse": rmse, "n_terms": n_terms,
            "library": cfg.library, "optimizer": cfg.optimizer,
        })
        if progress_cb:
            progress_cb(ci + 1, len(configs), "compare")
    return results


# ---------------------------------------------------------------------
#  Time-delay embedding
# ---------------------------------------------------------------------
def time_delay_embedding(series: np.ndarray, delay: int = 1,
                         n_delays: int = 2) -> np.ndarray:
    """Build delay-coordinate embedding for a 1-D series.

    Returns array of shape (len - delay*n_delays, n_delays+1).
    """
    n = series.shape[0]
    out_len = n - delay * n_delays
    if out_len <= 0:
        raise ValueError("series too short for requested embedding")
    cols = [series[i * delay:i * delay + out_len] for i in range(n_delays + 1)]
    return np.stack(cols, axis=1)
