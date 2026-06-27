"""Flow-field analysis: vorticity, strain, spectral analysis, POD, DMD,
turbulence statistics, comprehensive error metrics and region analysis.

All functions accept numpy arrays and require only numpy/scipy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import signal as _signal

from ._logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------
#  Kinematic fields
# ---------------------------------------------------------------------
def vorticity(u: np.ndarray, v: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """Out-of-plane vorticity omega_z = dv/dx - du/dy."""
    dv_dx = np.gradient(v, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    return dv_dx - du_dy


def strain_rate(u: np.ndarray, v: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> dict:
    """Symmetric strain-rate tensor components + scalar invariants."""
    du_dx = np.gradient(u, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    Sxx = du_dx
    Syy = dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    shear = 0.5 * (du_dy - dv_dx)  # rotation half
    mag = np.sqrt(2 * Sxx ** 2 + 2 * Syy ** 2 + 4 * Sxy ** 2)
    return {"Sxx": Sxx, "Syy": Syy, "Sxy": Sxy, "shear": shear,
            "magnitude": mag, "du_dx": du_dx, "du_dy": du_dy,
            "dv_dx": dv_dx, "dv_dy": dv_dy}


def divergence(u: np.ndarray, v: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    return np.gradient(u, dx, axis=1) + np.gradient(v, dy, axis=0)


def velocity_magnitude(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(u ** 2 + v ** 2)


# ---------------------------------------------------------------------
#  Spectral analysis
# ---------------------------------------------------------------------
def spatial_spectrum(field: np.ndarray) -> dict:
    """2-D FFT energy spectrum of a scalar field.

    Returns the radial-averaged spectrum (k, E(k)) and the full 2-D power.
    """
    F = np.fft.fft2(field)
    power = np.abs(F) ** 2
    h, w = field.shape[:2]
    ky = np.fft.fftfreq(h)
    kx = np.fft.fftfreq(w)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    kmax = min(h, w) // 2
    k_bins = np.linspace(0, np.max(K), kmax + 1)
    k_centres = 0.5 * (k_bins[:-1] + k_bins[1:])
    E = np.zeros_like(k_centres)
    for i in range(len(k_centres)):
        mask = (K >= k_bins[i]) & (K < k_bins[i + 1])
        if np.any(mask):
            E[i] = power[mask].mean()
    return {"k": k_centres, "E": E, "power2d": power}


def temporal_spectrum(series: np.ndarray, fs: float = 1.0) -> dict:
    """1-D power spectral density of a time series via Welch's method."""
    f, pxx = _signal.welch(series, fs=fs, nperseg=min(len(series), 256))
    return {"freq": f, "psd": pxx}


# ---------------------------------------------------------------------
#  Proper Orthogonal Decomposition (POD)
# ---------------------------------------------------------------------
def pod_decompose(stack: np.ndarray, n_modes: int = 10) -> dict:
    """Snapshot POD of a (T, H, W) or (T, H, W, C) stack.

    Returns modes (spatial), coefficients (temporal), energies and the
    cumulative energy fraction.
    """
    T = stack.shape[0]
    flat = stack.reshape(T, -1)
    flat = flat - flat.mean(axis=0, keepdims=True)
    # snapshot method: eig of T x T Gram matrix
    C = flat @ flat.T / T
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    n_modes = min(n_modes, T)
    # spatial modes
    modes = (flat.T @ eigvecs[:, :n_modes]).T.reshape(n_modes, *stack.shape[1:])
    coeffs = eigvecs[:, :n_modes] * np.sqrt(eigvals[:n_modes])
    energies = eigvals[:n_modes]
    energies = np.maximum(energies, 0)
    cum = np.cumsum(energies) / (energies.sum() + 1e-30)
    return {"modes": modes, "coefficients": coeffs, "energies": energies,
            "cumulative": cum, "n_modes": n_modes}


# ---------------------------------------------------------------------
#  Dynamic Mode Decomposition (DMD)
# ---------------------------------------------------------------------
def dmd_decompose(stack: np.ndarray, n_modes: int = 10, svd_thresh: float = 1e-6) -> dict:
    """Standard DMD on a (T, N) flattened stack (or (T,H,W) which we flatten).

    Returns modes, eigenvalues (continuous-time), amplitudes and the
    reconstructed approximation.
    """
    T = stack.shape[0]
    X = stack.reshape(T, -1).astype(np.float64)
    X1 = X[:-1].T
    X2 = X[1:].T
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    keep = max(1, min(n_modes, int(np.sum(S > svd_thresh * S[0]))))
    U = U[:, :keep]
    S = S[:keep]
    V = Vh[:keep].conj().T
    A_tilde = U.conj().T @ X2 @ V / S
    eigvals, eigvecs = np.linalg.eig(A_tilde)
    modes = X2 @ V @ np.diag(1.0 / S) @ eigvecs
    # continuous-time eigenvalues assuming unit dt
    ct_eig = np.log(eigvals + 1e-30)
    amps = np.linalg.lstsq(modes, X1[:, 0], rcond=None)[0]
    return {"modes": modes, "eigenvalues": eigvals, "ct_eigenvalues": ct_eig,
            "amplitudes": amps, "n_modes": keep}


# ---------------------------------------------------------------------
#  Turbulence statistics
# ---------------------------------------------------------------------
def turbulence_statistics(u: np.ndarray, v: np.ndarray, nu: float = 1e-6,
                          length_scale: float = 1.0) -> dict:
    """Compute common turbulence statistics for a velocity field."""
    mag = velocity_magnitude(u, v)
    u_mean = float(np.mean(u))
    v_mean = float(np.mean(v))
    u_rms = float(np.sqrt(np.mean((u - u_mean) ** 2)))
    v_rms = float(np.sqrt(np.mean((v - v_mean) ** 2)))
    tke = 0.5 * (u_rms ** 2 + v_rms ** 2)
    u_rms_tot = np.sqrt(u_rms ** 2 + v_rms ** 2)
    re = u_rms_tot * length_scale / max(nu, 1e-30)
    # Reynolds stress component
    uv = float(np.mean((u - u_mean) * (v - v_mean)))
    # turbulence intensity
    intensity = u_rms_tot / (np.mean(mag) + 1e-30)
    return {
        "u_mean": u_mean, "v_mean": v_mean, "u_rms": u_rms, "v_rms": v_rms,
        "tke": tke, "reynolds_number": float(re), "reynolds_stress_uv": uv,
        "turbulence_intensity": float(intensity),
        "mean_magnitude": float(np.mean(mag)),
        "max_magnitude": float(np.max(mag)),
    }


def kinetic_energy(u: np.ndarray, v: np.ndarray, rho: float = 1.0) -> np.ndarray:
    """Per-pixel kinetic energy density 0.5 * rho * (u^2 + v^2)."""
    return 0.5 * rho * (u ** 2 + v ** 2)


def structure_function(field: np.ndarray, max_sep: int = 10) -> dict:
    """Second-order longitudinal structure function S2(r)."""
    n = min(max_sep, min(field.shape) // 2)
    s2 = np.zeros(n)
    for r in range(1, n + 1):
        diff = field[:, r:] - field[:, :-r]
        s2[r - 1] = np.mean(diff ** 2)
    return {"r": np.arange(1, n + 1), "S2": s2}


def structure_function_scaling(field: np.ndarray, max_sep: int = 10,
                               order: int = 2) -> dict:
    """Structure function S_p(r) of arbitrary order + scaling exponent.

    Computes the p-th order structure function S_p(r) = <|u(x+r) - u(x)|^p>
    and fits a power-law scaling exponent zeta_p from S_p ~ r^{zeta_p}.

    For homogeneous isotropic turbulence, Kolmogorov K41 predicts:
        zeta_2 = 2/3  (inertial range)
        zeta_3 = 1    (exact 4/5 law)

    Returns dict with r, S_p, zeta (fitted exponent), and r_squared (fit quality).
    """
    n = min(max_sep, min(field.shape) // 2)
    sp = np.zeros(n)
    for r in range(1, n + 1):
        diff = np.abs(field[:, r:] - field[:, :-r])
        sp[r - 1] = np.mean(diff ** order)
    r_arr = np.arange(1, n + 1, dtype=float)
    # Fit log(S_p) = zeta * log(r) + c in the inertial range (skip r=1 boundary)
    if n >= 4:
        log_r = np.log(r_arr[1:])
        log_sp = np.log(sp[1:] + 1e-30)
        # linear regression
        coeffs = np.polyfit(log_r, log_sp, 1)
        zeta = float(coeffs[0])
        # R^2
        pred = np.polyval(coeffs, log_r)
        ss_res = np.sum((log_sp - pred) ** 2)
        ss_tot = np.sum((log_sp - np.mean(log_sp)) ** 2) + 1e-30
        r_squared = float(1 - ss_res / ss_tot)
    else:
        zeta = float("nan")
        r_squared = 0.0
    return {"r": r_arr, f"S{order}": sp, "zeta": zeta, "r_squared": r_squared}


def energy_spectrum(u: np.ndarray, v: np.ndarray,
                    dx: float = 1.0, dy: float = 1.0) -> dict:
    """Isotropic kinetic energy spectrum E(k) from a 2-D velocity field.

    Computes the 2-D FFT of the velocity field, forms the kinetic energy
    |u_hat|^2 + |v_hat|^2, and radially averages to get E(k).

    Returns dict with wavenumber k (in cycles per unit length), energy E(k),
    and the fitted Kolmogorov scaling exponent (E ~ k^{-5/3} in inertial range).
    """
    h, w = u.shape[:2]
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    energy_2d = np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2
    kx = np.fft.fftfreq(w, d=dx)
    ky = np.fft.fftfreq(h, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    kmax = min(h, w) // 2
    k_bins = np.linspace(0, np.max(K), kmax + 1)
    k_centres = 0.5 * (k_bins[:-1] + k_bins[1:])
    E = np.zeros_like(k_centres)
    for i in range(len(k_centres)):
        mask = (K >= k_bins[i]) & (K < k_bins[i + 1])
        if np.any(mask):
            E[i] = energy_2d[mask].mean()
    # Fit Kolmogorov exponent in log-log space (skip zero-energy bins)
    valid = (k_centres > 0) & (E > 0)
    if np.sum(valid) >= 4:
        log_k = np.log(k_centres[valid])
        log_e = np.log(E[valid])
        coeffs = np.polyfit(log_k, log_e, 1)
        exponent = float(coeffs[0])
        pred = np.polyval(coeffs, log_k)
        ss_res = np.sum((log_e - pred) ** 2)
        ss_tot = np.sum((log_e - np.mean(log_e)) ** 2) + 1e-30
        r_squared = float(1 - ss_res / ss_tot)
    else:
        exponent = float("nan")
        r_squared = 0.0
    return {"k": k_centres, "E": E, "kolmogorov_exponent": exponent,
            "r_squared": r_squared}


def velocity_pdf(u: np.ndarray, v: np.ndarray, bins: int = 50) -> dict:
    """Probability distribution of velocity components."""
    hist_u, edges_u = np.histogram(u, bins=bins, density=True)
    hist_v, edges_v = np.histogram(v, bins=bins, density=True)
    return {"u_centers": 0.5 * (edges_u[:-1] + edges_u[1:]), "u_pdf": hist_u,
            "v_centers": 0.5 * (edges_v[:-1] + edges_v[1:]), "v_pdf": hist_v}


# ---------------------------------------------------------------------
#  Error metrics
# ---------------------------------------------------------------------
def error_metrics(actual_u, actual_v, pred_u, pred_v) -> dict:
    """Comprehensive error metrics between actual and predicted fields."""
    au = np.asarray(actual_u); av = np.asarray(actual_v)
    pu = np.asarray(pred_u); pv = np.asarray(pred_v)
    du = au - pu
    dv = av - pv
    err2 = du ** 2 + dv ** 2
    err = np.sqrt(err2)
    mag_a = np.sqrt(au ** 2 + av ** 2)
    mag_p = np.sqrt(pu ** 2 + pv ** 2)
    return {
        "mse": float(np.mean(err2)),
        "rmse": float(np.sqrt(np.mean(err2))),
        "mae": float(np.mean(np.abs(du)) + np.mean(np.abs(dv))) / 2.0,
        "max_error": float(np.max(err)),
        "mean_error": float(np.mean(err)),
        "median_error": float(np.median(err)),
        "p95_error": float(np.percentile(err, 95)),
        "correlation_u": float(np.corrcoef(au.ravel(), pu.ravel())[0, 1]),
        "correlation_v": float(np.corrcoef(av.ravel(), pv.ravel())[0, 1]),
        "correlation_mag": float(np.corrcoef(mag_a.ravel(), mag_p.ravel())[0, 1]),
        "nrmse": float(np.sqrt(np.mean(err2)) / (np.max(mag_a) + 1e-30)),
    }


def per_frame_errors(u_stack, v_stack, pred_stack) -> dict:
    """Per-frame RMSE / MSE / MAE time series."""
    rmses, mses, maes = [], [], []
    for f in range(u_stack.shape[0]):
        m = error_metrics(u_stack[f], v_stack[f],
                          pred_stack[f, ..., 0], pred_stack[f, ..., 1])
        rmses.append(m["rmse"]); mses.append(m["mse"]); maes.append(m["mae"])
    return {"rmse": np.array(rmses), "mse": np.array(mses), "mae": np.array(maes)}


# ---------------------------------------------------------------------
#  Region-based analysis
# ---------------------------------------------------------------------
def region_statistics(u: np.ndarray, v: np.ndarray,
                      region: Optional[tuple] = None, dx: float = 1.0,
                      dy: float = 1.0, nu: float = 1e-6) -> dict:
    """Compute statistics over a sub-region (x0, y0, x1, y1) of the field."""
    if region is not None:
        x0, y0, x1, y1 = region
        u = u[y0:y1, x0:x1]
        v = v[y0:y1, x0:x1]
    stats = turbulence_statistics(u, v, nu=nu, length_scale=min(u.shape) * dx)
    stats["vorticity_mean"] = float(np.mean(vorticity(u, v, dx, dy)))
    stats["divergence_mean"] = float(np.mean(divergence(u, v, dx, dy)))
    stats["region"] = region
    return stats
