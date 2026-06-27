"""Pure-numpy SINDy implementation for mobile (no scipy/sklearn dependency).

Implements:
  - Polynomial library builder (degree 1-N)
  - STLSQ (Sequentially Thresholded Least Squares) optimizer
  - Model fitting and prediction
  - Equation string formatting

This is a lightweight replacement for pysindy that runs on-device.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------
#  Polynomial library
# ---------------------------------------------------------------------
def build_polynomial_library(X: np.ndarray, degree: int = 3) -> np.ndarray:
    """Build a polynomial library from state variables.

    For X with n_features columns and degree d, produces all monomials
    up to degree d (including the constant term).

    Example: X = [u, v], degree=2 → [1, u, v, u^2, u*v, v^2]
    """
    n_samples, n_features = X.shape
    # Generate all multi-indices (e1, e2, ...) with sum <= degree
    from itertools import combinations_with_replacement
    combos = []
    for d in range(degree + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            combos.append(combo)
    # Build library
    n_terms = len(combos)
    Theta = np.ones((n_samples, n_terms))
    for i, combo in enumerate(combos):
        if len(combo) == 0:
            continue  # constant term = 1
        for idx in combo:
            Theta[:, i] *= X[:, idx]
    return Theta, combos


def library_term_name(combo, var_names=("u", "v")):
    """Convert a multi-index combo to a human-readable term name."""
    if len(combo) == 0:
        return "1"
    parts = []
    for idx in combo:
        parts.append(var_names[idx] if idx < len(var_names) else f"x{idx}")
    # Combine repeated indices as powers
    from collections import Counter
    counts = Counter(combo)
    terms = []
    for idx, count in sorted(counts.items()):
        name = var_names[idx] if idx < len(var_names) else f"x{idx}"
        if count == 1:
            terms.append(name)
        else:
            terms.append(f"{name}^{count}")
    return " * ".join(terms)


# ---------------------------------------------------------------------
#  STLSQ optimizer
# ---------------------------------------------------------------------
def stlsq(Theta: np.ndarray, Xdot: np.ndarray, threshold: float = 0.1,
          max_iter: int = 20) -> np.ndarray:
    """Sequentially Thresholded Least Squares.

    Iteratively:
      1. Solve least squares for coefficients
      2. Zero out coefficients below threshold
      3. Re-solve using only non-zero columns
      4. Repeat until convergence

    Returns coefficient matrix (n_terms x n_features).
    """
    n_terms = Theta.shape[1]
    n_features = Xdot.shape[1]
    Xi = np.linalg.lstsq(Theta, Xdot, rcond=None)[0]
    for _ in range(max_iter):
        # Threshold small coefficients
        small = np.abs(Xi) < threshold
        Xi[small] = 0
        # Re-solve for each output dimension using only big coefficients
        for j in range(n_features):
            big = ~small[:, j]
            if np.any(big):
                Xi[big, j] = np.linalg.lstsq(Theta[:, big], Xdot[:, j],
                                             rcond=None)[0]
                Xi[~big, j] = 0
        # Check convergence
        new_small = np.abs(Xi) < threshold
        if np.array_equal(small, new_small):
            break
    return Xi


# ---------------------------------------------------------------------
#  SINDy fit / predict
# ---------------------------------------------------------------------
def fit_sindy(X: np.ndarray, Xdot: np.ndarray, degree: int = 3,
              threshold: float = 0.1) -> dict:
    """Fit a SINDy model.

    Parameters
    ----------
    X : state variables (n_samples, n_features)
    Xdot : time derivatives (n_samples, n_features)
    degree : max polynomial degree
    threshold : sparsity threshold for STLSQ

    Returns dict with coefficients, library combos, and equation strings.
    """
    Theta, combos = build_polynomial_library(X, degree)
    Xi = stlsq(Theta, Xdot, threshold=threshold)
    var_names = ["u", "v"][:X.shape[1]]
    equations = []
    for j in range(Xi.shape[1]):
        terms = []
        for i, coef in enumerate(Xi[:, j]):
            if abs(coef) > 1e-10:
                name = library_term_name(combos[i], var_names)
                sign = " + " if coef > 0 and terms else (
                    " - " if coef < 0 and terms else
                    ("" if coef > 0 else "-"))
                terms.append(f"{sign}{abs(coef):.4f} {name}")
        eq = f"d{var_names[j]}/dt = " + "".join(terms) if terms else f"d{var_names[j]}/dt = 0"
        equations.append(eq)
    return {
        "coefficients": Xi,
        "combos": combos,
        "equations": equations,
        "n_terms": int(np.count_nonzero(Xi)),
        "degree": degree,
        "threshold": threshold,
    }


def predict_sindy(model: dict, X: np.ndarray) -> np.ndarray:
    """Predict derivatives using a fitted SINDy model."""
    Theta, _ = build_polynomial_library(X, model["degree"])
    return Theta @ model["coefficients"]


def compute_gradient(X: np.ndarray, dt: float) -> np.ndarray:
    """Compute time derivative via finite differences.

    X : (n_samples, n_features) state trajectory
    dt : time step
    """
    Xdot = np.zeros_like(X)
    Xdot[1:-1] = (X[2:] - X[:-2]) / (2 * dt)
    Xdot[0] = (X[1] - X[0]) / dt
    Xdot[-1] = (X[-1] - X[-2]) / dt
    return Xdot
