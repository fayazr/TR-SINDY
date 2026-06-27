"""Tests for tr_sindy_app.analysis — kinematics, spectra, POD/DMD, metrics."""

from __future__ import annotations

import numpy as np
import pytest

from tr_sindy_app import analysis


# ---------------------------------------------------------------------
#  Vorticity
# ---------------------------------------------------------------------
class TestVorticity:
    def test_solid_body_rotation(self):
        """Solid-body rotation: omega_z = 2*angular_velocity (constant)."""
        h, w = 50, 50
        y, x = np.mgrid[:h, :w]
        cx, cy = w / 2, h / 2
        omega = 0.3
        u = -omega * (y - cy)
        v = omega * (x - cx)
        w_z = analysis.vorticity(u, v)
        # Interior should be ~2*omega (boundary effects from np.gradient).
        interior = w_z[5:-5, 5:-5]
        assert np.allclose(interior, 2 * omega, atol=1e-10)

    def test_uniform_field_zero_vorticity(self):
        u = np.ones((20, 20))
        v = np.zeros((20, 20))
        w_z = analysis.vorticity(u, v)
        assert np.allclose(w_z, 0.0)

    def test_shape_preserved(self):
        u = np.random.randn(30, 40)
        v = np.random.randn(30, 40)
        assert analysis.vorticity(u, v).shape == u.shape


# ---------------------------------------------------------------------
#  Divergence
# ---------------------------------------------------------------------
class TestDivergence:
    def test_incompressible_solid_body(self):
        h, w = 40, 40
        y, x = np.mgrid[:h, :w]
        cx, cy = w / 2, h / 2
        u = -(y - cy)
        v = (x - cx)
        div = analysis.divergence(u, v)
        assert np.allclose(div[5:-5, 5:-5], 0.0, atol=1e-10)

    def test_source_flow(self):
        """Radial outflow: div = 2*rate (constant in interior)."""
        h, w = 40, 40
        y, x = np.mgrid[:h, :w]
        cx, cy = w / 2, h / 2
        rate = 0.5
        u = rate * (x - cx)
        v = rate * (y - cy)
        div = analysis.divergence(u, v)
        assert np.allclose(div[5:-5, 5:-5], 2 * rate, atol=1e-10)


# ---------------------------------------------------------------------
#  Strain rate
# ---------------------------------------------------------------------
class TestStrainRate:
    def test_keys_present(self):
        s = analysis.strain_rate(np.zeros((10, 10)), np.zeros((10, 10)))
        for key in ("Sxx", "Syy", "Sxy", "shear", "magnitude",
                    "du_dx", "du_dy", "dv_dx", "dv_dy"):
            assert key in s

    def test_zero_field_zero_strain(self):
        s = analysis.strain_rate(np.zeros((10, 10)), np.zeros((10, 10)))
        assert np.allclose(s["magnitude"], 0.0)


# ---------------------------------------------------------------------
#  Error metrics
# ---------------------------------------------------------------------
class TestErrorMetrics:
    def test_perfect_prediction(self):
        u = np.random.randn(20, 20)
        v = np.random.randn(20, 20)
        m = analysis.error_metrics(u, v, u, v)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-12)
        assert m["mse"] == pytest.approx(0.0, abs=1e-12)
        assert m["correlation_u"] == pytest.approx(1.0)
        assert m["correlation_v"] == pytest.approx(1.0)

    def test_constant_offset(self):
        """A constant offset should give RMSE = offset magnitude."""
        u = np.zeros((20, 20))
        v = np.zeros((20, 20))
        pu = np.full_like(u, 3.0)
        pv = np.full_like(v, 4.0)
        m = analysis.error_metrics(u, v, pu, pv)
        # error magnitude = sqrt(3^2 + 4^2) = 5 everywhere
        assert m["rmse"] == pytest.approx(5.0)
        assert m["max_error"] == pytest.approx(5.0)

    def test_all_keys_present(self):
        m = analysis.error_metrics(np.zeros((5, 5)), np.zeros((5, 5)),
                                   np.zeros((5, 5)), np.zeros((5, 5)))
        for key in ("mse", "rmse", "mae", "max_error", "mean_error",
                    "median_error", "p95_error", "correlation_u",
                    "correlation_v", "correlation_mag", "nrmse"):
            assert key in m


# ---------------------------------------------------------------------
#  Structure function
# ---------------------------------------------------------------------
class TestStructureFunction:
    def test_returns_r_and_s2(self):
        field = np.random.randn(30, 30)
        sf = analysis.structure_function(field, max_sep=10)
        assert "r" in sf and "S2" in sf
        assert len(sf["r"]) == len(sf["S2"])

    def test_monotonic_for_smooth_field(self):
        """For a smooth (linear ramp) field, S2 should increase with r."""
        field = np.tile(np.arange(40, dtype=float), (40, 1))
        sf = analysis.structure_function(field, max_sep=10)
        assert np.all(np.diff(sf["S2"]) > 0)


class TestStructureFunctionScaling:
    def test_returns_zeta(self):
        rng = np.random.default_rng(42)
        field = rng.standard_normal((40, 40))
        result = analysis.structure_function_scaling(field, max_sep=10, order=2)
        assert "zeta" in result
        assert "r_squared" in result
        assert "S2" in result
        assert isinstance(result["zeta"], float)

    def test_smooth_field_positive_exponent(self):
        """A smooth field should have a positive scaling exponent."""
        field = np.tile(np.arange(50, dtype=float), (50, 1))
        result = analysis.structure_function_scaling(field, max_sep=15, order=2)
        assert result["zeta"] > 0


class TestEnergySpectrum:
    def test_returns_k_and_E(self):
        u = np.random.randn(32, 32)
        v = np.random.randn(32, 32)
        result = analysis.energy_spectrum(u, v)
        assert "k" in result and "E" in result
        assert "kolmogorov_exponent" in result
        assert len(result["k"]) == len(result["E"])

    def test_uniform_field_zero_energy(self):
        """A uniform field has no energy in any non-zero wavenumber."""
        u = np.ones((16, 16))
        v = np.zeros((16, 16))
        result = analysis.energy_spectrum(u, v)
        # Energy should be concentrated at k=0
        assert result["E"][0] > result["E"][5]


# ---------------------------------------------------------------------
#  POD
# ---------------------------------------------------------------------
class TestPOD:
    def test_orthogonal_modes(self):
        """POD modes should be orthogonal (flattened modes @ modes.T ~ diagonal)."""
        T = 20
        h, w = 8, 8
        rng = np.random.default_rng(42)
        stack = rng.standard_normal((T, h, w))
        res = analysis.pod_decompose(stack, n_modes=5)
        modes = res["modes"]  # (n_modes, H, W)
        flat = modes.reshape(5, -1)  # (n_modes, H*W)
        # Normalize and check orthogonality
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        normed = flat / (norms + 1e-30)
        gram = normed @ normed.T
        np.testing.assert_allclose(np.eye(5), gram, atol=1e-6)

    def test_energy_fractions_sum_to_one(self):
        rng = np.random.default_rng(0)
        stack = rng.standard_normal((15, 8, 8))
        res = analysis.pod_decompose(stack, n_modes=10)
        assert res["cumulative"].sum() >= 0.99  # cumulative[-1] ~ 1.0
        assert res["cumulative"][-1] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------
#  Velocity PDF
# ---------------------------------------------------------------------
class TestVelocityPDF:
    def test_pdf_integrates_to_one(self):
        u = np.random.randn(1000)
        v = np.random.randn(1000)
        pdf = analysis.velocity_pdf(u, v, bins=50)
        # density integrates to ~1
        integral_u = np.trapezoid(pdf["u_pdf"], pdf["u_centers"])
        assert integral_u == pytest.approx(1.0, abs=0.1)
