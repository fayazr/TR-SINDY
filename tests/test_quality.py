"""Tests for tr_sindy_app.quality — outlier detection and interpolation."""

from __future__ import annotations

import numpy as np
import pytest

from tr_sindy_app import quality


# ---------------------------------------------------------------------
#  Outlier detection
# ---------------------------------------------------------------------
class TestZScoreOutliers:
    def test_clean_data_no_outliers(self):
        rng = np.random.default_rng(42)
        field = rng.standard_normal(1000)
        mask = quality.zscore_outliers(field, thresh=4.0)
        assert mask.sum() == 0 or mask.sum() < 5  # very few false positives

    def test_detects_extreme_outlier(self):
        field = np.zeros(100)
        field[50] = 100.0  # extreme outlier
        mask = quality.zscore_outliers(field, thresh=4.0)
        assert mask[50]

    def test_constant_field_no_outliers(self):
        field = np.full(50, 3.14)
        mask = quality.zscore_outliers(field, thresh=4.0)
        assert mask.sum() == 0


class TestModifiedZScoreOutliers:
    def test_detects_outlier(self):
        # Need some variance for MAD to be nonzero
        rng = np.random.default_rng(42)
        field = rng.standard_normal(100)
        field[50] = 50.0  # extreme outlier
        mask = quality.modified_zscore_outliers(field, thresh=3.5)
        assert mask[50]

    def test_robust_to_multiple_outliers(self):
        """Modified z-score should be more robust than z-score."""
        rng = np.random.default_rng(0)
        field = rng.standard_normal(100)
        field[10] = 100
        field[20] = -100
        field[30] = 200
        mask = quality.modified_zscore_outliers(field, thresh=3.5)
        assert mask[10] and mask[20] and mask[30]


class TestIQROutliers:
    def test_detects_outside_whiskers(self):
        field = np.concatenate([np.ones(50), [100.0]])
        mask = quality.iqr_outliers(field, k=1.5)
        assert mask[-1]

    def test_no_outliers_in_normal_data(self):
        rng = np.random.default_rng(0)
        field = rng.standard_normal(500)
        mask = quality.iqr_outliers(field, k=3.0)
        assert mask.sum() <= 5


class TestDetectOutliers:
    def test_dispatch(self):
        rng = np.random.default_rng(42)
        field = rng.standard_normal(50)
        field[25] = 99
        for method in ("zscore", "modz", "iqr"):
            mask = quality.detect_outliers(field, method=method)
            assert mask.any(), f"{method} should detect the outlier"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            quality.detect_outliers(np.zeros(10), method="bogus")


# ---------------------------------------------------------------------
#  Outlier replacement
# ---------------------------------------------------------------------
class TestReplaceOutliers:
    def test_replaces_flagged_values(self):
        field = np.ones((10, 10))
        field[5, 5] = 999.0
        mask = field > 100
        result = quality.replace_outliers(field, mask, method="linear")
        assert result[5, 5] < 100  # replaced
        assert result[5, 5] == pytest.approx(1.0)  # interpolated from neighbors


# ---------------------------------------------------------------------
#  Noise estimation
# ---------------------------------------------------------------------
class TestEstimateNoise:
    def test_zero_noise_for_constant(self):
        result = quality.estimate_noise(np.zeros((20, 20)))
        assert result["noise_sigma"] == pytest.approx(0.0, abs=1e-10)

    def test_detects_noise(self):
        rng = np.random.default_rng(42)
        field = rng.standard_normal((50, 50)) * 5.0
        result = quality.estimate_noise(field)
        assert result["noise_sigma"] > 0
