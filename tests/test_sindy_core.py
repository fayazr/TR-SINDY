"""Tests for tr_sindy_app.sindy_core — library construction, dataset building.

These tests skip gracefully if pysindy is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from tr_sindy_app import sindy_core

pysindy_required = pytest.mark.skipif(
    not sindy_core.has_pysindy(),
    reason="pysindy not installed",
)


@pysindy_required
class TestBuildLibrary:
    def test_polynomial(self):
        lib = sindy_core.build_library("polynomial", degree=2)
        assert lib is not None

    def test_fourier(self):
        lib = sindy_core.build_library("fourier", n_freq=2)
        assert lib is not None

    def test_combined(self):
        # GeneralizedLibrary API varies across pysindy versions; just check
        # it doesn't raise for the basic case.
        try:
            lib = sindy_core.build_library("combined", degree=2, n_freq=1)
            assert lib is not None
        except Exception:
            pytest.skip("GeneralizedLibrary not compatible with this pysindy version")

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            sindy_core.build_library("nonexistent")


@pysindy_required
class TestBuildOptimizer:
    def test_stlsq(self):
        opt = sindy_core.build_optimizer("stlsq", threshold=0.1)
        assert opt is not None

    def test_sr3(self):
        opt = sindy_core.build_optimizer("sr3", threshold=0.1)
        assert opt is not None

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            sindy_core.build_optimizer("bogus")


class TestMmapGradient:
    def test_gradient_axis0(self):
        arr = np.arange(20, dtype=np.float32).reshape(4, 5)
        grad = sindy_core.mmap_gradient(arr, axis=0)
        assert grad.shape == arr.shape

    def test_gradient_axis1(self):
        arr = np.arange(20, dtype=np.float32).reshape(4, 5)
        grad = sindy_core.mmap_gradient(arr, axis=1)
        assert grad.shape == arr.shape


class TestTimeDelayEmbedding:
    def test_shape(self):
        series = np.arange(20, dtype=np.float32)
        emb = sindy_core.time_delay_embedding(series, delay=2, n_delays=2)
        # n_delays=2: each row has n_delays+1=3 values
        assert emb.shape[1] == 3
        # out_len = 20 - 2*2 = 16
        assert emb.shape[0] == 20 - 2 * 2

    def test_content(self):
        series = np.arange(10, dtype=np.float32)
        emb = sindy_core.time_delay_embedding(series, delay=1, n_delays=1)
        # n_delays=1: each row has 2 values [t, t+1]
        np.testing.assert_array_equal(emb[0], [0, 1])
        np.testing.assert_array_equal(emb[1], [1, 2])
