"""Tests for tr_sindy_app.export — metadata building and JSON serialization."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tr_sindy_app import export


class TestBuildMetadata:
    def test_basic_structure(self):
        project_state = {
            "video_file": "test.mp4",
            "roi_box": [10, 20, 100, 200],
            "calibration_px": 120.0,
            "calibration_m": 0.1,
            "meters_per_pixel": 0.1 / 120,
        }
        of_result = {"backend": "farneback", "frames": 100,
                     "roi_h": 180, "roi_w": 90, "FPS": 30.0}
        sindy_result = {"library": "polynomial", "n_features": 10}
        meta = export.build_metadata(project_state, of_result, sindy_result)
        assert meta["app"] == "Turbulence Realm - SINDy"
        assert meta["video_file"] == "test.mp4"
        assert meta["optical_flow"]["backend"] == "farneback"
        assert meta["sindy"]["library"] == "polynomial"
        # mmap paths should be stripped
        assert "u_mmap_path" not in meta["optical_flow"]
        assert "model" not in meta["sindy"]

    def test_extra_fields_merged(self):
        meta = export.build_metadata({}, {}, {}, extra={"custom": "value"})
        assert meta["custom"] == "value"

    def test_version_present(self):
        meta = export.build_metadata({}, {}, {})
        assert "version" in meta
        assert isinstance(meta["version"], str)


class TestExportMetadata:
    def test_writes_valid_json(self, tmp_path):
        meta = {"app": "test", "version": "1.0", "nested": {"a": [1, 2, 3]}}
        path = str(tmp_path / "meta.json")
        export.export_metadata(path, meta)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["app"] == "test"
        assert loaded["nested"]["a"] == [1, 2, 3]

    def test_numpy_serialization(self, tmp_path):
        """numpy types should be serialized to native Python."""
        meta = {"count": np.int64(42), "value": np.float64(3.14),
                "array": np.array([1, 2, 3])}
        path = str(tmp_path / "np.json")
        export.export_metadata(path, meta)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["count"] == 42
        assert loaded["value"] == pytest.approx(3.14)
        assert loaded["array"] == [1, 2, 3]


class TestExportCSV:
    def test_writes_per_frame_csvs(self, tmp_path):
        n, h, w = 3, 4, 4
        u = np.random.randn(n, h, w).astype(np.float32)
        v = np.random.randn(n, h, w).astype(np.float32)
        pred = np.random.randn(n, h, w, 2).astype(np.float32)
        out_dir = str(tmp_path / "csv_out")
        count = export.export_csv(out_dir, u, v, pred)
        assert count == n
        files = sorted(os.listdir(out_dir))
        assert len(files) == n
        assert files[0].startswith("frame_1")


class TestPackageMetadata:
    def test_version_string(self):
        from tr_sindy_app import __version__
        assert isinstance(__version__, str)
        # Should look like a semver
        parts = __version__.split(".")
        assert len(parts) >= 2
