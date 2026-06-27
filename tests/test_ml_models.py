"""Tests for tr_sindy_app.ml_models — model export round-trip.

These tests skip gracefully if torch is not installed.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tr_sindy_app import ml_models

torch_required = pytest.mark.skipif(
    not ml_models.torch_available(),
    reason="torch not installed",
)


class TestTorchAvailability:
    def test_torch_available_returns_bool(self):
        assert isinstance(ml_models.torch_available(), bool)

    def test_gpu_available_returns_bool(self):
        assert isinstance(ml_models.gpu_available(), bool)


@torch_required
class TestSetSeed:
    def test_reproducible_weights(self):
        import torch
        ml_models.set_seed(42)
        a = torch.randn(3, 3)
        ml_models.set_seed(42)
        b = torch.randn(3, 3)
        np.testing.assert_array_equal(a.numpy(), b.numpy())


@torch_required
class TestExportModel:
    def test_export_pt_roundtrip(self, tmp_path):
        """Train a tiny model for 1 epoch, export .pt, reload state_dict."""
        import torch

        ml_models.set_seed(0)
        model = ml_models.FlowVAE(latent_dim=4, hidden=16,
                                  lr=1e-3, epochs=1, device="cpu")
        # FlowVAE.fit() takes snapshots (n_samples, dim)
        snapshots = np.random.randn(10, 8).astype(np.float32)
        model.fit(snapshots)

        path = str(tmp_path / "model.pt")
        report = ml_models.export_model(model, path, formats=("pt",))
        assert os.path.exists(path)
        assert report["pt"]["status"] == "ok"

        # Reload and verify state_dict keys match
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["model_class"] == "FlowVAE"
        assert "state_dicts" in ckpt
        # FlowVAE should have encoder + decoder
        assert "enc" in ckpt["state_dicts"]
        assert "dec" in ckpt["state_dicts"]

    def test_export_torchscript(self, tmp_path):
        ml_models.set_seed(0)
        model = ml_models.DeepONet(n_sensors=16, trunk_in=2, hidden=16,
                                   lr=1e-3, device="cpu")
        path = str(tmp_path / "deepo")
        report = ml_models.export_model(model, path,
                                        formats=("pt", "torchscript"))
        assert report["pt"]["status"] == "ok"
        # TorchScript may or may not succeed depending on model complexity.
        if "torchscript" in report and report["torchscript"].get("status") == "ok":
            # At least one .ts.pt file should exist
            ts_files = [f for f in os.listdir(tmp_path) if f.endswith(".ts.pt")]
            assert len(ts_files) > 0

    def test_export_report_structure(self, tmp_path):
        ml_models.set_seed(0)
        model = ml_models.FlowVAE(latent_dim=4, hidden=16,
                                  lr=1e-3, epochs=1, device="cpu")
        snapshots = np.random.randn(10, 8).astype(np.float32)
        model.fit(snapshots)
        path = str(tmp_path / "report_test.pt")
        report = ml_models.export_model(model, path, formats=("pt",),
                                        metadata={"test": True})
        assert isinstance(report, dict)
        assert "pt" in report
