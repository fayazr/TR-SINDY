"""Tests for tr_sindy_app._provenance — reproducibility metadata."""

from __future__ import annotations

from tr_sindy_app._provenance import _file_sha256, collect_provenance


class TestCollectProvenance:
    def test_returns_dict_with_required_keys(self):
        p = collect_provenance()
        assert isinstance(p, dict)
        for key in ("app_version", "python_version", "platform",
                    "git_commit", "packages", "timestamp"):
            assert key in p

    def test_records_seed(self):
        p = collect_provenance(seed=42)
        assert p["random_seed"] == 42

    def test_records_config(self):
        p = collect_provenance(config={"backend": "farneback"})
        assert p["config"]["backend"] == "farneback"

    def test_records_input_file_hash(self, tmp_path):
        # Create a test file
        fpath = str(tmp_path / "test.bin")
        with open(fpath, "wb") as f:
            f.write(b"hello world")
        p = collect_provenance(input_file=fpath)
        assert p["input_file"] == "test.bin"
        assert p["input_sha256"] is not None
        assert len(p["input_sha256"]) == 64  # SHA-256 hex length

    def test_missing_input_file(self):
        p = collect_provenance(input_file="/nonexistent/file.mp4")
        assert p["input_sha256"] is None

    def test_package_versions_includes_numpy(self):
        p = collect_provenance()
        assert "numpy" in p["packages"]


class TestFileSha256:
    def test_known_hash(self, tmp_path):
        fpath = str(tmp_path / "test.txt")
        with open(fpath, "wb") as f:
            f.write(b"hello world")
        h = _file_sha256(fpath)
        # Known SHA-256 of "hello world"
        assert h == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

    def test_nonexistent_returns_none(self):
        assert _file_sha256("/nonexistent") is None
