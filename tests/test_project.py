"""Tests for tr_sindy_app.project — save/load, presets, history."""

from __future__ import annotations

import os

import numpy as np

from tr_sindy_app import project


# ---------------------------------------------------------------------
#  Presets
# ---------------------------------------------------------------------
class TestPresets:
    def test_builtin_presets_install(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project, "PRESETS_DIR", str(tmp_path / "presets"))
        project.install_builtin_presets()
        names = project.list_presets()
        assert "farneback-default" in names
        assert "tvl1-turbulent" in names

    def test_save_and_load_preset(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project, "PRESETS_DIR", str(tmp_path / "presets"))
        p = project.Preset("test-preset", "a test",
                           optical_flow={"backend": "farneback"},
                           sindy={"library": "polynomial"})
        project.save_preset(p)
        loaded = project.load_preset("test-preset")
        assert loaded.name == "test-preset"
        assert loaded.optical_flow["backend"] == "farneback"

    def test_delete_preset(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project, "PRESETS_DIR", str(tmp_path / "presets"))
        p = project.Preset("to-delete")
        project.save_preset(p)
        assert "to-delete" in project.list_presets()
        project.delete_preset("to-delete")
        assert "to-delete" not in project.list_presets()


# ---------------------------------------------------------------------
#  Processing history
# ---------------------------------------------------------------------
class TestProcessingHistory:
    def test_log_and_summary(self):
        h = project.ProcessingHistory()
        h.log("optical_flow", {"backend": "farneback"}, status="ok")
        h.log("sindy_fit", {"library": "polynomial"}, status="ok")
        assert len(h.entries) == 2
        summary = h.summary()
        assert "optical_flow" in summary
        assert "sindy_fit" in summary

    def test_round_trip(self):
        h = project.ProcessingHistory()
        h.log("step1", {"a": 1})
        lst = h.to_list()
        h2 = project.ProcessingHistory.from_list(lst)
        assert len(h2.entries) == 1
        assert h2.entries[0].step == "step1"


# ---------------------------------------------------------------------
#  Project save/load
# ---------------------------------------------------------------------
class TestProjectSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project, "RECENT_FILE",
                            str(tmp_path / "recent.json"))
        state = {
            "video_file": "test.mp4",
            "roi_box": [10, 20, 100, 200],
            "calibration_px": 120.0,
            "calibration_m": 0.1,
            "meters_per_pixel": 0.1 / 120,
            "optical_flow": {"backend": "farneback"},
            "sindy": {"library": "polynomial", "degree": 3},
        }
        path = str(tmp_path / "test_project.trsindy")
        project.save_project(path, state, mmap_paths=None, bundle_mmaps=False)
        assert os.path.exists(path)

        loaded = project.load_project(path)
        assert loaded["video_file"] == "test.mp4"
        assert loaded["roi_box"] == [10, 20, 100, 200]
        assert loaded["optical_flow"]["backend"] == "farneback"

    def test_bundles_memmaps(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project, "RECENT_FILE",
                            str(tmp_path / "recent.json"))
        # Create a fake memmap file in a subdirectory (not the project dir)
        mmap_dir = tmp_path / "mmaps"
        mmap_dir.mkdir()
        mmap_path = str(mmap_dir / "u.dat")
        arr = np.arange(100, dtype=np.float32)
        arr.tofile(mmap_path)

        state = {"video_file": "v.mp4"}
        proj_path = str(tmp_path / "bundled.trsindy")
        project.save_project(proj_path, state,
                             mmap_paths={"u": mmap_path},
                             bundle_mmaps=True)
        loaded = project.load_project(proj_path)
        assert "u" in loaded["bundled_mmaps"]
        assert os.path.exists(loaded["bundled_mmaps"]["u"])


# ---------------------------------------------------------------------
#  Recent files
# ---------------------------------------------------------------------
class TestRecentFiles:
    def test_register_and_load(self, tmp_path, monkeypatch):
        recent_file = str(tmp_path / "recent.json")
        monkeypatch.setattr(project, "RECENT_FILE", recent_file)
        project.register_recent("/a.trsindy")
        project.register_recent("/b.trsindy")
        project.register_recent("/c.trsindy")
        recent = project.load_recent(10)
        assert recent[0] == "/c.trsindy"
        assert recent[1] == "/b.trsindy"
        assert recent[2] == "/a.trsindy"

    def test_no_duplicates(self, tmp_path, monkeypatch):
        recent_file = str(tmp_path / "recent.json")
        monkeypatch.setattr(project, "RECENT_FILE", recent_file)
        project.register_recent("/x.trsindy")
        project.register_recent("/y.trsindy")
        project.register_recent("/x.trsindy")  # re-register
        recent = project.load_recent(10)
        assert recent.count("/x.trsindy") == 1
        assert recent[0] == "/x.trsindy"
