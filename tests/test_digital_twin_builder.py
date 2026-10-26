import json
from pathlib import Path

from PyQt5 import QtCore

import src.viewer.digital_twin_builder as builder_module
from src.viewer.digital_twin_builder import (
    REFINE_ITERATIONS,
    ColmapSource,
    DigitalTwinBuilder,
    build_preset,
    digital_twin_cache_path,
    find_colmap_source,
    write_provenance_file,
)


def _make_scene(tmp_path: Path, scene_name: str, frame_name: str = "frame_0000",
                 with_images: bool = True, with_sparse: bool = True) -> Path:
    scene_dir = tmp_path / scene_name
    plys_dir = scene_dir / "plys"
    plys_dir.mkdir(parents=True)
    (plys_dir / "frame_0000.ply").write_bytes(b"")

    frame_dir = scene_dir / "colmap_data" / frame_name
    frame_dir.mkdir(parents=True)
    if with_images:
        (frame_dir / "images").mkdir()
    if with_sparse:
        (frame_dir / "sparse").mkdir()
    return scene_dir


class TestFindColmapSource:
    def test_finds_matching_scene(self, tmp_path):
        scene_dir = _make_scene(tmp_path, "lego_bugatti")
        ply_path = scene_dir / "plys" / "frame_0000.ply"

        result = find_colmap_source(ply_path)

        assert result is not None
        assert result.scene_name == "lego_bugatti"
        assert result.frame_dir == scene_dir / "colmap_data" / "frame_0000"

    def test_returns_none_when_not_in_a_plys_folder(self, tmp_path):
        loose_ply = tmp_path / "some_folder" / "scene.ply"
        loose_ply.parent.mkdir(parents=True)
        loose_ply.write_bytes(b"")

        assert find_colmap_source(loose_ply) is None

    def test_returns_none_when_no_colmap_data_dir(self, tmp_path):
        plys_dir = tmp_path / "orphan_scene" / "plys"
        plys_dir.mkdir(parents=True)
        ply_path = plys_dir / "frame_0000.ply"
        ply_path.write_bytes(b"")

        assert find_colmap_source(ply_path) is None

    def test_returns_none_when_frame_dir_missing_images(self, tmp_path):
        scene_dir = _make_scene(tmp_path, "incomplete_scene", with_images=False)
        ply_path = scene_dir / "plys" / "frame_0000.ply"

        assert find_colmap_source(ply_path) is None

    def test_returns_none_when_frame_dir_missing_sparse(self, tmp_path):
        scene_dir = _make_scene(tmp_path, "incomplete_scene", with_sparse=False)
        ply_path = scene_dir / "plys" / "frame_0000.ply"

        assert find_colmap_source(ply_path) is None

    def test_picks_lexicographically_first_frame_when_multiple_exist(self, tmp_path):
        scene_dir = tmp_path / "multi_frame"
        plys_dir = scene_dir / "plys"
        plys_dir.mkdir(parents=True)
        (plys_dir / "frame_0000.ply").write_bytes(b"")
        for frame in ("frame_0001", "frame_0000"):
            frame_dir = scene_dir / "colmap_data" / frame
            (frame_dir / "images").mkdir(parents=True)
            (frame_dir / "sparse").mkdir()

        result = find_colmap_source(plys_dir / "frame_0000.ply")

        assert result.frame_dir.name == "frame_0000"

    def test_two_scenes_with_identical_frame_names_resolve_to_different_scene_names(self, tmp_path):
        bugatti_dir = _make_scene(tmp_path, "lego_bugatti")
        ferrari_dir = _make_scene(tmp_path, "lego_ferrari")

        bugatti = find_colmap_source(bugatti_dir / "plys" / "frame_0000.ply")
        ferrari = find_colmap_source(ferrari_dir / "plys" / "frame_0000.ply")

        assert bugatti.frame_dir.name == ferrari.frame_dir.name == "frame_0000"
        assert bugatti.scene_name != ferrari.scene_name
        assert bugatti.scene_name == "lego_bugatti"
        assert ferrari.scene_name == "lego_ferrari"


class TestDigitalTwinCachePath:
    def test_matches_existing_bonsai_naming_convention(self):
        path = digital_twin_cache_path("bonsai")
        assert path.name == "bonsai_digital_twin.npz"
        assert path.parent.name == "gaussian_scenes"

    def test_uses_the_given_scene_name(self):
        assert digital_twin_cache_path("lego_bugatti").name == "lego_bugatti_digital_twin.npz"


def _normalized_source(root: Path, images: int = 3, masks: int = 3) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    for i in range(images):
        (root / "images" / f"{i:04d}.jpg").write_bytes(b"x")
    for i in range(masks):
        (root / "masks" / f"{i:04d}.png").write_bytes(b"x")
    return root


class TestBuildRecordFile:
    def test_the_preset_is_the_one_the_build_uses(self):
        assert build_preset()["refine_iterations"] == REFINE_ITERATIONS
        assert set(build_preset()) >= {"baseline_iterations", "coarse_iterations", "refine_iterations", "mesh_triangles_target"}

    def test_the_record_is_written_to_a_temporary_file_for_the_export_script(self, tmp_path):
        source = _normalized_source(tmp_path / "lego_x")

        path = write_provenance_file("lego_x", source)

        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            assert record["scene"] == "lego_x" and record["photographs"] == 3 and record["masked"] is True
            assert record["preset"] == build_preset()
        finally:
            path.unlink()

    def test_a_record_that_cannot_be_made_gives_no_file_and_raises_nothing(self, tmp_path, monkeypatch):
        def refuse(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(builder_module, "build_record", refuse)

        assert write_provenance_file("lego_x", tmp_path) is None


class _FakeProcess:
    """Stands in for QProcess in the export step: records how it is set up, starts nothing."""

    MergedChannels = 0
    created = []

    def __init__(self, parent=None):
        self.arguments = []
        self.started = False
        self.readyRead = _FakeSignal()
        self.finished = _FakeSignal()
        _FakeProcess.created.append(self)

    def setProgram(self, program): ...
    def setArguments(self, arguments): self.arguments = list(arguments)
    def setWorkingDirectory(self, directory): ...
    def setProcessEnvironment(self, environment): ...
    def setProcessChannelMode(self, mode): ...
    def start(self): self.started = True


class _FakeSignal:
    def connect(self, slot):
        self.slot = slot


class TestExportStep:
    def _builder(self, monkeypatch, source_root: Path) -> DigitalTwinBuilder:
        _FakeProcess.created.clear()
        monkeypatch.setattr(QtCore, "QProcess", _FakeProcess)
        monkeypatch.setattr(builder_module, "NORMALIZED_SOURCE_ROOT", source_root)
        builder = DigitalTwinBuilder()
        builder._scene_name = "lego_x"
        return builder

    def test_the_export_is_given_the_build_record_and_the_file_is_removed_afterwards(self, tmp_path, monkeypatch):
        _normalized_source(tmp_path / "lego_x")
        builder = self._builder(monkeypatch, tmp_path)

        builder._start_export(tmp_path / "refined.pt")

        (process,) = _FakeProcess.created
        record_path = Path(process.arguments[process.arguments.index("--provenance") + 1])
        assert process.started and json.loads(record_path.read_text(encoding="utf-8"))["scene"] == "lego_x"
        done = tmp_path / "twin.npz"
        done.write_bytes(b"x")
        builder._on_export_finished(0, None, done)
        assert not record_path.exists()

    def test_the_rest_of_the_export_command_is_what_it_was(self, tmp_path, monkeypatch):
        _normalized_source(tmp_path / "lego_x")
        builder = self._builder(monkeypatch, tmp_path)

        builder._start_export(tmp_path / "refined.pt")

        arguments = _FakeProcess.created[0].arguments
        for flag in ("--scene-path", "--gs-output-dir", "--refined-checkpoint", "--iteration-to-load", "--out"):
            assert flag in arguments
        assert arguments[0] == "-u" and arguments[1].endswith("export_digital_twin_package.py")

    def test_a_build_goes_ahead_without_a_record_if_the_record_cannot_be_written(self, tmp_path, monkeypatch):
        builder = self._builder(monkeypatch, tmp_path)
        monkeypatch.setattr(builder_module, "write_provenance_file", lambda scene, source: None)

        builder._start_export(tmp_path / "refined.pt")

        (process,) = _FakeProcess.created
        assert process.started and "--provenance" not in process.arguments
        done = tmp_path / "twin.npz"
        done.write_bytes(b"x")
        builder._on_export_finished(0, None, done)  # nothing to remove, and no error
