"""Unit tests for the pure path-logic helpers in
src/viewer/digital_twin_builder.py (no GPU/subprocess needed). The QProcess
orchestration itself (DigitalTwinBuilder) is exercised manually against a
real scene -- see experiments/02_mesh_based_digital_twin_editing/README.md.
"""

from pathlib import Path

from src.viewer.digital_twin_builder import (
    ColmapSource,
    digital_twin_cache_path,
    find_colmap_source,
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
        """The real data/Test data/ layout: lego_bugatti and lego_ferrari
        both have a colmap_data/frame_0000/ -- the collision this whole
        feature has to avoid is in the *scene name* SuGaR derives output
        folders from, not the frame folder name itself."""
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
