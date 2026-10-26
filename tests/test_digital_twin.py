import json

import numpy as np
import pytest

from src.viewer.digital_twin import (
    DigitalTwinPackage,
    load_digital_twin_package,
    save_digital_twin_package,
)


def _make_strip_package():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
    )
    faces = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4]], dtype=np.int32)
    bary = np.tile(np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32), (3, 1))
    face_idx = np.arange(3, dtype=np.int32)
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    opacities = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    scales = np.ones((3, 3), dtype=np.float32)
    quaternions = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (3, 1))
    return DigitalTwinPackage(
        mesh_vertices=vertices,
        mesh_faces=faces,
        gaussian_face_idx=face_idx,
        gaussian_bary=bary,
        gaussian_colors=colors,
        gaussian_opacities=opacities,
        gaussian_scales=scales,
        gaussian_quaternions=quaternions,
    )


def test_recompute_gaussian_positions_is_centroid_of_bound_face():
    package = _make_strip_package()
    positions = package.recompute_gaussian_positions()

    expected = package.mesh_vertices[package.mesh_faces].mean(axis=1)
    np.testing.assert_allclose(positions, expected, atol=1e-6)


def test_recompute_gaussian_positions_follows_moved_vertex():
    package = _make_strip_package()
    before = package.recompute_gaussian_positions()

    package.mesh_vertices[1] += [10.0, 0.0, 0.0]
    after = package.recompute_gaussian_positions()

    assert not np.allclose(before, after)
    np.testing.assert_allclose(after - before, np.tile([10.0 / 3, 0.0, 0.0], (3, 1)), atol=1e-5)


def test_len_matches_gaussian_count():
    package = _make_strip_package()
    assert len(package) == 3


def test_clone_is_a_deep_copy():
    package = _make_strip_package()
    snapshot = package.clone()

    package.mesh_vertices[0] = [99, 99, 99]
    package.gaussian_colors[0] = [0, 0, 0]

    assert not np.allclose(snapshot.mesh_vertices[0], [99, 99, 99])
    assert not np.allclose(snapshot.gaussian_colors[0], [0, 0, 0])
    snapshot.mesh_vertices[1] = [-1, -1, -1]
    assert not np.allclose(package.mesh_vertices[1], [-1, -1, -1])


def test_remove_faces_by_vertex_selection_drops_touching_faces_and_gaussians():
    package = _make_strip_package()
    original_colors = package.gaussian_colors.copy()

    package.remove_faces_by_vertex_selection(np.array([2]))

    assert package.mesh_vertices.shape == (4, 3)
    assert package.mesh_faces.shape == (2, 3)
    assert len(package) == 2
    np.testing.assert_allclose(package.gaussian_colors, original_colors[:2])


def test_remove_faces_by_vertex_selection_remaps_indices_consistently():
    package = _make_strip_package()
    package.remove_faces_by_vertex_selection(np.array([2]))
    assert package.mesh_faces.max() < package.mesh_vertices.shape[0]
    assert package.gaussian_face_idx.max() < package.mesh_faces.shape[0]
    positions = package.recompute_gaussian_positions()
    assert positions.shape == (len(package), 3)


def test_remove_all_vertices_yields_empty_package():
    package = _make_strip_package()
    package.remove_faces_by_vertex_selection(np.arange(5))

    assert package.mesh_vertices.shape == (0, 3)
    assert package.mesh_faces.shape == (0, 3)
    assert len(package) == 0


def test_remove_faces_remaps_surviving_faces_exactly():
    package = _make_strip_package()

    package.remove_faces_by_vertex_selection(np.array([0]))

    np.testing.assert_array_equal(package.mesh_faces, np.array([[0, 3, 2], [0, 1, 3]]))


def test_save_and_load_round_trip(tmp_path):
    package = _make_strip_package()
    path = tmp_path / "twin.npz"

    save_digital_twin_package(str(path), package)
    loaded = load_digital_twin_package(str(path))

    np.testing.assert_allclose(loaded.mesh_vertices, package.mesh_vertices)
    np.testing.assert_array_equal(loaded.mesh_faces, package.mesh_faces)
    np.testing.assert_array_equal(loaded.gaussian_face_idx, package.gaussian_face_idx)
    np.testing.assert_allclose(loaded.gaussian_bary, package.gaussian_bary)
    np.testing.assert_allclose(loaded.gaussian_colors, package.gaussian_colors)
    np.testing.assert_allclose(loaded.gaussian_opacities, package.gaussian_opacities)
    np.testing.assert_allclose(loaded.gaussian_scales, package.gaussian_scales)
    np.testing.assert_allclose(loaded.gaussian_quaternions, package.gaussian_quaternions)
    assert len(loaded) == len(package)


def test_add_gaussians_appends_to_every_array():
    package = _make_strip_package()
    original_len = len(package)

    package.add_gaussians(
        face_idx=np.array([0, 1], dtype=np.int32),
        bary=np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]], dtype=np.float32),
        colors=np.array([[1, 1, 1], [0, 0, 0]], dtype=np.float32),
        opacities=np.array([0.9, 0.1], dtype=np.float32),
        scales=np.ones((2, 3), dtype=np.float32),
        quaternions=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (2, 1)),
    )

    assert len(package) == original_len + 2
    np.testing.assert_array_equal(package.gaussian_face_idx[-2:], [0, 1])
    positions = package.recompute_gaussian_positions()
    assert positions.shape[0] == original_len + 2


def test_remove_gaussians_by_index_drops_only_those_gaussians():
    package = _make_strip_package()
    original_colors = package.gaussian_colors.copy()

    package.remove_gaussians_by_index(np.array([1]))

    assert len(package) == 2
    np.testing.assert_allclose(package.gaussian_colors, original_colors[[0, 2]])
    # mesh itself is untouched by a Gaussian-only removal
    assert package.mesh_vertices.shape == (5, 3)
    assert package.mesh_faces.shape == (3, 3)


RECORD = {
    "schema": 1,
    "built_at": "2026-03-05T14:07:09",
    "scene": "lego_x",
    "masked": True,
    "preset": {"refine_iterations": 2000},
}


class TestProvenance:
    """The optional build record (twin_provenance.py) stored in a package."""

    def test_a_record_survives_a_save_and_load(self, tmp_path):
        package = _make_strip_package()
        package.provenance = RECORD

        save_digital_twin_package(str(tmp_path / "twin.npz"), package)

        assert load_digital_twin_package(str(tmp_path / "twin.npz")).provenance == RECORD

    def test_a_package_without_a_record_writes_no_extra_key_and_loads_with_none(self, tmp_path):
        save_digital_twin_package(str(tmp_path / "twin.npz"), _make_strip_package())

        with np.load(tmp_path / "twin.npz") as data:
            assert "provenance_json" not in data.files
        assert load_digital_twin_package(str(tmp_path / "twin.npz")).provenance is None

    def test_a_package_saved_before_records_existed_still_loads(self, tmp_path):
        package = _make_strip_package()
        np.savez(
            tmp_path / "old.npz",
            mesh_vertices=package.mesh_vertices, mesh_faces=package.mesh_faces,
            gaussian_face_idx=package.gaussian_face_idx, gaussian_bary=package.gaussian_bary,
            gaussian_colors=package.gaussian_colors, gaussian_opacities=package.gaussian_opacities,
            gaussian_scales=package.gaussian_scales, gaussian_quaternions=package.gaussian_quaternions,
        )

        loaded = load_digital_twin_package(str(tmp_path / "old.npz"))

        assert loaded.provenance is None and len(loaded) == 3

    def test_a_damaged_record_is_ignored_and_the_twin_still_loads(self, tmp_path):
        package = _make_strip_package()
        save_digital_twin_package(str(tmp_path / "twin.npz"), package)
        with np.load(tmp_path / "twin.npz") as data:
            arrays = {name: data[name] for name in data.files}
        np.savez(tmp_path / "damaged.npz", provenance_json=np.array("{not json"), **arrays)

        loaded = load_digital_twin_package(str(tmp_path / "damaged.npz"))

        assert loaded.provenance is None
        np.testing.assert_allclose(loaded.mesh_vertices, package.mesh_vertices)

    def test_the_record_is_stored_as_json_text(self, tmp_path):
        package = _make_strip_package()
        package.provenance = RECORD

        save_digital_twin_package(str(tmp_path / "twin.npz"), package)

        with np.load(tmp_path / "twin.npz") as data:
            assert json.loads(str(data["provenance_json"])) == RECORD

    def test_a_clone_has_its_own_copy_of_the_record(self):
        package = _make_strip_package()
        package.provenance = {"masked": True, "preset": {"refine_iterations": 2000}}

        clone = package.clone()
        package.provenance["preset"]["refine_iterations"] = 1

        assert clone.provenance == {"masked": True, "preset": {"refine_iterations": 2000}}

    def test_editing_the_mesh_keeps_the_record(self):
        package = _make_strip_package()
        package.provenance = RECORD

        package.remove_faces_by_vertex_selection(np.array([0]))
        package.add_gaussians(
            face_idx=np.array([0], dtype=np.int32), bary=np.array([[0.2, 0.3, 0.5]], dtype=np.float32),
            colors=np.ones((1, 3), dtype=np.float32), opacities=np.array([0.5], dtype=np.float32),
            scales=np.ones((1, 3), dtype=np.float32), quaternions=np.array([[1, 0, 0, 0]], dtype=np.float32),
        )
        package.remove_gaussians_by_index(np.array([0]))

        assert package.provenance == RECORD


class TestOrigins:

    def test_nothing_is_tracked_until_asked_and_every_method_behaves_as_before(self):
        package = _make_strip_package()

        package.remove_faces_by_vertex_selection(np.array([2]))
        package.remove_gaussians_by_index(np.array([0]))
        clone = package.clone()

        assert package.vertex_origin is None and package.face_origin is None
        assert clone.vertex_origin is None and clone.face_origin is None

    def test_starting_gives_every_vertex_and_face_its_own_index(self):
        package = _make_strip_package()

        package.start_tracking_origins()

        assert package.vertex_origin.tolist() == [0, 1, 2, 3, 4]
        assert package.face_origin.tolist() == [0, 1, 2]

    def test_after_two_deletes_every_vertex_and_face_points_back_to_where_it_was(self):
        package = _make_strip_package()
        package.start_tracking_origins()
        loaded_vertices, loaded_faces = package.mesh_vertices.copy(), package.mesh_faces.copy()

        package.remove_faces_by_vertex_selection(np.array([2]))  # takes face 2 and vertex 2
        package.remove_faces_by_vertex_selection(np.array([3]))  # vertex 3 is loaded vertex 4 now

        assert package.vertex_origin.tolist() == [0, 1, 3]
        assert package.face_origin.tolist() == [0]
        assert np.array_equal(loaded_vertices[package.vertex_origin], package.mesh_vertices)
        assert np.array_equal(loaded_vertices[loaded_faces[package.face_origin]], package.mesh_vertices[package.mesh_faces])

    def test_adding_or_removing_gaussians_leaves_the_origins_alone(self):
        package = _make_strip_package()
        package.start_tracking_origins()

        package.add_gaussians(
            face_idx=np.array([0], dtype=np.int32), bary=np.array([[0.2, 0.3, 0.5]], dtype=np.float32),
            colors=np.ones((1, 3), dtype=np.float32), opacities=np.array([0.5], dtype=np.float32),
            scales=np.ones((1, 3), dtype=np.float32), quaternions=np.array([[1, 0, 0, 0]], dtype=np.float32),
        )
        package.remove_gaussians_by_index(np.array([1]))

        assert package.vertex_origin.tolist() == [0, 1, 2, 3, 4]
        assert package.face_origin.tolist() == [0, 1, 2]

    def test_a_clone_has_its_own_origins_so_undo_gets_them_back(self):
        package = _make_strip_package()
        package.start_tracking_origins()
        snapshot = package.clone()

        package.remove_faces_by_vertex_selection(np.array([0]))

        assert snapshot.vertex_origin.tolist() == [0, 1, 2, 3, 4]
        assert snapshot.face_origin.tolist() == [0, 1, 2]

    def test_a_clone_does_not_share_its_origin_arrays(self):
        package = _make_strip_package()
        package.start_tracking_origins()
        snapshot = package.clone()

        package.vertex_origin[0] = package.face_origin[0] = 99  # written in place, before anything replaces them

        assert snapshot.vertex_origin[0] == 0 and snapshot.face_origin[0] == 0

    def test_origins_are_not_saved(self, tmp_path):
        package = _make_strip_package()
        package.start_tracking_origins()
        package.remove_faces_by_vertex_selection(np.array([0]))

        save_digital_twin_package(str(tmp_path / "twin.npz"), package)

        with np.load(tmp_path / "twin.npz") as data:
            assert not any("origin" in name for name in data.files)
        loaded = load_digital_twin_package(str(tmp_path / "twin.npz"))
        assert loaded.vertex_origin is None and loaded.face_origin is None
