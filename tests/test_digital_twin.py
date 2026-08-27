"""Unit tests for src/viewer/digital_twin.py: the mesh<->Gaussian binding
that powers both live editing (recompute_gaussian_positions) and the
Select/Delete/Undo tools (remove_faces_by_vertex_selection, clone).
"""

import numpy as np
import pytest

from src.viewer.digital_twin import (
    DigitalTwinPackage,
    load_digital_twin_package,
    save_digital_twin_package,
)


def _make_strip_package():
    """A small, hand-checkable mesh: a 2x1 quad strip (5 verts, 3 faces),
    one Gaussian per face sitting at its centroid."""
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

    package.mesh_vertices[1] += [10.0, 0.0, 0.0]  # vertex 1 touches all 3 faces
    after = package.recompute_gaussian_positions()

    assert not np.allclose(before, after)
    # Each face includes vertex 1 with weight 1/3, so every Gaussian should
    # shift by exactly (10, 0, 0) / 3.
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
    # And the reverse: mutating the clone must not affect the original.
    snapshot.mesh_vertices[1] = [-1, -1, -1]
    assert not np.allclose(package.mesh_vertices[1], [-1, -1, -1])


def test_remove_faces_by_vertex_selection_drops_touching_faces_and_gaussians():
    package = _make_strip_package()
    original_colors = package.gaussian_colors.copy()

    # Vertex 2 is touched only by face 2 ([1, 2, 4]).
    package.remove_faces_by_vertex_selection(np.array([2]))

    assert package.mesh_vertices.shape == (4, 3)
    assert package.mesh_faces.shape == (2, 3)
    assert len(package) == 2
    # Faces 0 and 1's Gaussians (red, green) should survive in order; face
    # 2's (blue) should be gone.
    np.testing.assert_allclose(package.gaussian_colors, original_colors[:2])


def test_remove_faces_by_vertex_selection_remaps_indices_consistently():
    package = _make_strip_package()
    package.remove_faces_by_vertex_selection(np.array([2]))

    # Every remapped face index must point at a valid (in-range) vertex,
    # and recompute_gaussian_positions must not raise/index out of bounds.
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
    """Vertex 0 is touched only by face 0 ([0, 1, 3]); faces 1 ([1, 4, 3])
    and 2 ([1, 2, 4]) must survive with their vertex indices shifted down
    by exactly one (vertex 0 removed, so old index i>0 becomes i-1)."""
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
