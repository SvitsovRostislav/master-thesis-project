import numpy as np
import pytest

import src.viewer.gaussian_deformation as gaussian_deformation
from src.viewer.digital_twin import DigitalTwinPackage
from src.viewer.gaussian_deformation import (
    DeformationBaseline,
    _matrices_to_quaternions,
    _quaternions_to_matrices,
    adapt_gaussians_to_deformation,
)

_RNG = np.random.default_rng(7)


def _random_rotations(n: int) -> np.ndarray:
    q = _RNG.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return _quaternions_to_matrices(q)


def _rotation_about(axis, degrees) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    a = np.radians(degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)


def _covariances(package: DigitalTwinPackage) -> np.ndarray:
    r = _quaternions_to_matrices(package.gaussian_quaternions)
    s = package.gaussian_scales.astype(np.float64)
    return np.matmul(r * s[:, None, :] ** 2, np.transpose(r, (0, 2, 1)))


def _triangle_package(n_gaussians: int = 5, vertices=None) -> DigitalTwinPackage:
    if vertices is None:
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    quats = _matrices_to_quaternions(_random_rotations(n_gaussians)).astype(np.float32)
    scales = _RNG.uniform(0.02, 0.2, size=(n_gaussians, 3)).astype(np.float32)
    return DigitalTwinPackage(
        mesh_vertices=vertices,
        mesh_faces=faces,
        gaussian_face_idx=np.zeros(n_gaussians, dtype=np.int32),
        gaussian_bary=np.tile(np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32), (n_gaussians, 1)),
        gaussian_colors=np.zeros((n_gaussians, 3), dtype=np.float32),
        gaussian_opacities=np.full(n_gaussians, 0.8, dtype=np.float32),
        gaussian_scales=scales,
        gaussian_quaternions=quats,
    )


class TestQuaternionMatrixConversion:
    def test_round_trip_recovers_the_rotation(self):
        rotations = _random_rotations(500)

        recovered = _quaternions_to_matrices(_matrices_to_quaternions(rotations))

        np.testing.assert_allclose(recovered, rotations, atol=1e-10)

    def test_identity_maps_to_the_identity_quaternion(self):
        q = _matrices_to_quaternions(np.eye(3)[None])

        np.testing.assert_allclose(q, [[1, 0, 0, 0]], atol=1e-12)

    def test_covers_all_four_shepperd_branches(self):
        rotations = np.stack(
            [np.eye(3), _rotation_about([1, 0, 0], 180), _rotation_about([0, 1, 0], 180), _rotation_about([0, 0, 1], 180)]
        )

        recovered = _quaternions_to_matrices(_matrices_to_quaternions(rotations))

        np.testing.assert_allclose(recovered, rotations, atol=1e-10)


class TestAdaptGaussians:
    def test_translation_changes_nothing(self):
        package = _triangle_package()
        baseline = DeformationBaseline.capture(package)
        scales, quats = package.gaussian_scales.copy(), package.gaussian_quaternions.copy()

        package.mesh_vertices += np.array([5.0, -2.0, 3.0], dtype=np.float32)
        updated = adapt_gaussians_to_deformation(package, baseline)

        assert updated == 0
        np.testing.assert_array_equal(package.gaussian_scales, scales)
        np.testing.assert_array_equal(package.gaussian_quaternions, quats)

    def test_rotating_the_triangle_rotates_each_gaussian_rigidly(self):
        package = _triangle_package()
        baseline = DeformationBaseline.capture(package)
        before = _covariances(package)
        rot = _rotation_about([1, 1, 0], 40)

        package.mesh_vertices = (package.mesh_vertices.astype(np.float64) @ rot.T).astype(np.float32)
        updated = adapt_gaussians_to_deformation(package, baseline)

        assert updated == len(package)
        expected = np.matmul(np.matmul(rot, before), rot.T)
        np.testing.assert_allclose(_covariances(package), expected, atol=2e-6)

    def test_in_plane_stretch_stretches_the_covariance_the_same_way(self):
        package = _triangle_package()
        baseline = DeformationBaseline.capture(package)
        before = _covariances(package)
        m = np.diag([2.0, 1.0, 1.0])

        package.mesh_vertices = (package.mesh_vertices.astype(np.float64) @ m.T).astype(np.float32)
        adapt_gaussians_to_deformation(package, baseline)

        np.testing.assert_allclose(_covariances(package), m @ before @ m.T, atol=2e-6)

    def test_shear_is_handled_exactly_too(self):
        package = _triangle_package()
        baseline = DeformationBaseline.capture(package)
        before = _covariances(package)
        m = np.array([[1.0, 0.6, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        package.mesh_vertices = (package.mesh_vertices.astype(np.float64) @ m.T).astype(np.float32)
        adapt_gaussians_to_deformation(package, baseline)

        np.testing.assert_allclose(_covariances(package), m @ before @ m.T, atol=2e-6)

    def test_stretch_then_squash_back_restores_the_original_covariance(self):
        package = _triangle_package()
        baseline = DeformationBaseline.capture(package)
        original = _covariances(package)
        rest = package.mesh_vertices.copy()

        package.mesh_vertices = rest * np.array([3.0, 0.5, 1.0], dtype=np.float32)
        adapt_gaussians_to_deformation(package, baseline)
        package.mesh_vertices = rest.copy()
        adapt_gaussians_to_deformation(package, baseline)

        np.testing.assert_allclose(_covariances(package), original, atol=1e-5)

    def test_a_second_call_with_no_further_change_is_a_no_op(self):
        package = _triangle_package()
        baseline = DeformationBaseline.capture(package)
        package.mesh_vertices = package.mesh_vertices * np.array([2.0, 1.0, 1.0], dtype=np.float32)
        assert adapt_gaussians_to_deformation(package, baseline) > 0
        after_first = _covariances(package)

        assert adapt_gaussians_to_deformation(package, baseline) == 0
        np.testing.assert_array_equal(_covariances(package), after_first)

    def test_only_gaussians_on_deformed_triangles_are_touched(self):
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 0], [6, 5, 0], [5, 6, 0]], dtype=np.float32
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        n = 6
        package = DigitalTwinPackage(
            mesh_vertices=vertices,
            mesh_faces=faces,
            gaussian_face_idx=np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
            gaussian_bary=np.full((n, 3), 1 / 3, dtype=np.float32),
            gaussian_colors=np.zeros((n, 3), dtype=np.float32),
            gaussian_opacities=np.full(n, 0.8, dtype=np.float32),
            gaussian_scales=_RNG.uniform(0.02, 0.2, size=(n, 3)).astype(np.float32),
            gaussian_quaternions=_matrices_to_quaternions(_random_rotations(n)).astype(np.float32),
        )
        baseline = DeformationBaseline.capture(package)
        scales, quats = package.gaussian_scales.copy(), package.gaussian_quaternions.copy()

        package.mesh_vertices[4] = [9.0, 5.0, 0.0]  # stretches only the second triangle
        updated = adapt_gaussians_to_deformation(package, baseline)

        assert updated == 3
        np.testing.assert_array_equal(package.gaussian_scales[:3], scales[:3])
        np.testing.assert_array_equal(package.gaussian_quaternions[:3], quats[:3])
        assert not np.allclose(package.gaussian_scales[3:], scales[3:])

    def test_a_collapsed_triangle_is_skipped_without_producing_nans(self):
        package = _triangle_package()
        baseline = DeformationBaseline.capture(package)
        scales, quats = package.gaussian_scales.copy(), package.gaussian_quaternions.copy()

        package.mesh_vertices[2] = [2.0, 0.0, 0.0]  # all three vertices now on a line

        assert adapt_gaussians_to_deformation(package, baseline) == 0
        np.testing.assert_array_equal(package.gaussian_scales, scales)
        np.testing.assert_array_equal(package.gaussian_quaternions, quats)

    def test_extreme_stretch_is_clamped(self):
        package = _triangle_package()
        baseline = DeformationBaseline.capture(package)
        largest_before = package.gaussian_scales.max()

        package.mesh_vertices = package.mesh_vertices * np.array([1000.0, 1.0, 1.0], dtype=np.float32)
        adapt_gaussians_to_deformation(package, baseline)

        assert np.all(np.isfinite(package.gaussian_scales))
        assert package.gaussian_scales.max() <= largest_before * gaussian_deformation.MAX_STRETCH * 1.001

    def test_stale_baseline_after_a_topology_change_is_a_no_op(self):
        package = _triangle_package()
        baseline = DeformationBaseline.capture(package)
        package.mesh_faces = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32)

        assert adapt_gaussians_to_deformation(package, baseline) == 0
