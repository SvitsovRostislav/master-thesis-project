import time

import numpy as np

import src.viewer.gaussian_density as gaussian_density
from src.viewer.digital_twin import DigitalTwinPackage
from src.viewer.gaussian_density import DensityBaseline, rebalance_density, triangle_areas


def _make_single_triangle_package(n_gaussians: int = 4) -> DigitalTwinPackage:
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    face_idx = np.zeros(n_gaussians, dtype=np.int32)
    # Spread the Gaussians out so donors aren't all identical.
    bary = np.tile(np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32), (n_gaussians, 1))
    colors = np.linspace(0, 1, n_gaussians)[:, None].repeat(3, axis=1).astype(np.float32)
    opacities = np.full(n_gaussians, 0.8, dtype=np.float32)
    scales = np.ones((n_gaussians, 3), dtype=np.float32)
    quaternions = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n_gaussians, 1))
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


class TestTriangleAreas:
    def test_right_triangle_area(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        np.testing.assert_allclose(triangle_areas(vertices, faces), [0.5], atol=1e-6)

    def test_scaling_the_triangle_scales_area_quadratically(self):
        vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        np.testing.assert_allclose(triangle_areas(vertices, faces), [2.0], atol=1e-6)


class TestDensityBaseline:
    def test_capture_matches_current_area_and_count(self):
        package = _make_single_triangle_package(n_gaussians=5)
        baseline = DensityBaseline.capture(package)

        np.testing.assert_allclose(baseline.area, [0.5], atol=1e-6)
        np.testing.assert_array_equal(baseline.count, [5])


class TestRebalanceDensity:
    def test_no_change_when_area_is_unchanged(self):
        package = _make_single_triangle_package(n_gaussians=4)
        baseline = DensityBaseline.capture(package)

        changed = rebalance_density(package, baseline)

        assert changed is False
        assert len(package) == 4

    def test_stretching_the_triangle_adds_gaussians(self):
        package = _make_single_triangle_package(n_gaussians=4)
        baseline = DensityBaseline.capture(package)

        # Double both legs -> area quadruples -> expect ~4x the Gaussians.
        package.mesh_vertices[1] = [2.0, 0.0, 0.0]
        package.mesh_vertices[2] = [0.0, 2.0, 0.0]

        changed = rebalance_density(package, baseline)

        assert changed is True
        assert len(package) == 16
        # New Gaussians must land inside the (now larger) triangle.
        positions = package.recompute_gaussian_positions()
        assert np.all(positions[:, 0] >= -1e-5)
        assert np.all(positions[:, 1] >= -1e-5)

    def test_compressing_never_drops_below_the_baseline_count(self):
        package = _make_single_triangle_package(n_gaussians=6)
        baseline = DensityBaseline.capture(package)

        # Shrink the triangle to a tenth of its area.
        package.mesh_vertices[1] = [0.1, 0.0, 0.0]
        package.mesh_vertices[2] = [0.0, 0.1, 0.0]

        rebalance_density(package, baseline)

        assert len(package) == 6  # floored at the original count, not below

    def test_grow_then_shrink_back_returns_to_the_original_count(self):
        package = _make_single_triangle_package(n_gaussians=4)
        baseline = DensityBaseline.capture(package)

        package.mesh_vertices[1] = [3.0, 0.0, 0.0]
        package.mesh_vertices[2] = [0.0, 3.0, 0.0]
        rebalance_density(package, baseline)
        assert len(package) > 4

        package.mesh_vertices[1] = [1.0, 0.0, 0.0]
        package.mesh_vertices[2] = [0.0, 1.0, 0.0]
        rebalance_density(package, baseline)

        assert len(package) == 4  # baseline is fixed, not ratcheted up by the stretch

    def test_returns_false_and_is_a_no_op_if_face_count_changed_since_capture(self):
        package = _make_single_triangle_package(n_gaussians=4)
        baseline = DensityBaseline.capture(package)
        package.mesh_faces = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32)

        changed = rebalance_density(package, baseline)

        assert changed is False

    def test_an_extreme_single_stretch_is_capped_per_triangle(self):
        """Regression test for the reported hang: one very large drag must
        not ask for thousands of new Gaussians on a single triangle."""
        package = _make_single_triangle_package(n_gaussians=4)
        baseline = DensityBaseline.capture(package)

        package.mesh_vertices[1] = [1000.0, 0.0, 0.0]
        package.mesh_vertices[2] = [0.0, 1000.0, 0.0]

        rebalance_density(package, baseline)

        assert len(package) <= 4 * gaussian_density.MAX_GROWTH_RATIO

    def test_many_faces_stretching_at_once_respects_the_global_budget(self):
        """Regression test for the reported hang: a brush touching many
        triangles at once must not let the *total* Gaussian count run
        away, even though each triangle's own cap would individually
        allow it."""
        n_faces = 500
        vertices = np.zeros((n_faces * 3, 3), dtype=np.float32)
        faces = np.arange(n_faces * 3, dtype=np.int32).reshape(n_faces, 3)
        for i in range(n_faces):
            vertices[i * 3 : i * 3 + 3] = [[i, 0, 0], [i + 1, 0, 0], [i, 1, 0]]
        gaussian_face_idx = np.repeat(np.arange(n_faces, dtype=np.int32), 2)
        n_gaussians = gaussian_face_idx.shape[0]
        package = DigitalTwinPackage(
            mesh_vertices=vertices,
            mesh_faces=faces,
            gaussian_face_idx=gaussian_face_idx,
            gaussian_bary=np.tile([1 / 3, 1 / 3, 1 / 3], (n_gaussians, 1)).astype(np.float32),
            gaussian_colors=np.zeros((n_gaussians, 3), dtype=np.float32),
            gaussian_opacities=np.full(n_gaussians, 0.8, dtype=np.float32),
            gaussian_scales=np.ones((n_gaussians, 3), dtype=np.float32),
            gaussian_quaternions=np.tile([1, 0, 0, 0], (n_gaussians, 1)).astype(np.float32),
        )
        baseline = DensityBaseline.capture(package)

        package.mesh_vertices[:, :2] *= 20.0

        started = time.perf_counter()
        rebalance_density(package, baseline)
        elapsed = time.perf_counter() - started

        original_total = n_gaussians
        assert len(package) <= original_total * gaussian_density.MAX_TOTAL_GROWTH_RATIO
        assert elapsed < 5.0  # guards against the old O(faces x gaussians) per-face scan
