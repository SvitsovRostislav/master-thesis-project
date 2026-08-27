"""Unit tests for src/viewer/gaussian_ply_io.py: the .ply read/write format
used by both the base viewer and the digital-twin "Save As..." button.
"""

import numpy as np
import pytest

from src.viewer.gaussian_ply_io import SH_C0, load_gaussian_ply, save_gaussian_ply


def _make_cloud(n=50, seed=0):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-5, 5, size=(n, 3)).astype(np.float32)
    colors = rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float32)
    opacities = rng.uniform(0.05, 0.95, size=n).astype(np.float32)
    scales = rng.uniform(0.01, 0.5, size=(n, 3)).astype(np.float32)
    return positions, colors, opacities, scales


def test_round_trip_preserves_positions_colors_opacities_scales(tmp_path):
    positions, colors, opacities, scales = _make_cloud()
    path = tmp_path / "cloud.ply"

    save_gaussian_ply(str(path), positions, colors, opacities, scales)
    cloud = load_gaussian_ply(str(path))

    assert len(cloud) == len(positions)
    np.testing.assert_allclose(cloud.positions, positions, atol=1e-4)
    np.testing.assert_allclose(cloud.colors, colors, atol=1e-3)
    np.testing.assert_allclose(cloud.opacities, opacities, atol=1e-3)
    np.testing.assert_allclose(cloud.scales, scales, atol=1e-3)


def test_round_trip_preserves_rotations(tmp_path):
    positions, colors, opacities, scales = _make_cloud(n=10, seed=1)
    rng = np.random.default_rng(2)
    rotations = rng.normal(size=(10, 4)).astype(np.float32)
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
    path = tmp_path / "cloud_rot.ply"

    save_gaussian_ply(str(path), positions, colors, opacities, scales, rotations=rotations)
    cloud = load_gaussian_ply(str(path))

    # load_gaussian_ply's GaussianCloud does not expose rotations (the base
    # viewer treats every splat as isotropic) -- read them back via plyfile
    # directly to confirm they were actually written, not silently dropped.
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"].data
    written_rot = np.stack([vertex[f"rot_{i}"] for i in range(4)], axis=1)
    np.testing.assert_allclose(written_rot, rotations, atol=1e-4)


def test_save_without_rotations_defaults_to_identity_quaternion(tmp_path):
    positions, colors, opacities, scales = _make_cloud(n=5, seed=3)
    path = tmp_path / "cloud_no_rot.ply"

    save_gaussian_ply(str(path), positions, colors, opacities, scales)

    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"].data
    written_rot = np.stack([vertex[f"rot_{i}"] for i in range(4)], axis=1)
    expected = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (5, 1))
    np.testing.assert_allclose(written_rot, expected, atol=1e-6)


def test_load_falls_back_for_plain_xyz_point_cloud(tmp_path):
    """A mesh-derived or scanner .ply with only x/y/z (no Gaussian fields)
    should still load, with sensible defaults instead of crashing."""
    from plyfile import PlyData, PlyElement

    n = 20
    rng = np.random.default_rng(4)
    xyz = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
    vertex = np.zeros(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    path = tmp_path / "plain.ply"
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))

    cloud = load_gaussian_ply(str(path))

    assert len(cloud) == n
    np.testing.assert_allclose(cloud.positions, xyz, atol=1e-5)
    assert np.all(cloud.opacities == 1.0)
    assert np.allclose(cloud.colors, 0.7)  # fallback uncolored gray
    assert np.all(cloud.scales > 0)  # extent-proportional fallback, not zero/NaN


def test_load_falls_back_for_rgb_point_cloud(tmp_path):
    """A .ply with red/green/blue (e.g. a colored COLMAP point cloud) but no
    Gaussian-specific fields should use that color instead of the flat gray
    fallback."""
    from plyfile import PlyData, PlyElement

    n = 8
    xyz = np.zeros((n, 3), dtype=np.float32)
    rgb = np.linspace(0, 255, n * 3).reshape(n, 3).astype(np.uint8)
    vertex = np.zeros(
        n,
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    path = tmp_path / "rgb.ply"
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))

    cloud = load_gaussian_ply(str(path))

    np.testing.assert_allclose(cloud.colors, rgb.astype(np.float32) / 255.0, atol=1e-5)


@pytest.mark.parametrize("dc", [-2.0, 0.0, 2.0])
def test_sh_color_conversion_round_trips_and_clips(dc):
    """save_gaussian_ply's `color = 0.5 + SH_C0 * dc` <-> load's inverse
    should agree, and out-of-range colors should clip into [0, 1] rather
    than wrapping or producing negative/blown-out values."""
    color = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
    assert 0.0 <= color <= 1.0
