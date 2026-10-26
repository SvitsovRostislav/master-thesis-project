import numpy as np
import pytest

from src.viewer.digital_twin import load_digital_twin_package
from src.viewer.gaussian_deformation import _quaternions_to_matrices
from src.viewer.mesh_detach import detach_selection
from src.viewer.mesh_fill import boundary_edges, close_cut, min_area_triangulation, simple_cycles
from src.viewer.sam_pick import ReferenceGeometry

N = 6


def _block(first_row, first_col, size, n=N):
    return np.array([i * n + j for i in range(first_row, first_row + size) for j in range(first_col, first_col + size)])


@pytest.fixture
def grid(tmp_path, make_grid_twin):
    make_grid_twin(tmp_path / "grid.npz", n=N)
    package = load_digital_twin_package(str(tmp_path / "grid.npz"))
    package.start_tracking_origins()
    return package


def _cut(package, vertices):
    result = detach_selection(package, vertices)
    return np.concatenate([result.cut_originals, result.cut_copies])


def _directed_edges(faces):
    faces = np.asarray(faces, dtype=np.int64)
    return np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])


class TestCloseCut:
    def test_both_sides_of_an_inner_cut_are_closed(self, grid):
        cut = _cut(grid, _block(1, 1, 3))
        faces_before = grid.mesh_faces.shape[0]

        result = close_cut(grid, cut)

        assert result.cycles_closed == 2  # the part's rim and the hole it left
        assert result.cut_edges_open_before == 16 and result.cut_edges_open_after == 0
        assert result.faces_added == 12 and result.gaussians_added == 12  # an 8-gon takes 6 triangles, twice
        assert result.vertices_added == 0
        assert grid.mesh_faces.shape[0] == faces_before + 12 and len(grid) == faces_before + 12

    def test_the_patches_are_oriented_like_the_mesh_and_leave_no_edge_with_three_triangles(self, grid):
        close_cut(grid, _cut(grid, _block(1, 1, 3)))

        directed = _directed_edges(grid.mesh_faces)
        assert np.unique(directed, axis=0).shape[0] == directed.shape[0]  # no edge runs the same way twice
        _, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
        assert counts.max() == 2

    def test_a_hole_the_cut_did_not_make_is_left_alone(self, grid):
        border_before = boundary_edges(grid.mesh_faces)[0].shape[0]  # the grid's own outer border

        cut = _cut(grid, _block(1, 1, 3))
        close_cut(grid, cut)

        assert boundary_edges(grid.mesh_faces)[0].shape[0] == border_before

    def test_a_cycle_that_is_mostly_an_older_hole_is_skipped(self, grid):
        # A corner block: the hole it leaves in the rest is the grid's whole outer border with a
        # notch, of which the cut is a small part; the part's own rim is half cut, half border.
        cut = _cut(grid, _block(0, 0, 3))

        result = close_cut(grid.clone(), cut, min_cut_share=0.6)
        assert result.cycles_closed == 0 and result.cycles_skipped == 2

        result = close_cut(grid, cut, min_cut_share=0.5)
        assert result.cycles_closed == 1 and result.cycles_skipped == 1

    def test_nothing_that_existed_changes(self, grid):
        cut = _cut(grid, _block(1, 1, 3))
        vertices, faces = grid.mesh_vertices.copy(), grid.mesh_faces.copy()
        positions, colors = grid.recompute_gaussian_positions().copy(), grid.gaussian_colors.copy()
        quaternions, scales = grid.gaussian_quaternions.copy(), grid.gaussian_scales.copy()

        close_cut(grid, cut)

        assert np.array_equal(grid.mesh_vertices, vertices)
        assert np.array_equal(grid.mesh_faces[: faces.shape[0]], faces)
        count = positions.shape[0]
        assert np.array_equal(grid.recompute_gaussian_positions()[:count], positions)
        assert np.array_equal(grid.gaussian_colors[:count], colors)
        assert np.array_equal(grid.gaussian_quaternions[:count], quaternions)
        assert np.array_equal(grid.gaussian_scales[:count], scales)

    def test_each_new_gaussian_lies_flat_on_its_triangle_with_the_look_of_the_rim(self, grid):
        cut = _cut(grid, _block(1, 1, 3))
        rim_colors = {tuple(c) for c in grid.gaussian_colors.round(6)}
        faces_before, count = grid.mesh_faces.shape[0], len(grid)

        close_cut(grid, cut)

        new = slice(count, len(grid))
        assert np.array_equal(grid.gaussian_face_idx[new], np.arange(faces_before, grid.mesh_faces.shape[0]))
        assert np.allclose(grid.gaussian_bary[new], 1.0 / 3.0)
        assert all(tuple(c) in rim_colors for c in grid.gaussian_colors[new].round(6))
        # A triangle between three rim vertices on one line has no plane; its Gaussian gets no size.
        corners = grid.mesh_vertices[grid.mesh_faces[faces_before:]].astype(np.float64)
        flat = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1) > 1e-9
        rotation = _quaternions_to_matrices(grid.gaussian_quaternions[new])
        thin = rotation[np.arange(rotation.shape[0]), :, np.argmin(grid.gaussian_scales[new], axis=1)]
        assert flat.sum() >= 10
        assert np.allclose(np.abs(thin[flat, 2]), 1.0, atol=1e-5)  # the grid lies in z = 0: the normal is z
        assert np.allclose(grid.gaussian_scales[new][~flat, :2], 0.0)  # its two in-plane sizes

    def test_sam_pick_still_maps_the_twin_and_a_rim_answer_takes_the_patch_along(self, grid):
        reference = ReferenceGeometry.capture(grid)  # as loaded
        cut = _cut(grid, _block(1, 1, 3))
        faces_before = grid.mesh_faces.shape[0]

        close_cut(grid, cut)

        assert reference.maps(grid)
        patch = np.arange(faces_before, grid.mesh_faces.shape[0])
        answer = np.isin(reference.gaussian_face_idx, grid.face_origin[patch])  # the rim triangles, as loaded
        selected = reference.selection_in(grid, answer)
        assert set(grid.mesh_faces[patch].reshape(-1).tolist()) <= set(selected.tolist())

    def test_closing_twice_adds_nothing_the_second_time(self, grid):
        cut = _cut(grid, _block(1, 1, 3))
        close_cut(grid, cut)

        again = close_cut(grid, cut)

        assert again.faces_added == 0 and again.cut_edges_open_before == 0


class TestTriangulation:
    def test_a_convex_polygon_takes_n_minus_two_triangles_covering_its_area(self):
        angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        points = np.stack([np.cos(angles), np.sin(angles), np.zeros(9)], axis=1)

        triangles = min_area_triangulation(points)

        assert triangles.shape == (7, 3)
        area = 0.5 * np.linalg.norm(np.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                                              points[triangles[:, 2]] - points[triangles[:, 0]]), axis=1).sum()
        assert area == pytest.approx(0.5 * 9 * np.sin(2 * np.pi / 9))

    def test_a_long_narrow_cycle_is_closed_by_a_strip_not_a_membrane(self):
        # A slit: two rows of points 0.01 apart along the same arc, 1.0 long, closed into one cycle.
        x = np.linspace(0.0, 1.0, 20)
        arc = 0.3 * np.sin(np.pi * x)
        points = np.concatenate([np.stack([x, np.zeros(20), arc], 1),
                                 np.stack([x[::-1], np.full(20, 0.01), arc[::-1]], 1)])

        triangles = min_area_triangulation(points)

        area = 0.5 * np.linalg.norm(np.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                                              points[triangles[:, 2]] - points[triangles[:, 0]]), axis=1).sum()
        centre = points.mean(axis=0)
        fan = 0.5 * np.linalg.norm(np.cross(points - centre, np.roll(points, -1, axis=0) - centre), axis=1).sum()
        assert area < 0.02 and area < 0.1 * fan

    def test_a_polygon_whose_every_chord_is_forbidden_cannot_be_closed(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        forbidden = np.ones((4, 4), dtype=bool)
        np.fill_diagonal(forbidden, False)

        assert min_area_triangulation(points, forbidden) is None

    def test_a_cycle_through_a_vertex_twice_is_split_into_simple_cycles(self):
        ring = np.array([10, 11, 12, 13, 11, 14, 15])  # passes vertex 11 twice

        cycles = simple_cycles(ring)

        assert sorted(np.concatenate(cycles).tolist()) == list(range(7))  # every edge once
        assert all(len(set(ring[c].tolist())) == len(c) for c in cycles)
