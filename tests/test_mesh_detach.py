import numpy as np
import pytest

from src.viewer.digital_twin import load_digital_twin_package
from src.viewer.mesh_detach import detach_selection, inside_faces, stretches_on_move
from src.viewer.mesh_editor import _build_vertex_adjacency, _connected_component
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


def _corner_positions(package):
    return package.mesh_vertices[package.mesh_faces]  # (F, 3, 3)


class TestDetach:
    def test_a_three_by_three_block_takes_its_eight_triangles_and_copies_its_eight_border_vertices(self, grid):
        result = detach_selection(grid, _block(1, 1, 3))

        assert result.faces_detached == 8
        assert result.vertices_duplicated == 8
        assert grid.mesh_vertices.shape[0] == N * N + 8
        assert len(result.selection) == 9  # the eight copies and the centre vertex, which was on no cut
        assert (2 * N + 2) in result.selection.tolist()

    def test_the_detached_part_shares_no_vertex_with_the_rest(self, grid):
        result = detach_selection(grid, _block(1, 1, 3))

        inside = inside_faces(grid.mesh_faces, result.selection, grid.mesh_vertices.shape[0])
        assert set(grid.mesh_faces[inside].reshape(-1)).isdisjoint(set(grid.mesh_faces[~inside].reshape(-1)))
        adj_ptr, adj_indices = _build_vertex_adjacency(grid.mesh_faces, grid.mesh_vertices.shape[0])
        component = _connected_component(int(result.selection[0]), adj_ptr, adj_indices, grid.mesh_vertices.shape[0])
        assert set(component.tolist()) == set(result.selection.tolist())  # Pick Object now picks just the part

    def test_nothing_else_changes_not_the_triangles_not_the_gaussians_not_the_old_vertices(self, grid):
        corners, positions = _corner_positions(grid).copy(), grid.recompute_gaussian_positions().copy()
        old_vertices, n_faces = grid.mesh_vertices.copy(), grid.mesh_faces.shape[0]
        gaussian_face_idx, colors = grid.gaussian_face_idx.copy(), grid.gaussian_colors.copy()

        detach_selection(grid, _block(1, 1, 3))

        assert grid.mesh_faces.shape[0] == n_faces
        assert np.array_equal(_corner_positions(grid), corners)  # every triangle is where it was
        assert np.array_equal(grid.recompute_gaussian_positions(), positions)
        assert np.array_equal(grid.gaussian_face_idx, gaussian_face_idx) and np.array_equal(grid.gaussian_colors, colors)
        assert np.array_equal(grid.mesh_vertices[: N * N], old_vertices)  # old numbers keep their meaning

    def test_moving_the_detached_part_stretches_nothing_else_and_without_it_the_border_stretches(self, grid):
        joined = grid.clone()
        block = _block(1, 1, 3)
        result = detach_selection(grid, block)
        inside = inside_faces(grid.mesh_faces, result.selection, grid.mesh_vertices.shape[0])
        before = _corner_positions(grid).copy()

        grid.mesh_vertices[result.selection] += np.float32([0.0, 0.0, 1.0])
        joined.mesh_vertices[block] += np.float32([0.0, 0.0, 1.0])

        assert np.array_equal(_corner_positions(grid)[~inside], before[~inside])  # the rest did not move
        stretched = ~np.all(_corner_positions(joined)[~inside] == before[~inside], axis=(1, 2))
        # Without the cut the ring of twelve cells around the block is pulled along: 24 triangles, less
        # the two corner ones whose diagonal keeps them clear of the selected vertices.
        assert stretched.sum() == 22

    def test_selected_vertices_on_no_whole_triangle_are_left_to_the_rest(self, grid):
        block = _block(1, 1, 3)
        stray = np.array([0])  # a corner vertex: its triangle has two corners outside the selection

        result = detach_selection(grid, np.concatenate([block, stray]))

        assert 0 not in result.selection.tolist()
        assert not set(result.selection.tolist()) & set(block.tolist()) - {2 * N + 2}  # the originals on the cut stay behind

    def test_a_selection_without_a_whole_triangle_changes_nothing(self, grid):
        faces, vertices = grid.mesh_faces.copy(), grid.mesh_vertices.copy()

        assert detach_selection(grid, np.array([0, 2 * N + 3, 4 * N + 5])) is None

        assert np.array_equal(grid.mesh_faces, faces) and np.array_equal(grid.mesh_vertices, vertices)

    def test_a_second_detach_copies_nothing(self, grid):
        result = detach_selection(grid, _block(1, 1, 3))
        n_vertices = grid.mesh_vertices.shape[0]

        again = detach_selection(grid, result.selection)

        assert again.vertices_duplicated == 0 and grid.mesh_vertices.shape[0] == n_vertices
        assert np.array_equal(again.selection, result.selection)

    def test_the_whole_mesh_is_a_piece_of_its_own_already(self, grid):
        result = detach_selection(grid, np.arange(N * N))

        assert result.vertices_duplicated == 0 and result.faces_detached == 2 * (N - 1) ** 2


class TestDetachAndSamPick:
    def test_the_copies_keep_the_origin_of_their_original_so_sam_pick_still_maps(self, grid):
        reference = ReferenceGeometry.capture(grid)
        loaded = grid.mesh_vertices.copy()

        result = detach_selection(grid, _block(1, 1, 3))

        assert reference.maps(grid)
        copies = np.arange(N * N, grid.mesh_vertices.shape[0])
        assert np.array_equal(loaded[grid.vertex_origin[copies]], grid.mesh_vertices[copies])
        for vertex in result.selection:
            assert np.allclose(reference.point_of(grid, vertex), grid.mesh_vertices[vertex])

    def test_an_answer_selects_the_detached_part_through_its_copies(self, grid):
        reference = ReferenceGeometry.capture(grid)
        result = detach_selection(grid, _block(1, 1, 3))
        answer = np.zeros(len(grid), dtype=bool)
        answer[np.nonzero(inside_faces(grid.mesh_faces, result.selection, grid.mesh_vertices.shape[0]))[0]] = True

        assert np.array_equal(reference.selection_in(grid, answer), result.selection)

    def test_without_origin_tracking_it_works_and_tracks_nothing(self, tmp_path, make_grid_twin):
        make_grid_twin(tmp_path / "g.npz", n=N)
        package = load_digital_twin_package(str(tmp_path / "g.npz"))

        result = detach_selection(package, _block(1, 1, 3))

        assert result.vertices_duplicated == 8 and package.vertex_origin is None


class TestStretchesOnMove:
    def test_a_block_joined_to_the_rest_would_stretch(self, grid):
        assert stretches_on_move(grid.mesh_faces, _block(1, 1, 3), grid.mesh_vertices.shape[0])

    def test_the_whole_mesh_and_a_detached_part_would_not(self, grid):
        n_vertices = grid.mesh_vertices.shape[0]
        assert not stretches_on_move(grid.mesh_faces, np.arange(n_vertices), n_vertices)

        result = detach_selection(grid, _block(1, 1, 3))

        assert not stretches_on_move(grid.mesh_faces, result.selection, grid.mesh_vertices.shape[0])

    def test_loose_vertices_would_stretch_their_triangles(self, grid):
        assert stretches_on_move(grid.mesh_faces, np.array([0, 14]), grid.mesh_vertices.shape[0])
