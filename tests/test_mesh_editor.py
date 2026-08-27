import numpy as np
import pytest

from src.viewer.mesh_editor import (
    BASE_MESH_COLOR,
    SELECTED_VERTEX_COLOR,
    MeshEditorPanel,
    _build_vertex_adjacency,
    _connected_component,
    _points_in_polygon,
)


class _FakeEvent:
    def __init__(self, pos, button=None, buttons=()):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.button = button
        self.buttons = buttons


class TestPointsInPolygon:
    def test_rectangle_inside_outside_and_boundary(self):
        rect = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        points = np.array([[5, 5], [-1, 5], [5, -1], [9.9, 9.9], [10.1, 5]])
        expected = [True, False, False, True, False]
        np.testing.assert_array_equal(_points_in_polygon(points, rect), expected)

    def test_triangle(self):
        triangle = np.array([[0, 0], [10, 0], [5, 10]], dtype=float)
        points = np.array([[5, 1], [1, 1], [5, 9], [9, 1], [0.1, 9]])
        expected = [True, True, True, True, False]
        np.testing.assert_array_equal(_points_in_polygon(points, triangle), expected)

    def test_empty_points_returns_empty_mask(self):
        rect = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        result = _points_in_polygon(np.zeros((0, 2)), rect)
        assert result.shape == (0,)

    def test_all_outside(self):
        rect = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        points = np.array([[100, 100], [-100, -100], [50, -5]])
        assert not _points_in_polygon(points, rect).any()


class TestConnectedComponent:
    def test_two_disconnected_triangles_stay_separate(self):
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        adj_ptr, adj_indices = _build_vertex_adjacency(faces, 6)

        table = _connected_component(0, adj_ptr, adj_indices, 6)
        cup = _connected_component(4, adj_ptr, adj_indices, 6)

        assert sorted(table.tolist()) == [0, 1, 2]
        assert sorted(cup.tolist()) == [3, 4, 5]

    def test_connected_quad_returns_all_vertices(self):
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        adj_ptr, adj_indices = _build_vertex_adjacency(faces, 4)

        for start in range(4):
            component = _connected_component(start, adj_ptr, adj_indices, 4)
            assert sorted(component.tolist()) == [0, 1, 2, 3]

    def test_vertex_with_no_faces_returns_itself_only(self):
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        adj_ptr, adj_indices = _build_vertex_adjacency(faces, 6)

        component = _connected_component(5, adj_ptr, adj_indices, 6)

        assert component.tolist() == [5]

    def test_three_separate_components(self):
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32)
        adj_ptr, adj_indices = _build_vertex_adjacency(faces, 9)

        components = {tuple(sorted(_connected_component(s, adj_ptr, adj_indices, 9).tolist())) for s in (0, 3, 6)}

        assert components == {(0, 1, 2), (3, 4, 5), (6, 7, 8)}


def _make_pyramid():
    vertices = np.array(
        [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 2]], dtype=np.float32
    )
    faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.int32)
    return vertices, faces


def _make_two_disconnected_triangles():
    vertices = np.array(
        [
            [-5, -5, 0], [-4, -5, 0], [-4.5, -4, 0],  # "table", verts 0-2
            [5, 5, 0], [6, 5, 0], [5.5, 6, 0],  # "cup", verts 3-5
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    return vertices, faces


class TestMeshEditorPanelLifecycle:
    def test_load_mesh_initializes_state(self, qapp):
        panel = MeshEditorPanel()
        vertices, faces = _make_pyramid()

        panel.load_mesh(vertices, faces)

        np.testing.assert_allclose(panel.vertices, vertices)
        np.testing.assert_array_equal(panel.faces, faces)
        assert panel.selected_vertices is None
        assert panel.brush_radius > 0
        assert panel._vertex_colors.shape == (5, 4)
        np.testing.assert_allclose(panel._vertex_colors, np.tile(BASE_MESH_COLOR, (5, 1)))

    def test_set_mode_controls_camera_interactivity(self, qapp):
        panel = MeshEditorPanel()
        panel.load_mesh(*_make_pyramid())

        panel.set_mode("sculpt")
        assert panel.mode == "sculpt"
        assert panel.view.camera.interactive is False

        panel.set_mode("select")
        assert panel.mode == "select"
        assert panel.view.camera.interactive is False

        panel.set_mode("pick")
        assert panel.mode == "pick"
        assert panel.view.camera.interactive is False

        panel.set_mode("orbit")
        assert panel.mode == "orbit"
        assert panel.view.camera.interactive is True

    def test_set_mode_rejects_unknown_mode(self, qapp):
        panel = MeshEditorPanel()
        panel.load_mesh(*_make_pyramid())
        with pytest.raises(AssertionError):
            panel.set_mode("bogus")

    def test_clear_selection_resets_highlight_and_fires_callback(self, qapp):
        received = []
        panel = MeshEditorPanel(on_selection_changed=lambda sel: received.append(sel))
        panel.load_mesh(*_make_pyramid())

        panel.selected_vertices = np.array([0, 1])
        panel._update_highlight()
        assert np.allclose(panel._vertex_colors[0], SELECTED_VERTEX_COLOR)

        panel.clear_selection()

        assert panel.selected_vertices is None
        np.testing.assert_allclose(panel._vertex_colors, np.tile(BASE_MESH_COLOR, (5, 1)))
        assert received == [None]

    def test_select_release_highlights_polygon_interior(self, qapp):
        """Directly exercises the same release-time logic a real
        rectangle-drag would hit, without needing a live mouse event."""
        panel = MeshEditorPanel()
        panel.load_mesh(*_make_pyramid())
        panel.set_mode("select")

        screen_verts = panel._project_to_canvas(panel.vertices)
        apex_screen = screen_verts[4]
        panel._select_drag_kind = "rect"
        panel._select_path = [
            (apex_screen[0] - 50, apex_screen[1] - 50),
            (apex_screen[0] + 50, apex_screen[1] + 50),
        ]

        panel._select_release()

        assert panel.selected_vertices is not None
        assert 4 in panel.selected_vertices
        assert panel._select_drag_kind is None

    def test_load_mesh_again_clears_previous_selection(self, qapp):
        panel = MeshEditorPanel()
        vertices, faces = _make_pyramid()
        panel.load_mesh(vertices, faces)
        panel.selected_vertices = np.array([0])

        panel.load_mesh(vertices, faces)

        assert panel.selected_vertices is None


class TestPickMode:
    def test_pick_press_selects_only_the_clicked_component(self, qapp):
        panel = MeshEditorPanel()
        panel.load_mesh(*_make_two_disconnected_triangles())
        panel.set_mode("pick")

        screen_verts = panel._project_to_canvas(panel.vertices)
        click_pos = screen_verts[4]

        panel._pick_press(_FakeEvent(click_pos))

        assert panel.selected_vertices is not None
        assert sorted(panel.selected_vertices.tolist()) == [3, 4, 5]

    def test_pick_press_fires_selection_changed_callback(self, qapp):
        received = []
        panel = MeshEditorPanel(on_selection_changed=lambda sel: received.append(sel))
        panel.load_mesh(*_make_two_disconnected_triangles())
        panel.set_mode("pick")

        screen_verts = panel._project_to_canvas(panel.vertices)
        panel._pick_press(_FakeEvent(screen_verts[0]))

        assert len(received) == 1
        assert sorted(received[0].tolist()) == [0, 1, 2]

    def test_pick_press_on_empty_space_leaves_selection_unchanged(self, qapp):
        panel = MeshEditorPanel()
        panel.load_mesh(*_make_two_disconnected_triangles())
        panel.set_mode("pick")

        screen_verts = panel._project_to_canvas(panel.vertices)
        panel._pick_press(_FakeEvent(screen_verts[0]))
        assert panel.selected_vertices is not None

        panel._pick_press(_FakeEvent(screen_verts[0] + 10_000))

        assert sorted(panel.selected_vertices.tolist()) == [0, 1, 2]

    def test_clicking_existing_selection_starts_a_move_not_a_new_pick(self, qapp):
        panel = MeshEditorPanel()
        panel.load_mesh(*_make_two_disconnected_triangles())
        panel.set_mode("pick")

        screen_verts = panel._project_to_canvas(panel.vertices)
        panel._pick_press(_FakeEvent(screen_verts[3]))
        assert sorted(panel.selected_vertices.tolist()) == [3, 4, 5]

        panel._pick_press(_FakeEvent(screen_verts[4]))

        assert panel._select_drag_kind == "move"
        assert sorted(panel.selected_vertices.tolist()) == [3, 4, 5]  # unchanged

    def test_pick_then_move_drags_the_whole_component_rigidly(self, qapp):
        received_edits = []
        panel = MeshEditorPanel(on_edited=lambda v: received_edits.append(v.copy()))
        panel.load_mesh(*_make_two_disconnected_triangles())
        panel.set_mode("pick")

        screen_verts = panel._project_to_canvas(panel.vertices)
        panel._pick_press(_FakeEvent(screen_verts[3]))
        original_cup = panel.vertices[[3, 4, 5]].copy()
        original_table = panel.vertices[[0, 1, 2]].copy()

        panel._pick_press(_FakeEvent(screen_verts[3], button=1, buttons=(1,)))
        panel._select_move(_FakeEvent(screen_verts[3] + [20.0, 15.0], buttons=(1,)))

        moved_cup = panel.vertices[[3, 4, 5]]
        deltas = moved_cup.astype(np.float64) - original_cup.astype(np.float64)
        assert np.linalg.norm(deltas.std(axis=0)) < 1e-3
        assert np.linalg.norm(deltas.mean(axis=0)) > 1e-4
        # The "table" must be untouched.
        np.testing.assert_allclose(panel.vertices[[0, 1, 2]], original_table)
        assert len(received_edits) == 1
