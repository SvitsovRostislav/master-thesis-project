import numpy as np

from src.viewer.app import MainWindow
from src.viewer.mesh_detach import stretches_on_move

N = 25
BLOCK = np.array([i * N + j for i in range(5, 10) for j in range(5, 10)])  # a 5 x 5 block of vertices
CENTRE = 7 * N + 7


def _open(qapp, tmp_path, make_grid_twin) -> MainWindow:
    twin = tmp_path / "grid_digital_twin.npz"
    make_grid_twin(twin)
    window = MainWindow(initial_path=None)
    window.show()
    window._load_digital_twin(str(twin))
    window.select_button.setChecked(True)
    qapp.processEvents()
    return window


def _event(pos, buttons=(1,)):
    return type("Event", (), {"pos": np.asarray(pos, dtype=np.float64), "button": 1, "buttons": buttons, "modifiers": ()})()


def _drag(window, vertex: int, offset=(30.0, 20.0)) -> str | None:
    panel = window.mesh_panel
    start = panel._project_to_canvas(panel.vertices[[vertex]])[0]
    press = panel._pick_press if panel.mode == "pick" else panel._select_press
    press(_event(start))
    started = panel._select_drag_kind
    panel._select_move(_event(start + np.asarray(offset)))
    panel._select_release()
    return started


def _edge_lengths(package):
    corners = package.mesh_vertices[package.mesh_faces].astype(np.float64)
    return np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2)


def test_with_the_switch_off_a_joined_selection_is_neither_moved_nor_cut(qapp, tmp_path, make_grid_twin):
    window = _open(qapp, tmp_path, make_grid_twin)
    package = window.digital_twin_package
    vertices, faces = package.mesh_vertices.copy(), package.mesh_faces.copy()
    gaussians = package.recompute_gaussian_positions().copy()
    window.mesh_panel.set_selection(BLOCK)

    started = _drag(window, CENTRE)

    assert started == "rect"  # the press starts a selection, not a move
    package = window.digital_twin_package
    assert np.array_equal(package.mesh_vertices, vertices) and np.array_equal(package.mesh_faces, faces)
    assert np.array_equal(package.recompute_gaussian_positions(), gaussians)
    assert not window.undo_button.isEnabled() and not window.close_cut_button.isEnabled()
    assert "turn Detach Selected on" in window.status_label.text()
    window.close()


def test_with_the_switch_off_a_piece_of_its_own_is_not_moved_either(qapp, tmp_path, make_grid_twin):
    window = _open(qapp, tmp_path, make_grid_twin)
    vertices = window.digital_twin_package.mesh_vertices.copy()
    window.mesh_panel.set_selection(np.arange(len(vertices)))  # the whole grid, as Pick Object selects it

    assert _drag(window, CENTRE) != "move"

    assert np.array_equal(window.digital_twin_package.mesh_vertices, vertices)
    window.close()


def test_with_the_switch_off_a_click_on_the_selection_keeps_it(qapp, tmp_path, make_grid_twin):
    window = _open(qapp, tmp_path, make_grid_twin)
    panel = window.mesh_panel
    panel.set_selection(BLOCK)

    panel._select_press(_event(panel._project_to_canvas(panel.vertices[[CENTRE]])[0]))
    panel._select_release()

    assert np.array_equal(np.sort(panel.selected_vertices), np.sort(BLOCK))
    window.close()


def test_with_the_switch_off_a_click_in_pick_mode_is_a_pick(qapp, tmp_path, make_grid_twin):
    window = _open(qapp, tmp_path, make_grid_twin)
    window.pick_button.setChecked(True)
    panel = window.mesh_panel
    vertices = window.digital_twin_package.mesh_vertices.copy()
    panel.set_selection(BLOCK)

    _drag(window, CENTRE)

    assert len(panel.selected_vertices) == N * N  # Pick Object took the whole (connected) grid
    assert np.array_equal(window.digital_twin_package.mesh_vertices, vertices)
    window.close()


def test_switched_on_the_selection_is_cut_and_moves_stretching_nothing(qapp, tmp_path, make_grid_twin):
    window = _open(qapp, tmp_path, make_grid_twin)
    window.mesh_panel.set_selection(BLOCK)
    window.detach_button.setChecked(True)
    package = window.digital_twin_package
    part = window.mesh_panel.selected_vertices.copy()
    lengths, before = _edge_lengths(package), package.mesh_vertices.copy()

    assert _drag(window, part[len(part) // 2]) == "move"

    moved = np.nonzero(np.any(package.mesh_vertices != before, axis=1))[0]
    assert np.array_equal(np.sort(moved), np.sort(part))
    assert np.allclose(_edge_lengths(package), lengths, atol=1e-5)
    window.close()


def test_switched_off_again_nothing_moves_and_on_again_it_moves(qapp, tmp_path, make_grid_twin):
    window = _open(qapp, tmp_path, make_grid_twin)
    package = window.digital_twin_package
    window.mesh_panel.set_selection(BLOCK)
    window.detach_button.setChecked(True)
    part = window.mesh_panel.selected_vertices.copy()

    window.detach_button.setChecked(False)
    before = package.mesh_vertices.copy()
    window.mesh_panel.set_selection(part)
    assert _drag(window, part[len(part) // 2]) != "move"
    assert np.array_equal(package.mesh_vertices, before)

    window.detach_button.setChecked(True)
    window.mesh_panel.set_selection(part)
    assert _drag(window, part[len(part) // 2]) == "move"
    assert not np.array_equal(package.mesh_vertices, before)
    window.close()


def test_switched_on_a_piece_of_its_own_moves_without_a_cut(qapp, tmp_path, make_grid_twin):
    window = _open(qapp, tmp_path, make_grid_twin)
    package = window.digital_twin_package
    before = package.mesh_vertices.copy()
    window.mesh_panel.set_selection(np.arange(len(before)))  # the whole grid, as Pick Object selects it
    window.detach_button.setChecked(True)

    assert _drag(window, CENTRE) == "move"

    assert package.mesh_vertices.shape == before.shape
    assert np.all(np.any(package.mesh_vertices != before, axis=1))
    assert not window.undo_button.isEnabled()  # nothing was cut
    window.close()


def test_switched_on_a_selection_made_later_is_cut_when_dragged(qapp, tmp_path, make_grid_twin):
    window = _open(qapp, tmp_path, make_grid_twin)
    window.detach_button.setChecked(True)  # no selection yet: nothing is cut
    package = window.digital_twin_package
    n_vertices = package.mesh_vertices.shape[0]
    lengths = _edge_lengths(package)
    faces = package.mesh_faces.copy()
    window.mesh_panel.set_selection(BLOCK)

    assert _drag(window, CENTRE) == "move"

    package = window.digital_twin_package
    assert package.mesh_vertices.shape[0] == n_vertices + 16
    assert np.allclose(_edge_lengths(package), lengths, atol=1e-5)
    assert not stretches_on_move(package.mesh_faces, window.mesh_panel.selected_vertices, package.mesh_vertices.shape[0])
    assert window.undo_button.text() == "Undo Detach"

    window._on_undo_delete()

    assert np.array_equal(window.digital_twin_package.mesh_faces, faces)
    assert window.digital_twin_package.mesh_vertices.shape[0] == n_vertices
    window.close()


def test_switched_on_selected_vertices_without_a_whole_triangle_stay_behind(qapp, tmp_path, make_grid_twin):
    window = _open(qapp, tmp_path, make_grid_twin)
    window.detach_button.setChecked(True)
    stray = 20 * N + 20
    before = window.digital_twin_package.mesh_vertices[stray].copy()
    window.mesh_panel.set_selection(np.append(BLOCK, stray))

    _drag(window, CENTRE)

    assert stray not in window.mesh_panel.selected_vertices
    assert np.array_equal(window.digital_twin_package.mesh_vertices[stray], before)
    assert "1 selected vertices without a whole triangle stay behind" in window.status_label.text()
    window.close()


def test_switched_on_loose_vertices_alone_are_not_moved(qapp, tmp_path, make_grid_twin):
    window = _open(qapp, tmp_path, make_grid_twin)
    window.detach_button.setChecked(True)
    vertices = window.digital_twin_package.mesh_vertices.copy()
    window.mesh_panel.set_selection(np.array([CENTRE]))

    assert _drag(window, CENTRE) != "move"

    assert np.array_equal(window.digital_twin_package.mesh_vertices, vertices)
    window.close()
