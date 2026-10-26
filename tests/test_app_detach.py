import numpy as np
from PyQt5 import QtCore

from src.viewer.app import MainWindow
from src.viewer.mesh_detach import inside_faces

N = 25
BLOCK = np.array([i * N + j for i in range(5, 10) for j in range(5, 10)])  # a 5 x 5 block of vertices


def _open(qapp, tmp_path, make_grid_twin) -> MainWindow:
    twin = tmp_path / "grid_digital_twin.npz"
    make_grid_twin(twin)
    window = MainWindow(initial_path=None)
    window.show()
    window._load_digital_twin(str(twin))
    qapp.processEvents()
    return window


def _detached(qapp, tmp_path, make_grid_twin):
    window = _open(qapp, tmp_path, make_grid_twin)
    window.mesh_panel.set_selection(BLOCK)
    window._on_detach_selected()
    return window


def test_the_switch_needs_a_twin_and_starts_off(qapp, tmp_path, make_grid_twin):
    window = MainWindow(initial_path=None)
    assert not window.detach_button.isEnabled()
    window.close()

    window = _open(qapp, tmp_path, make_grid_twin)
    assert window.detach_button.isCheckable() and window.detach_button.isEnabled()
    assert not window.detach_button.isChecked()
    window.mesh_panel.set_selection(BLOCK)
    window.mesh_panel.set_selection(None)
    assert window.detach_button.isEnabled()  # it can always be switched off again
    window.close()


def test_switching_it_on_cuts_the_selection_and_off_leaves_the_cut(qapp, tmp_path, make_grid_twin):
    window = _open(qapp, tmp_path, make_grid_twin)
    n_vertices = window.digital_twin_package.mesh_vertices.shape[0]
    window.mesh_panel.set_selection(BLOCK)

    window.detach_button.setChecked(True)

    assert window.digital_twin_package.mesh_vertices.shape[0] == n_vertices + 16
    assert "Detached 32 triangles" in window.status_label.text()

    window.detach_button.setChecked(False)

    assert window.digital_twin_package.mesh_vertices.shape[0] == n_vertices + 16
    assert "off" in window.status_label.text()
    window.close()


def test_switching_it_on_without_a_selection_cuts_nothing(qapp, tmp_path, make_grid_twin):
    window = _open(qapp, tmp_path, make_grid_twin)
    n_vertices = window.digital_twin_package.mesh_vertices.shape[0]

    window.detach_button.setChecked(True)

    assert window.digital_twin_package.mesh_vertices.shape[0] == n_vertices
    assert not window.undo_button.isEnabled()
    window.close()


def test_loading_a_twin_switches_it_off(qapp, tmp_path, make_grid_twin):
    window = _open(qapp, tmp_path, make_grid_twin)
    window.detach_button.setChecked(True)

    window._load_digital_twin(str(tmp_path / "grid_digital_twin.npz"))

    assert not window.detach_button.isChecked() and window.detach_button.isEnabled()
    window.close()


def test_detaching_cuts_the_mesh_and_selects_the_part_leaving_the_gaussians_alone(qapp, tmp_path, make_grid_twin):
    window = _open(qapp, tmp_path, make_grid_twin)
    package = window.digital_twin_package
    n_vertices, n_gaussians = package.mesh_vertices.shape[0], len(package)
    positions = package.recompute_gaussian_positions().copy()
    window.mesh_panel.set_selection(BLOCK)

    window._on_detach_selected()

    package = window.digital_twin_package
    assert package.mesh_vertices.shape[0] == n_vertices + 16  # the 16 border vertices of the block
    assert len(package) == n_gaussians
    assert np.array_equal(package.recompute_gaussian_positions(), positions)
    selection = window.mesh_panel.selected_vertices
    assert len(selection) == 25 and set(selection.tolist()).isdisjoint(set(BLOCK.tolist()) - {7 * N + j for j in range(6, 9)} - {i * N + j for i in (6, 8) for j in range(6, 9)})
    assert "Detached 32 triangles" in window.status_label.text()
    window.close()


def test_the_panel_and_the_package_still_share_their_vertices(qapp, tmp_path, make_grid_twin):
    window = _detached(qapp, tmp_path, make_grid_twin)

    assert window.digital_twin_package.mesh_vertices is window.mesh_panel.vertices  # a drag moves the Gaussians
    window.close()


def test_moving_the_detached_part_moves_only_its_own_gaussians(qapp, tmp_path, make_grid_twin):
    window = _detached(qapp, tmp_path, make_grid_twin)
    package = window.digital_twin_package
    inside = inside_faces(package.mesh_faces, window.mesh_panel.selected_vertices, package.mesh_vertices.shape[0])
    on_part = inside[package.gaussian_face_idx]
    before = package.recompute_gaussian_positions().copy()

    window.mesh_panel.vertices[window.mesh_panel.selected_vertices] += np.float32([0.0, 0.0, 2.0])  # what a drag does

    after = package.recompute_gaussian_positions()
    assert np.array_equal(after[~on_part], before[~on_part])
    assert np.allclose(after[on_part] - before[on_part], [0.0, 0.0, 2.0])
    window.close()


def test_the_view_is_left_where_it_was(qapp, tmp_path, make_grid_twin):
    window = _open(qapp, tmp_path, make_grid_twin)
    camera = window.mesh_panel.view.camera
    camera.azimuth, camera.elevation, camera.scale_factor = 77.0, 12.0, 3.5
    window.mesh_panel.set_selection(BLOCK)

    window._on_detach_selected()

    assert (camera.azimuth, camera.elevation, camera.scale_factor) == (77.0, 12.0, 3.5)
    window.close()


def test_pick_object_then_picks_the_detached_part_alone(qapp, tmp_path, make_grid_twin):
    window = _open(qapp, tmp_path, make_grid_twin)
    panel = window.mesh_panel
    window.pick_button.setChecked(True)
    # A vertex inside the part: on the cut itself a copy and its original coincide until the part
    # is moved, and the nearest-vertex search finds the original, which belongs to the rest.
    centre = 7 * N + 7

    def pick_centre():
        target = panel._project_to_canvas(panel.vertices[[centre]])[0]
        panel._pick_press(type("Event", (), {"pos": target, "button": 1, "modifiers": ()})())

    pick_centre()  # before the cut: the whole grid is one piece, and its connectivity is now cached
    assert len(panel.selected_vertices) == N * N
    panel.set_selection(BLOCK)
    window._on_detach_selected()
    part = panel.selected_vertices.copy()
    panel.set_selection(None)

    pick_centre()

    assert np.array_equal(np.sort(panel.selected_vertices), np.sort(part))
    window.close()


def test_undo_puts_the_mesh_back(qapp, tmp_path, make_grid_twin):
    window = _open(qapp, tmp_path, make_grid_twin)
    faces = window.digital_twin_package.mesh_faces.copy()
    n_vertices = window.digital_twin_package.mesh_vertices.shape[0]
    window.mesh_panel.set_selection(BLOCK)
    window._on_detach_selected()
    assert window.undo_button.isEnabled() and window.undo_button.text() == "Undo Detach"

    window._on_undo_delete()

    assert window.digital_twin_package.mesh_vertices.shape[0] == n_vertices
    assert np.array_equal(window.digital_twin_package.mesh_faces, faces)
    assert not window.undo_button.isEnabled()
    window.close()


def test_a_selection_without_a_whole_triangle_says_so_and_changes_nothing(qapp, tmp_path, make_grid_twin):
    window = _open(qapp, tmp_path, make_grid_twin)
    n_vertices = window.digital_twin_package.mesh_vertices.shape[0]
    window.mesh_panel.set_selection(np.array([0, 100, 300]))

    window._on_detach_selected()

    assert window.digital_twin_package.mesh_vertices.shape[0] == n_vertices
    assert "no triangle has all three corners selected" in window.status_label.text()
    assert not window.undo_button.isEnabled()
    window.close()


def test_a_part_that_is_separate_already_is_left_alone_and_needs_no_undo(qapp, tmp_path, make_grid_twin):
    window = _detached(qapp, tmp_path, make_grid_twin)
    n_vertices = window.digital_twin_package.mesh_vertices.shape[0]
    snapshot = window._undo_snapshot

    window._on_detach_selected()  # the same selection again

    assert window.digital_twin_package.mesh_vertices.shape[0] == n_vertices
    assert window._undo_snapshot is snapshot
    assert "piece of its own already" in window.status_label.text()
    window.close()


def test_delete_afterwards_labels_the_undo_as_a_delete(qapp, tmp_path, make_grid_twin):
    window = _detached(qapp, tmp_path, make_grid_twin)

    window._on_delete_selected()

    assert window.undo_button.text() == "Undo Delete"
    window.close()
