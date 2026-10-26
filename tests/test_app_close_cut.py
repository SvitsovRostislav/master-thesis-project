import numpy as np

from src.viewer.app import MainWindow

N = 25
BLOCK = np.array([i * N + j for i in range(5, 10) for j in range(5, 10)])


def _open(qapp, tmp_path, make_grid_twin) -> MainWindow:
    twin = tmp_path / "grid_digital_twin.npz"
    make_grid_twin(twin)
    window = MainWindow(initial_path=None)
    window.show()
    window._load_digital_twin(str(twin))
    qapp.processEvents()
    return window


def _detached(qapp, tmp_path, make_grid_twin) -> MainWindow:
    window = _open(qapp, tmp_path, make_grid_twin)
    window.mesh_panel.set_selection(BLOCK)
    window._on_detach_selected()
    return window


def test_the_button_is_enabled_by_a_cut_and_used_up_by_closing_it(qapp, tmp_path, make_grid_twin):
    window = _open(qapp, tmp_path, make_grid_twin)
    assert not window.close_cut_button.isEnabled()
    window.mesh_panel.set_selection(BLOCK)
    assert not window.close_cut_button.isEnabled()  # a selection alone has no cut to close

    window._on_detach_selected()
    assert window.close_cut_button.isEnabled()

    window._on_close_cut()
    assert not window.close_cut_button.isEnabled()
    window.close()


def test_closing_adds_the_patches_and_their_gaussians_to_the_panels(qapp, tmp_path, make_grid_twin):
    window = _detached(qapp, tmp_path, make_grid_twin)
    package = window.digital_twin_package
    faces, gaussians = package.mesh_faces.shape[0], len(package)

    window._on_close_cut()

    package = window.digital_twin_package
    assert package.mesh_faces.shape[0] == faces + 28  # a 16-gon takes 14 triangles, on each side
    assert len(package) == gaussians + 28
    assert window.mesh_panel.faces.shape[0] == package.mesh_faces.shape[0]
    assert window.viewer._sizes.shape[0] == len(package)  # the point-sprite view shows the new Gaussians
    assert package.mesh_vertices is window.mesh_panel.vertices  # a drag still moves the Gaussians
    assert "Closed 2 holes with 28 triangles" in window.status_label.text()
    window.close()


def test_the_view_is_left_where_it_was(qapp, tmp_path, make_grid_twin):
    window = _detached(qapp, tmp_path, make_grid_twin)
    mesh_camera, splat_camera = window.mesh_panel.view.camera, window.viewer.view.camera
    mesh_camera.azimuth, mesh_camera.elevation, mesh_camera.scale_factor = 77.0, 12.0, 3.5
    splat_camera.azimuth, splat_camera.scale_factor = 40.0, 2.5

    window._on_close_cut()

    assert (mesh_camera.azimuth, mesh_camera.elevation, mesh_camera.scale_factor) == (77.0, 12.0, 3.5)
    assert (splat_camera.azimuth, splat_camera.scale_factor) == (40.0, 2.5)
    window.close()


def test_the_selection_and_the_baselines_follow_the_new_triangles(qapp, tmp_path, make_grid_twin):
    window = _detached(qapp, tmp_path, make_grid_twin)
    window.adapt_button.setChecked(True)
    selection = window.mesh_panel.selected_vertices.copy()

    window._on_close_cut()

    package = window.digital_twin_package
    assert np.array_equal(window.mesh_panel.selected_vertices, selection)  # no vertex was added
    assert window._deformation_baseline.frame_inv.shape[0] == package.mesh_faces.shape[0]
    window.close()


def test_undo_takes_the_patches_away(qapp, tmp_path, make_grid_twin):
    window = _detached(qapp, tmp_path, make_grid_twin)
    faces, gaussians = window.digital_twin_package.mesh_faces.copy(), len(window.digital_twin_package)
    window._on_close_cut()
    assert window.undo_button.isEnabled() and window.undo_button.text() == "Undo Close Cut"

    window._on_undo_delete()

    assert np.array_equal(window.digital_twin_package.mesh_faces, faces)
    assert len(window.digital_twin_package) == gaussians
    window.close()


def test_a_delete_after_the_cut_withdraws_the_button(qapp, tmp_path, make_grid_twin):
    window = _detached(qapp, tmp_path, make_grid_twin)

    window._on_delete_selected()

    assert not window.close_cut_button.isEnabled()
    window.close()


def test_a_cut_with_nothing_left_open_says_so_and_needs_no_undo(qapp, tmp_path, make_grid_twin):
    window = _detached(qapp, tmp_path, make_grid_twin)
    window._on_close_cut()
    snapshot = window._undo_snapshot
    window._last_cut = np.concatenate([BLOCK, np.arange(N * N, N * N + 16)])  # the same cut, now closed

    window._on_close_cut()

    assert "no hole" in window.status_label.text()
    assert window._undo_snapshot is snapshot
    window.close()
