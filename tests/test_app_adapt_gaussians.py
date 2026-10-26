import numpy as np

from src.viewer.app import MainWindow


class _FakeEvent:
    def __init__(self, pos, buttons=()):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.buttons = buttons


def _open_grid_window(qapp, tmp_path, make_grid_twin) -> MainWindow:
    twin = tmp_path / "grid_digital_twin.npz"
    make_grid_twin(twin)
    window = MainWindow(initial_path=None)
    window.show()
    window._load_digital_twin(str(twin))
    qapp.processEvents()
    return window


def _sculpt_drag_far(window: MainWindow) -> None:
    panel = window.mesh_panel
    window.sculpt_button.setChecked(True)
    cx, cy = panel.canvas.size[0] / 2.0, panel.canvas.size[1] / 2.0
    screen_verts = panel._project_to_canvas(panel.vertices)
    start = screen_verts[int(np.argmin(np.linalg.norm(screen_verts - [cx, cy], axis=1)))]
    panel._sculpt_press(_FakeEvent(start))
    panel._sculpt_move(_FakeEvent(start + [150.0, 110.0], buttons=(1,)))
    panel._sculpt_release()


def test_adapt_button_is_disabled_until_a_twin_is_loaded_and_off_by_default(qapp, tmp_path, make_grid_twin):
    window = MainWindow(initial_path=None)
    assert not window.adapt_button.isEnabled()
    window.close()

    window = _open_grid_window(qapp, tmp_path, make_grid_twin)
    assert window.adapt_button.isEnabled()
    assert not window.adapt_button.isChecked()
    window.close()


def test_toggle_off_leaves_orientation_and_scale_exactly_as_before(qapp, tmp_path, make_grid_twin):
    window = _open_grid_window(qapp, tmp_path, make_grid_twin)
    package = window.digital_twin_package
    scales, quats = package.gaussian_scales.copy(), package.gaussian_quaternions.copy()

    _sculpt_drag_far(window)

    np.testing.assert_array_equal(package.gaussian_scales, scales)
    np.testing.assert_array_equal(package.gaussian_quaternions, quats)
    window.close()


def test_toggle_on_adapts_the_gaussians_of_deformed_triangles(qapp, tmp_path, make_grid_twin):
    window = _open_grid_window(qapp, tmp_path, make_grid_twin)
    package = window.digital_twin_package
    scales, quats = package.gaussian_scales.copy(), package.gaussian_quaternions.copy()
    n = len(package)
    window.adapt_button.setChecked(True)

    _sculpt_drag_far(window)

    assert len(package) == n  # adapting never adds or removes Gaussians
    assert not np.array_equal(package.gaussian_scales, scales)
    assert not np.array_equal(package.gaussian_quaternions, quats)
    assert np.all(np.isfinite(package.gaussian_scales))
    assert np.all(np.isfinite(package.gaussian_quaternions))
    # Far from the brush nothing moved, so those Gaussians must be untouched.
    untouched = np.all(package.gaussian_scales == scales, axis=1)
    assert untouched.sum() > n // 2
    assert "adapted" in window.status_label.text()
    window.close()


def test_edits_made_while_off_are_not_retroactively_adapted_on_toggle(qapp, tmp_path, make_grid_twin):
    window = _open_grid_window(qapp, tmp_path, make_grid_twin)
    package = window.digital_twin_package
    _sculpt_drag_far(window)  # toggle still off: position-only
    scales, quats = package.gaussian_scales.copy(), package.gaussian_quaternions.copy()

    window.adapt_button.setChecked(True)
    window._on_edit_finished()  # a release with no new geometry change

    np.testing.assert_array_equal(package.gaussian_scales, scales)
    np.testing.assert_array_equal(package.gaussian_quaternions, quats)
    window.close()


def test_adapt_and_fill_density_work_together(qapp, tmp_path, make_grid_twin):
    window = _open_grid_window(qapp, tmp_path, make_grid_twin)
    package = window.digital_twin_package
    original_scales = package.gaussian_scales.copy()
    original_quats = package.gaussian_quaternions.copy()
    n = len(package)
    window.adapt_button.setChecked(True)
    window.density_button.setChecked(True)

    _sculpt_drag_far(window)

    assert len(package) > n  # Fill Density topped up the stretched triangles
    assert not np.array_equal(package.gaussian_scales[:n], original_scales)
    assert not np.array_equal(package.gaussian_quaternions[:n], original_quats)
    assert np.all(np.isfinite(package.gaussian_scales))
    window.close()


def test_delete_recaptures_the_baseline_for_the_new_face_indexing(qapp, tmp_path, make_grid_twin):
    window = _open_grid_window(qapp, tmp_path, make_grid_twin)
    window.adapt_button.setChecked(True)
    faces_before = window.digital_twin_package.mesh_faces.shape[0]

    window.select_button.setChecked(True)
    panel = window.mesh_panel
    cx, cy = panel.canvas.size[0] / 2.0, panel.canvas.size[1] / 2.0
    panel._select_drag_kind = "rect"
    panel._select_path = [(cx - 60, cy - 60), (cx + 60, cy + 60)]
    panel._select_release()
    window._on_delete_selected()

    package = window.digital_twin_package
    assert package.mesh_faces.shape[0] < faces_before
    assert window._deformation_baseline is not None
    assert window._deformation_baseline.frame_inv.shape[0] == package.mesh_faces.shape[0]
    window.close()
