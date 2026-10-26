from src.viewer.app import MainWindow


def _open_grid_window(qapp, tmp_path, make_grid_twin, name: str = "grid") -> MainWindow:
    twin = tmp_path / f"{name}_digital_twin.npz"
    make_grid_twin(twin)
    window = MainWindow(initial_path=None)
    window.show()
    window._load_digital_twin(str(twin))
    qapp.processEvents()
    return window


def test_disabled_before_a_twin_is_loaded_and_off_by_default(qapp):
    window = MainWindow(initial_path=None)

    assert not window.sync_view_button.isEnabled()
    assert not window.sync_view_button.isChecked()
    window.close()


def test_enabled_once_a_twin_is_loaded(qapp, tmp_path, make_grid_twin):
    window = _open_grid_window(qapp, tmp_path, make_grid_twin)

    assert window.sync_view_button.isEnabled()
    assert not window.sync_view_button.isChecked()  # off until the user asks for it
    window.close()


def test_on_it_carries_orbiting_and_zoom_from_either_view_to_the_other(qapp, tmp_path, make_grid_twin):
    window = _open_grid_window(qapp, tmp_path, make_grid_twin)
    gaussian_camera, mesh_camera = window.viewer.view.camera, window.mesh_panel.view.camera
    window.sync_view_button.setChecked(True)

    gaussian_camera.azimuth = gaussian_camera.azimuth + 40
    gaussian_camera.elevation = gaussian_camera.elevation + 15
    gaussian_camera.scale_factor = gaussian_camera.scale_factor * 0.5

    assert mesh_camera.azimuth == gaussian_camera.azimuth
    assert mesh_camera.elevation == gaussian_camera.elevation
    assert mesh_camera.scale_factor == gaussian_camera.scale_factor

    mesh_camera.azimuth = mesh_camera.azimuth + 20  # the other direction works too
    assert gaussian_camera.azimuth == mesh_camera.azimuth
    window.close()


def test_off_by_default_orbiting_one_view_does_not_move_the_other(qapp, tmp_path, make_grid_twin):
    window = _open_grid_window(qapp, tmp_path, make_grid_twin)
    gaussian_camera, mesh_camera = window.viewer.view.camera, window.mesh_panel.view.camera
    before = mesh_camera.azimuth

    gaussian_camera.azimuth = gaussian_camera.azimuth + 40

    assert mesh_camera.azimuth == before
    window.close()


def test_turning_it_off_again_stops_the_sync(qapp, tmp_path, make_grid_twin):
    window = _open_grid_window(qapp, tmp_path, make_grid_twin)
    gaussian_camera, mesh_camera = window.viewer.view.camera, window.mesh_panel.view.camera
    window.sync_view_button.setChecked(True)
    window.sync_view_button.setChecked(False)
    before = mesh_camera.azimuth

    gaussian_camera.azimuth = gaussian_camera.azimuth + 40

    assert mesh_camera.azimuth == before
    window.close()


def test_panning_stays_independent_even_when_on(qapp, tmp_path, make_grid_twin):
    window = _open_grid_window(qapp, tmp_path, make_grid_twin)
    gaussian_camera, mesh_camera = window.viewer.view.camera, window.mesh_panel.view.camera
    window.sync_view_button.setChecked(True)
    before = tuple(mesh_camera.center)

    gaussian_camera.center = tuple(c + 1.0 for c in gaussian_camera.center)

    assert tuple(mesh_camera.center) == before
    window.close()


def test_loading_another_twin_turns_it_off_so_each_view_refits_on_its_own(qapp, tmp_path, make_grid_twin):
    window = _open_grid_window(qapp, tmp_path, make_grid_twin, name="first")
    window.sync_view_button.setChecked(True)

    second = tmp_path / "second_digital_twin.npz"
    make_grid_twin(second)
    window._load_digital_twin(str(second))
    qapp.processEvents()

    assert not window.sync_view_button.isChecked()
    gaussian_camera, mesh_camera = window.viewer.view.camera, window.mesh_panel.view.camera
    before = mesh_camera.azimuth
    gaussian_camera.azimuth = gaussian_camera.azimuth + 40
    assert mesh_camera.azimuth == before  # confirms the link itself, not just the button, is off
    window.close()
