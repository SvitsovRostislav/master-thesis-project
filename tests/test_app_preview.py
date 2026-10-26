from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui

from src.viewer.app import MainWindow


class _FakeRenderer(QtCore.QObject):
    finished = QtCore.pyqtSignal(bool, object)
    instances = []

    def __init__(self, parent=None):
        super().__init__(parent)
        self.started = None
        self.cancelled = False
        self.after_existed_at_start = False
        _FakeRenderer.instances.append(self)

    def start(self, scene_source, packages, out_dir, num_views=4, max_side=800):
        self.started = {"scene_source": scene_source, "packages": dict(packages), "out_dir": Path(out_dir)}
        self.after_existed_at_start = Path(packages["after"]).is_file()

    def cancel(self):
        self.cancelled = True


def _open(qapp, tmp_path, monkeypatch, make_grid_twin, with_source: bool) -> MainWindow:
    scene_source = tmp_path / "scene_source"
    scene_source.mkdir(exist_ok=True)
    monkeypatch.setattr(
        "src.viewer.app.find_scene_source_for_twin", lambda path: scene_source if with_source else None
    )
    twin = tmp_path / "grid_digital_twin.npz"
    make_grid_twin(twin)
    window = MainWindow(initial_path=None)
    window.show()
    window._load_digital_twin(str(twin))
    qapp.processEvents()
    return window


def _png(path: Path) -> str:
    image = QtGui.QImage(20, 20, QtGui.QImage.Format_RGB32)
    image.fill(QtGui.QColor("red"))
    image.save(str(path))
    return str(path)


def test_preview_button_is_disabled_before_a_twin_is_loaded(qapp):
    window = MainWindow(initial_path=None)
    assert not window.preview_button.isEnabled()
    window.close()


def test_preview_button_stays_disabled_when_the_scene_source_is_not_found(qapp, tmp_path, monkeypatch, make_grid_twin):
    window = _open(qapp, tmp_path, monkeypatch, make_grid_twin, with_source=False)

    assert not window.preview_button.isEnabled()
    window.close()


def test_preview_button_is_enabled_when_the_scene_source_is_found(qapp, tmp_path, monkeypatch, make_grid_twin):
    window = _open(qapp, tmp_path, monkeypatch, make_grid_twin, with_source=True)

    assert window.preview_button.isEnabled()
    window.close()


def test_rendering_compares_the_loaded_file_with_the_current_edit_then_cleans_up(qapp, tmp_path, monkeypatch, make_grid_twin):
    _FakeRenderer.instances.clear()
    monkeypatch.setattr("src.viewer.app.PreviewRenderer", _FakeRenderer)
    window = _open(qapp, tmp_path, monkeypatch, make_grid_twin, with_source=True)
    # Edit the in-memory twin so "after" genuinely differs from the file on disk.
    window.digital_twin_package.mesh_vertices[0] += np.float32(3.0)

    window._on_render_preview()

    renderer = _FakeRenderer.instances[-1]
    packages = renderer.started["packages"]
    assert packages["before"] == str(tmp_path / "grid_digital_twin.npz")
    assert renderer.after_existed_at_start  # the edited state was saved for the renderer
    assert not window.preview_button.isEnabled()  # no second render while one is running
    assert renderer.started["scene_source"] == tmp_path / "scene_source"

    work_dir = renderer.started["out_dir"]
    results = [(name, view, _png(work_dir / f"{name}_{view}.png")) for view in (0, 1) for name in ("before", "after")]
    renderer.finished.emit(True, results)

    assert window.preview_button.isEnabled()
    assert not (work_dir / "after.npz").exists()  # the large package is dropped once rendered
    assert window._preview_dialog.save_button.isEnabled()  # dialog got the images
    window._preview_dialog.close()

    assert not work_dir.exists()  # temp dir removed with the dialog
    assert window._preview_dialog is None
    window.close()


def test_a_failed_render_is_shown_in_the_dialog_and_re_enables_the_button(qapp, tmp_path, monkeypatch, make_grid_twin):
    _FakeRenderer.instances.clear()
    monkeypatch.setattr("src.viewer.app.PreviewRenderer", _FakeRenderer)
    window = _open(qapp, tmp_path, monkeypatch, make_grid_twin, with_source=True)
    window._on_render_preview()
    renderer = _FakeRenderer.instances[-1]

    renderer.finished.emit(False, "Preview render exited with code 1.")

    assert "code 1" in window._preview_dialog.status_label.text()
    assert window.preview_button.isEnabled()
    window._preview_dialog.close()
    window.close()


def test_closing_the_dialog_mid_render_cancels_the_renderer(qapp, tmp_path, monkeypatch, make_grid_twin):
    _FakeRenderer.instances.clear()
    monkeypatch.setattr("src.viewer.app.PreviewRenderer", _FakeRenderer)
    window = _open(qapp, tmp_path, monkeypatch, make_grid_twin, with_source=True)
    window._on_render_preview()
    renderer = _FakeRenderer.instances[-1]
    work_dir = renderer.started["out_dir"]

    window._preview_dialog.close()

    assert renderer.cancelled
    assert not work_dir.exists()
    assert window.preview_button.isEnabled()
    window.close()
