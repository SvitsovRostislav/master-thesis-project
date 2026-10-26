from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from src.viewer.app import MainWindow
from src.viewer.digital_twin import load_digital_twin_package, save_digital_twin_package
from src.viewer.twin_provenance import check_cache_path, load_cached_check, save_cached_check

RECORD = {
    "schema": 1, "built_at": "2026-03-05T14:07:09", "scene": "grid", "photographs": 40, "masks_available": 40,
    "masked": True, "preset": {"baseline_iterations": 3000, "coarse_iterations": 2000, "refine_iterations": 1500, "mesh_triangles_target": 100000},
}


class _FakeChecker(QtCore.QObject):

    finished = QtCore.pyqtSignal(bool, object)
    instances = []

    def __init__(self, parent=None):
        super().__init__(parent)
        self.starts = []
        self.cancelled = False
        _FakeChecker.instances.append(self)

    def start(self, twin_path, scene_source):
        self.starts.append((str(twin_path), Path(scene_source)))

    def cancel(self):
        self.cancelled = True

    def answer(self, share, considered=50_000, views=24):
        self.finished.emit(True, {"share": share, "considered": considered, "views": views})

    def fail(self, message="boom"):
        self.finished.emit(False, message)


def _make_twin(tmp_path, make_grid_twin, name="grid", record=None) -> Path:
    twin = tmp_path / f"{name}_digital_twin.npz"
    make_grid_twin(twin)
    if record is not None:
        package = load_digital_twin_package(str(twin))
        package.provenance = record
        save_digital_twin_package(str(twin), package)
    return twin


def _window(qapp, tmp_path, monkeypatch, masks=True, python=True) -> MainWindow:
    scene = tmp_path / "scene_source"
    (scene / "masks").mkdir(parents=True, exist_ok=True)
    if masks:
        (scene / "masks" / "0000.png").write_bytes(b"x")
    fake_python = tmp_path / "python.exe"
    if python:
        fake_python.write_bytes(b"")
    monkeypatch.setattr("src.viewer.app.find_scene_source_for_twin", lambda path: scene)
    monkeypatch.setattr("src.viewer.app.sam_checkpoint_available", lambda *args: False)
    monkeypatch.setattr("src.viewer.app.TwinMaskChecker", _FakeChecker)
    monkeypatch.setattr("src.viewer.app.GS_TRAIN_PYTHON", fake_python)
    _FakeChecker.instances.clear()
    window = MainWindow(initial_path=None)
    window.show()
    return window


def _load(qapp, window, twin) -> None:
    window._load_digital_twin(str(twin))
    qapp.processEvents()


def _notice_shown(window) -> bool:
    return window._twin_notice_action.isVisible()


class TestBuildRecord:
    def test_a_twin_with_a_record_shows_it_as_the_tooltip_of_the_status_text(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _window(qapp, tmp_path, monkeypatch)

        _load(qapp, window, _make_twin(tmp_path, make_grid_twin, record=RECORD))

        tip = window.status_label.toolTip()
        assert "2026-03-05 14:07:09" in tip and "40 photographs" in tip and "3000/2000/1500" in tip
        window.close()

    def test_a_twin_without_a_record_says_so(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _window(qapp, tmp_path, monkeypatch)

        _load(qapp, window, _make_twin(tmp_path, make_grid_twin))

        assert "No build record" in window.status_label.toolTip()
        window.close()

    def test_the_record_survives_editing_and_saving(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _window(qapp, tmp_path, monkeypatch)
        _load(qapp, window, _make_twin(tmp_path, make_grid_twin, record=RECORD))
        window.digital_twin_package.remove_faces_by_vertex_selection([0])
        saved = tmp_path / "edited.npz"

        save_digital_twin_package(str(saved), window.digital_twin_package)

        assert load_digital_twin_package(str(saved)).provenance == RECORD
        window.close()


class TestMaskNotice:
    def test_no_check_starts_for_a_scene_without_masks(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _window(qapp, tmp_path, monkeypatch, masks=False)

        _load(qapp, window, _make_twin(tmp_path, make_grid_twin))

        assert _FakeChecker.instances == [] and not _notice_shown(window)
        window.close()

    def test_no_check_starts_for_a_twin_whose_record_says_it_was_masked(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _window(qapp, tmp_path, monkeypatch)

        _load(qapp, window, _make_twin(tmp_path, make_grid_twin, record=RECORD))

        assert _FakeChecker.instances == [] and not _notice_shown(window)
        window.close()

    def test_no_check_starts_when_the_training_environment_is_missing(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _window(qapp, tmp_path, monkeypatch, python=False)

        _load(qapp, window, _make_twin(tmp_path, make_grid_twin))

        assert _FakeChecker.instances == [] and not _notice_shown(window)
        window.close()

    def test_a_twin_without_a_record_of_a_scene_with_masks_is_checked_in_the_background(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _window(qapp, tmp_path, monkeypatch)
        twin = _make_twin(tmp_path, make_grid_twin)

        _load(qapp, window, twin)

        (checker,) = _FakeChecker.instances
        assert checker.starts == [(str(twin), tmp_path / "scene_source")]
        assert not _notice_shown(window)  # nothing is said before the answer, and loading did not wait for it
        assert window.mesh_panel.canvas.native.isVisible()
        window.close()

    def test_a_low_share_shows_a_notice_with_the_reason_as_its_tooltip(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _window(qapp, tmp_path, monkeypatch)
        _load(qapp, window, _make_twin(tmp_path, make_grid_twin))

        _FakeChecker.instances[-1].answer(0.085)

        assert _notice_shown(window)
        assert "8%" in window.twin_notice_label.text()
        assert "without masking" in window.twin_notice_label.toolTip()
        window.close()

    def test_a_high_share_shows_nothing(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _window(qapp, tmp_path, monkeypatch)
        _load(qapp, window, _make_twin(tmp_path, make_grid_twin))

        _FakeChecker.instances[-1].answer(0.998)

        assert not _notice_shown(window) and window.twin_notice_label.toolTip() == ""
        window.close()

    def test_the_verdict_is_kept_next_to_the_twin(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _window(qapp, tmp_path, monkeypatch)
        twin = _make_twin(tmp_path, make_grid_twin)
        _load(qapp, window, twin)

        _FakeChecker.instances[-1].answer(0.085, considered=51809, views=24)

        assert load_cached_check(twin) == {"share": 0.085, "considered": 51809, "views": 24}
        window.close()

    def test_a_kept_verdict_is_used_instead_of_checking_again(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _window(qapp, tmp_path, monkeypatch)
        twin = _make_twin(tmp_path, make_grid_twin)
        save_cached_check(twin, 0.085, 51809, 24)

        _load(qapp, window, twin)

        assert _FakeChecker.instances == []
        assert _notice_shown(window) and "8%" in window.twin_notice_label.text()
        window.close()

    def test_a_kept_verdict_for_a_twin_that_has_changed_since_is_not_used(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _window(qapp, tmp_path, monkeypatch)
        twin = _make_twin(tmp_path, make_grid_twin)
        save_cached_check(twin, 0.085, 51809, 24)
        _make_twin(tmp_path, make_grid_twin, record={"masked": False})  # rewritten: a different file now
        assert load_cached_check(twin) is None

        _load(qapp, window, twin)

        assert len(_FakeChecker.instances) == 1 and not _notice_shown(window)
        window.close()

    def test_a_failed_check_says_nothing_and_shows_no_dialog(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _window(qapp, tmp_path, monkeypatch)
        twin = _make_twin(tmp_path, make_grid_twin)
        shown = []
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: shown.append(args))
        monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args: shown.append(args))
        _load(qapp, window, twin)

        _FakeChecker.instances[-1].fail("the scene could not be read")

        assert shown == [] and not _notice_shown(window)
        assert not check_cache_path(twin).exists()  # a failure is not a verdict
        window.close()

    def test_loading_another_twin_cancels_the_check_and_clears_the_notice(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _window(qapp, tmp_path, monkeypatch)
        first = _make_twin(tmp_path, make_grid_twin, name="first")
        _load(qapp, window, first)
        _FakeChecker.instances[-1].answer(0.085)
        assert _notice_shown(window)
        _load(qapp, window, _make_twin(tmp_path, make_grid_twin, name="second"))

        assert not _notice_shown(window)
        assert len(_FakeChecker.instances) == 2 and not _FakeChecker.instances[0].cancelled  # its answer had come; nothing was left to cancel
        _load(qapp, window, _make_twin(tmp_path, make_grid_twin, name="third"))
        assert _FakeChecker.instances[1].cancelled
        window.close()

    def test_a_late_answer_for_the_previous_twin_is_ignored(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _window(qapp, tmp_path, monkeypatch)
        first = _make_twin(tmp_path, make_grid_twin, name="first")
        _load(qapp, window, first)
        old = _FakeChecker.instances[-1]
        _load(qapp, window, _make_twin(tmp_path, make_grid_twin, name="second"))

        old.answer(0.01)

        assert not _notice_shown(window)
        assert not check_cache_path(first).exists()
        window.close()

    def test_a_twin_that_needs_no_check_clears_the_notice_of_the_one_before(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _window(qapp, tmp_path, monkeypatch)
        _load(qapp, window, _make_twin(tmp_path, make_grid_twin, name="first"))
        _FakeChecker.instances[-1].answer(0.085)

        _load(qapp, window, _make_twin(tmp_path, make_grid_twin, name="second", record=RECORD))

        assert not _notice_shown(window) and window.twin_notice_label.text() == ""
        window.close()

    def test_closing_the_window_cancels_a_check_that_is_running(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _window(qapp, tmp_path, monkeypatch)
        _load(qapp, window, _make_twin(tmp_path, make_grid_twin))
        checker = _FakeChecker.instances[-1]

        window.close()

        assert checker.cancelled

    def test_the_notice_does_not_get_in_the_way_of_editing(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _window(qapp, tmp_path, monkeypatch)
        _load(qapp, window, _make_twin(tmp_path, make_grid_twin))
        _FakeChecker.instances[-1].answer(0.085)

        assert window.sculpt_button.isEnabled() and window.select_button.isEnabled() and window.save_button.isEnabled()
        window.close()
