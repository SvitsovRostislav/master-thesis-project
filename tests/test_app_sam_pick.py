from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets

from src.viewer.app import MainWindow
from src.viewer.sam_pick import INCLUSIVE, MAJORITY


class _FakeRunner(QtCore.QObject):

    finished = QtCore.pyqtSignal(bool, object)
    log_line = QtCore.pyqtSignal(str)
    instances = []

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.busy = False
        self.closed = False
        self.starts = []
        self.requests = []
        _FakeRunner.instances.append(self)

    def is_running(self):
        return self.running

    def is_busy(self):
        return self.busy

    def start(self, scene_source, positions_path, checkpoint=None):
        self.running = True
        # Read now: the app removes the file with its work directory.
        self.starts.append({"scene_source": scene_source, "positions": np.load(positions_path), "positions_path": Path(positions_path)})

    def request(self, point, level, output_path, extra_points=(), negative_points=(), **vote):
        self.busy = True
        self.requests.append({
            "point": np.asarray(point), "level": level, "output": Path(output_path),
            "extra": np.asarray(extra_points).reshape(-1, 3), "negatives": np.asarray(negative_points).reshape(-1, 3),
            "vote": vote,
        })

    def close(self, wait_ms=0):
        self.closed, self.running, self.close_wait = True, False, wait_ms

    def answer(self, selected: np.ndarray, views: int = 12) -> None:
        request = self.requests[-1]
        np.save(request["output"], selected)
        self.busy = False
        self.finished.emit(True, {"path": str(request["output"]), "selected": int(selected.sum()), "views": views, "rounds": 3})

    def fail(self, message: str) -> None:
        self.busy = False
        self.finished.emit(False, message)


def _open(qapp, tmp_path, monkeypatch, make_grid_twin, source=True, checkpoint=True, name="grid") -> MainWindow:
    scene_source = tmp_path / "scene_source"
    scene_source.mkdir(exist_ok=True)
    monkeypatch.setattr("src.viewer.app.find_scene_source_for_twin", lambda path: scene_source if source else None)
    monkeypatch.setattr("src.viewer.app.sam_checkpoint_available", lambda *args: checkpoint)
    monkeypatch.setattr("src.viewer.app.SamPickRunner", _FakeRunner)
    _FakeRunner.instances.clear()
    twin = tmp_path / f"{name}_digital_twin.npz"
    make_grid_twin(twin)
    window = MainWindow(initial_path=None)
    window.show()
    window._load_digital_twin(str(twin))
    qapp.processEvents()
    return window


def _delete(window: MainWindow, vertices) -> None:
    window.mesh_panel.set_selection(np.asarray(vertices))
    window._on_delete_selected()


def _click(window: MainWindow, vertex: int = 40):
    window.sam_pick_button.setChecked(True)
    return window._resolve_pick(vertex)


class TestAvailability:
    def test_disabled_before_a_twin_is_loaded(self, qapp):
        window = MainWindow(initial_path=None)

        assert not window.sam_pick_button.isEnabled() and not window.sam_level_combo.isEnabled()
        window.close()

    def test_disabled_when_the_scene_source_is_not_found(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin, source=False)

        assert not window.sam_pick_button.isEnabled()
        window.close()

    def test_disabled_when_the_sam_checkpoint_is_missing(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin, checkpoint=False)

        assert not window.sam_pick_button.isEnabled()
        window.close()

    def test_enabled_with_both_and_the_level_defaults_to_object(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)

        assert window.sam_pick_button.isEnabled() and window.sam_level_combo.isEnabled()
        assert window.sam_level_combo.currentText() == "Object"
        assert not window.sam_pick_button.isChecked()  # off until asked for
        window.close()


class TestPicking:
    def test_off_it_changes_nothing_about_ordinary_picking(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)

        assert window._resolve_pick(40) is None  # None = the editor's own connectivity pick
        assert _FakeRunner.instances == []
        window.close()

    def test_the_worker_gets_the_reference_positions_and_the_click_as_the_vertex_was_loaded(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        reference = window._sam_reference
        original_vertex = reference.vertices[40].copy()
        original_positions = reference.positions.copy()
        window.digital_twin_package.mesh_vertices[40] += np.float32(5.0)  # Sculpt has moved it since
        window.sam_level_combo.setCurrentText("Whole")

        result = _click(window, 40)

        runner = _FakeRunner.instances[-1]
        assert len(result) == 0  # the selection arrives later
        assert np.allclose(runner.starts[0]["positions"], original_positions)
        assert np.allclose(runner.requests[0]["point"], original_vertex)
        assert runner.requests[0]["level"] == 2
        assert runner.starts[0]["scene_source"] == tmp_path / "scene_source"
        window.close()

    def test_the_worker_is_started_once_and_reused_for_later_clicks(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        _click(window, 40)
        runner = _FakeRunner.instances[-1]
        runner.answer(np.zeros(len(window.digital_twin_package), dtype=bool))

        window._resolve_pick(41)

        assert len(_FakeRunner.instances) == 1 and len(runner.starts) == 1
        assert [np.allclose(r["point"], window._sam_reference.vertices[v]) for r, v in zip(runner.requests, (40, 41))] == [True, True]
        window.close()

    def test_a_worker_that_has_died_is_started_again_by_the_next_click(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        shown = []
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda parent, title, text: shown.append(text))
        _click(window, 40)
        runner = _FakeRunner.instances[-1]
        runner.running = False  # it crashed ...
        runner.fail("SAM Pick worker exited with code 1.")  # ... and the request failed

        window._resolve_pick(41)

        assert len(runner.starts) == 2
        window.close()

    def test_a_second_click_while_one_is_running_does_not_ask_again(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        _click(window, 40)

        window._resolve_pick(41)

        assert len(_FakeRunner.instances[-1].requests) == 1
        assert "still working" in window.status_label.text()
        window.close()

    def test_the_answer_becomes_the_selection_and_only_its_own_file_goes(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        _click(window, 40)
        runner = _FakeRunner.instances[-1]
        package = window.digital_twin_package
        mask = np.zeros(len(package), dtype=bool)
        mask[:100] = True

        runner.answer(mask)

        expected = np.unique(package.mesh_faces[:100].reshape(-1))  # one Gaussian per face in the grid twin
        assert np.array_equal(np.sort(window.mesh_panel.selected_vertices), expected)
        assert window.delete_button.isEnabled()
        assert "100 Gaussians" in window.status_label.text() and "12 photographs" in window.status_label.text()
        assert not runner.requests[0]["output"].exists()  # the answer is consumed ...
        assert runner.starts[0]["positions_path"].exists()  # ... the worker's own input stays for the next click
        window.close()

    def test_an_answer_with_no_object_leaves_the_selection_alone(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        window.mesh_panel.set_selection(np.array([1, 2, 3]))
        _click(window, 40)

        _FakeRunner.instances[-1].answer(np.zeros(len(window.digital_twin_package), dtype=bool))

        assert np.array_equal(window.mesh_panel.selected_vertices, [1, 2, 3])
        assert "no object" in window.status_label.text()
        window.close()

    def test_a_failure_is_shown_and_frees_the_button(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        shown = []
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda parent, title, text: shown.append((title, text)))
        _click(window, 40)
        runner = _FakeRunner.instances[-1]

        runner.fail("SAM Pick exited with code 1.")

        assert shown == [("SAM Pick failed", "SAM Pick exited with code 1.")]
        window._resolve_pick(41)
        assert len(runner.requests) == 2  # free for the next click
        window.close()

    def test_a_result_file_that_cannot_be_read_is_reported_not_raised(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        shown = []
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda parent, title, text: shown.append(text))
        _click(window, 40)

        _FakeRunner.instances[-1].finished.emit(True, {"path": str(tmp_path / "missing.npy"), "selected": 1, "views": 1, "rounds": 1})

        assert shown and "could not be read" in shown[0]
        assert window.mesh_panel.selected_vertices is None
        window.close()

    def test_sam_pick_takes_priority_over_semantic_pick(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        window.semantic_pick_button.setEnabled(True)
        window.semantic_pick_button.setChecked(True)

        _click(window, 40)

        assert len(_FakeRunner.instances[-1].requests) == 1
        window.close()


class TestStaleGeometry:
    def test_moving_vertices_does_not_disable_it(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        window.digital_twin_package.mesh_vertices[:50] += np.float32(2.0)

        _click(window, 40)

        assert len(_FakeRunner.instances[-1].requests) == 1
        window.close()

    def test_after_a_delete_it_still_works_and_sends_the_click_from_where_the_vertex_was_loaded(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        reference = window._sam_reference
        _delete(window, [0, 1, 25, 26])
        package = window.digital_twin_package

        _click(window, 200)

        (request,) = _FakeRunner.instances[-1].requests
        assert package.vertex_origin[200] != 200  # the delete renumbered the vertices...
        assert np.allclose(request["point"], reference.vertices[package.vertex_origin[200]])
        assert np.allclose(request["point"], package.mesh_vertices[200])  # ...and it is the vertex that was clicked
        window.close()

    def test_an_answer_that_arrives_after_a_delete_selects_what_is_left_of_the_object(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        reference = window._sam_reference
        _click(window, 200)
        runner = _FakeRunner.instances[-1]
        _delete(window, [0, 1, 25, 26])
        package = window.digital_twin_package
        answer = np.zeros(len(reference.positions), dtype=bool)
        answer[:100] = True  # an object of which a corner has just been deleted

        runner.answer(answer)

        selected = window.mesh_panel.selected_vertices
        assert np.array_equal(np.sort(selected), reference.selection_in(package, answer))
        loaded_object = np.unique(reference.faces[:100].reshape(-1))
        assert set(package.vertex_origin[selected].tolist()) <= set(loaded_object.tolist())
        assert "discarded" not in window.status_label.text()
        window.close()

    def test_an_answer_that_lies_entirely_in_the_deleted_part_says_so(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        reference = window._sam_reference
        _delete(window, [0, 1, 2, 25, 26, 27])
        _click(window, 200)
        answer = np.zeros(len(reference.positions), dtype=bool)
        answer[:2] = True  # faces 0 and 1, both deleted

        _FakeRunner.instances[-1].answer(answer)

        assert window.mesh_panel.selected_vertices is None
        assert "deleted" in window.status_label.text()
        window.close()

    def test_a_point_that_no_photograph_shows_is_reported_as_such(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        _click(window, 40)

        _FakeRunner.instances[-1].answer(np.zeros(len(window._sam_reference.positions), dtype=bool), views=0)

        assert window.mesh_panel.selected_vertices is None
        assert "no training photograph" in window.status_label.text()
        window.close()

    def test_a_delete_does_not_restart_the_worker(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        _click(window, 40)
        runner = _FakeRunner.instances[-1]
        runner.answer(np.zeros(len(window._sam_reference.positions), dtype=bool))
        _delete(window, [0, 1, 25, 26])

        window._resolve_pick(200)

        assert len(_FakeRunner.instances) == 1 and len(runner.starts) == 1 and len(runner.requests) == 2
        window.close()

    def test_it_keeps_working_after_undo_and_after_fill_density_changes_the_gaussians(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        reference = window._sam_reference
        _delete(window, [0, 1, 25, 26])
        window._on_undo_delete()

        _click(window, 200)
        runner = _FakeRunner.instances[-1]
        assert np.allclose(runner.requests[-1]["point"], reference.vertices[200])  # undone: numbered as loaded again
        runner.answer(np.zeros(len(reference.positions), dtype=bool))

        package = window.digital_twin_package
        package.add_gaussians(  # what Fill Density does when a triangle is stretched
            face_idx=np.array([10, 11], dtype=np.int32), bary=np.full((2, 3), 1 / 3, dtype=np.float32),
            colors=np.ones((2, 3), dtype=np.float32), opacities=np.ones(2, dtype=np.float32),
            scales=np.full((2, 3), 0.01, dtype=np.float32), quaternions=np.tile(np.float32([1, 0, 0, 0]), (2, 1)),
        )
        window._resolve_pick(210)
        answer = np.zeros(len(reference.positions), dtype=bool)
        answer[10:30] = True
        runner.answer(answer)

        assert len(runner.requests) == 2
        assert np.array_equal(np.sort(window.mesh_panel.selected_vertices), np.unique(package.mesh_faces[10:30].reshape(-1)))
        window.close()

    def test_exclude_after_a_delete_sends_its_points_from_where_they_were_loaded(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        reference = window._sam_reference
        _delete(window, [0, 1, 25, 26])
        package = window.digital_twin_package
        _click(window, 40)
        runner = _FakeRunner.instances[-1]
        answer = np.zeros(len(reference.positions), dtype=bool)
        answer[200:400] = True
        runner.answer(answer)
        window.sam_mode_combo.setCurrentText("Exclude")

        _click(window, 300)

        request = runner.requests[-1]
        assert np.allclose(request["point"], reference.vertices[package.vertex_origin[40]])
        assert np.allclose(request["negatives"][-1], reference.vertices[package.vertex_origin[300]])
        window.close()

    def test_loading_another_twin_ends_the_worker_and_ignores_its_answer(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        _click(window, 40)
        runner = _FakeRunner.instances[-1]
        work_dir = runner.starts[0]["positions_path"].parent
        other = tmp_path / "other_digital_twin.npz"
        make_grid_twin(other)

        window._load_digital_twin(str(other))

        assert runner.closed and not work_dir.exists() and runner.close_wait == 0  # no blocking for a twin switch
        assert not window.sam_pick_button.isChecked()
        runner.finished.emit(True, {"path": str(tmp_path / "gone.npy"), "selected": 0, "views": 0, "rounds": 0})  # nobody listens
        assert window.mesh_panel.selected_vertices is None
        window.close()

    def test_closing_the_window_ends_the_worker_and_removes_its_files(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        _click(window, 40)
        runner = _FakeRunner.instances[-1]
        work_dir = runner.starts[0]["positions_path"].parent

        window.close()

        assert runner.closed and not work_dir.exists()
        assert runner.close_wait > 0  # the application waits for the process before it exits


class _Key:
    """What a vispy key looks like to the application: something with a name."""

    def __init__(self, name):
        self.name = name


def _answer_with_first(window, runner, count: int):
    mask = np.zeros(len(window.digital_twin_package), dtype=bool)
    mask[:count] = True
    runner.answer(mask)
    return np.unique(window.digital_twin_package.mesh_faces[:count].reshape(-1))


def _replaced(qapp, tmp_path, monkeypatch, make_grid_twin, vertex: int = 40, count: int = 100):
    """A window whose selection is the answer to one Replace click on `vertex`: the first `count`
    Gaussians. Returns the window, the runner, and the selected vertices."""
    window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
    _click(window, vertex)
    runner = _FakeRunner.instances[-1]
    selected = _answer_with_first(window, runner, count)
    return window, runner, selected


class TestClickModes:
    def test_the_selector_offers_the_four_modes_and_starts_on_replace(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)

        modes = [window.sam_mode_combo.itemText(i) for i in range(window.sam_mode_combo.count())]

        assert modes == ["Replace", "Add", "Subtract", "Exclude"]
        assert window.sam_mode_combo.currentText() == "Replace" and window.sam_mode_combo.isEnabled()
        window.close()

    def test_the_selector_is_disabled_when_sam_pick_is(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin, source=False)

        assert not window.sam_mode_combo.isEnabled()
        window.close()

    def test_replace_sends_only_the_click_as_it_always_did(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        window.mesh_panel.set_selection(np.array([1, 2, 3]))

        _click(window, 40)

        request = _FakeRunner.instances[-1].requests[0]
        assert len(request["extra"]) == 0 and len(request["negatives"]) == 0
        window.close()

    def test_replace_swaps_the_selection_for_the_answer(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        window.mesh_panel.set_selection(np.array([1, 2, 3]))
        _click(window, 40)

        wanted = _answer_with_first(window, _FakeRunner.instances[-1], 100)

        assert np.array_equal(np.sort(window.mesh_panel.selected_vertices), wanted)
        window.close()

    def test_add_joins_the_answer_to_the_selection(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        window.mesh_panel.set_selection(np.array([400, 401, 402]))
        window.sam_mode_combo.setCurrentText("Add")
        _click(window, 40)

        wanted = _answer_with_first(window, _FakeRunner.instances[-1], 100)

        assert np.array_equal(np.sort(window.mesh_panel.selected_vertices), np.union1d(wanted, [400, 401, 402]))
        assert "add" in window.status_label.text()
        window.close()

    def test_add_with_nothing_selected_just_selects_the_answer(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        window.sam_mode_combo.setCurrentText("Add")
        _click(window, 40)

        wanted = _answer_with_first(window, _FakeRunner.instances[-1], 100)

        assert np.array_equal(np.sort(window.mesh_panel.selected_vertices), wanted)
        window.close()

    def test_subtract_takes_the_answer_out_of_the_selection(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        everything = np.unique(window.digital_twin_package.mesh_faces[:300].reshape(-1))
        window.mesh_panel.set_selection(everything)
        window.sam_mode_combo.setCurrentText("Subtract")
        _click(window, 40)

        removed = _answer_with_first(window, _FakeRunner.instances[-1], 100)

        assert np.array_equal(np.sort(window.mesh_panel.selected_vertices), np.setdiff1d(everything, removed))
        window.close()

    def test_subtracting_the_whole_selection_clears_it(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        window.mesh_panel.set_selection(np.unique(window.digital_twin_package.mesh_faces[:100].reshape(-1)))
        window.sam_mode_combo.setCurrentText("Subtract")
        _click(window, 40)

        _answer_with_first(window, _FakeRunner.instances[-1], 100)

        assert window.mesh_panel.selected_vertices is None
        assert not window.delete_button.isEnabled()
        window.close()

    def test_subtract_and_exclude_need_a_selection_and_do_not_ask_sam_without_one(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        for mode in ("Subtract", "Exclude"):
            window.sam_mode_combo.setCurrentText(mode)

            _click(window, 40)

            assert "nothing is selected" in window.status_label.text()
        assert _FakeRunner.instances == []
        window.close()

    def test_shift_adds_and_control_subtracts_whatever_the_selector_says(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        window.mesh_panel.set_selection(np.array([400, 401]))
        window.sam_pick_button.setChecked(True)

        window.mesh_panel.pick_modifiers = (_Key("Shift"),)
        assert window._sam_click_mode() == "Add"
        window.mesh_panel.pick_modifiers = (_Key("Control"),)
        assert window._sam_click_mode() == "Subtract"
        window.mesh_panel.pick_modifiers = (_Key("Alt"),)  # not one of ours
        assert window._sam_click_mode() == "Replace"
        window.mesh_panel.pick_modifiers = ()
        window.sam_mode_combo.setCurrentText("Exclude")
        assert window._sam_click_mode() == "Exclude"
        window.close()

    def test_a_shift_click_adds_to_the_selection(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        window.mesh_panel.set_selection(np.array([400, 401]))
        window.mesh_panel.pick_modifiers = (_Key("Shift"),)
        _click(window, 40)

        wanted = _answer_with_first(window, _FakeRunner.instances[-1], 100)

        assert np.array_equal(np.sort(window.mesh_panel.selected_vertices), np.union1d(wanted, [400, 401]))
        window.close()

    def test_exclude_runs_the_click_that_made_the_selection_again_with_the_excluded_vertex_as_a_negative(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window, runner, _ = _replaced(qapp, tmp_path, monkeypatch, make_grid_twin, vertex=40)
        reference = window._sam_reference
        window.digital_twin_package.mesh_vertices[300] += np.float32(4.0)  # Sculpt moved the vertex since loading
        window.sam_mode_combo.setCurrentText("Exclude")

        _click(window, 300)

        request = runner.requests[-1]
        assert np.allclose(request["point"], reference.vertices[40])  # the first click, as loaded
        assert len(request["extra"]) == 0
        assert request["negatives"].shape == (1, 3)
        assert np.allclose(request["negatives"][0], reference.vertices[300])  # as loaded, not as moved
        window.close()

    def test_exclude_only_ever_removes_from_the_selection(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window, runner, first = _replaced(qapp, tmp_path, monkeypatch, make_grid_twin, count=100)
        window.sam_mode_combo.setCurrentText("Exclude")
        _click(window, 300)

        smaller = _answer_with_first(window, runner, 50)

        assert np.array_equal(np.sort(window.mesh_panel.selected_vertices), np.intersect1d(first, smaller))
        assert len(window.mesh_panel.selected_vertices) < len(first)
        assert "removed" in window.status_label.text()
        window.close()

    def test_an_exclude_answer_that_is_larger_adds_nothing(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window, runner, first = _replaced(qapp, tmp_path, monkeypatch, make_grid_twin, count=100)
        window.sam_mode_combo.setCurrentText("Exclude")
        _click(window, 300)

        _answer_with_first(window, runner, 400)

        assert np.array_equal(np.sort(window.mesh_panel.selected_vertices), first)
        window.close()

    def test_every_exclude_click_so_far_goes_along_as_a_negative_and_the_pick_stays_the_same(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window, runner, _ = _replaced(qapp, tmp_path, monkeypatch, make_grid_twin, vertex=40, count=100)
        reference = window._sam_reference
        window.sam_mode_combo.setCurrentText("Exclude")
        _click(window, 300)
        _answer_with_first(window, runner, 90)

        _click(window, 310)

        request = runner.requests[-1]
        assert np.allclose(request["point"], reference.vertices[40])
        assert np.allclose(request["negatives"], reference.vertices[[300, 310]])
        window.close()

    def test_exclude_repeats_the_level_of_the_click_it_corrects_whatever_the_selector_says_now(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        window.sam_level_combo.setCurrentText("Whole")
        _click(window, 40)
        runner = _FakeRunner.instances[-1]
        _answer_with_first(window, runner, 100)
        window.sam_level_combo.setCurrentText("Part")
        window.sam_mode_combo.setCurrentText("Exclude")

        _click(window, 300)

        assert runner.requests[0]["level"] == 2 and runner.requests[-1]["level"] == 2
        window.close()

    def test_a_new_replace_click_starts_the_negatives_over(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window, runner, _ = _replaced(qapp, tmp_path, monkeypatch, make_grid_twin, count=100)
        window.sam_mode_combo.setCurrentText("Exclude")
        _click(window, 300)
        _answer_with_first(window, runner, 90)
        window.sam_mode_combo.setCurrentText("Replace")
        _click(window, 60)
        _answer_with_first(window, runner, 100)
        window.sam_mode_combo.setCurrentText("Exclude")

        _click(window, 310)

        assert runner.requests[-1]["negatives"].shape == (1, 3)
        window.close()

    def test_exclude_says_what_it_needs_when_the_selection_is_not_the_result_of_one_pick(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        # made by another tool
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        window.mesh_panel.set_selection(np.arange(0, 120))
        window.sam_mode_combo.setCurrentText("Exclude")
        _click(window, 300)
        assert "Click with Replace first" in window.status_label.text()
        assert _FakeRunner.instances == []
        window.close()

        # changed by Add
        window, runner, _ = _replaced(qapp, tmp_path, monkeypatch, make_grid_twin, count=100)
        window.sam_mode_combo.setCurrentText("Add")
        _click(window, 200)
        _answer_with_first(window, runner, 150)
        window.sam_mode_combo.setCurrentText("Exclude")
        n = len(runner.requests)
        _click(window, 300)
        assert "Click with Replace first" in window.status_label.text() and len(runner.requests) == n
        window.close()

        # changed by hand afterwards
        window, runner, _ = _replaced(qapp, tmp_path, monkeypatch, make_grid_twin, count=100)
        window.mesh_panel.set_selection(np.array([5, 6, 7]))
        window.sam_mode_combo.setCurrentText("Exclude")
        n = len(runner.requests)
        _click(window, 300)
        assert "Click with Replace first" in window.status_label.text() and len(runner.requests) == n
        window.close()

    def test_an_exclude_that_would_leave_nothing_changes_nothing_and_can_be_tried_again(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window, runner, first = _replaced(qapp, tmp_path, monkeypatch, make_grid_twin, count=100)
        window.sam_mode_combo.setCurrentText("Exclude")
        _click(window, 300)
        mask = np.zeros(len(window.digital_twin_package), dtype=bool)
        mask[500:600] = True  # nowhere near the selection

        runner.answer(mask)

        assert np.array_equal(np.sort(window.mesh_panel.selected_vertices), first)
        assert "nothing was changed" in window.status_label.text()
        _click(window, 310)
        assert len(runner.requests) == 3  # still allowed: the selection is still the pick's result
        window.close()

    def test_an_exclude_that_finds_no_object_leaves_the_selection_and_the_pick_alone(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window, runner, first = _replaced(qapp, tmp_path, monkeypatch, make_grid_twin, count=100)
        window.sam_mode_combo.setCurrentText("Exclude")
        _click(window, 300)

        runner.answer(np.zeros(len(window.digital_twin_package), dtype=bool))

        assert np.array_equal(np.sort(window.mesh_panel.selected_vertices), first)
        assert "no object" in window.status_label.text()
        window.close()

    def test_loading_another_twin_forgets_the_click_that_made_the_selection(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window, _, _ = _replaced(qapp, tmp_path, monkeypatch, make_grid_twin, count=100)
        assert window._sam_last_pick is not None
        other = tmp_path / "other_digital_twin.npz"
        make_grid_twin(other)

        window._load_digital_twin(str(other))

        assert window._sam_last_pick is None
        window.close()

    def test_the_mode_is_forgotten_after_the_answer(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        window.mesh_panel.set_selection(np.array([400]))
        window.mesh_panel.pick_modifiers = (_Key("Shift"),)
        _click(window, 40)
        _answer_with_first(window, _FakeRunner.instances[-1], 10)

        assert window._sam_pending_mode == "Replace"
        window.close()


class TestVote:
    def test_the_selector_follows_the_twin_and_starts_at_majority(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        assert window.sam_vote_combo.isEnabled() and window.sam_vote_combo.currentText() == "Majority"
        window.close()

        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin, checkpoint=False)
        assert not window.sam_vote_combo.isEnabled()
        window.close()

    def test_the_default_vote_sends_the_request_it_always_sent(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)

        _click(window)

        assert _FakeRunner.instances[0].requests[0]["vote"] == {}
        window.close()

    def test_inclusive_sends_the_lower_vote(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        window.sam_vote_combo.setCurrentText("Inclusive")

        _click(window)

        assert _FakeRunner.instances[0].requests[0]["vote"] == {"min_ratio": INCLUSIVE}
        window.close()

    def test_auto_is_inclusive_for_a_twin_built_with_masks(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        window.digital_twin_package.provenance = {"masked": True}
        window.sam_vote_combo.setCurrentText("Auto")

        _click(window)

        assert _FakeRunner.instances[0].requests[0]["vote"] == {"min_ratio": INCLUSIVE}
        assert "built with masks" in window.status_label.text()
        window.close()

    def test_auto_follows_the_mask_check_when_there_is_no_record(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        window.sam_vote_combo.setCurrentText("Auto")
        runner_requests = []

        for share, expected in ((0.998, INCLUSIVE), (0.085, MAJORITY), (None, MAJORITY)):
            window._twin_mask_share = share
            ratio, _ = window._sam_vote()
            runner_requests.append(ratio)

        assert runner_requests == [INCLUSIVE, MAJORITY, MAJORITY]
        window.close()

    def test_exclude_repeats_the_click_with_the_vote_it_was_made_with(self, qapp, tmp_path, monkeypatch, make_grid_twin):
        window = _open(qapp, tmp_path, monkeypatch, make_grid_twin)
        window.sam_vote_combo.setCurrentText("Inclusive")
        _click(window)
        runner = _FakeRunner.instances[0]
        runner.answer(np.arange(len(window.digital_twin_package)) < 200)
        window.sam_vote_combo.setCurrentText("Majority")  # changed after the click
        window.sam_mode_combo.setCurrentText("Exclude")

        _click(window, vertex=41)

        assert runner.requests[-1]["vote"] == {"min_ratio": INCLUSIVE}
        window.close()
