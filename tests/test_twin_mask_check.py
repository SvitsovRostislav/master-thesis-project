import json
import sys
from pathlib import Path

import pytest
from PyQt5 import QtCore

from src.viewer.twin_mask_check import TwinMaskChecker, scene_has_masks

_STUB_CHECK = '''
import json, sys, time
from pathlib import Path

args = sys.argv[1:]
twin = args[args.index("--twin") + 1]
Path(twin + ".argv.json").write_text(json.dumps(sys.argv))
print("Reading cameras...", flush=True)
name = Path(twin).name
if "fails" in name:
    print("Traceback: boom", flush=True)
    sys.exit(3)
if "silent" in name:
    sys.exit(0)
if "hangs" in name:
    time.sleep(60)
if "garbled" in name:
    print("TWINCHECK not-a-number 5 24 24", flush=True)
    sys.exit(0)
print("8.5% of 1,234 visible Gaussians lie inside the masks of 24 views.", flush=True)
print("TWINCHECK 0.085000 1234 24 24", flush=True)
'''


@pytest.fixture
def stub_script(tmp_path) -> Path:
    path = tmp_path / "stub_check.py"
    path.write_text(_STUB_CHECK)
    return path


def _wait_for(qapp, condition, timeout_ms: int = 15_000) -> bool:
    loop, ticker = QtCore.QEventLoop(), QtCore.QTimer()
    ticker.timeout.connect(lambda: loop.quit() if condition() else None)
    ticker.start(20)
    QtCore.QTimer.singleShot(timeout_ms, loop.quit)
    if not condition():
        loop.exec_()
    ticker.stop()
    return condition()


def _check(qapp, twin: Path, scene: Path, script: Path, python_exe=None) -> dict:
    checker = TwinMaskChecker(python_exe=python_exe or sys.executable, script_path=script)
    outcome = {"answers": []}
    checker.finished.connect(lambda ok, payload: outcome["answers"].append((ok, payload)))
    checker.start(twin, scene)
    _wait_for(qapp, lambda: outcome["answers"])
    _wait_for(qapp, lambda: False, timeout_ms=150)  # would a second answer follow?
    outcome["success"], outcome["payload"] = outcome["answers"][0] if outcome["answers"] else (None, None)
    return outcome


class TestCheck:
    def test_the_answer_carries_the_share_and_what_it_was_made_of(self, qapp, tmp_path, stub_script):
        outcome = _check(qapp, tmp_path / "lego_x_digital_twin.npz", tmp_path / "scene", stub_script)

        assert outcome["success"] is True
        assert outcome["payload"] == {"share": 0.085, "considered": 1234, "views": 24}
        assert len(outcome["answers"]) == 1

    def test_the_script_is_given_the_twin_and_the_scene_even_when_their_paths_contain_spaces(self, qapp, tmp_path, stub_script):
        twin = tmp_path / "my scenes" / "lego x_digital_twin.npz"
        twin.parent.mkdir()
        scene = tmp_path / "scene source"

        _check(qapp, twin, scene, stub_script)

        argv = json.loads(Path(str(twin) + ".argv.json").read_text())
        assert argv[argv.index("--twin") + 1] == str(twin)
        assert argv[argv.index("--scene-path") + 1] == str(scene)

    def test_a_script_that_fails_reports_its_last_lines(self, qapp, tmp_path, stub_script):
        outcome = _check(qapp, tmp_path / "fails_digital_twin.npz", tmp_path, stub_script)

        assert outcome["success"] is False
        assert "exited with code 3" in outcome["payload"] and "Traceback: boom" in outcome["payload"]

    def test_a_script_that_ends_without_a_result_is_a_failure(self, qapp, tmp_path, stub_script):
        outcome = _check(qapp, tmp_path / "silent_digital_twin.npz", tmp_path, stub_script)

        assert outcome["success"] is False and "no result" in outcome["payload"]

    def test_a_result_line_that_cannot_be_read_is_no_result(self, qapp, tmp_path, stub_script):
        outcome = _check(qapp, tmp_path / "garbled_digital_twin.npz", tmp_path, stub_script)

        assert outcome["success"] is False and "no result" in outcome["payload"]

    def test_a_python_that_is_not_there_is_reported_once(self, qapp, tmp_path, stub_script):
        outcome = _check(qapp, tmp_path / "t_digital_twin.npz", tmp_path, stub_script, python_exe=tmp_path / "no_python.exe")

        assert outcome["success"] is False and "Could not start" in outcome["payload"]
        assert len(outcome["answers"]) == 1

    def test_cancelling_ends_the_process_and_answers_cancelled_before_returning(self, qapp, tmp_path, stub_script):
        checker = TwinMaskChecker(python_exe=sys.executable, script_path=stub_script)
        answers = []
        checker.finished.connect(lambda ok, payload: answers.append((ok, payload)))
        checker.start(tmp_path / "hangs_digital_twin.npz", tmp_path)
        assert _wait_for(qapp, lambda: Path(str(tmp_path / "hangs_digital_twin.npz") + ".argv.json").is_file())  # it is running

        checker.cancel()

        assert answers == [(False, "Cancelled.")]

    def test_cancelling_a_check_that_has_not_started_does_nothing(self, qapp):
        TwinMaskChecker().cancel()


class TestSceneHasMasks:
    def test_a_folder_of_png_masks_counts(self, tmp_path):
        (tmp_path / "masks").mkdir()
        (tmp_path / "masks" / "0000.png").write_bytes(b"x")

        assert scene_has_masks(tmp_path)

    def test_the_suffix_may_be_in_capitals(self, tmp_path):
        (tmp_path / "masks").mkdir()
        (tmp_path / "masks" / "0000.PNG").write_bytes(b"x")

        assert scene_has_masks(tmp_path)

    def test_an_empty_folder_or_one_of_other_files_does_not(self, tmp_path):
        (tmp_path / "masks").mkdir()
        assert not scene_has_masks(tmp_path)

        (tmp_path / "masks" / "readme.txt").write_bytes(b"x")
        assert not scene_has_masks(tmp_path)

    def test_a_scene_without_the_folder_does_not(self, tmp_path):
        assert not scene_has_masks(tmp_path)
        assert not scene_has_masks(tmp_path / "missing")
