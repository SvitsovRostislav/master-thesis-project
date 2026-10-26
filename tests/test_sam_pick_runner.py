import json
import sys
from pathlib import Path

import pytest
from PyQt5 import QtCore

from src.viewer.sam_pick_runner import SamPickRunner, sam_checkpoint_available

_STUB_WORKER = '''
import json, sys, time
from pathlib import Path

args = sys.argv[1:]
Path(args[args.index("--positions") + 1] + ".argv.json").write_text(json.dumps(args))
print("Loading COLMAP cameras...", flush=True)
print("Ready.", flush=True)
served = 0
for line in sys.stdin:
    request = json.loads(line)
    Path(request["output"] + ".request.json").write_text(json.dumps(request))
    if request["level"] == 9:
        sys.exit(4)
    if request["level"] == 8:
        time.sleep(60)
    if request["level"] == 7:
        print("SAMPICK_ERROR boom", flush=True)
        continue
    served += 1
    Path(request["output"]).write_bytes(b"npy")
    print(f"round 1: prompting {served} view(s)", flush=True)
    print(f"SAMPICK {served} 7 3 {request['output']}", flush=True)
'''


@pytest.fixture
def stub_worker(tmp_path) -> Path:
    path = tmp_path / "stub_worker.py"
    path.write_text(_STUB_WORKER)
    return path


@pytest.fixture
def positions_file(tmp_path) -> Path:
    path = tmp_path / "positions.npy"
    path.write_bytes(b"x")
    return path


def _runner(stub_worker, **kwargs) -> SamPickRunner:
    return SamPickRunner(python_exe=sys.executable, script_path=stub_worker, **kwargs)


def _wait_for(qapp, condition, timeout_ms: int = 15_000) -> bool:
    loop, ticker = QtCore.QEventLoop(), QtCore.QTimer()
    ticker.timeout.connect(lambda: loop.quit() if condition() else None)
    ticker.start(20)
    QtCore.QTimer.singleShot(timeout_ms, loop.quit)
    if not condition():
        loop.exec_()
    ticker.stop()
    return condition()


def _ask(qapp, runner: SamPickRunner, level: int, output: Path, point=(1.0, 2.0, 3.0), cancel_after_ms=None) -> dict:
    """One request; returns the answer (and the log lines seen while waiting)."""
    outcome = {"logs": []}

    def on_finished(ok, payload):
        outcome.update(success=ok, payload=payload)

    runner.finished.connect(on_finished)
    runner.log_line.connect(outcome["logs"].append)
    if cancel_after_ms is not None:
        QtCore.QTimer.singleShot(cancel_after_ms, runner.cancel)
    runner.request(point, level, output)
    _wait_for(qapp, lambda: "success" in outcome)
    runner.finished.disconnect(on_finished)
    runner.log_line.disconnect(outcome["logs"].append)
    return outcome


class TestWorker:
    def test_one_worker_serves_successive_requests(self, qapp, tmp_path, stub_worker, positions_file):
        runner = _runner(stub_worker)
        runner.start(tmp_path / "scene", positions_file)

        first = _ask(qapp, runner, 0, tmp_path / "one.npy")
        second = _ask(qapp, runner, 0, tmp_path / "two.npy")

        assert first["success"] and second["success"]
        assert first["payload"]["selected"] == 1 and second["payload"]["selected"] == 2  # the same process counted both
        assert first["logs"].count("Ready.") + second["logs"].count("Ready.") <= 1  # it started once
        runner.close()

    def test_the_answer_carries_the_result_even_when_the_path_contains_spaces(self, qapp, tmp_path, stub_worker, positions_file):
        (tmp_path / "out dir").mkdir()
        output = tmp_path / "out dir" / "selected.npy"
        runner = _runner(stub_worker)
        runner.start(tmp_path, positions_file)

        outcome = _ask(qapp, runner, 0, output)

        assert outcome["payload"] == {"selected": 1, "views": 7, "rounds": 3, "path": str(output)}
        assert output.is_file()
        runner.close()

    def test_progress_lines_are_forwarded(self, qapp, tmp_path, stub_worker, positions_file):
        runner = _runner(stub_worker)
        runner.start(tmp_path, positions_file)

        outcome = _ask(qapp, runner, 0, tmp_path / "o.npy")

        assert "round 1: prompting 1 view(s)" in outcome["logs"]
        runner.close()

    def test_the_worker_is_started_with_the_scene_positions_and_checkpoint(self, qapp, tmp_path, stub_worker, positions_file):
        runner = _runner(stub_worker)
        runner.start(tmp_path / "scene", positions_file, checkpoint=tmp_path / "sam.pth")

        _ask(qapp, runner, 0, tmp_path / "o.npy")

        args = json.loads(Path(str(positions_file) + ".argv.json").read_text())
        assert "--serve" in args
        assert args[args.index("--scene-path") + 1] == str(tmp_path / "scene")
        assert args[args.index("--positions") + 1] == str(positions_file)
        assert args[args.index("--sam-checkpoint") + 1] == str(tmp_path / "sam.pth")
        runner.close()

    def test_the_request_reaches_the_worker_intact(self, qapp, tmp_path, stub_worker, positions_file):
        runner = _runner(stub_worker)
        runner.start(tmp_path, positions_file)
        output = tmp_path / "o.npy"

        _ask(qapp, runner, 2, output, point=(0.5, -1.25, 3.0))

        request = json.loads(Path(str(output) + ".request.json").read_text())
        assert request == {"point": [0.5, -1.25, 3.0], "level": 2, "output": str(output)}
        runner.close()

    def test_extra_and_negative_points_travel_with_the_request_only_when_given(self, qapp, tmp_path, stub_worker, positions_file):
        runner = _runner(stub_worker)
        runner.start(tmp_path, positions_file)
        plain, refined = tmp_path / "plain.npy", tmp_path / "refined.npy"

        _ask(qapp, runner, 1, plain)
        outcome = {}
        runner.finished.connect(lambda ok, payload: outcome.update(ok=ok))
        runner.request((0, 0, 0), 1, refined, extra_points=[(1, 2, 3), (4, 5, 6)], negative_points=[(7, 8, 9)])
        _wait_for(qapp, lambda: "ok" in outcome)

        assert set(json.loads(Path(str(plain) + ".request.json").read_text())) == {"point", "level", "output"}
        sent = json.loads(Path(str(refined) + ".request.json").read_text())
        assert sent["extra"] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] and sent["negatives"] == [[7.0, 8.0, 9.0]]
        runner.close()

    def test_the_vote_travels_with_the_request_only_when_given(self, qapp, tmp_path, stub_worker, positions_file):
        runner = _runner(stub_worker)
        runner.start(tmp_path, positions_file)
        plain, inclusive = tmp_path / "plain.npy", tmp_path / "inclusive.npy"

        _ask(qapp, runner, 1, plain)
        outcome = {}
        runner.finished.connect(lambda ok, payload: outcome.update(ok=ok))
        runner.request((0, 0, 0), 1, inclusive, min_ratio=0.3)
        _wait_for(qapp, lambda: "ok" in outcome)

        assert "min_ratio" not in json.loads(Path(str(plain) + ".request.json").read_text())
        assert json.loads(Path(str(inclusive) + ".request.json").read_text())["min_ratio"] == 0.3
        runner.close()

    def test_a_request_can_be_asked_for_before_the_worker_has_finished_starting(self, qapp, tmp_path, stub_worker, positions_file):
        runner = _runner(stub_worker)
        runner.start(tmp_path, positions_file)  # no waiting for "Ready."

        outcome = _ask(qapp, runner, 0, tmp_path / "o.npy")

        assert outcome["success"] is True
        runner.close()

    def test_is_busy_only_while_a_request_is_pending(self, qapp, tmp_path, stub_worker, positions_file):
        runner = _runner(stub_worker)
        runner.start(tmp_path, positions_file)
        assert not runner.is_busy()

        runner.request((0, 0, 0), 0, tmp_path / "o.npy")
        assert runner.is_busy()

        _wait_for(qapp, lambda: not runner.is_busy())
        assert not runner.is_busy()
        runner.close()


class TestFailures:
    def test_an_error_answer_fails_that_request_and_leaves_the_worker_running(self, qapp, tmp_path, stub_worker, positions_file):
        runner = _runner(stub_worker)
        runner.start(tmp_path, positions_file)

        failed = _ask(qapp, runner, 7, tmp_path / "a.npy")
        after = _ask(qapp, runner, 0, tmp_path / "b.npy")

        assert failed["success"] is False and failed["payload"] == "boom"
        assert after["success"] is True and after["payload"]["selected"] == 1  # the same worker went on
        runner.close()

    def test_a_crash_fails_the_request_and_the_next_start_launches_a_new_worker(self, qapp, tmp_path, stub_worker, positions_file):
        runner = _runner(stub_worker)
        runner.start(tmp_path, positions_file)

        crashed = _ask(qapp, runner, 9, tmp_path / "a.npy")

        assert crashed["success"] is False and "code 4" in crashed["payload"]
        assert _wait_for(qapp, lambda: not runner.is_running())
        runner.start(tmp_path, positions_file)
        again = _ask(qapp, runner, 0, tmp_path / "b.npy")
        assert again["success"] is True and again["payload"]["selected"] == 1
        runner.close()

    def test_cancel_kills_the_worker_and_says_so(self, qapp, tmp_path, stub_worker, positions_file):
        runner = _runner(stub_worker)
        runner.start(tmp_path, positions_file)

        outcome = _ask(qapp, runner, 8, tmp_path / "o.npy", cancel_after_ms=1500)

        assert outcome["success"] is False and outcome["payload"] == "Cancelled."
        assert _wait_for(qapp, lambda: not runner.is_running())

    def test_a_missing_python_fails_the_request_instead_of_hanging(self, qapp, tmp_path, stub_worker, positions_file):
        runner = SamPickRunner(python_exe=tmp_path / "no_such_python.exe", script_path=stub_worker)
        runner.start(tmp_path, positions_file)

        outcome = _ask(qapp, runner, 0, tmp_path / "o.npy")

        assert outcome["success"] is False and "Could not start" in outcome["payload"]


class TestShutdown:
    def test_close_ends_an_idle_worker_by_closing_its_input(self, qapp, tmp_path, stub_worker, positions_file):
        runner = _runner(stub_worker)
        runner.start(tmp_path, positions_file)
        _ask(qapp, runner, 0, tmp_path / "o.npy")

        runner.close()

        assert _wait_for(qapp, lambda: not runner.is_running(), timeout_ms=5000)

    def test_close_can_wait_until_the_worker_is_gone(self, qapp, tmp_path, stub_worker, positions_file):
        runner = _runner(stub_worker)
        runner.start(tmp_path, positions_file)
        _ask(qapp, runner, 0, tmp_path / "o.npy")

        runner.close(wait_ms=5000)

        assert not runner.is_running()  # no event loop was needed

    def test_close_during_a_request_kills_the_worker_at_once(self, qapp, tmp_path, stub_worker, positions_file):
        runner = _runner(stub_worker)
        runner.start(tmp_path, positions_file)
        runner.request((0, 0, 0), 8, tmp_path / "o.npy")  # hangs for a minute
        _wait_for(qapp, lambda: False, timeout_ms=800)  # let it start

        runner.close()

        assert _wait_for(qapp, lambda: not runner.is_running(), timeout_ms=5000)

    def test_an_idle_worker_shuts_itself_down(self, qapp, tmp_path, stub_worker, positions_file):
        runner = _runner(stub_worker, idle_ms=400)
        runner.start(tmp_path, positions_file)
        _ask(qapp, runner, 0, tmp_path / "o.npy")
        assert runner.is_running()

        assert _wait_for(qapp, lambda: not runner.is_running(), timeout_ms=6000)

    def test_a_worker_that_is_asked_again_does_not_shut_down_under_the_request(self, qapp, tmp_path, stub_worker, positions_file):
        runner = _runner(stub_worker, idle_ms=1200)
        runner.start(tmp_path, positions_file)
        _ask(qapp, runner, 0, tmp_path / "a.npy")
        _wait_for(qapp, lambda: False, timeout_ms=800)  # inside the idle window

        second = _ask(qapp, runner, 0, tmp_path / "b.npy")

        assert second["success"] is True and second["payload"]["selected"] == 2
        runner.close()


def test_checkpoint_availability_follows_the_file(tmp_path):
    checkpoint = tmp_path / "sam.pth"

    assert not sam_checkpoint_available(checkpoint)
    checkpoint.write_bytes(b"x")
    assert sam_checkpoint_available(checkpoint)
