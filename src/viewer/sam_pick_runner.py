import json
import re
from pathlib import Path

from PyQt5 import QtCore

from .digital_twin_builder import GS_TRAIN_PYTHON, REPO_ROOT, _make_process_environment

SAM_SCRIPT = REPO_ROOT / "scripts" / "sam_point_segment.py"
SAM_CHECKPOINT = REPO_ROOT / "data" / "checkpoints" / "sam_vit_b_01ec64.pth"

IDLE_SHUTDOWN_MS = 10 * 60 * 1000
_GRACEFUL_EXIT_MS = 3000

_RESULT_PREFIX = "SAMPICK "
_ERROR_PREFIX = "SAMPICK_ERROR "


def sam_checkpoint_available(path=None) -> bool:
    return Path(path or SAM_CHECKPOINT).is_file()


class SamPickRunner(QtCore.QObject):

    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(bool, object)

    def __init__(self, parent=None, python_exe=None, script_path=None, idle_ms: int = IDLE_SHUTDOWN_MS):
        super().__init__(parent)
        self._python = str(python_exe or GS_TRAIN_PYTHON)
        self._script = str(script_path or SAM_SCRIPT)
        self._process: QtCore.QProcess | None = None
        self._buffer = ""
        self._recent_lines: list[str] = []
        self._busy = False
        self._cancelled = False
        self._idle_ms = idle_ms
        self._idle_timer = QtCore.QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.timeout.connect(self.close)

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() != QtCore.QProcess.NotRunning

    def is_busy(self) -> bool:
        return self._busy

    def start(self, scene_source, positions_path, checkpoint=None) -> None:
        if self.is_running():
            return
        self._buffer, self._recent_lines, self._cancelled = "", [], False
        process = QtCore.QProcess(self)
        process.setProgram(self._python)
        process.setArguments([
            "-u", self._script, "--serve",
            "--scene-path", str(scene_source),
            "--positions", str(positions_path),
            "--sam-checkpoint", str(checkpoint or SAM_CHECKPOINT),
        ])
        process.setProcessEnvironment(_make_process_environment())
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.readyRead.connect(self._on_ready_read)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        self._process = process
        process.start()

    def request(self, point, level: int, output_path, extra_points=(), negative_points=(), min_ratio=None) -> None:
        self._busy = True
        self._idle_timer.stop()
        if not self.is_running():
            # The launch failed (or the worker is gone); say so once the caller has returned.
            QtCore.QTimer.singleShot(0, lambda: self._finish(False, f"Could not start SAM:\n{self._python}"))
            return
        message = {"point": [float(v) for v in point], "level": int(level), "output": str(output_path)}
        if len(extra_points):
            message["extra"] = [[float(v) for v in p] for p in extra_points]
        if len(negative_points):
            message["negatives"] = [[float(v) for v in p] for p in negative_points]
        if min_ratio is not None:
            message["min_ratio"] = float(min_ratio)
        self._process.write((json.dumps(message) + "\n").encode("utf-8"))

    def close(self, wait_ms: int = 0) -> None:
        self._idle_timer.stop()
        if not self.is_running():
            return
        if self._busy:
            self.cancel()
        else:
            self._process.closeWriteChannel()
            QtCore.QTimer.singleShot(_GRACEFUL_EXIT_MS, self._kill_if_running)
        if wait_ms and not self._process.waitForFinished(wait_ms):
            self._kill_if_running()
            self._process.waitForFinished(1000)

    def cancel(self) -> None:
        self._cancelled = True
        self._kill_if_running()

    def _kill_if_running(self) -> None:
        if self.is_running():
            self._process.kill()

    def _finish(self, success: bool, payload) -> None:
        if not self._busy:
            return
        self._busy = False
        if self._idle_ms > 0 and self.is_running():
            self._idle_timer.start(self._idle_ms)
        self.finished.emit(success, payload)

    def _on_ready_read(self) -> None:
        self._buffer += bytes(self._process.readAll()).decode("utf-8", errors="replace")
        *complete, self._buffer = re.split(r"[\r\n]+", self._buffer)
        for line in complete:
            if line:
                self._handle_line(line)

    def _handle_line(self, line: str) -> None:
        self.log_line.emit(line)
        self._recent_lines = (self._recent_lines + [line])[-30:]
        if line.startswith(_ERROR_PREFIX):
            self._finish(False, line[len(_ERROR_PREFIX):])
        elif line.startswith(_RESULT_PREFIX):
            parts = line[len(_RESULT_PREFIX):].split(" ", 3)
            if len(parts) == 4 and all(p.isdigit() for p in parts[:3]):
                self._finish(True, {"selected": int(parts[0]), "views": int(parts[1]), "rounds": int(parts[2]), "path": parts[3]})

    def _on_error(self, error) -> None:
        if error == QtCore.QProcess.FailedToStart:
            self._finish(False, f"Could not start SAM:\n{self._python}")

    def _on_finished(self, exit_code: int, _exit_status) -> None:
        self._idle_timer.stop()
        if self._buffer:
            self._handle_line(self._buffer)
            self._buffer = ""
        if self._cancelled:
            self._finish(False, "Cancelled.")
        elif self._busy:
            tail = "\n".join(self._recent_lines[-15:])
            self._finish(False, f"SAM Pick worker exited with code {exit_code}.\n\n{tail}")
