import re
from pathlib import Path

from PyQt5 import QtCore

from .digital_twin_builder import GS_TRAIN_PYTHON, REPO_ROOT, _make_process_environment

CHECK_SCRIPT = REPO_ROOT / "scripts" / "check_twin_masks.py"
_RESULT_PREFIX = "TWINCHECK "


def scene_has_masks(scene_source) -> bool:
    masks = Path(scene_source) / "masks"
    return masks.is_dir() and any(p.suffix.lower() == ".png" for p in masks.iterdir())


class TwinMaskChecker(QtCore.QObject):

    finished = QtCore.pyqtSignal(bool, object)

    def __init__(self, parent=None, python_exe=None, script_path=None):
        super().__init__(parent)
        self._python = str(python_exe or GS_TRAIN_PYTHON)
        self._script = str(script_path or CHECK_SCRIPT)
        self._process: QtCore.QProcess | None = None
        self._buffer = ""
        self._recent_lines: list[str] = []
        self._result: dict | None = None
        self._cancelled = False
        self._reported = False

    def start(self, twin_path, scene_source) -> None:
        process = QtCore.QProcess(self)
        process.setProgram(self._python)
        process.setArguments(["-u", self._script, "--twin", str(twin_path), "--scene-path", str(scene_source)])
        process.setProcessEnvironment(_make_process_environment())
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.readyRead.connect(self._on_ready_read)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        self._process = process
        process.start()

    def cancel(self) -> None:
        self._cancelled = True
        process = self._process
        if process is not None and process.state() != QtCore.QProcess.NotRunning:
            process.kill()
            process.waitForFinished(2000)

    def _report(self, success: bool, payload) -> None:
        if not self._reported:
            self._reported = True
            self.finished.emit(success, payload)

    def _on_ready_read(self) -> None:
        self._buffer += bytes(self._process.readAll()).decode("utf-8", errors="replace")
        *complete, self._buffer = re.split(r"[\r\n]+", self._buffer)
        for line in complete:
            if line:
                self._handle_line(line)

    def _handle_line(self, line: str) -> None:
        self._recent_lines = (self._recent_lines + [line])[-30:]
        if line.startswith(_RESULT_PREFIX):
            parts = line[len(_RESULT_PREFIX):].split()
            if len(parts) >= 3:
                try:
                    self._result = {"share": float(parts[0]), "considered": int(parts[1]), "views": int(parts[2])}
                except ValueError:
                    pass

    def _on_error(self, error) -> None:
        if error == QtCore.QProcess.FailedToStart:
            self._report(False, f"Could not start the mask check:\n{self._python}")

    def _on_finished(self, exit_code: int, _exit_status) -> None:
        if self._buffer:
            self._handle_line(self._buffer)
            self._buffer = ""
        if self._cancelled:
            self._report(False, "Cancelled.")
        elif exit_code != 0:
            self._report(False, f"The mask check exited with code {exit_code}.\n\n" + "\n".join(self._recent_lines[-10:]))
        elif self._result is None:
            self._report(False, "The mask check finished but reported no result.")
        else:
            self._report(True, dict(self._result))
