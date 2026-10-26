import re
import shutil
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

from .digital_twin_builder import GS_TRAIN_PYTHON, NORMALIZED_SOURCE_ROOT, REPO_ROOT, _make_process_environment

RENDER_SCRIPT = REPO_ROOT / "scripts" / "render_preview.py"

SCENE_SOURCE_LAYOUTS = [
    (NORMALIZED_SOURCE_ROOT, "{scene}"),
    (REPO_ROOT / "data" / "raw", "mipnerf360_{scene}"),
]

_TWIN_SUFFIX = "_digital_twin"
_RENDERED_PREFIX = "RENDERED "


def find_scene_source_for_twin(npz_path) -> Path | None:
    stem = Path(npz_path).stem
    if not stem.endswith(_TWIN_SUFFIX):
        return None
    scene = stem[: -len(_TWIN_SUFFIX)]
    for parent, pattern in SCENE_SOURCE_LAYOUTS:
        candidate = Path(parent) / pattern.format(scene=scene)
        if (candidate / "sparse" / "0").is_dir() and (candidate / "images").is_dir():
            return candidate
    return None


class PreviewRenderer(QtCore.QObject):

    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(bool, object)

    def __init__(self, parent=None, python_exe=None, script_path=None):
        super().__init__(parent)
        self._python = str(python_exe or GS_TRAIN_PYTHON)
        self._script = str(script_path or RENDER_SCRIPT)
        self._process: QtCore.QProcess | None = None
        self._buffer = ""
        self._recent_lines: list[str] = []
        self._results: list[tuple[str, int, str]] = []
        self._cancelled = False
        self._reported = False

    def start(self, scene_source, packages: dict, out_dir, num_views: int = 4, max_side: int = 800) -> None:
        args = [
            "-u", self._script,
            "--scene-path", str(scene_source),
            "--out-dir", str(out_dir),
            "--num-views", str(num_views),
            "--max-side", str(max_side),
        ]
        for name, path in packages.items():
            args += ["--package", f"{name}={path}"]

        process = QtCore.QProcess(self)
        process.setProgram(self._python)
        process.setArguments(args)
        process.setProcessEnvironment(_make_process_environment())
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.readyRead.connect(self._on_ready_read)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        self._process = process
        process.start()

    def cancel(self) -> None:
        self._cancelled = True
        if self._process is not None and self._process.state() != QtCore.QProcess.NotRunning:
            self._process.kill()

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
        self.log_line.emit(line)
        self._recent_lines = (self._recent_lines + [line])[-30:]
        if line.startswith(_RENDERED_PREFIX):
            parts = line[len(_RENDERED_PREFIX):].split(" ", 2)
            if len(parts) == 3 and parts[1].isdigit():
                self._results.append((parts[0], int(parts[1]), parts[2]))

    def _on_error(self, error) -> None:
        if error == QtCore.QProcess.FailedToStart:
            self._report(False, f"Could not start the renderer:\n{self._python}")

    def _on_finished(self, exit_code: int, _exit_status) -> None:
        if self._buffer:
            self._handle_line(self._buffer)
            self._buffer = ""
        if self._cancelled:
            self._report(False, "Cancelled.")
        elif exit_code != 0:
            tail = "\n".join(self._recent_lines[-15:])
            self._report(False, f"Preview render exited with code {exit_code}.\n\n{tail}")
        elif not self._results:
            self._report(False, "The renderer finished but produced no images.")
        else:
            self._report(True, list(self._results))


class PreviewDialog(QtWidgets.QDialog):

    _IMAGE_SIZE = QtCore.QSize(520, 390)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preview render")
        self.resize(1140, 620)
        self._results: dict[int, dict[str, str]] = {}
        self._names: list[str] = []

        self.status_label = QtWidgets.QLabel("Rendering with the 3DGS rasterizer...")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)

        self._image_row = QtWidgets.QHBoxLayout()
        self._captions: list[QtWidgets.QLabel] = []
        self._images: list[QtWidgets.QLabel] = []

        self.view_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.view_slider.setVisible(False)
        self.view_slider.valueChanged.connect(self._show_view)
        self.view_label = QtWidgets.QLabel("")

        self.save_button = QtWidgets.QPushButton("Save images...")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save_images)
        self.close_button = QtWidgets.QPushButton("Cancel")
        self.close_button.clicked.connect(self.close)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.view_label)
        controls.addWidget(self.view_slider, 1)
        controls.addWidget(self.save_button)
        controls.addWidget(self.close_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addLayout(self._image_row, 1)
        layout.addLayout(controls)

    def set_failed(self, message: str) -> None:
        self.progress_bar.setVisible(False)
        self.status_label.setText(message)
        self.close_button.setText("Close")

    def show_results(self, results: list) -> None:
        self._results = {}
        self._names = []
        for name, view, path in results:
            self._results.setdefault(view, {})[name] = path
            if name not in self._names:
                self._names.append(name)

        for name in self._names:
            column = QtWidgets.QVBoxLayout()
            caption = QtWidgets.QLabel(name)
            caption.setAlignment(QtCore.Qt.AlignCenter)
            image = QtWidgets.QLabel()
            image.setAlignment(QtCore.Qt.AlignCenter)
            image.setMinimumSize(self._IMAGE_SIZE)
            column.addWidget(caption)
            column.addWidget(image, 1)
            self._image_row.addLayout(column)
            self._captions.append(caption)
            self._images.append(image)

        views = sorted(self._results)
        self.view_slider.setRange(0, len(views) - 1)
        self.view_slider.setVisible(len(views) > 1)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Same cameras for every image; colour is the base spherical-harmonics term only.")
        self.save_button.setEnabled(True)
        self.close_button.setText("Close")
        self._show_view(0)

    def _show_view(self, slider_position: int) -> None:
        views = sorted(self._results)
        if not views:
            return
        view = views[slider_position]
        self.view_label.setText(f"Camera {view}  ({slider_position + 1} of {len(views)})")
        for name, label in zip(self._names, self._images):
            path = self._results[view].get(name)
            pixmap = QtGui.QPixmap(path) if path else QtGui.QPixmap()
            label.setPixmap(
                pixmap.scaled(self._IMAGE_SIZE, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            )

    def _save_images(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Save preview images to...")
        if not folder:
            return
        for by_name in self._results.values():
            for path in by_name.values():
                shutil.copy2(path, Path(folder) / Path(path).name)
        self.status_label.setText(f"Saved images to {folder}")
