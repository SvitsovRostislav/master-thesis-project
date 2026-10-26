import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PyQt5 import QtCore

from .twin_provenance import build_record

REPO_ROOT = Path(__file__).resolve().parents[2]
SUGAR_ROOT = REPO_ROOT / "external" / "SuGaR"
GS_TRAIN_ENV_DIR = Path(r"D:\conda-envs\gs-train")
GS_TRAIN_PYTHON = GS_TRAIN_ENV_DIR / "python.exe"

NORMALIZED_SOURCE_ROOT = REPO_ROOT / "data" / "gaussian_scenes" / "_colmap_src"
BASELINE_ITERATIONS = 3_000
COARSE_ITERATIONS = 7_000
REFINED_MESH_TRIANGLES = 100_000
REFINE_ITERATIONS = 2_000
GPU_MEMORY_FRACTION = 0.6

_STAGE_MARKERS = [
    ("Optimizing ", f"Training baseline 3DGS ({BASELINE_ITERATIONS // 1000}k iterations)"),
    ("Depth-Normal consistency factor:", "Training coarse SuGaR"),
    ("Loading the coarse SuGaR model from path", "Extracting mesh from coarse model"),
    ("Surface mesh to bind to:", "Refining SuGaR (mesh-bound training)"),
    ("Exporting ply file with refined Gaussians", "Exporting refined Gaussians"),
]

_PROGRESS_PATTERNS = [
    re.compile(r"\d+%\|.*?\|\s*(\d+)/(\d+)"),
    re.compile(r"\[\s*(\d+)/\s*(\d+)\]"),
    re.compile(r"(\d+)\s*/\s*(\d+)"),
]


@dataclass
class ColmapSource:
    scene_name: str
    frame_dir: Path


def find_colmap_source(ply_path) -> ColmapSource | None:
    plys_dir = Path(ply_path).resolve().parent
    if plys_dir.name != "plys":
        return None
    scene_dir = plys_dir.parent
    colmap_data_dir = scene_dir / "colmap_data"
    if not colmap_data_dir.is_dir():
        return None
    frame_dirs = sorted(
        d for d in colmap_data_dir.iterdir()
        if d.is_dir() and (d / "sparse").is_dir() and (d / "images").is_dir()
    )
    if not frame_dirs:
        return None
    return ColmapSource(scene_name=scene_dir.name, frame_dir=frame_dirs[0])


def build_preset() -> dict:
    return {
        "baseline_iterations": BASELINE_ITERATIONS,
        "coarse_iterations": COARSE_ITERATIONS,
        "refine_iterations": REFINE_ITERATIONS,
        "mesh_triangles_target": REFINED_MESH_TRIANGLES,
        "gaussians_per_triangle": 1,
        "gpu_memory_fraction": GPU_MEMORY_FRACTION,
    }


def write_provenance_file(scene_name: str, normalized_source) -> Path | None:
    try:
        record = build_record(scene_name, normalized_source, build_preset())
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", prefix="twin_provenance_", delete=False, encoding="utf-8")
        with handle:
            json.dump(record, handle)
        return Path(handle.name)
    except (OSError, ValueError):
        return None


def digital_twin_cache_path(scene_name: str) -> Path:
    return REPO_ROOT / "data" / "gaussian_scenes" / f"{scene_name}_digital_twin.npz"


def _write_masked_images(colmap: ColmapSource, dest_images_dir: Path) -> None:
    from PIL import Image
    import numpy as np

    masks_dir = colmap.frame_dir / "masks"
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted((colmap.frame_dir / "images").iterdir()):
        out_path = dest_images_dir / image_path.name
        if out_path.exists():
            continue
        mask_path = masks_dir / (image_path.stem + ".png")
        if not mask_path.is_file():
            shutil.copy2(image_path, out_path)
            continue
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L").resize(image.size)
        masked = np.array(image) * (np.array(mask) > 127)[:, :, None]
        Image.fromarray(masked.astype(np.uint8)).save(out_path, quality=95)


def prepare_normalized_source(colmap: ColmapSource) -> Path:
    dest = NORMALIZED_SOURCE_ROOT / colmap.scene_name
    sparse0 = dest / "sparse" / "0"
    images_dir = dest / "images"
    masks_src = colmap.frame_dir / "masks"
    images_dir_is_junction = images_dir.is_dir() and os.path.realpath(images_dir) != str(images_dir)
    already_prepared = sparse0.is_dir() and any(sparse0.iterdir()) and images_dir.is_dir() and not images_dir_is_junction
    if already_prepared:
        return dest

    dest.mkdir(parents=True, exist_ok=True)

    if images_dir_is_junction:
        images_dir.rmdir()
    if masks_src.is_dir():
        _write_masked_images(colmap, images_dir)
    elif not images_dir.exists():
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(images_dir), str(colmap.frame_dir / "images")],
            check=True, capture_output=True, text=True,
        )

    masks_dest = dest / "masks"
    if masks_src.is_dir() and not masks_dest.exists():
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(masks_dest), str(masks_src)],
            check=True, capture_output=True, text=True,
        )

    sparse0.mkdir(parents=True, exist_ok=True)
    for bin_file in (colmap.frame_dir / "sparse").glob("*.bin"):
        shutil.copy2(bin_file, sparse0 / bin_file.name)

    return dest


def _make_process_environment() -> QtCore.QProcessEnvironment:
    env = QtCore.QProcessEnvironment.systemEnvironment()
    env.insert("CUDA_HOME", str(GS_TRAIN_ENV_DIR / "Library"))
    env.insert("PYTHONUNBUFFERED", "1")
    env.insert("SUGAR_GPU_MEMORY_FRACTION", str(GPU_MEMORY_FRACTION))
    gs_train_bins = [
        str(GS_TRAIN_ENV_DIR),
        str(GS_TRAIN_ENV_DIR / "Scripts"),
        str(GS_TRAIN_ENV_DIR / "Library" / "bin"),
    ]
    existing_path = env.value("PATH")
    env.insert("PATH", ";".join(gs_train_bins) + ";" + existing_path)
    return env


class DigitalTwinBuilder(QtCore.QObject):
    stage_changed = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int, int)
    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process: QtCore.QProcess | None = None
        self._scene_name: str | None = None
        self._buffer = ""
        self._recent_lines: list[str] = []
        self._stage = "starting"
        self._provenance_file: Path | None = None

    def start(self, ply_path) -> None:
        colmap = find_colmap_source(ply_path)
        if colmap is None:
            self.finished.emit(False, f"No COLMAP source found next to {ply_path}")
            return
        self._scene_name = colmap.scene_name
        self._recent_lines = []

        try:
            normalized_source = prepare_normalized_source(colmap)
        except Exception as exc:
            self.finished.emit(False, f"Failed to prepare COLMAP source: {exc}")
            return

        self._stage = "starting"
        self.stage_changed.emit("Starting SuGaR pipeline...")
        self.progress.emit(-1, -1)

        process = QtCore.QProcess(self)
        process.setProgram(str(GS_TRAIN_PYTHON))
        process.setArguments([
            "-u", str(SUGAR_ROOT / "train_full_pipeline.py"),
            "-s", str(normalized_source),
            "-r", "dn_consistency",
            "--vanilla_gs_iterations", str(BASELINE_ITERATIONS),
            "--vanilla_gs_resolution", "2",
            "--coarse_iterations", str(COARSE_ITERATIONS),
            "-v", str(REFINED_MESH_TRIANGLES),
            "-f", str(REFINE_ITERATIONS),
            "--export_obj", "False",
            "--export_ply", "False",
            "--eval", "False",
            "--gpu", "0",
        ])
        process.setWorkingDirectory(str(SUGAR_ROOT))
        process.setProcessEnvironment(_make_process_environment())
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.readyRead.connect(self._on_ready_read)
        process.finished.connect(self._on_training_finished)
        process.errorOccurred.connect(self._on_process_error)
        self._process = process
        process.start()

    def cancel(self) -> None:
        if self._process is not None:
            self._process.kill()

    def _on_ready_read(self) -> None:
        chunk = bytes(self._process.readAll()).decode("utf-8", errors="replace")
        self._buffer += chunk
        *complete_lines, self._buffer = re.split(r"[\r\n]+", self._buffer)
        for line in complete_lines:
            if line:
                self._handle_line(line)

    def _handle_line(self, line: str) -> None:
        self.log_line.emit(line)
        self._recent_lines.append(line)
        self._recent_lines = self._recent_lines[-30:]

        for marker, label in _STAGE_MARKERS:
            if marker in line:
                self._stage = label
                self.stage_changed.emit(label)
                self.progress.emit(-1, -1)
                break

        for pattern in _PROGRESS_PATTERNS:
            match = pattern.search(line)
            if match:
                current, total = int(match.group(1)), int(match.group(2))
                if total > 0 and current <= total:
                    self.progress.emit(current, total)
                break

    def _on_process_error(self, _error) -> None:
        pass

    def _on_training_finished(self, exit_code: int, _exit_status) -> None:
        if self._buffer:
            self._handle_line(self._buffer)
            self._buffer = ""

        if exit_code != 0:
            tail = "\n".join(self._recent_lines[-15:])
            self.finished.emit(False, f"SuGaR pipeline exited with code {exit_code}.\n\n{tail}")
            return

        checkpoint = self._find_latest_refined_checkpoint()
        if checkpoint is None:
            self.finished.emit(False, "Pipeline finished but no refined checkpoint was found.")
            return

        self.stage_changed.emit("Exporting digital-twin package...")
        self.progress.emit(-1, -1)
        self._start_export(checkpoint)

    def _find_latest_refined_checkpoint(self) -> Path | None:
        pattern = str(SUGAR_ROOT / "output" / "refined" / self._scene_name / "*" / "*.pt")
        candidates = glob.glob(pattern)
        if not candidates:
            return None
        return Path(max(candidates, key=lambda p: Path(p).stat().st_mtime))

    def _start_export(self, checkpoint: Path) -> None:
        gs_output_dir = SUGAR_ROOT / "output" / "vanilla_gs" / self._scene_name
        out_path = digital_twin_cache_path(self._scene_name)
        self._provenance_file = write_provenance_file(self._scene_name, NORMALIZED_SOURCE_ROOT / self._scene_name)

        process = QtCore.QProcess(self)
        process.setProgram(str(GS_TRAIN_PYTHON))
        process.setArguments([
            "-u", str(REPO_ROOT / "scripts" / "export_digital_twin_package.py"),
            "--scene-path", str(NORMALIZED_SOURCE_ROOT / self._scene_name),
            "--gs-output-dir", str(gs_output_dir) + "\\",
            "--refined-checkpoint", str(checkpoint),
            "--iteration-to-load", str(BASELINE_ITERATIONS),
            "--out", str(out_path),
        ] + (["--provenance", str(self._provenance_file)] if self._provenance_file else []))
        process.setWorkingDirectory(str(SUGAR_ROOT))
        process.setProcessEnvironment(_make_process_environment())
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.readyRead.connect(self._on_ready_read)
        process.finished.connect(lambda code, status: self._on_export_finished(code, status, out_path))
        self._process = process
        process.start()

    def _on_export_finished(self, exit_code: int, _exit_status, out_path: Path) -> None:
        if self._buffer:
            self._handle_line(self._buffer)
            self._buffer = ""
        if self._provenance_file is not None:
            self._provenance_file.unlink(missing_ok=True)
            self._provenance_file = None

        if exit_code != 0 or not out_path.exists():
            tail = "\n".join(self._recent_lines[-15:])
            self.finished.emit(False, f"Export step exited with code {exit_code}.\n\n{tail}")
            return

        self.finished.emit(True, str(out_path))
