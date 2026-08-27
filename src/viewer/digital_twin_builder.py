"""Auto-build a digital-twin package the first time a scene is edited.

Producing a digital-twin `.npz` (see digital_twin.py) normally means running
the full SuGaR pipeline by hand: baseline 3DGS -> SuGaR coarse training ->
mesh extraction -> SuGaR refinement -> scripts/export_digital_twin_package.py
(that's how the bonsai package was made, over several hours at SuGaR's
"long"/high_poly/30k-baseline settings). This module drives the same
pipeline automatically, at a lighter preset, from inside the app -- via
`train_full_pipeline.py`'s own iteration/mesh-size arguments (some added by
us specifically for this, see environment/README.md):

- Baseline 3DGS: `--vanilla_gs_iterations` (`BASELINE_ITERATIONS`), plus
  `--vanilla_gs_resolution 2` to halve the (already auto-capped-to-1.6K)
  training resolution.
- SuGaR coarse stage: `--coarse_iterations` (`COARSE_ITERATIONS`). This
  stage's own iteration *counter* starts at `BASELINE_ITERATIONS - 1`
  (SuGaR's own continuity convention -- patched to track the actual
  baseline length instead of an assumed 7k, see environment/README.md), so
  `COARSE_ITERATIONS` must stay comfortably above `BASELINE_ITERATIONS`
  for any real coarse training to happen at all -- it's the *ending*
  iteration count, not a step budget.
- Mesh extraction: `-v` (`REFINED_MESH_VERTICES`) target vertex count.
- SuGaR refinement: `-f` (`REFINE_ITERATIONS`), passed directly rather than
  via a --refinement_time preset.
- Optional textured-mesh/ply exports skipped (`--export_obj False
  --export_ply False`) -- the digital-twin `.npz` only needs the refined
  `.pt` checkpoint, always saved regardless.
- `SUGAR_GPU_MEMORY_FRACTION` caps this process's share of GPU VRAM (see
  `GPU_MEMORY_FRACTION` below) -- without it, training can creep toward the
  GPU's full capacity over a long run.

**Tuning history** (real, timed/observed against lego_bugatti/lego_ferrari
-- see experiments/02_mesh_based_digital_twin_editing/README.md for the
full account): an aggressive first pass (~1.5k baseline, ~1.5k real coarse
steps, 50k-vertex mesh, 300 refine iterations) hit ~9.5-13 min but produced
a visibly fragmented, disconnected mesh -- coarse's density/normal
regularization hadn't run long enough to align Gaussians into a coherent
surface before extraction. The opposite extreme (7k baseline, ~8k real
coarse steps, 200k-vertex mesh, 15k refine iterations, no resolution
downscale) ran for almost 2 hours without finishing and pushed GPU memory
to 15.9/16.3 GB -- real OOM risk, and still slower than practical for an
interactive "click and wait" feature. The current constants below are a
middle ground between those two data points, not yet re-validated at the
same level of scrutiny as the first (too-aggressive) preset was -- see
Status in the experiments doc for whatever was last actually measured.

This is communicated to the user as a real wait with a progress dialog, not
implied to be instant. Needs the heavier `gs-train` conda env (torch/CUDA/
SuGaR); this module only *drives* it as a subprocess (via QProcess, so it
doesn't block the Qt event loop) -- it does not import torch itself.
"""

import glob
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PyQt5 import QtCore

REPO_ROOT = Path(__file__).resolve().parents[2]
SUGAR_ROOT = REPO_ROOT / "external" / "SuGaR"
GS_TRAIN_ENV_DIR = Path(r"D:\conda-envs\gs-train")
GS_TRAIN_PYTHON = GS_TRAIN_ENV_DIR / "python.exe"

NORMALIZED_SOURCE_ROOT = REPO_ROOT / "data" / "gaussian_scenes" / "_colmap_src"

# Balanced preset -- see "Tuning history" in the module docstring.
# COARSE_ITERATIONS must stay well above BASELINE_ITERATIONS -- it's an
# ending iteration count, not a step budget (see docstring).
BASELINE_ITERATIONS = 3_000
COARSE_ITERATIONS = 7_000
REFINED_MESH_VERTICES = 100_000
REFINE_ITERATIONS = 2_000

# Cap this process's share of GPU VRAM (fraction of total, 0-1) so a long
# run degrades to a clean, immediate CUDA OOM error instead of silently
# creeping toward the GPU's full capacity -- see SUGAR_GPU_MEMORY_FRACTION
# in the two patched entry points (gaussian_splatting/train.py, train.py).
GPU_MEMORY_FRACTION = 0.6

# Stage markers: distinctive substrings SuGaR/gaussian-splatting already
# print at the start of each stage (checked directly against the vendored
# source, see the accompanying plan) -- used to label the progress dialog
# without needing to patch those scripts to report structured progress.
_STAGE_MARKERS = [
    ("Optimizing ", "Training baseline 3DGS (7k iterations)"),
    ("Depth-Normal consistency factor:", "Training coarse SuGaR"),
    ("Loading the coarse SuGaR model from path", "Extracting mesh from coarse model"),
    ("Surface mesh to bind to:", "Refining SuGaR (mesh-bound training)"),
    ("Exporting ply file with refined Gaussians", "Exporting refined Gaussians"),
]

# Progress-within-a-stage patterns, tried in order against each output line:
# tqdm's default bar rendering, SuGaR's own "loss: ... [iter/total]" lines,
# and a bare "N/M" fallback (e.g. "Processing frame 5/130...").
_PROGRESS_PATTERNS = [
    re.compile(r"\d+%\|.*?\|\s*(\d+)/(\d+)"),
    re.compile(r"\[\s*(\d+)/\s*(\d+)\]"),
    re.compile(r"(\d+)\s*/\s*(\d+)"),
]


@dataclass
class ColmapSource:
    scene_name: str
    frame_dir: Path  # contains images/, optionally masks/, and sparse/


def find_colmap_source(ply_path) -> ColmapSource | None:
    """Recognize the `data/Test data/<scene>/plys/*.ply` convention and
    return its sibling COLMAP capture, or None if `ply_path` doesn't match
    (e.g. a plain .ply with no associated multi-view source -- SuGaR
    fundamentally needs images + camera poses, not just a point cloud)."""
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


def digital_twin_cache_path(scene_name: str) -> Path:
    """Same naming convention already used for bonsai_digital_twin.npz."""
    return REPO_ROOT / "data" / "gaussian_scenes" / f"{scene_name}_digital_twin.npz"


def _write_masked_images(colmap: ColmapSource, dest_images_dir: Path) -> None:
    """Multiply each training photo by its background mask (masks/<name>.png
    -- white=subject, black=background) instead of just junctioning the raw
    images through.

    Found this to matter a lot, not just be a nice-to-have: these are
    glossy toy-car turntable captures, and training on the *unmasked*
    photos (background, turntable, reflections and all) produced visibly
    fragmented, disconnected meshes even after generously increasing
    coarse SuGaR's iteration budget -- the background was competing for
    the density regularization's attention alongside the actual object.
    Masking out the background before training removes that competing
    signal entirely, which iteration count alone can't fix.
    """
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
    """Build (or reuse) a per-scene COLMAP source directory SuGaR/gaussian-
    splatting can consume directly.

    Two real gotchas in data/Test data/, discovered by inspecting the
    actual files: (1) every scene's capture sits under the same generic
    leaf folder name (`colmap_data/frame_0000/`), which would collide in
    SuGaR's own `output/<scene_name>/` directories if passed straight
    through, since SuGaR names those from the last path component; (2) the
    COLMAP reconstruction sits directly under `.../sparse/`, not the
    `sparse/0/` nesting gaussian-splatting's reader expects. A third,
    discovered later: the raw `images/` need to be background-masked
    first (see `_write_masked_images`) for a coherent mesh at all.
    """
    dest = NORMALIZED_SOURCE_ROOT / colmap.scene_name
    sparse0 = dest / "sparse" / "0"
    images_dir = dest / "images"
    masks_src = colmap.frame_dir / "masks"
    # A real (masked) images/ folder, not a directory junction, marks this
    # scene as already prepared with masking applied -- a junction there
    # means an older, pre-masking prepared copy, which needs replacing, not
    # reusing. Path.is_symlink() does NOT detect Windows junctions (verified
    # empirically -- it's False for them); comparing realpath is reliable.
    images_dir_is_junction = images_dir.is_dir() and os.path.realpath(images_dir) != str(images_dir)
    already_prepared = sparse0.is_dir() and any(sparse0.iterdir()) and images_dir.is_dir() and not images_dir_is_junction
    if already_prepared:
        return dest

    dest.mkdir(parents=True, exist_ok=True)

    if images_dir_is_junction:
        images_dir.rmdir()  # drop a stale junction from an older, unmasked prepared copy (target untouched)
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
    # Every stage this drives is itself a chain of `os.system("python ...")`
    # calls inside train_full_pipeline.py; PYTHONUNBUFFERED (rather than
    # `-u` on just the top-level process) makes sure *all* of them flush
    # progress promptly instead of block-buffering into our pipe.
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
    """Drives the fast-preset pipeline for one scene as a background
    QProcess, so the Qt event loop (and the rest of the app) stays
    responsive. Reuses train_full_pipeline.py and
    scripts/export_digital_twin_package.py exactly as they already exist --
    no reimplementation of the pipeline itself, just orchestration."""

    stage_changed = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int, int)  # (current, total); (-1, -1) = indeterminate
    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(bool, str)  # (success, npz_path_or_error_message)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process: QtCore.QProcess | None = None
        self._scene_name: str | None = None
        self._buffer = ""
        self._recent_lines: list[str] = []
        self._stage = "starting"

    def start(self, ply_path) -> None:
        colmap = find_colmap_source(ply_path)
        if colmap is None:
            self.finished.emit(False, f"No COLMAP source found next to {ply_path}")
            return
        self._scene_name = colmap.scene_name
        self._recent_lines = []

        try:
            normalized_source = prepare_normalized_source(colmap)
        except Exception as exc:  # noqa: BLE001 -- surface any prep failure to the UI
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
            "-v", str(REFINED_MESH_VERTICES),
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
        pass  # surfaced via the exit code in _on_training_finished

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

        process = QtCore.QProcess(self)
        process.setProgram(str(GS_TRAIN_PYTHON))
        process.setArguments([
            "-u", str(REPO_ROOT / "scripts" / "export_digital_twin_package.py"),
            "--scene-path", str(NORMALIZED_SOURCE_ROOT / self._scene_name),
            "--gs-output-dir", str(gs_output_dir) + "\\",
            "--refined-checkpoint", str(checkpoint),
            "--iteration-to-load", str(BASELINE_ITERATIONS),
            "--out", str(out_path),
        ])
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

        if exit_code != 0 or not out_path.exists():
            tail = "\n".join(self._recent_lines[-15:])
            self.finished.emit(False, f"Export step exited with code {exit_code}.\n\n{tail}")
            return

        self.finished.emit(True, str(out_path))
