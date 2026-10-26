import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets
from vispy import app as vispy_app
from vispy import scene

from .digital_twin import DigitalTwinPackage, load_digital_twin_package, save_digital_twin_package
from .digital_twin_builder import GS_TRAIN_PYTHON, DigitalTwinBuilder, digital_twin_cache_path, find_colmap_source
from .gaussian_density import DensityBaseline, rebalance_density
from .gaussian_deformation import DeformationBaseline, adapt_gaussians_to_deformation
from .gaussian_ply_io import load_gaussian_ply, save_gaussian_ply
from .mesh_detach import detach_selection, inside_faces, stretches_on_move
from .mesh_fill import close_cut
from .mesh_editor import MeshEditorPanel
from .preview_render import PreviewDialog, PreviewRenderer, find_scene_source_for_twin
from .sam_pick import DEFAULT_LEVEL, INCLUSIVE, LEVEL_NAMES, MAJORITY, ReferenceGeometry
from .sam_pick_runner import SamPickRunner, sam_checkpoint_available
from .semantic_segmentation import label_at_vertex, labels_to_vertex_selection, load_semantic_labels
from .twin_mask_check import TwinMaskChecker, scene_has_masks
from .twin_provenance import MASK_SHARE_WARNING, assess, describe, load_cached_check, save_cached_check

REPO_ROOT = Path(__file__).resolve().parents[2]
_TEST_DATA_DIR = REPO_ROOT / "data" / "Test data"
DEFAULT_OPEN_DIR = _TEST_DATA_DIR if _TEST_DATA_DIR.is_dir() else REPO_ROOT / "data" / "raw" / "test_models"

SAM_MODES = ("Replace", "Add", "Subtract", "Exclude")
SAM_VOTES = ("Majority", "Auto", "Inclusive")
SYNCED_CAMERA_PROPS = ("azimuth", "elevation", "roll", "scale_factor")

MARKER_DIAMETER_IN_SCALES = 2.0
MIN_MARKER_PX = 1.0
MAX_MARKER_PX = 40.0

EXTENT_DISTANCE_PERCENTILE = 80.0

CONTROLS_HELP = (
    'Controls:  left-drag = rotate   scroll = zoom   right-drag = pan   '
    '"Open File..." button (or O key) = open file   Esc = quit'
)


def pick_ply_file(initial_dir: Path) -> str | None:
    if QtWidgets.QApplication.instance() is None:
        QtWidgets.QApplication(sys.argv)
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        "Select a Gaussian Splat / point cloud .ply file",
        str(initial_dir),
        "PLY files (*.ply);;All files (*)",
    )
    return path or None


def pick_digital_twin_file(initial_dir: Path) -> str | None:
    if QtWidgets.QApplication.instance() is None:
        QtWidgets.QApplication(sys.argv)
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        "Select a digital-twin package (.npz)",
        str(initial_dir),
        "Digital-twin package (*.npz);;All files (*)",
    )
    return path or None


def pick_save_ply_path(initial_dir: Path) -> str | None:
    if QtWidgets.QApplication.instance() is None:
        QtWidgets.QApplication(sys.argv)
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        None,
        "Save Gaussian Splat as .ply",
        str(initial_dir),
        "PLY files (*.ply);;All files (*)",
    )
    return path or None


class GaussianViewer:
    def __init__(self, on_open_requested=None, on_quit_requested=None):
        self.canvas = scene.SceneCanvas(
            keys="interactive", size=(1024, 720), bgcolor="#141414"
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(fov=45.0)
        self.markers = scene.visuals.Markers(parent=self.view.scene)
        self.on_open_requested = on_open_requested
        self.on_quit_requested = on_quit_requested
        self._face_colors = None
        self._sizes = None
        self.canvas.events.key_press.connect(self._on_key_press)

    def load(self, ply_path: str) -> int:
        cloud = load_gaussian_ply(ply_path)
        count = self._display(cloud.positions, cloud.colors, cloud.opacities, cloud.scales)
        print(f"Loaded {count:,} splats from {ply_path}")
        return count

    def load_from_package(self, package: DigitalTwinPackage, keep_view: bool = False) -> int:
        """Display an in-memory Gaussian cloud from a digital-twin package. With `keep_view` the
        camera stays where it is instead of being fitted to the cloud."""
        positions = package.recompute_gaussian_positions()
        count = self._display(
            positions, package.gaussian_colors, package.gaussian_opacities, package.gaussian_scales, keep_view
        )
        print(f"Loaded {count:,} splats from digital-twin package")
        return count

    def update_positions(self, positions: np.ndarray) -> None:
        self.markers.set_data(
            pos=positions, face_color=self._face_colors, size=self._sizes, edge_width=0
        )

    def _display(
        self,
        positions: np.ndarray,
        colors: np.ndarray,
        opacities: np.ndarray,
        scales: np.ndarray,
        keep_view: bool = False,
    ) -> int:
        face_colors = np.concatenate([colors, opacities[:, None]], axis=1)  # RGBA

        self.markers.set_data(
            pos=positions,
            face_color=face_colors,
            size=1.0,
            edge_width=0,
        )

        center, radius = self._robust_center_and_radius(positions)
        lo, hi = center - radius, center + radius
        if not keep_view:
            self.view.camera.set_range(x=(lo[0], hi[0]), y=(lo[1], hi[1]), z=(lo[2], hi[2]))

        sizes = self._marker_sizes(radius, scales)
        self.markers.set_data(pos=positions, face_color=face_colors, size=sizes, edge_width=0)

        self._face_colors = face_colors
        self._sizes = sizes
        return positions.shape[0]

    @staticmethod
    def _robust_center_and_radius(positions: np.ndarray) -> tuple[np.ndarray, float]:
        center = np.median(positions, axis=0)
        dist_from_center = np.linalg.norm(positions - center, axis=1)
        radius = float(np.percentile(dist_from_center, EXTENT_DISTANCE_PERCENTILE))
        return center, max(radius, 1e-6)

    def _marker_sizes(self, radius: float, scales: np.ndarray) -> np.ndarray:
        canvas_height_px = self.canvas.size[1]
        px_per_world_unit = canvas_height_px / max(2.0 * radius, 1e-6)

        mean_scale = scales.mean(axis=1)
        pixel_size = MARKER_DIAMETER_IN_SCALES * mean_scale * px_per_world_unit
        return np.clip(pixel_size, MIN_MARKER_PX, MAX_MARKER_PX)

    def _on_key_press(self, event) -> None:
        if event.text.lower() == "o" and self.on_open_requested:
            self.on_open_requested()
        elif event.key == "Escape" and self.on_quit_requested:
            self.on_quit_requested()


class BuildProgressDialog(QtWidgets.QDialog):
    cancelled = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Building digital twin...")
        self.resize(560, 320)

        self.stage_label = QtWidgets.QLabel("Starting SuGaR pipeline...")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self._on_cancel_clicked)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.stage_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_view)
        layout.addWidget(cancel_button)

    def set_stage(self, stage: str) -> None:
        self.stage_label.setText(stage)

    def set_progress(self, current: int, total: int) -> None:
        if current < 0 or total <= 0:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)

    def append_log(self, line: str) -> None:
        self.log_view.appendPlainText(line)

    def _on_cancel_clicked(self) -> None:
        self.cancelled.emit()
        self.close()


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, initial_path: str | None = None):
        super().__init__()
        self.setWindowTitle("3DGS Viewer")
        self.resize(1600, 800)

        self.digital_twin_package: DigitalTwinPackage | None = None
        self._undo_snapshot: DigitalTwinPackage | None = None
        self._twin_mask_share: float | None = None
        self._last_cut: np.ndarray | None = None
        self._digital_twin_name: str = ""
        self._current_ply_path: Path | None = None
        self._builder: DigitalTwinBuilder | None = None
        self._build_dialog: BuildProgressDialog | None = None
        self._density_baseline: DensityBaseline | None = None
        self._deformation_baseline: DeformationBaseline | None = None
        self.semantic_labels: np.ndarray | None = None
        self._digital_twin_path: str | None = None
        self._preview_scene_source: Path | None = None
        self._preview_renderer: PreviewRenderer | None = None
        self._preview_dialog: PreviewDialog | None = None
        self._preview_work_dir: Path | None = None
        self._sam_reference: ReferenceGeometry | None = None
        self._sam_runner: SamPickRunner | None = None
        self._sam_work_dir: Path | None = None
        self._sam_pending_mode = "Replace"
        self._sam_pending_pick: dict | None = None
        self._sam_last_pick: dict | None = None
        self._mask_checker: TwinMaskChecker | None = None

        self.viewer = GaussianViewer(
            on_open_requested=self._open_file_dialog,
            on_quit_requested=self.close,
        )
        self.mesh_panel = MeshEditorPanel(
            on_edited=self._on_mesh_edited,
            on_selection_changed=self._on_selection_changed,
            on_sculpt_finished=self._on_sculpt_finished,
            on_edit_finished=self._on_edit_finished,
            pick_resolver=self._resolve_pick,
            selection_move_allowed=self._selection_move_allowed,
        )
        self.viewer.view.camera.link(self.mesh_panel.view.camera, props=())

        open_button = QtWidgets.QPushButton("Open File...")
        open_button.clicked.connect(self._open_file_dialog)

        self.digital_twin_button = QtWidgets.QPushButton("Edit with Digital Twin...")
        self.digital_twin_button.clicked.connect(self._open_digital_twin_dialog)

        self.sculpt_button = QtWidgets.QPushButton("Sculpt")
        self.sculpt_button.setCheckable(True)
        self.sculpt_button.setEnabled(False)
        self.sculpt_button.toggled.connect(self._on_sculpt_toggled)

        self.density_button = QtWidgets.QPushButton("Fill Density")
        self.density_button.setCheckable(True)
        self.density_button.setEnabled(False)
        self.density_button.setToolTip(
            "When on, stretching the mesh with Sculpt adds new Gaussians to fill "
            "the gap, and compressing it removes extras -- never below the "
            "original count for that triangle."
        )

        self.adapt_button = QtWidgets.QPushButton("Adapt Gaussians")
        self.adapt_button.setCheckable(True)
        self.adapt_button.setEnabled(False)
        self.adapt_button.setToolTip(
            "When on, after each Sculpt or move every Gaussian's orientation and "
            "scale follow the local rotation/stretch of the triangle it is bound "
            "to (position always follows the mesh). Applies to edits made while "
            "on. Works together with Fill Density."
        )
        self.adapt_button.toggled.connect(self._on_adapt_toggled)

        self.select_button = QtWidgets.QPushButton("Select")
        self.select_button.setCheckable(True)
        self.select_button.setEnabled(False)
        self.select_button.toggled.connect(self._on_select_toggled)

        self.pick_button = QtWidgets.QPushButton("Pick Object")
        self.pick_button.setCheckable(True)
        self.pick_button.setEnabled(False)
        self.pick_button.toggled.connect(self._on_pick_toggled)

        self.semantic_pick_button = QtWidgets.QPushButton("Semantic Pick")
        self.semantic_pick_button.setCheckable(True)
        self.semantic_pick_button.setEnabled(False)
        self.semantic_pick_button.setToolTip(
            "When on, Pick Object selects by semantic label from a precomputed "
            "*_semantic_labels.npy sidecar instead of mesh connectivity -- lets "
            "it separate objects that got fused into one mesh piece. Only "
            "enabled when such a sidecar exists for the loaded scene."
        )

        self.sam_pick_button = QtWidgets.QPushButton("SAM Pick")
        self.sam_pick_button.setCheckable(True)
        self.sam_pick_button.setEnabled(False)
        self.sam_pick_button.setToolTip(
            "When on, Pick Object selects the object under the click by asking "
            "Segment Anything for it in the training photographs (runs in the "
            "gs-train environment; the first click starts SAM and takes 10-20 seconds, "
            "later ones one to a few) -- separates "
            "objects that are fused into one mesh piece. Keeps working after Sculpt, "
            "Delete and Fill Density: the photographs show the twin as it was loaded, "
            "so a part deleted since still hides what was behind it there. Takes "
            "priority over Semantic Pick. Enabled when the scene's COLMAP "
            "source and the SAM checkpoint are found."
        )
        self.sam_level_combo = QtWidgets.QComboBox()
        self.sam_level_combo.addItems(LEVEL_NAMES)
        self.sam_level_combo.setCurrentIndex(DEFAULT_LEVEL)
        self.sam_level_combo.setEnabled(False)
        self.sam_level_combo.setToolTip(
            "How much SAM Pick selects around the click: Part (a detail), Object, or "
            "Whole (everything that belongs to the thing under the click)."
        )

        self.sam_mode_combo = QtWidgets.QComboBox()
        self.sam_mode_combo.addItems(SAM_MODES)
        self.sam_mode_combo.setEnabled(False)
        self.sam_mode_combo.setToolTip(
            "What a SAM Pick click does with the current selection. Replace: select the object under "
            "the click. Add: add it. Subtract: remove it. Exclude: run the click that made the "
            "selection again with this click marked as not part of it, and keep only what stays "
            "selected (fixes a mask that spilled over; works on the result of one Replace click, "
            "which Add, Subtract and the other tools end). Holding Shift adds and holding Ctrl "
            "subtracts, whatever is chosen here."
        )

        self.sam_vote_combo = QtWidgets.QComboBox()
        self.sam_vote_combo.addItems(SAM_VOTES)
        self.sam_vote_combo.setCurrentIndex(0)
        self.sam_vote_combo.setEnabled(False)
        self.sam_vote_combo.setToolTip(
            "How many of the photographs that see a Gaussian must place it inside Segment Anything's "
            "mask. Majority: more than half (the default). Inclusive: more than three in ten, which "
            "recovers what partial masks miss but takes background along if the twin holds some. "
            "Auto: inclusive when the twin holds only its object (built with masks, or passed the "
            "mask check), majority otherwise."
        )

        self.sync_view_button = QtWidgets.QPushButton("Sync View")
        self.sync_view_button.setCheckable(True)
        self.sync_view_button.setEnabled(False)
        self.sync_view_button.setToolTip(
            "When on, orbiting (left-drag) or zooming (mouse wheel or right-drag) either view "
            "turns or zooms the other one the same way, to compare the twin's mesh with its "
            "Gaussians from the same angle. Panning (Shift+drag) stays independent. "
            "Turned off, and re-fitted independently, whenever a twin is (re)loaded."
        )
        self.sync_view_button.toggled.connect(self._on_sync_view_toggled)

        self.preview_button = QtWidgets.QPushButton("Render Preview")
        self.preview_button.setEnabled(False)
        self.preview_button.setToolTip(
            "Renders the edited scene next to the originally loaded one, from real "
            "training cameras, with the 3DGS rasterizer (runs in the gs-train "
            "environment, takes a few seconds to a minute). Enabled when the "
            "scene's COLMAP source is found next to the loaded twin."
        )
        self.preview_button.clicked.connect(self._on_render_preview)

        self.delete_button = QtWidgets.QPushButton("Delete Selected")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self._on_delete_selected)

        self.detach_button = QtWidgets.QPushButton("Detach Selected")
        self.detach_button.setCheckable(True)
        self.detach_button.setEnabled(False)
        self.detach_button.setToolTip(
            "A switch. On: the selection is cut out of the mesh along its edge -- the triangles with "
            "all three corners selected become a piece of their own (the vertices on the cut are "
            "duplicated) -- and selections can be dragged; a selection made while it is on is cut "
            "out when it is dragged, so nothing is ever stretched. Off: no selection is moved, not "
            "even an object picked whole, and nothing is cut. The Gaussians are not changed. Undo "
            "puts the mesh back."
        )
        self.detach_button.toggled.connect(self._on_detach_toggled)

        self.close_cut_button = QtWidgets.QPushButton("Close Cut")
        self.close_cut_button.setEnabled(False)
        self.close_cut_button.setToolTip(
            "Closes the holes that the last Detach Selected opened, on both sides of the cut: each "
            "hole along the cut is filled with the triangles of least area between its edge's "
            "vertices, and every new triangle gets one Gaussian that takes the look of the surface "
            "next to it. Holes the mesh had before the cut are left alone. Undo removes the patches."
        )
        self.close_cut_button.clicked.connect(self._on_close_cut)

        self.undo_button = QtWidgets.QPushButton("Undo Delete")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self._on_undo_delete)

        self.save_button = QtWidgets.QPushButton("Save As...")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._on_save_as)

        self.status_label = QtWidgets.QLabel("No file loaded")
        self.twin_notice_label = QtWidgets.QLabel()
        self.twin_notice_label.setStyleSheet("color: #b45309;")

        toolbar = QtWidgets.QToolBar()
        toolbar.setMovable(False)
        toolbar.addWidget(open_button)
        toolbar.addWidget(self.digital_twin_button)
        toolbar.addWidget(self.sculpt_button)
        toolbar.addWidget(self.density_button)
        toolbar.addWidget(self.adapt_button)
        toolbar.addWidget(self.select_button)
        toolbar.addWidget(self.pick_button)
        toolbar.addWidget(self.semantic_pick_button)
        toolbar.addWidget(self.sam_pick_button)
        toolbar.addWidget(self.sam_level_combo)
        toolbar.addWidget(self.sam_mode_combo)
        toolbar.addWidget(self.sam_vote_combo)
        toolbar.addWidget(self.delete_button)
        toolbar.addWidget(self.detach_button)
        toolbar.addWidget(self.close_cut_button)
        toolbar.addWidget(self.undo_button)
        toolbar.addWidget(self.sync_view_button)
        toolbar.addWidget(self.preview_button)
        toolbar.addWidget(self.save_button)
        toolbar.addSeparator()
        toolbar.addWidget(self.status_label)
        self._twin_notice_action = toolbar.addWidget(self.twin_notice_label)
        self._twin_notice_action.setVisible(False)
        self.addToolBar(toolbar)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.addWidget(self.viewer.canvas.native)
        self.splitter.addWidget(self.mesh_panel.canvas.native)
        self.mesh_panel.canvas.native.hide()
        self.setCentralWidget(self.splitter)

        if initial_path:
            self._load(initial_path)

    def _open_file_dialog(self) -> None:
        path = pick_ply_file(DEFAULT_OPEN_DIR)
        if path:
            self._load(path)

    def _load(self, path: str) -> None:
        count = self.viewer.load(path)
        name = Path(path).name
        self._current_ply_path = Path(path)
        self.setWindowTitle(f"3DGS Viewer -- {name}")
        self.status_label.setText(f"{name}   ({count:,} splats)")
        self.status_label.setToolTip("")

    def _open_digital_twin_dialog(self) -> None:
        if self._current_ply_path is not None:
            colmap = find_colmap_source(self._current_ply_path)
            if colmap is not None:
                cache_path = digital_twin_cache_path(colmap.scene_name)
                if cache_path.exists():
                    self._load_digital_twin(str(cache_path))
                    return
                self._confirm_and_build(colmap.scene_name)
                return

        path = pick_digital_twin_file(DEFAULT_OPEN_DIR)
        if path:
            self._load_digital_twin(path)

    def _confirm_and_build(self, scene_name: str) -> None:
        reply = QtWidgets.QMessageBox.question(
            self,
            "Build digital twin?",
            f'No digital-twin package exists yet for "{scene_name}".\n\n'
            "Building one runs the SuGaR pipeline (baseline 3DGS training, "
            "mesh extraction, refinement) at a fast preset. This is a real "
            "wait -- roughly 10-20 minutes on a recent GPU, longer on slower "
            "hardware, not instant -- with progress shown as it runs. Once "
            "built, it's cached and will load instantly next time.\n\nStart now?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self._start_build()

    def _start_build(self) -> None:
        self.digital_twin_button.setEnabled(False)
        self._build_dialog = BuildProgressDialog(self)
        self._build_dialog.cancelled.connect(self._on_build_cancelled)

        self._builder = DigitalTwinBuilder(self)
        self._builder.stage_changed.connect(self._build_dialog.set_stage)
        self._builder.progress.connect(self._build_dialog.set_progress)
        self._builder.log_line.connect(self._build_dialog.append_log)
        self._builder.finished.connect(self._on_build_finished)

        self._build_dialog.show()
        self._builder.start(str(self._current_ply_path))

    def _on_build_cancelled(self) -> None:
        if self._builder is not None:
            self._builder.cancel()

    def _on_build_finished(self, success: bool, result: str) -> None:
        if self._build_dialog is not None:
            self._build_dialog.close()
            self._build_dialog = None
        self.digital_twin_button.setEnabled(True)

        if success:
            self._load_digital_twin(result)
        else:
            QtWidgets.QMessageBox.critical(self, "Digital-twin build failed", result)

    def _load_digital_twin(self, path: str) -> None:
        self.digital_twin_package = load_digital_twin_package(path)
        self.digital_twin_package.start_tracking_origins()
        self._undo_snapshot = None
        self.undo_button.setEnabled(False)
        self._digital_twin_name = Path(path).name
        self.sync_view_button.setChecked(False)
        self.detach_button.blockSignals(True)
        self.detach_button.setChecked(False)
        self.detach_button.blockSignals(False)
        self._reload_panels_from_package()

        self.semantic_labels = load_semantic_labels(path, len(self.digital_twin_package))
        self.semantic_pick_button.setEnabled(self.semantic_labels is not None)

        self._digital_twin_path = path
        self._preview_scene_source = find_scene_source_for_twin(path)
        self.preview_button.setEnabled(self._preview_scene_source is not None)

        self._cancel_sam_pick()
        sam_available = self._preview_scene_source is not None and sam_checkpoint_available()
        self._sam_reference = ReferenceGeometry.capture(self.digital_twin_package) if sam_available else None
        self.sam_pick_button.setChecked(False)
        self.sam_pick_button.setEnabled(sam_available)
        self.sam_level_combo.setEnabled(sam_available)
        self.sam_mode_combo.setEnabled(sam_available)
        self.sam_mode_combo.setCurrentIndex(0)
        self.sam_vote_combo.setEnabled(sam_available)
        self.sam_vote_combo.setCurrentIndex(0)
        self._twin_mask_share = None

        self._start_mask_check(path)

        self.mesh_panel.canvas.native.show()
        self.splitter.setSizes([1, 1])
        self.sculpt_button.setEnabled(True)
        self.density_button.setEnabled(True)
        self.adapt_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.pick_button.setEnabled(True)
        self.sync_view_button.setEnabled(True)
        self.save_button.setEnabled(True)

        self.setWindowTitle(f"3DGS Viewer -- {self._digital_twin_name} (digital twin)")

    def _reload_panels_from_package(self) -> None:
        package = self.digital_twin_package
        self.mesh_panel.load_mesh(package.mesh_vertices, package.mesh_faces)
        package.mesh_vertices = self.mesh_panel.vertices
        self.delete_button.setEnabled(False)
        self.detach_button.setEnabled(True)
        self._last_cut = None
        self.close_cut_button.setEnabled(False)
        self._density_baseline = DensityBaseline.capture(package)
        self._deformation_baseline = (
            DeformationBaseline.capture(package) if self.adapt_button.isChecked() else None
        )
        self.semantic_labels = None
        self.semantic_pick_button.setChecked(False)
        self.semantic_pick_button.setEnabled(False)

        count = self.viewer.load_from_package(package)
        self.status_label.setText(f"{self._digital_twin_name}   ({count:,} splats, {len(package.mesh_vertices):,} verts)")

    def _on_mesh_edited(self, _vertices: np.ndarray) -> None:
        if self.digital_twin_package is None:
            return
        positions = self.digital_twin_package.recompute_gaussian_positions()
        self.viewer.update_positions(positions)

    def _on_adapt_toggled(self, checked: bool) -> None:
        if not checked or self.digital_twin_package is None:
            self._deformation_baseline = None
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self._deformation_baseline = DeformationBaseline.capture(self.digital_twin_package)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _on_edit_finished(self) -> None:
        if not self.adapt_button.isChecked() or self.digital_twin_package is None:
            return
        if self._deformation_baseline is None:
            return
        package = self.digital_twin_package
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            updated = adapt_gaussians_to_deformation(package, self._deformation_baseline)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if updated:
            self.status_label.setText(
                f"{self._digital_twin_name}   ({len(package):,} splats, "
                f"{len(package.mesh_vertices):,} verts)   adapted {updated:,} Gaussians"
            )

    def _on_sculpt_finished(self) -> None:
        if not self.density_button.isChecked() or self.digital_twin_package is None:
            return
        if self._density_baseline is None:
            return
        changed = rebalance_density(self.digital_twin_package, self._density_baseline)
        if not changed:
            return
        self.semantic_labels = None
        self.semantic_pick_button.setChecked(False)
        self.semantic_pick_button.setEnabled(False)
        package = self.digital_twin_package
        count = self.viewer.load_from_package(package)
        self.status_label.setText(
            f"{self._digital_twin_name}   ({count:,} splats, {len(package.mesh_vertices):,} verts)"
        )

    def _resolve_semantic_pick(self, vertex_index: int) -> np.ndarray | None:
        if not self.semantic_pick_button.isChecked() or self.semantic_labels is None:
            return None
        package = self.digital_twin_package
        target_label = label_at_vertex(
            package.mesh_faces, package.gaussian_face_idx, self.semantic_labels, vertex_index
        )
        return labels_to_vertex_selection(
            package.mesh_faces, package.gaussian_face_idx, self.semantic_labels, target_label
        )

    def _resolve_pick(self, vertex_index: int) -> np.ndarray | None:
        if self.sam_pick_button.isChecked():
            return self._resolve_sam_pick(vertex_index)
        return self._resolve_semantic_pick(vertex_index)

    def _keep_selection(self) -> np.ndarray:
        current = self.mesh_panel.selected_vertices
        return current if current is not None else np.empty(0, dtype=np.int64)

    def _sam_click_mode(self) -> str:
        held = {getattr(key, "name", str(key)) for key in self.mesh_panel.pick_modifiers}
        if "Shift" in held:
            return "Add"
        if "Control" in held:
            return "Subtract"
        return self.sam_mode_combo.currentText()

    def _resolve_sam_pick(self, vertex_index: int) -> np.ndarray:
        package, reference = self.digital_twin_package, self._sam_reference
        if package is None or reference is None:
            return self._keep_selection()
        if self._sam_runner is not None and self._sam_runner.is_busy():
            self.status_label.setText("SAM Pick: still working on the previous click...")
            return self._keep_selection()
        if not reference.maps(package):
            self.status_label.setText(
                "SAM Pick cannot trace this twin back to the one that was loaded; reload the twin."
            )
            return self._keep_selection()

        mode = self._sam_click_mode()
        current = self.mesh_panel.selected_vertices
        if mode in ("Subtract", "Exclude") and current is None:
            self.status_label.setText(f"SAM Pick ({mode.lower()}): nothing is selected yet.")
            return self._keep_selection()
        point, level, negatives = reference.point_of(package, vertex_index), self.sam_level_combo.currentIndex(), np.empty((0, 3))
        min_ratio, vote_reason = self._sam_vote()
        if mode == "Exclude":
            last = self._sam_last_pick
            if last is None or not np.array_equal(np.unique(current), last["selection"]):
                self.status_label.setText(
                    "SAM Pick (exclude): Exclude corrects the result of one SAM Pick click, and this selection "
                    "was made or changed otherwise (Add, Subtract, another tool). Click with Replace first."
                )
                return self._keep_selection()
            point, level = last["point"], last["level"]
            min_ratio, vote_reason = last.get("min_ratio"), last.get("vote_reason", "")
            negatives = np.vstack([last["negatives"], reference.point_of(package, vertex_index)])

        if self._sam_work_dir is None:
            self._sam_work_dir = Path(tempfile.mkdtemp(prefix="dt_sampick_"))
            np.save(self._sam_work_dir / "positions.npy", reference.positions)
        if self._sam_runner is None:
            self._sam_runner = SamPickRunner(self)
            self._sam_runner.finished.connect(self._on_sam_finished)
            self._sam_runner.log_line.connect(self._on_sam_log_line)
        runner = self._sam_runner
        cold = not runner.is_running()
        if cold:
            runner.start(self._preview_scene_source, self._sam_work_dir / "positions.npy")
        vote = f" (vote: {vote_reason})" if vote_reason else ""
        self.status_label.setText(("SAM Pick: starting SAM and segmenting..." if cold else "SAM Pick: segmenting...") + vote)
        self._sam_pending_mode = mode
        self._sam_pending_pick = {"point": point, "level": level, "negatives": negatives,
                                  "min_ratio": min_ratio, "vote_reason": vote_reason}
        extra = {} if min_ratio is None else {"min_ratio": min_ratio}
        runner.request(point, level, self._sam_work_dir / "selected.npy", negative_points=negatives, **extra)
        return self._keep_selection()

    def _sam_vote(self) -> tuple[float | None, str]:
        choice = self.sam_vote_combo.currentText()
        if choice == "Inclusive":
            return INCLUSIVE, "inclusive"
        if choice != "Auto":
            return None, ""
        record = self.digital_twin_package.provenance if self.digital_twin_package is not None else None
        if record and record.get("masked"):
            return INCLUSIVE, "inclusive, the twin was built with masks"
        share = self._twin_mask_share
        if share is not None and share >= MASK_SHARE_WARNING:
            return INCLUSIVE, f"inclusive, {share:.0%} of the twin lies inside the masks"
        if share is not None:
            return MAJORITY, f"majority, only {share:.0%} of the twin lies inside the masks"
        return MAJORITY, "majority, nothing is known about what the twin holds"

    def _on_sam_log_line(self, line: str) -> None:
        if line.startswith("round"):
            self.status_label.setText(f"SAM Pick: {line}")

    def _on_sam_finished(self, success: bool, payload) -> None:
        path = payload["path"] if success else None
        mask = None
        if success:
            try:
                mask = np.load(path)
            except (OSError, ValueError) as error:
                success, payload = False, f"SAM Pick reported a result that could not be read:\n{error}"
        if path is not None:
            Path(path).unlink(missing_ok=True)
        if not success:
            self.status_label.setText("SAM Pick failed")
            QtWidgets.QMessageBox.warning(self, "SAM Pick failed", payload)
            return

        package, reference = self.digital_twin_package, self._sam_reference
        if package is None or reference is None or mask.shape[0] != len(reference.positions) or not reference.maps(package):
            self.status_label.setText("SAM Pick: result discarded -- the twin changed while SAM was running.")
            return
        selection = reference.selection_in(package, mask)
        if selection is None:
            if payload.get("views", 1) == 0:
                self.status_label.setText(
                    "SAM Pick: no training photograph shows that point -- a part that is deleted "
                    "now may hide it in all of them."
                )
            elif mask.any():
                self.status_label.setText("SAM Pick: everything SAM found under that point has been deleted.")
            else:
                self.status_label.setText("SAM Pick: SAM found no object under that point.")
            return
        mode, self._sam_pending_mode = self._sam_pending_mode, "Replace"
        pick, self._sam_pending_pick = self._sam_pending_pick, None
        current = self.mesh_panel.selected_vertices
        found = f"{int(mask.sum()):,} Gaussians, {len(selection):,} vertices"
        removed = 0
        if mode == "Add" and current is not None:
            selection = np.union1d(current, selection)
        elif mode == "Subtract":
            selection = np.setdiff1d(current, selection) if current is not None else selection[:0]
        elif mode == "Exclude":
            kept = np.intersect1d(current, selection) if current is not None else selection[:0]
            if kept.size == 0:
                self.status_label.setText("SAM Pick (exclude): that would leave nothing selected, so nothing was changed.")
                return
            removed, selection = len(current) - len(kept), kept
        self.mesh_panel.set_selection(selection)
        self._sam_last_pick = (
            {**pick, "selection": np.unique(selection)} if pick is not None and mode in ("Replace", "Exclude") else None
        )
        if mode == "Replace":
            self.status_label.setText(
                f"SAM Pick: {int(mask.sum()):,} Gaussians, {len(selection):,} vertices selected "
                f"(from {payload['views']} photographs)"
            )
        elif mode == "Exclude":
            self.status_label.setText(
                f"SAM Pick (exclude): {removed:,} vertices removed; {len(selection):,} vertices selected now"
            )
        else:
            self.status_label.setText(
                f"SAM Pick ({mode.lower()}): found {found}; {len(selection):,} vertices selected now"
            )

    def _start_mask_check(self, path: str) -> None:
        self._cancel_mask_check()
        self._set_twin_notice(None)
        record = self.digital_twin_package.provenance
        self.status_label.setToolTip(describe(record))
        source = self._preview_scene_source
        if source is None or (record and record.get("masked")) or not scene_has_masks(source):
            return
        cached = load_cached_check(path)
        if cached is not None:
            self._twin_mask_share = cached["share"]
            self._set_twin_notice(assess(record, cached["share"]), cached["share"])
            return
        if not GS_TRAIN_PYTHON.is_file():
            return
        checker = TwinMaskChecker(self)
        checker.finished.connect(lambda success, payload, who=checker, twin=path: self._on_mask_check_finished(who, twin, success, payload))
        self._mask_checker = checker
        checker.start(path, source)

    def _on_mask_check_finished(self, checker: TwinMaskChecker, twin_path: str, success: bool, payload) -> None:
        if checker is not self._mask_checker:
            return
        self._mask_checker = None
        checker.deleteLater()
        if not success:
            return
        save_cached_check(twin_path, payload["share"], payload["considered"], payload["views"])
        self._twin_mask_share = payload["share"]
        self._set_twin_notice(assess(self.digital_twin_package.provenance, payload["share"]), payload["share"])

    def _set_twin_notice(self, message: str | None, share: float | None = None) -> None:
        self._twin_notice_action.setVisible(message is not None)
        self.twin_notice_label.setText(f"Masks: only {share:.0%} of the twin inside" if message and share is not None else "")
        self.twin_notice_label.setToolTip(message or "")

    def _cancel_mask_check(self) -> None:
        checker, self._mask_checker = self._mask_checker, None
        if checker is not None:
            checker.finished.disconnect()
            checker.cancel()
            checker.deleteLater()

    def _cancel_sam_pick(self, wait_ms: int = 0) -> None:
        runner, work_dir = self._sam_runner, self._sam_work_dir
        self._sam_runner = self._sam_work_dir = None
        self._sam_pending_pick = self._sam_last_pick = None
        if runner is not None:
            runner.finished.disconnect(self._on_sam_finished)
            runner.close(wait_ms)
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)

    def closeEvent(self, event) -> None:
        self._cancel_mask_check()
        self._cancel_sam_pick(wait_ms=3000)
        super().closeEvent(event)

    def _on_sculpt_toggled(self, checked: bool) -> None:
        if checked:
            self.select_button.setChecked(False)
            self.pick_button.setChecked(False)
        self.mesh_panel.set_mode("sculpt" if checked else "orbit")

    def _on_select_toggled(self, checked: bool) -> None:
        if checked:
            self.sculpt_button.setChecked(False)
            self.pick_button.setChecked(False)
        self.mesh_panel.set_mode("select" if checked else "orbit")

    def _on_pick_toggled(self, checked: bool) -> None:
        if checked:
            self.sculpt_button.setChecked(False)
            self.select_button.setChecked(False)
        self.mesh_panel.set_mode("pick" if checked else "orbit")

    def _on_sync_view_toggled(self, checked: bool) -> None:
        self.viewer.view.camera.link(
            self.mesh_panel.view.camera, props=SYNCED_CAMERA_PROPS if checked else ()
        )

    def _on_selection_changed(self, selected_vertices: np.ndarray | None) -> None:
        self.delete_button.setEnabled(selected_vertices is not None)

    def _on_delete_selected(self) -> None:
        if self.mesh_panel.selected_vertices is None or self.digital_twin_package is None:
            return
        self._undo_snapshot = self.digital_twin_package.clone()
        self.digital_twin_package.remove_faces_by_vertex_selection(self.mesh_panel.selected_vertices)
        self._reload_panels_from_package()
        self.undo_button.setText("Undo Delete")
        self.undo_button.setEnabled(True)

    def _on_detach_selected(self) -> None:
        package, selected = self.digital_twin_package, self.mesh_panel.selected_vertices
        if package is None or selected is None:
            return
        snapshot = package.clone()
        result = detach_selection(package, selected)
        if result is None:
            self.status_label.setText("Detach: no triangle has all three corners selected -- select a larger area.")
            return
        if result.vertices_duplicated == 0:
            self.mesh_panel.set_selection(result.selection)
            self.status_label.setText(
                f"Detach: the selection is a piece of its own already ({result.faces_detached:,} triangles) "
                f"and can be moved."
            )
            return
        self._take_cut(snapshot, result)
        self.status_label.setText(
            f"Detached {result.faces_detached:,} triangles; {result.vertices_duplicated:,} vertices on the cut "
            f"were duplicated. The selection now moves on its own."
        )

    def _take_cut(self, snapshot, result) -> None:
        package = self.digital_twin_package
        self._undo_snapshot = snapshot
        self.mesh_panel.replace_mesh(package.mesh_vertices, package.mesh_faces, result.selection)
        package.mesh_vertices = self.mesh_panel.vertices
        self._last_cut = np.concatenate([result.cut_originals, result.cut_copies])
        self.close_cut_button.setEnabled(True)
        self.undo_button.setText("Undo Detach")
        self.undo_button.setEnabled(True)

    def _on_detach_toggled(self, checked: bool) -> None:
        if not checked:
            self.status_label.setText("Detach Selected is off: selections are not moved.")
            return
        if self.mesh_panel.selected_vertices is None:
            self.status_label.setText(
                "Detach Selected is on: a selection is cut out of the mesh when it is dragged."
            )
            return
        self._on_detach_selected()

    def _selection_move_allowed(self) -> bool:
        if not self.detach_button.isChecked():
            self.status_label.setText(
                "Moving is off: turn Detach Selected on to cut the selection out and move it."
            )
            return False
        package, selected = self.digital_twin_package, self.mesh_panel.selected_vertices
        if package is None or selected is None:
            return False
        faces, n_vertices = package.mesh_faces, package.mesh_vertices.shape[0]
        if not stretches_on_move(faces, selected, n_vertices):
            return True
        left = np.setdiff1d(selected, faces[inside_faces(faces, selected, n_vertices)]).size  # without a whole triangle
        snapshot = package.clone()
        result = detach_selection(package, selected)
        if result is None:
            self.status_label.setText(
                "Detach: no triangle has all three corners selected, so the selection is not moved."
            )
            return False
        if result.vertices_duplicated == 0:  # a piece of its own already, with loose vertices selected besides
            self.mesh_panel.set_selection(result.selection)
            return True
        self._take_cut(snapshot, result)
        self.status_label.setText(
            f"Detached {result.faces_detached:,} triangles to move them ({result.vertices_duplicated:,} vertices "
            f"on the cut duplicated)" + (f"; {left:,} selected vertices without a whole triangle stay behind" if left else "")
            + ". Undo puts the part back."
        )
        return True

    def _on_close_cut(self) -> None:
        package, cut = self.digital_twin_package, self._last_cut
        if package is None or cut is None:
            return
        snapshot = package.clone()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            result = close_cut(package, cut)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if result.faces_added == 0:
            self.status_label.setText("Close Cut: the cut left no hole that could be closed.")
            return
        self._undo_snapshot = snapshot
        self._last_cut = None
        self.close_cut_button.setEnabled(False)
        selection = self.mesh_panel.selected_vertices
        self.mesh_panel.replace_mesh(package.mesh_vertices, package.mesh_faces, selection)
        package.mesh_vertices = self.mesh_panel.vertices
        self._density_baseline = DensityBaseline.capture(package)
        self._deformation_baseline = (
            DeformationBaseline.capture(package) if self.adapt_button.isChecked() else None
        )
        self.semantic_labels = None
        self.semantic_pick_button.setChecked(False)
        self.semantic_pick_button.setEnabled(False)
        self.viewer.load_from_package(package, keep_view=True)
        self.undo_button.setText("Undo Close Cut")
        self.undo_button.setEnabled(True)
        still_open = (
            f"; {result.cut_edges_open_after:,} edges of the cut stay open where it ran into holes the mesh had before"
            if result.cut_edges_open_after else ""
        )
        self.status_label.setText(
            f"Closed {result.cycles_closed:,} holes with {result.faces_added:,} triangles and as many Gaussians{still_open}."
        )

    def _on_undo_delete(self) -> None:
        if self._undo_snapshot is None:
            return
        self.digital_twin_package = self._undo_snapshot
        self._undo_snapshot = None
        self._reload_panels_from_package()
        self.undo_button.setEnabled(False)

    def _on_render_preview(self) -> None:
        package = self.digital_twin_package
        if package is None or self._preview_scene_source is None or self._digital_twin_path is None:
            return
        if self._preview_dialog is not None:
            self._preview_dialog.close()

        self._preview_work_dir = Path(tempfile.mkdtemp(prefix="dt_preview_"))
        edited_path = self._preview_work_dir / "after.npz"
        save_digital_twin_package(str(edited_path), package)

        dialog = PreviewDialog(self)
        renderer = PreviewRenderer(self)
        renderer.finished.connect(self._on_preview_finished)
        dialog.finished.connect(self._on_preview_dialog_closed)
        self._preview_dialog, self._preview_renderer = dialog, renderer

        self.preview_button.setEnabled(False)
        dialog.show()
        renderer.start(
            self._preview_scene_source,
            {"before": self._digital_twin_path, "after": edited_path},
            self._preview_work_dir,
        )

    def _on_preview_finished(self, success: bool, payload) -> None:
        self.preview_button.setEnabled(self._preview_scene_source is not None)
        if self._preview_work_dir is not None:
            # until the dialog closes.
            (self._preview_work_dir / "after.npz").unlink(missing_ok=True)
        if self._preview_dialog is None:
            return
        if success:
            self._preview_dialog.show_results(payload)
        else:
            self._preview_dialog.set_failed(payload)

    def _on_preview_dialog_closed(self, _result: int) -> None:
        if self._preview_renderer is not None:
            self._preview_renderer.cancel()
        if self._preview_work_dir is not None:
            shutil.rmtree(self._preview_work_dir, ignore_errors=True)
        self._preview_dialog = self._preview_renderer = self._preview_work_dir = None
        self.preview_button.setEnabled(self._preview_scene_source is not None)

    def _on_save_as(self) -> None:
        if self.digital_twin_package is None:
            return
        path = pick_save_ply_path(DEFAULT_OPEN_DIR)
        if not path:
            return
        package = self.digital_twin_package
        save_gaussian_ply(
            path,
            positions=package.recompute_gaussian_positions(),
            colors=package.gaussian_colors,
            opacities=package.gaussian_opacities,
            scales=package.gaussian_scales,
            rotations=package.gaussian_quaternions,
        )
        self.status_label.setText(f"Saved {len(package.gaussian_opacities):,} splats -- {Path(path).name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="View a 3D Gaussian Splatting .ply scene (rotate/zoom with the mouse)."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help='Path to a .ply file to open immediately. Otherwise, use the "Open File..." button.',
    )
    args = parser.parse_args()

    vispy_app.use_app("pyqt5")
    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    window = MainWindow(initial_path=args.path)
    window.show()
    print(CONTROLS_HELP)
    vispy_app.run()


if __name__ == "__main__":
    main()
