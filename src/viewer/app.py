"""Interactive base viewer for 3D Gaussian Splatting scenes.

Lets the user pick a .ply file (a real trained 3DGS scene, or the synthetic
sample produced by scripts/generate_sample_gaussian_ply.py) via an "Open
File..." button in the toolbar, and inspect it with the mouse: left-drag to
orbit, scroll to zoom, right-drag (or Shift+left-drag) to pan. Every Gaussian
is drawn as a flat, isotropic point sprite sized/colored from its scale and
degree-0 SH color -- see gaussian_ply_io.py for the format details and
simplifications.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets
from vispy import app as vispy_app
from vispy import scene

from .digital_twin import DigitalTwinPackage, load_digital_twin_package
from .digital_twin_builder import DigitalTwinBuilder, digital_twin_cache_path, find_colmap_source
from .gaussian_ply_io import load_gaussian_ply, save_gaussian_ply
from .mesh_editor import MeshEditorPanel

REPO_ROOT = Path(__file__).resolve().parents[2]
# Real trained scenes live in data/Test data/<object>/plys/*.ply once
# present; data/raw/test_models/ (a synthetic placeholder .ply) is the
# fallback for a fresh checkout that doesn't have real data yet.
_TEST_DATA_DIR = REPO_ROOT / "data" / "Test data"
DEFAULT_OPEN_DIR = _TEST_DATA_DIR if _TEST_DATA_DIR.is_dir() else REPO_ROOT / "data" / "raw" / "test_models"

# Marker sizes are in screen pixels (vispy's 3D world-space marker scaling is
# not reliable yet, see https://github.com/vispy/vispy/issues/2357). Instead
# of a fixed empirical pixel constant, the world-to-pixel ratio is estimated
# per scene from the canvas size and the point cloud's extent (see
# `_marker_sizes`), since TurntableCamera.set_range() frames each scene's
# extent to roughly fill the canvas regardless of the scene's absolute scale.
MARKER_DIAMETER_IN_SCALES = 2.0  # splat pixel diameter, as a multiple of its mean world-space scale
MIN_MARKER_PX = 1.0
MAX_MARKER_PX = 40.0

# Real trained 3DGS scenes -- especially unbounded outdoor captures -- often
# spread a long tail of Gaussians (distant background, sky, stray SfM points)
# far beyond the densely reconstructed subject. E.g. in the Tanks & Temples
# "train" scene at 7k iterations, 50% of Gaussians sit within a radius of
# ~2.6 world units of the median center, but the 90th-percentile radius jumps
# to ~21 and the exact max to ~180: framing the camera from the *exact*
# bounding box would zoom out to fit that tail and shrink the actual subject
# to a speck. Framing is instead based on this percentile of each Gaussian's
# distance from the scene's median center, which stays close to the dense
# core regardless of how far a minority of points are scattered.
EXTENT_DISTANCE_PERCENTILE = 80.0

CONTROLS_HELP = (
    'Controls:  left-drag = rotate   scroll = zoom   right-drag = pan   '
    '"Open File..." button (or O key) = open file   Esc = quit'
)


def pick_ply_file(initial_dir: Path) -> str | None:
    """Show a native "open file" dialog and return the chosen path, or None."""
    if QtWidgets.QApplication.instance() is None:
        QtWidgets.QApplication(sys.argv)  # PyQt5 requires a live QApplication before any dialog/widget
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        "Select a Gaussian Splat / point cloud .ply file",
        str(initial_dir),
        "PLY files (*.ply);;All files (*)",
    )
    return path or None


def pick_digital_twin_file(initial_dir: Path) -> str | None:
    """Show a native "open file" dialog for a digital-twin .npz package."""
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
    """Show a native "save file" dialog for exporting a Gaussian Splat .ply."""
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
    """Owns the vispy canvas/camera/markers for one Gaussian Splat point cloud.

    Rendering only -- no window chrome or file dialogs, so it can be embedded
    in a Qt window (see MainWindow) or driven headlessly (see the offscreen
    smoke tests). `on_open_requested` / `on_quit_requested` are optional
    no-argument callbacks fired by the matching keyboard shortcuts (O / Esc),
    so a host window can wire them to the same handlers as its own button.
    """

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
        """Load a .ply file, refresh the displayed point cloud, and return its splat count."""
        cloud = load_gaussian_ply(ply_path)
        count = self._display(cloud.positions, cloud.colors, cloud.opacities, cloud.scales)
        print(f"Loaded {count:,} splats from {ply_path}")
        return count

    def load_from_package(self, package: DigitalTwinPackage) -> int:
        """Display an in-memory Gaussian cloud from a digital-twin package."""
        positions = package.recompute_gaussian_positions()
        count = self._display(
            positions, package.gaussian_colors, package.gaussian_opacities, package.gaussian_scales
        )
        print(f"Loaded {count:,} splats from digital-twin package")
        return count

    def update_positions(self, positions: np.ndarray) -> None:
        """Refresh only Gaussian positions (color/opacity/size unchanged).

        Used for live updates while dragging the linked mesh -- cheap enough
        to call once per mouse-move event, and deliberately does not re-fit
        the camera (that would be visually jarring mid-edit).
        """
        self.markers.set_data(
            pos=positions, face_color=self._face_colors, size=self._sizes, edge_width=0
        )

    def _display(
        self, positions: np.ndarray, colors: np.ndarray, opacities: np.ndarray, scales: np.ndarray
    ) -> int:
        face_colors = np.concatenate([colors, opacities[:, None]], axis=1)  # RGBA

        self.markers.set_data(
            pos=positions,
            face_color=face_colors,
            size=1.0,  # placeholder; real sizes are set below once the camera is fitted
            edge_width=0,
        )

        center, radius = self._robust_center_and_radius(positions)
        lo, hi = center - radius, center + radius
        self.view.camera.set_range(x=(lo[0], hi[0]), y=(lo[1], hi[1]), z=(lo[2], hi[2]))

        sizes = self._marker_sizes(radius, scales)
        self.markers.set_data(pos=positions, face_color=face_colors, size=sizes, edge_width=0)

        self._face_colors = face_colors
        self._sizes = sizes
        return positions.shape[0]

    @staticmethod
    def _robust_center_and_radius(positions: np.ndarray) -> tuple[np.ndarray, float]:
        """Median center and an outlier-robust framing radius (see note above)."""
        center = np.median(positions, axis=0)
        dist_from_center = np.linalg.norm(positions - center, axis=1)
        radius = float(np.percentile(dist_from_center, EXTENT_DISTANCE_PERCENTILE))
        return center, max(radius, 1e-6)

    def _marker_sizes(self, radius: float, scales: np.ndarray) -> np.ndarray:
        """Convert each Gaussian's world-space scale into an on-screen pixel diameter.

        Estimates the world-to-pixel ratio from the canvas height and the
        framing radius used in `load()`, then sizes each splat to a small
        multiple of its own mean scale. This is a simplification: real 3DGS
        rendering draws an oriented, view-dependent 2D ellipse per Gaussian;
        here every splat is an isotropic circular sprite of fixed pixel size
        (see module docstring).
        """
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
    """Shows stage/progress/log while a DigitalTwinBuilder runs in the
    background. Non-modal (see MainWindow._start_build) so the rest of the
    app -- e.g. rotating the still-open Gaussian view -- stays usable while
    it waits; "Cancel" kills the underlying subprocess, closing the window
    via the titlebar does not (the build keeps running, just unobserved --
    MainWindow._on_build_finished still fires and loads the result when
    it's done either way).
    """

    cancelled = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Building digital twin...")
        self.resize(560, 320)

        self.stage_label = QtWidgets.QLabel("Starting SuGaR pipeline...")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate until a stage reports real numbers
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
    """Application window.

    Normally just a toolbar ("Open File...") above the 3DGS view. Clicking
    "Edit with Digital Twin..." loads a mesh bound to the same Gaussians
    (see digital_twin.py) and reveals a second, editable-mesh panel in a
    splitter alongside it. "Sculpt" mode smoothly deforms the mesh with a
    brush; "Select" mode drags a rectangle (or, holding Shift, a lasso) to
    select a region; "Pick Object" instead selects with one click, by mesh
    connectivity (e.g. a cup sitting on, not fused to, a table is its own
    component -- one click grabs just the cup). Either selection tool then
    drags the highlighted selection rigidly by click-dragging on it, and
    "Delete Selected"/"Undo Delete" remove or restore it. All three modes
    recompute and live-update the Gaussians on the left every edited frame
    (see mesh_editor.py, digital_twin.py). "Save As..." exports the current
    Gaussians to a standard 3DGS .ply (gaussian_ply_io.py).

    If the open scene has no digital-twin package yet but does have a
    locatable multi-view COLMAP source (the `data/Test data/<scene>/plys/*.ply`
    convention -- see digital_twin_builder.py), "Edit with Digital Twin..."
    offers to build one automatically instead of just opening a file picker.
    """

    def __init__(self, initial_path: str | None = None):
        super().__init__()
        self.setWindowTitle("3DGS Viewer")
        self.resize(1600, 800)

        self.digital_twin_package: DigitalTwinPackage | None = None
        self._undo_snapshot: DigitalTwinPackage | None = None
        self._digital_twin_name: str = ""
        self._current_ply_path: Path | None = None
        self._builder: DigitalTwinBuilder | None = None
        self._build_dialog: BuildProgressDialog | None = None

        self.viewer = GaussianViewer(
            on_open_requested=self._open_file_dialog,
            on_quit_requested=self.close,
        )
        self.mesh_panel = MeshEditorPanel(
            on_edited=self._on_mesh_edited,
            on_selection_changed=self._on_selection_changed,
        )

        open_button = QtWidgets.QPushButton("Open File...")
        open_button.clicked.connect(self._open_file_dialog)

        self.digital_twin_button = QtWidgets.QPushButton("Edit with Digital Twin...")
        self.digital_twin_button.clicked.connect(self._open_digital_twin_dialog)

        self.sculpt_button = QtWidgets.QPushButton("Sculpt")
        self.sculpt_button.setCheckable(True)
        self.sculpt_button.setEnabled(False)
        self.sculpt_button.toggled.connect(self._on_sculpt_toggled)

        self.select_button = QtWidgets.QPushButton("Select")
        self.select_button.setCheckable(True)
        self.select_button.setEnabled(False)
        self.select_button.toggled.connect(self._on_select_toggled)

        self.pick_button = QtWidgets.QPushButton("Pick Object")
        self.pick_button.setCheckable(True)
        self.pick_button.setEnabled(False)
        self.pick_button.toggled.connect(self._on_pick_toggled)

        self.delete_button = QtWidgets.QPushButton("Delete Selected")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self._on_delete_selected)

        self.undo_button = QtWidgets.QPushButton("Undo Delete")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self._on_undo_delete)

        self.save_button = QtWidgets.QPushButton("Save As...")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._on_save_as)

        self.status_label = QtWidgets.QLabel("No file loaded")

        toolbar = QtWidgets.QToolBar()
        toolbar.setMovable(False)
        toolbar.addWidget(open_button)
        toolbar.addWidget(self.digital_twin_button)
        toolbar.addWidget(self.sculpt_button)
        toolbar.addWidget(self.select_button)
        toolbar.addWidget(self.pick_button)
        toolbar.addWidget(self.delete_button)
        toolbar.addWidget(self.undo_button)
        toolbar.addWidget(self.save_button)
        toolbar.addSeparator()
        toolbar.addWidget(self.status_label)
        self.addToolBar(toolbar)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.addWidget(self.viewer.canvas.native)
        self.splitter.addWidget(self.mesh_panel.canvas.native)
        self.mesh_panel.canvas.native.hide()  # revealed once a digital-twin package is loaded
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

    def _open_digital_twin_dialog(self) -> None:
        if self._current_ply_path is not None:
            colmap = find_colmap_source(self._current_ply_path)
            if colmap is not None:
                cache_path = digital_twin_cache_path(colmap.scene_name)
                if cache_path.exists():
                    self._load_digital_twin(str(cache_path))  # cached -- never rebuild
                    return
                self._confirm_and_build(colmap.scene_name)
                return

        # No open scene, or no locatable COLMAP source to build from --
        # fall back to picking an already-exported package by hand.
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
            "wait -- realistically 45-70 minutes, not instant -- with "
            "progress shown as it runs. Once built, it's cached and will "
            "load instantly next time.\n\nStart now?",
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
        self._undo_snapshot = None
        self.undo_button.setEnabled(False)
        self._digital_twin_name = Path(path).name
        self._reload_panels_from_package()

        self.mesh_panel.canvas.native.show()
        self.splitter.setSizes([1, 1])
        self.sculpt_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.pick_button.setEnabled(True)
        self.save_button.setEnabled(True)

        self.setWindowTitle(f"3DGS Viewer -- {self._digital_twin_name} (digital twin)")

    def _reload_panels_from_package(self) -> None:
        """Refresh both panels from `self.digital_twin_package` (used after
        the initial load, and again after any delete/undo, since those
        change the vertex/face/Gaussian counts)."""
        package = self.digital_twin_package
        self.mesh_panel.load_mesh(package.mesh_vertices, package.mesh_faces)
        # Editing mutates mesh_panel.vertices in place; re-point the package
        # at that same array so recompute_gaussian_positions() sees live edits.
        package.mesh_vertices = self.mesh_panel.vertices
        self.delete_button.setEnabled(False)  # load_mesh() just cleared the selection

        count = self.viewer.load_from_package(package)
        self.status_label.setText(f"{self._digital_twin_name}   ({count:,} splats, {len(package.mesh_vertices):,} verts)")

    def _on_mesh_edited(self, _vertices: np.ndarray) -> None:
        if self.digital_twin_package is None:
            return
        positions = self.digital_twin_package.recompute_gaussian_positions()
        self.viewer.update_positions(positions)

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

    def _on_selection_changed(self, selected_vertices: np.ndarray | None) -> None:
        self.delete_button.setEnabled(selected_vertices is not None)

    def _on_delete_selected(self) -> None:
        if self.mesh_panel.selected_vertices is None or self.digital_twin_package is None:
            return
        self._undo_snapshot = self.digital_twin_package.clone()
        self.digital_twin_package.remove_faces_by_vertex_selection(self.mesh_panel.selected_vertices)
        self._reload_panels_from_package()
        self.undo_button.setEnabled(True)

    def _on_undo_delete(self) -> None:
        if self._undo_snapshot is None:
            return
        self.digital_twin_package = self._undo_snapshot
        self._undo_snapshot = None
        self._reload_panels_from_package()
        self.undo_button.setEnabled(False)

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
