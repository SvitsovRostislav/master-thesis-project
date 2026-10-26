import sys
from pathlib import Path

import numpy as np
import pytest
from PyQt5 import QtWidgets

from src.viewer.app import MainWindow

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PLY = REPO_ROOT / "data" / "raw" / "test_models" / "sample_gaussians.ply"
BONSAI_TWIN = REPO_ROOT / "data" / "gaussian_scenes" / "bonsai_digital_twin.npz"


class _FakeEvent:
    def __init__(self, pos, buttons=()):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.buttons = buttons


@pytest.mark.skipif(not SAMPLE_PLY.exists(), reason=f"sample fixture missing: {SAMPLE_PLY}")
def test_open_plain_ply(qapp):
    window = MainWindow(initial_path=None)
    window._load(str(SAMPLE_PLY))

    assert window.viewer._sizes is not None
    assert window.viewer._sizes.shape[0] > 0
    assert SAMPLE_PLY.name in window.status_label.text()
    window.close()


def test_digital_twin_controls_disabled_until_a_package_is_loaded(qapp):
    window = MainWindow(initial_path=None)

    assert not window.sculpt_button.isEnabled()
    assert not window.density_button.isEnabled()
    assert not window.select_button.isEnabled()
    assert not window.pick_button.isEnabled()
    assert not window.semantic_pick_button.isEnabled()
    assert not window.save_button.isEnabled()
    assert not window.delete_button.isEnabled()
    assert not window.undo_button.isEnabled()
    window.close()


@pytest.mark.skipif(not BONSAI_TWIN.exists(), reason=f"digital-twin fixture missing: {BONSAI_TWIN}")
class TestDigitalTwinEndToEnd:

    def test_load_enables_editing_controls(self, qapp):
        window = MainWindow(initial_path=None)
        window.show()
        window._load_digital_twin(str(BONSAI_TWIN))
        qapp.processEvents()

        assert window.digital_twin_package is not None
        assert window.sculpt_button.isEnabled()
        assert window.density_button.isEnabled()
        assert window.select_button.isEnabled()
        assert window.pick_button.isEnabled()
        assert window.save_button.isEnabled()
        assert not window.semantic_pick_button.isEnabled()
        assert not window.delete_button.isEnabled()
        window.close()

    def test_select_move_delete_undo_round_trip(self, qapp):
        window = MainWindow(initial_path=None)
        window.show()
        window._load_digital_twin(str(BONSAI_TWIN))
        qapp.processEvents()
        panel = window.mesh_panel

        original_verts = panel.vertices.shape[0]
        original_gaussians = len(window.digital_twin_package)

        window.select_button.setChecked(True)
        assert panel.mode == "select"

        cx, cy = panel.canvas.size[0] / 2.0, panel.canvas.size[1] / 2.0
        panel._select_drag_kind = "rect"
        panel._select_path = [(cx - 40, cy - 40), (cx + 40, cy + 40)]
        panel._select_release()

        assert panel.selected_vertices is not None
        n_selected = len(panel.selected_vertices)
        assert 0 < n_selected < original_verts
        assert window.delete_button.isEnabled()

        window._on_delete_selected()
        assert panel.vertices.shape[0] == original_verts - n_selected
        assert len(window.digital_twin_package) < original_gaussians
        assert window.undo_button.isEnabled()

        window._on_undo_delete()
        assert panel.vertices.shape[0] == original_verts
        assert len(window.digital_twin_package) == original_gaussians
        assert not window.undo_button.isEnabled()

        window.close()

    def test_pick_delete_undo_round_trip(self, qapp):
        window = MainWindow(initial_path=None)
        window.show()
        window._load_digital_twin(str(BONSAI_TWIN))
        qapp.processEvents()
        panel = window.mesh_panel

        original_verts = panel.vertices.shape[0]
        original_gaussians = len(window.digital_twin_package)

        window.pick_button.setChecked(True)
        assert panel.mode == "pick"
        assert not window.sculpt_button.isChecked()
        assert not window.select_button.isChecked()

        cx, cy = panel.canvas.size[0] / 2.0, panel.canvas.size[1] / 2.0
        screen_verts = panel._project_to_canvas(panel.vertices)
        nearest = int(np.argmin(np.linalg.norm(screen_verts - [cx, cy], axis=1)))
        panel._pick_press(_FakeEvent(screen_verts[nearest]))

        assert panel.selected_vertices is not None
        n_selected = len(panel.selected_vertices)
        assert 0 < n_selected <= original_verts
        assert window.delete_button.isEnabled()

        window._on_delete_selected()
        assert panel.vertices.shape[0] == original_verts - n_selected
        assert len(window.digital_twin_package) < original_gaussians
        assert window.undo_button.isEnabled()

        window._on_undo_delete()
        assert panel.vertices.shape[0] == original_verts
        assert len(window.digital_twin_package) == original_gaussians
        assert not window.undo_button.isEnabled()

        window.close()

    def test_save_as_round_trips(self, qapp, tmp_path, monkeypatch):
        window = MainWindow(initial_path=None)
        window.show()
        window._load_digital_twin(str(BONSAI_TWIN))
        qapp.processEvents()

        save_path = str(tmp_path / "saved.ply")
        monkeypatch.setattr("src.viewer.app.pick_save_ply_path", lambda initial_dir: save_path)
        window._on_save_as()

        from src.viewer.gaussian_ply_io import load_gaussian_ply

        cloud = load_gaussian_ply(save_path)
        assert len(cloud) == len(window.digital_twin_package)
        expected_positions = window.digital_twin_package.recompute_gaussian_positions()
        np.testing.assert_allclose(cloud.positions[:500], expected_positions[:500], atol=1e-3)

        window.close()

    def _drag_a_vertex_far_away(self, panel):
        """Sculpt-drags the mesh vertex closest to the canvas center a long
        way, which stretches the triangles around it substantially."""
        cx, cy = panel.canvas.size[0] / 2.0, panel.canvas.size[1] / 2.0
        screen_verts = panel._project_to_canvas(panel.vertices)
        nearest = screen_verts[int(np.argmin(np.linalg.norm(screen_verts - [cx, cy], axis=1)))]

        panel._sculpt_press(_FakeEvent(nearest))
        panel._sculpt_move(_FakeEvent(nearest + [250.0, 250.0], buttons=(1,)))
        panel._sculpt_release()

    def test_fill_density_toggle_off_leaves_gaussian_count_unchanged(self, qapp):
        """Regression guard: with the toggle off, Sculpt must behave exactly
        as it did before this feature existed."""
        window = MainWindow(initial_path=None)
        window.show()
        window._load_digital_twin(str(BONSAI_TWIN))
        qapp.processEvents()
        original_gaussians = len(window.digital_twin_package)

        window.sculpt_button.setChecked(True)
        assert not window.density_button.isChecked()  # off by default

        self._drag_a_vertex_far_away(window.mesh_panel)

        assert len(window.digital_twin_package) == original_gaussians
        window.close()

    def test_fill_density_toggle_on_adds_gaussians_when_stretched(self, qapp):
        window = MainWindow(initial_path=None)
        window.show()
        window._load_digital_twin(str(BONSAI_TWIN))
        qapp.processEvents()
        original_gaussians = len(window.digital_twin_package)

        window.sculpt_button.setChecked(True)
        window.density_button.setChecked(True)

        self._drag_a_vertex_far_away(window.mesh_panel)

        assert len(window.digital_twin_package) > original_gaussians
        # Positions must stay finite and consistent with the (larger) arrays.
        positions = window.digital_twin_package.recompute_gaussian_positions()
        assert positions.shape[0] == len(window.digital_twin_package)
        assert np.all(np.isfinite(positions))
        window.close()

    def test_semantic_pick_resolver_uses_a_sidecar_when_present(self, qapp, tmp_path):
        from src.viewer.digital_twin import load_digital_twin_package, save_digital_twin_package
        from src.viewer.semantic_segmentation import semantic_labels_path

        original = load_digital_twin_package(str(BONSAI_TWIN))
        twin_path = tmp_path / "bonsai_digital_twin.npz"
        save_digital_twin_package(str(twin_path), original)

        n = len(original)
        labels = np.zeros(n, dtype=np.int32)
        labels[: n // 2] = 1
        labels[n // 2:] = 2
        np.save(semantic_labels_path(twin_path), labels)

        window = MainWindow(initial_path=None)
        window.show()
        window._load_digital_twin(str(twin_path))
        qapp.processEvents()

        assert window.semantic_pick_button.isEnabled()
        window.semantic_pick_button.setChecked(False)
        face_with_label_1 = int(window.digital_twin_package.gaussian_face_idx[0])
        vertex = int(window.digital_twin_package.mesh_faces[face_with_label_1][0])

        assert window._resolve_semantic_pick(vertex) is None

        window.semantic_pick_button.setChecked(True)
        selection = window._resolve_semantic_pick(vertex)

        assert selection is not None
        assert selection.size > 0
        assert vertex in selection.tolist()
        window.close()

    def test_delete_invalidates_the_semantic_labels_sidecar(self, qapp, tmp_path):
        from src.viewer.digital_twin import load_digital_twin_package, save_digital_twin_package
        from src.viewer.semantic_segmentation import semantic_labels_path

        original = load_digital_twin_package(str(BONSAI_TWIN))
        twin_path = tmp_path / "bonsai_digital_twin.npz"
        save_digital_twin_package(str(twin_path), original)
        np.save(semantic_labels_path(twin_path), np.zeros(len(original), dtype=np.int32))

        window = MainWindow(initial_path=None)
        window.show()
        window._load_digital_twin(str(twin_path))
        qapp.processEvents()
        assert window.semantic_pick_button.isEnabled()

        window.select_button.setChecked(True)
        cx, cy = window.mesh_panel.canvas.size[0] / 2.0, window.mesh_panel.canvas.size[1] / 2.0
        window.mesh_panel._select_drag_kind = "rect"
        window.mesh_panel._select_path = [(cx - 40, cy - 40), (cx + 40, cy + 40)]
        window.mesh_panel._select_release()
        window._on_delete_selected()

        assert window.semantic_labels is None
        assert not window.semantic_pick_button.isEnabled()
        window.close()
