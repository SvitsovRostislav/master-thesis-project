"""End-to-end smoke tests for the actual application (src/viewer/app.py).

These exercise MainWindow the same way a user would (open a file, load a
digital-twin package, select/move/delete/undo/save), calling the real
button-handler methods directly rather than reimplementing their logic --
regressions here would mean the app itself is broken, not just a helper
function. The digital-twin scenario needs a real exported package
(data/gaussian_scenes/bonsai_digital_twin.npz, ~95 MB, produced by
scripts/export_digital_twin_package.py from a trained SuGaR checkpoint) --
it is not committed as a repo fixture, so that test is skipped if absent.
"""

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
    """Minimal stand-in for a vispy mouse event -- just enough attributes
    for the handler under test to read."""

    def __init__(self, pos):
        self.pos = np.asarray(pos, dtype=np.float64)


@pytest.mark.skipif(not SAMPLE_PLY.exists(), reason=f"sample fixture missing: {SAMPLE_PLY}")
def test_open_plain_ply(qapp):
    window = MainWindow(initial_path=None)
    window._load(str(SAMPLE_PLY))

    assert window.viewer._sizes is not None  # only set once _display() has run
    assert window.viewer._sizes.shape[0] > 0
    assert SAMPLE_PLY.name in window.status_label.text()
    window.close()


def test_digital_twin_controls_disabled_until_a_package_is_loaded(qapp):
    window = MainWindow(initial_path=None)

    assert not window.sculpt_button.isEnabled()
    assert not window.select_button.isEnabled()
    assert not window.pick_button.isEnabled()
    assert not window.save_button.isEnabled()
    assert not window.delete_button.isEnabled()
    assert not window.undo_button.isEnabled()
    window.close()


@pytest.mark.skipif(not BONSAI_TWIN.exists(), reason=f"digital-twin fixture missing: {BONSAI_TWIN}")
class TestDigitalTwinEndToEnd:
    """Same scenario as the manual verification run during development,
    kept here so a future change can't silently break it."""

    def test_load_enables_editing_controls(self, qapp):
        window = MainWindow(initial_path=None)
        window.show()
        window._load_digital_twin(str(BONSAI_TWIN))
        qapp.processEvents()

        assert window.digital_twin_package is not None
        assert window.sculpt_button.isEnabled()
        assert window.select_button.isEnabled()
        assert window.pick_button.isEnabled()
        assert window.save_button.isEnabled()
        assert not window.delete_button.isEnabled()  # nothing selected yet
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

        # Select a small box around the canvas center (same technique as
        # the manual verification driver).
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
        """Same scenario as test_select_move_delete_undo_round_trip, but
        selecting via "Pick Object" (click -> mesh-connectivity component)
        instead of a rectangle drag."""
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
