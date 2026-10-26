import json
import sys
from pathlib import Path

import pytest
from PyQt5 import QtCore, QtGui

import src.viewer.preview_render as preview_render
from src.viewer.preview_render import PreviewDialog, PreviewRenderer, find_scene_source_for_twin

_STUB_SCRIPT = '''
import json, sys, time
from pathlib import Path

args = sys.argv[1:]
out = Path(args[args.index("--out-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "argv.json").write_text(json.dumps(args))
print("Loading COLMAP cameras...", flush=True)
mode = args[args.index("--num-views") + 1]
if mode == "99":
    time.sleep(60)
if mode == "98":
    print("Traceback: boom", flush=True)
    sys.exit(3)
for spec in [a for i, a in enumerate(args) if args[i - 1] == "--package"]:
    name = spec.split("=")[0]
    for view in (0, 5):
        p = out / f"{name}_{view:04d}.png"
        p.write_bytes(b"png")
        print(f"RENDERED {name} {view} {p}", flush=True)
'''


def _make_scene(root: Path, folder: str) -> Path:
    scene = root / folder
    (scene / "sparse" / "0").mkdir(parents=True)
    (scene / "images").mkdir()
    return scene


class TestFindSceneSource:
    def test_finds_a_normalized_source_by_the_twins_scene_name(self, tmp_path, monkeypatch):
        expected = _make_scene(tmp_path, "lego_bugatti")
        monkeypatch.setattr(preview_render, "SCENE_SOURCE_LAYOUTS", [(tmp_path, "{scene}")])

        assert find_scene_source_for_twin("x/lego_bugatti_digital_twin.npz") == expected

    def test_falls_through_to_a_later_layout(self, tmp_path, monkeypatch):
        raw = tmp_path / "raw"
        expected = _make_scene(raw, "mipnerf360_bonsai")
        monkeypatch.setattr(
            preview_render, "SCENE_SOURCE_LAYOUTS", [(tmp_path / "nowhere", "{scene}"), (raw, "mipnerf360_{scene}")]
        )

        assert find_scene_source_for_twin("bonsai_digital_twin.npz") == expected

    def test_none_without_the_digital_twin_suffix(self, tmp_path, monkeypatch):
        _make_scene(tmp_path, "scene")
        monkeypatch.setattr(preview_render, "SCENE_SOURCE_LAYOUTS", [(tmp_path, "{scene}")])

        assert find_scene_source_for_twin("scene.npz") is None

    def test_none_when_the_folder_lacks_cameras_or_images(self, tmp_path, monkeypatch):
        (tmp_path / "half_built").mkdir()
        (tmp_path / "half_built" / "images").mkdir()  # no sparse/0
        monkeypatch.setattr(preview_render, "SCENE_SOURCE_LAYOUTS", [(tmp_path, "{scene}")])

        assert find_scene_source_for_twin("half_built_digital_twin.npz") is None


@pytest.fixture
def stub_script(tmp_path) -> Path:
    path = tmp_path / "stub_render.py"
    path.write_text(_STUB_SCRIPT)
    return path


def _run(qapp, renderer: PreviewRenderer, timeout_ms: int = 30_000, **start_kwargs):
    outcome = {}
    loop = QtCore.QEventLoop()

    def on_finished(success, payload):
        outcome.update(success=success, payload=payload)
        loop.quit()

    renderer.finished.connect(on_finished)
    QtCore.QTimer.singleShot(timeout_ms, loop.quit)
    renderer.start(**start_kwargs)
    if not outcome:  # a failed launch reports synchronously, before any loop runs
        loop.exec_()
    return outcome


class TestPreviewRenderer:
    def test_parses_rendered_lines_even_when_paths_contain_spaces(self, qapp, tmp_path, stub_script):
        out_dir = tmp_path / "out dir with spaces"
        renderer = PreviewRenderer(python_exe=sys.executable, script_path=stub_script)

        outcome = _run(
            qapp, renderer,
            scene_source=tmp_path / "scene", packages={"before": "a.npz", "after": "b.npz"}, out_dir=out_dir,
        )

        assert outcome["success"] is True
        assert [(name, view) for name, view, _ in outcome["payload"]] == [
            ("before", 0), ("before", 5), ("after", 0), ("after", 5)
        ]
        assert all(Path(path).is_file() for _, _, path in outcome["payload"])

    def test_passes_every_package_and_option_to_the_script(self, qapp, tmp_path, stub_script):
        out_dir = tmp_path / "out"
        renderer = PreviewRenderer(python_exe=sys.executable, script_path=stub_script)

        _run(
            qapp, renderer, scene_source=tmp_path / "scene",
            packages={"before": "a.npz", "after": "b.npz"}, out_dir=out_dir, num_views=7, max_side=640,
        )

        args = json.loads((out_dir / "argv.json").read_text())
        assert args[args.index("--scene-path") + 1] == str(tmp_path / "scene")
        assert args[args.index("--num-views") + 1] == "7"
        assert args[args.index("--max-side") + 1] == "640"
        assert [a for i, a in enumerate(args) if args[i - 1] == "--package"] == ["before=a.npz", "after=b.npz"]

    def test_a_failing_script_reports_its_exit_code_and_output(self, qapp, tmp_path, stub_script):
        renderer = PreviewRenderer(python_exe=sys.executable, script_path=stub_script)

        outcome = _run(
            qapp, renderer, scene_source=tmp_path, packages={"before": "a.npz"}, out_dir=tmp_path / "o", num_views=98
        )

        assert outcome["success"] is False
        assert "code 3" in outcome["payload"]
        assert "boom" in outcome["payload"]

    def test_cancel_kills_a_running_render_and_reports_it(self, qapp, tmp_path, stub_script):
        renderer = PreviewRenderer(python_exe=sys.executable, script_path=stub_script)
        QtCore.QTimer.singleShot(1500, renderer.cancel)

        outcome = _run(
            qapp, renderer, timeout_ms=20_000,
            scene_source=tmp_path, packages={"before": "a.npz"}, out_dir=tmp_path / "o", num_views=99,
        )

        assert outcome == {"success": False, "payload": "Cancelled."}

    def test_a_missing_interpreter_is_reported_not_swallowed(self, qapp, tmp_path, stub_script):
        renderer = PreviewRenderer(python_exe=tmp_path / "no_such_python.exe", script_path=stub_script)

        outcome = _run(qapp, renderer, scene_source=tmp_path, packages={"before": "a.npz"}, out_dir=tmp_path / "o")

        assert outcome["success"] is False
        assert "Could not start" in outcome["payload"]


def _png(path: Path, color: str) -> str:
    image = QtGui.QImage(40, 30, QtGui.QImage.Format_RGB32)
    image.fill(QtGui.QColor(color))
    assert image.save(str(path))
    return str(path)


class TestPreviewDialog:
    def test_shows_one_column_per_package_and_steps_through_the_cameras(self, qapp, tmp_path):
        dialog = PreviewDialog()
        results = [
            (name, view, _png(tmp_path / f"{name}_{view}.png", color))
            for view, color in ((0, "red"), (5, "green"), (9, "blue"))
            for name in ("before", "after")
        ]

        dialog.show_results(results)

        assert [c.text() for c in dialog._captions] == ["before", "after"]
        assert dialog.view_slider.maximum() == 2 and dialog.view_slider.isVisibleTo(dialog)
        assert not dialog._images[0].pixmap().isNull()
        assert "Camera 0" in dialog.view_label.text()
        dialog.view_slider.setValue(2)
        assert "Camera 9" in dialog.view_label.text()
        assert dialog.save_button.isEnabled()

    def test_a_single_view_hides_the_slider(self, qapp, tmp_path):
        dialog = PreviewDialog()

        dialog.show_results([("before", 0, _png(tmp_path / "b.png", "red"))])

        assert not dialog.view_slider.isVisibleTo(dialog)

    def test_failure_shows_the_message_and_offers_close(self, qapp):
        dialog = PreviewDialog()

        dialog.set_failed("Preview render exited with code 3.")

        assert "code 3" in dialog.status_label.text()
        assert dialog.close_button.text() == "Close"
        assert not dialog.save_button.isEnabled()
