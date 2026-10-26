import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import pytest

from src.viewer.semantic_segmentation import Camera
from src.viewer.twin_provenance import (
    MASK_SHARE_WARNING,
    assess,
    build_record,
    check_cache_path,
    describe,
    inside_mask_share,
    load_cached_check,
    save_cached_check,
    source_masking,
    visible_gaussians,
    with_counts,
)

SIZE = 200
FOV_DEG = 35.0


def look_at(eye, size=SIZE) -> Camera:
    eye = np.asarray(eye, float)
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    focal = size / (2 * np.tan(np.radians(FOV_DEG) / 2))
    return Camera(rotation, -rotation @ eye, focal, focal, size / 2, size / 2, size, size)


def ring(n=24, radius=7.0, height=1.8) -> list[Camera]:
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [look_at([radius * np.cos(a), radius * np.sin(a), height]) for a in angles]


def disc_mask(camera: Camera, radius=1.0) -> np.ndarray:
    """The silhouette of a sphere of the given radius at the origin."""
    centre = camera.translation  # the origin in camera coordinates
    depth = centre[2]
    cx, cy = camera.fx * centre[0] / depth + camera.cx, camera.fy * centre[1] / depth + camera.cy
    r = camera.fx * radius / np.sqrt(depth ** 2 - radius ** 2)
    yy, xx = np.mgrid[0:camera.height, 0:camera.width]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def sphere_points(n=1500) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def floor_points(n=60, half=2.0, z=-1.2) -> np.ndarray:
    xs = np.linspace(-half, half, n)
    gx, gy = np.meshgrid(xs, xs)
    return np.stack([gx.ravel(), gy.ravel(), np.full(n * n, z)], axis=1)


class TestInsideMaskShare:
    def _masks(self, cameras):
        return {view: disc_mask(camera) for view, camera in enumerate(cameras)}

    def test_a_twin_of_the_object_alone_lies_inside_its_masks(self):
        cameras = ring()

        share, count = inside_mask_share(sphere_points(), cameras, self._masks(cameras))

        assert share > 0.95 and count > 1000

    def test_a_twin_that_also_holds_the_floor_does_not(self):
        cameras = ring()
        positions = np.vstack([sphere_points(), floor_points()])

        share, count = inside_mask_share(positions, cameras, self._masks(cameras))

        assert share < MASK_SHARE_WARNING
        assert count > len(sphere_points())  # the floor was looked at, not skipped

    def test_gaussians_the_check_is_told_to_leave_out_do_not_count(self):
        cameras = ring()
        sphere, floor = sphere_points(), floor_points()
        positions = np.vstack([sphere, floor])
        considered = np.arange(len(positions)) < len(sphere)

        share, count = inside_mask_share(positions, cameras, self._masks(cameras), considered=considered)

        assert share > 0.95 and count <= len(sphere)

    def test_a_subset_of_the_views_gives_the_same_verdict(self):
        cameras = ring()
        masks = self._masks(cameras)
        with_floor = np.vstack([sphere_points(), floor_points()])

        assert inside_mask_share(sphere_points(), cameras, masks, max_views=6)[0] > 0.95
        assert inside_mask_share(with_floor, cameras, masks, max_views=6)[0] < MASK_SHARE_WARNING

    def test_too_few_views_give_no_verdict(self):
        cameras = ring()
        masks = {view: disc_mask(cameras[view]) for view in (0, 12)}  # framed twice, and three are needed

        assert inside_mask_share(sphere_points(), cameras, masks) == (0.0, 0)

    def test_views_without_a_mask_are_skipped(self):
        cameras = ring()
        masks = {view: disc_mask(cameras[view]) for view in range(0, 24, 3)}  # eight of the 24

        share, _ = inside_mask_share(sphere_points(), cameras, masks)

        assert share > 0.95


class TestVisibleGaussians:
    def test_only_opaque_gaussians_that_are_not_black_show_against_a_black_background(self):
        colors = np.array([[0.8, 0.7, 0.6], [0.0, 0.0, 0.0], [0.8, 0.7, 0.6], [0.02, 0.02, 0.05]], dtype=np.float32)
        opacities = np.array([0.9, 0.9, 0.05, 0.9], dtype=np.float32)

        assert visible_gaussians(colors, opacities).tolist() == [True, False, False, False]


def _source(root, images=3, masks=2, extra_in_masks=()):
    (root / "images").mkdir(parents=True)
    for i in range(images):
        (root / "images" / f"{i:04d}.jpg").write_bytes(b"x")
    if masks or extra_in_masks:
        (root / "masks").mkdir()
    for i in range(masks):
        (root / "masks" / f"{i:04d}.png").write_bytes(b"x")
    for name in extra_in_masks:
        (root / "masks" / name).write_bytes(b"x")
    return root


class TestSourceMasking:
    def test_photographs_and_masks_are_counted_and_the_scene_counts_as_masked(self, tmp_path):
        assert source_masking(_source(tmp_path / "s", images=5, masks=4)) == (5, 4, True)

    def test_a_scene_without_masks_is_not_masked(self, tmp_path):
        assert source_masking(_source(tmp_path / "s", images=5, masks=0)) == (5, 0, False)

    def test_only_png_files_count_as_masks(self, tmp_path):
        source = _source(tmp_path / "s", images=3, masks=1, extra_in_masks=("notes.txt", "0009.jpg"))

        assert source_masking(source) == (3, 1, True)

    def test_a_folder_that_does_not_exist_counts_as_nothing(self, tmp_path):
        assert source_masking(tmp_path / "missing") == (0, 0, False)

    @pytest.mark.skipif(sys.platform != "win32", reason="NTFS junctions")
    def test_photographs_linked_to_the_originals_are_not_masked_even_with_masks_present(self, tmp_path):
        originals = tmp_path / "originals"
        originals.mkdir()
        (originals / "0000.jpg").write_bytes(b"x")
        source = tmp_path / "s"
        (source / "masks").mkdir(parents=True)
        (source / "masks" / "0000.png").write_bytes(b"x")
        subprocess.run(["cmd", "/c", "mklink", "/J", str(source / "images"), str(originals)], check=True, capture_output=True)

        assert source_masking(source) == (1, 1, False)


class TestBuildRecord:
    PRESET = {"baseline_iterations": 3000, "refine_iterations": 2000, "mesh_triangles_target": 100000}

    def test_it_says_what_the_twin_was_built_from(self, tmp_path):
        source = _source(tmp_path / "lego_x", images=40, masks=40)

        record = build_record("lego_x", source, self.PRESET, now=datetime(2026, 3, 5, 14, 7, 9))

        assert record["scene"] == "lego_x" and record["source"] == str(source)
        assert record["built_at"] == "2026-03-05T14:07:09"
        assert (record["photographs"], record["masks_available"], record["masked"]) == (40, 40, True)
        assert record["preset"] == self.PRESET

    def test_a_build_without_masks_says_so(self, tmp_path):
        record = build_record("s", _source(tmp_path / "s", images=10, masks=0), self.PRESET)

        assert record["masked"] is False and record["masks_available"] == 0

    def test_it_is_plain_json_and_does_not_share_the_preset_dict(self, tmp_path):
        preset = dict(self.PRESET)
        record = build_record("s", _source(tmp_path / "s"), preset)
        preset["refine_iterations"] = 1

        assert json.loads(json.dumps(record)) == record
        assert record["preset"]["refine_iterations"] == 2000

    def test_the_counts_of_the_exported_twin_are_added_without_touching_the_original(self, tmp_path):
        record = build_record("s", _source(tmp_path / "s"), self.PRESET)

        counted = with_counts(record, np.int64(5), np.int32(3), 7)

        assert (counted["n_vertices"], counted["n_faces"], counted["n_gaussians"]) == (5, 3, 7)
        assert json.dumps(counted)  # numpy integers were turned into plain ones
        assert "n_vertices" not in record


class TestDescribe:
    def test_a_twin_without_a_record_says_so(self):
        assert "No build record" in describe(None)
        assert "No build record" in describe({})

    def test_a_record_reads_as_one_line(self, tmp_path):
        record = build_record("s", _source(tmp_path / "s", images=40, masks=40), {
            "baseline_iterations": 3000, "coarse_iterations": 2000, "refine_iterations": 1500, "mesh_triangles_target": 100000,
        }, now=datetime(2026, 3, 5, 14, 7, 9))

        text = describe(record)

        assert "2026-03-05 14:07:09" in text and "40 photographs" in text and "masked" in text and "not masked" not in text
        assert "3000/2000/1500" in text and "100000 triangles" in text
        assert "\n" not in text

    def test_masks_that_were_available_but_not_used_are_pointed_out(self, tmp_path):
        source = _source(tmp_path / "s", images=40, masks=40)
        record = {**build_record("s", source, {}), "masked": False}

        assert "not masked (the scene has 40 masks)" in describe(record)


class TestAssess:
    def test_a_twin_inside_its_masks_gets_no_warning(self):
        assert assess(None, 0.99) is None
        assert assess({"masked": False}, MASK_SHARE_WARNING) is None

    def test_an_unknown_share_gets_no_warning(self):
        assert assess(None, None) is None

    def test_a_low_share_without_a_record_says_it_looks_unmasked(self):
        text = assess(None, 0.085)

        assert text.startswith("only 8% of this twin's visible Gaussians") and "without masking" in text

    def test_a_low_share_against_a_record_that_claims_masking_says_so(self):
        assert "although its build record says it was built with masking" in assess({"masked": True}, 0.1)


class TestCheckCache:
    def _twin(self, tmp_path, content=b"twin"):
        path = tmp_path / "lego_x_digital_twin.npz"
        path.write_bytes(content)
        return path

    def test_it_sits_next_to_the_twin(self, tmp_path):
        assert check_cache_path(tmp_path / "lego_x_digital_twin.npz") == tmp_path / "lego_x_digital_twin_mask_check.json"

    def test_a_saved_verdict_comes_back(self, tmp_path):
        twin = self._twin(tmp_path)

        save_cached_check(twin, 0.085, 51809, 24)

        assert load_cached_check(twin) == {"share": 0.085, "considered": 51809, "views": 24}

    def test_nothing_is_stored_until_a_check_has_run(self, tmp_path):
        assert load_cached_check(self._twin(tmp_path)) is None

    def test_a_changed_twin_file_invalidates_it(self, tmp_path):
        twin = self._twin(tmp_path)
        save_cached_check(twin, 0.085, 51809, 24)

        twin.write_bytes(b"a different, longer twin")

        assert load_cached_check(twin) is None

    def test_a_twin_rewritten_at_the_same_size_at_a_later_time_invalidates_it(self, tmp_path):
        twin = self._twin(tmp_path)
        save_cached_check(twin, 0.085, 51809, 24)
        stat = twin.stat()

        os.utime(twin, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))

        assert load_cached_check(twin) is None

    def test_a_damaged_verdict_file_is_ignored(self, tmp_path):
        twin = self._twin(tmp_path)
        check_cache_path(twin).write_text("{not json", encoding="utf-8")

        assert load_cached_check(twin) is None

    def test_a_verdict_missing_a_field_is_ignored(self, tmp_path):
        twin = self._twin(tmp_path)
        save_cached_check(twin, 0.085, 51809, 24)
        stored = json.loads(check_cache_path(twin).read_text(encoding="utf-8"))
        del stored["share"]
        check_cache_path(twin).write_text(json.dumps(stored), encoding="utf-8")

        assert load_cached_check(twin) is None

    def test_saving_for_a_twin_that_is_gone_or_a_folder_that_is_read_only_does_not_raise(self, tmp_path):
        save_cached_check(tmp_path / "gone.npz", 0.5, 1, 1)  # no twin file to stat

        twin = self._twin(tmp_path)
        check_cache_path(twin).mkdir()  # a folder where the file should go: writing fails
        save_cached_check(twin, 0.5, 1, 1)
