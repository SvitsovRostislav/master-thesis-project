import math

import numpy as np
import pytest

from src.viewer.digital_twin import load_digital_twin_package
import src.viewer.sam_pick as sam_pick
from src.viewer.sam_pick import (INCLUSIVE, MAJORITY, 
    ReferenceGeometry,
    SceneGeometry,
    Voter,
    _DepthMap,
    evenly_spaced,
    initial_prompts,
    near_points,
    pick_fitting_mask,
    pick_level_mask,
    segment_object,
    spread_points,
    vote_selection,
)
from src.viewer.semantic_segmentation import Camera, labels_to_vertex_selection

SIZE = 200
FOV_DEG = 35.0
SPHERE_RADIUS = 1.0
CENTERS = {"A": np.array([-1.3, 0.0, 0.0]), "B": np.array([1.3, 0.0, 0.0])}


def look_at(eye, target=(0.0, 0.0, 0.0), size=SIZE) -> Camera:
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    focal = size / (2 * np.tan(np.radians(FOV_DEG) / 2))
    return Camera(rotation, -rotation @ eye, focal, focal, size / 2, size / 2, size, size)


def ring(n=24, radius=7.0, height=1.8) -> list[Camera]:
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [look_at([radius * np.cos(a), radius * np.sin(a), height]) for a in angles]


def sphere_points(center, n=1500) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    unit = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
    return center + SPHERE_RADIUS * unit


def two_spheres():
    a, b = sphere_points(CENTERS["A"]), sphere_points(CENTERS["B"])
    return np.vstack([a, b]).astype(np.float32), np.arange(len(a) + len(b)) < len(a)


def two_spheres_with_hidden_cores():
    shell, is_a = two_spheres()
    cores = [CENTERS[name] + 0.3 * sphere_points(np.zeros(3), n=200) for name in "AB"]
    kind = np.concatenate([np.where(is_a, "A", "B"), np.full(200, "core A"), np.full(200, "core B")])
    return np.vstack([shell, *cores]).astype(np.float32), kind


class StandInSam:

    def __init__(self, cameras):
        self.cameras = cameras
        self.calls = []

    def _discs(self, camera):
        yy, xx = np.mgrid[0:camera.height, 0:camera.width]
        owner = np.full((camera.height, camera.width), "", dtype=object)
        nearest = np.full((camera.height, camera.width), np.inf)
        for name, center in CENTERS.items():
            pixel, depth, _ = _project_one(camera, center)
            radius = camera.fx * SPHERE_RADIUS / np.sqrt(depth ** 2 - SPHERE_RADIUS ** 2)
            inside = (xx - pixel[0]) ** 2 + (yy - pixel[1]) ** 2 <= radius ** 2
            claim = inside & (depth < nearest)
            owner[claim], nearest[claim] = name, depth
        return owner

    def __call__(self, view, points):
        self.calls.append(view)
        camera = self.cameras[view]
        owner = self._discs(camera)
        xy = points[0]
        hit = owner[int(round(xy[1])), int(round(xy[0]))]
        empty = np.zeros((camera.height, camera.width), bool)
        if hit == "":
            return np.stack([empty, empty, empty])
        medium = owner == hit
        _, depth, _ = _project_one(camera, CENTERS[hit])
        yy, xx = np.mgrid[0:camera.height, 0:camera.width]
        patch_radius = 0.35 * camera.fx * SPHERE_RADIUS / depth
        patch = medium & ((xx - xy[0]) ** 2 + (yy - xy[1]) ** 2 <= patch_radius ** 2)
        return np.stack([patch, medium, owner != ""])


def _project_one(camera, point):
    cam = camera.rotation @ point + camera.translation
    return np.array([camera.fx * cam[0] / cam[2] + camera.cx, camera.fy * cam[1] / cam[2] + camera.cy]), cam[2], cam


def click_on_a():
    return CENTERS["A"] + np.array([-SPHERE_RADIUS, 0.0, 0.0])  # the far-left pole, facing away from B


class TestVisibility:
    def _camera(self):
        return Camera(np.eye(3), np.zeros(3), 100.0, 100.0, 50.0, 50.0, 100, 100)

    def test_a_point_behind_another_along_the_ray_is_hidden(self):
        front = np.array([[0.0, 0.0, 5.0], [0.3, 0.2, 5.0]])
        back = front * (9.0 / 5.0)  # the same rays, further out
        depth_map = _DepthMap(self._camera(), np.vstack([front, back]))

        assert depth_map.visible_points().tolist() == [True, True, False, False]

    def test_a_slanted_surface_is_not_mistaken_for_an_occluder(self):
        xs = np.linspace(-1, 1, 200)
        plane = np.stack([xs, np.zeros_like(xs), 5.0 + 0.5 * xs], axis=1)  # ~27 degrees off face-on

        assert _DepthMap(self._camera(), plane).visible_points().all()

    def test_points_behind_the_camera_or_outside_the_frame_are_not_visible(self):
        points = np.array([[0.0, 0.0, -2.0], [50.0, 0.0, 5.0], [0.0, 0.0, 5.0]])

        assert _DepthMap(self._camera(), points).visible_points().tolist() == [False, False, True]


class TestMaskChoice:
    def test_the_level_picks_the_candidate_of_that_area_rank(self):
        small, medium, large = (np.zeros((10, 10), bool) for _ in range(3))
        small[:2], medium[:5], large[:9] = True, True, True
        shuffled = np.stack([large, small, medium])

        assert (pick_level_mask(shuffled, 0) == small).all()
        assert (pick_level_mask(shuffled, 1) == medium).all()
        assert (pick_level_mask(shuffled, 2) == large).all()

    def test_growth_keeps_the_requested_extent_while_it_covers_the_selection(self):
        camera = Camera(np.eye(3), np.zeros(3), 10.0, 10.0, 5.0, 5.0, 10, 10)
        part, object_, everything = (np.zeros((10, 10), bool) for _ in range(3))
        part[:3, :3], object_[:6, :6], everything[:] = True, True, True
        candidates = np.stack([everything, part, object_])
        selected_pixels = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 0.5]])  # inside even the smallest

        assert (pick_fitting_mask(candidates, selected_pixels, camera, level=0) == part).all()
        assert (pick_fitting_mask(candidates, selected_pixels, camera, level=1) == object_).all()
        assert (pick_fitting_mask(candidates, selected_pixels, camera, level=2) == everything).all()

    def test_growth_moves_up_to_a_larger_mask_when_the_requested_one_misses_the_selection(self):
        camera = Camera(np.eye(3), np.zeros(3), 10.0, 10.0, 5.0, 5.0, 10, 10)
        part, object_, everything = (np.zeros((10, 10), bool) for _ in range(3))
        part[:3, :3], object_[:6, :6], everything[:] = True, True, True
        selected_pixels = np.array([[1.5, 1.5], [4.5, 4.5], [2.5, 5.5]])  # spread over the object

        chosen = pick_fitting_mask(np.stack([part, object_, everything]), selected_pixels, camera, level=0)

        assert (chosen == object_).all()

    def test_growth_falls_back_to_the_best_covering_mask_when_none_is_good_enough(self):
        camera = Camera(np.eye(3), np.zeros(3), 10.0, 10.0, 5.0, 5.0, 10, 10)
        left, right = np.zeros((10, 10), bool), np.zeros((10, 10), bool)
        left[:, :4], right[:, :2] = True, True
        selected_pixels = np.array([[1.0, 5.0], [3.0, 5.0], [3.5, 5.0], [7.0, 5.0], [8.0, 5.0]])  # left covers 3 of 5

        assert (pick_fitting_mask(np.stack([right, left]), selected_pixels, camera) == left).all()

    def test_growth_rejects_masks_that_miss_most_of_the_selection(self):
        camera = Camera(np.eye(3), np.zeros(3), 10.0, 10.0, 5.0, 5.0, 10, 10)
        left = np.zeros((10, 10), bool)
        left[:, :4] = True
        selected_pixels = np.array([[1.0, 5.0], [3.0, 5.0], [7.0, 5.0], [8.0, 5.0], [9.0, 5.0]])  # left covers 2 of 5

        assert pick_fitting_mask(left[None], selected_pixels, camera) is None


class TestSegmentObject:
    def setup_method(self):
        self.positions, self.is_a = two_spheres()
        self.cameras = ring()

    def _segment(self, **kwargs):
        sam = StandInSam(self.cameras)
        result = segment_object(self.positions, self.cameras, click_on_a(), sam, **kwargs)
        return result, sam

    def test_selects_the_clicked_object_all_the_way_round_and_not_its_neighbour(self):
        result, _ = self._segment()

        recall = result.selected[self.is_a].mean()
        leak = result.selected[~self.is_a].mean()
        assert recall > 0.9, f"only {recall:.0%} of the clicked object was selected"
        assert leak < 0.02, f"{leak:.0%} of the neighbouring object was selected"

    def test_the_default_vote_is_a_strict_majority(self):
        default, _ = self._segment()
        explicit, _ = self._segment(min_ratio=MAJORITY)

        assert np.array_equal(default.selected, explicit.selected)

    def test_an_inclusive_vote_recovers_what_partial_masks_miss_and_takes_no_neighbour(self):
        honest = StandInSam(self.cameras)

        def partial(view, points):
            masks = honest(view, points)
            if view % 5 < 3:  # in three views out of five the mask covers one half of the object
                pixel, _, _ = _project_one(self.cameras[view], CENTERS["A"])
                masks[:, :, int(pixel[0]):] = False
            return masks

        majority = segment_object(self.positions, self.cameras, click_on_a(), partial)
        inclusive = segment_object(self.positions, self.cameras, click_on_a(), partial, min_ratio=INCLUSIVE)

        assert inclusive.selected[self.is_a].mean() > majority.selected[self.is_a].mean() + 0.05
        assert inclusive.selected[~self.is_a].mean() < 0.02

    def test_growth_reaches_the_side_the_click_views_cannot_see(self):
        one_view, _ = self._segment(initial_views=1, max_rounds=1)  # a single photograph of the click
        grown, _ = self._segment(initial_views=1)

        assert one_view.selected[self.is_a].mean() < 0.7  # one photograph sees about one side
        assert grown.selected[self.is_a].mean() > 0.9
        assert grown.selected[~self.is_a].mean() < 0.02
        assert grown.rounds > 1

    def test_the_level_sets_the_extent_of_the_selection(self):
        part, _ = self._segment(level=0)
        whole, _ = self._segment(level=2)

        assert part.selected.sum() < 0.5 * self._segment(level=1)[0].selected.sum()
        assert whole.selected[~self.is_a].mean() > 0.8  # the largest candidate covers both objects

    def test_never_asks_for_more_views_than_the_budget(self):
        result, sam = self._segment(max_views=7)

        assert len(sam.calls) == len(set(sam.calls)) <= 7
        assert result.views_used == sorted(sam.calls)

    def test_a_point_no_camera_can_see_selects_nothing_and_asks_nothing(self):
        sam = StandInSam(self.cameras)

        result = segment_object(self.positions, self.cameras, np.array([0.0, 0.0, 500.0]), sam)

        assert not result.selected.any()
        assert sam.calls == [] and result.rounds == 0

    def test_a_prompt_that_lands_on_background_selects_nothing(self):
        def background_only(view, points):
            return np.zeros((3, SIZE, SIZE), bool)

        result = segment_object(self.positions, self.cameras, click_on_a(), background_only)

        assert not result.selected.any()

    def test_the_hidden_inside_of_the_clicked_object_is_taken_along_but_not_the_neighbours(self):
        positions, kind = two_spheres_with_hidden_cores()
        sam = StandInSam(self.cameras)

        # The rule itself, unrestricted: this core is a solid cluster far from the surface, unlike the
        # hidden surfaces of a real twin, which lie close to visible ones.
        result = segment_object(positions, self.cameras, click_on_a(), sam, hull_neighbourhood=math.inf)

        assert result.selected[kind == "core A"].mean() > 0.9  # inside the silhouette in every view
        assert result.selected[kind == "core B"].mean() < 0.05  # inside it only where B hides behind A
        assert result.selected[kind == "A"].mean() > 0.9
        assert result.selected[kind == "B"].mean() < 0.02

    def test_views_whose_prompt_lands_on_a_detail_do_not_erase_the_selection(self):
        honest = StandInSam(self.cameras)
        first_round = set(initial_prompts(self.cameras, self.positions, click_on_a(), 6))
        asked_late = []

        def flaky(view, points):
            masks = honest(view, points)
            if view in first_round:
                return masks
            asked_late.append(view)
            tiny = np.zeros_like(masks[0])
            tiny[100:103, 100:103] = True  # the prompt landed on a small detail
            return np.stack([tiny, tiny, tiny])

        result = segment_object(self.positions, self.cameras, click_on_a(), flaky)

        assert asked_late and set(asked_late).isdisjoint(result.views_used)
        assert result.selected[self.is_a].mean() > 0.4  # what the click's own views established survives
        assert result.selected[~self.is_a].mean() < 0.02

    def test_a_click_view_whose_mask_is_a_speck_does_not_vote(self):
        honest = StandInSam(self.cameras)
        first_round = sorted(initial_prompts(self.cameras, self.positions, click_on_a(), 6))
        specked = first_round[0]

        def one_bad(view, points):
            masks = honest(view, points)
            if view != specked:
                return masks
            speck = np.zeros_like(masks[0])
            speck[50:52, 50:52] = True
            return np.stack([speck, speck, speck])

        result = segment_object(self.positions, self.cameras, click_on_a(), one_bad)

        assert specked not in result.views_used
        assert result.selected[self.is_a].mean() > 0.9

    def test_click_views_are_spread_over_the_capture(self):
        views = initial_prompts(self.cameras, self.positions, click_on_a(), max_views=4)

        assert 2 <= len(views) <= 4
        assert evenly_spaced(sorted(views), 99) == sorted(views)


class TestPrefetch:

    def setup_method(self):
        self.positions, self.is_a = two_spheres()
        self.cameras = ring()

    def test_it_announces_each_round_s_views_before_any_of_them_is_asked_about(self):
        sam = StandInSam(self.cameras)
        calls: list[list[int]] = []

        def prefetch(views):
            calls.append(list(views))
            assert sam.calls == sum(calls[:-1], [])  # only earlier rounds have been asked about so far

        result = segment_object(self.positions, self.cameras, click_on_a(), sam, initial_views=1, prefetch=prefetch)

        assert len(calls) == result.rounds > 1  # one initial view forces growth: more than one round
        assert sam.calls == sum(calls, [])  # exactly the announced views, round by round, in that order

    def test_without_it_nothing_about_the_answer_or_the_calls_made_changes(self):
        with_hook = segment_object(self.positions, self.cameras, click_on_a(), StandInSam(self.cameras), initial_views=1, prefetch=lambda views: None)
        without_hook = segment_object(self.positions, self.cameras, click_on_a(), StandInSam(self.cameras), initial_views=1)

        assert np.array_equal(with_hook.selected, without_hook.selected)
        assert with_hook.views_used == without_hook.views_used and with_hook.rounds == without_hook.rounds

    def test_a_click_that_asks_nothing_calls_it_zero_times(self):
        calls = []

        segment_object(self.positions, self.cameras, np.array([0.0, 0.0, 500.0]), StandInSam(self.cameras), prefetch=calls.append)

        assert calls == []


def _extra_gaussians(faces) -> dict:
    n = len(faces)
    return dict(
        face_idx=np.asarray(faces, dtype=np.int32), bary=np.full((n, 3), 1 / 3, dtype=np.float32),
        colors=np.ones((n, 3), dtype=np.float32), opacities=np.ones(n, dtype=np.float32),
        scales=np.full((n, 3), 0.01, dtype=np.float32), quaternions=np.tile(np.float32([1, 0, 0, 0]), (n, 1)),
    )


class TestReferenceGeometry:

    def _loaded(self, tmp_path, make_grid_twin, n=6):
        make_grid_twin(tmp_path / "t.npz", n=n)
        package = load_digital_twin_package(str(tmp_path / "t.npz"))
        package.start_tracking_origins()  # what the application does when it loads a twin
        return package, ReferenceGeometry.capture(package)

    def test_moving_vertices_keeps_the_reference_valid_and_leaves_it_where_it_was(self, tmp_path, make_grid_twin):
        package, reference = self._loaded(tmp_path, make_grid_twin)
        before, loaded_vertex = reference.positions.copy(), reference.vertices[3].copy()

        package.mesh_vertices[3] += np.float32(5.0)

        assert reference.maps(package)
        assert np.array_equal(reference.positions, before)
        assert not np.allclose(package.recompute_gaussian_positions(), before)
        assert np.allclose(reference.point_of(package, 3), loaded_vertex)  # the click goes out as loaded

    def test_deletes_and_density_changes_keep_it_mapping(self, tmp_path, make_grid_twin):
        package, reference = self._loaded(tmp_path, make_grid_twin)

        package.remove_faces_by_vertex_selection(np.array([0, 1, 7]))
        assert reference.maps(package)
        package.add_gaussians(**_extra_gaussians([0, 3]))
        package.remove_gaussians_by_index(np.array([0, 1]))
        assert reference.maps(package)
        package.remove_faces_by_vertex_selection(np.array([0]))
        assert reference.maps(package)

    def test_a_package_that_does_not_keep_its_origins_cannot_be_mapped(self, tmp_path, make_grid_twin):
        make_grid_twin(tmp_path / "t.npz", n=5)
        package = load_digital_twin_package(str(tmp_path / "t.npz"))
        reference = ReferenceGeometry.capture(package)
        assert not reference.maps(package)

        package.start_tracking_origins()
        assert reference.maps(package)

        package.face_origin = package.face_origin[:-1]  # out of step with the faces
        assert not reference.maps(package)

    def test_after_a_delete_a_click_is_sent_from_where_the_vertex_was_loaded(self, tmp_path, make_grid_twin):
        package, reference = self._loaded(tmp_path, make_grid_twin)
        loaded = package.mesh_vertices.copy()

        package.remove_faces_by_vertex_selection(np.array([0, 1, 2]))
        package.mesh_vertices[4] += np.float32(3.0)  # and sculpted afterwards

        for vertex in (0, 4, package.mesh_vertices.shape[0] - 1):
            assert np.allclose(reference.point_of(package, vertex), loaded[package.vertex_origin[vertex]])
        assert not np.allclose(reference.point_of(package, 0), loaded[0])  # the numbers moved; the point did not

    def test_for_a_twin_as_loaded_the_answer_selects_exactly_what_it_did_before(self, tmp_path, make_grid_twin):
        package, reference = self._loaded(tmp_path, make_grid_twin)
        rng = np.random.default_rng(0)

        for _ in range(5):
            answer = rng.random(len(package)) < 0.2
            before = labels_to_vertex_selection(package.mesh_faces, package.gaussian_face_idx, answer.astype(np.int32), 1)
            assert np.array_equal(reference.selection_in(package, answer), before)
        assert reference.selection_in(package, np.zeros(len(package), dtype=bool)) is None

    def test_after_a_delete_the_answer_selects_what_is_left_of_the_object(self, tmp_path, make_grid_twin):
        package, reference = self._loaded(tmp_path, make_grid_twin)
        answer = np.zeros(len(package), dtype=bool)
        answer[:20] = True  # the object: the first twenty triangles
        before = reference.selection_in(package, answer)

        package.remove_faces_by_vertex_selection(np.array([0, 1]))  # part of it deleted
        after = reference.selection_in(package, answer)

        surviving = np.isin(package.face_origin, np.arange(20))
        assert np.array_equal(after, np.unique(package.mesh_faces[surviving]))  # all of what is left of it
        assert set(package.vertex_origin[after].tolist()) <= set(before.tolist())  # and nothing else
        assert 0 < after.size < before.size

    def test_an_answer_that_lies_entirely_in_the_deleted_part_selects_nothing(self, tmp_path, make_grid_twin):
        package, reference = self._loaded(tmp_path, make_grid_twin)
        answer = np.zeros(len(package), dtype=bool)
        answer[:2] = True  # faces 0 and 1, which both use vertex 1

        package.remove_faces_by_vertex_selection(np.array([1]))

        assert reference.selection_in(package, answer) is None

    def test_gaussians_that_fill_density_adds_or_removes_do_not_change_the_answer(self, tmp_path, make_grid_twin):
        package, reference = self._loaded(tmp_path, make_grid_twin)
        answer = np.zeros(len(package), dtype=bool)
        answer[5:15] = True
        before = reference.selection_in(package, answer)

        package.add_gaussians(**_extra_gaussians([5, 30]))
        package.remove_gaussians_by_index(np.array([6, 7]))  # the only Gaussians of faces 6 and 7

        assert np.array_equal(reference.selection_in(package, answer), before)


def _reference_vote(cameras, positions, masks, min_ratio=0.5):
    """The voting rule written out plainly, all views at once -- what Voter must equal."""
    n = positions.shape[0]
    seen, inside, framed, inside_framed = (np.zeros(n, dtype=np.int32) for _ in range(4))
    for view, mask in masks.items():
        depth_map = _DepthMap(cameras[view], positions)
        in_frame = np.nonzero(depth_map.in_frame)[0]
        height, width = mask.shape
        x = np.clip((depth_map.pixels[in_frame, 0] * width / cameras[view].width).astype(int), 0, width - 1)
        y = np.clip((depth_map.pixels[in_frame, 1] * height / cameras[view].height).astype(int), 0, height - 1)
        in_mask = mask[y, x]
        visible = depth_map.visible_points()[in_frame]
        framed[in_frame] += 1
        inside_framed[in_frame] += in_mask
        seen[in_frame[visible]] += 1
        inside[in_frame[visible]] += in_mask[visible]
    return ((seen > 0) & (inside > min_ratio * seen)) | (
        (seen == 0) & (framed >= sam_pick.HULL_MIN_VIEWS) & (inside_framed >= sam_pick.HULL_RATIO * framed)
    )


class TestVoter:
    def setup_method(self):
        self.positions, _ = two_spheres_with_hidden_cores()[0], None
        self.cameras = ring()
        sam = StandInSam(self.cameras)
        rng = np.random.default_rng(1)
        self.masks = {
            v: sam(v, np.array([[100.0 + rng.uniform(-8, 8), 100.0 + rng.uniform(-8, 8)]]))[1] for v in range(0, 24, 2)
        }

    def test_adding_views_one_at_a_time_gives_the_vote_of_all_of_them(self):
        voter = Voter(self.positions)
        added = {}
        for view, mask in self.masks.items():
            voter.add(self.cameras[view], mask)
            added[view] = mask

            assert np.array_equal(voter.selection(), _reference_vote(self.cameras, self.positions, added))

    def test_vote_selection_is_the_same_rule(self):
        assert np.array_equal(
            vote_selection(self.cameras, self.positions, self.masks),
            _reference_vote(self.cameras, self.positions, self.masks),
        )


class TestSceneGeometry:
    def setup_method(self):
        self.positions, self.is_a = two_spheres()
        self.cameras = ring()

    def _run(self, geometry=None, **kwargs):
        return segment_object(self.positions, self.cameras, click_on_a(), StandInSam(self.cameras), geometry=geometry, **kwargs)

    def test_a_second_click_reuses_the_planning_work_and_gets_the_same_answer(self):
        geometry = SceneGeometry(self.positions, self.cameras)

        first = self._run(geometry)
        misses_after_first = geometry.planning_misses
        second = self._run(geometry)

        assert geometry.planning_misses == misses_after_first and geometry.planning_hits > 0
        assert np.array_equal(first.selected, second.selected)

    def test_a_shared_geometry_changes_nothing_about_the_answer(self):
        shared = SceneGeometry(self.positions, self.cameras)
        self._run(shared)  # warm it with another click first
        warm = self._run(shared, level=2)
        fresh = self._run(level=2)

        assert np.array_equal(warm.selected, fresh.selected)
        assert warm.views_used == fresh.views_used and warm.rounds == fresh.rounds

    def test_the_cache_stays_within_its_budget_without_changing_the_answer(self, monkeypatch):
        monkeypatch.setattr(sam_pick, "PLANNING_CACHE_BYTES", 1)  # the minimum capacity: four views
        small = SceneGeometry(self.positions, self.cameras)

        result = self._run(small)

        assert len(small._maps) <= 4
        assert np.array_equal(result.selected, self._run().selected)

    def test_a_geometry_built_for_other_gaussians_is_refused(self):
        other = SceneGeometry(self.positions[:100], self.cameras)

        with pytest.raises(ValueError):
            self._run(other)

    def test_each_view_is_voted_on_once_however_many_rounds_run(self, monkeypatch):
        added = []
        original = Voter.add
        monkeypatch.setattr(Voter, "add", lambda self, camera, mask: (added.append(id(camera)), original(self, camera, mask))[1])

        result = self._run()

        assert result.rounds > 1
        assert len(added) == len(result.views_used) == len(set(added))


class TestNearPoints:
    def test_finds_every_point_within_the_radius_and_none_far_beyond_it(self):
        rng = np.random.default_rng(4)
        points, anchors = rng.uniform(0, 10, size=(400, 3)), rng.uniform(0, 10, size=(60, 3))
        radius = 0.8
        nearest = np.linalg.norm(points[:, None, :] - anchors[None, :, :], axis=2).min(axis=1)

        near = near_points(points, anchors, radius)

        assert near[nearest <= radius].all()
        assert not near[nearest > 2 * np.sqrt(3) * radius].any()  # the grid can be off by at most a cell diagonal

    def test_nothing_is_near_when_there_are_no_anchors_or_no_radius(self):
        points = np.random.default_rng(0).uniform(size=(20, 3))

        assert not near_points(points, np.zeros((0, 3)), 1.0).any()
        assert not near_points(points, points, 0.0).any()


class TestHullNeighbourhood:

    def _voter(self):
        rng = np.random.default_rng(2)
        object_ = rng.uniform(-1, 1, size=(50, 3))
        near = object_ + 0.01  # a hair from a visibly selected Gaussian
        far = rng.uniform(-1, 1, size=(50, 3)) * 0.05 + [0.0, 0.0, 6.0]
        other = rng.uniform(-1, 1, size=(50, 3)) * 0.05 + [6.0, 0.0, 0.0]
        voter = Voter(np.vstack([object_, near, far, other]).astype(np.float32))
        group = np.repeat(np.arange(4), 50)
        voter._seen[group == 0], voter._inside[group == 0] = 3, 3
        voter._framed[group >= 1], voter._inside_framed[group >= 1] = 10, 10
        voter._inside_framed[group == 3] = 2  # inside the mask in only a fifth of its views: not a hull point
        return voter, group

    def test_unrestricted_the_rule_takes_every_hidden_gaussian_inside_the_masks(self):
        voter, group = self._voter()

        selected = voter.selection()

        assert selected[group == 1].all() and selected[group == 2].all() and not selected[group == 3].any()

    def test_a_neighbourhood_keeps_the_ones_next_to_the_object_and_drops_the_far_ones(self):
        voter, group = self._voter()

        selected = voter.selection(neighbourhood=0.05)

        assert selected[group == 0].all()
        assert selected[group == 1].all() and not selected[group == 2].any()

    def test_the_visible_selection_is_the_same_with_and_without_a_neighbourhood(self):
        voter, group = self._voter()

        assert np.array_equal(voter.selection()[group == 0], voter.selection(neighbourhood=0.02)[group == 0])

    def test_without_a_visible_selection_no_hidden_gaussian_is_kept(self):
        voter, group = self._voter()
        voter._seen[:] = 0  # nothing is visibly selected any more

        assert not voter.selection(neighbourhood=0.5).any()

    def test_the_call_and_the_module_setting_decide_what_the_driver_asks_for(self, monkeypatch):
        positions, _ = two_spheres()
        cameras = ring()
        asked = []
        original = Voter.selection
        monkeypatch.setattr(Voter, "selection", lambda self, min_ratio=0.5, neighbourhood=None: (asked.append(neighbourhood), original(self, min_ratio, neighbourhood))[1])

        segment_object(positions, cameras, click_on_a(), StandInSam(cameras), max_rounds=1)
        monkeypatch.setattr(sam_pick, "HULL_NEIGHBOURHOOD", 0.03)
        segment_object(positions, cameras, click_on_a(), StandInSam(cameras), max_rounds=1)
        segment_object(positions, cameras, click_on_a(), StandInSam(cameras), max_rounds=1, hull_neighbourhood=0.2)

        assert asked == [0.01, 0.03, 0.2]  # the default, the module setting, the call

    def test_infinity_asks_for_the_rule_as_it_was_and_none_falls_back_to_the_default(self, monkeypatch):
        positions, _ = two_spheres()
        cameras = ring()
        asked = []
        original = Voter.selection
        monkeypatch.setattr(Voter, "selection", lambda self, min_ratio=0.5, neighbourhood=None: (asked.append(neighbourhood), original(self, min_ratio, neighbourhood))[1])

        segment_object(positions, cameras, click_on_a(), StandInSam(cameras), max_rounds=1, hull_neighbourhood=math.inf)
        monkeypatch.setattr(sam_pick, "HULL_NEIGHBOURHOOD", None)
        segment_object(positions, cameras, click_on_a(), StandInSam(cameras), max_rounds=1)

        assert asked == [None, None]

    def test_the_default_is_the_measured_one(self):
        assert sam_pick.HULL_NEIGHBOURHOOD == 0.01


class LeakySam(StandInSam):

    def __init__(self, cameras, leaks=True):
        super().__init__(cameras)
        self.leaks = leaks
        self.log = []

    def __call__(self, view, points, negatives=None):
        self.calls.append(view)
        negatives = np.zeros((0, 2)) if negatives is None else negatives
        self.log.append((view, len(points), len(negatives)))
        camera = self.cameras[view]
        owner = self._discs(camera)

        def under(pixels):
            return {owner[int(round(y)), int(round(x))] for x, y in pixels} - {""}

        wanted = under(points)
        if self.leaks and len(points) == 1 and len(negatives) == 0:
            wanted = {"A", "B"}
        wanted -= under(negatives)
        empty = np.zeros((camera.height, camera.width), bool)
        if not wanted:
            return np.stack([empty] * (3 if len(points) + len(negatives) == 1 else 1))
        mask = np.isin(owner, sorted(wanted))
        if len(points) + len(negatives) == 1:
            return np.stack([mask & (owner == owner[int(round(points[0][1])), int(round(points[0][0]))]), mask, owner != ""])
        return mask[None]


def top_of(name):
    return CENTERS[name] + np.array([0.0, 0.0, SPHERE_RADIUS])


class TestNegativeAndExtraPoints:
    def setup_method(self):
        self.positions, self.is_a = two_spheres()
        self.cameras = ring()

    def _run(self, sam=None, cameras=None, **kwargs):
        cameras = cameras or self.cameras
        sam = sam or LeakySam(cameras)
        return segment_object(self.positions, cameras, click_on_a(), sam, level=1, **kwargs), sam

    def test_the_leaking_answer_takes_the_neighbour_along(self):
        result, _ = self._run()

        assert result.selected[~self.is_a].mean() > 0.5  # the fault the next tests correct

    def test_a_negative_point_that_most_views_can_see_takes_the_neighbour_out_again(self):
        # Cameras high enough to see the top of the neighbour in every view.
        high = ring(height=3.0)

        result, _ = self._run(cameras=high, negative_points=np.array([top_of("B")]))

        assert result.selected[~self.is_a].mean() < 0.05
        assert result.selected[self.is_a].mean() > 0.8

    def test_further_points_of_the_object_settle_the_ambiguity_where_the_negative_is_hidden(self):
        extra = np.array([CENTERS["A"] + [0.0, 0.0, 1.0], CENTERS["A"] + [0.0, -1.0, 0.0]])

        negative_alone, _ = self._run(negative_points=np.array([top_of("B")]))
        settled, _ = self._run(negative_points=np.array([top_of("B")]), extra_points=extra)

        assert negative_alone.selected[~self.is_a].mean() > 0.5
        assert settled.selected[~self.is_a].mean() < 0.05
        assert settled.selected[self.is_a].mean() > 0.9

    def test_extra_points_are_prompted_together_with_the_click_in_the_first_round(self):
        extra = np.array([CENTERS["A"] + [0.0, 0.0, 1.0]])
        sam = LeakySam(self.cameras, leaks=False)

        self._run(sam=sam, extra_points=extra)

        assert any(points == 2 for _, points, _ in sam.log)  # the click and the extra point, together

    def test_the_negative_point_reaches_sam_only_in_the_views_that_can_see_it(self):
        _, sam = self._run(negative_points=np.array([top_of("B")]))

        with_negatives = [view for view, _, negatives in sam.log if negatives]
        without = [view for view, _, negatives in sam.log if not negatives]
        assert with_negatives  # it was used
        assert without  # and skipped where the neighbour hides it
        assert all(negatives <= 1 for _, _, negatives in sam.log)

    def test_without_them_the_predictor_is_called_exactly_as_before(self):
        seen = []
        honest = StandInSam(self.cameras)

        def two_arguments_only(view, points):  # a predictor that knows nothing of negatives
            seen.append(len(points))
            return honest(view, points)

        result = segment_object(self.positions, self.cameras, click_on_a(), two_arguments_only, level=1)

        assert seen and result.selected[self.is_a].mean() > 0.9

    def test_spread_points_returns_given_points_in_three_dimensions_too(self):
        cloud = np.random.default_rng(1).uniform(size=(500, 3))

        chosen = spread_points(cloud, 4)

        assert chosen.shape == (4, 3)
        assert all((cloud == row).all(axis=1).any() for row in chosen)
