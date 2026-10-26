from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .semantic_segmentation import Camera

LEVEL_NAMES = ("Part", "Object", "Whole")
DEFAULT_LEVEL = 1

MAX_CANDIDATE_VIEWS = 90
MAX_PLANNING_POINTS = 300_000
PLANNING_CACHE_BYTES = 600 * 2**20
MIN_SELECTED_POINTS_IN_VIEW = 30
MIN_MASK_RECALL = 0.8
MIN_USABLE_RECALL = 0.5
MIN_RELATIVE_AREA = 0.25
GROWTH_POINTS = 5
DEPTH_SLACK_CELLS = 3.0
TARGET_POINTS_PER_CELL = 8.0
HULL_MIN_VIEWS = 3
HULL_RATIO = 0.9
MAJORITY = 0.5
INCLUSIVE = 0.3
HULL_NEIGHBOURHOOD: float | None = 0.01
MIN_HULL_ANCHORS = 10
CONVERGED_FRACTION = 0.001

Predictor = Callable[..., np.ndarray]


@dataclass
class ReferenceGeometry:

    vertices: np.ndarray
    faces: np.ndarray
    gaussian_face_idx: np.ndarray
    positions: np.ndarray

    @classmethod
    def capture(cls, package) -> "ReferenceGeometry":
        return cls(
            vertices=np.array(package.mesh_vertices, dtype=np.float32),
            faces=np.array(package.mesh_faces),
            gaussian_face_idx=np.array(package.gaussian_face_idx),
            positions=np.array(package.recompute_gaussian_positions(), dtype=np.float32),
        )

    def maps(self, package) -> bool:
        vertex_origin = getattr(package, "vertex_origin", None)
        face_origin = getattr(package, "face_origin", None)
        if vertex_origin is None or face_origin is None:
            return False
        if vertex_origin.shape[0] != package.mesh_vertices.shape[0] or face_origin.shape[0] != package.mesh_faces.shape[0]:
            return False
        return (
            (vertex_origin.size == 0 or (vertex_origin.min() >= 0 and vertex_origin.max() < self.vertices.shape[0]))
            and (face_origin.size == 0 or (face_origin.min() >= 0 and face_origin.max() < self.faces.shape[0]))
        )

    def point_of(self, package, vertex_index: int) -> np.ndarray:
        return self.vertices[package.vertex_origin[vertex_index]]

    def selection_in(self, package, selected: np.ndarray) -> np.ndarray | None:
        faces_hit = np.unique(self.gaussian_face_idx[np.asarray(selected, dtype=bool)])
        if faces_hit.size == 0:
            return None
        current = np.isin(package.face_origin, faces_hit)
        if not current.any():
            return None
        return np.unique(package.mesh_faces[current].reshape(-1))


@dataclass
class SegmentationResult:
    selected: np.ndarray
    views_used: list[int]
    rounds: int


def project(camera: Camera, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cam_space = points @ camera.rotation.T + camera.translation
    depth = cam_space[:, 2]
    in_front = depth > 1e-6
    safe = np.where(in_front, depth, 1.0)
    x = camera.fx * cam_space[:, 0] / safe + camera.cx
    y = camera.fy * cam_space[:, 1] / safe + camera.cy
    in_frame = in_front & (x >= 0) & (x < camera.width) & (y >= 0) & (y < camera.height)
    return np.stack([x, y], axis=1), depth, in_frame


def _visible_pixels(geometry: "SceneGeometry", view: int, points: np.ndarray | None) -> np.ndarray:
    if points is None or len(points) == 0:
        return np.zeros((0, 2))
    pixels, depth, in_frame = project(geometry.cameras[view], np.asarray(points, dtype=np.float64).reshape(-1, 3))
    return pixels[geometry.planning_map(view).visible(pixels, depth, in_frame, neighbourhood=1)]


class _DepthMap:

    def __init__(self, camera: Camera, points: np.ndarray):
        self.camera = camera
        self.pixels, self.depth, self.in_frame = project(camera, points)
        idx = np.nonzero(self.in_frame)[0]
        self.cell = self._cell_size(camera, self.pixels[idx])
        self._nx = camera.width // self.cell + 1
        self._ny = camera.height // self.cell + 1
        self._zbuf = np.full(self._nx * self._ny, np.inf)
        np.minimum.at(self._zbuf, self._cells(self.pixels[idx]), self.depth[idx])
        self._widened: dict[int, np.ndarray] = {}
        self._visible: dict[int, np.ndarray] = {}

    @staticmethod
    def _cell_size(camera: Camera, pixels: np.ndarray) -> int:
        longest = max(camera.width, camera.height)
        if pixels.shape[0] < 2:
            return max(2, longest // 160)
        sample = pixels[:: max(1, pixels.shape[0] // 50_000)]
        low, high = np.percentile(sample, [1, 99], axis=0)
        covered_area = max(float(np.prod(high - low)), 1.0)
        cell = int(np.ceil(np.sqrt(TARGET_POINTS_PER_CELL * covered_area / pixels.shape[0])))
        return int(np.clip(cell, 2, max(2, longest // 16)))

    def _cells(self, pixels: np.ndarray) -> np.ndarray:
        return (pixels[:, 1] // self.cell).astype(np.int64) * self._nx + (pixels[:, 0] // self.cell).astype(np.int64)

    def _nearest_depths(self, neighbourhood: int) -> np.ndarray:
        if neighbourhood == 0:
            return self._zbuf
        if neighbourhood not in self._widened:
            grid = self._zbuf.reshape(self._ny, self._nx)
            padded = np.pad(grid, neighbourhood, constant_values=np.inf)
            widened = np.full_like(grid, np.inf)
            for dy in range(2 * neighbourhood + 1):
                for dx in range(2 * neighbourhood + 1):
                    widened = np.minimum(widened, padded[dy:dy + self._ny, dx:dx + self._nx])
            self._widened[neighbourhood] = widened.ravel()
        return self._widened[neighbourhood]

    def visible(self, pixels: np.ndarray, depth: np.ndarray, in_frame: np.ndarray, neighbourhood: int = 0) -> np.ndarray:
        result = np.zeros(pixels.shape[0], dtype=bool)
        idx = np.nonzero(in_frame)[0]
        if idx.size:
            nearest = self._nearest_depths(neighbourhood)[self._cells(pixels[idx])]
            slack = DEPTH_SLACK_CELLS * self.cell * nearest / self.camera.fx
            result[idx] = depth[idx] <= nearest + slack
        return result

    def visible_points(self, neighbourhood: int = 0) -> np.ndarray:
        if neighbourhood not in self._visible:
            self._visible[neighbourhood] = self.visible(self.pixels, self.depth, self.in_frame, neighbourhood)
        return self._visible[neighbourhood]

    def release_point_arrays(self) -> None:
        self.depth = self.in_frame = None


def evenly_spaced(items: list, count: int) -> list:
    if len(items) <= count:
        return list(items)
    picks = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in sorted(set(picks.tolist()))]


def _planning_subset(n_points: int) -> np.ndarray:
    if n_points <= MAX_PLANNING_POINTS:
        return np.arange(n_points)
    return np.sort(np.random.default_rng(0).choice(n_points, MAX_PLANNING_POINTS, replace=False))


class SceneGeometry:

    def __init__(self, positions: np.ndarray, cameras: list[Camera]):
        self.positions = np.asarray(positions, dtype=np.float32)
        self.cameras = cameras
        self.subset_index = _planning_subset(self.positions.shape[0])
        self.planning_positions = self.positions[self.subset_index]
        per_view = self.planning_positions.shape[0] * 17 + 4096
        self._capacity = max(4, PLANNING_CACHE_BYTES // per_view)
        self._maps: OrderedDict[int, _DepthMap] = OrderedDict()
        self.planning_hits = 0
        self.planning_misses = 0

    def planning_map(self, view: int) -> _DepthMap:
        cached = self._maps.get(view)
        if cached is not None:
            self._maps.move_to_end(view)
            self.planning_hits += 1
            return cached
        depth_map = _DepthMap(self.cameras[view], self.planning_positions)
        depth_map.visible_points(neighbourhood=1)
        depth_map.release_point_arrays()
        self._maps[view] = depth_map
        while len(self._maps) > self._capacity:
            self._maps.popitem(last=False)
        self.planning_misses += 1
        return depth_map


def initial_prompts(
    cameras: list[Camera], positions: np.ndarray, point: np.ndarray, max_views: int, geometry: SceneGeometry | None = None
) -> dict[int, np.ndarray]:
    geometry = geometry or SceneGeometry(positions, cameras)
    query = np.asarray(point, dtype=np.float64).reshape(1, 3)
    seeing: dict[int, np.ndarray] = {}
    for view in evenly_spaced(list(range(len(cameras))), MAX_CANDIDATE_VIEWS):
        pixels, depth, in_frame = project(cameras[view], query)
        if geometry.planning_map(view).visible(pixels, depth, in_frame, neighbourhood=1)[0]:
            seeing[view] = pixels[:1].copy()
    return {v: seeing[v] for v in evenly_spaced(sorted(seeing), max_views)}


def spread_points(pixels: np.ndarray, count: int) -> np.ndarray:
    if pixels.shape[0] <= count:
        return pixels.copy()
    rng = np.random.default_rng(0)
    sample = pixels[rng.choice(pixels.shape[0], min(pixels.shape[0], 2000), replace=False)]
    centres = [sample[np.argmin(((sample - np.median(sample, axis=0)) ** 2).sum(axis=1))]]
    while len(centres) < count:
        nearest = np.min([((sample - c) ** 2).sum(axis=1) for c in centres], axis=0)
        centres.append(sample[np.argmax(nearest)])
    centres = np.array(centres)
    for _ in range(4):
        owner = np.argmin(((sample[:, None, :] - centres[None]) ** 2).sum(axis=2), axis=1)
        for j in range(count):
            if np.any(owner == j):
                centres[j] = sample[owner == j].mean(axis=0)
    return np.array([sample[np.argmin(((sample - c) ** 2).sum(axis=1))] for c in centres])


def growth_prompts(
    cameras: list[Camera],
    positions: np.ndarray,
    selected: np.ndarray,
    exclude: set[int],
    max_views: int,
    geometry: SceneGeometry | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    geometry = geometry or SceneGeometry(positions, cameras)
    subset_selected = selected[geometry.subset_index]
    minimum = max(MIN_SELECTED_POINTS_IN_VIEW, int(0.005 * subset_selected.sum()))

    qualifying: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for view in evenly_spaced(list(range(len(cameras))), MAX_CANDIDATE_VIEWS):
        if view in exclude:
            continue
        depth_map = geometry.planning_map(view)
        on_selection = depth_map.visible_points(neighbourhood=1) & subset_selected
        if on_selection.sum() < minimum:
            continue
        pixels = depth_map.pixels[on_selection]
        qualifying[view] = (spread_points(pixels, GROWTH_POINTS), pixels)
    return {v: qualifying[v] for v in evenly_spaced(sorted(qualifying), max_views)}


def pick_level_mask(candidates: np.ndarray, level: int) -> np.ndarray:
    """The candidate of rank `level` by area (0 smallest)."""
    areas = candidates.reshape(candidates.shape[0], -1).sum(axis=1)
    order = np.argsort(areas, kind="stable")
    return candidates[order[min(max(level, 0), len(order) - 1)]]


def pick_fitting_mask(
    candidates: np.ndarray, selected_pixels: np.ndarray, camera: Camera, level: int = DEFAULT_LEVEL
) -> np.ndarray | None:
    height, width = candidates.shape[1:]
    x = np.clip((selected_pixels[:, 0] * width / camera.width).astype(int), 0, width - 1)
    y = np.clip((selected_pixels[:, 1] * height / camera.height).astype(int), 0, height - 1)
    recall = np.array([m[y, x].mean() for m in candidates])
    order = np.argsort(candidates.reshape(candidates.shape[0], -1).sum(axis=1), kind="stable")
    for rank in range(min(max(level, 0), len(order) - 1), len(order)):
        if recall[order[rank]] >= MIN_MASK_RECALL:
            return candidates[order[rank]]
    best = int(np.argmax(recall))
    return candidates[best] if recall[best] >= MIN_USABLE_RECALL else None


def near_points(points: np.ndarray, anchors: np.ndarray, radius: float) -> np.ndarray:
    near = np.zeros(points.shape[0], dtype=bool)
    if radius <= 0 or anchors.shape[0] == 0 or points.shape[0] == 0:
        return near
    origin = np.minimum(points.min(axis=0), anchors.min(axis=0)) - radius
    cells_anchor = np.floor((anchors - origin) / radius).astype(np.int64) + 1
    cells_point = np.floor((points - origin) / radius).astype(np.int64) + 1
    size = np.maximum(cells_anchor.max(axis=0), cells_point.max(axis=0)) + 3

    def key(cells: np.ndarray) -> np.ndarray:
        return (cells[:, 0] * size[1] + cells[:, 1]) * size[2] + cells[:, 2]

    occupied = np.unique(key(cells_anchor))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                wanted = key(cells_point + np.array([dx, dy, dz]))
                position = np.minimum(np.searchsorted(occupied, wanted), occupied.shape[0] - 1)
                near |= occupied[position] == wanted
    return near


class Voter:

    def __init__(self, positions: np.ndarray):
        self.positions = positions
        n_points = positions.shape[0]
        self._seen = np.zeros(n_points, dtype=np.int32)
        self._inside = np.zeros(n_points, dtype=np.int32)
        self._framed = np.zeros(n_points, dtype=np.int32)
        self._inside_framed = np.zeros(n_points, dtype=np.int32)

    def add(self, camera: Camera, mask: np.ndarray) -> None:
        depth_map = _DepthMap(camera, self.positions)
        in_frame = np.nonzero(depth_map.in_frame)[0]
        height, width = mask.shape
        x = np.clip((depth_map.pixels[in_frame, 0] * width / camera.width).astype(int), 0, width - 1)
        y = np.clip((depth_map.pixels[in_frame, 1] * height / camera.height).astype(int), 0, height - 1)
        in_mask = mask[y, x]
        visible = depth_map.visible_points()[in_frame]

        self._framed[in_frame] += 1
        self._inside_framed[in_frame] += in_mask
        self._seen[in_frame[visible]] += 1
        self._inside[in_frame[visible]] += in_mask[visible]

    def selection(self, min_ratio: float = 0.5, neighbourhood: float | None = None) -> np.ndarray:
        seen_and_inside = (self._seen > 0) & (self._inside > min_ratio * self._seen)
        inside_hull = (
            (self._seen == 0) & (self._framed >= HULL_MIN_VIEWS) & (self._inside_framed >= HULL_RATIO * self._framed)
        )
        if neighbourhood is not None and inside_hull.any():
            inside_hull = self._near_the_visible_selection(inside_hull, seen_and_inside, neighbourhood)
        return seen_and_inside | inside_hull

    def _near_the_visible_selection(self, candidates: np.ndarray, visible: np.ndarray, fraction: float) -> np.ndarray:
        if int(visible.sum()) < MIN_HULL_ANCHORS:
            return np.zeros_like(candidates)
        anchors = self.positions[visible]
        low, high = np.percentile(anchors, [1, 99], axis=0)
        kept = np.zeros_like(candidates)
        kept[candidates] = near_points(self.positions[candidates], anchors, fraction * float(np.linalg.norm(high - low)))
        return kept


def vote_selection(
    cameras: list[Camera],
    positions: np.ndarray,
    masks: dict[int, np.ndarray],
    min_ratio: float = 0.5,
    neighbourhood: float | None = None,
) -> np.ndarray:
    voter = Voter(positions)
    for view, mask in masks.items():
        voter.add(cameras[view], mask)
    return voter.selection(min_ratio, neighbourhood)


def _drop_undersized(masks: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    areas = {view: int(mask.sum()) for view, mask in masks.items()}
    typical = float(np.median(list(areas.values()))) if len(areas) >= 3 else 0.0
    return {v: m for v, m in masks.items() if areas[v] > 0 and areas[v] >= MIN_RELATIVE_AREA * typical}


def segment_object(
    positions: np.ndarray,
    cameras: list[Camera],
    point: np.ndarray,
    predict: Predictor,
    level: int = DEFAULT_LEVEL,
    initial_views: int = 6,
    growth_views: int = 8,
    max_rounds: int = 4,
    max_views: int = 36,
    progress: Callable[[str], None] | None = None,
    geometry: SceneGeometry | None = None,
    hull_neighbourhood: float | None = None,
    extra_points: np.ndarray | None = None,
    negative_points: np.ndarray | None = None,
    prefetch: Callable[[list[int]], None] | None = None,
    min_ratio: float = MAJORITY,
) -> SegmentationResult:
    if geometry is None:
        geometry = SceneGeometry(positions, cameras)
    elif geometry.positions.shape != np.shape(positions):
        raise ValueError("geometry was built for a different set of Gaussians")
    positions = geometry.positions
    neighbourhood = HULL_NEIGHBOURHOOD if hull_neighbourhood is None else hull_neighbourhood
    if neighbourhood is not None and not np.isfinite(neighbourhood):
        neighbourhood = None
    say = progress or (lambda _message: None)
    voter = Voter(positions)
    used: list[int] = []
    tried: set[int] = set()
    selected: np.ndarray | None = None
    rounds = 0

    for round_index in range(max_rounds):
        room = max_views - len(tried)
        if room <= 0:
            break
        if round_index == 0:
            first = initial_prompts(cameras, positions, point, min(initial_views, room), geometry)
            plan = {v: (points, None) for v, points in first.items()}
        else:
            plan = growth_prompts(cameras, positions, selected, tried, min(growth_views, room), geometry)
        if not plan:
            break
        rounds += 1
        say(f"round {rounds}: prompting {len(plan)} view(s)")
        if prefetch is not None:
            prefetch(list(plan))

        usable: dict[int, np.ndarray] = {}
        for view, (points, selected_pixels) in plan.items():
            tried.add(view)
            if selected_pixels is None:
                points = np.vstack([points, _visible_pixels(geometry, view, extra_points)])
            negatives = _visible_pixels(geometry, view, negative_points)
            candidates = predict(view, points) if len(negatives) == 0 else predict(view, points, negatives)
            mask = (
                pick_level_mask(candidates, level)
                if selected_pixels is None
                else pick_fitting_mask(candidates, selected_pixels, cameras[view], level)
            )
            if mask is not None:
                usable[view] = mask
        for view, mask in (_drop_undersized(usable) if round_index == 0 else usable).items():
            voter.add(cameras[view], mask)
            used.append(view)
        if not used:
            break

        new_selected = voter.selection(min_ratio=min_ratio, neighbourhood=neighbourhood)
        changed = selected is None or int((new_selected != selected).sum()) > CONVERGED_FRACTION * positions.shape[0]
        selected = new_selected
        skipped = len(plan) - len(usable)
        say(
            f"round {rounds}: {int(selected.sum()):,} Gaussians selected from {len(used)} view(s)"
            + (f", {skipped} skipped (mask did not fit)" if skipped else "")
        )
        if not selected.any() or not changed:
            break

    if selected is None:
        selected = np.zeros(positions.shape[0], dtype=bool)
    return SegmentationResult(selected=selected, views_used=sorted(used), rounds=rounds)
