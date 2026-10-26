from dataclasses import dataclass

import numpy as np

from .digital_twin import DigitalTwinPackage

_RNG = np.random.default_rng()

MAX_GROWTH_RATIO = 8.0
MAX_TOTAL_GROWTH_RATIO = 8.0


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def _random_barycentric(n: int) -> np.ndarray:
    r = _RNG.random((n, 2)).astype(np.float32)
    flip = r.sum(axis=1) > 1.0
    r[flip] = 1.0 - r[flip]
    u, v = r[:, 0], r[:, 1]
    return np.stack([1.0 - u - v, u, v], axis=1)


def _face_gaussian_groups(gaussian_face_idx: np.ndarray, n_faces: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(gaussian_face_idx, kind="stable")
    sorted_faces = gaussian_face_idx[order]
    ptr = np.searchsorted(sorted_faces, np.arange(n_faces + 1))
    return ptr, order


@dataclass
class DensityBaseline:

    area: np.ndarray
    count: np.ndarray

    @classmethod
    def capture(cls, package: DigitalTwinPackage) -> "DensityBaseline":
        n_faces = package.mesh_faces.shape[0]
        area = triangle_areas(package.mesh_vertices, package.mesh_faces).astype(np.float32)
        count = np.bincount(package.gaussian_face_idx, minlength=n_faces).astype(np.int64)
        return cls(area=area, count=count)


def rebalance_density(package: DigitalTwinPackage, baseline: DensityBaseline) -> bool:
    n_faces = package.mesh_faces.shape[0]
    if baseline.area.shape[0] != n_faces:
        return False

    current_area = triangle_areas(package.mesh_vertices, package.mesh_faces)
    current_count = np.bincount(package.gaussian_face_idx, minlength=n_faces)

    safe_baseline_area = np.where(baseline.area > 1e-12, baseline.area, 1e-12)
    stretch_ratio = current_area / safe_baseline_area
    uncapped_target = np.round(baseline.count * stretch_ratio).astype(np.int64)
    per_triangle_cap = np.ceil(baseline.count * MAX_GROWTH_RATIO).astype(np.int64)
    target_count = np.maximum(baseline.count, np.minimum(uncapped_target, per_triangle_cap))

    grow_faces = np.nonzero(target_count > current_count)[0]
    shrink_faces = np.nonzero(target_count < current_count)[0]

    grew = False
    if grow_faces.size > 0:
        n_new_per_face = _apply_global_growth_budget(
            n_new_per_face=target_count[grow_faces] - current_count[grow_faces],
            current_total=int(current_count.sum()),
            baseline_total=int(baseline.count.sum()),
        )
        grew = _grow(package, grow_faces, n_new_per_face)

    shrank = False
    if shrink_faces.size > 0:
        shrank = _shrink(package, shrink_faces, current_count[shrink_faces] - target_count[shrink_faces])

    return grew or shrank


def _apply_global_growth_budget(n_new_per_face: np.ndarray, current_total: int, baseline_total: int) -> np.ndarray:
    global_cap = int(np.ceil(baseline_total * MAX_TOTAL_GROWTH_RATIO))
    budget = max(global_cap - current_total, 0)
    total_requested = int(n_new_per_face.sum())
    if total_requested <= budget:
        return n_new_per_face
    if budget <= 0:
        return np.zeros_like(n_new_per_face)
    scale = budget / total_requested
    return np.floor(n_new_per_face * scale).astype(np.int64)


def _grow(package: DigitalTwinPackage, faces: np.ndarray, n_new_per_face: np.ndarray) -> bool:
    ptr, order = _face_gaussian_groups(package.gaussian_face_idx, package.mesh_faces.shape[0])
    face_idx, bary, colors, opacities, scales, quats = [], [], [], [], [], []
    for face, n_new in zip(faces, n_new_per_face):
        n_new = int(n_new)
        if n_new <= 0:
            continue
        donors_pool = order[ptr[face]:ptr[face + 1]]
        if donors_pool.size == 0:
            continue  # nothing on this triangle to interpolate appearance from
        donors = donors_pool[_RNG.integers(0, donors_pool.size, size=n_new)]
        face_idx.append(np.full(n_new, face, dtype=np.int32))
        bary.append(_random_barycentric(n_new))
        colors.append(package.gaussian_colors[donors])
        opacities.append(package.gaussian_opacities[donors])
        scales.append(package.gaussian_scales[donors])
        quats.append(package.gaussian_quaternions[donors])

    if not face_idx:
        return False
    package.add_gaussians(
        face_idx=np.concatenate(face_idx),
        bary=np.concatenate(bary),
        colors=np.concatenate(colors),
        opacities=np.concatenate(opacities),
        scales=np.concatenate(scales),
        quaternions=np.concatenate(quats),
    )
    return True


def _shrink(package: DigitalTwinPackage, faces: np.ndarray, n_remove_per_face: np.ndarray) -> bool:
    ptr, order = _face_gaussian_groups(package.gaussian_face_idx, package.mesh_faces.shape[0])
    doomed = []
    for face, n_remove in zip(faces, n_remove_per_face):
        n_remove = int(n_remove)
        if n_remove <= 0:
            continue
        candidates = order[ptr[face]:ptr[face + 1]]
        n_remove = min(n_remove, candidates.size)
        if n_remove == 0:
            continue
        doomed.append(_RNG.choice(candidates, size=n_remove, replace=False))

    if not doomed:
        return False
    package.remove_gaussians_by_index(np.concatenate(doomed))
    return True
