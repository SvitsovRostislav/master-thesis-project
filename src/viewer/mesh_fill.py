from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .gaussian_deformation import _matrices_to_quaternions, _quaternions_to_matrices

MIN_CUT_SHARE = 0.5


@dataclass
class FillResult:
    cycles_closed: int
    cycles_skipped: int
    cut_edges_open_before: int
    cut_edges_open_after: int
    faces_added: int
    vertices_added: int
    gaussians_added: int


def boundary_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    faces = np.asarray(faces, dtype=np.int64)
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    owner = np.concatenate([np.arange(faces.shape[0])] * 3)
    _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    single = counts[inverse.reshape(-1)] == 1
    return directed[single], owner[single]


def boundary_cycles(edges: np.ndarray) -> list[np.ndarray]:
    outgoing = defaultdict(list)
    for index, (start, _) in enumerate(edges):
        outgoing[int(start)].append(index)
    used = np.zeros(len(edges), dtype=bool)
    cycles = []
    for first in range(len(edges)):
        if used[first]:
            continue
        used[first] = True
        chain, origin, current = [first], int(edges[first][0]), first
        while int(edges[current][1]) != origin:
            options = [e for e in outgoing[int(edges[current][1])] if not used[e]]
            if not options:
                chain = None
                break
            current = options[0]
            used[current] = True
            chain.append(current)
        if chain is not None:
            cycles.append(np.array(chain, dtype=np.int64))
    return cycles


def min_area_triangulation(points: np.ndarray, forbidden: np.ndarray | None = None) -> np.ndarray | None:
    n = points.shape[0]
    if n < 3:
        return np.empty((0, 3), dtype=np.int64)
    p = points.astype(np.float64)
    cost = np.zeros((n, n))
    split = np.zeros((n, n), dtype=np.int64)
    for span in range(2, n):
        i = np.arange(0, n - span)
        j = i + span
        k = i[:, None] + np.arange(1, span)[None, :]
        area = 0.5 * np.linalg.norm(np.cross(p[k] - p[i][:, None], p[j][:, None] - p[i][:, None]), axis=2)
        total = cost[i[:, None], k] + cost[k, j[:, None]] + area
        best = np.argmin(total, axis=1)
        cost[i, j] = total[np.arange(i.size), best]
        split[i, j] = k[np.arange(i.size), best]
        if forbidden is not None and span < n - 1:
            cost[i[forbidden[i, j]], j[forbidden[i, j]]] = np.inf
    if not np.isfinite(cost[0, n - 1]):
        return None
    triangles, stack = [], [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        k = int(split[i, j])
        triangles.append((i, k, j))
        stack.extend([(i, k), (k, j)])
    return np.array(triangles, dtype=np.int64)


def simple_cycles(ring: np.ndarray) -> list[np.ndarray]:
    pending, done = [np.arange(ring.size)], []
    while pending:
        positions = pending.pop()
        seen = {}
        for place, vertex in enumerate(ring[positions]):
            if int(vertex) in seen:
                first = seen[int(vertex)]
                pending.append(positions[first:place])
                pending.append(np.concatenate([positions[:first], positions[place:]]))
                break
            seen[int(vertex)] = place
        else:
            if positions.size >= 3:
                done.append(positions)
    return done


class _Neighbours:

    def __init__(self, faces: np.ndarray, n_vertices: int):
        faces = np.asarray(faces, dtype=np.int64)
        a = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
        b = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
        order = np.argsort(a, kind="stable")
        self._targets = b[order]
        self._start = np.searchsorted(a[order], np.arange(n_vertices + 1))
        self._added: dict[int, set] = defaultdict(set)
        self._place = np.full(n_vertices, -1, dtype=np.int64)

    def add_triangle(self, face) -> None:
        for x, y in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            self._added[x].add(y)
            self._added[y].add(x)

    def among(self, ring: np.ndarray) -> np.ndarray:
        n = ring.size
        forbidden = np.zeros((n, n), dtype=bool)
        self._place[ring] = np.arange(n)
        for p, vertex in enumerate(ring):
            others = self._targets[self._start[vertex]:self._start[vertex + 1]]
            if vertex in self._added:
                others = np.concatenate([others, np.fromiter(self._added[vertex], dtype=np.int64)])
            q = self._place[others]
            q = q[q >= 0]
            forbidden[p, q] = True
            forbidden[q, p] = True
        self._place[ring] = -1
        np.fill_diagonal(forbidden, False)
        return forbidden


def _new_gaussians(package, faces_new: np.ndarray, across: np.ndarray, vertices: np.ndarray) -> dict:
    by_face = defaultdict(list)
    wanted = set(across.tolist())
    for index, face in enumerate(package.gaussian_face_idx):
        if int(face) in wanted:
            by_face[int(face)].append(index)
    fallback = np.arange(len(package))
    donors = [by_face.get(int(face)) or fallback[:1] for face in across]

    colors = np.stack([package.gaussian_colors[d].mean(axis=0) for d in donors])
    opacities = np.array([package.gaussian_opacities[d].mean() for d in donors])
    donor_scales = np.stack([package.gaussian_scales[d].mean(axis=0) for d in donors]).astype(np.float64)

    v = vertices.astype(np.float64)

    def geometry(tri):
        e1, e2 = v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]]
        cross = np.cross(e1, e2)
        return e1, np.linalg.norm(cross, axis=1), cross

    e1, twice_area, cross = geometry(faces_new)
    _, twice_area_across, _ = geometry(package.mesh_faces[across])
    flat = twice_area > 1e-12 * np.maximum(twice_area_across, 1e-30)
    normal = cross / np.maximum(twice_area, 1e-30)[:, None]
    t1 = e1 / np.maximum(np.linalg.norm(e1, axis=1), 1e-30)[:, None]
    t2 = np.cross(normal, t1)
    rotation = np.stack([t1, t2, normal], axis=-1)
    if not flat.all():
        donor_rotation = _quaternions_to_matrices(np.stack([package.gaussian_quaternions[d[0]] for d in donors]))
        rotation[~flat] = donor_rotation[~flat]

    ordered = np.sort(donor_scales, axis=1)
    in_plane = ordered[:, 1:].mean(axis=1) * np.sqrt(twice_area / np.maximum(twice_area_across, 1e-30))
    scales = np.stack([in_plane, in_plane, ordered[:, 0]], axis=1)
    return {
        "colors": colors.astype(np.float32),
        "opacities": opacities.astype(np.float32),
        "scales": scales.astype(np.float32),
        "quaternions": _matrices_to_quaternions(rotation).astype(np.float32),
    }


def close_cut(package, cut_vertices, min_cut_share: float = MIN_CUT_SHARE) -> FillResult:
    on_cut = np.zeros(package.mesh_vertices.shape[0], dtype=bool)
    on_cut[np.asarray(cut_vertices, dtype=np.int64)] = True
    edges, owner = boundary_edges(package.mesh_faces)
    edge_on_cut = on_cut[edges].all(axis=1)
    open_before = int(edge_on_cut.sum())

    n_vertices = package.mesh_vertices.shape[0]
    neighbours = _Neighbours(package.mesh_faces, n_vertices)

    new_faces, across = [], []
    closed = skipped = 0
    for cycle in boundary_cycles(edges):
        if not edge_on_cut[cycle].any():
            continue
        if edge_on_cut[cycle].mean() < min_cut_share:
            skipped += 1
            continue
        whole = edges[cycle][:, 0]
        for positions in simple_cycles(whole):
            ring = whole[positions]
            forbidden = neighbours.among(ring)
            triangles = min_area_triangulation(package.mesh_vertices[ring], forbidden)
            if triangles is None:
                skipped += 1
                continue
            for i, k, j in triangles:
                face = (int(ring[j]), int(ring[k]), int(ring[i]))
                new_faces.append(face)
                across.append(owner[cycle][positions[i]])
                neighbours.add_triangle(face)
            closed += 1

    if not new_faces:
        return FillResult(0, skipped, open_before, open_before, 0, 0, 0)

    faces_new = np.array(new_faces, dtype=np.int64)
    across = np.array(across, dtype=np.int64)
    look = _new_gaussians(package, faces_new, across, package.mesh_vertices)

    first_new_face = package.mesh_faces.shape[0]
    package.mesh_faces = np.concatenate([package.mesh_faces, faces_new.astype(package.mesh_faces.dtype)])
    if package.face_origin is not None:
        package.face_origin = np.concatenate([package.face_origin, package.face_origin[across]])
    count = faces_new.shape[0]
    package.add_gaussians(
        face_idx=np.arange(first_new_face, first_new_face + count),
        bary=np.full((count, 3), 1.0 / 3.0),
        colors=look["colors"],
        opacities=look["opacities"],
        scales=look["scales"],
        quaternions=look["quaternions"],
    )

    edges_after, _ = boundary_edges(package.mesh_faces)
    open_after = int(on_cut[edges_after].all(axis=1).sum())
    return FillResult(closed, skipped, open_before, open_after, count, 0, count)


def thin_axis_is_normal(package, sample: int = 2000) -> float:
    count = min(sample, len(package))
    rotation = _quaternions_to_matrices(package.gaussian_quaternions[:count])
    thin = rotation[np.arange(count), :, np.argmin(package.gaussian_scales[:count], axis=1)]
    v = package.mesh_vertices.astype(np.float64)
    f = package.mesh_faces[package.gaussian_face_idx[:count]]
    normal = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    return float(np.median(np.abs((thin * normal).sum(axis=1))))
