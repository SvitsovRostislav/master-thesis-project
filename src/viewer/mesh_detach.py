from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetachResult:
    selection: np.ndarray
    faces_detached: int
    vertices_duplicated: int
    cut_originals: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    cut_copies: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))


def inside_faces(faces: np.ndarray, vertex_indices, n_vertices: int) -> np.ndarray:
    selected = np.zeros(n_vertices, dtype=bool)
    selected[np.asarray(vertex_indices, dtype=np.int64)] = True
    return selected[faces].all(axis=1)


def stretches_on_move(faces: np.ndarray, vertex_indices, n_vertices: int) -> bool:
    selected = np.zeros(n_vertices, dtype=bool)
    selected[np.asarray(vertex_indices, dtype=np.int64)] = True
    corners = selected[faces]
    return bool((corners.any(axis=1) & ~corners.all(axis=1)).any())


def detach_selection(package, vertex_indices) -> DetachResult | None:
    n_vertices = package.mesh_vertices.shape[0]
    faces = package.mesh_faces
    inside = inside_faces(faces, vertex_indices, n_vertices)
    if not inside.any():
        return None

    used_inside = np.zeros(n_vertices, dtype=bool)
    used_inside[faces[inside].reshape(-1)] = True
    used_outside = np.zeros(n_vertices, dtype=bool)
    used_outside[faces[~inside].reshape(-1)] = True
    on_cut = np.nonzero(used_inside & used_outside)[0]

    if on_cut.size:
        copy_of = np.arange(n_vertices, dtype=np.int64)
        copy_of[on_cut] = n_vertices + np.arange(on_cut.size)
        new_faces = faces.copy()
        new_faces[inside] = copy_of[faces[inside]].astype(faces.dtype)
        package.mesh_faces = new_faces
        package.mesh_vertices = np.concatenate([package.mesh_vertices, package.mesh_vertices[on_cut]])
        if package.vertex_origin is not None:
            package.vertex_origin = np.concatenate([package.vertex_origin, package.vertex_origin[on_cut]])

    selection = np.unique(package.mesh_faces[inside].reshape(-1))
    return DetachResult(selection=selection, faces_detached=int(inside.sum()), vertices_duplicated=int(on_cut.size),
                        cut_originals=on_cut.astype(np.int64), cut_copies=n_vertices + np.arange(on_cut.size, dtype=np.int64))
