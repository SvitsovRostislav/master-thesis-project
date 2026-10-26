from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Camera:

    rotation: np.ndarray
    translation: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def _project(camera: Camera, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cam_space = positions @ camera.rotation.T + camera.translation
    in_front = cam_space[:, 2] > 1e-6
    z = np.where(in_front, cam_space[:, 2], 1.0)
    x = camera.fx * cam_space[:, 0] / z + camera.cx
    y = camera.fy * cam_space[:, 1] / z + camera.cy
    in_bounds = (x >= 0) & (x < camera.width) & (y >= 0) & (y < camera.height)
    return np.stack([x, y], axis=1), in_front & in_bounds


def vote_gaussian_labels(
    positions: np.ndarray, cameras: list[Camera], masks: list[np.ndarray]
) -> np.ndarray:
    n_gaussians = positions.shape[0]
    max_label = max((int(m.max()) for m in masks if m.size), default=0)
    votes = np.zeros((n_gaussians, max_label + 1), dtype=np.int32)

    for camera, mask in zip(cameras, masks):
        pixel_xy, visible = _project(camera, positions)
        if not np.any(visible):
            continue
        px = pixel_xy[visible, 0].astype(np.int64)
        py = pixel_xy[visible, 1].astype(np.int64)
        labels_here = mask[py, px]
        gaussian_indices = np.nonzero(visible)[0]
        np.add.at(votes, (gaussian_indices, labels_here), 1)

    return votes.argmax(axis=1).astype(np.int32)


def label_at_vertex(
    mesh_faces: np.ndarray, gaussian_face_idx: np.ndarray, semantic_id: np.ndarray, vertex_index: int
) -> int:
    touching_faces = np.nonzero((mesh_faces == vertex_index).any(axis=1))[0]
    on_those_faces = np.isin(gaussian_face_idx, touching_faces)
    labels_here = semantic_id[on_those_faces]
    labels_here = labels_here[labels_here != 0]
    if labels_here.size == 0:
        return 0
    values, counts = np.unique(labels_here, return_counts=True)
    return int(values[np.argmax(counts)])


def labels_to_vertex_selection(
    mesh_faces: np.ndarray, gaussian_face_idx: np.ndarray, semantic_id: np.ndarray, target_label: int
) -> np.ndarray | None:
    if target_label == 0:
        return None
    matching_faces = np.unique(gaussian_face_idx[semantic_id == target_label])
    if matching_faces.size == 0:
        return None
    return np.unique(mesh_faces[matching_faces].reshape(-1))


def semantic_labels_path(npz_path) -> Path:
    npz_path = Path(npz_path)
    return npz_path.with_name(npz_path.stem + "_semantic_labels.npy")


def load_semantic_labels(npz_path, expected_count: int) -> np.ndarray | None:
    path = semantic_labels_path(npz_path)
    if not path.is_file():
        return None
    labels = np.load(path)
    if labels.shape[0] != expected_count:
        return None
    return labels.astype(np.int32)
