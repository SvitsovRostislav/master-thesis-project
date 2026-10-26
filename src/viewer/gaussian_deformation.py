from dataclasses import dataclass

import numpy as np

from .digital_twin import DigitalTwinPackage

MIN_CHANGE = 1e-3
MAX_STRETCH = 10.0
MIN_SIN_ANGLE = 1e-2

_CHUNK = 200_000
_IDENTITY = np.eye(3)


def face_frames(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = vertices.astype(np.float64)
    v0, v1, v2 = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    cross = np.cross(e1, e2)
    cross_norm = np.linalg.norm(cross, axis=1)
    edge_product = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
    valid = (edge_product > 1e-24) & (cross_norm > MIN_SIN_ANGLE * edge_product)
    normal = cross / np.where(valid, cross_norm, 1.0)[:, None]
    return np.stack([e1, e2, normal], axis=-1), valid


@dataclass
class DeformationBaseline:

    frame_inv: np.ndarray  # (F, 3, 3) float64
    valid: np.ndarray  # (F,) bool

    @classmethod
    def capture(cls, package: DigitalTwinPackage) -> "DeformationBaseline":
        frames, valid = face_frames(package.mesh_vertices, package.mesh_faces)
        frame_inv = np.zeros_like(frames)
        if valid.any():
            frame_inv[valid] = np.linalg.inv(frames[valid])
        return cls(frame_inv=frame_inv, valid=valid)


def _quaternions_to_matrices(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = np.where(norm > 1e-12, q / np.where(norm > 1e-12, norm, 1.0), np.array([1.0, 0.0, 0.0, 0.0]))
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((q.shape[0], 3, 3))
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def _matrices_to_quaternions(r: np.ndarray) -> np.ndarray:
    m00, m11, m22 = r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]
    t = np.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], axis=1)
    which = t.argmax(axis=1)
    s = 2.0 * np.sqrt(np.maximum(np.take_along_axis(t, which[:, None], axis=1)[:, 0], 1e-12))

    w_from = [s / 4, (r[:, 2, 1] - r[:, 1, 2]) / s, (r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 1, 0] - r[:, 0, 1]) / s]
    x_from = [(r[:, 2, 1] - r[:, 1, 2]) / s, s / 4, (r[:, 0, 1] + r[:, 1, 0]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s]
    y_from = [(r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 0, 1] + r[:, 1, 0]) / s, s / 4, (r[:, 1, 2] + r[:, 2, 1]) / s]
    z_from = [(r[:, 1, 0] - r[:, 0, 1]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s, (r[:, 1, 2] + r[:, 2, 1]) / s, s / 4]

    def pick(options):
        return np.choose(which, options)

    q = np.stack([pick(w_from), pick(x_from), pick(y_from), pick(z_from)], axis=1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q * np.where(q[:, :1] < 0, -1.0, 1.0)


def adapt_gaussians_to_deformation(package: DigitalTwinPackage, baseline: DeformationBaseline) -> int:
    n_faces = package.mesh_faces.shape[0]
    if baseline.frame_inv.shape[0] != n_faces:
        return 0

    frames_now, valid_now = face_frames(package.mesh_vertices, package.mesh_faces)
    gradient = np.matmul(frames_now, baseline.frame_inv)
    deviation = np.linalg.norm(gradient - _IDENTITY, axis=(1, 2))
    changed = baseline.valid & valid_now & (deviation > MIN_CHANGE)
    changed_faces = np.nonzero(changed)[0]
    if changed_faces.size == 0:
        return 0

    u, sigma, vt = np.linalg.svd(gradient[changed_faces])
    clamped = np.clip(sigma, 1.0 / MAX_STRETCH, MAX_STRETCH)
    full = np.matmul(u * clamped[:, None, :], vt)

    row_of_face = np.full(n_faces, -1, dtype=np.int64)
    row_of_face[changed_faces] = np.arange(changed_faces.size)
    gaussians = np.nonzero(row_of_face[package.gaussian_face_idx] >= 0)[0]
    rows = row_of_face[package.gaussian_face_idx[gaussians]]

    for start in range(0, gaussians.size, _CHUNK):
        idx = gaussians[start : start + _CHUNK]
        face_row = rows[start : start + _CHUNK]
        r = _quaternions_to_matrices(package.gaussian_quaternions[idx])
        s = package.gaussian_scales[idx].astype(np.float64)
        a = np.matmul(full[face_row], r * s[:, None, :])
        u_g, d_g, _ = np.linalg.svd(a)
        u_g[np.linalg.det(u_g) < 0, :, 2] *= -1.0
        package.gaussian_quaternions[idx] = _matrices_to_quaternions(u_g).astype(np.float32)
        package.gaussian_scales[idx] = d_g.astype(np.float32)

    baseline.frame_inv[changed_faces] = np.linalg.inv(frames_now[changed_faces])
    return int(gaussians.size)
