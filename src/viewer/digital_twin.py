"""Digital-twin package: a mesh bound to Gaussians via fixed barycentric weights.

Produced offline by scripts/export_digital_twin_package.py from a SuGaR
refined checkpoint (see external/SuGaR/sugar_scene/sugar_model.py). Every
Gaussian sits at a fixed barycentric position within one mesh triangle, so
moving the triangle's vertices and recomputing the barycentric sum gives the
Gaussian's new world-space position -- the mechanism SuGaR itself uses to
let mesh edits (in Blender, etc.) drive the Gaussians.

This module only follows *position* on edit: orientation/scale re-fitting to
the triangle's local deformation (SuGaR's `get_edited_quaternions_and_scales`)
is a documented simplification, not implemented here -- see project README.
Deliberately pure numpy: no torch/CUDA needed to load or edit a package,
only to produce one (that happens once, offline, in the gs-train environment).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DigitalTwinPackage:
    """A mesh (editable) plus the Gaussians bound to it (position-following)."""

    mesh_vertices: np.ndarray  # (V, 3) float32, mutable -- edited in place
    mesh_faces: np.ndarray  # (F, 3) int32
    gaussian_face_idx: np.ndarray  # (N,) int32, which face each Gaussian belongs to
    gaussian_bary: np.ndarray  # (N, 3) float32, barycentric weights within that face
    gaussian_colors: np.ndarray  # (N, 3) float32, RGB in [0, 1]
    gaussian_opacities: np.ndarray  # (N,) float32, alpha in [0, 1]
    gaussian_scales: np.ndarray  # (N, 3) float32, world-space scale (fixed, not re-fit on edit)
    gaussian_quaternions: np.ndarray  # (N, 4) float32, orientation (fixed, not re-fit on edit)

    def __len__(self) -> int:
        return self.gaussian_face_idx.shape[0]

    def recompute_gaussian_positions(self) -> np.ndarray:
        """Recompute every Gaussian's world position from the current mesh vertices."""
        face_verts = self.mesh_vertices[self.mesh_faces[self.gaussian_face_idx]]  # (N, 3, 3)
        return (face_verts * self.gaussian_bary[:, :, None]).sum(axis=1)  # (N, 3)

    def clone(self) -> "DigitalTwinPackage":
        """Deep-copy every array -- used to snapshot the package for undo."""
        return DigitalTwinPackage(
            mesh_vertices=self.mesh_vertices.copy(),
            mesh_faces=self.mesh_faces.copy(),
            gaussian_face_idx=self.gaussian_face_idx.copy(),
            gaussian_bary=self.gaussian_bary.copy(),
            gaussian_colors=self.gaussian_colors.copy(),
            gaussian_opacities=self.gaussian_opacities.copy(),
            gaussian_scales=self.gaussian_scales.copy(),
            gaussian_quaternions=self.gaussian_quaternions.copy(),
        )

    def remove_faces_by_vertex_selection(self, vertex_indices: np.ndarray) -> None:
        """Delete the given vertices, any face touching them, and any
        Gaussian bound to a removed face -- then compact and remap every
        array so indices stay consistent (standard "delete vertex removes
        incident faces" mesh-editing convention).
        """
        remove_mask = np.zeros(self.mesh_vertices.shape[0], dtype=bool)
        remove_mask[vertex_indices] = True

        face_removed = remove_mask[self.mesh_faces].any(axis=1)  # (F,)
        keep_faces = ~face_removed
        keep_verts = ~remove_mask

        new_vertex_index = np.full(self.mesh_vertices.shape[0], -1, dtype=np.int64)
        new_vertex_index[keep_verts] = np.arange(int(keep_verts.sum()))

        new_face_index = np.full(self.mesh_faces.shape[0], -1, dtype=np.int64)
        new_face_index[keep_faces] = np.arange(int(keep_faces.sum()))

        self.mesh_vertices = self.mesh_vertices[keep_verts].copy()
        self.mesh_faces = new_vertex_index[self.mesh_faces[keep_faces]].astype(np.int32)

        keep_gaussians = keep_faces[self.gaussian_face_idx]
        self.gaussian_face_idx = new_face_index[self.gaussian_face_idx[keep_gaussians]].astype(np.int32)
        self.gaussian_bary = self.gaussian_bary[keep_gaussians]
        self.gaussian_colors = self.gaussian_colors[keep_gaussians]
        self.gaussian_opacities = self.gaussian_opacities[keep_gaussians]
        self.gaussian_scales = self.gaussian_scales[keep_gaussians]
        self.gaussian_quaternions = self.gaussian_quaternions[keep_gaussians]


def load_digital_twin_package(path: str) -> DigitalTwinPackage:
    """Load a digital-twin package saved by scripts/export_digital_twin_package.py."""
    with np.load(path) as data:
        return DigitalTwinPackage(
            mesh_vertices=data["mesh_vertices"].astype(np.float32).copy(),
            mesh_faces=data["mesh_faces"].astype(np.int32),
            gaussian_face_idx=data["gaussian_face_idx"].astype(np.int32),
            gaussian_bary=data["gaussian_bary"].astype(np.float32),
            gaussian_colors=data["gaussian_colors"].astype(np.float32),
            gaussian_opacities=data["gaussian_opacities"].astype(np.float32),
            gaussian_scales=data["gaussian_scales"].astype(np.float32),
            gaussian_quaternions=data["gaussian_quaternions"].astype(np.float32),
        )


def save_digital_twin_package(path: str, package: DigitalTwinPackage) -> None:
    """Save a digital-twin package (used by the export script)."""
    np.savez(
        path,
        mesh_vertices=package.mesh_vertices.astype(np.float32),
        mesh_faces=package.mesh_faces.astype(np.int32),
        gaussian_face_idx=package.gaussian_face_idx.astype(np.int32),
        gaussian_bary=package.gaussian_bary.astype(np.float32),
        gaussian_colors=package.gaussian_colors.astype(np.float32),
        gaussian_opacities=package.gaussian_opacities.astype(np.float32),
        gaussian_scales=package.gaussian_scales.astype(np.float32),
        gaussian_quaternions=package.gaussian_quaternions.astype(np.float32),
    )
