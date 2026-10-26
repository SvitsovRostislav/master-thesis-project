import copy
import json
from dataclasses import dataclass

import numpy as np


@dataclass
class DigitalTwinPackage:

    mesh_vertices: np.ndarray
    mesh_faces: np.ndarray
    gaussian_face_idx: np.ndarray
    gaussian_bary: np.ndarray
    gaussian_colors: np.ndarray
    gaussian_opacities: np.ndarray
    gaussian_scales: np.ndarray
    gaussian_quaternions: np.ndarray
    provenance: dict | None = None
    vertex_origin: np.ndarray | None = None
    face_origin: np.ndarray | None = None

    def __len__(self) -> int:
        return self.gaussian_face_idx.shape[0]

    def recompute_gaussian_positions(self) -> np.ndarray:
        face_verts = self.mesh_vertices[self.mesh_faces[self.gaussian_face_idx]]
        return (face_verts * self.gaussian_bary[:, :, None]).sum(axis=1)

    def clone(self) -> "DigitalTwinPackage":
        return DigitalTwinPackage(
            mesh_vertices=self.mesh_vertices.copy(),
            mesh_faces=self.mesh_faces.copy(),
            gaussian_face_idx=self.gaussian_face_idx.copy(),
            gaussian_bary=self.gaussian_bary.copy(),
            gaussian_colors=self.gaussian_colors.copy(),
            gaussian_opacities=self.gaussian_opacities.copy(),
            gaussian_scales=self.gaussian_scales.copy(),
            gaussian_quaternions=self.gaussian_quaternions.copy(),
            provenance=copy.deepcopy(self.provenance),
            vertex_origin=None if self.vertex_origin is None else self.vertex_origin.copy(),
            face_origin=None if self.face_origin is None else self.face_origin.copy(),
        )

    def start_tracking_origins(self) -> None:
        self.vertex_origin = np.arange(self.mesh_vertices.shape[0], dtype=np.int64)
        self.face_origin = np.arange(self.mesh_faces.shape[0], dtype=np.int64)

    def remove_faces_by_vertex_selection(self, vertex_indices: np.ndarray) -> None:
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
        if self.vertex_origin is not None:
            self.vertex_origin = self.vertex_origin[keep_verts]
        if self.face_origin is not None:
            self.face_origin = self.face_origin[keep_faces]

        keep_gaussians = keep_faces[self.gaussian_face_idx]
        self.gaussian_face_idx = new_face_index[self.gaussian_face_idx[keep_gaussians]].astype(np.int32)
        self.gaussian_bary = self.gaussian_bary[keep_gaussians]
        self.gaussian_colors = self.gaussian_colors[keep_gaussians]
        self.gaussian_opacities = self.gaussian_opacities[keep_gaussians]
        self.gaussian_scales = self.gaussian_scales[keep_gaussians]
        self.gaussian_quaternions = self.gaussian_quaternions[keep_gaussians]

    def add_gaussians(
        self,
        face_idx: np.ndarray,
        bary: np.ndarray,
        colors: np.ndarray,
        opacities: np.ndarray,
        scales: np.ndarray,
        quaternions: np.ndarray,
    ) -> None:
        self.gaussian_face_idx = np.concatenate([self.gaussian_face_idx, face_idx.astype(np.int32)])
        self.gaussian_bary = np.concatenate([self.gaussian_bary, bary.astype(np.float32)])
        self.gaussian_colors = np.concatenate([self.gaussian_colors, colors.astype(np.float32)])
        self.gaussian_opacities = np.concatenate([self.gaussian_opacities, opacities.astype(np.float32)])
        self.gaussian_scales = np.concatenate([self.gaussian_scales, scales.astype(np.float32)])
        self.gaussian_quaternions = np.concatenate([self.gaussian_quaternions, quaternions.astype(np.float32)])

    def remove_gaussians_by_index(self, gaussian_indices: np.ndarray) -> None:
        keep = np.ones(len(self), dtype=bool)
        keep[gaussian_indices] = False
        self.gaussian_face_idx = self.gaussian_face_idx[keep]
        self.gaussian_bary = self.gaussian_bary[keep]
        self.gaussian_colors = self.gaussian_colors[keep]
        self.gaussian_opacities = self.gaussian_opacities[keep]
        self.gaussian_scales = self.gaussian_scales[keep]
        self.gaussian_quaternions = self.gaussian_quaternions[keep]


def load_digital_twin_package(path: str) -> DigitalTwinPackage:
    with np.load(path) as data:
        provenance = None
        if "provenance_json" in data.files:
            try:
                provenance = json.loads(str(data["provenance_json"]))
            except ValueError:
                provenance = None
        return DigitalTwinPackage(
            mesh_vertices=data["mesh_vertices"].astype(np.float32).copy(),
            mesh_faces=data["mesh_faces"].astype(np.int32),
            gaussian_face_idx=data["gaussian_face_idx"].astype(np.int32),
            gaussian_bary=data["gaussian_bary"].astype(np.float32),
            gaussian_colors=data["gaussian_colors"].astype(np.float32),
            gaussian_opacities=data["gaussian_opacities"].astype(np.float32),
            gaussian_scales=data["gaussian_scales"].astype(np.float32),
            gaussian_quaternions=data["gaussian_quaternions"].astype(np.float32),
            provenance=provenance,
        )


def save_digital_twin_package(path: str, package: DigitalTwinPackage) -> None:
    extra = {"provenance_json": np.array(json.dumps(package.provenance))} if package.provenance else {}
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
        **extra,
    )
