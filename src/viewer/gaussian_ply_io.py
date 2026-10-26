from dataclasses import dataclass

import numpy as np
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814


@dataclass
class GaussianCloud:
    positions: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray
    scales: np.ndarray

    def __len__(self) -> int:
        return self.positions.shape[0]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _inverse_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def load_gaussian_ply(path: str) -> GaussianCloud:
    ply_data = PlyData.read(str(path))
    vertex = ply_data["vertex"].data
    names = vertex.dtype.names

    positions = np.stack(
        [vertex["x"], vertex["y"], vertex["z"]], axis=1
    ).astype(np.float32)
    n = positions.shape[0]

    if "f_dc_0" in names:
        dc = np.stack(
            [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1
        ).astype(np.float32)
        colors = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
    elif "red" in names:
        colors = (
            np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(
                np.float32
            )
            / 255.0
        )
    else:
        colors = np.full((n, 3), 0.7, dtype=np.float32)

    if "opacity" in names:
        opacities = _sigmoid(np.asarray(vertex["opacity"], dtype=np.float32))
    else:
        opacities = np.ones(n, dtype=np.float32)

    if "scale_0" in names:
        scales = np.exp(
            np.stack(
                [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1
            ).astype(np.float32)
        )
    else:
        extent = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
        radius = max(extent, 1.0) * 0.002
        scales = np.full((n, 3), radius, dtype=np.float32)

    return GaussianCloud(
        positions=positions, colors=colors, opacities=opacities, scales=scales
    )


def save_gaussian_ply(
    path: str,
    positions: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray = None,
    sh_degree: int = 3,
) -> None:
    n = positions.shape[0]
    if rotations is None:
        rotations = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1)
        )

    dc = (np.clip(colors, 0.0, 1.0).astype(np.float32) - 0.5) / SH_C0
    n_rest = 3 * ((sh_degree + 1) ** 2 - 1)
    rest = np.zeros((n, n_rest), dtype=np.float32)

    raw_opacity = _inverse_sigmoid(np.asarray(opacities, dtype=np.float32))
    raw_scale = np.log(np.clip(scales, 1e-8, None)).astype(np.float32)

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    dtype += [(f"f_dc_{i}", "f4") for i in range(3)]
    dtype += [(f"f_rest_{i}", "f4") for i in range(n_rest)]
    dtype += [("opacity", "f4")]
    dtype += [(f"scale_{i}", "f4") for i in range(3)]
    dtype += [(f"rot_{i}", "f4") for i in range(4)]

    vertex = np.zeros(n, dtype=dtype)
    vertex["x"], vertex["y"], vertex["z"] = positions[:, 0], positions[:, 1], positions[:, 2]
    for i in range(3):
        vertex[f"f_dc_{i}"] = dc[:, i]
    for i in range(n_rest):
        vertex[f"f_rest_{i}"] = rest[:, i]
    vertex["opacity"] = raw_opacity
    for i in range(3):
        vertex[f"scale_{i}"] = raw_scale[:, i]
    for i in range(4):
        vertex[f"rot_{i}"] = rotations[:, i]

    element = PlyElement.describe(vertex, "vertex")
    PlyData([element], text=False).write(str(path))
