import sys

import numpy as np
import pytest
from PyQt5 import QtWidgets
from vispy import app as vispy_app


@pytest.fixture(scope="session")
def qapp():
    vispy_app.use_app("pyqt5")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    yield app


@pytest.fixture
def make_grid_twin():
    from src.viewer.digital_twin import DigitalTwinPackage, save_digital_twin_package

    def make(path, n: int = 25, spacing: float = 0.4) -> None:
        rng = np.random.default_rng(3)
        xs = np.arange(n) * spacing
        gx, gy = np.meshgrid(xs, xs)
        vertices = np.stack([gx.ravel(), gy.ravel(), np.zeros(n * n)], axis=1).astype(np.float32)
        faces = []
        for i in range(n - 1):
            for j in range(n - 1):
                a = i * n + j
                b, c = a + 1, a + n
                faces += [[a, b, c], [b, c + 1, c]]
        faces = np.array(faces, dtype=np.int32)
        n_faces = faces.shape[0]
        quats = rng.normal(size=(n_faces, 4)).astype(np.float32)
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        save_digital_twin_package(
            str(path),
            DigitalTwinPackage(
                mesh_vertices=vertices,
                mesh_faces=faces,
                gaussian_face_idx=np.arange(n_faces, dtype=np.int32),
                gaussian_bary=np.full((n_faces, 3), 1 / 3, dtype=np.float32),
                gaussian_colors=rng.uniform(size=(n_faces, 3)).astype(np.float32),
                gaussian_opacities=np.full(n_faces, 0.8, dtype=np.float32),
                gaussian_scales=rng.uniform(0.01, 0.08, size=(n_faces, 3)).astype(np.float32),
                gaussian_quaternions=quats,
            ),
        )

    return make
