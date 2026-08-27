"""Generate a synthetic Gaussian Splat .ply from one of the bundled test meshes.

There is no trained 3D Gaussian Splatting scene in this project yet (that
requires a full capture -> COLMAP -> 3DGS-training pipeline, see
experiments/01_baseline_3dgs_reconstruction/). This script produces a
stand-in .ply in the same format, by sampling points on a test mesh's
surface and coloring them from its texture (if any) or its surface normals,
so that scripts/view_gaussians.py can be exercised end-to-end right away.

Usage:
    .venv\\Scripts\\python scripts\\generate_sample_gaussian_ply.py
    .venv\\Scripts\\python scripts\\generate_sample_gaussian_ply.py --mesh data\\raw\\test_models\\stanford-bunny.obj
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # allow `from src...` when run as a plain script

from src.viewer.gaussian_ply_io import save_gaussian_ply  # noqa: E402

TEST_MODELS_DIR = REPO_ROOT / "data" / "raw" / "test_models"


def load_obj(path: Path):
    """Minimal Wavefront OBJ loader (v / vt / f only, fan-triangulated).

    Returns (vertices, texcoords_or_None, v_faces, vt_faces_or_None). Enough
    for the simple single-object test meshes in data/raw/test_models --
    not a general-purpose OBJ parser (no vn, groups, materials, etc.).
    """
    vertices, texcoords = [], []
    v_faces, vt_faces = [], []
    has_uv = None

    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "v":
                vertices.append([float(x) for x in parts[1:4]])
            elif tag == "vt":
                texcoords.append([float(x) for x in parts[1:3]])
            elif tag == "f":
                v_idx, vt_idx = [], []
                for token in parts[1:]:
                    fields = token.split("/")
                    v_idx.append(int(fields[0]) - 1)  # OBJ indices are 1-based
                    if len(fields) > 1 and fields[1] != "":
                        vt_idx.append(int(fields[1]) - 1)
                if has_uv is None:
                    has_uv = len(vt_idx) == len(v_idx) and len(texcoords) > 0
                # Fan-triangulate polygons with more than 3 vertices.
                for i in range(1, len(v_idx) - 1):
                    v_faces.append((v_idx[0], v_idx[i], v_idx[i + 1]))
                    if has_uv:
                        vt_faces.append((vt_idx[0], vt_idx[i], vt_idx[i + 1]))

    vertices = np.array(vertices, dtype=np.float64)
    texcoords = np.array(texcoords, dtype=np.float64) if has_uv else None
    vt_faces = vt_faces if has_uv else None
    return vertices, texcoords, v_faces, vt_faces


def sample_points_on_mesh(vertices: np.ndarray, faces: list, n_points: int, rng):
    """Sample points uniformly (by triangle area) on the mesh surface.

    Returns positions, interpolated face normals, and the barycentric
    sampling info (face_idx, u, v, w) so callers can interpolate other
    per-vertex attributes (e.g. UV coordinates) consistently.
    """
    tri = vertices[np.array(faces)]  # (F, 3, 3)
    edge1 = tri[:, 1] - tri[:, 0]
    edge2 = tri[:, 2] - tri[:, 0]
    raw_normals = np.cross(edge1, edge2)
    areas = 0.5 * np.linalg.norm(raw_normals, axis=1)
    normals = raw_normals / np.clip(np.linalg.norm(raw_normals, axis=1, keepdims=True), 1e-12, None)

    probs = areas / areas.sum()
    face_idx = rng.choice(len(faces), size=n_points, p=probs)

    # Uniform sampling in a triangle via reflected random barycentric coords.
    u = rng.random(n_points)
    v = rng.random(n_points)
    outside = (u + v) > 1
    u[outside], v[outside] = 1 - u[outside], 1 - v[outside]
    w = 1 - u - v

    tri_sel = tri[face_idx]
    points = w[:, None] * tri_sel[:, 0] + u[:, None] * tri_sel[:, 1] + v[:, None] * tri_sel[:, 2]
    point_normals = normals[face_idx]
    return points, point_normals, face_idx, u, v, w


def colors_from_normals(normals: np.ndarray) -> np.ndarray:
    """Fallback coloring for untextured meshes: normal direction -> RGB."""
    return np.clip(0.5 + 0.5 * normals, 0.0, 1.0).astype(np.float32)


def colors_from_texture(
    texcoords: np.ndarray, vt_faces: list, face_idx: np.ndarray, u, v, w, texture: np.ndarray
) -> np.ndarray:
    """Interpolate UVs at the sampled points and look them up in the texture image."""
    uv_tri = texcoords[np.array(vt_faces)]  # (F, 3, 2)
    uv_sel = uv_tri[face_idx]
    uvs = w[:, None] * uv_sel[:, 0] + u[:, None] * uv_sel[:, 1] + v[:, None] * uv_sel[:, 2]

    height, width = texture.shape[:2]
    px = np.clip((uvs[:, 0] * width).astype(np.int64), 0, width - 1)
    py = np.clip(((1.0 - uvs[:, 1]) * height).astype(np.int64), 0, height - 1)  # OBJ v=0 is image bottom
    return texture[py, px, :3].astype(np.float32) / 255.0


def find_texture(mesh_path: Path) -> Path | None:
    """Look for a real UV texture image next to the mesh file.

    Only the "<name>_texture.*" naming convention is trusted (verified to be
    an actual UV-space texture for e.g. spot_texture.png). A same-basename
    "<name>.png" next to these test meshes is typically just a rendered
    preview thumbnail from the source repository, NOT a UV texture -- naively
    sampling it by UV coordinates produces meaningless colors, so it is
    deliberately not used here.
    """
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = mesh_path.with_name(f"{mesh_path.stem}_texture{ext}")
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh", default=str(TEST_MODELS_DIR / "spot.obj"), help="Input .obj mesh path"
    )
    parser.add_argument(
        "--out", default=str(TEST_MODELS_DIR / "sample_gaussians.ply"), help="Output .ply path"
    )
    parser.add_argument("--n-points", type=int, default=100_000, help="Number of Gaussians to generate")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    rng = np.random.default_rng(args.seed)

    vertices, texcoords, v_faces, vt_faces = load_obj(mesh_path)
    points, normals, face_idx, u, v, w = sample_points_on_mesh(vertices, v_faces, args.n_points, rng)

    texture_path = find_texture(mesh_path) if vt_faces is not None else None
    if texture_path is not None:
        texture = np.array(Image.open(texture_path).convert("RGB"))
        colors = colors_from_texture(texcoords, vt_faces, face_idx, u, v, w, texture)
        print(f"Colored from texture: {texture_path.name}")
    else:
        colors = colors_from_normals(normals)
        print("No texture found for this mesh, coloring from surface normals instead.")

    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    point_spacing = extent / (args.n_points ** (1 / 3))
    scales = np.full((args.n_points, 3), point_spacing * 0.6, dtype=np.float32)
    opacities = np.full(args.n_points, 0.95, dtype=np.float32)

    save_gaussian_ply(args.out, points.astype(np.float32), colors, opacities, scales)
    print(f"Wrote {args.n_points:,} synthetic Gaussians to {args.out}")


if __name__ == "__main__":
    main()
