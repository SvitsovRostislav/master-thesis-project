import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SUGAR_ROOT = REPO_ROOT / "external" / "SuGaR"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SUGAR_ROOT))

from src.viewer.digital_twin import DigitalTwinPackage, save_digital_twin_package  # noqa: E402
from src.viewer.gaussian_ply_io import SH_C0  # noqa: E402
from src.viewer.twin_provenance import with_counts  # noqa: E402

from sugar_scene.gs_model import GaussianSplattingWrapper  # noqa: E402
from sugar_scene.sugar_model import load_refined_model  # noqa: E402


def build_gaussian_face_bindings(n_faces: int, bary_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_gaussians_per_face = bary_coords.shape[0]
    face_idx = np.repeat(np.arange(n_faces, dtype=np.int32), n_gaussians_per_face)
    bary = np.tile(bary_coords[:, :, 0], (n_faces, 1))  # pattern is identical per face
    return face_idx, bary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-path", required=True, help="COLMAP source path (e.g. data/raw/mipnerf360_bonsai)")
    parser.add_argument("--gs-output-dir", required=True, help="Baseline 3DGS checkpoint dir (e.g. data/gaussian_scenes/bonsai_baseline)")
    parser.add_argument("--refined-checkpoint", required=True, help="Path to SuGaR's refined .pt checkpoint")
    parser.add_argument("--iteration-to-load", type=int, default=30000)
    parser.add_argument("--out", required=True, help="Output .npz path")
    parser.add_argument("--provenance", default=None, help="JSON build record (twin_provenance.build_record) to store in the package")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading baseline 3DGS cameras (needed by SuGaR's own checkpoint loader)...")
    nerfmodel = GaussianSplattingWrapper(
        source_path=args.scene_path,
        output_path=args.gs_output_dir,
        iteration_to_load=args.iteration_to_load,
        load_gt_images=False,
        eval_split=False,
        white_background=False,
    )

    print(f"Loading refined SuGaR checkpoint: {args.refined_checkpoint}")
    refined_sugar = load_refined_model(args.refined_checkpoint, nerfmodel, device=device)

    with torch.no_grad():
        mesh_vertices = refined_sugar._points.detach().cpu().numpy().astype(np.float32)
        mesh_faces = refined_sugar._surface_mesh_faces.detach().cpu().numpy().astype(np.int32)
        n_faces = mesh_faces.shape[0]

        bary_coords = refined_sugar.surface_triangle_bary_coords.detach().cpu().numpy()
        gaussian_face_idx, gaussian_bary = build_gaussian_face_bindings(n_faces, bary_coords)

        sh_dc = refined_sugar._sh_coordinates_dc.detach().cpu().numpy()[:, 0, :]  # (N, 3)
        gaussian_colors = np.clip(0.5 + SH_C0 * sh_dc, 0.0, 1.0).astype(np.float32)

        gaussian_opacities = refined_sugar.strengths.detach().cpu().numpy().reshape(-1).astype(np.float32)
        gaussian_scales = refined_sugar.scaling.detach().cpu().numpy().astype(np.float32)
        gaussian_quaternions = refined_sugar.quaternions.detach().cpu().numpy().astype(np.float32)

    n_gaussians = gaussian_face_idx.shape[0]
    print(f"{mesh_vertices.shape[0]:,} mesh vertices, {n_faces:,} faces, {n_gaussians:,} Gaussians.")
    assert gaussian_colors.shape[0] == n_gaussians
    assert gaussian_scales.shape[0] == n_gaussians
    assert gaussian_quaternions.shape[0] == n_gaussians

    package = DigitalTwinPackage(
        mesh_vertices=mesh_vertices,
        mesh_faces=mesh_faces,
        gaussian_face_idx=gaussian_face_idx,
        gaussian_bary=gaussian_bary,
        gaussian_colors=gaussian_colors,
        gaussian_opacities=gaussian_opacities,
        gaussian_scales=gaussian_scales,
        gaussian_quaternions=gaussian_quaternions,
        provenance=(
            with_counts(json.loads(Path(args.provenance).read_text(encoding="utf-8")), mesh_vertices.shape[0], n_faces, n_gaussians)
            if args.provenance else None
        ),
    )
    save_digital_twin_package(args.out, package)
    print(f"Saved digital-twin package to {args.out}")


if __name__ == "__main__":
    main()
