import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_fidelity import REPO_ROOT, _as_png, _render, _View

import torch
from lpipsPyTorch import lpips
from scene.dataset_readers import readColmapSceneInfo
from utils.image_utils import psnr
from utils.loss_utils import ssim

from src.viewer.digital_twin import load_digital_twin_package
from src.viewer.gaussian_deformation import DeformationBaseline, _matrices_to_quaternions, _quaternions_to_matrices, adapt_gaussians_to_deformation
from src.viewer.mesh_detach import detach_selection, inside_faces


def _rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    angle = math.radians(degrees)
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def _tensors(package, means, quats, scales) -> dict:
    q = torch.tensor(quats, dtype=torch.float32, device="cuda")
    return {
        "means": torch.tensor(means, dtype=torch.float32, device="cuda"),
        "colors": torch.tensor(np.clip(package.gaussian_colors, 0.0, 1.0), dtype=torch.float32, device="cuda"),
        "opacities": torch.tensor(package.gaussian_opacities[:, None], dtype=torch.float32, device="cuda"),
        "scales": torch.tensor(scales, dtype=torch.float32, device="cuda"),
        "rotations": torch.nn.functional.normalize(q, dim=1),
    }


def _covariances(quats: np.ndarray, scales: np.ndarray) -> np.ndarray:
    r = _quaternions_to_matrices(quats)
    s = scales.astype(np.float64)
    return np.matmul(r * (s * s)[:, None, :], np.transpose(r, (0, 2, 1)))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="plant")
    parser.add_argument("--part", default="plant", help="the selection kept by evaluate_sam_pick.py --pair")
    parser.add_argument("--rest", default="pot", help="the other selection, which gives the axis")
    parser.add_argument("--angles", type=float, nargs="+", default=[15.0, 45.0, 90.0])
    parser.add_argument("--every", type=int, default=2, help="render every N-th training camera")
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--out", default=None, help="JSON file (default thesis/figures/rigid_edit_<scene>.json)")
    args = parser.parse_args()

    figures = REPO_ROOT / "thesis" / "figures"
    out = Path(args.out) if args.out else figures / f"rigid_edit_{args.scene}.json"
    package = load_digital_twin_package(str(REPO_ROOT / "data" / "gaussian_scenes" / f"{args.scene}_digital_twin.npz"))

    def vertices_of(name: str) -> np.ndarray:
        gaussians = np.load(figures / f"sam_pick_eval_{args.scene}_{name}.npy")
        return np.unique(package.mesh_faces[np.unique(package.gaussian_face_idx[gaussians])].reshape(-1))

    part_vertices, rest_vertices = vertices_of(args.part), vertices_of(args.rest)
    axis = package.mesh_vertices[part_vertices].mean(axis=0) - package.mesh_vertices[rest_vertices].mean(axis=0)
    cut = detach_selection(package, part_vertices)
    part_vertices = cut.selection
    inside = inside_faces(package.mesh_faces, part_vertices, package.mesh_vertices.shape[0])
    part = inside[package.gaussian_face_idx]
    centre = package.mesh_vertices[part_vertices].astype(np.float64).mean(axis=0)
    print(f"{args.scene}: part of {int(part.sum()):,} Gaussians on {cut.faces_detached:,} triangles "
          f"({cut.vertices_duplicated:,} vertices duplicated by the cut)", flush=True)

    info = readColmapSceneInfo(str(REPO_ROOT / "data" / "gaussian_scenes" / "_colmap_src" / args.scene), "images", eval=False)
    cams = sorted(info.train_cameras, key=lambda c: c.image_name)[:: args.every]
    views = []
    for cam in cams:
        scale = min(1.0, args.max_side / max(cam.width, cam.height))
        views.append(_View(cam, max(1, round(cam.width * scale)), max(1, round(cam.height * scale))))

    rest_positions = package.recompute_gaussian_positions().astype(np.float64)
    record = {"scene": args.scene, "part_gaussians": int(part.sum()), "gaussians": int(len(package)),
              "triangles_detached": cut.faces_detached, "vertices_duplicated": cut.vertices_duplicated,
              "views": [c.image_name for c in cams], "angles": {}}
    for angle in args.angles:
        rotation = _rotation(axis, angle)

        exact_means = rest_positions.copy()
        exact_means[part] = (exact_means[part] - centre) @ rotation.T + centre
        exact_quats = package.gaussian_quaternions.copy()
        exact_quats[part] = _matrices_to_quaternions(
            np.matmul(rotation[None], _quaternions_to_matrices(package.gaussian_quaternions[part]))).astype(np.float32)

        moved = package.clone()
        moved.mesh_vertices[part_vertices] = ((moved.mesh_vertices[part_vertices] - centre) @ rotation.T + centre).astype(np.float32)
        positions_only = moved.recompute_gaussian_positions()

        adapted = package.clone()
        baseline = DeformationBaseline.capture(adapted)
        adapted.mesh_vertices[:] = moved.mesh_vertices
        updated = adapt_gaussians_to_deformation(adapted, baseline)

        exact_cov = _covariances(exact_quats[part], package.gaussian_scales[part])
        adapted_cov = _covariances(adapted.gaussian_quaternions[part], adapted.gaussian_scales[part])
        norm = np.linalg.norm(exact_cov, axis=(1, 2))
        matched = np.linalg.norm(adapted_cov - exact_cov, axis=(1, 2)) <= 1e-3 * np.maximum(norm, 1e-12)

        versions = {
            "exact": _tensors(package, exact_means, exact_quats, package.gaussian_scales),
            "positions_only": _tensors(package, positions_only, package.gaussian_quaternions, package.gaussian_scales),
            "adapted": _tensors(adapted, adapted.recompute_gaussian_positions(), adapted.gaussian_quaternions, adapted.gaussian_scales),
        }
        indicator = {**versions["exact"], "colors": torch.tensor(part, dtype=torch.float32, device="cuda")[:, None].expand(-1, 3).contiguous()}
        rows = {name: {"psnr": [], "ssim": [], "lpips": [], "psnr_part": []} for name in ("positions_only", "adapted")}
        with torch.no_grad():
            for view in views:
                reference = _as_png(_render("twin", versions["exact"], view))
                region = _render("twin", indicator, view)[0] > 0.5
                for name in rows:
                    image = _as_png(_render("twin", versions[name], view))
                    rows[name]["psnr"].append(float(psnr(image, reference).mean()))
                    rows[name]["ssim"].append(float(ssim(image, reference)))
                    rows[name]["lpips"].append(float(lpips(image, reference, net_type="vgg")))
                    if region.any():
                        mse = ((image[0] - reference[0]) ** 2).mean(dim=0)[region].mean()
                        rows[name]["psnr_part"].append(float(20 * torch.log10(1.0 / torch.sqrt(mse))) if mse > 0 else float("inf"))
        entry = {"gaussians_updated_by_adapt": updated, "part_gaussians_matching_exact": int(matched.sum()),
                 "max_position_difference": float(np.abs(positions_only - exact_means).max())}
        for name, values in rows.items():
            finite = {k: [x for x in v if np.isfinite(x)] for k, v in values.items()}
            entry[name] = {"mean": {k: float(np.mean(v)) if v else float("inf") for k, v in finite.items()},
                           "identical_views": int(sum(1 for x in values["psnr"] if not np.isfinite(x))), "per_view": values}
        record["angles"][f"{angle:g}"] = entry
        print(f"  {angle:5.1f} deg: adapt updated {updated:,}; {int(matched.sum()):,} of {int(part.sum()):,} part Gaussians "
              f"match the exact covariance; max position gap {entry['max_position_difference']:.2e}", flush=True)
        for name in rows:
            m = entry[name]["mean"]
            print(f"      {name:15s} PSNR {m['psnr']:.2f}  in part {m['psnr_part']:.2f}  SSIM {m['ssim']:.4f}  LPIPS {m['lpips']:.4f}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
