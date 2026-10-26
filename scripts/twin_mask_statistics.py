import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sam_point_segment import REPO_ROOT, _build_cameras

from PIL import Image

from scene.dataset_readers import readColmapSceneInfo
from src.viewer.digital_twin import load_digital_twin_package
from src.viewer.sam_pick_metrics import carve_object
from src.viewer.twin_provenance import MASK_SHARE_WARNING, inside_mask_share, visible_gaussians


def statistics(scene: str) -> dict:
    twin = REPO_ROOT / "data" / "gaussian_scenes" / f"{scene}_digital_twin.npz"
    source = REPO_ROOT / "data" / "gaussian_scenes" / "_colmap_src" / scene
    package = load_digital_twin_package(str(twin))
    positions = package.recompute_gaussian_positions()
    infos = readColmapSceneInfo(str(source), images="images", eval=False).train_cameras
    cameras = _build_cameras(infos, 512)
    silhouettes = {
        view: np.asarray(Image.open(source / "masks" / (Path(info.image_path).stem + ".png")).convert("L").resize((camera.width, camera.height))) > 127
        for view, (info, camera) in enumerate(zip(infos, cameras))
    }
    inside = carve_object(cameras, positions, silhouettes)
    brightness, opacity = package.gaussian_colors.mean(axis=1), package.gaussian_opacities

    def part(selected: np.ndarray) -> dict:
        if not selected.any():
            return {"gaussians": 0}
        return {
            "gaussians": int(selected.sum()),
            "share_of_twin": float(selected.mean()),
            "mean_brightness": float(brightness[selected].mean()),
            "share_darker_than_0.1": float((brightness[selected] < 0.1).mean()),
            "mean_opacity": float(opacity[selected].mean()),
            "share_opacity_above_0.5": float((opacity[selected] > 0.5).mean()),
        }

    visible = visible_gaussians(package.gaussian_colors, package.gaussian_opacities)
    share, considered = inside_mask_share(positions, cameras, silhouettes, considered=visible)
    return {
        "scene": scene,
        "gaussians": int(len(positions)),
        "inside_masks": part(inside),
        "outside_masks": part(~inside),
        "check": {"visible_gaussians_considered": considered, "share_inside_masks": share, "warns": bool(share < MASK_SHARE_WARNING)},
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--out", default=None, help="JSON file for the numbers")
    args = parser.parse_args()

    results = []
    print(f"{'scene':14s} {'Gaussians':>9s} {'inside':>7s} | {'outside':>8s} {'brightness':>10s} {'dark':>5s} {'opacity':>8s} | {'check: visible':>14s} {'inside masks':>12s}")
    for scene in args.scenes:
        record = statistics(scene)
        results.append(record)
        inside, outside, check = record["inside_masks"], record["outside_masks"], record["check"]
        print(f"{scene:14s} {record['gaussians']:9,d} {inside['share_of_twin']:7.1%} | {outside['gaussians']:8,d} "
              f"{outside.get('mean_brightness', float('nan')):10.3f} {outside.get('share_darker_than_0.1', float('nan')):5.0%} "
              f"{outside.get('mean_opacity', float('nan')):8.2f} | {check['visible_gaussians_considered']:14,d} {check['share_inside_masks']:12.1%}"
              f"{'  <- warns' if check['warns'] else ''}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
