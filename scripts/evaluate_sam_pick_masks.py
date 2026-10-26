import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_fidelity import _render, _View
from evaluate_sam_pick import DEFAULT_CHECKPOINT, _load_silhouettes, _random_clicks
from sam_point_segment import REPO_ROOT, SamSession

import torch

from src.viewer.digital_twin import load_digital_twin_package
from src.viewer.sam_pick import LEVEL_NAMES, SceneGeometry, segment_object
from src.viewer.sam_pick_metrics import carve_object, confusion_metrics

THRESHOLD = 0.5


def _twin_tensors(package) -> dict:
    quats = torch.tensor(package.gaussian_quaternions, dtype=torch.float32, device="cuda")
    return {
        "means": torch.tensor(package.recompute_gaussian_positions(), dtype=torch.float32, device="cuda"),
        "opacities": torch.tensor(package.gaussian_opacities[:, None], dtype=torch.float32, device="cuda"),
        "scales": torch.tensor(package.gaussian_scales, dtype=torch.float32, device="cuda"),
        "rotations": torch.nn.functional.normalize(quats, dim=1),
    }


def _mask_scores(twin: dict, selected: np.ndarray, views: list, truths: list) -> dict:
    """IoU and pixel accuracy of the rendered selection against each view's mask."""
    colors = torch.tensor(selected, dtype=torch.float32, device="cuda")[:, None].expand(-1, 3).contiguous()
    model = {**twin, "colors": colors}
    ious, accuracies = [], []
    with torch.no_grad():
        for view, truth in zip(views, truths):
            rendered = _render("twin", model, view)[0] > THRESHOLD
            union = int((rendered | truth).sum())
            ious.append(int((rendered & truth).sum()) / union if union else 1.0)
            accuracies.append(float((rendered == truth).float().mean()))
    return {"iou": float(np.mean(ious)), "accuracy": float(np.mean(accuracies)), "iou_per_view": ious, "accuracy_per_view": accuracies}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--twin", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--sam-checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--clicks", type=int, default=6, help="random clicks on the object, as in evaluate_sam_pick.py")
    parser.add_argument("--levels", type=int, nargs="+", default=[2], help="0 part, 1 object, 2 whole (default: whole)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--every", type=int, default=1, help="score every N-th view")
    parser.add_argument("--min-ratio", type=float, default=None,
                        help="the vote a visible Gaussian needs (default: a strict majority; 0.3 is sam_pick.INCLUSIVE); "
                             "with it the selections differ from the stored run, so the check against it is skipped")
    parser.add_argument("--reference", default=None,
                        help="evaluate_sam_pick.py record to check the selections against "
                             "(default thesis/figures/sam_pick_eval_<scene>.json, if it exists)")
    parser.add_argument("--out", default=None, help="JSON file (default thesis/figures/sam_pick_masks_<scene>.json)")
    args = parser.parse_args()

    twin_path = Path(args.twin) if args.twin else REPO_ROOT / "data" / "gaussian_scenes" / f"{args.scene}_digital_twin.npz"
    source = Path(args.source) if args.source else REPO_ROOT / "data" / "gaussian_scenes" / "_colmap_src" / args.scene
    out = Path(args.out) if args.out else REPO_ROOT / "thesis" / "figures" / f"sam_pick_masks_{args.scene}.json"
    reference = Path(args.reference) if args.reference else REPO_ROOT / "thesis" / "figures" / f"sam_pick_eval_{args.scene}.json"

    package = load_digital_twin_package(str(twin_path))
    positions = package.recompute_gaussian_positions()
    session = SamSession(source, args.sam_checkpoint)
    cameras = session.cameras
    geometry = SceneGeometry(positions, cameras)
    silhouettes = _load_silhouettes(source, session)
    truth = carve_object(cameras, positions, silhouettes)

    scored = list(range(0, len(cameras), args.every))
    views = [_View(session.cam_infos[v], cameras[v].width, cameras[v].height) for v in scored]
    truths = [torch.tensor(silhouettes[v], device="cuda") for v in scored]
    twin = _twin_tensors(package)
    ceiling = _mask_scores(twin, truth, views, truths)
    print(f"{args.scene}: {len(positions):,} Gaussians, {len(scored)} of {len(cameras)} views scored, "
          f"{cameras[0].width} x {cameras[0].height}; true object rendered: IoU {ceiling['iou']:.3f}, "
          f"accuracy {ceiling['accuracy']:.3f}", flush=True)

    stored = {}
    if reference.is_file() and args.min_ratio is None:
        for run in json.loads(reference.read_text(encoding="utf-8"))["runs"]:
            stored[(run["level"], run["click"]["gaussian"])] = run["selected"]

    record = {"scene": args.scene, "gaussians": int(len(positions)), "true_object": int(truth.sum()), "seed": args.seed,
              "threshold": THRESHOLD, "views": scored, "width": cameras[0].width, "height": cameras[0].height,
              "true_object_rendered": ceiling, "runs": []}
    clicks = _random_clicks(np.random.default_rng(args.seed), session, positions, silhouettes, args.clicks)
    for level in args.levels:
        for click in clicks:
            started = time.perf_counter()
            result = segment_object(positions, cameras, positions[click["gaussian"]], session.predict, level=level,
                                    geometry=geometry, prefetch=session.prefetch,
                                    **({} if args.min_ratio is None else {"min_ratio": args.min_ratio}))
            seconds = time.perf_counter() - started
            selected = int(result.selected.sum())
            expected = stored.get((level, click["gaussian"]))
            entry = {"level": level, "click": click, "seconds": seconds, "selected": selected,
                     "same_as_reference": None if expected is None else expected == selected,
                     **{f"gaussian_{k}": v for k, v in confusion_metrics(result.selected, truth).items()},
                     **_mask_scores(twin, result.selected, views, truths)}
            record["runs"].append(entry)
            check = "" if expected is None else ("  (as recorded)" if expected == selected else f"  (recorded {expected:,})")
            print(f"  level {LEVEL_NAMES[level]:6s} view {click['view']:3d}: {selected:,} selected{check}; "
                  f"2D IoU {entry['iou']:.3f}, accuracy {entry['accuracy']:.3f}; 3D IoU {entry['gaussian_iou']:.2f}", flush=True)

    print("\n| Level | 2D IoU | 2D accuracy | 2D IoU of the true object | 3D IoU |\n| --- | --- | --- | --- | --- |")
    for level in args.levels:
        rows = [r for r in record["runs"] if r["level"] == level]
        iou, acc, iou3 = (np.array([r[k] for r in rows]) for k in ("iou", "accuracy", "gaussian_iou"))
        print(f"| {LEVEL_NAMES[level]} | {iou.mean():.3f} ± {iou.std():.3f} | {acc.mean():.3f} ± {acc.std():.3f} | "
              f"{ceiling['iou']:.3f} | {iou3.mean():.2f} ± {iou3.std():.2f} |")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
