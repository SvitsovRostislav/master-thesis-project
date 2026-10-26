import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_sam_pick as ev
from sam_point_segment import REPO_ROOT, SamSession

import src.viewer.sam_pick as sam_pick
from src.viewer.digital_twin import load_digital_twin_package
from src.viewer.sam_pick_metrics import carve_object

RADII = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--clicks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="JSON file for the numbers")
    args = parser.parse_args()

    captured = []

    class Spy(sam_pick.Voter):
        def __init__(self, positions):
            super().__init__(positions)
            captured.append(self)

    sam_pick.Voter = Spy
    sam_pick.HULL_NEIGHBOURHOOD = None

    twin = REPO_ROOT / "data" / "gaussian_scenes" / f"{args.scene}_digital_twin.npz"
    source = REPO_ROOT / "data" / "gaussian_scenes" / "_colmap_src" / args.scene
    positions = load_digital_twin_package(str(twin)).recompute_gaussian_positions()
    session = SamSession(source, REPO_ROOT / "data" / "checkpoints" / "sam_vit_b_01ec64.pth")
    silhouettes = ev._load_silhouettes(source, session)
    truth = carve_object(session.cameras, positions, silhouettes)
    geometry = sam_pick.SceneGeometry(positions, session.cameras)
    clicks = ev._random_clicks(np.random.default_rng(args.seed), session, positions, silhouettes, args.clicks)

    per_click, good_all, bad_all = [], [], []
    for click in clicks:
        captured.clear()
        sam_pick.segment_object(
            positions, session.cameras, positions[click["gaussian"]], session.predict, level=2, geometry=geometry,
            prefetch=session.prefetch,
        )
        voter = captured[-1]
        visible = (voter._seen > 0) & (voter._inside > 0.5 * voter._seen)
        hull = (voter._seen == 0) & (voter._framed >= sam_pick.HULL_MIN_VIEWS) & (voter._inside_framed >= sam_pick.HULL_RATIO * voter._framed)
        if visible.sum() < sam_pick.MIN_HULL_ANCHORS or not hull.any():
            print(f"click in view {click['view']}: {int(visible.sum())} visibly selected, {int(hull.sum())} decided by the hull -- skipped")
            continue
        low, high = np.percentile(positions[visible], [1, 99], axis=0)
        distance = cKDTree(positions[visible]).query(positions[hull])[0] / float(np.linalg.norm(high - low))
        right, wrong = truth[hull], ~truth[hull]
        good_all.append(distance[right])
        bad_all.append(distance[wrong])
        quantiles = lambda d: np.round(np.quantile(d, [0.1, 0.5, 0.9]), 3).tolist() if len(d) else None
        per_click.append({"view": click["view"], "hull_decided": int(hull.sum()), "in_truth": int(right.sum()), "not_in_truth": int(wrong.sum()),
                          "distance_quantiles_10_50_90_in_truth": quantiles(distance[right]), "distance_quantiles_10_50_90_not_in_truth": quantiles(distance[wrong])})
        print(f"click in view {click['view']:3d}: hull-decided {hull.sum():6,} (in truth {right.sum():6,}, not {wrong.sum():6,}); "
              f"distance / diagonal, quantiles 10/50/90: in truth {quantiles(distance[right])}, not {quantiles(distance[wrong])}", flush=True)

    good = np.concatenate(good_all) if good_all else np.array([])
    bad = np.concatenate(bad_all) if bad_all else np.array([])
    print(f"\n{args.scene}, pooled over {len(per_click)} clicks: {len(good):,} hull-decided Gaussians in the truth, {len(bad):,} not")
    kept = []
    for radius in RADII:
        row = {"radius": radius, "kept_in_truth": float((good <= radius).mean()) if len(good) else None,
               "kept_not_in_truth": float((bad <= radius).mean()) if len(bad) else None}
        kept.append(row)
        print("  radius {:.3f} of the diagonal keeps {} of those in the truth and {} of those not".format(
            radius, "-" if row["kept_in_truth"] is None else f"{row['kept_in_truth']:.0%}", "-" if row["kept_not_in_truth"] is None else f"{row['kept_not_in_truth']:.0%}"))
    if args.out:
        Path(args.out).write_text(json.dumps({"scene": args.scene, "seed": args.seed, "clicks": per_click, "pooled": {"in_truth": int(len(good)), "not_in_truth": int(len(bad)), "kept": kept}}, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
