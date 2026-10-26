import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_sam_pick as ev
from sam_point_segment import REPO_ROOT, SamSession

import src.viewer.sam_pick as sam_pick
from src.viewer.digital_twin import load_digital_twin_package
from src.viewer.sam_pick_metrics import carve_object


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--clicks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--click-min-opacity", type=float, default=0.0,
                        help="the added click lands only on Gaussians at least this opaque (default: any)")
    args = parser.parse_args()

    package = load_digital_twin_package(str(REPO_ROOT / "data" / "gaussian_scenes" / f"{args.scene}_digital_twin.npz"))
    positions = package.recompute_gaussian_positions()
    brightness, opacity = package.gaussian_colors.mean(axis=1), package.gaussian_opacities
    source = REPO_ROOT / "data" / "gaussian_scenes" / "_colmap_src" / args.scene
    session = SamSession(source, REPO_ROOT / "data" / "checkpoints" / "sam_vit_b_01ec64.pth")
    silhouettes = ev._load_silhouettes(source, session)
    truth = carve_object(session.cameras, positions, silhouettes)
    geometry = sam_pick.SceneGeometry(positions, session.cameras)
    clicks = ev._random_clicks(np.random.default_rng(args.seed), session, positions, silhouettes, args.clicks)
    refine_rng = np.random.default_rng(args.seed + 1)
    clickable = opacity >= args.click_min_opacity

    def describe(name: str, chosen: np.ndarray) -> None:
        if not chosen.any():
            print(f"    {name}: none")
            return
        print(f"    {name}: {int(chosen.sum()):7,d} Gaussians, brightness {brightness[chosen].mean():.3f}, "
              f"dark (<0.1) {np.mean(brightness[chosen] < 0.1):.0%}, opacity {opacity[chosen].mean():.2f}, "
              f"opaque (>0.5) {np.mean(opacity[chosen] > 0.5):.0%}")

    print(f"{args.scene}: true object {int(truth.sum()):,} of {len(positions):,}; brightness of the true object {brightness[truth].mean():.3f}")
    for click in clicks:
        start = sam_pick.segment_object(
            positions, session.cameras, positions[click["gaussian"]], session.predict, level=2, geometry=geometry,
            prefetch=session.prefetch,
        ).selected
        missed = ev._visible_among(refine_rng, session, positions, truth & ~start & clickable)
        if missed is None:
            print(f"click in view {click['view']}: nothing missed to click on")
            continue
        found = sam_pick.segment_object(
            positions, session.cameras, positions[missed], session.predict, level=2, geometry=geometry,
            prefetch=session.prefetch,
        ).selected
        after = start | found
        before_precision = (start & truth).sum() / max(start.sum(), 1)
        after_precision = (after & truth).sum() / max(after.sum(), 1)
        print(f"click in view {click['view']:3d}: precision {before_precision:.2f} -> {after_precision:.2f}; the clicked Gaussian has "
              f"brightness {brightness[missed]:.3f}, opacity {opacity[missed]:.2f}")
        describe("added by the click, wrong", found & ~start & ~truth)
        describe("added by the click, right", found & ~start & truth)


if __name__ == "__main__":
    main()
