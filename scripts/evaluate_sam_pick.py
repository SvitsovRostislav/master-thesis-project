import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sam_point_segment import REPO_ROOT, SamSession

from PIL import Image

from src.viewer import sam_pick
from src.viewer.digital_twin import load_digital_twin_package
from src.viewer.sam_pick import LEVEL_NAMES, SceneGeometry, _DepthMap, segment_object, spread_points
from src.viewer.sam_pick_metrics import carve_object, confusion_metrics, overlap_fraction, silhouette_agreement

DEFAULT_CHECKPOINT = REPO_ROOT / "data" / "checkpoints" / "sam_vit_b_01ec64.pth"
SPREAD_POINTS = 3


def _load_silhouettes(source: Path, session: SamSession) -> dict[int, np.ndarray]:
    silhouettes = {}
    for view, (info, camera) in enumerate(zip(session.cam_infos, session.cameras)):
        path = source / "masks" / (Path(info.image_path).stem + ".png")
        if not path.is_file():
            raise SystemExit(f"No object mask for view {view}: {path}")
        silhouettes[view] = np.asarray(Image.open(path).convert("L").resize((camera.width, camera.height))) > 127
    return silhouettes


def _gaussian_at_pixel(camera, positions: np.ndarray, x: float, y: float):
    depth_map = _DepthMap(camera, positions)
    visible = np.nonzero(depth_map.visible_points(neighbourhood=1))[0]
    if visible.size == 0:
        return None, np.inf
    distance = np.hypot(depth_map.pixels[visible, 0] - x, depth_map.pixels[visible, 1] - y)
    nearest = int(np.argmin(distance))
    return int(visible[nearest]), float(distance[nearest])


def _random_clicks(rng, session, positions, silhouettes, count: int) -> list[dict]:
    clicks, attempts = [], 0
    while len(clicks) < count and attempts < 50 * max(count, 1):
        attempts += 1
        view = int(rng.integers(len(session.cameras)))
        rows, cols = np.nonzero(silhouettes[view])
        if rows.size == 0:
            continue
        pick = int(rng.integers(rows.size))
        x, y = float(cols[pick]), float(rows[pick])
        gaussian, distance = _gaussian_at_pixel(session.cameras[view], positions, x, y)
        if gaussian is not None and distance <= 6.0:
            clicks.append({"view": view, "pixel": [x, y], "gaussian": gaussian})
    return clicks


def _visible_among(rng, session, positions, candidates: np.ndarray, attempts: int = 40):
    for _ in range(attempts):
        view = int(rng.integers(len(session.cameras)))
        depth_map = _DepthMap(session.cameras[view], positions)
        options = np.nonzero(depth_map.visible_points(neighbourhood=1) & candidates)[0]
        if options.size:
            return int(rng.choice(options))
    return None


def _steps_table(title: str, runs: list[dict]) -> None:
    longest = max(len(r["steps"]) for r in runs)
    print(f"\n{title}\n| Clicks after the first | Runs | Precision | Recall | IoU |\n| --- | --- | --- | --- | --- |")
    for k in range(longest):
        rows = [r["steps"][k] for r in runs if len(r["steps"]) > k]
        print(f"| {k} | {len(rows)} | {_mean_std([s['precision'] for s in rows])} | {_mean_std([s['recall'] for s in rows])} | "
              f"{_mean_std([s['iou'] for s in rows])} |")


def _mean_std(values) -> str:
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.2f} ± {values.std():.2f}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--twin", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--sam-checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--clicks", type=int, default=6, help="random clicks on the object, each run at every level")
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--views-2d", type=int, default=12, help="views the silhouette score is averaged over")
    parser.add_argument("--hull-off", action="store_true",
                        help="switch SAM Pick's visual-hull rule off for the selection (the ground truth keeps it)")
    parser.add_argument("--hull-radius", type=float, default=None,
                        help="restrict the visual-hull rule to Gaussians within this fraction of the visible "
                             "selection's diagonal of a visibly selected one (default: sam_pick.HULL_NEIGHBOURHOOD; "
                             "the ground truth is never restricted)")
    parser.add_argument("--min-ratio", type=float, default=None,
                        help="the vote a visible Gaussian needs (default: sam_pick.MAJORITY, a strict majority; "
                             "sam_pick.INCLUSIVE is 0.3); the ground truth keeps the majority")
    parser.add_argument("--hull-unrestricted", action="store_true",
                        help="the visual-hull rule as it was before the restriction was introduced")
    parser.add_argument("--add-clicks", type=int, default=0, help="simulated Add clicks after each starting click")
    parser.add_argument("--exclude-clicks", type=int, default=0, help="simulated Exclude clicks after each starting click")
    parser.add_argument("--click-min-opacity", type=float, default=0.0,
                        help="corrective (Add, Exclude) clicks land only on Gaussians at least this opaque (default: any)")
    parser.add_argument("--exclude-variant", default="click-subset", choices=["spread", "spread-subset", "click", "click-subset"],
                        help="how an Exclude click makes the selection again (see the top of this file)")
    parser.add_argument("--pair", action="append", default=[], metavar="NAME=VIEW:X:Y:LEVEL")
    parser.add_argument("--out", default=None, help="JSON file (default thesis/figures/sam_pick_eval_<scene>.json)")
    args = parser.parse_args()
    vote = {} if args.min_ratio is None else {"min_ratio": args.min_ratio}

    twin = Path(args.twin) if args.twin else REPO_ROOT / "data" / "gaussian_scenes" / f"{args.scene}_digital_twin.npz"
    source = Path(args.source) if args.source else REPO_ROOT / "data" / "gaussian_scenes" / "_colmap_src" / args.scene
    out = Path(args.out) if args.out else REPO_ROOT / "thesis" / "figures" / f"sam_pick_eval_{args.scene}.json"

    package = load_digital_twin_package(str(twin))
    positions = package.recompute_gaussian_positions()
    clickable = package.gaussian_opacities >= args.click_min_opacity
    session = SamSession(source, args.sam_checkpoint)
    cameras = session.cameras
    geometry = SceneGeometry(positions, cameras)
    silhouettes = _load_silhouettes(source, session)
    truth = carve_object(cameras, positions, silhouettes)
    if args.hull_off:
        sam_pick.HULL_RATIO = 2.0
    if args.hull_radius is not None:
        sam_pick.HULL_NEIGHBOURHOOD = args.hull_radius
    if args.hull_unrestricted:
        sam_pick.HULL_NEIGHBOURHOOD = None
    print(f"{args.scene}: {len(positions):,} Gaussians, {len(cameras)} views, true object = {int(truth.sum()):,} ({truth.mean():.1%})", flush=True)

    spread = np.linspace(0, len(cameras) - 1, min(args.views_2d, len(cameras))).round().astype(int)

    def score(selected: np.ndarray) -> dict:
        ratios = [silhouette_agreement(cameras[v], positions, selected, truth, silhouettes[v])["ratio"] for v in spread]
        return {**confusion_metrics(selected, truth), "silhouette_ratio": float(np.mean(ratios)), "selected": int(selected.sum())}

    def run(click: dict, level: int):
        started = time.perf_counter()
        result = segment_object(
            positions, cameras, positions[click["gaussian"]], session.predict, level=level, geometry=geometry,
            prefetch=session.prefetch, **vote,
        )
        return result, time.perf_counter() - started

    record = {"scene": args.scene, "gaussians": int(len(positions)), "true_object": int(truth.sum()), "seed": args.seed, "runs": [], "pairs": []}
    if args.min_ratio is not None:
        record["min_ratio"] = args.min_ratio
    if args.exclude_clicks:
        record["exclude_variant"] = args.exclude_variant
    if args.add_clicks or args.exclude_clicks:
        record["click_min_opacity"] = args.click_min_opacity

    clicks = _random_clicks(np.random.default_rng(args.seed), session, positions, silhouettes, args.clicks)
    for level in args.levels:
        for click in clicks:
            result, seconds = run(click, level)
            entry = {"level": level, "click": click, "seconds": seconds, "views": len(result.views_used), "rounds": result.rounds, **score(result.selected)}
            record["runs"].append(entry)
            print(f"  level {LEVEL_NAMES[level]:6s} view {click['view']:3d}: recall {entry['recall']:.2f} precision {entry['precision']:.2f} "
                  f"IoU {entry['iou']:.2f} silhouette {entry['silhouette_ratio']:.2f}  {seconds:.1f} s, {entry['views']} views", flush=True)

    if record["runs"]:
        print("\n| Level | Precision | Recall | IoU | Silhouette ratio | Seconds | Views |\n| --- | --- | --- | --- | --- | --- | --- |")
        for level in args.levels:
            rows = [r for r in record["runs"] if r["level"] == level]
            print(f"| {LEVEL_NAMES[level]} | {_mean_std([r['precision'] for r in rows])} | {_mean_std([r['recall'] for r in rows])} | "
                  f"{_mean_std([r['iou'] for r in rows])} | {_mean_std([r['silhouette_ratio'] for r in rows])} | "
                  f"{np.mean([r['seconds'] for r in rows]):.1f} | {np.mean([r['views'] for r in rows]):.0f} |")

    if args.add_clicks or args.exclude_clicks:
        refine_rng = np.random.default_rng(args.seed + 1)
        record["add"], record["exclude"] = [], []
        for level in args.levels:
            for click in clicks:
                start = segment_object(
                    positions, cameras, positions[click["gaussian"]], session.predict, level=level, geometry=geometry,
                    prefetch=session.prefetch, **vote,
                ).selected

                selected, steps = start.copy(), [score(start)]
                for _ in range(args.add_clicks):
                    missed = _visible_among(refine_rng, session, positions, truth & ~selected & clickable)
                    if missed is None:
                        break
                    found = segment_object(
                        positions, cameras, positions[missed], session.predict, level=level, geometry=geometry,
                        prefetch=session.prefetch, **vote,
                    )
                    selected = selected | found.selected
                    steps.append(score(selected))
                if args.add_clicks:
                    record["add"].append({"level": level, "click": click, "steps": steps})

                selected, steps, excluded = start.copy(), [score(start)], []
                for _ in range(args.exclude_clicks):
                    wrong = _visible_among(refine_rng, session, positions, selected & ~truth & clickable)
                    if wrong is None:
                        break
                    excluded.append(wrong)
                    if args.exclude_variant.startswith("click"):
                        first, extra, negatives = positions[click["gaussian"]], None, positions[excluded]
                    else:
                        chosen = spread_points(positions[selected], SPREAD_POINTS)
                        first, extra, negatives = chosen[0], chosen[1:], positions[wrong][None]
                    again = segment_object(
                        positions, cameras, first, session.predict, level=level, geometry=geometry,
                        extra_points=extra, negative_points=negatives, prefetch=session.prefetch, **vote,
                    )
                    selected = again.selected & selected if args.exclude_variant.endswith("-subset") else again.selected
                    steps.append(score(selected))
                if args.exclude_clicks:
                    record["exclude"].append({"level": level, "click": click, "steps": steps})
            for name, runs in (("Add", [r for r in record["add"] if r["level"] == level]),
                               ("Exclude", [r for r in record["exclude"] if r["level"] == level])):
                if runs:
                    _steps_table(f"{name} clicks, level {LEVEL_NAMES[level]}", runs)

    if args.pair:
        selections = {}
        for spec in args.pair:
            name, _, rest = spec.partition("=")
            view, fx, fy, level = rest.split(":")
            camera = cameras[int(view)]
            gaussian, distance = _gaussian_at_pixel(camera, positions, float(fx) * camera.width, float(fy) * camera.height)
            result, seconds = run({"gaussian": gaussian}, int(level))
            selections[name] = result.selected
            np.save(out.with_name(f"{out.stem}_{name}.npy"), result.selected)
            record["pairs"].append({"name": name, "spec": spec, "distance_px": distance, "seconds": seconds, **score(result.selected)})
            print(f"pair {name}: {int(result.selected.sum()):,} Gaussians ({result.selected.mean():.1%}), {seconds:.1f} s")
        names = list(selections)
        union = np.zeros(len(positions), dtype=bool)
        for selection in selections.values():
            union |= selection
        summary = {"union": score(union), "overlap": {f"{a}/{b}": overlap_fraction(selections[a], selections[b]) for i, a in enumerate(names) for b in names[i + 1:]}}
        record["pair_summary"] = summary
        print(f"together: recall {summary['union']['recall']:.2f}, silhouette {summary['union']['silhouette_ratio']:.2f}; overlap {summary['overlap']}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"\nembedding cache: {session.embedding_hits} hits, {session.embedding_misses} misses; "
          f"planning maps: {geometry.planning_hits} hits, {geometry.planning_misses} misses; wrote {out}")


if __name__ == "__main__":
    main()
