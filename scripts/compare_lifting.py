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

from src.viewer import sam_pick
from src.viewer.digital_twin import load_digital_twin_package
from src.viewer.sam_pick import SceneGeometry, segment_object
from src.viewer.sam_pick_metrics import carve_object, confusion_metrics

GAMMAS = (-0.4, 0.0, 0.4)
MIN_RATIOS = (0.3, 0.4)


class _RecordingVoter(sam_pick.Voter):

    last = None

    def __init__(self, positions):
        super().__init__(positions)
        self.masks = []
        _RecordingVoter.last = self

    def add(self, camera, mask):
        super().add(camera, mask)
        self.masks.append((camera, mask))


def contributions(twin: dict, views: list, masks: list[tuple[int, np.ndarray]]) -> torch.Tensor:
    count = twin["means"].shape[0]
    totals = torch.zeros((count, 2), dtype=torch.float64, device="cuda")
    for view, mask in masks:
        colors = torch.zeros((count, 3), dtype=torch.float32, device="cuda", requires_grad=True)
        image = _render("twin", {**twin, "colors": colors}, views[view])
        inside = torch.tensor(mask, dtype=torch.float32, device="cuda")
        ((image[0] * inside).sum() + (image[1] * (1.0 - inside)).sum()).backward()
        totals[:, 1] += colors.grad[:, 0].double()
        totals[:, 0] += colors.grad[:, 1].double()
    return totals


def flashsplat(totals: torch.Tensor, gamma: float) -> np.ndarray:
    total = totals.sum(dim=1)
    seen = total > 0
    share_in = torch.where(seen, totals[:, 1] / total.clamp_min(1e-30), torch.zeros_like(total))
    return (seen & (share_in > (1.0 - share_in) + gamma)).cpu().numpy()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--twin", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--sam-checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--clicks", type=int, default=6)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="JSON file (default thesis/figures/lifting_comparison_<scene>.json)")
    args = parser.parse_args()

    twin_path = Path(args.twin) if args.twin else REPO_ROOT / "data" / "gaussian_scenes" / f"{args.scene}_digital_twin.npz"
    source = Path(args.source) if args.source else REPO_ROOT / "data" / "gaussian_scenes" / "_colmap_src" / args.scene
    out = Path(args.out) if args.out else REPO_ROOT / "thesis" / "figures" / f"lifting_comparison_{args.scene}.json"

    package = load_digital_twin_package(str(twin_path))
    positions = package.recompute_gaussian_positions()
    session = SamSession(source, args.sam_checkpoint)
    cameras = session.cameras
    camera_index = {id(camera): index for index, camera in enumerate(cameras)}
    geometry = SceneGeometry(positions, cameras)
    silhouettes = _load_silhouettes(source, session)
    truth = carve_object(cameras, positions, silhouettes)
    views = [_View(session.cam_infos[v], cameras[v].width, cameras[v].height) for v in range(len(cameras))]
    truths = [torch.tensor(silhouettes[v], device="cuda") for v in range(len(cameras))]
    twin = _twin_tensors(package)

    def score(selected: np.ndarray) -> dict:
        scores = _mask_scores(twin, selected, views, truths)
        return {"selected": int(selected.sum()), "iou_2d": scores["iou"], "accuracy_2d": scores["accuracy"],
                **{f"{k}_3d": v for k, v in confusion_metrics(selected, truth).items()}}

    record = {"scene": args.scene, "gaussians": int(len(positions)), "true_object": int(truth.sum()), "seed": args.seed,
              "level": args.level, "gammas": list(GAMMAS), "views": len(cameras), "clicks": []}

    # the rules alone: the object mask of every view
    started = time.perf_counter()
    with torch.enable_grad():
        totals = contributions(twin, views, [(v, silhouettes[v]) for v in range(len(cameras))])
    seconds = time.perf_counter() - started
    record["object_masks"] = {"sam_pick_rule": score(truth), "seconds_flashsplat": seconds,
                              **{f"sam_pick_rule_{r:.1f}": score(sam_pick.vote_selection(cameras, positions, silhouettes, min_ratio=r))
                                 for r in MIN_RATIOS},
                              **{f"flashsplat_{g:+.1f}": score(flashsplat(totals, g)) for g in GAMMAS}}
    line = "  ".join(f"{k}: 2D {v['iou_2d']:.3f} 3D {v['iou_3d']:.2f}" for k, v in record["object_masks"].items() if isinstance(v, dict))
    print(f"{args.scene}, object masks of all {len(cameras)} views -- {line}", flush=True)

    saved_voter = sam_pick.Voter
    sam_pick.Voter = _RecordingVoter
    try:
        clicks = _random_clicks(np.random.default_rng(args.seed), session, positions, silhouettes, args.clicks)
        for click in clicks:
            started = time.perf_counter()
            result = segment_object(positions, cameras, positions[click["gaussian"]], session.predict, level=args.level,
                                    geometry=geometry, prefetch=session.prefetch)
            seconds_pick = time.perf_counter() - started
            voter = _RecordingVoter.last
            masks = [(camera_index[id(camera)], mask) for camera, mask in voter.masks]
            again = saved_voter(positions)
            for view, mask in masks:
                again.add(cameras[view], mask)
            assert np.array_equal(again.selection(neighbourhood=sam_pick.HULL_NEIGHBOURHOOD), result.selected), \
                "the recorded masks do not reproduce SAM Pick's selection"
            hidden = result.selected & (voter._seen == 0)

            started = time.perf_counter()
            with torch.enable_grad():
                totals = contributions(twin, views, masks)
            seconds_fs = time.perf_counter() - started
            entry = {"click": click, "views_voted": len(masks), "seconds_sam_pick": seconds_pick,
                     "seconds_flashsplat_lifting": seconds_fs, "hidden_selected_by_hull": int(hidden.sum()),
                     "sam_pick": score(result.selected),
                     **{f"sam_pick_{r:.1f}": score(again.selection(min_ratio=r, neighbourhood=sam_pick.HULL_NEIGHBOURHOOD))
                        for r in MIN_RATIOS},
                     **{f"flashsplat_{g:+.1f}": score(flashsplat(totals, g)) for g in GAMMAS},
                     "flashsplat_+0.0_with_hull": score(flashsplat(totals, 0.0) | hidden)}
            record["clicks"].append(entry)
            print(f"  view {click['view']:3d}: {len(masks)} masks; SAM Pick 2D {entry['sam_pick']['iou_2d']:.3f} / 3D "
                  f"{entry['sam_pick']['iou_3d']:.2f};  " + "  ".join(
                      f"FS{g:+.1f} 2D {entry[f'flashsplat_{g:+.1f}']['iou_2d']:.3f} / 3D {entry[f'flashsplat_{g:+.1f}']['iou_3d']:.2f}"
                      for g in GAMMAS) + f";  FS+hull 2D {entry['flashsplat_+0.0_with_hull']['iou_2d']:.3f} / 3D "
                  f"{entry['flashsplat_+0.0_with_hull']['iou_3d']:.2f}  ({seconds_fs:.2f} s lifting)", flush=True)
    finally:
        sam_pick.Voter = saved_voter

    keys = ["sam_pick", *[f"sam_pick_{r:.1f}" for r in MIN_RATIOS], *[f"flashsplat_{g:+.1f}" for g in GAMMAS],
            "flashsplat_+0.0_with_hull"]
    print("\n| Rule | 2D IoU | 3D IoU | 3D recall | 3D precision |\n| --- | --- | --- | --- | --- |")
    for key in keys:
        rows = [c[key] for c in record["clicks"]]
        print(f"| {key} | {np.mean([r['iou_2d'] for r in rows]):.3f} | {np.mean([r['iou_3d'] for r in rows]):.2f} | "
              f"{np.mean([r['recall_3d'] for r in rows]):.2f} | {np.mean([r['precision_3d'] for r in rows]):.2f} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
