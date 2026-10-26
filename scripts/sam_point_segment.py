import argparse
import json
import math
import os
import sys
import time
import traceback
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SUGAR_ROOT = REPO_ROOT / "external" / "SuGaR"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SUGAR_ROOT))
sys.path.insert(0, str(SUGAR_ROOT / "gaussian_splatting"))

import torch
from PIL import Image

from src.viewer.sam_pick import MAJORITY, SceneGeometry, segment_object
from src.viewer.semantic_segmentation import Camera

from scene.dataset_readers import readColmapSceneInfo

_fraction = os.environ.get("SUGAR_GPU_MEMORY_FRACTION")
if _fraction and torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(float(_fraction))

EMBEDDING_CACHE_VIEWS = 96


def _say(message: str) -> None:
    print(message, flush=True)


def _build_cameras(cam_infos, max_side: int) -> list[Camera]:
    cameras = []
    for info in cam_infos:
        scale = min(1.0, max_side / max(info.width, info.height))
        width, height = max(1, round(info.width * scale)), max(1, round(info.height * scale))
        cameras.append(
            Camera(
                rotation=np.asarray(info.R, dtype=np.float64).T,
                translation=np.asarray(info.T, dtype=np.float64),
                fx=width / (2.0 * math.tan(info.FovX / 2.0)),
                fy=height / (2.0 * math.tan(info.FovY / 2.0)),
                cx=width / 2.0,
                cy=height / 2.0,
                width=width,
                height=height,
            )
        )
    return cameras


class SamSession:

    def __init__(self, scene_path, checkpoint, model_type: str = "vit_b", max_side: int = 1024,
                 cache_views: int = EMBEDDING_CACHE_VIEWS):
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise SystemExit("segment-anything is not installed in this environment (pip install segment-anything).") from exc

        _say("Loading COLMAP cameras...")
        self.cam_infos = readColmapSceneInfo(str(scene_path), images="images", eval=False).train_cameras
        self.cameras = _build_cameras(self.cam_infos, max_side)
        _say(f"{len(self.cameras)} cameras")

        _say(f"Loading SAM ({model_type})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._predictor = SamPredictor(sam_model_registry[model_type](checkpoint=str(checkpoint)).to(device))
        self._embeddings: OrderedDict = OrderedDict()
        self._cache_views = cache_views
        self.embedding_hits = 0
        self.embedding_misses = 0
        self._reads = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sam-pick-read")
        self._pending: dict[int, Future] = {}

    def _read_photograph(self, view: int) -> np.ndarray:
        camera = self.cameras[view]
        photograph = Image.open(self.cam_infos[view].image_path).convert("RGB").resize((camera.width, camera.height))
        return np.asarray(photograph)

    def prefetch(self, views) -> None:
        for view in views:
            if view not in self._embeddings and view not in self._pending:
                self._pending[view] = self._reads.submit(self._read_photograph, view)

    def _select_image(self, view: int) -> None:
        predictor, cached = self._predictor, self._embeddings.get(view)
        if cached is not None:
            self._embeddings.move_to_end(view)
            predictor.features, predictor.original_size, predictor.input_size = cached
            predictor.is_image_set = True
            self.embedding_hits += 1
            return
        future = self._pending.pop(view, None)
        photograph = future.result() if future is not None else self._read_photograph(view)
        predictor.set_image(photograph)
        self._embeddings[view] = (predictor.features, predictor.original_size, predictor.input_size)
        while len(self._embeddings) > self._cache_views:
            self._embeddings.popitem(last=False)
        self.embedding_misses += 1

    def predict(self, view: int, points: np.ndarray, negatives: np.ndarray | None = None) -> np.ndarray:
        coordinates = np.asarray(points, dtype=np.float32)
        labels = np.ones(len(coordinates), dtype=np.int32)
        if negatives is not None and len(negatives):
            coordinates = np.concatenate([coordinates, np.asarray(negatives, dtype=np.float32)])
            labels = np.concatenate([labels, np.zeros(len(negatives), dtype=np.int32)])
        with torch.inference_mode():
            self._select_image(view)
            masks, _, _ = self._predictor.predict(
                point_coords=coordinates, point_labels=labels, multimask_output=len(coordinates) == 1
            )
        return masks


def serve(args) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    positions = np.load(args.positions).astype(np.float32)
    session = SamSession(args.scene_path, args.sam_checkpoint, args.sam_model_type, args.max_side)
    geometry = SceneGeometry(positions, session.cameras)
    _say(f"Ready. {positions.shape[0]:,} Gaussians.")

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            started = time.time()
            extra, negatives = request.get("extra"), request.get("negatives")
            min_ratio = float(request.get("min_ratio", MAJORITY))
            result = segment_object(
                positions, session.cameras, np.asarray(request["point"], dtype=np.float64), session.predict,
                level=int(request["level"]), max_views=args.max_views, progress=_say, geometry=geometry,
                extra_points=np.asarray(extra, dtype=np.float64) if extra else None,
                negative_points=np.asarray(negatives, dtype=np.float64) if negatives else None,
                prefetch=session.prefetch, min_ratio=min_ratio,
            )
            np.save(request["output"], result.selected)
            _say(f"Finished in {time.time() - started:.1f} s (embeddings so far: {session.embedding_hits} reused, "
                 f"{session.embedding_misses} computed).")
            print(f"SAMPICK {int(result.selected.sum())} {len(result.views_used)} {result.rounds} {request['output']}", flush=True)
        except Exception as error:  # a bad request must not end the worker
            traceback.print_exc()
            print("SAMPICK_ERROR " + f"{type(error).__name__}: {error}".replace("\n", " "), flush=True)
        finally:
            torch.cuda.empty_cache()  # keep the weights and embeddings, give back the activations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-path", required=True, help="COLMAP source the cameras and photographs are read from")
    parser.add_argument("--input", help="request .npz (one-shot mode)")
    parser.add_argument("--output", help="result .npy (one-shot mode)")
    parser.add_argument("--serve", action="store_true", help="stay alive and answer requests from stdin")
    parser.add_argument("--positions", help="(N, 3) .npy of Gaussian centres (serve mode)")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--sam-model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--max-side", type=int, default=1024, help="longest photograph side shown to SAM, in pixels")
    parser.add_argument("--max-views", type=int, default=36, help="most photographs SAM is asked about")
    args = parser.parse_args()

    if args.serve:
        if not args.positions:
            parser.error("--serve needs --positions")
        return serve(args)
    if not (args.input and args.output):
        parser.error("one-shot mode needs --input and --output (or use --serve)")

    started = time.time()
    request = np.load(args.input)
    positions, point, level = request["positions"], request["point"], int(request["level"])
    _say(f"{positions.shape[0]:,} Gaussians, level {level}, click at {np.round(point, 3).tolist()}")

    session = SamSession(args.scene_path, args.sam_checkpoint, args.sam_model_type, args.max_side)
    result = segment_object(
        positions, session.cameras, point, session.predict, level=level, max_views=args.max_views, progress=_say,
        prefetch=session.prefetch,
    )

    np.save(args.output, result.selected)
    _say(f"Finished in {time.time() - started:.0f} s.")
    print(f"SAMPICK {int(result.selected.sum())} {len(result.views_used)} {result.rounds} {args.output}", flush=True)


if __name__ == "__main__":
    main()
