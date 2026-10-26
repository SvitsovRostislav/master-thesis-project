import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from .sam_pick import evenly_spaced, project
from .semantic_segmentation import Camera

SCHEMA_VERSION = 1

MASK_SHARE_WARNING = 0.5
CHECK_VIEWS = 24
MIN_VIEWS_FRAMED = 3
VISIBLE_OPACITY = 0.3
VISIBLE_BRIGHTNESS = 0.1


def visible_gaussians(colors: np.ndarray, opacities: np.ndarray) -> np.ndarray:
    return (opacities > VISIBLE_OPACITY) & (colors.mean(axis=1) > VISIBLE_BRIGHTNESS)


def source_masking(normalized_source) -> tuple[int, int, bool]:
    source = Path(normalized_source)
    images, masks = source / "images", source / "masks"
    n_images = sum(1 for p in images.iterdir() if p.is_file()) if images.is_dir() else 0
    n_masks = sum(1 for p in masks.iterdir() if p.suffix.lower() == ".png") if masks.is_dir() else 0
    images_is_junction = images.is_dir() and os.path.normcase(os.path.realpath(images)) != os.path.normcase(os.path.abspath(images))
    return n_images, n_masks, n_masks > 0 and not images_is_junction


def build_record(scene: str, source, preset: dict, now: datetime | None = None) -> dict:
    images, masks, masked = source_masking(source)
    return {
        "schema": SCHEMA_VERSION,
        "built_at": (now or datetime.now()).isoformat(timespec="seconds"),
        "scene": scene,
        "source": str(source),
        "photographs": images,
        "masks_available": masks,
        "masked": masked,
        "preset": dict(preset),
    }


def with_counts(record: dict, n_vertices: int, n_faces: int, n_gaussians: int) -> dict:
    return {**record, "n_vertices": int(n_vertices), "n_faces": int(n_faces), "n_gaussians": int(n_gaussians)}


def describe(record: dict | None) -> str:
    if not record:
        return "No build record in this twin (it was built before records existed)."
    preset = record.get("preset", {})
    iterations = "/".join(str(preset.get(k, "?")) for k in ("baseline_iterations", "coarse_iterations", "refine_iterations"))
    masking = "masked" if record.get("masked") else "not masked"
    if record.get("masks_available") and not record.get("masked"):
        masking += f" (the scene has {record['masks_available']} masks)"
    return (
        f"Built {record.get('built_at', '?').replace('T', ' ')} from {record.get('photographs', '?')} photographs, "
        f"{masking}; iterations {iterations}; mesh target {preset.get('mesh_triangles_target', '?')} triangles."
    )


def inside_mask_share(
    positions: np.ndarray,
    cameras: list[Camera],
    silhouettes: dict[int, np.ndarray],
    max_views: int = CHECK_VIEWS,
    considered: np.ndarray | None = None,
) -> tuple[float, int]:
    framed = np.zeros(positions.shape[0], dtype=np.int32)
    inside = np.zeros(positions.shape[0], dtype=np.int32)
    for view in evenly_spaced(sorted(silhouettes), max_views):
        camera, mask = cameras[view], silhouettes[view]
        pixels, _, in_frame = project(camera, positions)
        idx = np.nonzero(in_frame)[0]
        height, width = mask.shape
        x = np.clip((pixels[idx, 0] * width / camera.width).astype(int), 0, width - 1)
        y = np.clip((pixels[idx, 1] * height / camera.height).astype(int), 0, height - 1)
        framed[idx] += 1
        inside[idx] += mask[y, x]
    known = framed >= MIN_VIEWS_FRAMED
    if considered is not None:
        known &= considered
    if not known.any():
        return 0.0, 0
    return float(np.mean(inside[known] >= 0.5 * framed[known])), int(known.sum())


def assess(record: dict | None, share: float | None) -> str | None:
    """A warning for the user, or None when nothing is wrong (or nothing is known)."""
    if share is None or share >= MASK_SHARE_WARNING:
        return None
    text = f"only {share:.0%} of this twin's visible Gaussians lie inside its scene's object masks"
    if record and record.get("masked"):
        return text + ", although its build record says it was built with masking"
    return text + ": it looks built without masking, or holds background"



def check_cache_path(twin_path) -> Path:
    twin_path = Path(twin_path)
    return twin_path.with_name(twin_path.stem + "_mask_check.json")


def load_cached_check(twin_path) -> dict | None:
    path = check_cache_path(twin_path)
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        stat = Path(twin_path).stat()
        if stored["twin_size"] == stat.st_size and stored["twin_mtime_ns"] == stat.st_mtime_ns:
            return {"share": float(stored["share"]), "considered": int(stored["considered"]), "views": int(stored["views"])}
    except (OSError, ValueError, KeyError):
        pass
    return None


def save_cached_check(twin_path, share: float, considered: int, views: int) -> None:
    try:
        stat = Path(twin_path).stat()
        payload = {
            "share": share, "considered": considered, "views": views,
            "twin_size": stat.st_size, "twin_mtime_ns": stat.st_mtime_ns,
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }
        check_cache_path(twin_path).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    except OSError:
        pass
