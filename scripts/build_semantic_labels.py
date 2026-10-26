import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SUGAR_ROOT = REPO_ROOT / "external" / "SuGaR"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SUGAR_ROOT))
sys.path.insert(0, str(SUGAR_ROOT / "gaussian_splatting"))

from src.viewer.digital_twin import load_digital_twin_package
from src.viewer.semantic_segmentation import Camera, semantic_labels_path, vote_gaussian_labels

from gaussian_splatting.scene.dataset_readers import readColmapSceneInfo


def _load_cameras(scene_path: str) -> tuple[list[Camera], list[str]]:
    """Reads the same COLMAP cameras train.py itself trains against. Note
    dataset_readers.CameraInfo.R is stored *transposed* relative to the
    actual world-to-camera rotation ("R is stored transposed due to 'glm'
    in CUDA code" -- see gaussian_splatting/scene/dataset_readers.py), so
    it has to be transposed back here."""
    scene_info = readColmapSceneInfo(scene_path, images="images", eval=False)
    cameras, image_paths = [], []
    for cam_info in scene_info.train_cameras:
        rotation_w2c = np.asarray(cam_info.R, dtype=np.float64).T
        translation_w2c = np.asarray(cam_info.T, dtype=np.float64)
        cameras.append(
            Camera(
                rotation=rotation_w2c,
                translation=translation_w2c,
                fx=cam_info.width / (2.0 * np.tan(cam_info.FovX / 2.0)),
                fy=cam_info.height / (2.0 * np.tan(cam_info.FovY / 2.0)),
                cx=cam_info.width / 2.0,
                cy=cam_info.height / 2.0,
                width=cam_info.width,
                height=cam_info.height,
            )
        )
        image_paths.append(cam_info.image_path)
    return cameras, image_paths


def _segment_images(image_paths: list[str], checkpoint: str, model_type: str) -> list[np.ndarray]:
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise SystemExit(
            "segment-anything is not installed in this environment.\n"
            "Install it in gs-train with: pip install segment-anything\n"
            "and download a checkpoint from the official SAM repo before running this script."
        ) from exc
    import torch
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    generator = SamAutomaticMaskGenerator(sam)

    masks = []
    for path in image_paths:
        image = np.array(Image.open(path).convert("RGB"))
        instance_masks = generator.generate(image)
        label_image = np.zeros(image.shape[:2], dtype=np.int32)
        for label, m in enumerate(sorted(instance_masks, key=lambda d: -d["area"]), start=1):
            label_image[m["segmentation"]] = label
        masks.append(label_image)
        print(f"  {Path(path).name}: {len(instance_masks)} SAM regions")
    return masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-path", required=True, help="Normalized COLMAP source (same one train_full_pipeline.py used)")
    parser.add_argument("--digital-twin", required=True, help="Path to the exported .npz to label")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--sam-model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--out", default=None, help="Defaults to <digital-twin>_semantic_labels.npy")
    args = parser.parse_args()

    print("Loading digital-twin package...")
    package = load_digital_twin_package(args.digital_twin)
    positions = package.recompute_gaussian_positions()

    print("Loading COLMAP cameras...")
    cameras, image_paths = _load_cameras(args.scene_path)
    print(f"{len(cameras)} training cameras.")

    print("Running Segment Anything on each training photograph...")
    masks = _segment_images(image_paths, args.sam_checkpoint, args.sam_model_type)

    print("Lifting 2D masks into per-Gaussian labels via multi-view voting...")
    labels = vote_gaussian_labels(positions, cameras, masks)

    out_path = Path(args.out) if args.out else semantic_labels_path(args.digital_twin)
    np.save(out_path, labels)
    n_objects = int(labels.max())
    n_unlabeled = int((labels == 0).sum())
    print(
        f"Wrote {out_path} -- {n_objects} object label(s) found, "
        f"{n_unlabeled:,}/{labels.shape[0]:,} Gaussians unlabeled."
    )


if __name__ == "__main__":
    main()
