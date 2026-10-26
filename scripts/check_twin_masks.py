import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sam_point_segment import _build_cameras

from PIL import Image

from scene.dataset_readers import readColmapSceneInfo
from src.viewer.digital_twin import load_digital_twin_package
from src.viewer.sam_pick import evenly_spaced
from src.viewer.twin_provenance import CHECK_VIEWS, inside_mask_share, visible_gaussians


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--twin", required=True)
    parser.add_argument("--scene-path", required=True, help="COLMAP source of the scene; its masks/ folder holds the masks")
    parser.add_argument("--views", type=int, default=CHECK_VIEWS)
    args = parser.parse_args()

    package = load_digital_twin_package(args.twin)
    positions = package.recompute_gaussian_positions()
    infos = readColmapSceneInfo(args.scene_path, images="images", eval=False).train_cameras
    cameras = _build_cameras(infos, 512)

    masks_dir = Path(args.scene_path) / "masks"
    silhouettes, found = {}, 0
    for view in evenly_spaced(list(range(len(cameras))), args.views):
        path = masks_dir / (Path(infos[view].image_path).stem + ".png")
        if path.is_file():
            found += 1
            silhouettes[view] = np.asarray(Image.open(path).convert("L").resize((cameras[view].width, cameras[view].height))) > 127
    if not silhouettes:
        print("No object masks found for the checked views.", flush=True)
        sys.exit(2)

    visible = visible_gaussians(package.gaussian_colors, package.gaussian_opacities)
    share, considered = inside_mask_share(positions, cameras, silhouettes, max_views=args.views, considered=visible)
    print(f"{share:.1%} of {considered:,} visible Gaussians lie inside the masks of {found} views.", flush=True)
    print(f"TWINCHECK {share:.6f} {considered} {found} {found}", flush=True)


if __name__ == "__main__":
    main()
