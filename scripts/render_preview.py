import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SUGAR_ROOT = REPO_ROOT / "external" / "SuGaR"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SUGAR_ROOT))
sys.path.insert(0, str(SUGAR_ROOT / "gaussian_splatting"))

import torch
from PIL import Image

from src.viewer.digital_twin import load_digital_twin_package

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.dataset_readers import readColmapSceneInfo
from utils.graphics_utils import getProjectionMatrix, getWorld2View2

ZNEAR, ZFAR = 0.01, 100.0

_fraction = os.environ.get("SUGAR_GPU_MEMORY_FRACTION")
if _fraction and torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(float(_fraction))


def _pick_views(n_cameras: int, num_views: int) -> list[int]:
    if n_cameras <= num_views:
        return list(range(n_cameras))
    return sorted(set(np.linspace(0, n_cameras - 1, num_views).round().astype(int).tolist()))


def _camera_tensors(cam_info, max_side: int) -> dict:
    """The matrices the rasterizer wants, built exactly as scene/cameras.py does.
    Only the pixel size changes with max_side -- the field of view is kept."""
    scale = max_side / max(cam_info.width, cam_info.height)
    width = max(1, int(round(cam_info.width * scale)))
    height = max(1, int(round(cam_info.height * scale)))
    world_view = torch.tensor(getWorld2View2(cam_info.R, cam_info.T)).transpose(0, 1).cuda()
    projection = getProjectionMatrix(ZNEAR, ZFAR, cam_info.FovX, cam_info.FovY).transpose(0, 1).cuda()
    full_proj = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    return {
        "width": width,
        "height": height,
        "tanfovx": math.tan(cam_info.FovX * 0.5),
        "tanfovy": math.tan(cam_info.FovY * 0.5),
        "world_view": world_view,
        "full_proj": full_proj,
        "center": world_view.inverse()[3, :3],
    }


def _load_gaussians(path: str) -> dict:
    package = load_digital_twin_package(path)
    quats = torch.tensor(package.gaussian_quaternions, dtype=torch.float32, device="cuda")
    return {
        "means": torch.tensor(package.recompute_gaussian_positions(), dtype=torch.float32, device="cuda"),
        "colors": torch.tensor(package.gaussian_colors, dtype=torch.float32, device="cuda"),
        "opacities": torch.tensor(package.gaussian_opacities, dtype=torch.float32, device="cuda").reshape(-1, 1),
        "scales": torch.tensor(package.gaussian_scales, dtype=torch.float32, device="cuda"),
        # The CUDA kernel assumes unit quaternions (its own normalization is commented out).
        "rotations": torch.nn.functional.normalize(quats, dim=1),
    }


@torch.no_grad()
def _render(gaussians: dict, cam: dict, background: float) -> np.ndarray:
    settings = GaussianRasterizationSettings(
        image_height=cam["height"],
        image_width=cam["width"],
        tanfovx=cam["tanfovx"],
        tanfovy=cam["tanfovy"],
        bg=torch.full((3,), background, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=cam["world_view"],
        projmatrix=cam["full_proj"],
        sh_degree=0,
        campos=cam["center"],
        prefiltered=False,
        debug=False,
    )
    image, _ = GaussianRasterizer(raster_settings=settings)(
        means3D=gaussians["means"],
        means2D=torch.zeros_like(gaussians["means"]),
        shs=None,
        colors_precomp=gaussians["colors"],
        opacities=gaussians["opacities"],
        scales=gaussians["scales"],
        rotations=gaussians["rotations"],
        cov3D_precomp=None,
    )
    return (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-path", required=True, help="COLMAP source the cameras are read from")
    parser.add_argument("--package", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--max-side", type=int, default=800, help="longest image side in pixels")
    parser.add_argument("--white-background", action="store_true")
    args = parser.parse_args()

    packages = []
    for spec in args.package:
        name, _, path = spec.partition("=")
        if not name or not path:
            raise SystemExit(f"--package expects NAME=PATH, got {spec!r}")
        packages.append((name, path))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading COLMAP cameras...", flush=True)
    scene_info = readColmapSceneInfo(args.scene_path, images="images", eval=False)
    cameras = scene_info.train_cameras
    views = _pick_views(len(cameras), args.num_views)
    print(f"{len(cameras)} cameras, rendering views {views}", flush=True)
    camera_tensors = {v: _camera_tensors(cameras[v], args.max_side) for v in views}
    background = 1.0 if args.white_background else 0.0

    for name, path in packages:
        print(f"Loading package '{name}'...", flush=True)
        gaussians = _load_gaussians(path)
        for v in views:
            pixels = _render(gaussians, camera_tensors[v], background)
            out_path = out_dir / f"{name}_{v:04d}.png"
            Image.fromarray(pixels).save(out_path)
            print(f"RENDERED {name} {v} {out_path}", flush=True)
        del gaussians
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
