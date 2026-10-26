import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SUGAR_ROOT = REPO_ROOT / "external" / "SuGaR"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SUGAR_ROOT))
sys.path.insert(0, str(SUGAR_ROOT / "gaussian_splatting"))

import torch
from PIL import Image

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_renderer import render as render_3dgs
from lpipsPyTorch import lpips
from scene.dataset_readers import readColmapSceneInfo
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getProjectionMatrix, getWorld2View2
from utils.image_utils import psnr
from utils.loss_utils import ssim

from src.viewer.digital_twin import load_digital_twin_package

ZNEAR, ZFAR = 0.01, 100.0


class _View:
    def __init__(self, info, width: int, height: int):
        self.FoVx, self.FoVy = info.FovX, info.FovY
        self.image_width, self.image_height = width, height
        self.world_view_transform = torch.tensor(getWorld2View2(info.R, info.T)).transpose(0, 1).cuda()
        projection = getProjectionMatrix(ZNEAR, ZFAR, info.FovX, info.FovY).transpose(0, 1).cuda()
        self.full_proj_transform = self.world_view_transform.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class _Pipe:
    debug = False
    compute_cov3D_python = False
    convert_SHs_python = False


def _load_model(path: Path):
    sh0 = path.name.endswith("@sh0")
    if sh0:
        path = path.with_name(path.name[: -len("@sh0")])
    if path.suffix == ".npz":
        package = load_digital_twin_package(str(path))
        quats = torch.tensor(package.gaussian_quaternions, dtype=torch.float32, device="cuda")
        return "twin", {
            "means": torch.tensor(package.recompute_gaussian_positions(), dtype=torch.float32, device="cuda"),
            "colors": torch.tensor(np.clip(package.gaussian_colors, 0.0, 1.0), dtype=torch.float32, device="cuda"),
            "opacities": torch.tensor(package.gaussian_opacities[:, None], dtype=torch.float32, device="cuda"),
            "scales": torch.tensor(package.gaussian_scales, dtype=torch.float32, device="cuda"),
            # The CUDA kernel assumes unit quaternions (its own normalization is commented out).
            "rotations": torch.nn.functional.normalize(quats, dim=1),
            "count": len(package),
        }
    model = GaussianModel(3)
    model.load_ply(str(path))
    if sh0:
        model.active_sh_degree = 0
    return "3dgs", model


def _render(kind: str, model, view: _View) -> torch.Tensor:
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    if kind == "3dgs":
        return render_3dgs(view, model, _Pipe(), background)["render"]
    settings = GaussianRasterizationSettings(
        image_height=view.image_height, image_width=view.image_width,
        tanfovx=math.tan(view.FoVx * 0.5), tanfovy=math.tan(view.FoVy * 0.5),
        bg=background, scale_modifier=1.0,
        viewmatrix=view.world_view_transform, projmatrix=view.full_proj_transform,
        sh_degree=0, campos=view.camera_center, prefiltered=False, debug=False,
    )
    image, _ = GaussianRasterizer(raster_settings=settings)(
        means3D=model["means"], means2D=torch.zeros_like(model["means"], requires_grad=False),
        shs=None, colors_precomp=model["colors"], opacities=model["opacities"],
        scales=model["scales"], rotations=model["rotations"], cov3D_precomp=None,
    )
    return image


def _as_png(image: torch.Tensor) -> torch.Tensor:
    return (torch.round(image.clamp(0.0, 1.0) * 255.0) / 255.0).unsqueeze(0)


def _photo(info, images_dir: Path | None, resolution: int) -> torch.Tensor:
    path = Path(info.image_path) if images_dir is None else images_dir / Path(info.image_path).name
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if resolution != 1:
        image = image.resize((round(width / resolution), round(height / resolution)))  # as 3DGS loadCam does
    array = torch.tensor(np.asarray(image), dtype=torch.float32, device="cuda").permute(2, 0, 1) / 255.0
    return array.unsqueeze(0)


def _mask(info, mask_dir: Path, size: tuple[int, int]) -> torch.Tensor | None:
    path = mask_dir / (Path(info.image_path).stem + ".png")
    if not path.is_file():
        return None
    mask = np.asarray(Image.open(path).convert("L").resize(size)) > 127
    return torch.tensor(mask, device="cuda")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-path", required=True, help="COLMAP source (sparse/0 and the photographs)")
    parser.add_argument("--images", default="images", help="photograph folder inside --scene-path, as the 3DGS -i option")
    parser.add_argument("--images-dir", default=None, help="read the photographs from here instead, by file name")
    parser.add_argument("--resolution", type=int, default=1, help="downscale factor, as the 3DGS -r option")
    parser.add_argument("--split", choices=["test", "train", "all"], default="all")
    parser.add_argument("--every", type=int, default=1, help="use every N-th view of the split")
    parser.add_argument("--mask-dir", default=None, help="object masks, for a PSNR inside the object as well")
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    info = readColmapSceneInfo(args.scene_path, args.images, eval=args.split != "all")
    cameras = {"test": info.test_cameras, "train": info.train_cameras, "all": info.train_cameras}[args.split]
    cameras = sorted(cameras, key=lambda c: c.image_name)[:: args.every]
    images_dir = Path(args.images_dir) if args.images_dir else None
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    print(f"{len(cameras)} views ({args.split}, every {args.every}), resolution 1/{args.resolution}", flush=True)

    photos, views, masks = [], [], []
    for cam in cameras:
        photo = _photo(cam, images_dir, args.resolution)
        _, _, height, width = photo.shape
        photos.append(photo)
        views.append(_View(cam, width, height))
        masks.append(_mask(cam, mask_dir, (width, height)) if mask_dir else None)
    print(f"image size {views[0].image_width} x {views[0].image_height}", flush=True)

    record = {
        "scene_path": args.scene_path, "images": args.images, "images_dir": args.images_dir,
        "resolution": args.resolution, "split": args.split, "every": args.every,
        "width": views[0].image_width, "height": views[0].image_height,
        "views": [c.image_name for c in cameras], "models": {},
    }
    for spec in args.model:
        name, _, path = spec.partition("=")
        kind, model = _load_model(Path(path))
        count = model["count"] if kind == "twin" else int(model.get_xyz.shape[0])
        rows = {"psnr": [], "ssim": [], "lpips": [], "psnr_in_mask": [], "render_ms": []}
        with torch.no_grad():
            _render(kind, model, views[0])  # warm-up: the first call compiles and allocates, and is not timed
            for photo, view, mask in zip(photos, views, masks):
                torch.cuda.synchronize()
                started = time.perf_counter()
                image = _render(kind, model, view)
                torch.cuda.synchronize()
                rows["render_ms"].append(1000.0 * (time.perf_counter() - started))
                rendered = _as_png(image)
                rows["psnr"].append(float(psnr(rendered, photo).mean()))
                rows["ssim"].append(float(ssim(rendered, photo)))
                rows["lpips"].append(float(lpips(rendered, photo, net_type="vgg")))
                if mask is not None and mask.any():
                    mse = ((rendered[0] - photo[0]) ** 2).mean(dim=0)[mask].mean()
                    rows["psnr_in_mask"].append(float(20 * torch.log10(1.0 / torch.sqrt(mse))))
        summary = {key: float(np.mean(values)) for key, values in rows.items() if values}
        record["models"][name] = {"path": path, "kind": kind, "gaussians": count, "mean": summary, "per_view": rows}
        print(f"{name:12s} {count:>10,d} Gaussians  PSNR {summary['psnr']:.2f}  SSIM {summary['ssim']:.3f}  "
              f"LPIPS {summary['lpips']:.3f}"
              + (f"  PSNR in mask {summary['psnr_in_mask']:.2f}" if "psnr_in_mask" in summary else "")
              + f"  render {summary['render_ms']:.1f} ms", flush=True)
        del model
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
