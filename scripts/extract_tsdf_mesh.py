import argparse
import math
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import open3d as o3d

OPACITY_CUT = 0.5
MESH_RESOLUTION = 1024
TRUNCATION_VOXELS = 5.0
CLUSTERS_KEPT = 50
MIN_CLUSTER_TRIANGLES = 50


def bounding_radius(camera_to_world: np.ndarray) -> tuple[np.ndarray, float]:
    directions, origins = camera_to_world[:, :3, 2:3], camera_to_world[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    center = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return center, float(np.linalg.norm(camera_to_world[:, :3, 3] - center, axis=1).min())


def fuse_depth_maps(frames: Iterable[tuple[np.ndarray, tuple, np.ndarray]], voxel: float, depth_cut: float,
                    truncation_voxels: float = TRUNCATION_VOXELS) -> o3d.geometry.TriangleMesh:
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=truncation_voxels * voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )
    for depth, (width, height, fx, fy, cx, cy), extrinsic in frames:
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros((height, width, 3), dtype=np.uint8)),
            o3d.geometry.Image(np.ascontiguousarray(depth, dtype=np.float32)),
            depth_scale=1.0, depth_trunc=depth_cut, convert_rgb_to_intensity=False,
        )
        camera = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        volume.integrate(rgbd, camera, np.asarray(extrinsic, dtype=np.float64))
    return volume.extract_triangle_mesh()


def keep_largest_clusters(mesh: o3d.geometry.TriangleMesh, keep: int = CLUSTERS_KEPT,
                          minimum: int = MIN_CLUSTER_TRIANGLES) -> o3d.geometry.TriangleMesh:
    clusters, sizes, _ = mesh.cluster_connected_triangles()
    clusters, sizes = np.asarray(clusters), np.asarray(sizes)
    if sizes.size == 0:
        return mesh
    cut = max(int(np.sort(sizes)[-min(keep, sizes.size)]), minimum)
    kept = o3d.geometry.TriangleMesh(mesh)
    kept.remove_triangles_by_mask(sizes[clusters] < cut)
    kept.remove_unreferenced_vertices()
    kept.remove_degenerate_triangles()
    return kept


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-path", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--model", required=True, help="a 3DGS point_cloud.ply")
    parser.add_argument("--resolution", type=int, default=2, help="downscale factor of the depth maps")
    parser.add_argument("--triangles-of", default=None, help="a twin .npz whose triangle count the mesh is decimated to")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate_fidelity import _View
    import torch
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from scene.dataset_readers import readColmapSceneInfo
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import getWorld2View2

    from src.viewer.digital_twin import load_digital_twin_package

    gaussians = GaussianModel(3)
    gaussians.load_ply(args.model)
    means = gaussians.get_xyz.detach().contiguous()
    opacities = gaussians.get_opacity.detach().contiguous()
    scales = gaussians.get_scaling.detach().contiguous()
    rotations = torch.nn.functional.normalize(gaussians.get_rotation.detach(), dim=1).contiguous()
    homogeneous = torch.cat([means, torch.ones_like(means[:, :1])], dim=1)

    def render_depth(view) -> tuple[np.ndarray, np.ndarray]:
        depth = (homogeneous @ view.world_view_transform)[:, 2:3]
        colors = torch.cat([depth, torch.ones_like(depth), torch.zeros_like(depth)], dim=1)
        settings = GaussianRasterizationSettings(
            image_height=view.image_height, image_width=view.image_width,
            tanfovx=math.tan(view.FoVx * 0.5), tanfovy=math.tan(view.FoVy * 0.5),
            bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
            viewmatrix=view.world_view_transform, projmatrix=view.full_proj_transform,
            sh_degree=0, campos=view.camera_center, prefiltered=False, debug=False,
        )
        with torch.no_grad():
            image, _ = GaussianRasterizer(raster_settings=settings)(
                means3D=means, means2D=torch.zeros_like(means), shs=None, colors_precomp=colors,
                opacities=opacities, scales=scales, rotations=rotations, cov3D_precomp=None,
            )
        blended, opacity = image[0].cpu().numpy(), image[1].cpu().numpy()
        return np.where(opacity > OPACITY_CUT, blended / np.maximum(opacity, 1e-6), 0.0).astype(np.float32), opacity

    cameras = readColmapSceneInfo(args.scene_path, args.images, eval=False).train_cameras
    world_to_camera = [getWorld2View2(info.R, info.T).astype(np.float64) for info in cameras]
    center, radius = bounding_radius(np.linalg.inv(np.stack(world_to_camera)))
    depth_cut = 2.0 * radius
    voxel = depth_cut / MESH_RESOLUTION

    def frames():
        for info, extrinsic in zip(cameras, world_to_camera):
            width, height = round(info.width / args.resolution), round(info.height / args.resolution)
            depth, _ = render_depth(_View(info, width, height))
            fx, fy = width / (2 * math.tan(info.FovX / 2)), height / (2 * math.tan(info.FovY / 2))
            yield depth, (width, height, fx, fy, width / 2, height / 2), extrinsic

    raw = fuse_depth_maps(frames(), voxel, depth_cut)
    mesh = keep_largest_clusters(raw)
    print(f"{len(cameras)} depth maps; radius {radius:.3f} around {np.round(center, 3).tolist()}, voxel {voxel:.5f}: "
          f"{len(raw.triangles):,} triangles, {len(mesh.triangles):,} in the {CLUSTERS_KEPT} largest clusters", flush=True)
    if args.triangles_of:
        target = int(load_digital_twin_package(args.triangles_of).mesh_faces.shape[0])
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
        mesh.remove_unreferenced_vertices()
        print(f"decimated to {len(mesh.triangles):,} triangles (the twin has {target:,})", flush=True)
    o3d.io.write_triangle_mesh(args.out, mesh)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
