import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_sam_pick import _load_silhouettes
from sam_point_segment import REPO_ROOT, _build_cameras

from scene.dataset_readers import readColmapSceneInfo

from src.viewer.digital_twin import load_digital_twin_package
from src.viewer.sam_pick_metrics import carve_object

PAIR_SCENES = {"plant": ("pot", "plant")}
TRANSFER_CAP = 4.0


def component_labels(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    ea = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    eb = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    labels = np.arange(n_vertices)
    while True:
        la, lb = labels[ea], labels[eb]
        low = np.minimum(la, lb)
        new = labels.copy()
        np.minimum.at(new, la, low)
        np.minimum.at(new, lb, low)
        while True:
            jumped = new[new]
            if np.array_equal(jumped, new):
                break
            new = jumped
        if np.array_equal(new, labels):
            return labels
        labels = new


def face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def object_scores(face_component: np.ndarray, face_label: np.ndarray, target: int, names: dict,
                  weights: np.ndarray | None = None) -> dict:
    mine = face_label == target
    if weights is None:
        comps, totals = np.unique(face_component[mine], return_counts=True)
    else:
        comps, inverse = np.unique(face_component[mine], return_inverse=True)
        totals = np.bincount(inverse, weights=weights[mine])
    order = np.argsort(totals)[::-1]
    comps, totals = comps[order], totals[order]
    cumulative = np.cumsum(totals) / totals.sum()
    main = comps[0]
    in_main = face_component == main
    if weights is None:
        others = {names[label]: int(((face_label == label) & in_main).sum()) for label in names if label != target}
        main_total = in_main.sum()
    else:
        others = {names[label]: float(weights[(face_label == label) & in_main].sum()) for label in names if label != target}
        main_total = weights[in_main].sum()
    return {
        "triangles": int(mine.sum()),
        "components_touched": int(comps.size),
        "largest_share": float(totals[0] / totals.sum()),
        "pieces_for_95_percent": int(np.searchsorted(cumulative, 0.95) + 1),
        "main_component_triangles": int(in_main.sum()),
        "impurity": float(1.0 - totals[0] / main_total),
        "impurity_by_label": others,
    }


def nearest_twin(twin_points: np.ndarray, targets: np.ndarray, cap: float = TRANSFER_CAP) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    tree = cKDTree(twin_points)
    spacing = float(np.median(tree.query(twin_points, k=2)[0][:, 1]))
    distance, nearest = tree.query(targets, k=1)
    return nearest, distance <= cap * spacing


def _carver(scene: str):
    source = REPO_ROOT / "data" / "gaussian_scenes" / "_colmap_src" / scene
    infos = readColmapSceneInfo(str(source), "images", eval=False).train_cameras
    session = SimpleNamespace(cam_infos=infos, cameras=_build_cameras(infos, 1024))
    silhouettes = _load_silhouettes(source, session)
    return lambda positions: carve_object(session.cameras, positions, silhouettes)


def _twin(scene: str, figures: Path, carve) -> SimpleNamespace:
    path = REPO_ROOT / "data" / "gaussian_scenes" / f"{scene}_digital_twin.npz"
    package = load_digital_twin_package(str(path))
    faces = package.mesh_faces
    used = np.zeros(faces.shape[0], dtype=bool)
    used[package.gaussian_face_idx] = True
    vertex_component = component_labels(faces[used], package.mesh_vertices.shape[0])
    face_component = np.full(faces.shape[0], -1, dtype=np.int64)
    face_component[used] = vertex_component[faces[used][:, 0]]

    face_label = np.zeros(faces.shape[0], dtype=np.int64)
    if scene in PAIR_SCENES:
        names = {0: "rest"}
        for label, name in enumerate(PAIR_SCENES[scene], start=1):
            selected = np.load(figures / f"sam_pick_eval_{scene}_{name}.npy")
            face_label[package.gaussian_face_idx[selected & (face_label[package.gaussian_face_idx] == 0)]] = label
            names[label] = name
        targets = list(range(1, len(PAIR_SCENES[scene]) + 1))
    else:
        truth = carve(package.recompute_gaussian_positions())
        face_label[package.gaussian_face_idx[truth]] = 1
        names, targets = {0: "background", 1: "object"}, [1]
    return SimpleNamespace(path=path, package=package, used=used, face_component=face_component,
                           face_label=face_label, names=names, targets=targets)


def _read_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.triangles) == 0:
        raise SystemExit(f"No triangles in {path}")
    before = len(mesh.vertices)
    mesh.remove_duplicated_vertices()
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles), before - len(mesh.vertices)


def _both_scores(face_component, face_label, names, targets, areas) -> dict:
    return {names[t]: {"by_count": object_scores(face_component, face_label, t, names),
                       "by_area": object_scores(face_component, face_label, t, names, weights=areas)}
            for t in targets}


def _twins_table(scenes: list[str], figures: Path, out: Path) -> None:
    record = {}
    print("| Twin | Object | Triangles | Components touched | Largest share | Pieces for 95 % | Impurity of its main component |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for scene in scenes:
        twin = _twin(scene, figures, None if scene in PAIR_SCENES else _carver(scene))
        used, face_component, face_label, names = twin.used, twin.face_component, twin.face_label, twin.names
        components = int(np.unique(face_component[used]).size)
        record[scene] = {"components": components, "triangles_with_gaussians": int(used.sum()), "objects": {}}
        for target in twin.targets:
            scores = object_scores(face_component[used], face_label[used], target, names)
            record[scene]["objects"][names[target]] = scores
            print(f"| `{scene}` | {names[target]} | {scores['triangles']:,} | {scores['components_touched']:,} | "
                  f"{scores['largest_share']:.3f} | {scores['pieces_for_95_percent']:,} | {scores['impurity']:.3f} "
                  f"({', '.join(f'{k} {v:,}' for k, v in scores['impurity_by_label'].items())}) |", flush=True)
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"wrote {out}")


def _benchmark(meshes: dict[str, list[tuple[str, Path]]], figures: Path, out: Path) -> None:
    record = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    rows = []
    for scene, entries in meshes.items():
        carve = None if scene in PAIR_SCENES else _carver(scene)
        twin = _twin(scene, figures, carve)
        used, names, targets = twin.used, twin.names, twin.targets
        faces = twin.package.mesh_faces[used]
        areas = face_areas(twin.package.mesh_vertices, faces)
        scene_record = record.setdefault(scene, {})
        scene_record["sugar_twin"] = {
            "path": str(twin.path.relative_to(REPO_ROOT)), "triangles": int(used.sum()),
            "components": int(np.unique(twin.face_component[used]).size),
            "objects": _both_scores(twin.face_component[used], twin.face_label[used], names, targets, areas),
        }
        rows.append((scene, "sugar_twin", scene_record["sugar_twin"]))

        twin_points = twin.package.recompute_gaussian_positions()
        twin_labels = twin.face_label[twin.package.gaussian_face_idx]
        for name, path in entries:
            vertices, faces, merged = _read_mesh(path)
            centroids = vertices[faces].mean(axis=1)
            nearest, near = nearest_twin(twin_points, centroids)
            face_label = np.where(near, twin_labels[nearest], 0) if carve is None else carve(centroids).astype(np.int64)
            face_component = component_labels(faces, vertices.shape[0])[faces[:, 0]]
            scene_record[name] = {
                "path": str(path), "triangles": int(faces.shape[0]), "merged_vertices": int(merged),
                "components": int(np.unique(face_component).size), "near_twin_share": float(near.mean()),
                "objects": _both_scores(face_component, face_label, names, targets, face_areas(vertices, faces)),
            }
            if near.mean() < 0.5:
                print(f"warning: only {near.mean():.0%} of the triangles of {path} lie near the twin's surface; "
                      "is the mesh in the scene's COLMAP frame?", flush=True)
            rows.append((scene, name, scene_record[name]))

    print("| Scene | Mesh | Object | Triangles | Components | Largest share (area) | Pieces for 95 % (area) | "
          "Impurity (area) | Largest share (count) | Impurity (count) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for scene, name, entry in rows:
        for obj, scores in entry["objects"].items():
            area, count = scores["by_area"], scores["by_count"]
            print(f"| `{scene}` | {name} | {obj} | {entry['triangles']:,} | {entry['components']:,} | "
                  f"{area['largest_share']:.3f} | {area['pieces_for_95_percent']:,} | {area['impurity']:.3f} | "
                  f"{count['largest_share']:.3f} | {count['impurity']:.3f} |", flush=True)
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenes", nargs="*", help="twins to score (thesis/figures/separability_metrics.json)")
    parser.add_argument("--mesh", action="append", default=[], metavar="SCENE:NAME=PATH",
                        help="another mesh of a scene, scored beside its twin (repeatable)")
    parser.add_argument("--out", default=None,
                        help="JSON file (default thesis/figures/separability_metrics.json, or with --mesh "
                             "thesis/figures/separability_benchmark.json, whose other entries are kept)")
    args = parser.parse_args()
    if bool(args.scenes) == bool(args.mesh):
        parser.error("give either twins to score or --mesh entries, not both")
    figures = REPO_ROOT / "thesis" / "figures"

    if args.scenes:
        _twins_table(args.scenes, figures, Path(args.out) if args.out else figures / "separability_metrics.json")
        return
    meshes: dict[str, list[tuple[str, Path]]] = {}
    for entry in args.mesh:
        scene, sep, rest = entry.partition(":")
        name, sep2, path = rest.partition("=")
        if not (sep and sep2 and scene and name and path):
            parser.error(f"--mesh {entry}: expected SCENE:NAME=PATH")
        if name == "sugar_twin":
            parser.error("the name sugar_twin is the twin's own")
        if not Path(path).is_file():
            parser.error(f"--mesh {entry}: no such file")
        meshes.setdefault(scene, []).append((name, Path(path)))
    _benchmark(meshes, figures, Path(args.out) if args.out else figures / "separability_benchmark.json")


if __name__ == "__main__":
    main()
