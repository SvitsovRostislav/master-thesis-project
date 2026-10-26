import numpy as np

from .sam_pick import _DepthMap, project, vote_selection
from .semantic_segmentation import Camera


def carve_object(cameras: list[Camera], positions: np.ndarray, silhouettes: dict[int, np.ndarray]) -> np.ndarray:
    return vote_selection(cameras, positions, silhouettes)


def confusion_metrics(selected: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    hit = int((selected & truth).sum())
    return {
        "precision": hit / max(int(selected.sum()), 1),
        "recall": hit / max(int(truth.sum()), 1),
        "iou": hit / max(int((selected | truth).sum()), 1),
    }


def overlap_fraction(a: np.ndarray, b: np.ndarray) -> float:
    return int((a & b).sum()) / max(min(int(a.sum()), int(b.sum())), 1)


def _resampled(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask
    rows = np.minimum((np.arange(height) * mask.shape[0] / height).astype(int), mask.shape[0] - 1)
    cols = np.minimum((np.arange(width) * mask.shape[1] / width).astype(int), mask.shape[1] - 1)
    return mask[np.ix_(rows, cols)]


def silhouette_agreement(
    camera: Camera, positions: np.ndarray, selected: np.ndarray, truth: np.ndarray, silhouette: np.ndarray
) -> dict[str, float]:
    pixels, _, in_frame = project(camera, positions)
    framed = np.nonzero(in_frame)[0]
    cell = _DepthMap._cell_size(camera, pixels[framed])
    nx, ny = camera.width // cell + 1, camera.height // cell + 1

    def occupied(indices: np.ndarray) -> np.ndarray:
        grid = np.zeros(nx * ny, dtype=bool)
        points = pixels[indices]
        grid[(points[:, 1] // cell).astype(np.int64) * nx + (points[:, 0] // cell).astype(np.int64)] = True
        return grid.reshape(ny, nx)

    resampled = _resampled(silhouette, camera.height, camera.width).astype(np.float32)
    padded = np.zeros((ny * cell, nx * cell), dtype=np.float32)
    padded[: camera.height, : camera.width] = resampled
    true_cells = padded.reshape(ny, cell, nx, cell).mean(axis=(1, 3)) >= 0.5

    def iou(cells: np.ndarray) -> float:
        return int((cells & true_cells).sum()) / max(int((cells | true_cells).sum()), 1)

    selected_iou, ceiling = iou(occupied(framed[selected[framed]])), iou(occupied(framed[truth[framed]]))
    return {"iou": selected_iou, "ceiling": ceiling, "ratio": selected_iou / ceiling if ceiling > 0 else 0.0}
