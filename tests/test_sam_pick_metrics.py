import numpy as np
import pytest

from src.viewer.sam_pick_metrics import carve_object, confusion_metrics, overlap_fraction, silhouette_agreement
from test_sam_pick import CENTERS, SPHERE_RADIUS, StandInSam, _project_one, ring, two_spheres

CAMERAS = ring()


def sphere_silhouettes(name: str, occluded_by_the_other: bool = False) -> dict[int, np.ndarray]:
    if occluded_by_the_other:
        stand_in = StandInSam(CAMERAS)
        return {view: stand_in._discs(camera) == name for view, camera in enumerate(CAMERAS)}
    silhouettes = {}
    for view, camera in enumerate(CAMERAS):
        yy, xx = np.mgrid[0:camera.height, 0:camera.width]
        pixel, depth, _ = _project_one(camera, CENTERS[name])
        radius = camera.fx * SPHERE_RADIUS / np.sqrt(depth ** 2 - SPHERE_RADIUS ** 2)
        silhouettes[view] = (xx - pixel[0]) ** 2 + (yy - pixel[1]) ** 2 <= radius ** 2
    return silhouettes


class TestConfusionMetrics:
    def test_values(self):
        truth = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
        selected = np.array([1, 1, 0, 0, 1, 0], dtype=bool)

        result = confusion_metrics(selected, truth)

        assert result == {"precision": pytest.approx(2 / 3), "recall": pytest.approx(0.5), "iou": pytest.approx(0.4)}

    def test_empty_selection_scores_zero_not_an_error(self):
        result = confusion_metrics(np.zeros(5, bool), np.ones(5, bool))

        assert result == {"precision": 0.0, "recall": 0.0, "iou": 0.0}

    def test_overlap_is_relative_to_the_smaller_selection(self):
        a = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
        b = np.array([0, 0, 1, 1, 1, 1], dtype=bool)

        assert overlap_fraction(a, b) == pytest.approx(0.5)
        assert overlap_fraction(a, np.zeros(6, bool)) == 0.0


class TestCarveObject:
    def test_the_true_masks_recover_one_sphere_including_its_hidden_core(self):
        shell, is_a = two_spheres()
        core = CENTERS["A"] + 0.3 * (shell[:200] - CENTERS["A"]) / SPHERE_RADIUS  # points inside sphere A, seen by no camera
        positions = np.vstack([shell, core]).astype(np.float32)
        truth_a = np.concatenate([is_a, np.ones(200, bool)])

        carved = carve_object(CAMERAS, positions, sphere_silhouettes("A"))

        assert confusion_metrics(carved, truth_a)["iou"] > 0.95

    def test_a_hidden_core_behind_an_occluder_in_some_views_is_lost_from_visible_only_masks(self):
        """The limit of the visual-hull rule: masks that show only what is visible drop
        an object's hidden interior once a neighbour covers it in more than a tenth of the views."""
        shell, is_a = two_spheres()
        core = CENTERS["A"] + 0.3 * (shell[:200] - CENTERS["A"]) / SPHERE_RADIUS
        positions = np.vstack([shell, core]).astype(np.float32)

        carved = carve_object(CAMERAS, positions, sphere_silhouettes("A", occluded_by_the_other=True))

        assert carved[: len(shell)][is_a].mean() > 0.9  # the visible shell is fine
        assert carved[len(shell):].mean() < 0.5  # the hidden core is not


class TestSilhouetteAgreement:
    def setup_method(self):
        self.positions, self.is_a = two_spheres()
        self.camera = CAMERAS[6]
        self.silhouette = sphere_silhouettes("A")[6]

    def test_selecting_the_object_reaches_the_ceiling(self):
        result = silhouette_agreement(self.camera, self.positions, self.is_a, self.is_a, self.silhouette)

        assert result["ratio"] > 0.97
        assert 0.6 < result["ceiling"] <= 1.0

    def test_selecting_the_other_object_scores_near_zero(self):
        result = silhouette_agreement(self.camera, self.positions, ~self.is_a, self.is_a, self.silhouette)

        assert result["ratio"] < 0.1

    def test_selecting_half_of_the_object_scores_about_half(self):
        pixel, _, _ = _project_one(self.camera, CENTERS["A"])
        pixels = np.array([_project_one(self.camera, p)[0] for p in self.positions])
        left_half = self.is_a & (pixels[:, 0] < pixel[0])

        result = silhouette_agreement(self.camera, self.positions, left_half, self.is_a, self.silhouette)

        assert 0.35 < result["ratio"] < 0.65

    def test_a_silhouette_at_another_resolution_is_resampled(self):
        half_size = self.silhouette[::2, ::2]

        full = silhouette_agreement(self.camera, self.positions, self.is_a, self.is_a, self.silhouette)
        small = silhouette_agreement(self.camera, self.positions, self.is_a, self.is_a, half_size)

        assert small["ratio"] == pytest.approx(full["ratio"], abs=0.05)
