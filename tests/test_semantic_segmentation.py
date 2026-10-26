import numpy as np

from src.viewer.semantic_segmentation import (
    Camera,
    label_at_vertex,
    labels_to_vertex_selection,
    load_semantic_labels,
    semantic_labels_path,
    vote_gaussian_labels,
)


def _centered_camera(width=200, height=200, focal=50.0) -> Camera:
    return Camera(
        rotation=np.eye(3),
        translation=np.zeros(3),
        fx=focal,
        fy=focal,
        cx=width / 2,
        cy=height / 2,
        width=width,
        height=height,
    )


class TestVoteGaussianLabels:
    def test_recovers_the_correct_label_per_gaussian_from_a_split_mask(self):
        positions = np.array(
            [[-1.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 0.0, -5.0]], dtype=np.float64
        )
        camera = _centered_camera()
        mask = np.zeros((200, 200), dtype=np.int32)
        mask[:, :100] = 1
        mask[:, 100:] = 2

        labels = vote_gaussian_labels(positions, [camera, camera], [mask, mask])

        np.testing.assert_array_equal(labels, [1, 2, 0])

    def test_a_point_never_seen_under_a_labeled_region_stays_unlabeled(self):
        positions = np.array([[0.0, 0.0, 5.0]], dtype=np.float64)
        camera = _centered_camera()
        blank_mask = np.zeros((200, 200), dtype=np.int32)  # all background

        labels = vote_gaussian_labels(positions, [camera], [blank_mask])

        assert labels.tolist() == [0]

    def test_majority_vote_wins_across_disagreeing_views(self):
        positions = np.array([[0.0, 0.0, 5.0]], dtype=np.float64)
        camera = _centered_camera()
        mask_a = np.full((200, 200), 1, dtype=np.int32)
        mask_b = np.full((200, 200), 1, dtype=np.int32)
        mask_c = np.full((200, 200), 2, dtype=np.int32)

        labels = vote_gaussian_labels(positions, [camera, camera, camera], [mask_a, mask_b, mask_c])

        assert labels.tolist() == [1]


def _two_disconnected_triangles_with_labels():
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    gaussian_face_idx = np.array([0, 0, 1, 1], dtype=np.int32)
    semantic_id = np.array([1, 1, 2, 2], dtype=np.int32)
    return faces, gaussian_face_idx, semantic_id


class TestLabelAtVertex:
    def test_returns_the_label_of_the_triangle_a_vertex_belongs_to(self):
        faces, gaussian_face_idx, semantic_id = _two_disconnected_triangles_with_labels()

        assert label_at_vertex(faces, gaussian_face_idx, semantic_id, vertex_index=0) == 1
        assert label_at_vertex(faces, gaussian_face_idx, semantic_id, vertex_index=4) == 2

    def test_vertex_with_no_labeled_gaussians_returns_background(self):
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        gaussian_face_idx = np.array([0], dtype=np.int32)
        semantic_id = np.array([0], dtype=np.int32)

        assert label_at_vertex(faces, gaussian_face_idx, semantic_id, vertex_index=0) == 0


class TestLabelsToVertexSelection:
    def test_selects_only_the_vertices_of_the_matching_triangle(self):
        faces, gaussian_face_idx, semantic_id = _two_disconnected_triangles_with_labels()

        selection_1 = labels_to_vertex_selection(faces, gaussian_face_idx, semantic_id, target_label=1)
        selection_2 = labels_to_vertex_selection(faces, gaussian_face_idx, semantic_id, target_label=2)

        np.testing.assert_array_equal(sorted(selection_1.tolist()), [0, 1, 2])
        np.testing.assert_array_equal(sorted(selection_2.tolist()), [3, 4, 5])

    def test_background_label_returns_none(self):
        faces, gaussian_face_idx, semantic_id = _two_disconnected_triangles_with_labels()

        assert labels_to_vertex_selection(faces, gaussian_face_idx, semantic_id, target_label=0) is None

    def test_unmatched_label_returns_none(self):
        faces, gaussian_face_idx, semantic_id = _two_disconnected_triangles_with_labels()

        assert labels_to_vertex_selection(faces, gaussian_face_idx, semantic_id, target_label=99) is None


class TestSemanticLabelsSidecar:
    def test_path_convention(self):
        assert semantic_labels_path("scene_digital_twin.npz").name == "scene_digital_twin_semantic_labels.npy"

    def test_missing_sidecar_returns_none(self, tmp_path):
        assert load_semantic_labels(tmp_path / "nope.npz", expected_count=5) is None

    def test_stale_sidecar_with_wrong_length_returns_none(self, tmp_path):
        npz_path = tmp_path / "twin.npz"
        np.save(semantic_labels_path(npz_path), np.zeros(5, dtype=np.int32))

        assert load_semantic_labels(npz_path, expected_count=6) is None

    def test_matching_sidecar_loads(self, tmp_path):
        npz_path = tmp_path / "twin.npz"
        np.save(semantic_labels_path(npz_path), np.array([1, 1, 2], dtype=np.int32))

        loaded = load_semantic_labels(npz_path, expected_count=3)

        np.testing.assert_array_equal(loaded, [1, 1, 2])
