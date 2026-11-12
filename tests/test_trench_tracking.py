"""Fixed-geometry and fail-closed tests for trench-constrained tracking."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from skimage.draw import disk

from napari_raman_widget.trench_tracking import (
    DEFAULT_TRENCH_CONFIG,
    TrenchConstraintTracker,
    TrenchPose,
    TrenchTarget,
    filter_labels_to_trenches,
    reference_lattice_vertices,
)


SHAPE = DEFAULT_TRENCH_CONFIG.reference_image_shape_yx
POINT = np.asarray((468.0, 900.0))


def cell_mask(center_yx, radius=12):
    mask = np.zeros(SHAPE, dtype=bool)
    rows, columns = disk(center_yx, radius, shape=SHAPE)
    mask[rows, columns] = True
    return mask


class LearnedTrenchGeometryTests(unittest.TestCase):
    def test_one_closed_shape_is_repeated_only_by_translation(self):
        vertices = reference_lattice_vertices()

        self.assertEqual(vertices.shape, (10, 7, 2))
        self.assertTrue(np.allclose(vertices[:, 0], vertices[:, -1]))
        translations = np.diff(vertices, axis=0)
        expected = np.broadcast_to(translations[0], translations.shape)
        np.testing.assert_allclose(translations, expected, atol=1e-10)

    def test_global_pose_round_trip_preserves_points(self):
        pose = TrenchPose(angle_deg=2.75, shift_yx=(7.5, -13.0))
        points = np.asarray(((100.0, 700.0), (468.0, 900.0)))

        restored = pose.inverse(pose.apply(points, SHAPE), SHAPE)

        np.testing.assert_allclose(restored, points, atol=1e-10)


class TrenchConstraintTrackerTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros(SHAPE, dtype=np.float32)
        self.labels = np.zeros((256, 336), dtype=np.uint16)
        self.pose = TrenchPose(score=1.0)
        self.target = TrenchTarget("cell-1", POINT, POINT)

    def update(self, tracker, time_index, target, mask, pose=None):
        with patch(
            "napari_raman_widget.trench_tracking._cell_watershed",
            return_value=mask,
        ):
            return tracker.update(
                0,
                time_index,
                self.image,
                [target],
                self.labels,
                4.0,
                pose_override=self.pose if pose is None else pose,
            )

    def initialize(self, tracker=None):
        tracker = tracker or TrenchConstraintTracker()
        update = self.update(tracker, 0, self.target, cell_mask(POINT))
        self.assertTrue(update.accepted[0], update.reasons)
        return tracker, update

    def test_more_than_fifteen_pixels_is_held_without_poisoning_next_frame(self):
        tracker, _ = self.initialize()
        far = POINT + (0, 16)
        with patch(
            "napari_raman_widget.trench_tracking._template_candidate",
            return_value=(far, 1.0, True),
        ):
            rejected = self.update(
                tracker,
                1,
                TrenchTarget("cell-1", POINT, far),
                cell_mask(far),
            )

        self.assertFalse(rejected.accepted[0])
        self.assertIn("maximum_step", rejected.reasons[0])
        np.testing.assert_allclose(rejected.points_yx[0], POINT)

        near = POINT + (0, 5)
        with patch(
            "napari_raman_widget.trench_tracking._template_candidate",
            return_value=(near, 1.0, True),
        ):
            accepted = self.update(
                tracker,
                2,
                TrenchTarget("cell-1", POINT, near),
                cell_mask(near),
            )
        self.assertTrue(accepted.accepted[0], accepted.reasons)
        np.testing.assert_allclose(accepted.points_yx[0], near)

    def test_motion_gate_is_corrected_for_global_device_translation(self):
        tracker, _ = self.initialize()
        translated_pose = TrenchPose(shift_yx=(0.0, 20.0), score=1.0)
        predicted = POINT + (0, 20)
        center = predicted + (0, 5)
        with patch(
            "napari_raman_widget.trench_tracking._template_candidate",
            return_value=(center, 1.0, True),
        ):
            update = self.update(
                tracker,
                1,
                TrenchTarget("cell-1", POINT, center),
                cell_mask(center),
                pose=translated_pose,
            )

        self.assertTrue(update.accepted[0], update.reasons)
        self.assertAlmostEqual(update.results[0].step_px, 5.0)
        self.assertGreater(np.linalg.norm(center - POINT), 15.0)

    def test_motion_gate_is_corrected_for_global_device_rotation(self):
        tracker, _ = self.initialize()
        rotated_pose = TrenchPose(angle_deg=5.0, score=1.0)
        predicted = rotated_pose.apply(POINT, SHAPE)
        center = predicted + (0, 3)
        with patch(
            "napari_raman_widget.trench_tracking._template_candidate",
            return_value=(center, 1.0, True),
        ):
            update = self.update(
                tracker,
                1,
                TrenchTarget("cell-1", POINT, center),
                cell_mask(center),
                pose=rotated_pose,
            )

        self.assertTrue(update.accepted[0], update.reasons)
        self.assertAlmostEqual(update.results[0].step_px, 3.0, delta=0.1)
        self.assertGreater(np.linalg.norm(center - POINT), 15.0)

    def test_center_near_wall_is_rejected_even_when_inside_raw_polygon(self):
        near_wall = np.asarray((437.0, 900.0))
        tracker = TrenchConstraintTracker()
        target = TrenchTarget("wall-cell", near_wall, near_wall)

        update = self.update(tracker, 0, target, cell_mask(near_wall))

        self.assertFalse(update.accepted[0])
        self.assertIn("same_trench_interior", update.reasons[0])
        np.testing.assert_allclose(update.points_yx[0], near_wall)

    def test_track_cannot_jump_to_an_adjacent_trench(self):
        config = replace(DEFAULT_TRENCH_CONFIG, maximum_step_px=200.0)
        tracker, _ = self.initialize(TrenchConstraintTracker(config))
        repeat = (
            reference_lattice_vertices(config)[5, 0]
            - reference_lattice_vertices(config)[4, 0]
        )
        adjacent = POINT + repeat
        with patch(
            "napari_raman_widget.trench_tracking._template_candidate",
            return_value=(adjacent, 1.0, True),
        ):
            update = self.update(
                tracker,
                1,
                TrenchTarget("cell-1", POINT, adjacent),
                cell_mask(adjacent),
            )

        self.assertFalse(update.accepted[0])
        self.assertIn("same_trench_interior", update.reasons[0])
        self.assertEqual(update.results[0].trench_index, 4)
        np.testing.assert_allclose(update.points_yx[0], POINT)

    def test_downstream_mask_keeps_only_cell_sized_labels_in_safe_trenches(self):
        labels = np.zeros((512, 672), dtype=np.uint16)
        inside = np.rint(POINT / 2).astype(int)
        rows, columns = disk(inside, 6, shape=labels.shape)
        labels[rows, columns] = 3
        rows, columns = disk((50, 50), 6, shape=labels.shape)
        labels[rows, columns] = 7

        update = TrenchConstraintTracker().update(
            0,
            0,
            self.image,
            [],
            labels,
            2.0,
            pose_override=self.pose,
        )

        self.assertNotEqual(update.labels[tuple(inside)], 0)
        self.assertEqual(update.labels[50, 50], 0)

    def test_existing_base_label_is_not_reintroduced_as_a_duplicate(self):
        labels = np.zeros((256, 336), dtype=np.uint16)
        center = np.rint(POINT / 4).astype(int)
        rows, columns = disk(center, 7, shape=labels.shape)
        labels[rows, columns] = 9
        tracker = TrenchConstraintTracker()

        with patch(
            "napari_raman_widget.trench_tracking._cell_watershed",
            return_value=cell_mask(POINT),
        ):
            update = tracker.update(
                0,
                0,
                self.image,
                [self.target],
                labels,
                4.0,
                pose_override=self.pose,
            )

        self.assertTrue(update.accepted[0], update.reasons)
        np.testing.assert_array_equal(
            np.unique(update.labels),
            np.asarray((0, 1), dtype=np.uint32),
        )

    def test_incompatible_image_shape_fails_closed(self):
        update = TrenchConstraintTracker().update(
            0,
            0,
            np.zeros((512, 672), dtype=np.float32),
            [TrenchTarget("cell", (200, 400))],
            np.ones((128, 168), dtype=np.uint16),
            4.0,
            pose_override=self.pose,
        )

        self.assertFalse(update.geometry_valid)
        self.assertEqual(update.reasons, ("geometry_shape",))
        self.assertFalse(np.any(update.labels))


class AutomatedSelectionTrenchFilterTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros(SHAPE, dtype=np.float32)
        self.pose = TrenchPose(score=1.0)

    def test_only_cell_sized_components_inside_safe_trenches_remain(self):
        labels = np.zeros((256, 336), dtype=np.uint16)
        inside = np.rint(POINT / 4).astype(int)
        rows, columns = disk(inside, 5, shape=labels.shape)
        labels[rows, columns] = 3
        rows, columns = disk((50, 50), 5, shape=labels.shape)
        labels[rows, columns] = 7

        result = filter_labels_to_trenches(
            self.image,
            labels,
            pose_override=self.pose,
        )

        self.assertTrue(result.geometry_valid)
        self.assertNotEqual(result.filtered_labels[tuple(inside)], 0)
        self.assertEqual(result.filtered_labels[50, 50], 0)
        self.assertTrue(result.safe_interior_mask[tuple(inside)])
        self.assertTrue(result.exclusion_mask[50, 50])

    def test_one_raw_label_crossing_trenches_is_split_before_centering(self):
        labels = np.zeros((256, 336), dtype=np.uint16)
        vertices = reference_lattice_vertices()
        centers = []
        for trench_index in (4, 5):
            center = np.mean(vertices[trench_index, 1:5], axis=0) / 4
            centers.append(np.rint(center).astype(int))
        for center in centers:
            rows, columns = disk(center, 5, shape=labels.shape)
            labels[rows, columns] = 9

        result = filter_labels_to_trenches(
            self.image,
            labels,
            pose_override=self.pose,
        )

        kept = np.unique(result.filtered_labels)
        kept = kept[kept != 0]
        self.assertEqual(len(kept), 2)
        for label_value in kept:
            center = np.rint(
                np.argwhere(result.filtered_labels == label_value).mean(axis=0)
            ).astype(int)
            self.assertTrue(result.safe_interior_mask[tuple(center)])

    def test_low_confidence_or_incompatible_geometry_fails_closed(self):
        labels = np.ones((256, 336), dtype=np.uint16)
        cases = (
            (self.image, TrenchPose(score=0.1)),
            (np.zeros((512, 672), dtype=np.float32), self.pose),
        )
        for image, pose in cases:
            with self.subTest(shape=image.shape, score=pose.score):
                result = filter_labels_to_trenches(
                    image,
                    labels,
                    pose_override=pose,
                )
                self.assertFalse(result.geometry_valid)
                self.assertFalse(np.any(result.filtered_labels))
                self.assertFalse(np.any(result.safe_interior_mask))


if __name__ == "__main__":
    unittest.main()
