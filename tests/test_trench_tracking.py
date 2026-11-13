"""Fixed-geometry and fail-closed tests for trench-constrained tracking."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from skimage.draw import disk, line

from napari_raman_widget.trench_tracking import (
    DEFAULT_TRENCH_CONFIG,
    TrenchConstraintTracker,
    TrenchPose,
    TrenchTarget,
    filter_labels_to_trenches,
    registered_trench_geometry,
    reference_lattice_vertices,
    _feature,
)


SHAPE = DEFAULT_TRENCH_CONFIG.reference_image_shape_yx
POINT = np.asarray((468.0, 900.0))


def cell_mask(center_yx, radius=12):
    mask = np.zeros(SHAPE, dtype=bool)
    rows, columns = disk(center_yx, radius, shape=SHAPE)
    mask[rows, columns] = True
    return mask


def trench_axis(pose, trench_index=4, config=DEFAULT_TRENCH_CONFIG):
    vertices = pose.apply(
        reference_lattice_vertices(config),
        config.reference_image_shape_yx,
    )[trench_index]
    tip_midpoint = (vertices[0] + vertices[5]) / 2
    shoulder_midpoint = (vertices[1] + vertices[4]) / 2
    inward_axis = shoulder_midpoint - tip_midpoint
    inward_axis /= np.linalg.norm(inward_axis)
    tip_length = np.linalg.norm(vertices[5] - vertices[0])
    return vertices, tip_midpoint, inward_axis, tip_length


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

    def test_safe_mask_excludes_two_tip_lengths_inward(self):
        pose = TrenchPose(angle_deg=2.75, shift_yx=(7.5, -13.0), score=1.0)
        geometry = registered_trench_geometry(
            np.zeros(SHAPE, dtype=np.float32),
            pose_override=pose,
        )
        _, tip, inward, tip_length = trench_axis(pose)
        exclusion_depth = 2.0 * tip_length
        inside_cap = np.rint(tip + inward * (exclusion_depth - 5)).astype(int)
        # This location was excluded by 2.5d but must be available at 2d.
        past_cap = np.rint(tip + inward * (exclusion_depth + 5)).astype(int)

        self.assertAlmostEqual(tip_length, 17.0, places=10)
        self.assertTrue(geometry.geometry_valid)
        self.assertTrue(geometry.trench_masks[4][tuple(inside_cap)])
        self.assertFalse(geometry.safe_masks[4][tuple(inside_cap)])
        self.assertTrue(geometry.safe_masks[4][tuple(past_cap)])

        safe_pixels = np.argwhere(geometry.safe_masks[4])
        safe_depths = (safe_pixels - tip) @ inward
        self.assertGreaterEqual(safe_depths.min(), exclusion_depth - 1e-9)

    def test_negative_tip_exclusion_multiplier_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "tip_exclusion"):
            replace(DEFAULT_TRENCH_CONFIG, tip_exclusion_depth_multiplier=-0.1)


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
                # Isolate geometry/motion gates from Cellpose association.
                # The identity tests below supply explicit current labels.
                None,
                1.0,
                pose_override=self.pose if pose is None else pose,
            )

    def initialize(self, tracker=None):
        # These tests isolate the legacy geometry/overlap gates with a blank
        # image. Real appearance matching is covered separately with texture.
        tracker = tracker or TrenchConstraintTracker(
            replace(DEFAULT_TRENCH_CONFIG, track_cell_appearance=False)
        )
        update = self.update(tracker, 0, self.target, cell_mask(POINT))
        self.assertTrue(update.accepted[0], update.reasons)
        return tracker, update

    def test_more_than_forty_five_pixels_is_held_without_poisoning_next_frame(self):
        tracker, _ = self.initialize()
        far = POINT + (0, 46)
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

        boundary = POINT + (0, 45)
        with patch(
            "napari_raman_widget.trench_tracking._template_candidate",
            return_value=(boundary, 1.0, True),
        ):
            accepted = self.update(
                tracker,
                2,
                TrenchTarget("cell-1", POINT, boundary),
                cell_mask(boundary),
            )
        self.assertTrue(accepted.accepted[0], accepted.reasons)
        self.assertAlmostEqual(accepted.results[0].step_px, 45.0)
        np.testing.assert_allclose(accepted.points_yx[0], boundary)

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

    def test_custom_movement_limit_rejects_and_accepts_relative_steps(self):
        config = replace(DEFAULT_TRENCH_CONFIG, maximum_step_px=12.0)
        tracker, _ = self.initialize(TrenchConstraintTracker(config))
        for time_index, distance, expected in ((1, 13, False), (2, 12, True)):
            center = POINT + (0, distance)
            with patch(
                "napari_raman_widget.trench_tracking._template_candidate",
                return_value=(center, 1.0, True),
            ):
                update = self.update(
                    tracker, time_index, TrenchTarget("cell-1", POINT, center),
                    cell_mask(center),
                )
            self.assertEqual(bool(update.accepted[0]), expected, update.reasons)
            np.testing.assert_allclose(
                update.points_yx[0], center if expected else POINT
            )

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

    def test_target_inside_closed_tip_cap_is_held(self):
        _, tip, inward, tip_length = trench_axis(self.pose)
        in_cap = tip + inward * (2.0 * tip_length - 10)
        tracker = TrenchConstraintTracker()

        update = self.update(
            tracker,
            0,
            TrenchTarget("tip-wall", in_cap, in_cap),
            cell_mask(in_cap),
        )

        self.assertFalse(update.accepted[0])
        self.assertIn("template_confidence", update.reasons[0])
        np.testing.assert_allclose(update.points_yx[0], in_cap)

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

    def test_refinement_ignores_better_template_match_on_neighbor(self):
        tracker, _ = self.initialize()
        neighbor = POINT + (0, 40)
        original = cell_mask(POINT)
        labels = original.astype(np.uint16)
        labels[cell_mask(neighbor)] = 2
        with patch(
            "napari_raman_widget.trench_tracking._template_candidate",
            return_value=(neighbor, 1.0, True),
        ) as match, patch(
            "napari_raman_widget.trench_tracking._cell_watershed",
            return_value=original,
        ) as segment:
            update = tracker.update(
                0, 1, self.image, [self.target], labels,
                pose_override=self.pose,
            )
        match.assert_not_called()
        np.testing.assert_allclose(segment.call_args.args[1], POINT)
        np.testing.assert_array_equal(segment.call_args.args[2], original)
        self.assertTrue(update.accepted[0], update.reasons)
        np.testing.assert_allclose(update.points_yx[0], POINT)
        self.assertEqual(update.labels[tuple(POINT.astype(int))], 1)

    def test_proposal_on_neighbor_is_held_when_original_cell_is_visible(self):
        tracker, _ = self.initialize()
        trusted = tracker._positions[0].targets["cell-1"]
        neighbor = POINT + (0, 40)
        labels = cell_mask(POINT).astype(np.uint16)
        labels[cell_mask(neighbor)] = 7
        update = tracker.update(
            0, 1, self.image, [TrenchTarget("cell-1", POINT, neighbor)],
            labels, pose_override=self.pose,
        )
        self.assertEqual(update.reasons, ("cell_identity",))
        np.testing.assert_allclose(update.points_yx[0], POINT)
        self.assertIs(tracker._positions[0].targets["cell-1"], trusted)

    def test_refinement_cannot_escape_proposal_into_neighbor_of_merged_mask(self):
        tracker, _ = self.initialize()
        neighbor = POINT + (0, 35)
        labels = np.zeros(SHAPE, dtype=np.uint16)
        labels[456:481, 887:948] = 1  # A merged Cellpose object covers both.
        with patch(
            "napari_raman_widget.trench_tracking._cell_watershed",
            return_value=cell_mask(neighbor),
        ):
            update = tracker.update(
                0, 1, self.image, [self.target], labels,
                pose_override=self.pose,
            )
        self.assertIn("same_cell", update.reasons[0])
        np.testing.assert_allclose(update.points_yx[0], POINT)

    def test_missing_original_mask_cannot_reassign_to_neighbor_within_45_pixels(self):
        tracker, _ = self.initialize()
        trusted = tracker._positions[0].targets["cell-1"]
        neighbor = POINT + (0, 40)
        labels = cell_mask(neighbor).astype(np.uint16) * 77
        update = tracker.update(
            0, 1, self.image, [TrenchTarget("cell-1", POINT, neighbor)],
            labels, pose_override=self.pose,
        )
        self.assertEqual(update.reasons, ("cell_identity",))
        np.testing.assert_allclose(update.points_yx[0], POINT)
        self.assertIs(tracker._positions[0].targets["cell-1"], trusted)

    def test_rejected_update_keeps_exact_previous_aim_despite_registered_drift(self):
        tracker, _ = self.initialize()
        update = tracker.update(
            0, 1, self.image, [self.target], np.zeros((512, 672), dtype=np.uint16),
            pose_override=TrenchPose(shift_yx=(4, 12)),
        )
        self.assertFalse(update.accepted[0])
        np.testing.assert_array_equal(update.points_yx[0], POINT)
        self.assertFalse(np.any(update.labels))

    def test_cell_identity_overlap_accounts_for_device_rotation_and_translation(self):
        tracker, _ = self.initialize()
        pose = TrenchPose(angle_deg=5, shift_yx=(0, 70))
        center = pose.apply(POINT, SHAPE) + (0, 3)
        mask = cell_mask(center)
        with patch(
            "napari_raman_widget.trench_tracking._cell_watershed",
            return_value=mask,
        ):
            update = tracker.update(
                0, 1, self.image, [TrenchTarget("cell-1", POINT, center)],
                mask.astype(np.uint16) * 99, pose_override=pose,
            )
        self.assertTrue(update.accepted[0], update.reasons)
        np.testing.assert_allclose(update.points_yx[0], np.argwhere(mask).mean(axis=0))
        self.assertAlmostEqual(update.results[0].step_px, 3.0, delta=0.1)

    def test_missing_proposed_cell_holds_and_recovers_after_label_renumbering(self):
        tracker, _ = self.initialize()
        trusted = tracker._positions[0].targets["cell-1"]
        labels = cell_mask(POINT + (0, 35)).astype(np.uint16)
        update = tracker.update(
            0, 1, self.image, [self.target], labels,
            pose_override=self.pose,
        )
        self.assertEqual(update.reasons, ("missing_proposed_cell",))
        np.testing.assert_allclose(update.points_yx[0], POINT)
        self.assertIs(tracker._positions[0].targets["cell-1"], trusted)
        shifted = POINT + (0, 3)
        mask = cell_mask(shifted)
        with patch(
            "napari_raman_widget.trench_tracking._cell_watershed",
            return_value=mask,
        ):
            recovered = tracker.update(
                0, 2, self.image, [TrenchTarget("cell-1", POINT, shifted)],
                mask.astype(np.uint16) * 99, pose_override=self.pose,
            )
        self.assertTrue(recovered.accepted[0], recovered.reasons)
        np.testing.assert_allclose(recovered.points_yx[0], shifted)

    def test_disconnected_same_label_cannot_expand_tracked_cell(self):
        mask = cell_mask(POINT)
        labels = (mask | cell_mask(POINT + (0, 40))).astype(np.uint16) * 9
        with patch(
            "napari_raman_widget.trench_tracking._cell_watershed",
            return_value=mask,
        ) as segment:
            update = TrenchConstraintTracker().update(
                0, 0, self.image, [self.target], labels,
                pose_override=self.pose,
            )
        segment.assert_not_called()
        self.assertTrue(update.accepted[0], update.reasons)
        np.testing.assert_array_equal(update.labels > 0, mask)

    def test_feature_extraction_does_not_change_template_or_image_view(self):
        image = np.arange(80 * 90, dtype=np.float32).reshape(80, 90)
        original = image.copy()
        first = _feature(image[10:50, 20:60])
        second = _feature(image[10:50, 20:60])
        np.testing.assert_array_equal(image, original)
        np.testing.assert_array_equal(first, second)

    def test_central_trench_excludes_neighbor_tracking_and_labels(self):
        repeat = (
            reference_lattice_vertices()[5, 0]
            - reference_lattice_vertices()[4, 0]
        )
        for pose in (
            self.pose,
            TrenchPose(angle_deg=3.0, shift_yx=(8.0, 12.0)),
        ):
            with self.subTest(pose=pose):
                center, neighbor = pose.apply(np.asarray([POINT, POINT + repeat]))
                central_mask = cell_mask(center)
                neighbor_mask = cell_mask(neighbor)
                labels = np.zeros(SHAPE, dtype=np.uint16)
                labels[central_mask] = 1
                labels[neighbor_mask] = 2
                with patch(
                    "napari_raman_widget.trench_tracking._cell_watershed",
                    return_value=central_mask,
                ) as watershed:
                    update = TrenchConstraintTracker().update(
                        0, 0, self.image,
                        [TrenchTarget("central", center), TrenchTarget("other", neighbor)],
                        labels,
                        pose_override=pose,
                        trench_center_yx=center,
                    )

                watershed.assert_not_called()
                self.assertTrue(update.accepted[0], update.reasons)
                self.assertFalse(update.accepted[1])
                self.assertEqual(update.reasons[1], "outside_selected_trench")
                np.testing.assert_allclose(update.points_yx[1], neighbor)
                self.assertTrue(np.any(update.labels[central_mask]))
                self.assertFalse(np.any(update.labels[neighbor_mask]))

    def test_empty_target_list_still_filters_discovery_to_central_trench(self):
        repeat = (
            reference_lattice_vertices()[5, 0]
            - reference_lattice_vertices()[4, 0]
        )
        labels = np.zeros(SHAPE, dtype=np.uint16)
        central_mask = cell_mask(POINT)
        neighbor_mask = cell_mask(POINT + repeat)
        labels[central_mask] = 1
        labels[neighbor_mask] = 2
        with patch(
            "napari_raman_widget.trench_tracking._cell_watershed"
        ) as watershed:
            update = TrenchConstraintTracker().update(
                0, 1, self.image, [], labels,
                pose_override=self.pose,
                trench_center_yx=POINT,
            )

        watershed.assert_not_called()
        self.assertTrue(np.any(update.labels[central_mask]))
        self.assertFalse(np.any(update.labels[neighbor_mask]))

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

    def test_closed_tip_component_is_removed_but_deeper_cell_survives(self):
        labels = np.zeros((256, 336), dtype=np.uint16)
        _, tip, inward, _ = trench_axis(self.pose)
        cap_center = np.rint((tip + inward * 20) / 4).astype(int)
        cell_center = np.rint((tip + inward * 70) / 4).astype(int)
        rows, columns = disk(cap_center, 3, shape=labels.shape)
        labels[rows, columns] = 9
        rows, columns = disk(cell_center, 4, shape=labels.shape)
        labels[rows, columns] = 9

        result = filter_labels_to_trenches(
            self.image,
            labels,
            pose_override=self.pose,
        )

        kept = np.unique(result.filtered_labels)
        self.assertEqual(np.count_nonzero(kept), 1)
        self.assertEqual(result.filtered_labels[tuple(cap_center)], 0)
        self.assertNotEqual(result.filtered_labels[tuple(cell_center)], 0)
        self.assertFalse(result.safe_interior_mask[tuple(cap_center)])
        self.assertTrue(result.safe_interior_mask[tuple(cell_center)])

    def test_attached_tip_nose_is_clipped_before_center_is_calculated(self):
        labels = np.zeros((256, 336), dtype=np.uint16)
        _, tip, inward, tip_length = trench_axis(self.pose)
        nose = np.rint((tip + inward * 10) / 4).astype(int)
        cell_center = np.rint((tip + inward * 65) / 4).astype(int)
        rows, columns = line(*nose, *cell_center)
        labels[rows, columns] = 5
        rows, columns = disk(cell_center, 4, shape=labels.shape)
        labels[rows, columns] = 5
        raw_center = np.argwhere(labels == 5).mean(axis=0) * 4

        result = filter_labels_to_trenches(
            self.image,
            labels,
            pose_override=self.pose,
        )

        kept = result.filtered_labels != 0
        self.assertTrue(np.any(kept))
        filtered_pixels = np.argwhere(kept) * 4
        filtered_depths = (filtered_pixels - tip) @ inward
        filtered_center = filtered_pixels.mean(axis=0)
        self.assertGreaterEqual(filtered_depths.min(), 2.0 * tip_length - 4)
        self.assertGreater(
            (filtered_center - tip) @ inward,
            (raw_center - tip) @ inward,
        )

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
