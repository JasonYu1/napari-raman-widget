import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from scipy import ndimage as ndi

from napari_raman_widget.cell_identity import CellMatch, cell_reference, match_cell
from napari_raman_widget.trench_tracking import (
    DEFAULT_TRENCH_CONFIG, TrenchConstraintTracker, TrenchPose, TrenchTarget,
)


def textured_cell(shape, point, seed=1):
    image = np.zeros(shape, dtype=np.float32)
    y, x = np.indices((61, 61))
    radius = np.hypot(y - 30, x - 30)
    noise = ndi.gaussian_filter(np.random.default_rng(seed).normal(size=(61, 61)), 1)
    cell = noise * (radius < 22) + np.exp(-((radius - 23) / 2) ** 2)
    row, col = np.rint(point).astype(int)
    image[row-30:row+31, col-30:col+31] = cell
    return image


class CellIdentityTests(unittest.TestCase):
    def test_translated_cell_is_found_despite_brightness_change_and_neighbor(self):
        point = np.asarray((150.25, 200.1))
        original = textured_cell((320, 420), point)
        target = point + (-4, 33)
        current = textured_cell(original.shape, target) * 1.7 + 500
        current += textured_cell(original.shape, point + (0, -30), seed=7)
        before = original.copy()
        reference = cell_reference(original, point, 45)
        result = match_cell(reference, current, point, 45)
        self.assertTrue(result.accepted, result)
        np.testing.assert_allclose(result.point_yx, target)
        self.assertLessEqual(result.reverse_error, 1)
        np.testing.assert_array_equal(original, before)

    def test_missing_cell_rejects_and_does_not_poison_reference(self):
        point = np.asarray((150., 200.))
        original = textured_cell((320, 420), point)
        reference = cell_reference(original, point, 45)
        before = reference.image.copy()
        rejected = match_cell(reference, np.zeros_like(original), point, 45)
        self.assertFalse(rejected.accepted)
        np.testing.assert_array_equal(reference.image, before)
        recovered = match_cell(reference, ndi.shift(original, (2, 12)), point, 45)
        self.assertTrue(recovered.accepted)
        np.testing.assert_allclose(recovered.point_yx, point + (2, 12))

    def test_two_indistinguishable_candidates_are_ambiguous(self):
        point = np.asarray((150., 200.))
        original = textured_cell((320, 420), point)
        current = (
            textured_cell(original.shape, point + (0, -30))
            + textured_cell(original.shape, point + (0, 30))
        )
        result = match_cell(cell_reference(original, point, 45), current, point, 45)
        self.assertFalse(result.accepted, result)
        self.assertLess(result.margin, 0.05)

    def test_custom_movement_limit_and_image_bounds(self):
        point = np.asarray((150., 200.))
        original = textured_cell((320, 420), point)
        current = ndi.shift(original, (0, 20))
        result = match_cell(cell_reference(original, point, 12), current, point, 12)
        self.assertFalse(result.accepted, result)
        self.assertIsNone(cell_reference(original, (4, 4), 45))

    def test_same_cell_is_identified_then_centered_without_moving_identity_reference(self):
        point = np.asarray((468., 900.))
        shape = (1024, 1344)
        original = textured_cell(shape, point)
        tracker = TrenchConstraintTracker()
        yy, xx = np.indices(shape)
        original_mask = (yy-point[0])**2 + (xx-point[1])**2 < 18**2
        with patch('napari_raman_widget.trench_tracking._cell_watershed') as watershed:
            first = tracker.update(0, 0, original, [TrenchTarget('a', point)],
                original_mask.astype(np.uint16), pose_override=TrenchPose())
        watershed.assert_not_called()
        self.assertTrue(first.accepted[0], first.reasons)
        shifted = point + (0, 30)
        current = textured_cell(shape, shifted)
        # Ignore an upstream proposal on a neighbor. BF identifies the mask;
        # its asymmetric outline then centers the aim, not the identity anchor.
        merged = np.zeros(shape, dtype=np.uint16)
        merged[450:488, 905:970] = 77
        with patch('napari_raman_widget.trench_tracking._cell_watershed') as watershed:
            result = tracker.update(0, 1, current,
                [TrenchTarget('a', point, point + (0, -35))], merged,
                pose_override=TrenchPose())
        watershed.assert_not_called()
        self.assertTrue(result.accepted[0], result.reasons)
        np.testing.assert_allclose(result.points_yx[0], np.argwhere(merged).mean(0))
        np.testing.assert_array_equal(result.labels > 0, merged > 0)
        state = tracker._positions[0].appearance_targets['a']
        np.testing.assert_allclose(state.reference.point_yx, shifted)
        self.assertGreater(np.linalg.norm(result.points_yx[0]-shifted), 6)
        # A mask-boundary change must not walk the next BF reference across
        # the image. The same feature still identifies this cell at t2.
        third_mask = merged.copy()
        third_mask[:, 960:] = 0
        third = tracker.update(0, 2, current,
            [TrenchTarget('a', result.points_yx[0])], third_mask,
            pose_override=TrenchPose())
        self.assertTrue(third.accepted[0], third.reasons)
        np.testing.assert_allclose(third.points_yx[0], np.argwhere(third_mask).mean(0))
        np.testing.assert_allclose(tracker._positions[0].appearance_targets['a'].reference.point_yx, shifted)
        trusted = tracker._positions[0].appearance_targets['a']
        lost = tracker.update(0, 3, np.zeros(shape), [TrenchTarget('a', third.points_yx[0])],
            np.zeros(shape, dtype=np.uint16), pose_override=TrenchPose(shift_yx=(0, 8)))
        self.assertFalse(lost.accepted[0])
        np.testing.assert_array_equal(lost.points_yx[0], third.points_yx[0])
        self.assertIs(tracker._positions[0].appearance_targets['a'], trusted)

    def test_cellpose_outline_and_holes_are_preserved_at_each_scale(self):
        point = np.asarray((468.25, 900.2))
        shape = (1024, 1344)
        image = textured_cell(shape, point)
        for scale in (1, 2, 4, 2.5):
            with self.subTest(scale=scale):
                label_shape = tuple(int(round(v / scale)) for v in shape)
                y, x = np.indices(label_shape)
                y = (y + 0.5) * shape[0] / label_shape[0]
                x = (x + 0.5) * shape[1] / label_shape[1]
                # An asymmetric Cellpose outline with a notch and a small
                # hole: neither watershed nor hole filling may reshape it.
                cell = ((y >= 452) & (y < 484) & (x >= 878) & (x < 934)
                    & ~((y < 460) & (x > 920))
                    & ~((y >= 456) & (y < 464) & (x >= 886) & (x < 894)))
                neighbor = (y >= 456) & (y < 480) & (x >= 970) & (x < 998)
                labels = cell.astype(np.uint16) * 17
                labels[neighbor] = 95
                before = labels.copy()
                with patch('napari_raman_widget.trench_tracking._cell_watershed') as watershed:
                    result = TrenchConstraintTracker().update(0, 0, image,
                        [TrenchTarget('a', point)], labels, scale,
                        pose_override=TrenchPose())
                self.assertTrue(result.accepted[0], result.reasons)
                watershed.assert_not_called()
                np.testing.assert_array_equal(result.labels == 1, cell)
                np.testing.assert_array_equal(result.labels == 2, neighbor)
                np.testing.assert_array_equal(labels, before)
                expected = (np.argwhere(cell).mean(0)+0.5)*np.asarray(shape)/labels.shape-0.5
                np.testing.assert_allclose(result.points_yx[0], expected)

    def test_missing_cellpose_mask_holds_without_fabricating_and_recovers(self):
        point = np.asarray((468., 900.))
        shape = (1024, 1344)
        yy, xx = np.indices(shape)
        def labels_at(center):
            return (((yy-center[0])**2 + (xx-center[1])**2) < 18**2).astype(np.uint16)
        tracker = TrenchConstraintTracker()
        with patch('napari_raman_widget.trench_tracking._cell_watershed') as watershed:
            first = tracker.update(0, 0, textured_cell(shape, point),
                [TrenchTarget('a', point)], labels_at(point), pose_override=TrenchPose())
            self.assertTrue(first.accepted[0], first.reasons)
            trusted = tracker._positions[0].appearance_targets['a']
            shifted = point + (0, 6)
            missed = tracker.update(0, 1, textured_cell(shape, shifted),
                [TrenchTarget('a', point)], labels_at(point+(0, 40)),
                pose_override=TrenchPose())
            self.assertFalse(missed.accepted[0])
            self.assertEqual(missed.reasons, ('missing_cellpose_mask',))
            np.testing.assert_array_equal(missed.points_yx[0], point)
            self.assertIs(tracker._positions[0].appearance_targets['a'], trusted)
            self.assertEqual(missed.labels[tuple(shifted.astype(int))], 0)
            recovered = tracker.update(0, 2, textured_cell(shape, shifted),
                [TrenchTarget('a', point)], labels_at(shifted)*99,
                pose_override=TrenchPose())
            self.assertTrue(recovered.accepted[0], recovered.reasons)
            np.testing.assert_allclose(recovered.points_yx[0], shifted)
            watershed.assert_not_called()

    def test_large_cellpose_mother_cell_keeps_its_outline_and_identity_at_each_scale(self):
        shape = (1024, 1344)
        point = np.asarray((468., 900.))
        for scale in (1, 2, 4):
            with self.subTest(scale=scale):
                tracker = TrenchConstraintTracker()
                label_shape = tuple(size // scale for size in shape)
                yy, xx = np.indices(label_shape)
                yy = (yy + .5) * scale - .5
                xx = (xx + .5) * scale - .5
                previous = point.copy()
                for t, shift in enumerate((0, 8)):
                    center = point + (0, shift)
                    cell = ((yy-center[0])/17)**2 + ((xx-center[1])/55)**2 < 1
                    labels = cell.astype(np.uint16) * (17 + t)
                    update = tracker.update(0, t, textured_cell(shape, center),
                        [TrenchTarget('mother', previous, point + (0, -35))],
                        labels, scale, pose_override=TrenchPose())
                    self.assertTrue(update.accepted[0], update.reasons)
                    self.assertGreater(update.results[0].area_px, 2600)
                    np.testing.assert_array_equal(update.labels == 1, cell)
                    centroid = (np.argwhere(cell).mean(0)+.5)*scale-.5
                    np.testing.assert_allclose(update.points_yx[0], centroid)
                    state = tracker._positions[0].appearance_targets['mother']
                    np.testing.assert_allclose(state.reference.point_yx, center)
                    self.assertLessEqual(update.results[0].step_px, 45)
                    previous = update.points_yx[0]

    def test_oversized_cellpose_mask_still_holds_without_poisoning_identity(self):
        shape = (1024, 1344)
        point = np.asarray((468., 900.))
        yy, xx = np.indices(shape)
        image = textured_cell(shape, point)
        cell = ((yy-point[0])/19)**2 + ((xx-point[1])/50)**2 < 1
        tracker = TrenchConstraintTracker()
        first = tracker.update(0, 0, image, [TrenchTarget('mother', point)],
            cell.astype(np.uint16), pose_override=TrenchPose())
        self.assertTrue(first.accepted[0], first.reasons)
        trusted = tracker._positions[0].appearance_targets['mother']
        oversized = ((yy-point[0])/22)**2 + ((xx-point[1])/90)**2 < 1
        rejected = tracker.update(0, 1, image,
            [TrenchTarget('mother', first.points_yx[0])],
            oversized.astype(np.uint16), pose_override=TrenchPose())
        self.assertFalse(rejected.accepted[0])
        self.assertIn('cell_area', rejected.reasons[0])
        np.testing.assert_array_equal(rejected.points_yx, first.points_yx)
        self.assertFalse(np.any(rejected.labels))
        self.assertIs(tracker._positions[0].appearance_targets['mother'], trusted)
        recovered = tracker.update(0, 2, image,
            [TrenchTarget('mother', first.points_yx[0])],
            cell.astype(np.uint16)*99, pose_override=TrenchPose())
        self.assertTrue(recovered.accepted[0], recovered.reasons)
        np.testing.assert_array_equal(recovered.labels == 1, cell)

    def test_centroid_movement_limit_holds_without_poisoning_and_accepts_boundary(self):
        point = np.asarray((468., 900.))
        shape = (1024, 1344)
        yy, xx = np.indices(shape)
        original_mask = ((yy-point[0])**2+(xx-point[1])**2 < 18**2).astype(np.uint16)
        image = textured_cell(shape, point)
        for limit in (45, 12):
            with self.subTest(limit=limit):
                tracker = TrenchConstraintTracker(replace(DEFAULT_TRENCH_CONFIG, maximum_step_px=limit))
                first = tracker.update(0, 0, image, [TrenchTarget('a', point)],
                    original_mask, pose_override=TrenchPose())
                self.assertTrue(first.accepted[0], first.reasons)
                trusted = tracker._positions[0].appearance_targets['a']
                extended = np.zeros(shape, dtype=np.uint16)
                extended[462:475, 890:900+2*(limit+1)+11] = 8
                rejected = tracker.update(0, 1, image, [TrenchTarget('a', point)],
                    extended, pose_override=TrenchPose())
                self.assertFalse(rejected.accepted[0])
                self.assertIn('maximum_step', rejected.reasons[0])
                np.testing.assert_array_equal(rejected.points_yx[0], point)
                self.assertIs(tracker._positions[0].appearance_targets['a'], trusted)
                extended[:, 900+2*limit+11:] = 0
                accepted = tracker.update(0, 2, image, [TrenchTarget('a', point)],
                    extended, pose_override=TrenchPose())
                self.assertTrue(accepted.accepted[0], accepted.reasons)
                np.testing.assert_allclose(accepted.points_yx[0], point+(0, limit))
                self.assertAlmostEqual(accepted.results[0].step_px, limit)

    def test_centroid_in_mask_hole_is_held_instead_of_projected(self):
        shape = (1024, 1344)
        point = np.asarray((468., 900.))
        image = textured_cell(shape, point)
        yy, xx = np.indices(shape)
        mask = (yy >= 450) & (yy < 487) & (xx >= 890) & (xx < 931)
        mask &= ~((yy >= 462) & (yy < 475) & (xx >= 904) & (xx < 917))
        update = TrenchConstraintTracker().update(0, 0, image,
            [TrenchTarget('a', point)], mask.astype(np.uint16), pose_override=TrenchPose())
        self.assertFalse(update.accepted[0])
        self.assertIn('centroid_in_cell', update.reasons[0])
        np.testing.assert_array_equal(update.points_yx[0], point)

    def test_movement_is_measured_from_previous_centroid_not_identity_feature(self):
        point = np.asarray((468., 900.))
        shape = (1024, 1344)
        yy, xx = np.indices(shape)
        tracker = TrenchConstraintTracker(replace(DEFAULT_TRENCH_CONFIG, maximum_step_px=12))
        def labels_at(center):
            return ((yy-center[0])**2+(xx-center[1])**2 < 18**2).astype(np.uint16)
        first_center = point+(0, 8)
        first = tracker.update(0, 0, textured_cell(shape, point),
            [TrenchTarget('a', point)], labels_at(first_center), pose_override=TrenchPose())
        self.assertTrue(first.accepted[0], first.reasons)
        np.testing.assert_allclose(first.points_yx[0], first_center)
        shifted = point+(0, 10)
        next_center = first_center+(0, 10)
        second = tracker.update(0, 1, textured_cell(shape, shifted),
            [TrenchTarget('a', first.points_yx[0])], labels_at(next_center), pose_override=TrenchPose())
        self.assertTrue(second.accepted[0], second.reasons)
        np.testing.assert_allclose(second.points_yx[0], next_center)
        self.assertAlmostEqual(second.results[0].step_px, 10)
        np.testing.assert_allclose(tracker._positions[0].appearance_targets['a'].reference.point_yx, shifted)

    def test_centroid_limit_compensates_device_motion_and_stays_in_trench(self):
        point = np.asarray((468., 900.))
        shape = (1024, 1344)
        yy, xx = np.indices(shape)
        tracker = TrenchConstraintTracker()
        labels = ((yy-point[0])**2+(xx-point[1])**2 < 18**2).astype(np.uint16)
        first = tracker.update(0, 0, textured_cell(shape, point),
            [TrenchTarget('a', point)], labels, pose_override=TrenchPose())
        self.assertTrue(first.accepted[0], first.reasons)
        pose = TrenchPose(angle_deg=3, shift_yx=(4, 70))
        predicted = pose.apply(point, shape)
        center = predicted+(0, 4)
        current_labels = ((yy-center[0])**2+(xx-center[1])**2 < 18**2).astype(np.uint16)
        with patch('napari_raman_widget.trench_tracking.match_cell',
            return_value=CellMatch(predicted, True, 1., 0., 1.)):
            result = tracker.update(0, 1, textured_cell(shape, predicted),
                [TrenchTarget('a', first.points_yx[0])], current_labels,
                pose_override=pose, trench_center_yx=predicted)
        self.assertTrue(result.accepted[0], result.reasons)
        np.testing.assert_allclose(result.points_yx[0], np.argwhere(current_labels).mean(0))
        self.assertLess(result.results[0].step_px, 5)
        self.assertGreater(np.linalg.norm(result.points_yx[0]-point), 45)


if __name__ == '__main__':
    unittest.main()
