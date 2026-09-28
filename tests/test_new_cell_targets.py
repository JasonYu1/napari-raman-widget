"""Tests for selecting targets from newly appearing segmentation labels."""

from __future__ import annotations

import unittest

import numpy as np

from napari_raman_widget.new_cell_targets import find_new_cell_targets


class TestNewCellTargets(unittest.TestCase):
    def test_tracked_ids_find_only_current_only_track(self) -> None:
        previous = np.zeros((30, 30), dtype=np.uint16)
        current = np.zeros_like(previous)
        previous[3:10, 3:10] = 10
        current[4:11, 4:11] = 99
        current[18:25, 19:26] = 42

        tracked = np.zeros((2, 30, 30), dtype=np.uint16)
        tracked[0, 3:10, 3:10] = 1
        tracked[1, 4:11, 4:11] = 1
        tracked[1, 18:25, 19:26] = 2

        label_ids, points = find_new_cell_targets(
            previous,
            current,
            np.empty((0, 2)),
            scale=2,
            tracked_labels=tracked,
        )

        np.testing.assert_array_equal(label_ids, [2])
        self.assertEqual(points.shape, (1, 2))
        pixel = np.floor(points[0] / 2).astype(int)
        self.assertEqual(tracked[1, pixel[0], pixel[1]], 2)

    def test_manual_point_on_new_track_prevents_duplicate(self) -> None:
        previous = np.zeros((20, 20), dtype=np.int32)
        current = np.zeros_like(previous)
        current[10:17, 10:17] = 8
        tracked = np.stack((previous, current))

        label_ids, points = find_new_cell_targets(
            previous,
            current,
            np.array([[26.0, 28.0]]),
            scale=2,
            tracked_labels=tracked,
        )

        self.assertEqual(label_ids.shape, (0,))
        self.assertEqual(points.shape, (0, 2))

    def test_raw_fallback_ignores_renumbered_nearby_object(self) -> None:
        previous = np.zeros((35, 35), dtype=np.int32)
        current = np.zeros_like(previous)
        previous[3:10, 3:10] = 1
        current[4:11, 4:11] = 73
        current[24:31, 24:31] = 1

        label_ids, points = find_new_cell_targets(
            previous,
            current,
            np.empty((0, 2)),
            fallback_match_distance=3,
        )

        # Raw ID 1 is reused by a different, far-away object and raw ID 73 is
        # a continuation.  Geometry, not either numeric coincidence, decides.
        np.testing.assert_array_equal(label_ids, [1])
        self.assertEqual(points.shape, (1, 2))
        pixel = np.floor(points[0]).astype(int)
        self.assertEqual(current[pixel[0], pixel[1]], 1)

    def test_raw_fallback_treats_all_cells_as_new_after_empty_mask(self) -> None:
        previous = np.zeros((30, 30), dtype=np.int32)
        current = np.zeros_like(previous)
        current[2:9, 2:9] = 9
        current[18:25, 18:25] = 4

        label_ids, points = find_new_cell_targets(
            previous,
            current,
            np.empty((0, 2)),
            fallback_match_distance=100,
        )

        np.testing.assert_array_equal(label_ids, [4, 9])
        self.assertEqual(points.shape, (2, 2))

    def test_filters_small_and_thin_objects(self) -> None:
        previous = np.zeros((35, 35), dtype=np.int32)
        current = np.zeros_like(previous)
        current[2:4, 2:4] = 1  # Too little area.
        current[10:30, 10] = 2  # Enough area, but no safe interior radius.
        current[20:27, 22:29] = 3
        tracked = np.stack((previous, current))

        label_ids, points = find_new_cell_targets(
            previous,
            current,
            np.empty((0, 2)),
            tracked_labels=tracked,
            minimum_area=16,
            minimum_interior_radius=2,
        )

        np.testing.assert_array_equal(label_ids, [3])
        self.assertEqual(points.shape, (1, 2))

    def test_deepest_point_is_inside_concave_label_and_scaled(self) -> None:
        previous = np.zeros((20, 20), dtype=np.int32)
        current = np.zeros_like(previous)
        current[3:15, 3:7] = 5
        current[11:15, 3:15] = 5
        tracked = np.stack((previous, current))

        label_ids, points = find_new_cell_targets(
            previous,
            current,
            np.empty((0, 2)),
            scale=4,
            tracked_labels=tracked,
        )

        np.testing.assert_array_equal(label_ids, [5])
        pixel = np.floor(points[0] / 4).astype(int)
        self.assertEqual(current[pixel[0], pixel[1]], 5)
        np.testing.assert_array_equal(points[0] % 4, [0, 0])

    def test_cap_prefers_cells_closest_to_reference_center(self) -> None:
        previous = np.zeros((40, 40), dtype=np.int32)
        current = np.zeros_like(previous)
        current[2:9, 2:9] = 1
        current[17:24, 17:24] = 3
        current[29:36, 29:36] = 2
        tracked = np.stack((previous, current))

        label_ids, _ = find_new_cell_targets(
            previous,
            current,
            np.empty((0, 2)),
            tracked_labels=tracked,
            center_yx=(20, 20),
            max_new_cells=2,
        )

        np.testing.assert_array_equal(label_ids, [3, 2])

    def test_invalid_tracked_shape_is_rejected(self) -> None:
        labels = np.zeros((10, 10), dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "shape \\(2, Y, X\\)"):
            find_new_cell_targets(
                labels,
                labels,
                np.empty((0, 2)),
                tracked_labels=np.zeros((3, 10, 10), dtype=np.int32),
            )


if __name__ == "__main__":
    unittest.main()
