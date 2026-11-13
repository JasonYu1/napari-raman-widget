"""Learned-trench safeguards for initial automated cell selection."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from napari_raman_widget.selection import automatic


class _Core:
    def __init__(self, image: np.ndarray):
        self.image = image

    def setXYPosition(self, x, y):
        del x, y

    def waitForSystem(self):
        return None

    def snapImage(self):
        return None

    def getImage(self):
        return self.image


class _PillarFilter:
    def __init__(self, filtered_labels: np.ndarray, removed_mask: np.ndarray):
        self.filtered_labels = filtered_labels
        self.removed_mask = removed_mask
        self.config = SimpleNamespace(
            pillar_length=78.0,
            pillar_width=56.0,
            minimum_rectangle_fit=0.75,
        )

    def filter(self, labels, *, image_shape):
        self.labels = labels
        self.image_shape = image_shape
        return SimpleNamespace(
            filtered_labels=self.filtered_labels,
            removed_mask=self.removed_mask,
            measurements=(),
        )


class AutomaticSelectionTrenchTests(unittest.TestCase):
    def setUp(self):
        self.image = np.arange(36, dtype=np.uint16).reshape(6, 6)
        self.sequence = SimpleNamespace(
            stage_positions=[SimpleNamespace(x=1.0, y=2.0)]
        )
        self.raw_labels = np.zeros((6, 6), dtype=np.uint16)
        self.raw_labels[0:2, 0:2] = 1
        self.raw_labels[2:5, 2:5] = 2
        self.viewer = MagicMock()
        self.trench_input_safe = np.zeros(self.image.shape, dtype=bool)
        self.trench_input_safe[1:5, 1:5] = True
        self.pose = object()
        self.geometry = SimpleNamespace(
            geometry_valid=True,
            geometry_score=0.91,
            safe_union=self.trench_input_safe,
            trench_masks=(self.trench_input_safe.copy(),),
            safe_masks=(self.trench_input_safe.copy(),),
            pose=self.pose,
        )

    def test_filter_runs_after_pillars_and_before_selection_and_display(self):
        after_pillars = self.raw_labels.copy()
        after_pillars[after_pillars == 1] = 0
        pillar_pixels = self.raw_labels == 1
        pillar_filter = _PillarFilter(after_pillars, pillar_pixels)
        safe_labels = np.zeros_like(self.raw_labels, dtype=np.uint32)
        safe_labels[3:5, 3:5] = 7
        trench_filter = MagicMock(
            return_value=SimpleNamespace(
                filtered_labels=safe_labels,
                safe_interior_mask=safe_labels != 0,
                geometry_valid=True,
                geometry_score=0.91,
            )
        )
        selector = MagicMock(return_value=np.asarray([[3.5, 3.5]]))
        finish = MagicMock(return_value=("sources", "autofocus", "sequence"))

        segmenter = MagicMock(
            return_value=np.repeat(
                np.repeat(self.raw_labels, 2, axis=0),
                2,
                axis=1,
            )
        )
        with (
            patch.object(
                automatic,
                "get_seq_from_napari",
                return_value=self.sequence,
            ),
            patch.object(
                automatic,
                "make_pillar_shape_filter",
                return_value=pillar_filter,
            ),
            patch.object(
                automatic,
                "segment_single_img",
                new=segmenter,
            ),
            patch.object(
                automatic,
                "registered_trench_geometry",
                return_value=self.geometry,
            ),
            patch.object(
                automatic,
                "filter_labels_to_trenches",
                new=trench_filter,
            ),
            patch.object(
                automatic,
                "get_n_most_centered_coms",
                new=selector,
            ),
            patch.object(
                automatic,
                "_finish_original_position_selection",
                new=finish,
            ),
            patch.object(
                automatic,
                "tqdm",
                side_effect=lambda values, **_: values,
            ),
        ):
            result = automatic.automated_point_selections(
                _Core(self.image),
                viewer=self.viewer,
                main_window=SimpleNamespace(),
                point_transformer=SimpleNamespace(),
                N=2,
                center=(3.0, 3.0),
                radius=3.0,
                autofocus_object="glass",
                bkd_thres=0,
                stage_settle_time=0,
                image_settle_time=0,
                suppress_pillars=True,
                constrain_trenches=True,
                cellpose_upsample=2,
                show_masks=True,
                cell_point_refiner=lambda points: np.zeros_like(points),
            )

        self.assertEqual(result, ("sources", "autofocus", "sequence"))
        expected_input = self.image.copy()
        expected_input[~self.trench_input_safe] = np.median(
            self.image[self.trench_input_safe]
        )
        expected_input = np.repeat(
            np.repeat(expected_input, 2, axis=0),
            2,
            axis=1,
        )
        np.testing.assert_array_equal(segmenter.call_args.args[0], expected_input)
        self.assertIs(segmenter.call_args.kwargs["crop"], False)
        self.assertNotIn("circle_radius", segmenter.call_args.kwargs)
        self.assertNotIn("circle_center", segmenter.call_args.kwargs)
        self.assertIs(trench_filter.call_args.args[0], self.image)
        np.testing.assert_array_equal(
            trench_filter.call_args.args[1],
            after_pillars,
        )
        self.assertIs(
            trench_filter.call_args.kwargs["pose_override"],
            self.pose,
        )
        np.testing.assert_array_equal(selector.call_args.args[0], safe_labels)
        self.assertIsNone(selector.call_args.kwargs["autofocus_object"])
        np.testing.assert_array_equal(
            selector.call_args.kwargs["exclusion_mask"],
            pillar_pixels | ~(safe_labels != 0),
        )
        self.viewer.add_labels.assert_called_once()
        np.testing.assert_array_equal(
            self.viewer.add_labels.call_args.args[0],
            safe_labels,
        )
        self.assertEqual(
            self.viewer.add_labels.call_args.kwargs["name"],
            "Cellpose mask p0",
        )
        # The first point is background inside the circle and outside the raw
        # trench; the refined cell remains at its original wall-safe center.
        selected = finish.call_args.kwargs["selected_by_position"][0]
        background_pixel = tuple(selected[0].astype(int))
        selection_circle = automatic._circle_selection_mask(
            self.image.shape,
            (3.0, 3.0),
            3.0,
        )
        self.assertTrue(selection_circle[background_pixel])
        self.assertFalse(self.trench_input_safe[background_pixel])
        np.testing.assert_allclose(
            selected[1:],
            [[3.5, 3.5]],
        )

    def test_low_confidence_geometry_skips_field_without_targets(self):
        selector = MagicMock()
        finish = MagicMock()
        failed_geometry = SimpleNamespace(
            geometry_valid=False,
            geometry_score=0.2,
            safe_union=np.zeros_like(self.raw_labels, dtype=bool),
            pose=None,
        )
        segmenter = MagicMock()
        trench_filter = MagicMock()

        with (
            patch.object(
                automatic,
                "get_seq_from_napari",
                return_value=self.sequence,
            ),
            patch.object(
                automatic,
                "make_pillar_shape_filter",
                return_value=None,
            ),
            patch.object(
                automatic,
                "segment_single_img",
                new=segmenter,
            ),
            patch.object(
                automatic,
                "filter_labels_to_trenches",
                new=trench_filter,
            ),
            patch.object(
                automatic,
                "registered_trench_geometry",
                return_value=failed_geometry,
            ),
            patch.object(
                automatic,
                "get_n_most_centered_coms",
                new=selector,
            ),
            patch.object(
                automatic,
                "_finish_original_position_selection",
                new=finish,
            ),
            patch.object(
                automatic,
                "tqdm",
                side_effect=lambda values, **_: values,
            ),
        ):
            result = automatic.automated_point_selections(
                _Core(self.image),
                viewer=self.viewer,
                main_window=SimpleNamespace(),
                point_transformer=SimpleNamespace(),
                N=1,
                stage_settle_time=0,
                image_settle_time=0,
                constrain_trenches=True,
                show_masks=True,
            )

        self.assertEqual(result, finish.return_value)
        segmenter.assert_not_called()
        trench_filter.assert_not_called()
        selector.assert_not_called()
        finish.assert_called_once()
        selected = finish.call_args.kwargs["selected_by_position"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].shape, (0, 2))
        self.viewer.add_labels.assert_called_once()
        np.testing.assert_array_equal(
            self.viewer.add_labels.call_args.args[0],
            np.zeros_like(self.raw_labels),
        )

    def test_refinement_must_remain_in_the_same_safe_label(self):
        labels = np.zeros((8, 8), dtype=np.uint16)
        labels[1:4, 1:4] = 2
        labels[5:8, 5:8] = 3
        original = np.asarray([[2.0, 2.0], [6.0, 6.0], [2.0, 2.0]])
        proposed = np.asarray([[3.0, 3.0], [2.0, 2.0], [-1.0, 4.0]])

        retained = automatic._retain_same_label_refinements(
            original,
            proposed,
            labels,
        )

        np.testing.assert_allclose(
            retained,
            [[3.0, 3.0], [6.0, 6.0], [2.0, 2.0]],
        )

    def test_cellpose_receives_only_tight_circle_trench_crop(self):
        image = np.arange(30 * 40, dtype=np.uint16).reshape(30, 40)
        safe = np.zeros(image.shape, dtype=bool)
        safe[8:18, 15:30] = True
        circle = automatic._circle_selection_mask(
            image.shape,
            center=(12.0, 21.0),
            radius=5.0,
        )
        allowed = safe & circle
        received = {}

        def segmenter(segment_image, **kwargs):
            received["image"] = np.array(segment_image, copy=True)
            received["kwargs"] = dict(kwargs)
            return np.ones(segment_image.shape, dtype=np.uint16)

        with patch.object(automatic, "segment_single_img", new=segmenter):
            labels = automatic._segment_trench_cellpose_input(
                image,
                allowed,
                upsample=2,
                cellpose_model="cyto2",
            )

        pixels = np.argwhere(allowed)
        lower = np.maximum(pixels.min(axis=0) - 4, 0)
        upper = np.minimum(pixels.max(axis=0) + 5, image.shape)
        crop_shape = tuple((upper - lower) * 2)
        self.assertEqual(received["image"].shape, crop_shape)
        self.assertLess(received["image"].size, image.size * 4)
        self.assertFalse(received["kwargs"]["crop"])
        np.testing.assert_array_equal(labels != 0, allowed)

    def test_background_is_inside_circle_and_outside_raw_trenches(self):
        labels = np.zeros((30, 30), dtype=np.uint16)
        labels[12:15, 12:15] = 1
        circle = automatic._circle_selection_mask(
            labels.shape,
            center=(15.0, 15.0),
            radius=10.0,
        )
        raw_trenches = np.zeros(labels.shape, dtype=bool)
        raw_trenches[10:21, 8:23] = True
        allowed_background = circle & ~raw_trenches

        point = automatic._background_point_in_region(
            labels,
            allowed_background,
            threshold=0,
            center=(15.0, 15.0),
        )

        pixel = tuple(point.astype(int))
        self.assertTrue(circle[pixel])
        self.assertFalse(raw_trenches[pixel])

    def test_background_falls_back_when_clearance_is_impossible(self):
        labels = np.zeros((20, 20), dtype=np.uint16)
        labels[9:12, 9:12] = 1
        allowed = np.zeros(labels.shape, dtype=bool)
        allowed[2:18, 2:18] = True

        point = automatic._background_point_in_region(
            labels,
            allowed,
            threshold=10_000,
            center=(10.0, 10.0),
        )

        pixel = tuple(point.astype(int))
        self.assertTrue(allowed[pixel])
        self.assertEqual(labels[pixel], 0)

    def test_background_gap_between_neighboring_trench_labels_is_filled(self):
        labels = np.zeros((24, 60), dtype=np.uint16)
        safe = np.zeros(labels.shape, dtype=bool)
        safe[5:19, 5:55] = True
        labels[8:16, 10:18] = 1
        labels[8:16, 38:46] = 2

        filled = automatic._fill_interlabel_trench_gaps(
            labels,
            (safe,),
        )

        self.assertFalse(np.any(filled[5:19, 18:38] == 0))
        self.assertTrue(np.any(filled[5:19, 18:38] == 1))
        self.assertTrue(np.any(filled[5:19, 18:38] == 2))
        self.assertTrue(np.all(filled[5:19, 5:10] == 0))
        self.assertTrue(np.all(filled[5:19, 46:55] == 0))

    def test_labels_in_different_trenches_do_not_fill_between_trenches(self):
        labels = np.zeros((30, 60), dtype=np.uint16)
        upper = np.zeros(labels.shape, dtype=bool)
        lower = np.zeros(labels.shape, dtype=bool)
        upper[3:11, 5:55] = True
        lower[19:27, 5:55] = True
        labels[5:9, 20:28] = 1
        labels[21:25, 32:40] = 2

        filled = automatic._fill_interlabel_trench_gaps(
            labels,
            (upper, lower),
        )

        np.testing.assert_array_equal(filled, labels)


if __name__ == "__main__":
    unittest.main()
