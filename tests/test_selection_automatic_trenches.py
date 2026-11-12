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
                return_value=self.raw_labels,
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
                N=1,
                center=(3.0, 3.0),
                radius=3.0,
                autofocus_object=None,
                stage_settle_time=0,
                image_settle_time=0,
                suppress_pillars=True,
                constrain_trenches=True,
                show_masks=True,
                cell_point_refiner=lambda points: np.zeros_like(points),
            )

        self.assertEqual(result, ("sources", "autofocus", "sequence"))
        self.assertIs(trench_filter.call_args.args[0], self.image)
        np.testing.assert_array_equal(
            trench_filter.call_args.args[1],
            after_pillars,
        )
        self.assertIs(selector.call_args.args[0], safe_labels)
        np.testing.assert_array_equal(
            selector.call_args.kwargs["exclusion_mask"],
            pillar_pixels,
        )
        self.viewer.add_labels.assert_called_once_with(
            safe_labels,
            name="Cellpose mask p0",
        )
        # The demo refiner proposed (0, 0), outside the selected safe label,
        # so the original wall-safe center must reach source construction.
        np.testing.assert_allclose(
            finish.call_args.kwargs["selected_by_position"][0],
            [[3.5, 3.5]],
        )

    def test_low_confidence_geometry_skips_field_without_targets(self):
        selector = MagicMock()
        finish = MagicMock()
        failed = SimpleNamespace(
            filtered_labels=np.zeros_like(self.raw_labels),
            safe_interior_mask=np.zeros_like(self.raw_labels, dtype=bool),
            geometry_valid=False,
            geometry_score=0.2,
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
                return_value=None,
            ),
            patch.object(
                automatic,
                "segment_single_img",
                return_value=self.raw_labels,
            ),
            patch.object(
                automatic,
                "filter_labels_to_trenches",
                return_value=failed,
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


if __name__ == "__main__":
    unittest.main()
