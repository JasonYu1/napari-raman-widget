"""Post-segmentation pillar exclusion checks for automated selection."""

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
        pass

    def waitForSystem(self):
        pass

    def snapImage(self):
        pass

    def getImage(self):
        return self.image


class _ShapeFilter:
    def __init__(self, filtered_labels, removed_mask):
        self.filtered_labels = filtered_labels
        self.removed_mask = removed_mask
        self.inputs = []
        self.config = SimpleNamespace(
            pillar_length=78.0,
            pillar_width=56.0,
            minimum_rectangle_fit=0.75,
        )

    def filter(self, labels, *, image_shape):
        self.inputs.append((labels, image_shape))
        return SimpleNamespace(
            filtered_labels=self.filtered_labels,
            removed_mask=self.removed_mask,
            removed_labels=(1,),
            measurements=(
                SimpleNamespace(
                    label=1,
                    rectangle_fit_score=0.9,
                    removed=True,
                ),
            ),
        )


class AutomaticSelectionPillarTests(unittest.TestCase):
    def _run_one_position(self, *, upsample: int):
        raw = np.arange(36, dtype=np.uint16).reshape(6, 6)
        original = raw.copy()
        sequence = SimpleNamespace(
            stage_positions=[SimpleNamespace(x=1.0, y=2.0)]
        )
        captured: list[np.ndarray] = []
        cellpose_labels = np.zeros_like(raw, dtype=np.uint16)
        cellpose_labels[0:2, 0:2] = 1
        cellpose_labels[3:6, 3:6] = 2
        filtered_labels = cellpose_labels.copy()
        filtered_labels[filtered_labels == 1] = 0
        pillar_mask = cellpose_labels == 1
        shape_filter = _ShapeFilter(filtered_labels, pillar_mask)
        viewer = MagicMock()
        selector = MagicMock(
            return_value=np.array([[2.0, 2.0]])
        )

        def segment_direct(image, **kwargs):
            captured.append(image)
            return cellpose_labels

        def segment_upsampled(image, **kwargs):
            captured.append(image)
            return cellpose_labels

        patches = (
            patch.object(automatic, "get_seq_from_napari", return_value=sequence),
            patch.object(
                automatic,
                "make_pillar_shape_filter",
                return_value=shape_filter,
            ),
            patch.object(automatic, "segment_single_img", side_effect=segment_direct),
            patch.object(
                automatic,
                "get_n_most_centered_coms",
                new=selector,
            ),
            patch.object(
                automatic,
                "_finish_original_position_selection",
                return_value=("sources", "autofocus", "sequence"),
            ),
            patch.object(automatic, "tqdm", side_effect=lambda values, **_: values),
            patch(
                "napari_raman_widget.demo.cellpose."
                "segment_upsampled_demo_region",
                side_effect=segment_upsampled,
            ),
        )

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[
            5
        ], patches[6]:
            result = automatic.automated_point_selections(
                _Core(raw),
                viewer=viewer,
                main_window=SimpleNamespace(),
                point_transformer=SimpleNamespace(),
                N=1,
                center=(3.0, 3.0),
                radius=3.0,
                autofocus_object=None,
                stage_settle_time=0,
                image_settle_time=0,
                cellpose_upsample=upsample,
                suppress_pillars=True,
                show_masks=True,
            )

        self.assertEqual(result, ("sources", "autofocus", "sequence"))
        self.assertIs(captured[0], raw)
        self.assertEqual(captured[0].shape, raw.shape)
        self.assertIs(shape_filter.inputs[0][0], cellpose_labels)
        self.assertEqual(shape_filter.inputs[0][1], raw.shape)
        self.assertIs(selector.call_args.args[0], filtered_labels)
        np.testing.assert_array_equal(
            selector.call_args.kwargs["exclusion_mask"],
            pillar_mask,
        )
        self.assertEqual(
            selector.call_args.kwargs["exclusion_margin"],
            0,
        )
        np.testing.assert_array_equal(raw, original)
        viewer.add_labels.assert_called_once_with(
            filtered_labels,
            name="Cellpose mask p0",
        )

    def test_direct_cellpose_receives_exact_raw_frame(self):
        self._run_one_position(upsample=1)

    def test_demo_upsampling_receives_exact_raw_frame_before_crop(self):
        self._run_one_position(upsample=3)


if __name__ == "__main__":
    unittest.main()
