import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from napari_raman_widget.trench_segmentation import (
    TrenchCellposeSegmenter,
)


class _Delegate:
    def __init__(self):
        self.calls = []

    def segment(self, image, **kwargs):
        self.calls.append((np.asarray(image).copy(), dict(kwargs)))
        scale = int(kwargs["scale"])
        shape = tuple(np.round(np.asarray(image.shape) / scale).astype(int))
        return np.ones(shape, dtype=np.uint16)


def _geometry(valid=True):
    raw_a = np.zeros((20, 40), dtype=bool)
    raw_a[2:13, 2:18] = True
    raw_b = np.zeros_like(raw_a)
    raw_b[2:13, 22:38] = True
    safe_a = np.zeros_like(raw_a)
    safe_a[4:11, 7:16] = True
    safe_b = np.zeros_like(raw_a)
    safe_b[4:11, 25:35] = True
    return SimpleNamespace(
        geometry_valid=valid,
        trench_masks=(raw_a, raw_b) if valid else (),
        safe_masks=(safe_a, safe_b) if valid else (),
    )


class TrenchCellposeSegmenterTests(unittest.TestCase):
    def test_segments_only_central_trench_without_circular_clipping(self):
        delegate = _Delegate()
        segmenter = TrenchCellposeSegmenter(delegate)
        image = np.arange(800, dtype=np.uint16).reshape(20, 40)

        with patch(
            "napari_raman_widget.trench_segmentation."
            "registered_trench_geometry",
            return_value=_geometry(),
        ):
            labels = segmenter.segment(
                image,
                scale=2,
                crop=True,
                circle_center=(6, 28),
                circle_radius=1,
            )

        self.assertEqual(labels.shape, (10, 20))
        self.assertEqual(len(delegate.calls), 1)
        self.assertEqual(set(np.unique(labels)), {0, 1})
        np.testing.assert_array_equal(
            labels != 0, _geometry().safe_masks[1][1::2, 1::2]
        )
        for prepared, kwargs in delegate.calls:
            self.assertEqual(prepared.ndim, 2)
            self.assertIs(kwargs["crop"], False)
            self.assertIsNone(kwargs["circle_center"])
            self.assertTrue(np.isinf(kwargs["circle_radius"]))

    def test_labels_are_clipped_to_cap_aware_safe_mask(self):
        delegate = _Delegate()
        segmenter = TrenchCellposeSegmenter(delegate)

        with patch(
            "napari_raman_widget.trench_segmentation."
            "registered_trench_geometry",
            return_value=_geometry(),
        ):
            # The circle center is in the excluded cap of the first trench.
            labels = segmenter.segment(
                np.ones((20, 40)), scale=1, circle_center=(3, 3)
            )

        expected = _geometry().safe_masks[0]
        np.testing.assert_array_equal(labels != 0, expected)

    def test_center_between_trenches_uses_only_nearest_trench(self):
        delegate = _Delegate()
        segmenter = TrenchCellposeSegmenter(delegate)
        with patch(
            "napari_raman_widget.trench_segmentation."
            "registered_trench_geometry",
            return_value=_geometry(),
        ):
            labels = segmenter.segment(
                np.ones((20, 40)),
                scale=1,
                circle_center=(6, 20.8),
                circle_radius=1000,
            )

        self.assertEqual(len(delegate.calls), 1)
        np.testing.assert_array_equal(labels != 0, _geometry().safe_masks[1])

    def test_input_neutralizes_everything_outside_selected_safe_interior(self):
        delegate = _Delegate()
        segmenter = TrenchCellposeSegmenter(delegate)
        image = np.arange(800, dtype=np.uint16).reshape(20, 40)
        with patch(
            "napari_raman_widget.trench_segmentation."
            "registered_trench_geometry",
            return_value=_geometry(),
        ):
            segmenter.segment(
                image,
                scale=1,
                circle_center=(6, 10),
                circle_radius=1,
                crop_padding=30,
            )

        prepared, _ = delegate.calls[0]
        safe = _geometry().safe_masks[0]
        np.testing.assert_array_equal(prepared[safe], image[safe])
        np.testing.assert_array_equal(
            prepared[~safe], np.full(np.count_nonzero(~safe), np.median(image[safe]))
        )

    def test_missing_circle_center_defaults_to_image_center(self):
        delegate = _Delegate()
        segmenter = TrenchCellposeSegmenter(delegate)
        with patch(
            "napari_raman_widget.trench_segmentation."
            "registered_trench_geometry",
            return_value=_geometry(),
        ):
            labels = segmenter.segment(np.ones((20, 40)), scale=1)

        self.assertEqual(len(delegate.calls), 1)
        np.testing.assert_array_equal(labels != 0, _geometry().safe_masks[0])

    def test_invalid_or_empty_selected_geometry_fails_closed(self):
        empty_safe = _geometry()
        empty_safe.safe_masks[0][:] = False
        empty_raw = _geometry()
        for mask in empty_raw.trench_masks:
            mask[:] = False
        for geometry in (_geometry(valid=False), empty_safe, empty_raw):
            with self.subTest(geometry=geometry):
                delegate = _Delegate()
                segmenter = TrenchCellposeSegmenter(delegate)
                with patch(
                    "napari_raman_widget.trench_segmentation."
                    "registered_trench_geometry",
                    return_value=geometry,
                ):
                    labels = segmenter.segment(
                        np.ones((20, 40)), scale=2, circle_center=(6, 10)
                    )

                self.assertEqual(labels.shape, (10, 20))
                self.assertFalse(np.any(labels))
                self.assertEqual(delegate.calls, [])

    def test_invalid_circle_center_is_rejected(self):
        segmenter = TrenchCellposeSegmenter(_Delegate())
        with patch(
            "napari_raman_widget.trench_segmentation."
            "registered_trench_geometry",
            return_value=_geometry(),
        ):
            for center in ((np.nan, 10), (1, 2, 3)):
                with self.subTest(center=center), self.assertRaises(ValueError):
                    segmenter.segment(np.ones((20, 40)), circle_center=center)


if __name__ == "__main__":
    unittest.main()
