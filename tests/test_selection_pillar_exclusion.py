"""Hard exclusion guarantees for pillar-aware target selection."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.ndimage import binary_dilation

from napari_raman_widget.selection.masks import (
    find_clear_center_point,
    get_n_most_centered_coms,
)


class PillarExclusionSelectionTests(unittest.TestCase):
    def test_small_normal_cell_keeps_old_com_when_exclusion_is_off(self):
        labels = np.zeros((20, 20), dtype=np.uint16)
        labels[8:11, 12:15] = 1

        points = get_n_most_centered_coms(
            labels,
            N=1,
            autofocus_object=None,
        )

        np.testing.assert_allclose(points, [[9.0, 13.0]])

    def test_nonoverlapping_cell_is_unchanged_by_distant_pillar_mask(self):
        labels = np.zeros((20, 20), dtype=np.uint16)
        labels[2:5, 3:6] = 1
        exclusion = np.zeros_like(labels, dtype=bool)
        exclusion[12:18, 12:18] = True

        points = get_n_most_centered_coms(
            labels,
            N=1,
            autofocus_object=None,
            exclusion_mask=exclusion,
        )

        np.testing.assert_allclose(points, [[3.0, 4.0]])

    def test_autofocus_never_uses_suppressed_pillar_as_background(self):
        labels = np.zeros((31, 31), dtype=np.uint16)
        exclusion = np.zeros_like(labels, dtype=bool)
        exclusion[10:21, 10:21] = True

        point = find_clear_center_point(
            labels,
            threshold=1,
            center=(15, 15),
            exclusion_mask=exclusion,
        )

        y, x = point.astype(int)
        self.assertFalse(exclusion[y, x])

    def test_cell_target_is_inside_label_and_outside_pillar_hole(self):
        labels = np.zeros((31, 31), dtype=np.uint16)
        labels[5:26, 5:26] = 1
        labels[10:21, 10:21] = 0
        exclusion = np.zeros_like(labels, dtype=bool)
        exclusion[9:22, 9:22] = True

        points = get_n_most_centered_coms(
            labels,
            N=1,
            center=(15, 15),
            radius=np.inf,
            autofocus_object=None,
            exclusion_mask=exclusion,
        )

        self.assertEqual(points.shape, (1, 2))
        y, x = points[0].astype(int)
        self.assertEqual(labels[y, x], 1)
        self.assertFalse(exclusion[y, x])

    def test_pillar_only_label_is_not_selected(self):
        labels = np.zeros((32, 32), dtype=np.uint16)
        labels[4:28, 14:18] = 1
        exclusion = labels == 1

        points = get_n_most_centered_coms(
            labels,
            N=1,
            autofocus_object=None,
            exclusion_mask=exclusion,
        )

        self.assertEqual(points.shape, (0, 2))

    def test_cell_lobe_touching_pillar_survives_outside_margin(self):
        yy, xx = np.ogrid[:72, :72]
        pillar = (
            (yy >= 8)
            & (yy < 64)
            & (xx >= 25)
            & (xx < 32)
        )
        cell = (yy - 36) ** 2 + (xx - 43) ** 2 <= 12**2
        labels = np.zeros((72, 72), dtype=np.uint16)
        labels[pillar | cell] = 1
        exclusion = pillar.copy()
        exclusion_margin = 4
        expanded_exclusion = binary_dilation(
            exclusion,
            iterations=exclusion_margin,
        )

        # This merged Cellpose label crosses the former whole-label rejection
        # threshold, while retaining a substantial round cell body away from
        # the pillar safety margin.
        overlap_fraction = np.count_nonzero(
            (labels == 1) & expanded_exclusion
        ) / np.count_nonzero(labels == 1)
        self.assertGreater(overlap_fraction, 0.25)
        self.assertTrue(np.any(cell & ~expanded_exclusion))

        points = get_n_most_centered_coms(
            labels,
            N=1,
            center=(36, 43),
            autofocus_object=None,
            exclusion_mask=exclusion,
            exclusion_margin=exclusion_margin,
        )

        self.assertEqual(points.shape, (1, 2))
        y, x = np.rint(points[0]).astype(int)
        self.assertTrue(cell[y, x])
        self.assertEqual(labels[y, x], 1)
        self.assertFalse(expanded_exclusion[y, x])

    def test_thin_residual_beside_pillar_is_not_selected(self):
        labels = np.zeros((40, 40), dtype=np.uint16)
        labels[5:35, 16:21] = 1
        exclusion = np.zeros_like(labels, dtype=bool)
        exclusion[5:35, 16:20] = True

        # Removing the pillar leaves a one-pixel-wide, 30-pixel-long sliver:
        # enough area to fool an area-only check, but no viable cell interior.
        residual = (labels == 1) & ~exclusion
        self.assertGreaterEqual(np.count_nonzero(residual), 16)
        self.assertEqual(np.count_nonzero(residual, axis=1).max(), 1)

        points = get_n_most_centered_coms(
            labels,
            N=1,
            autofocus_object=None,
            exclusion_mask=exclusion,
        )

        self.assertEqual(points.shape, (0, 2))

    def test_exclusion_shape_must_match_labels(self):
        with self.assertRaisesRegex(ValueError, "match label_mask"):
            get_n_most_centered_coms(
                np.zeros((10, 10), dtype=np.uint16),
                exclusion_mask=np.zeros((5, 5), dtype=bool),
            )


if __name__ == "__main__":
    unittest.main()
