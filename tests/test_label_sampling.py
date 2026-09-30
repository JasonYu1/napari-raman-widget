"""All-label mask sampling, immutable previews, and camera alignment."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from qtpy.QtWidgets import QApplication

from napari_raman_widget.label_sampling import make_label_roi, sample_label_plan
from napari_raman_widget.scan_preview import ScanPreviewDialog, build_grid_preview
from napari_raman_widget.spatial_mapping import snapshot_scan_labels


def labels_fixture():
    labels = np.zeros((20, 24), dtype=np.int32)
    labels[1:8, 2:9] = 3
    labels[3:5, 4:6] = 0  # A background hole, not another ROI.
    labels[12:18, 14:22] = 11
    return labels


class LabelSamplingTests(unittest.TestCase):
    def test_spacing_matches_independent_lattice_including_both_ids_not_holes(self):
        labels = labels_fixture()
        roi = make_label_roi(labels)
        points, pitch = sample_label_plan(roi, mode="spacing", spacing_px=2)
        expected = np.array([[y, x] for x in range(2, 22, 2) for y in range(1, 18, 2)
                             if labels[y, x] != 0])
        np.testing.assert_array_equal(points, expected)
        self.assertEqual(pitch, 2)
        self.assertEqual(set(roi.labels_at(points)), {3, 11})
        self.assertEqual(roi.label_values, (3, 11))
        self.assertFalse(np.any((points[:, 0] >= 3) & (points[:, 0] < 5)
                               & (points[:, 1] >= 4) & (points[:, 1] < 6)))

    def test_count_mode_is_complete_regular_grid_not_one_point_per_label(self):
        roi = make_label_roi(labels_fixture())
        points, pitch = sample_label_plan(roi, total_points=40)
        explicit, same_pitch = sample_label_plan(roi, mode="spacing", spacing_px=pitch)
        np.testing.assert_array_equal(points, explicit)
        self.assertEqual(pitch, same_pitch)
        self.assertGreater(len(points), len(roi.label_values))
        indices = (points - [1, 2]) / pitch
        np.testing.assert_allclose(indices, np.rint(indices), atol=1e-12)
        self.assertTrue(np.all(roi.labels_at(points) != 0))
        self.assertLess(abs(len(points) - 40), 10)

    def test_square_count_target_400_uses_uniform_20_by_20_grid(self):
        roi = make_label_roi(np.ones((20, 20), dtype=np.uint8))
        points, pitch = sample_label_plan(roi, total_points=400)
        self.assertEqual(len(points), 400)
        self.assertEqual(pitch, 1)
        expected = np.array([[y, x] for x in range(20) for y in range(20)])
        np.testing.assert_array_equal(points, expected)

    def test_fractional_spacing_uses_nearest_label_pixel(self):
        labels = labels_fixture()
        roi = make_label_roi(labels)
        points, pitch = sample_label_plan(roi, mode="spacing", spacing_px=.7)
        candidates = np.array([[y, x] for x in 2 + np.arange(28) * .7
                               for y in 1 + np.arange(23) * .7])
        indices = np.floor(candidates + .5).astype(int)
        expected = candidates[labels[indices[:, 0], indices[:, 1]] != 0]
        np.testing.assert_allclose(points, expected, atol=1e-12)
        self.assertEqual(pitch, .7)

    def test_snapshot_is_immutable_and_original_mask_is_unchanged(self):
        labels = labels_fixture()
        original = labels.copy()
        roi = make_label_roi(labels)
        sample_label_plan(roi)
        np.testing.assert_array_equal(labels, original)
        labels[:] = 0
        np.testing.assert_array_equal(roi.labels, original)
        with self.assertRaises(ValueError):
            roi.labels.setflags(write=True)

    def test_negative_and_large_unsigned_nonzero_ids_are_preserved(self):
        for dtype, value in ((np.int64, -7), (np.uint64, 2**63 + 29)):
            with self.subTest(dtype=dtype):
                data = np.full((3, 3), value, dtype=dtype)
                roi = make_label_roi(data)
                points, _ = sample_label_plan(roi, mode="spacing", spacing_px=1)
                self.assertEqual(roi.labels.dtype, dtype)
                self.assertEqual(roi.label_values, (value,))
                np.testing.assert_array_equal(roi.labels_at(points), np.full(9, value, dtype=dtype))

    def test_empty_float_and_3d_labels_are_rejected(self):
        for data in (np.zeros((4, 4), int), np.ones((2, 3, 4), int), np.ones((4, 4), float)):
            with self.subTest(shape=data.shape), self.assertRaises(ValueError):
                make_label_roi(data)

    def test_oversized_lazy_data_is_rejected_before_loading(self):
        with patch("napari_raman_widget.label_sampling.np.asarray") as materialize:
            with self.assertRaisesRegex(ValueError, "16 million"):
                make_label_roi(SimpleNamespace(shape=(100_000, 100_000)))
            materialize.assert_not_called()

    def test_invalid_or_too_dense_spacing_and_point_limits_are_rejected(self):
        roi = make_label_roi(np.ones((501, 501), dtype=np.uint8))
        for spacing in (0, -1, float("nan"), float("inf"), 1e-300, 1):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                sample_label_plan(roi, mode="spacing", spacing_px=spacing)
        for count in (1, 250001, 2.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                sample_label_plan(roi, total_points=count)

    def test_too_sparse_mask_cannot_produce_fake_or_duplicate_points(self):
        roi = make_label_roi(np.ones((2, 2), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "Fewer than two"):
            sample_label_plan(roi, mode="spacing", spacing_px=10)
        roi = make_label_roi(np.ones((1, 1), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "Cannot find"):
            sample_label_plan(roi, total_points=10)


class LabelsAlignmentTests(unittest.TestCase):
    def layer(self, *, matrix=None, shift=(0, 0), data=None):
        matrix = np.eye(2) if matrix is None else np.asarray(matrix)
        shift = np.asarray(shift)
        return SimpleNamespace(
            name="Segmentation", ndim=2, data=labels_fixture() if data is None else data,
            selected_label=3, show_selected_label=True,
            data_to_world=lambda p: matrix @ p + shift,
            world_to_data=lambda p: np.linalg.solve(matrix, p - shift),
        )

    def test_selected_id_and_selected_only_display_do_not_filter_data(self):
        roi = snapshot_scan_labels(self.layer(), width=24, height=20)
        self.assertEqual(roi.label_values, (3, 11))

    def test_matching_rotated_scaled_and_sheared_image_labels_are_aligned(self):
        matrix = [[2, .4], [.3, 1.5]]
        labels = self.layer(matrix=matrix, shift=[100, 200])
        image = self.layer(matrix=matrix, shift=[100, 200])
        image.name = "Camera"
        roi = snapshot_scan_labels(labels, image_layer=image, width=24, height=20)
        np.testing.assert_allclose(roi.image_to_world_yx, [[2, .4, 100], [.3, 1.5, 200], [0, 0, 1]])
        self.assertEqual(roi.image_layer_name, "Camera")
        np.testing.assert_array_equal(roi.labels, labels.data)

    def test_misaligned_labels_or_transformed_labels_without_image_are_rejected(self):
        labels = self.layer(shift=[1, 0])
        for image in (None, self.layer()):
            with self.subTest(image=image), self.assertRaisesRegex(ValueError, "Align"):
                snapshot_scan_labels(labels, image_layer=image, width=24, height=20)

    def test_wrong_dimensions_and_multiscale_are_rejected(self):
        for changes in ({"ndim": 3}, {"multiscale": True}, {"data": np.ones((10, 12), int)}):
            layer = self.layer()
            layer.__dict__.update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "2D Labels"):
                snapshot_scan_labels(layer, width=24, height=20)


class LabelPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def plan(self, data=None, spacing=2):
        return build_grid_preview(
            None, label_roi=make_label_roi(labels_fixture() if data is None else data),
            sampling_mode="spacing", spacing_px=spacing, z_offsets_um=[-1, 1],
            exposure_ms=100, output_path="label-test.zarr", layer_name="Labels",
        )

    def test_preview_preserves_label_ids_and_counts_actual_spectra(self):
        plan = self.plan()
        self.assertEqual(set(plan.point_label_ids), {3, 11})
        self.assertEqual(len(plan.point_label_ids), len(plan.points_yx))
        self.assertEqual(plan.spectrum_count, 2 * len(plan.points_yx))
        self.assertIn("all non-zero labels", plan.summary_text)
        self.assertIn("Label IDs sampled: 2 / 2", plan.summary_text)

    def test_preview_warns_when_coarse_grid_misses_small_labels(self):
        data = np.zeros((40, 40), int)
        data[:10, :10] = 1
        data[39, 39] = 2
        plan = self.plan(data, spacing=5)
        self.assertIn("Label IDs sampled: 1 / 2", plan.summary_text)
        self.assertIn("Some small labels have no grid point", plan.summary_text)

    def test_preview_draws_the_mask_not_a_filled_bounding_rectangle(self):
        plan = self.plan()
        dialog = ScanPreviewDialog(plan)
        try:
            axis = dialog.plot_panel.figure.axes[0]
            self.assertEqual(len(axis.images), 1)
            self.assertEqual(len(axis.lines), 0)
            self.assertEqual(len(axis.collections[0].get_offsets()), len(plan.points_yx))
            shown = axis.images[0].get_array()
            self.assertTrue(np.ma.getmaskarray(shown)[2, 2])  # Hole at full-image (3,4).
            self.assertFalse(np.ma.getmaskarray(shown)[0, 0])
            self.assertTrue(axis.yaxis_inverted())
        finally:
            dialog.close()


if __name__ == "__main__":
    unittest.main()
