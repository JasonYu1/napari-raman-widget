import os
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QDialog

from napari_raman_widget.scan_preview import (
    ScanPreviewDialog,
    build_axial_preview,
    build_grid_preview,
)


class ScanPreviewPlanTests(unittest.TestCase):
    def grid(self, **overrides):
        values = dict(
            shape_yx=[[10, 100], [10, 200], [30, 200], [30, 100]],
            points_per_axis=3,
            z_offsets_um=[-2, 0, 2],
            exposure_ms=100,
            output_path="preview-grid.zarr",
            extra_channels=[("GFP", 50)],
            z_offset_um=4,
            layer_name="Scan rectangle",
        )
        values.update(overrides)
        return build_grid_preview(**values)

    def axial(self, **overrides):
        values = dict(
            point_yx=[10, 20], repeats=5, search_range_um=10,
            search_points=20, exposure_ms=1000,
            output_path="reference/preview-axial.zarr", layer_name="Cells",
        )
        values.update(overrides)
        return build_axial_preview(**values)

    def test_grid_geometry_order_and_exposure_units_match_acquisition(self):
        plan = self.grid()
        rows, columns = np.meshgrid([10, 20, 30], [100, 150, 200])
        np.testing.assert_array_equal(
            plan.grid_points_yx, np.column_stack((rows.ravel(), columns.ravel()))
        )
        self.assertEqual(plan.spectrum_count, 27)
        self.assertEqual(plan.repeats, 1)
        self.assertEqual(plan.brightfield_frames, 5)
        self.assertAlmostEqual(plan.minimum_duration_seconds, 2.8)
        self.assertEqual(plan.output_path, str(Path("preview-grid.zarr").resolve()))
        self.assertIn("X 100 to 200; Y 10 to 30", plan.summary_text)
        self.assertIn("initial Z minus 4 µm", plan.summary_text)
        self.assertIn("exposure only", plan.summary_text)
        self.assertIn("actual duration will be longer", plan.summary_text)

    def test_axial_is_one_measured_sweep_with_repeats_not_fine_interpolation(self):
        plan = self.axial()
        np.testing.assert_allclose(plan.z_offsets_um, np.linspace(-10, 10, 20))
        self.assertEqual(plan.spectrum_count, 100)
        self.assertEqual(plan.minimum_duration_seconds, 100)
        self.assertEqual(plan.brightfield_frames, 0)
        self.assertIn("X 20, Y 10", plan.summary_text)

    def test_plan_copies_inputs_and_is_frozen(self):
        shape = np.array([[10, 100], [30, 200]], dtype=float)
        z_offsets = np.array([0.0])
        channels = [["GFP", 50]]
        plan = self.grid(shape_yx=shape, z_offsets_um=z_offsets, extra_channels=channels)
        shape[:] = 0
        z_offsets[:] = 100
        channels[0][1] = 1000
        self.assertEqual(plan.points_yx[0], (10.0, 100.0))
        self.assertEqual(plan.z_offsets_um, (0.0,))
        self.assertEqual(plan.extra_channels, (("GFP", 50.0),))
        with self.assertRaises(FrozenInstanceError):
            plan.repeats = 3

    def test_invalid_counts_and_nonfinite_settings_are_rejected(self):
        for overrides in (
            {"points_per_axis": 10**9}, {"points_per_axis": 2.5},
            {"exposure_ms": 0}, {"exposure_ms": float("nan")},
            {"z_offsets_um": []}, {"z_offsets_um": [float("inf")]},
            {"z_offset_um": float("nan")}, {"shape_yx": [[0, 1, 2], [1, 2, 3]]},
            {"shape_yx": [[0, 1], [float("nan"), 2]]}, {"output_path": ""},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.grid(**overrides)
        for overrides in (
            {"repeats": 0}, {"repeats": 10**9}, {"search_points": 1},
            {"search_points": 10**9}, {"search_range_um": -1},
            {"point_yx": [1, float("nan")]},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.axial(**overrides)

    def test_large_counts_rejected_before_mesh_allocation(self):
        with patch("napari_raman_widget.scan_preview.np.linspace") as linspace:
            with self.assertRaises(ValueError):
                self.grid(points_per_axis=1000000)
            linspace.assert_not_called()

    def test_builders_do_not_write_output(self):
        with patch.object(Path, "mkdir") as mkdir, patch.object(Path, "open") as open_file:
            self.grid()
            self.axial()
        mkdir.assert_not_called()
        open_file.assert_not_called()


class ScanPreviewDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_cancel_is_default_and_only_explicit_start_accepts(self):
        plan = ScanPreviewPlanTests().grid()
        dialog = ScanPreviewDialog(plan)
        try:
            self.assertTrue(dialog.cancel_button.isDefault())
            self.assertFalse(dialog.start_button.isDefault())
            self.assertFalse(dialog.start_button.autoDefault())
            self.assertFalse(dialog.plot_panel.white_background_check.isChecked())
            dialog.cancel_button.click()
            self.assertEqual(dialog.result(), QDialog.Rejected)
            dialog.start_button.click()
            self.assertEqual(dialog.result(), QDialog.Accepted)
        finally:
            dialog.close()

    def test_escape_rejects_axial_preview_and_does_not_mutate_plan(self):
        plan = ScanPreviewPlanTests().axial()
        dialog = ScanPreviewDialog(plan)
        try:
            original = plan.summary_text
            dialog.show()
            self.app.processEvents()
            QTest.keyClick(dialog, Qt.Key_Escape)
            self.assertEqual(dialog.result(), QDialog.Rejected)
            self.assertEqual(plan.summary_text, original)
            self.assertEqual(len(dialog.plot_panel.figure.axes), 1)
        finally:
            dialog.close()

    def test_return_keeps_cancel_as_safe_default(self):
        dialog = ScanPreviewDialog(ScanPreviewPlanTests().grid())
        try:
            accepted = []
            dialog.accepted.connect(lambda: accepted.append(True))
            dialog.show()
            self.app.processEvents()
            QTest.keyClick(dialog, Qt.Key_Return)
            self.assertEqual(dialog.result(), QDialog.Rejected)
            self.assertFalse(accepted)
        finally:
            dialog.close()

    def test_large_grid_display_is_sampled_without_changing_acquisition_count(self):
        plan = ScanPreviewPlanTests().grid(points_per_axis=100)
        dialog = ScanPreviewDialog(plan)
        try:
            xy_axis = dialog.plot_panel.figure.axes[0]
            self.assertEqual(len(xy_axis.collections[0].get_offsets()), 2500)
            self.assertIn("sampled", xy_axis.get_title())
            self.assertEqual(len(plan.points_yx), 10000)
            self.assertEqual(plan.spectrum_count, 30000)
        finally:
            dialog.close()


if __name__ == "__main__":
    unittest.main()
