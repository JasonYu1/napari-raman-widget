"""Display-only maps preserve irregular-grid point indexing and raw spectra."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import xarray as xr
from matplotlib import colormaps
from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtWidgets import QApplication

from napari_raman_widget.plot_windows import GridScanPlotWindow


class BandMapControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.windows = []
        self.points = np.array([[1, 1], [2, 4], [4, 2]], dtype=float)
        self.specs = np.array([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [3, 6, 9, 12, 15]], dtype=float)

    def tearDown(self):
        for window in self.windows:
            window.close()
            window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def window(self, *, z=False, specs=None, calibration=None, image=True):
        specs = self.specs.copy() if specs is None else specs
        variables = {"grid_pos": (("point", "coord"), self.points)}
        if image:
            variables["BF"] = (("Y", "X"), np.zeros((6, 6)))
        if z:
            variables.update({
                "specs": (("z", "point", "pixel"), np.stack([specs, specs * 10])),
                "BF_z": (("z", "Y", "X"), np.zeros((2, 6, 6))),
                "z_range": ("z", [-1, 1]),
            })
        else:
            variables["specs"] = (("point", "pixel"), specs)
        window = GridScanPlotWindow(xr.Dataset(variables), spectral_calibration=calibration)
        self.windows.append(window)
        return window

    def apply_area(self, window, lo=1, hi=3):
        control = window.band_map_controls
        control.mode_combo.setCurrentIndex(1)
        control.a_low.setValue(lo)
        control.a_high.setValue(hi)
        control.apply_button.click()
        return control

    def test_default_is_points_only_with_pixel_ranges_and_no_colorbar(self):
        window = self.window()
        control = window.band_map_controls
        self.assertEqual(control.mode_combo.currentData(), "points")
        self.assertFalse(control.toggle.isChecked())
        self.assertTrue(control.body.isHidden())
        self.assertEqual(control._unit, "pixel")
        self.assertFalse(control.apply_button.isEnabled())
        self.assertIsNone(control.values)
        self.assertIsNone(control.colorbar)
        self.assertEqual(control.colormap_combo.currentText(), "viridis")
        self.assertFalse(control.reverse_check.isChecked())
        self.assertFalse(window.show_wavenumber_check.isEnabled())
        np.testing.assert_array_equal(window._scat.get_offsets(), self.points[:, ::-1])

    def test_area_colors_real_points_and_picking_keeps_exact_spectrum(self):
        window = self.window()
        control = self.apply_area(window)
        np.testing.assert_allclose(control.values, [6, 12, 18])
        np.testing.assert_allclose(window._scat.get_array(), [6, 12, 18])
        self.assertEqual(len(window._scat.get_edgecolors()), 0)
        self.assertIsNotNone(control.colorbar)
        self.assertEqual(len(control._spans), 1)
        self.assertIn("pixel", control.colorbar.ax.get_ylabel())
        np.testing.assert_array_equal(window._scat.get_offsets(), self.points[:, ::-1])
        window._on_pick(SimpleNamespace(artist=window._scat, ind=[2]))
        np.testing.assert_array_equal(window._spec_line.get_ydata(), self.specs[2])
        np.testing.assert_array_equal(window._hl.get_offsets(), [[2, 4]])
        self.assertIn("Selected point 2: 18", control.status.text())
        np.testing.assert_array_equal(window.ds.specs.values, self.specs)

    def test_palette_choices_are_registered_with_previews(self):
        control = self.window().band_map_controls
        names = []
        for index in range(control.colormap_combo.count()):
            name = control.colormap_combo.itemText(index)
            names.append(name)
            self.assertIn(name, colormaps)
            self.assertFalse(name.endswith("_r"))
            self.assertFalse(control.colormap_combo.itemIcon(index).isNull())
        self.assertEqual(len(names), len(set(names)))
        for name in ("viridis", "plasma", "cividis", "coolwarm", "gray"):
            self.assertIn(name, names)

    def test_colormap_changes_only_colors_and_updates_existing_colorbar(self):
        window = self.window()
        control = self.apply_area(window)
        window._on_pick(SimpleNamespace(artist=window._scat, ind=[2]))
        values, norm, colorbar = control.values, window._scat.norm, control.colorbar
        spans, status = tuple(control._spans), control.status.text()
        limits = window._scat.get_clim()
        axes = tuple(window.fig.axes)
        with patch("napari_raman_widget.band_map_controls.band_map_values") as integrate:
            control.colormap_combo.setCurrentText("plasma")
            self.assertEqual(window._scat.get_cmap().name, "plasma")
            self.assertEqual(colorbar.cmap.name, "plasma")
            control.reverse_check.setChecked(True)
            self.assertEqual(colorbar.cmap.name, "plasma_r")
            np.testing.assert_allclose(window._scat.get_cmap()(0.0), colormaps["plasma"](1.0))
            np.testing.assert_allclose(window._scat.get_cmap()(1.0), colormaps["plasma"](0.0))
            integrate.assert_not_called()
        self.assertIs(control.values, values)
        self.assertIs(window._scat.norm, norm)
        self.assertIs(control.colorbar, colorbar)
        self.assertEqual(window._scat.get_clim(), limits)
        self.assertEqual(tuple(window.fig.axes), axes)
        self.assertEqual(tuple(control._spans), spans)
        self.assertEqual(control.status.text(), status)
        self.assertTrue(control._applied)
        self.assertEqual(window._sel, 2)
        np.testing.assert_array_equal(window.ds.specs.values, self.specs)
        window.canvas.draw()

    def test_palette_choice_does_not_apply_pending_bands_or_points_only(self):
        window = self.window()
        control = window.band_map_controls
        with patch("napari_raman_widget.band_map_controls.band_map_values") as integrate:
            control.colormap_combo.setCurrentText("cividis")
            control.reverse_check.setChecked(True)
            self.assertIsNone(control.values)
            self.assertIsNone(window._scat.get_array())
            self.assertFalse(control._applied)
            integrate.assert_not_called()
        self.apply_area(window)
        self.assertEqual(window._scat.get_cmap().name, "cividis_r")
        control.a_high.setValue(4)
        with patch("napari_raman_widget.band_map_controls.band_map_values") as integrate:
            control.colormap_combo.setCurrentText("magma")
            self.assertIsNone(control.values)
            self.assertIsNone(control.colorbar)
            self.assertIsNone(window._scat.get_array())
            self.assertIn("click Apply", control.status.text())
            self.assertFalse(control._applied)
            integrate.assert_not_called()
        control.apply_button.click()
        self.assertEqual(window._scat.get_cmap().name, "magma_r")
        np.testing.assert_allclose(control.values, [10.5, 21, 31.5])

    def test_palette_persists_across_z_mode_and_axis_changes(self):
        calibration = SimpleNamespace(transform=lambda x: 1000 + 2 * np.asarray(x))
        window = self.window(z=True, calibration=calibration)
        control = self.apply_area(window)
        control.colormap_combo.setCurrentText("coolwarm")
        control.reverse_check.setChecked(True)
        window._z_slider.setValue(1)
        self.assertEqual(window._scat.get_cmap().name, "coolwarm_r")
        np.testing.assert_allclose(control.values, [60, 120, 180])
        control.mode_combo.setCurrentIndex(2)
        control.apply_button.click()
        self.assertEqual(window._scat.get_cmap().name, "coolwarm_r")
        window.show_wavenumber_check.setChecked(True)
        control.apply_button.click()
        self.assertEqual(window._scat.get_cmap().name, "coolwarm_r")
        self.assertEqual(control.colormap_combo.currentText(), "coolwarm")
        self.assertTrue(control.reverse_check.isChecked())

    def test_ratio_masks_zero_denominator_and_nonfinite_needed_samples(self):
        specs = np.array([[1, 1, 0, 2, 2], [1, 1, 0, 0, 0], [np.nan, 1, 0, 1, 1]])
        window = self.window(specs=specs)
        control = window.band_map_controls
        control.mode_combo.setCurrentIndex(2)
        control.a_low.setValue(0)
        control.a_high.setValue(1)
        control.b_low.setValue(3)
        control.b_high.setValue(4)
        control.apply_button.click()
        np.testing.assert_allclose(control.values, [0.5, np.nan, np.nan], equal_nan=True)
        self.assertEqual(len(control._spans), 2)
        self.assertIn("2 gray/invalid", control.status.text())
        window.canvas.draw()
        colors = window._scat.get_facecolors()
        np.testing.assert_allclose(colors[1, :3], [140 / 255] * 3)
        np.testing.assert_allclose(colors[2, :3], [140 / 255] * 3)
        # Invalid map values do not remove measured points from the overlay.
        self.assertEqual(len(window._scat.get_offsets()), 3)
        window._on_pick(SimpleNamespace(artist=window._scat, ind=[1]))
        np.testing.assert_array_equal(window._spec_line.get_ydata(), specs[1])
        self.assertIn("Selected point 1: invalid", control.status.text())
        registered_bad = colormaps["plasma"].get_bad().copy()
        control.colormap_combo.setCurrentText("plasma")
        control.reverse_check.setChecked(True)
        window.canvas.draw()
        colors = window._scat.get_facecolors()
        np.testing.assert_allclose(colors[1:, :3], np.full((2, 3), 140 / 255))
        np.testing.assert_array_equal(colormaps["plasma"].get_bad(), registered_bad)

    def test_settings_change_clears_old_map_until_apply(self):
        window = self.window()
        control = self.apply_area(window)
        control.a_high.setValue(4)
        self.assertIsNone(control.values)
        self.assertIsNone(control.colorbar)
        self.assertFalse(control._applied)
        self.assertIn("click Apply", control.status.text())
        control.apply_button.click()
        np.testing.assert_allclose(control.values, [10.5, 21, 31.5])
        control.a_low.setValue(4)
        control.apply_button.click()
        self.assertIsNone(control.values)
        self.assertIn("Cannot apply map", control.status.text())

    def test_z_plane_recomputes_map_without_changing_selected_point(self):
        window = self.window(z=True)
        control = self.apply_area(window)
        window._on_pick(SimpleNamespace(artist=window._scat, ind=[1]))
        original_axes = len(window.fig.axes)
        window._z_slider.setValue(1)
        np.testing.assert_allclose(control.values, [60, 120, 180])
        self.assertEqual(window._sel, 1)
        np.testing.assert_array_equal(window._spec_line.get_ydata(), self.specs[1] * 10)
        self.assertEqual(len(window.fig.axes), original_axes)
        window.canvas.draw()

    def test_calibrated_ranges_follow_pixels_and_unit_switch_clears_map(self):
        calibration = SimpleNamespace(transform=lambda x: 1000 + 2 * np.asarray(x))
        window = self.window(calibration=calibration)
        control = self.apply_area(window)
        window.show_wavenumber_check.setChecked(True)
        self.assertIsNone(control.values)
        self.assertEqual(control._unit, "cm⁻¹")
        self.assertEqual(control.a_low.value(), 1002)
        self.assertEqual(control.a_high.value(), 1006)
        self.assertGreater(window._ax_spec.get_xlim()[0], 900)
        control.apply_button.click()
        np.testing.assert_allclose(control.values, [12, 24, 36])
        self.assertIn("cm⁻¹", control.colorbar.ax.get_ylabel())
        window.show_wavenumber_check.setChecked(False)
        self.assertEqual(control.a_low.value(), 1)
        self.assertEqual(control.a_high.value(), 3)

    def test_decreasing_calibration_reorders_bounds_without_negative_area(self):
        calibration = SimpleNamespace(transform=lambda x: 1000 - 2 * np.asarray(x))
        window = self.window(calibration=calibration)
        control = self.apply_area(window)
        window.show_wavenumber_check.setChecked(True)
        self.assertEqual(control.a_low.value(), 994)
        self.assertEqual(control.a_high.value(), 998)
        control.apply_button.click()
        np.testing.assert_allclose(control.values, [12, 24, 36])

    def test_nonmonotonic_calibration_disables_map_and_pixel_mode_recovers(self):
        calibration = SimpleNamespace(transform=lambda x: (np.asarray(x) - 2) ** 2)
        window = self.window(calibration=calibration)
        control = self.apply_area(window)
        window.show_wavenumber_check.setChecked(True)
        self.assertFalse(control.apply_button.isEnabled())
        self.assertIsNone(control.values)
        window.show_wavenumber_check.setChecked(False)
        self.assertTrue(control.apply_button.isEnabled())
        control.apply_button.click()
        self.assertIsNotNone(control.values)

    def test_processing_affects_spectrum_only_not_raw_map_values(self):
        window = self.window()
        control = self.apply_area(window)
        window.baseline_subtraction_check.setChecked(True)
        np.testing.assert_allclose(control.values, [6, 12, 18])
        control.apply_button.click()
        np.testing.assert_allclose(control.values, [6, 12, 18])
        np.testing.assert_array_equal(window.ds.specs.values, self.specs)

    def test_all_invalid_ratio_is_gray_without_fabricated_color_scale(self):
        window = self.window(specs=np.zeros((3, 5)))
        control = window.band_map_controls
        control.mode_combo.setCurrentIndex(2)
        control.apply_button.click()
        self.assertTrue(np.isnan(control.values).all())
        self.assertIsNone(control.colorbar)
        self.assertIn("0/3 valid", control.status.text())
        control.colormap_combo.setCurrentText("inferno")
        control.reverse_check.setChecked(True)
        self.assertIsNone(control.colorbar)
        window.canvas.draw()
        np.testing.assert_allclose(window._scat.get_facecolors()[:, :3], np.full((3, 3), 140 / 255))

    def test_repeated_on_off_and_white_background_do_not_accumulate_axes(self):
        window = self.window()
        axes = len(window.fig.axes)
        for _ in range(3):
            control = self.apply_area(window)
            self.assertEqual(len(window.fig.axes), axes + 1)
            window.white_background_check.setChecked(True)
            window.canvas.draw()
            window.white_background_check.setChecked(False)
            control.mode_combo.setCurrentIndex(0)
            self.assertEqual(len(window.fig.axes), axes)
            self.assertEqual(len(control._spans), 0)
            self.assertIsNone(window._scat.get_array())
            window.canvas.draw()

    def test_missing_image_disables_map_without_breaking_spectrum(self):
        window = self.window(image=False)
        control = window.band_map_controls
        control.mode_combo.setCurrentIndex(1)
        self.assertFalse(control.apply_button.isEnabled())
        self.assertIn("Map unavailable", control.status.text())

    def test_colorbar_removal_restores_image_geometry_without_layout_engine(self):
        window = self.window()
        window.fig.set_layout_engine(None)
        ax = window._scat.axes
        original = ax.get_position(original=True).bounds
        anchor = ax.get_anchor()
        for _ in range(3):
            control = self.apply_area(window)
            control.mode_combo.setCurrentIndex(0)
            np.testing.assert_allclose(ax.get_position(original=True).bounds, original)
            self.assertEqual(ax.get_anchor(), anchor)


if __name__ == "__main__":
    unittest.main()
