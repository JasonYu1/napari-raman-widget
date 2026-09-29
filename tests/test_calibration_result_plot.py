import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import xarray as xr
from matplotlib.backend_bases import MouseButton, MouseEvent
from qtpy.QtWidgets import QApplication

from napari_raman_widget.plot_windows import CalibrationPlotWindow
from napari_raman_widget.spectral_calibration import (
    PixelToWavenumberCalibration,
)


class CalibrationResultPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def dataset() -> xr.Dataset:
        spectra = np.array(
            [
                [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
                [[10, 20, 30, 40, 50], [12, 22, 32, 42, 52]],
                [[5, 4, 3, 2, 1], [5, 4, 3, 2, 1]],
            ],
            dtype=float,
        )
        return xr.Dataset(
            {
                "imgs": (
                    ("idx", "Y", "X"),
                    np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8),
                ),
                "rel_BF_pos": (
                    ("idx", "rel_BF"),
                    [[1.0, 1.0], [6.0, 2.0], [4.0, 6.0]],
                ),
                "laser_pos": (
                    ("idx", "volt"),
                    [[-0.5, -0.25], [0.0, 0.25], [0.5, 0.75]],
                ),
                "specs": (("idx", "N", "spec_dim"), spectra),
            }
        )

    @staticmethod
    def click_point(window, point_index: int) -> None:
        window.canvas.draw()
        point = window._point_positions[point_index]
        x, y = window.ax_image.transData.transform(point)
        event = MouseEvent(
            "button_press_event",
            window.canvas,
            x,
            y,
            button=MouseButton.LEFT,
        )
        window.canvas.callbacks.process("button_press_event", event)
        window.canvas.draw()

    def test_canvas_click_selects_corresponding_acquired_spectrum(self) -> None:
        ds = self.dataset()
        original = ds["specs"].values.copy()
        window = CalibrationPlotWindow(ds)
        try:
            self.assertEqual(window.selected_index, 0)
            self.assertEqual(window.image_artist.get_array().dtype, np.uint16)
            np.testing.assert_allclose(
                window.spectrum_line.get_ydata(), [1, 2, 3, 4, 5]
            )

            self.click_point(window, 1)

            self.assertEqual(window.selected_index, 1)
            np.testing.assert_allclose(
                window.selected_spectrum, [11, 21, 31, 41, 51]
            )
            np.testing.assert_allclose(
                window.spectrum_line.get_ydata(), [11, 21, 31, 41, 51]
            )
            np.testing.assert_allclose(
                window.selection_artist.get_offsets(), [[6.0, 2.0]]
            )
            self.assertIn("Point 2 of 3", window.selection_label.text())
            self.assertIn("galvo (x=0 V, y=0.25 V)", window.selection_label.text())
            np.testing.assert_array_equal(ds["specs"].values, original)
        finally:
            window.close()

    def test_pixels_are_default_and_toolbar_pan_blocks_selection(self) -> None:
        calibration = PixelToWavenumberCalibration(
            [0.0, 4.0], [100.0, 140.0], degree=1
        )
        window = CalibrationPlotWindow(
            self.dataset(), spectral_calibration=calibration
        )
        try:
            self.assertFalse(window.show_wavenumber_check.isChecked())
            np.testing.assert_allclose(
                window.spectrum_line.get_xdata(), np.arange(5)
            )
            window.show_wavenumber_check.setChecked(True)
            window.canvas.draw()
            np.testing.assert_allclose(
                window.spectrum_line.get_xdata(),
                calibration.transform(np.arange(5)),
            )

            initial_index = window.selected_index
            window.toolbar.pan()
            try:
                self.assertTrue(window.toolbar.mode)
                self.click_point(window, 2)
                self.assertEqual(window.selected_index, initial_index)
            finally:
                window.toolbar.pan()
        finally:
            window.close()

    def test_missing_spectrum_and_invalid_points_do_not_invent_a_trace(self) -> None:
        missing_spectrum = xr.Dataset(
            {
                "imgs": (
                    ("idx", "Y", "X"),
                    np.zeros((2, 4, 4), dtype=float),
                ),
                "rel_BF_pos": (
                    ("idx", "rel_BF"),
                    [[1.0, 1.0], [2.0, 2.0]],
                ),
            }
        )
        invalid_points = xr.Dataset(
            {
                "imgs": (
                    ("idx", "Y", "X"),
                    np.zeros((2, 4, 4), dtype=float),
                ),
                "rel_BF_pos": (
                    ("idx", "rel_BF"),
                    [[np.nan, 1.0], [2.0, np.inf]],
                ),
                "specs": (
                    ("idx", "N", "spec_dim"),
                    np.ones((2, 2, 5), dtype=float),
                ),
            }
        )
        windows = [
            CalibrationPlotWindow(missing_spectrum),
            CalibrationPlotWindow(invalid_points),
        ]
        try:
            no_spectrum, no_points = windows
            no_spectrum.canvas.draw()
            self.assertEqual(no_spectrum.selected_index, 0)
            self.assertIsNone(no_spectrum.spectrum_line)
            self.assertIn(
                "spectrum unavailable", no_spectrum.selection_label.text()
            )

            no_points.canvas.draw()
            self.assertIsNone(no_points.selected_index)
            self.assertIsNone(no_points.spectrum_line)
            self.assertIn(
                "No finite calibration acquisition points",
                no_points.selection_label.text(),
            )
        finally:
            for window in windows:
                window.close()

    def test_scalar_per_point_is_not_treated_as_a_spectrum(self) -> None:
        ds = self.dataset().drop_vars("specs")
        ds["specs"] = (("idx",), [1.0, 2.0, 3.0])
        window = CalibrationPlotWindow(ds)
        try:
            window.canvas.draw()
            self.assertIsNone(window.spectrum_line)
            self.assertIn("spectrum unavailable", window.selection_label.text())
            self.assertTrue(window.select_point(1))
            self.assertIsNone(window.selected_spectrum)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
