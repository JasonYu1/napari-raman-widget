"""Irregular ROI grids preserve point-to-spectrum mapping in result plots."""

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import xarray as xr
from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtWidgets import QApplication

from napari_raman_widget.plot_windows import GridScanPlotWindow


class IrregularROIPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        # Seven points are neither a square grid nor a complete rectangle.
        self.points = np.array(
            [[1, 1], [1, 4], [2, 2], [2, 4], [4, 1], [4, 3], [5, 2]],
            dtype=float,
        )
        self.spectra = np.arange(7)[:, None] * 100 + np.arange(11)[None, :] ** 2
        self.windows = []

    def tearDown(self):
        for window in self.windows:
            window.close()
            window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def make_window(self, dataset):
        window = GridScanPlotWindow(dataset)
        self.windows.append(window)
        return window

    def assert_selected_spectrum(self, window, index, expected):
        window._on_pick(SimpleNamespace(artist=window._scat, ind=[index]))
        self.assertEqual(window._sel, index)
        self.assertFalse(window._show_average)
        np.testing.assert_array_equal(window._spec_line.get_ydata(), expected)
        np.testing.assert_array_equal(
            window._hl.get_offsets(), self.points[index, ::-1].reshape(1, 2)
        )

    def test_single_plane_seven_point_roi_clicks_show_exact_original_spectra(self):
        dataset = xr.Dataset({
            "BF": (("Y", "X"), np.zeros((8, 8))),
            "end_BF": (("Y", "X"), np.ones((8, 8))),
            "grid_pos": (("idx", "xy"), self.points),
            "laser_pos": (("idx", "volt"), self.points / 10),
            "specs": (("N", "spec_dim"), self.spectra),
        })
        window = self.make_window(dataset)

        self.assertFalse(window._has_zscan)
        self.assertEqual(window._specs.shape, (7, 11))
        np.testing.assert_array_equal(window._scat.get_offsets(), self.points[:, ::-1])
        np.testing.assert_array_equal(
            window._spec_line.get_ydata(), self.spectra.mean(axis=0)
        )
        for index in (0, 1, 6):
            with self.subTest(point=index):
                self.assert_selected_spectrum(window, index, self.spectra[index])

        window._mode_btn.click()
        self.assertTrue(window._show_average)
        np.testing.assert_array_equal(
            window._spec_line.get_ydata(), self.spectra.mean(axis=0)
        )
        np.testing.assert_array_equal(dataset["specs"].values, self.spectra)

    def test_multiz_seven_point_roi_keeps_point_index_when_changing_z_plane(self):
        spectra = self.spectra[None, :, :] + np.arange(3)[:, None, None] * 10_000
        dataset = xr.Dataset({
            "BF": (("Y", "X"), np.zeros((8, 8))),
            "BF_z": (("z", "Y", "X"), np.zeros((3, 8, 8))),
            "z_range": ("z", [-2.0, 0.0, 2.0]),
            "grid_pos": (("idx", "xy"), self.points),
            "laser_pos": (("idx", "volt"), self.points / 10),
            "specs": (("z", "idx", "spec_dim"), spectra),
        })
        window = self.make_window(dataset)

        self.assertTrue(window._has_zscan)
        self.assertEqual(window._specs.shape, (3, 7, 11))
        np.testing.assert_array_equal(window._scat.get_offsets(), self.points[:, ::-1])
        np.testing.assert_array_equal(
            window._spec_line.get_ydata(), spectra[0].mean(axis=0)
        )
        for index in (0, 1, 6):
            with self.subTest(point=index):
                self.assert_selected_spectrum(window, index, spectra[0, index])

        window._z_slider.setValue(2)
        self.assertEqual(window._sel, 6)
        np.testing.assert_array_equal(window._spec_line.get_ydata(), spectra[2, 6])
        self.assert_selected_spectrum(window, 1, spectra[2, 1])
        window._mode_btn.click()
        np.testing.assert_array_equal(
            window._spec_line.get_ydata(), spectra[2].mean(axis=0)
        )
        window._z_slider.setValue(1)
        np.testing.assert_array_equal(
            window._spec_line.get_ydata(), spectra[1].mean(axis=0)
        )
        np.testing.assert_array_equal(dataset["specs"].values, spectra)


if __name__ == "__main__":
    unittest.main()
