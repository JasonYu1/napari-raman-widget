import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from qtpy.QtWidgets import QApplication

from napari_raman_widget.plot_windows import (
    DetectorImageWindow,
    SpectrumWindow,
)
from napari_raman_widget.spectra import smooth_spectra


class FixedYScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_spectrum_scale_can_be_fixed_and_released(self) -> None:
        window = SpectrumWindow([[0.0, 1.0, 2.0]])
        try:
            initial_limits = window.ax.get_ylim()
            window.fix_y_scale_check.setChecked(True)

            window.update_spectrum([[0.0, 100.0, 200.0]])

            np.testing.assert_allclose(window.ax.get_ylim(), initial_limits)
            window.fix_y_scale_check.setChecked(False)
            self.assertGreater(window.ax.get_ylim()[1], 200.0)
        finally:
            window.close()

    def test_spectral_bias_is_a_per_window_display_toggle(self) -> None:
        raw = np.array([[10.0, 20.0, 30.0]])
        bias = np.array([1.0, 2.0, 3.0])
        window = SpectrumWindow(raw, spectral_bias=bias)
        try:
            np.testing.assert_array_equal(window.spec, raw)
            self.assertFalse(window.remove_spectral_bias_check.isHidden())
            self.assertFalse(window.remove_spectral_bias_check.isChecked())
            np.testing.assert_allclose(
                window.ax.lines[0].get_ydata(),
                raw[0],
            )

            window.remove_spectral_bias_check.setChecked(True)

            np.testing.assert_array_equal(window.spec, raw)
            np.testing.assert_allclose(
                window.ax.lines[0].get_ydata(),
                raw[0] - bias,
            )
        finally:
            window.close()

    def test_smoothing_is_an_odd_window_display_toggle(self) -> None:
        raw = np.array([[0, 1, 9, 2, 0, 8, 1, 0, 7, 2, 0]], dtype=float)
        window = SpectrumWindow(raw)
        try:
            self.assertTrue(window.smoothing_window_controls.isHidden())
            self.assertEqual(window.smoothing_window_label.text(), "Window: 5")

            window.smoothing_window_slider.setValue(1)
            self.assertEqual(window.smoothing_window_label.text(), "Window: 7")
            window.smoothing_check.setChecked(True)

            self.assertFalse(window.smoothing_window_controls.isHidden())
            np.testing.assert_array_equal(window.spec, raw)
            np.testing.assert_allclose(
                window.ax.lines[0].get_ydata(),
                smooth_spectra(raw[0], 7),
            )
        finally:
            window.close()

    def test_row_sum_scale_can_be_fixed_and_released(self) -> None:
        window = DetectorImageWindow(np.ones((1, 4, 6)))
        try:
            window._toggle_view()
            initial_limits = window.ax.get_ylim()
            window.fix_y_scale_check.setChecked(True)

            window.update_frames(np.full((1, 4, 6), 100.0))

            np.testing.assert_allclose(window.ax.get_ylim(), initial_limits)
            window.fix_y_scale_check.setChecked(False)
            self.assertGreater(window.ax.get_ylim()[1], 400.0)
        finally:
            window.close()

    def test_detector_row_sum_can_be_smoothed(self) -> None:
        frames = np.zeros((1, 4, 11), dtype=float)
        frames[0, :, 5] = 10.0
        window = DetectorImageWindow(frames)
        try:
            window._toggle_view()
            raw_row_sum = frames.mean(axis=0).sum(axis=0)
            window.smoothing_window_slider.setValue(1)
            window.smoothing_check.setChecked(True)

            np.testing.assert_allclose(
                window.ax.lines[0].get_ydata(),
                smooth_spectra(raw_row_sum, 7),
            )
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
