import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from qtpy.QtWidgets import QApplication

from napari_raman_widget.plot_windows import (
    DetectorImageWindow,
    SpectrumWindow,
)


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


if __name__ == "__main__":
    unittest.main()
