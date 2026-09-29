import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import xarray as xr
from matplotlib.colors import to_rgba
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from napari_raman_widget.calibration.calibrator import ManualImageSelector
from napari_raman_widget.calibration.stage_points import StagePointPicker
from napari_raman_widget.figure_panel import FigurePanel


class FigurePanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_stage_picker_uses_installed_canvas(self) -> None:
        panel = FigurePanel("Point picker", "Click, then press Enter")
        picker = StagePointPicker(
            np.zeros((2, 8, 8)),
            figure=panel.figure,
        )
        try:
            self.assertIs(picker.figure, panel.figure)
            self.assertIs(picker.figure.canvas, panel.canvas)
            self.assertIn(picker.axes, panel.figure.axes)
            self.assertEqual(panel.canvas.focusPolicy(), Qt.StrongFocus)
        finally:
            picker.close()
            panel.close()

    def test_embedded_picker_does_not_close_figure_at_last_frame(self) -> None:
        panel = FigurePanel("Point picker")
        picker = StagePointPicker(
            np.zeros((1, 8, 8)),
            figure=panel.figure,
        )
        try:
            picker._advance()
            self.assertIs(picker.figure.canvas, panel.canvas)
            self.assertEqual(len(panel.figure.axes), 1)
        finally:
            picker.close()
            panel.close()

    def test_panel_runs_selector_cleanup_once(self) -> None:
        panel = FigurePanel("Point picker")
        calls = []
        panel.add_cleanup(lambda: calls.append("closed"))

        panel.close()
        panel.close()

        self.assertEqual(calls, ["closed"])

    def test_background_control_survives_selector_axes_clear(self) -> None:
        panel = FigurePanel("Point picker")
        picker = StagePointPicker(
            np.zeros((2, 8, 8)), figure=panel.figure
        )
        panel.add_cleanup(picker.close)
        try:
            panel.canvas.draw()
            self.assertFalse(panel.white_background_check.isChecked())
            np.testing.assert_allclose(
                panel.figure.get_facecolor(), to_rgba("none")
            )
            np.testing.assert_allclose(
                picker.axes.get_facecolor(), to_rgba("none")
            )
            rgba = np.asarray(panel.canvas.buffer_rgba())
            self.assertEqual(int(rgba[0, 0, 3]), 0)

            panel.white_background_check.setChecked(True)
            picker.axes.clear()
            picker.axes.set_title("Selection")
            panel.canvas.draw()
            panel.canvas.draw()
            np.testing.assert_allclose(
                panel.figure.get_facecolor(), to_rgba("white")
            )
            np.testing.assert_allclose(
                picker.axes.get_facecolor(), to_rgba("white")
            )
            self.assertLess(to_rgba(picker.axes.title.get_color())[0], 0.3)

            panel.white_background_check.setChecked(False)
            picker.axes.clear()
            panel.canvas.draw()
            panel.canvas.draw()
            np.testing.assert_allclose(
                picker.axes.get_facecolor(), to_rgba("none")
            )
        finally:
            panel.close()

    def test_manual_selector_uses_and_releases_installed_canvas(self) -> None:
        panel = FigurePanel("Manual calibration")
        dataset = xr.Dataset(
            {
                "imgs": xr.DataArray(
                    np.zeros((2, 16, 16)),
                    dims=("idx", "Y", "X"),
                )
            }
        )
        selector = ManualImageSelector(dataset, figure=panel.figure)
        panel.add_cleanup(selector.close)

        self.assertIs(selector.fig, panel.figure)
        self.assertIs(selector.fig.canvas, panel.canvas)
        self.assertFalse(selector._owns_figure)

        selector.current_idx = selector.num_images - 1
        with patch("matplotlib.pyplot.close") as close_figure:
            selector.on_key_press(SimpleNamespace(key="enter"))
        close_figure.assert_not_called()
        self.assertIs(selector.fig.canvas, panel.canvas)

        panel.close()
        self.assertIsNone(selector.cid)


if __name__ == "__main__":
    unittest.main()
