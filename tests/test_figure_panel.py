import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import xarray as xr
from matplotlib.colors import to_rgba
from qtpy.QtCore import QCoreApplication, QEvent, QTimer, Qt
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

    def test_theme_event_filter_is_safe_during_partial_teardown(self) -> None:
        panel = FigurePanel("Theme teardown")
        manager = panel._matplotlib_background_theme
        errors = []

        try:
            del manager._applying
            with (
                patch.object(manager, "schedule_refresh") as schedule_refresh,
                patch.object(
                    sys,
                    "excepthook",
                    new=lambda error_type, error, traceback: errors.append(
                        (error_type, error, traceback)
                    ),
                ),
            ):
                QCoreApplication.sendEvent(
                    panel, QEvent(QEvent.PaletteChange)
                )
                filtered = manager.eventFilter(
                    panel, QEvent(QEvent.PaletteChange)
                )

            self.assertFalse(filtered)
            self.assertEqual(errors, [])
            schedule_refresh.assert_not_called()
        finally:
            manager._applying = False
            panel.close()

    def test_queued_theme_refresh_is_safe_after_qt_children_are_deleted(
        self,
    ) -> None:
        self.app.processEvents()
        panel = FigurePanel("Queued theme teardown")
        background_theme = panel._matplotlib_background_theme
        icon_theme = panel.toolbar._raman_icon_contrast
        errors = []

        QTimer.singleShot(0, background_theme.refresh)
        QTimer.singleShot(0, icon_theme.refresh)
        with patch.object(
            sys,
            "excepthook",
            new=lambda error_type, error, traceback: errors.append(
                (error_type, error, traceback)
            ),
        ):
            panel.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            self.app.processEvents()
            self.app.processEvents()

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
