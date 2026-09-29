import os
import unittest
from io import BytesIO

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import to_rgba
from matplotlib.image import imread
from qtpy.QtWidgets import QApplication, QMainWindow, QSizePolicy, QWidget

from napari_raman_widget.plot_windows import (
    CalibrationPlotWindow,
    DatasetViewerWindow,
    DetectorImageWindow,
    GridScanPlotWindow,
    ReferenceSpectraWindow,
    SpectrumWindow,
)
from napari_raman_widget.spectra import asls_baseline, smooth_spectra


class FixedYScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def assert_plot_background(self, panel, *, white: bool) -> None:
        panel.canvas.draw()
        expected = to_rgba("white" if white else "none")
        np.testing.assert_allclose(panel.fig.get_facecolor(), expected)
        for ax in panel.fig.axes:
            np.testing.assert_allclose(ax.get_facecolor(), expected)

    @staticmethod
    def text_luminance(text_artist) -> float:
        red, green, blue, _alpha = to_rgba(text_artist.get_color())
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue

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

    def test_plot_classes_are_dockable_widget_panels(self) -> None:
        for panel_type in (
            CalibrationPlotWindow,
            DetectorImageWindow,
            SpectrumWindow,
            ReferenceSpectraWindow,
            GridScanPlotWindow,
            DatasetViewerWindow,
        ):
            with self.subTest(panel_type=panel_type.__name__):
                self.assertTrue(issubclass(panel_type, QWidget))
                self.assertFalse(issubclass(panel_type, QMainWindow))

        window = SpectrumWindow(np.arange(11, dtype=float))
        try:
            self.assertIsNotNone(window.layout())
            self.assertEqual(
                window.canvas.sizePolicy().horizontalPolicy(),
                QSizePolicy.Expanding,
            )
            self.assertEqual(
                window.canvas.sizePolicy().verticalPolicy(),
                QSizePolicy.Expanding,
            )
            self.assertEqual(
                window.layout().stretch(window.layout().indexOf(window.canvas)),
                1,
            )
        finally:
            window.close()

    def test_static_and_grid_panels_construct_as_widgets(self) -> None:
        calibration_ds = xr.Dataset(
            {
                "imgs": (("point", "Y", "X"), np.zeros((2, 4, 4))),
                "rel_BF_pos": (("point", "coord"), [[1.0, 1.0], [2.0, 2.0]]),
            }
        )
        grid_ds = xr.Dataset(
            {
                "BF": (("Y", "X"), np.zeros((4, 4))),
                "grid_pos": (("point", "coord"), [[1.0, 1.0], [2.0, 2.0]]),
                "specs": (("point", "pixel"), np.ones((2, 11))),
            }
        )
        dataset_index = pd.MultiIndex.from_tuples(
            [(0, 0, 0, 0)], names=["t", "p", "z", "pt"]
        )
        dataset_df = pd.DataFrame(
            [[0.0, 1.0, 2.0, 1.0, 2.0, 0.0]],
            index=dataset_index,
            columns=[0, 1, 2, "X", "Y", "time"],
        )
        dataset_da = xr.DataArray(
            np.zeros((1, 1, 1, 1, 4, 4)),
            dims=["t", "p", "c", "z", "y", "x"],
            coords={"t": [0], "p": [0], "c": [0], "z": [0]},
        )
        panels = [
            CalibrationPlotWindow(calibration_ds),
            DetectorImageWindow(np.ones((1, 4, 11))),
            ReferenceSpectraWindow(
                np.ones((2, 3, 11)), np.array([-1.0, 1.0])
            ),
            GridScanPlotWindow(grid_ds),
            DatasetViewerWindow(dataset_df, dataset_da),
        ]
        try:
            for panel in panels:
                with self.subTest(panel_type=type(panel).__name__):
                    self.assertIsNotNone(panel.layout())
                    self.assertIsNotNone(panel.canvas)
                    self.assertTrue(
                        hasattr(panel, "white_background_check")
                    )
                    self.assertFalse(panel.white_background_check.isChecked())
                    self.assert_plot_background(panel, white=False)
        finally:
            for panel in panels:
                panel.close()

    def test_background_control_round_trip_survives_axes_clear(self) -> None:
        panels = [
            SpectrumWindow(np.arange(11, dtype=float)),
            DetectorImageWindow(np.ones((1, 4, 11))),
        ]
        try:
            for panel in panels:
                with self.subTest(panel_type=type(panel).__name__):
                    self.assertFalse(panel.white_background_check.isChecked())
                    self.assert_plot_background(panel, white=False)
                    rgba = np.asarray(panel.canvas.buffer_rgba())
                    self.assertEqual(int(rgba[0, 0, 3]), 0)

                    panel.white_background_check.setChecked(True)
                    if isinstance(panel, SpectrumWindow):
                        panel.update_spectrum(np.arange(11, dtype=float) * 2)
                    else:
                        panel.update_frames(np.full((1, 4, 11), 2.0))
                    self.assert_plot_background(panel, white=True)
                    rgba = np.asarray(panel.canvas.buffer_rgba())
                    np.testing.assert_array_equal(
                        rgba[0, 0], np.array([255, 255, 255, 255])
                    )

                    panel.white_background_check.setChecked(False)
                    if isinstance(panel, SpectrumWindow):
                        panel.update_spectrum(np.arange(11, dtype=float) * 3)
                    else:
                        panel.update_frames(np.full((1, 4, 11), 3.0))
                    self.assert_plot_background(panel, white=False)
        finally:
            for panel in panels:
                panel.close()

    def test_saved_png_matches_background_immediately_after_clear(self) -> None:
        panel = SpectrumWindow(np.arange(11, dtype=float))
        try:
            panel.update_spectrum(np.arange(11, dtype=float) * 2)
            transparent_output = BytesIO()
            panel.fig.savefig(transparent_output, format="png")
            transparent_output.seek(0)
            transparent_rgba = imread(transparent_output, format="png")
            self.assertEqual(float(transparent_rgba[0, 0, 3]), 0.0)

            panel.white_background_check.setChecked(True)
            panel.update_spectrum(np.arange(11, dtype=float) * 3)
            white_output = BytesIO()
            panel.fig.savefig(white_output, format="png")
            white_output.seek(0)
            white_rgba = imread(white_output, format="png")
            np.testing.assert_allclose(
                white_rgba[0, 0], np.ones(4), atol=1 / 255
            )
        finally:
            panel.close()

    def test_colorbars_follow_dark_host_and_white_background(self) -> None:
        detector = DetectorImageWindow(np.ones((1, 4, 11)))
        reference = ReferenceSpectraWindow(
            np.ones((2, 3, 11)), np.array([-1.0, 1.0])
        )
        panels = [detector, reference]
        try:
            for panel in panels:
                panel.setStyleSheet(
                    "QWidget { background: #202124; color: #f0f2f5; }"
                )
                panel.show()
            self.app.processEvents()
            self.app.processEvents()

            for panel in panels:
                panel._matplotlib_background_theme.refresh()
                panel.canvas.draw()
                colorbar_ax = panel.fig.axes[1]
                self.assertGreater(
                    self.text_luminance(colorbar_ax.yaxis.label), 0.7
                )
                self.assertTrue(colorbar_ax.get_yticklabels())
                self.assertGreater(
                    self.text_luminance(colorbar_ax.get_yticklabels()[0]),
                    0.7,
                )

                panel.white_background_check.setChecked(True)
                panel.canvas.draw()
                self.assertLess(
                    self.text_luminance(colorbar_ax.yaxis.label), 0.3
                )

            detector.white_background_check.setChecked(False)
            original_colorbar_ax = detector._colorbar.ax
            detector._toggle_view()
            detector.canvas.draw()
            self.assertFalse(original_colorbar_ax.get_visible())
            detector._toggle_view()
            detector.canvas.draw()
            self.assertTrue(original_colorbar_ax.get_visible())
            self.assertIs(detector._colorbar.ax, original_colorbar_ax)
            self.assertEqual(len(detector.fig.axes), 2)
            self.assertGreater(
                self.text_luminance(original_colorbar_ax.yaxis.label), 0.7
            )
        finally:
            for panel in panels:
                panel.close()

    def test_processing_expands_before_panel_is_shown(self) -> None:
        window = SpectrumWindow(np.arange(11, dtype=float))
        try:
            self.assertFalse(window.processing_toggle.isChecked())
            self.assertTrue(window.processing_panel.isHidden())

            window.processing_toggle.setChecked(True)

            self.assertFalse(window.processing_panel.isHidden())
            window.processing_toggle.setChecked(False)
            self.assertTrue(window.processing_panel.isHidden())
        finally:
            window.close()

    def test_processing_numeric_inputs_stay_in_sync_with_sliders(self) -> None:
        window = SpectrumWindow(np.arange(21, dtype=float))
        try:
            window.smoothing_window_input.setValue(9)
            self.assertEqual(window.smoothing_window_slider.value(), 2)
            self.assertEqual(window.smoothing_window_label.text(), "Window: 9")

            window.smoothing_window_input.setValue(8)
            self.assertEqual(window.smoothing_window_input.value(), 9)
            self.assertEqual(window.smoothing_window_slider.value(), 2)

            window.baseline_lambda_input.setValue(1e5)
            self.assertEqual(window.baseline_lambda_slider.value(), 20)
            self.assertEqual(window.baseline_lambda_label.text(), "λ: 1e5")

            window.baseline_lambda_slider.setValue(24)
            self.assertEqual(window.baseline_lambda_input.value(), 1e6)
            self.assertEqual(window.baseline_lambda_label.text(), "λ: 1e6")
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

    def test_baseline_subtraction_is_a_display_only_toggle(self) -> None:
        x = np.linspace(0, 1, 101)
        raw = np.array([
            8 + 4 * x + 20 * np.exp(-((x - 0.5) / 0.04) ** 2)
        ])
        window = SpectrumWindow(raw)
        try:
            self.assertFalse(window.baseline_subtraction_check.isChecked())
            self.assertTrue(window.baseline_lambda_controls.isHidden())
            self.assertEqual(window.baseline_lambda_label.text(), "λ: 1e6")

            window.baseline_subtraction_check.setChecked(True)

            self.assertFalse(window.baseline_lambda_controls.isHidden())
            np.testing.assert_array_equal(window.spec, raw)
            np.testing.assert_allclose(
                window.ax.lines[0].get_ydata(),
                raw[0] - asls_baseline(raw[0], lam=1e6),
            )

            window.baseline_lambda_slider.setValue(20)

            self.assertEqual(window.baseline_lambda_label.text(), "λ: 1e5")
            np.testing.assert_allclose(
                window.ax.lines[0].get_ydata(),
                raw[0] - asls_baseline(raw[0], lam=1e5),
            )
        finally:
            window.close()

    def test_short_live_spectrum_disables_display_filters_safely(self) -> None:
        window = SpectrumWindow(np.arange(11, dtype=float))
        try:
            window.baseline_subtraction_check.setChecked(True)
            window.smoothing_check.setChecked(True)

            window.update_spectrum([1.0, 2.0])

            self.assertFalse(window.baseline_subtraction_check.isChecked())
            self.assertFalse(window.smoothing_check.isChecked())
            self.assertTrue(window.baseline_lambda_controls.isHidden())
            self.assertTrue(window.smoothing_window_controls.isHidden())
            np.testing.assert_allclose(
                window.ax.lines[0].get_ydata(),
                [1.0, 2.0],
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

    def test_dataset_spectrum_scale_can_be_fixed_and_released(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [(0, 0, 0, 0), (1, 0, 0, 0)],
            names=["t", "p", "z", "pt"],
        )
        df = pd.DataFrame(
            [
                [0.0, 1.0, 2.0, 1.0, 2.0, 0.0],
                [0.0, 100.0, 200.0, 1.0, 2.0, 1.0],
            ],
            index=index,
            columns=[0, 1, 2, "X", "Y", "time"],
        )
        da = xr.DataArray(
            np.zeros((2, 1, 1, 1, 4, 4)),
            dims=["t", "p", "c", "z", "y", "x"],
            coords={"t": [0, 1], "p": [0], "c": [0], "z": [0]},
        )
        window = DatasetViewerWindow(df, da)
        try:
            initial_limits = window.ax_spec.get_ylim()
            window.fix_y_scale_check.setChecked(True)

            window.t_slider.setValue(1)

            np.testing.assert_allclose(
                window.ax_spec.get_ylim(), initial_limits
            )
            window.fix_y_scale_check.setChecked(False)
            self.assertGreater(window.ax_spec.get_ylim()[1], 200.0)
        finally:
            window.close()

    def test_dataset_navigation_has_synchronized_numeric_inputs(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [(0, 0, 0, 0), (1, 0, 0, 0)],
            names=["t", "p", "z", "pt"],
        )
        df = pd.DataFrame(
            [
                [0.0, 1.0, 2.0, 1.0, 2.0, 0.0],
                [3.0, 4.0, 5.0, 1.0, 2.0, 1.0],
            ],
            index=index,
            columns=[0, 1, 2, "X", "Y", "time"],
        )
        da = xr.DataArray(
            np.zeros((2, 1, 1, 1, 4, 4)),
            dims=["t", "p", "c", "z", "y", "x"],
            coords={"t": [0, 1], "p": [0], "c": [0], "z": [0]},
        )
        window = DatasetViewerWindow(df, da)
        try:
            self.assertIsNot(window.navigation_group, window.view_controls_group)
            window.t_input.setValue(1)
            self.assertEqual(window.t_slider.value(), 1)
            self.assertIn("t=1", window.ax_img.get_title())

            window.t_slider.setValue(0)
            self.assertEqual(window.t_input.value(), 0)
            self.assertIn("t=0", window.ax_img.get_title())
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

    def test_detector_row_sum_can_have_its_baseline_subtracted(self) -> None:
        x = np.linspace(0, 1, 101)
        row = 5 + 2 * x + 10 * np.exp(-((x - 0.5) / 0.04) ** 2)
        frames = np.tile(row, (1, 4, 1))
        window = DetectorImageWindow(frames)
        try:
            self.assertTrue(window.baseline_subtraction_check.isHidden())
            window._toggle_view()
            self.assertFalse(window.baseline_subtraction_check.isHidden())
            raw_row_sum = frames.mean(axis=0).sum(axis=0)

            window.baseline_subtraction_check.setChecked(True)

            np.testing.assert_allclose(
                window.ax.lines[0].get_ydata(),
                raw_row_sum - asls_baseline(raw_row_sum, lam=1e6),
            )
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
