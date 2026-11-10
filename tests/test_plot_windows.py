import os
import unittest
from io import BytesIO
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import to_rgba
from matplotlib.image import imread
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QWidget,
)

from napari_raman_widget.plot_windows import (
    CalibrationPlotWindow,
    DatasetViewerWindow,
    DetectorImageWindow,
    GridScanPlotWindow,
    ReferenceSpectraWindow,
    SpectrumWindow,
)
from napari_raman_widget.spectra import asls_baseline, smooth_spectra
from napari_raman_widget.spectral_calibration import (
    PixelToWavenumberCalibration,
)


class _FakeCosmicRayCalibration:
    def __init__(
        self, *, offset=2.0, compatible=True, reason="", failure=None
    ):
        self.offset = float(offset)
        self.compatible = bool(compatible)
        self.reason = reason
        self.failure = failure
        self.compatibility_calls = []
        self.correct_calls = []

    def compatibility(self, values, *, exposure_ms=None, read_mode=None):
        self.compatibility_calls.append(
            (np.asarray(values).shape, exposure_ms, read_mode)
        )
        return self.compatible, self.reason

    def correct(self, values):
        values = np.asarray(values)
        self.correct_calls.append(np.array(values, copy=True))
        if self.failure is not None:
            raise self.failure
        return values - self.offset


class _DarkOnlyPeakCalibration(_FakeCosmicRayCalibration):
    read_mode = "fvb"
    open_reference_path = None

    def __init__(self, detector_length):
        super().__init__(offset=0.0)
        self.fvb_dark_sigma_adu = np.ones(detector_length, dtype=float)

    def correct(self, values):
        values = np.asarray(values)
        self.correct_calls.append(np.array(values, copy=True))
        corrected = np.array(values, dtype=float, copy=True)
        corrected[2] = 10.0
        if corrected[-1] > 500.0:
            corrected[-1] = 10.0
        return corrected


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

    def test_spectral_axis_defaults_to_pixels_and_opt_in_wavenumber(self) -> None:
        calibration = PixelToWavenumberCalibration(
            [0.0, 10.0], [100.0, 200.0], degree=1
        )
        calibrated = SpectrumWindow(
            np.arange(11, dtype=float),
            spectral_calibration=calibration,
        )
        uncalibrated = SpectrumWindow(np.arange(11, dtype=float))
        try:
            self.assertTrue(calibrated.show_wavenumber_check.isEnabled())
            self.assertFalse(calibrated.show_wavenumber_check.isChecked())
            self.assertEqual(calibrated.ax.get_xlabel(), "Pixel")
            np.testing.assert_allclose(
                calibrated.ax.lines[0].get_xdata(), np.arange(11)
            )

            calibrated.show_wavenumber_check.setChecked(True)

            self.assertEqual(
                calibrated.ax.get_xlabel(), "Raman shift (cm⁻¹)"
            )
            np.testing.assert_allclose(
                calibrated.ax.lines[0].get_xdata(),
                calibration.transform(np.arange(11)),
            )
            calibrated.show_wavenumber_check.setChecked(False)
            self.assertEqual(calibrated.ax.get_xlabel(), "Pixel")

            self.assertFalse(uncalibrated.show_wavenumber_check.isEnabled())
            self.assertFalse(uncalibrated.show_wavenumber_check.isChecked())
            self.assertEqual(uncalibrated.ax.get_xlabel(), "Pixel")
        finally:
            calibrated.close()
            uncalibrated.close()

    def test_display_choices_share_one_compact_view_row(self) -> None:
        calibration = PixelToWavenumberCalibration(
            [0.0, 10.0], [100.0, 200.0], degree=1
        )
        spectrum = SpectrumWindow(
            np.arange(11, dtype=float),
            spectral_calibration=calibration,
        )
        detector = DetectorImageWindow(
            np.ones((1, 4, 11)), spectral_calibration=calibration
        )
        calibration_result = CalibrationPlotWindow(
            xr.Dataset(
                {
                    "imgs": (("point", "Y", "X"), np.zeros((2, 4, 4))),
                    "rel_BF_pos": (
                        ("point", "coord"),
                        [[1.0, 1.0], [2.0, 2.0]],
                    ),
                    "specs": (
                        ("point", "repeat", "pixel"),
                        np.ones((2, 2, 11)),
                    ),
                }
            ),
            spectral_calibration=calibration,
        )
        index = pd.MultiIndex.from_tuples(
            [(0, 0, 0, 0)], names=["t", "p", "z", "pt"]
        )
        frame = pd.DataFrame(
            [[0.0, 1.0, 2.0, 1.0, 2.0, 0.0]],
            index=index,
            columns=[0, 1, 2, "X", "Y", "time"],
        )
        images = xr.DataArray(
            np.zeros((1, 1, 1, 1, 4, 4)),
            dims=["t", "p", "c", "z", "y", "x"],
            coords={"t": [0], "p": [0], "c": [0], "z": [0]},
        )
        dataset = DatasetViewerWindow(
            frame, images, spectral_calibration=calibration
        )
        panels = [spectrum, detector, calibration_result, dataset]
        try:
            for panel in panels:
                with self.subTest(panel_type=type(panel).__name__):
                    grid = panel.view_controls_group.layout()
                    controls = [
                        panel.fix_y_scale_check,
                        panel.show_wavenumber_check,
                        panel.white_background_check,
                    ]
                    positions = [
                        grid.getItemPosition(grid.indexOf(control))
                        for control in controls
                    ]
                    self.assertEqual(len({position[0] for position in positions}), 1)
                    self.assertLess(positions[0][1], positions[1][1])
                    self.assertLess(positions[1][1], positions[2][1])
                    self.assertIs(
                        panel.white_background_check.parentWidget(),
                        panel.view_controls_group,
                    )
        finally:
            for panel in panels:
                panel.close()

    def test_show_all_sits_beside_axis_calibration_and_remains_functional(self) -> None:
        for bias in (None, np.ones(11)):
            with self.subTest(has_spectral_bias=bias is not None):
                window = SpectrumWindow(
                    np.arange(22, dtype=float).reshape(2, 11),
                    spectral_bias=bias,
                )
                try:
                    grid = window.view_controls_group.layout()

                    def position(control):
                        return grid.getItemPosition(grid.indexOf(control))

                    toggle = position(window.toggle_btn)
                    calibrate = position(window.calibration_btn)
                    self.assertEqual(toggle[0], calibrate[0])
                    self.assertEqual(toggle[1] + toggle[3], calibrate[1])
                    self.assertNotEqual(
                        toggle[0], position(window.view_mode_label)[0]
                    )
                    self.assertEqual(
                        position(window.remove_spectral_bias_check)[0], toggle[0]
                    )
                    self.assertEqual(
                        window.remove_spectral_bias_check.isHidden(), bias is None
                    )
                    view_rows = {
                        position(control)[0] for control in (
                            window.fix_y_scale_check,
                            window.show_wavenumber_check,
                            window.white_background_check,
                        )
                    }
                    self.assertEqual(len(view_rows), 1)
                    self.assertNotIn(toggle[0], view_rows)

                    self.assertEqual(window.toggle_btn.text(), "Show all")
                    window.toggle_btn.click()
                    self.assertEqual(window.toggle_btn.text(), "Show mean")
                    self.assertEqual(len(window.ax.lines), 2)
                    window.toggle_btn.click()
                    self.assertEqual(window.toggle_btn.text(), "Show all")
                    self.assertEqual(len(window.ax.lines), 1)
                    window.repeat_slider.setValue(2)
                    window.calibration_btn.click()
                    self.assertFalse(window.toggle_btn.isEnabled())
                    self.assertEqual(window.repeat_slider.value(), 0)
                    self.assertFalse(window.repeat_controls.isEnabled())
                    window.toggle_btn.click()
                    self.assertTrue(window._show_mean)
                    window.cancel_calibration_btn.click()
                    self.assertTrue(window.toggle_btn.isEnabled())
                    self.assertTrue(window.repeat_controls.isEnabled())
                    window.toggle_btn.click()
                    self.assertEqual(window.toggle_btn.text(), "Show mean")
                    self.assertEqual(len(window.ax.lines), 2)
                finally:
                    window.close()

    def test_spectrum_repeat_slider_shows_individual_and_aggregate_views(
        self,
    ) -> None:
        raw = np.arange(15, dtype=float).reshape(3, 5)
        original = raw.copy()
        window = SpectrumWindow(raw)
        try:
            self.assertFalse(window.repeat_controls.isHidden())
            self.assertEqual(window.repeat_slider.maximum(), 3)
            self.assertEqual(window.repeat_slider.value(), 0)
            self.assertEqual(window.repeat_input.specialValueText(), "Mean")
            np.testing.assert_allclose(
                window.ax.lines[0].get_ydata(),
                raw.mean(axis=0),
            )

            window.repeat_slider.setValue(2)

            self.assertEqual(window.repeat_input.value(), 2)
            self.assertEqual(len(window.ax.lines), 1)
            np.testing.assert_array_equal(
                window.ax.lines[0].get_ydata(), raw[1]
            )
            self.assertIn("repeat 2 of 3", window.view_mode_label.text())
            self.assertIn("Repeat 2 of 3", window.ax.get_title())

            window.toggle_btn.click()

            self.assertEqual(window.repeat_slider.value(), 0)
            self.assertFalse(window._show_mean)
            self.assertEqual(len(window.ax.lines), 3)

            window.repeat_input.setValue(3)

            self.assertTrue(window._show_mean)
            self.assertEqual(len(window.ax.lines), 1)
            np.testing.assert_array_equal(
                window.ax.lines[0].get_ydata(), raw[2]
            )
            window.repeat_slider.setValue(0)
            np.testing.assert_allclose(
                window.ax.lines[0].get_ydata(), raw.mean(axis=0)
            )
            np.testing.assert_array_equal(window.spec, original)
        finally:
            window.close()

    def test_detector_repeat_slider_selects_image_and_row_sum(self) -> None:
        frames = np.arange(24, dtype=float).reshape(3, 2, 4)
        original = frames.copy()
        window = DetectorImageWindow(frames)
        try:
            self.assertFalse(window.repeat_controls.isHidden())
            np.testing.assert_allclose(
                np.asarray(window.ax.images[0].get_array()),
                frames.mean(axis=0),
            )

            window.repeat_slider.setValue(2)

            np.testing.assert_array_equal(
                np.asarray(window.ax.images[0].get_array()), frames[1]
            )
            self.assertIn("repeat 2 of 3", window.view_mode_label.text())
            window._toggle_view()
            np.testing.assert_array_equal(
                window.ax.lines[0].get_ydata(), frames[1].sum(axis=0)
            )

            window.repeat_input.setValue(0)

            np.testing.assert_allclose(
                window.ax.lines[0].get_ydata(),
                frames.mean(axis=0).sum(axis=0),
            )
            np.testing.assert_array_equal(window.raw_frames, original)
        finally:
            window.close()

    def test_repeat_controls_hide_for_one_repeat_and_reset_on_update(self):
        spectrum = SpectrumWindow(np.arange(5, dtype=float))
        detector = DetectorImageWindow(np.ones((2, 4), dtype=float))
        try:
            for panel in (spectrum, detector):
                self.assertTrue(panel.repeat_controls.isHidden())
                self.assertFalse(panel.repeat_controls.isEnabled())

            spectrum.update_spectrum(np.ones((3, 5), dtype=float))
            detector.update_frames(np.ones((3, 2, 4), dtype=float))
            for panel in (spectrum, detector):
                self.assertFalse(panel.repeat_controls.isHidden())
                self.assertTrue(panel.repeat_controls.isEnabled())
                panel.repeat_slider.setValue(3)

            spectrum.update_spectrum(np.ones((1, 5), dtype=float))
            detector.update_frames(np.ones((1, 2, 4), dtype=float))
            for panel in (spectrum, detector):
                self.assertEqual(panel.repeat_slider.value(), 0)
                self.assertTrue(panel.repeat_controls.isHidden())
                self.assertFalse(panel.repeat_controls.isEnabled())
        finally:
            spectrum.close()
            detector.close()

    def test_axis_calibration_forces_pixels_then_restores_preference(
        self,
    ) -> None:
        calibration = PixelToWavenumberCalibration(
            [0.0, 10.0], [100.0, 200.0], degree=1
        )
        existing = SpectrumWindow(
            np.arange(11, dtype=float),
            spectral_calibration=calibration,
        )
        created = SpectrumWindow(np.arange(11, dtype=float))
        try:
            existing.show_wavenumber_check.setChecked(True)
            existing._start_calibration()
            self.assertFalse(existing.show_wavenumber_check.isChecked())
            self.assertFalse(existing.show_wavenumber_check.isEnabled())
            self.assertEqual(existing.ax.get_xlabel(), "Pixel")

            existing._cancel_calibration()
            self.assertTrue(existing.show_wavenumber_check.isChecked())
            self.assertTrue(existing.show_wavenumber_check.isEnabled())
            self.assertEqual(
                existing.ax.get_xlabel(), "Raman shift (cm⁻¹)"
            )

            created._start_calibration()
            created.calibration_degree_input.setValue(1)
            created._calibration_pixels = [0, 10]
            created._known_shifts = [100.0, 200.0]
            with (
                patch.object(
                    QFileDialog,
                    "getSaveFileName",
                    return_value=("calibration.json", "JSON files (*.json)"),
                ),
                patch(
                    "napari_raman_widget.plot_windows."
                    "save_pixel_to_wavenumber_calibration",
                    return_value="calibration.json",
                ),
                patch.object(QMessageBox, "information"),
            ):
                created._finish_calibration()

            self.assertIsNotNone(created.spectral_calibration)
            self.assertTrue(created.show_wavenumber_check.isEnabled())
            self.assertFalse(created.show_wavenumber_check.isChecked())
            self.assertEqual(created.ax.get_xlabel(), "Pixel")
        finally:
            existing.close()
            created.close()

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

    def test_cosmic_ray_control_is_present_default_off_and_informative(self):
        unavailable = SpectrumWindow(np.arange(5, dtype=float))
        incompatible_calibration = _FakeCosmicRayCalibration(
            compatible=False,
            reason="exposure does not match the calibration",
        )
        incompatible = SpectrumWindow(
            np.arange(5, dtype=float),
            cosmic_ray_calibration=incompatible_calibration,
            exposure_ms=50.0,
        )
        try:
            for window in (unavailable, incompatible):
                self.assertEqual(
                    window.cosmic_ray_removal_check.text(),
                    "Remove cosmic rays (RACER)",
                )
                self.assertFalse(window.cosmic_ray_removal_check.isChecked())
                self.assertFalse(window.cosmic_ray_removal_check.isEnabled())
                self.assertTrue(window.cosmic_ray_removal_check.toolTip())
            self.assertIn(
                "load a compatible", unavailable.cosmic_ray_removal_check.toolTip()
            )
            self.assertIn(
                "exposure does not match",
                incompatible.cosmic_ray_removal_check.toolTip(),
            )
            self.assertEqual(
                incompatible_calibration.compatibility_calls,
                [((1, 5), 50.0, "fvb")],
            )
        finally:
            unavailable.close()
            incompatible.close()

    def test_cosmic_ray_removal_is_available_in_all_spectral_panels(self):
        offset = 2.0
        calibration_raw = np.array(
            [
                [[10, 11, 12, 13, 14], [10, 11, 12, 13, 14]],
                [[20, 21, 22, 23, 24], [20, 21, 22, 23, 24]],
            ],
            dtype=float,
        )
        calibration_ds = xr.Dataset(
            {
                "imgs": (("point", "Y", "X"), np.zeros((2, 3, 3))),
                "rel_BF_pos": (
                    ("point", "coord"),
                    [[1.0, 1.0], [2.0, 2.0]],
                ),
                "specs": (
                    ("point", "repeat", "pixel"), calibration_raw
                ),
            }
        )
        detector_raw = np.stack(
            [np.arange(10, dtype=float).reshape(2, 5) + 10] * 2
        )
        spectrum_raw = np.array(
            [[30, 31, 32, 33, 34], [30, 31, 32, 33, 34]],
            dtype=float,
        )
        reference_raw = np.array(
            [
                [[40, 41, 42, 43, 44], [40, 41, 42, 43, 44]],
                [[50, 51, 52, 53, 54], [50, 51, 52, 53, 54]],
            ],
            dtype=float,
        )
        grid_raw = np.array(
            [[60, 61, 62, 63, 64], [60, 61, 62, 63, 64]],
            dtype=float,
        )
        grid_ds = xr.Dataset(
            {
                "BF": (("Y", "X"), np.zeros((3, 3))),
                "grid_pos": (
                    ("point", "coord"), [[1.0, 1.0], [2.0, 2.0]]
                ),
                "specs": (("point", "pixel"), grid_raw),
            }
        )
        dataset_raw = np.array([70, 71, 72, 73, 74], dtype=float)
        dataset_index = pd.MultiIndex.from_tuples(
            [(0, 0, 0, 0)], names=["t", "p", "z", "pt"]
        )
        dataset_df = pd.DataFrame(
            [[*dataset_raw, 1.0, 2.0, 0.0]],
            index=dataset_index,
            columns=[0, 1, 2, 3, 4, "X", "Y", "time"],
        )
        dataset_da = xr.DataArray(
            np.zeros((1, 1, 1, 1, 3, 3)),
            dims=["t", "p", "c", "z", "y", "x"],
            coords={"t": [0], "p": [0], "c": [0], "z": [0]},
        )

        calibrations = [_FakeCosmicRayCalibration(offset=offset) for _ in range(6)]
        panels = [
            CalibrationPlotWindow(
                calibration_ds,
                cosmic_ray_calibration=calibrations[0],
                exposure_ms=25.0,
            ),
            DetectorImageWindow(
                detector_raw,
                cosmic_ray_calibration=calibrations[1],
                exposure_ms=25.0,
            ),
            SpectrumWindow(
                spectrum_raw,
                cosmic_ray_calibration=calibrations[2],
                exposure_ms=25.0,
            ),
            ReferenceSpectraWindow(
                reference_raw,
                [-1.0, 1.0],
                cosmic_ray_calibration=calibrations[3],
                exposure_ms=25.0,
            ),
            GridScanPlotWindow(
                grid_ds,
                cosmic_ray_calibration=calibrations[4],
                exposure_ms=25.0,
            ),
            DatasetViewerWindow(
                dataset_df,
                dataset_da,
                cosmic_ray_calibration=calibrations[5],
                exposure_ms=25.0,
            ),
        ]
        raw_snapshots = [
            calibration_raw,
            detector_raw,
            spectrum_raw,
            reference_raw,
            grid_raw,
            dataset_raw.reshape(1, -1),
        ]
        try:
            for panel in panels:
                self.assertFalse(panel.cosmic_ray_removal_check.isChecked())
                self.assertTrue(panel.cosmic_ray_removal_check.isEnabled())

            for panel in panels:
                panel.cosmic_ray_removal_check.setChecked(True)

            np.testing.assert_allclose(
                panels[0].spectrum_line.get_ydata(),
                calibration_raw[0, 0] - offset,
            )
            np.testing.assert_allclose(
                np.asarray(panels[1].ax.images[0].get_array()),
                detector_raw.mean(axis=0) - offset,
            )
            np.testing.assert_allclose(
                panels[2].ax.lines[0].get_ydata(),
                spectrum_raw[0] - offset,
            )
            for line, expected in zip(panels[3]._spectral_lines, reference_raw):
                np.testing.assert_allclose(
                    line.get_ydata(), expected[0] - offset
                )
            np.testing.assert_allclose(
                panels[4]._spec_line.get_ydata(), grid_raw[0] - offset
            )
            np.testing.assert_allclose(
                panels[5].spec_line.get_ydata(), dataset_raw - offset
            )

            retained_raw = [
                panels[0].raw_spectra,
                panels[1].raw_frames,
                panels[2].spec,
                panels[3]._raw_reference_spectra,
                panels[4]._specs,
                panels[5]._raw_dataset_spectra,
            ]
            for actual, expected in zip(retained_raw, raw_snapshots):
                np.testing.assert_array_equal(actual, expected)

            expected_calls = [4, 2, 2, 4, 2, 1]
            self.assertEqual(
                [len(calibration.correct_calls) for calibration in calibrations],
                expected_calls,
            )
            expected_compatibility_shapes = [
                (4, 5),
                (2, 2, 5),
                (2, 5),
                (4, 5),
                (2, 5),
                (1, 5),
            ]
            self.assertEqual(
                [
                    calibration.compatibility_calls[0][0]
                    for calibration in calibrations
                ],
                expected_compatibility_shapes,
            )
            self.assertEqual(calibrations[1].compatibility_calls[0][2], "image")
        finally:
            for panel in panels:
                panel.close()

    def test_cosmic_ray_correction_is_ordered_cached_and_invalidated(self):
        raw = np.array(
            [[20, 21, 22, 23, 24], [20, 21, 22, 23, 24]], dtype=float
        )
        original = raw.copy()
        bias = np.arange(5, dtype=float)
        calibration = _FakeCosmicRayCalibration(offset=3.0)
        window = SpectrumWindow(
            raw,
            spectral_bias=bias,
            cosmic_ray_calibration=calibration,
        )
        try:
            window.cosmic_ray_removal_check.setChecked(True)
            self.assertEqual(len(calibration.correct_calls), 2)
            np.testing.assert_allclose(
                window.ax.lines[0].get_ydata(), raw[0] - 3.0
            )

            window.remove_spectral_bias_check.setChecked(True)
            np.testing.assert_allclose(
                window.ax.lines[0].get_ydata(), raw[0] - 3.0 - bias
            )
            window._redraw()
            window._toggle()
            self.assertEqual(len(calibration.correct_calls), 2)
            np.testing.assert_array_equal(window.spec, original)

            replacement = np.array([[50, 51, 52, 53, 54]], dtype=float)
            window.update_spectrum(replacement)
            self.assertEqual(len(calibration.correct_calls), 3)
            np.testing.assert_array_equal(window.spec, replacement)
            np.testing.assert_allclose(
                window.ax.lines[0].get_ydata(), replacement[0] - 3.0 - bias
            )
        finally:
            window.close()

    def test_selected_repeat_uses_cached_cosmic_correction_then_bias(self):
        raw = np.array(
            [[10, 11, 12, 13, 14], [30, 31, 32, 33, 34]], dtype=float
        )
        bias = np.arange(5, dtype=float)
        calibration = _FakeCosmicRayCalibration(offset=2.0)
        window = SpectrumWindow(
            raw,
            spectral_bias=bias,
            cosmic_ray_calibration=calibration,
        )
        try:
            window.cosmic_ray_removal_check.setChecked(True)
            window.remove_spectral_bias_check.setChecked(True)
            window.repeat_slider.setValue(2)

            np.testing.assert_allclose(
                window.ax.lines[0].get_ydata(), raw[1] - 2.0 - bias
            )
            self.assertEqual(len(calibration.correct_calls), 2)

            window.repeat_slider.setValue(1)

            np.testing.assert_allclose(
                window.ax.lines[0].get_ydata(), raw[0] - 2.0 - bias
            )
            self.assertEqual(len(calibration.correct_calls), 2)
        finally:
            window.close()

    def test_dark_only_fvb_preserves_repeatable_peak_and_removes_one_off(self):
        raw = np.array(
            [
                [10.0, 11.0, 100.0, 13.0, 14.0],
                [10.0, 12.0, 101.0, 13.0, 1000.0],
                [11.0, 11.0, 99.0, 14.0, 15.0],
            ]
        )
        original = raw.copy()
        calibration = _DarkOnlyPeakCalibration(raw.shape[1])
        window = SpectrumWindow(raw, cosmic_ray_calibration=calibration)
        try:
            self.assertTrue(window.cosmic_ray_removal_check.isEnabled())
            window.cosmic_ray_removal_check.setChecked(True)

            corrected = window._cosmic_ray_corrected_cache
            np.testing.assert_array_equal(corrected[:, 2], raw[:, 2])
            self.assertEqual(corrected[1, -1], 10.0)
            np.testing.assert_array_equal(corrected[[0, 2], -1], raw[[0, 2], -1])
            np.testing.assert_array_equal(window.spec, original)
            self.assertIn("corrected 1 detector value", window.cosmic_ray_status_label.text())
            self.assertIn("Preserved 3", window.cosmic_ray_status_label.text())
        finally:
            window.close()

    def test_dark_only_fvb_repeat_guard_handles_two_repeats(self):
        raw = np.array(
            [
                [10.0, 11.0, 100.0, 13.0, 14.0],
                [10.0, 12.0, 101.0, 13.0, 1000.0],
            ]
        )
        calibration = _DarkOnlyPeakCalibration(raw.shape[1])
        window = SpectrumWindow(raw, cosmic_ray_calibration=calibration)
        try:
            window.cosmic_ray_removal_check.setChecked(True)
            corrected = window._cosmic_ray_corrected_cache
            np.testing.assert_array_equal(corrected[:, 2], raw[:, 2])
            self.assertEqual(corrected[1, -1], 10.0)
        finally:
            window.close()

    def test_dark_only_fvb_is_disabled_for_one_spectrum(self):
        calibration = _DarkOnlyPeakCalibration(5)
        window = SpectrumWindow(
            np.arange(5, dtype=float),
            cosmic_ray_calibration=calibration,
        )
        try:
            self.assertFalse(window.cosmic_ray_removal_check.isEnabled())
            self.assertIn(
                "at least two repeats",
                window.cosmic_ray_removal_check.toolTip(),
            )
            self.assertIn(
                "at least two repeats", window.cosmic_ray_status_label.text()
            )
        finally:
            window.close()

    def test_detector_cosmic_ray_cache_covers_image_and_row_sum_updates(self):
        frames = np.arange(30, dtype=float).reshape(2, 3, 5)
        calibration = _FakeCosmicRayCalibration(offset=1.0)
        window = DetectorImageWindow(
            frames, cosmic_ray_calibration=calibration
        )
        try:
            window.cosmic_ray_removal_check.setChecked(True)
            self.assertEqual(len(calibration.correct_calls), 2)
            self.assertIn(
                "corrected 30 detector values across 2 frames",
                window.cosmic_ray_status_label.text(),
            )
            window._redraw()
            window._toggle_view()
            self.assertEqual(len(calibration.correct_calls), 2)
            np.testing.assert_allclose(
                window.ax.lines[0].get_ydata(),
                (frames.mean(axis=0) - 1.0).sum(axis=0),
            )
            window.repeat_slider.setValue(2)
            np.testing.assert_allclose(
                window.ax.lines[0].get_ydata(),
                (frames[1] - 1.0).sum(axis=0),
            )
            self.assertEqual(len(calibration.correct_calls), 2)

            replacement = np.full((1, 3, 5), 10.0)
            window.update_frames(replacement)
            self.assertEqual(len(calibration.correct_calls), 3)
            np.testing.assert_array_equal(window.raw_frames, replacement)
            np.testing.assert_allclose(
                window.ax.lines[0].get_ydata(),
                np.full(5, 27.0),
            )
        finally:
            window.close()

    def test_cosmic_ray_failure_disables_control_without_changing_raw(self):
        raw = np.arange(5, dtype=float).reshape(1, -1)
        calibration = _FakeCosmicRayCalibration(
            failure=RuntimeError("model could not process this spectrum")
        )
        window = SpectrumWindow(raw, cosmic_ray_calibration=calibration)
        try:
            window.cosmic_ray_removal_check.setChecked(True)

            self.assertFalse(window.cosmic_ray_removal_check.isChecked())
            self.assertFalse(window.cosmic_ray_removal_check.isEnabled())
            self.assertIn(
                "correction failed", window.cosmic_ray_removal_check.toolTip()
            )
            self.assertIn("model could not process", window._cosmic_ray_error)
            np.testing.assert_array_equal(window.spec, raw)
            np.testing.assert_allclose(window.ax.lines[0].get_ydata(), raw[0])
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
