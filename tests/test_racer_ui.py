"""Session RACER calibration controls and plot wiring tests."""

from __future__ import annotations

import ast
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from qtpy.QtWidgets import QApplication, QComboBox, QLabel, QVBoxLayout, QWidget

from napari_raman_widget import racer_ui
from napari_raman_widget.racer_preferences import (
    load_calibration_options,
    save_calibration_options,
)
from napari_raman_widget.cosmic_ray_removal import RacerCalibration
from napari_raman_widget.plot_windows import SpectrumWindow


PACKAGE = Path(__file__).resolve().parents[1] / "napari_raman_widget"


class _Calibration:
    def __init__(
        self,
        read_mode="fvb",
        compatible=True,
        reason="matched",
        reference_path=Path("reference.zarr"),
    ):
        self.read_mode = read_mode
        self.exposure_ms = 100.0
        self.detector_shape = (4,) if read_mode == "fvb" else (4, 5)
        self.open_reference_path = reference_path
        self.source_paths = (Path("closed.zarr"), reference_path)
        self.gain_e_per_adu = 0.7
        self.saturation_adu = 65535.0
        self.metadata_warnings = ()
        self.compatible = compatible
        self.reason = reason
        self.calls = []

    def compatibility(self, values, *, exposure_ms, read_mode):
        self.calls.append((np.asarray(values).shape, exposure_ms, read_mode))
        return self.compatible, self.reason


class _Owner(QWidget):
    def __init__(self):
        super().__init__()
        self.status = QLabel("Status: unchanged", self)
        self.collect_read_mode_combo = QComboBox(self)
        self.collect_read_mode_combo.addItem("FVB", "fvb")
        self.collect_read_mode_combo.addItem("Image", "image")
        self.collect_read_mode_combo.addItem("Single track", "single_track")
        self.collector = object()
        self._plot_windows = []
        self.layout = QVBoxLayout(self)
        racer_ui.add_racer_calibration_controls(self, self.layout)

    def select_mode(self, mode):
        index = self.collect_read_mode_combo.findData(mode)
        self.collect_read_mode_combo.setCurrentIndex(index)


class RacerCalibrationUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        load_preferences = patch.object(
            racer_ui, "load_calibration_options", return_value={}
        )
        save_preferences = patch.object(racer_ui, "save_calibration_options")
        self.load_preferences = load_preferences.start()
        self.save_preferences = save_preferences.start()
        self.addCleanup(load_preferences.stop)
        self.addCleanup(save_preferences.stop)
        load_paths = patch.object(racer_ui, "load_calibration_paths", return_value={})
        save_paths = patch.object(racer_ui, "save_calibration_paths")
        self.load_paths = load_paths.start()
        self.save_paths = save_paths.start()
        self.addCleanup(load_paths.stop)
        self.addCleanup(save_paths.stop)
        self.owner = _Owner()
        self.addCleanup(self.owner.close)

    def test_controls_start_with_session_only_unloaded_state(self):
        self.assertIsNone(self.owner.racer_calibration)
        self.assertEqual(
            self.owner.load_racer_calibration_btn.text(),
            "Load RACER calibration\N{HORIZONTAL ELLIPSIS}",
        )
        self.assertEqual(self.owner.clear_racer_calibration_btn.text(), "Clear")
        self.assertFalse(self.owner.clear_racer_calibration_btn.isEnabled())
        self.assertIn(
            "session-only",
            self.owner.load_racer_calibration_btn.toolTip(),
        )
        self.assertIn(
            "optional for RACER-1D",
            self.owner.load_racer_calibration_btn.toolTip(),
        )

    def test_cancelling_first_directory_cancels_the_load(self):
        with patch.object(
            racer_ui.QFileDialog,
            "getExistingDirectory",
            return_value="",
        ) as choose, patch.object(
            racer_ui,
            "load_racer_calibration",
        ) as load:
            result = racer_ui.load_racer_calibration_from_dialogs(self.owner)

        self.assertIsNone(result)
        choose.assert_called_once()
        load.assert_not_called()
        self.assertIn("loading cancelled", self.owner.status.text())

    def test_settings_button_opens_dialog_on_request(self):
        self.assertEqual(self.owner.racer_calibration_options, {})
        with patch.object(racer_ui, "show_racer_calibration_settings") as show:
            self.owner.racer_calibration_settings_btn.click()
        show.assert_called_once_with(self.owner)

    def test_saving_settings_changes_next_load_without_replacing_active_fit(self):
        active = _Calibration()
        self.owner.racer_calibration = active
        options = {"image": {"trace_half_width_rows": 64}}
        with patch.object(racer_ui, "RacerCalibrationSettingsDialog") as make_dialog:
            dialog = make_dialog.return_value
            dialog.exec_.return_value = racer_ui.QDialog.Accepted
            dialog.calibration_options = options
            racer_ui.show_racer_calibration_settings(self.owner)
        self.assertEqual(self.owner.racer_calibration_options, options)
        self.assertIs(self.owner.racer_calibration, active)
        self.save_preferences.assert_called_once_with(options)
        self.assertIn("future sessions", self.owner.status.text())
        self.assertIn("load a calibration", self.owner.status.text())
        self.assertIn("trace_half_width_rows=64", self.owner.racer_calibration_settings_btn.toolTip())

        with patch.object(racer_ui.QFileDialog, "getExistingDirectory",
                          side_effect=["closed.zarr", "reference.zarr"]), patch.object(
            racer_ui, "load_racer_calibration", return_value=active
        ) as load:
            racer_ui.load_racer_calibration_from_dialogs(self.owner)
        load.assert_called_once_with(
            Path("closed.zarr"), Path("reference.zarr"), calibration_options=options
        )

    def test_cancel_settings_keeps_saved_choices(self):
        self.owner.racer_calibration_options = {"image": {"trace_half_width_rows": 64}}
        with patch.object(racer_ui, "RacerCalibrationSettingsDialog") as make_dialog:
            make_dialog.return_value.exec_.return_value = racer_ui.QDialog.Rejected
            make_dialog.return_value.calibration_options = {}
            racer_ui.show_racer_calibration_settings(self.owner)
        self.assertEqual(self.owner.racer_calibration_options, {"image": {"trace_half_width_rows": 64}})
        self.assertEqual(self.owner.status.text(), "Status: unchanged")
        self.save_preferences.assert_not_called()

    def test_saved_values_survive_recreating_widget_and_reset_survives_too(self):
        options = {"image": {"trace_half_width_rows": 64},
                   "spectrum_dark": {"temporal_iterations": 3}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "racer-settings.json"
            self.load_preferences.side_effect = lambda: load_calibration_options(path)
            self.save_preferences.side_effect = (
                lambda values: save_calibration_options(values, path)
            )
            with patch.object(racer_ui, "RacerCalibrationSettingsDialog") as make:
                dialog = make.return_value
                dialog.exec_.return_value = racer_ui.QDialog.Accepted
                dialog.calibration_options = options
                racer_ui.show_racer_calibration_settings(self.owner)
                reopened = _Owner()
                self.addCleanup(reopened.close)
                self.assertEqual(reopened.racer_calibration_options, options)
                self.assertIsNone(reopened.racer_calibration)
                self.assertFalse(reopened.clear_racer_calibration_btn.isEnabled())
                self.assertIn("trace_half_width_rows=64",
                              reopened.racer_calibration_settings_btn.toolTip())

                # Restored options are used without reopening the settings dialog.
                with patch.object(racer_ui.QFileDialog, "getExistingDirectory",
                                  side_effect=["closed.zarr", "reference.zarr"]), patch.object(
                    racer_ui, "load_racer_calibration", return_value=_Calibration()
                ) as load:
                    racer_ui.load_racer_calibration_from_dialogs(reopened)
                load.assert_called_once_with(
                    Path("closed.zarr"), Path("reference.zarr"),
                    calibration_options=options,
                )
                dialog.calibration_options = {}
                racer_ui.show_racer_calibration_settings(reopened)
                reset = _Owner()
                self.addCleanup(reset.close)
                self.assertEqual(reset.racer_calibration_options, {})
                self.assertEqual(load_calibration_options(path), {})

    def test_corrupt_preferences_do_not_prevent_widget_startup(self):
        self.load_preferences.side_effect = ValueError("invalid JSON")
        owner = _Owner()
        self.addCleanup(owner.close)
        self.assertEqual(owner.racer_calibration_options, {})
        self.assertIn("could not be read", owner.racer_calibration_settings_btn.toolTip())
        with patch.object(racer_ui, "RacerCalibrationSettingsDialog") as make:
            make.return_value.exec_.return_value = racer_ui.QDialog.Rejected
            racer_ui.show_racer_calibration_settings(owner)
        self.assertIn("invalid JSON", make.call_args.kwargs["settings_warning"])
        self.save_preferences.assert_not_called()

    def test_save_error_reports_session_only_values_and_can_retry(self):
        active = _Calibration()
        self.owner.racer_calibration = active
        options = {"image": {"trace_half_width_rows": 64}}
        self.save_preferences.side_effect = OSError("disk is read-only")
        with patch.object(racer_ui, "RacerCalibrationSettingsDialog") as make:
            make.return_value.exec_.return_value = racer_ui.QDialog.Accepted
            make.return_value.calibration_options = options
            racer_ui.show_racer_calibration_settings(self.owner)
            self.assertEqual(self.owner.racer_calibration_options, options)
            self.assertIs(self.owner.racer_calibration, active)
            self.assertIn("could not be saved", self.owner.status.text())
            self.assertIn("disk is read-only", self.owner.racer_calibration_settings_btn.toolTip())
            self.save_preferences.side_effect = None
            racer_ui.show_racer_calibration_settings(self.owner)
        self.assertIn("saved for future sessions", self.owner.status.text())
        self.assertNotIn("read-only", self.owner.racer_calibration_settings_btn.toolTip())

    def test_settings_import_failure_is_reported_without_changing_calibration(self):
        active = _Calibration()
        self.owner.racer_calibration = active
        with patch.object(racer_ui, "RacerCalibrationSettingsDialog",
                          side_effect=RuntimeError("RACER is not installed")):
            racer_ui.show_racer_calibration_settings(self.owner)
        self.assertIn("settings unavailable", self.owner.status.text())
        self.assertIs(self.owner.racer_calibration, active)

    def test_loaded_tooltip_reports_applied_settings(self):
        calibration = _Calibration(read_mode="image")
        calibration.calibration_options = {"image": {"trace_half_width_rows": 64}}
        self.assertIn("trace_half_width_rows=64", racer_ui._loaded_tooltip(calibration))

    def test_omitting_open_reference_loads_dark_only_fvb(self):
        calibration = _Calibration(reference_path=None)
        calibration.metadata_warnings = (
            "signal-dependent shot noise is not modeled",
        )
        with patch.object(
            racer_ui.QFileDialog,
            "getExistingDirectory",
            side_effect=["closed.zarr", ""],
        ), patch.object(
            racer_ui,
            "load_racer_calibration",
            return_value=calibration,
        ) as load:
            result = racer_ui.load_racer_calibration_from_dialogs(self.owner)

        self.assertIs(result, calibration)
        load.assert_called_once_with(Path("closed.zarr"), None)
        self.assertIs(self.owner.racer_calibration, calibration)
        self.assertIn(
            "closed-shutter variance only",
            self.owner.load_racer_calibration_btn.toolTip(),
        )
        self.assertIn(
            "signal-dependent shot noise is not modeled",
            self.owner.status.text(),
        )

    def test_omitting_reference_for_image_surfaces_loader_error(self):
        previous = _Calibration()
        self.owner.racer_calibration = previous
        self.owner.clear_racer_calibration_btn.setEnabled(True)
        with patch.object(
            racer_ui.QFileDialog,
            "getExistingDirectory",
            side_effect=["closed-image.zarr", ""],
        ), patch.object(
            racer_ui,
            "load_racer_calibration",
            side_effect=ValueError(
                "an illuminated open-reference acquisition is required for "
                "RACER-2D Image calibration"
            ),
        ) as load:
            result = racer_ui.load_racer_calibration_from_dialogs(self.owner)

        self.assertIsNone(result)
        load.assert_called_once_with(Path("closed-image.zarr"), None)
        self.assertIs(self.owner.racer_calibration, previous)
        self.assertIn("required for RACER-2D", self.owner.status.text())

    def test_calibration_mode_is_detected_without_collect_mode_control(self):
        del self.owner.collect_read_mode_combo
        calibration = _Calibration(read_mode="fvb")
        with patch.object(
            racer_ui.QFileDialog,
            "getExistingDirectory",
            side_effect=["closed.zarr", "reference.zarr"],
        ), patch.object(
            racer_ui,
            "load_racer_calibration",
            return_value=calibration,
        ) as load:
            result = racer_ui.load_racer_calibration_from_dialogs(self.owner)

        self.assertIs(result, calibration)
        load.assert_called_once_with(Path("closed.zarr"), Path("reference.zarr"))

    def test_two_directories_load_and_clear_one_session_calibration(self):
        closed = Path("C:/data/closed.zarr")
        reference = Path("C:/data/reference.zarr")
        calibration = _Calibration()
        with patch.object(
            racer_ui.QFileDialog,
            "getExistingDirectory",
            side_effect=[str(closed), str(reference)],
        ) as choose, patch.object(
            racer_ui,
            "load_racer_calibration",
            return_value=calibration,
        ) as load:
            self.owner.load_racer_calibration_btn.click()

        self.assertEqual(choose.call_count, 2)
        load.assert_called_once_with(closed, reference)
        self.assertIs(self.owner.racer_calibration, calibration)
        self.assertTrue(self.owner.clear_racer_calibration_btn.isEnabled())
        self.assertIn("closed.zarr", self.owner.load_racer_calibration_btn.toolTip())
        self.assertIn("loaded for this session", self.owner.status.text())

        self.owner.clear_racer_calibration_btn.click()
        self.assertIsNone(self.owner.racer_calibration)
        self.assertFalse(self.owner.clear_racer_calibration_btn.isEnabled())
        self.assertIn("cleared", self.owner.status.text())

    def test_load_error_preserves_the_previous_calibration(self):
        previous = _Calibration()
        self.owner.racer_calibration = previous
        self.owner.clear_racer_calibration_btn.setEnabled(True)
        with patch.object(
            racer_ui.QFileDialog,
            "getExistingDirectory",
            side_effect=["closed.zarr", "reference.zarr"],
        ), patch.object(
            racer_ui,
            "load_racer_calibration",
            side_effect=ValueError("not enough frames"),
        ):
            result = racer_ui.load_racer_calibration_from_dialogs(self.owner)

        self.assertIsNone(result)
        self.assertIs(self.owner.racer_calibration, previous)
        self.assertTrue(self.owner.clear_racer_calibration_btn.isEnabled())
        self.assertIn("not enough frames", self.owner.status.text())
        self.assertIn(
            "not enough frames",
            self.owner.load_racer_calibration_btn.toolTip(),
        )
        self.assertIn(
            "closed.zarr", self.owner.load_racer_calibration_btn.toolTip()
        )

    def test_fvb_load_ignores_current_image_collect_mode(self):
        self.owner.select_mode("image")
        calibration = _Calibration(read_mode="fvb")
        with patch.object(
            racer_ui.QFileDialog,
            "getExistingDirectory",
            side_effect=["closed.zarr", "reference.zarr"],
        ), patch.object(
            racer_ui,
            "load_racer_calibration",
            return_value=calibration,
        ):
            result = racer_ui.load_racer_calibration_from_dialogs(self.owner)

        self.assertIs(result, calibration)
        self.assertIs(self.owner.racer_calibration, calibration)
        self.assertIn("fvb", self.owner.load_racer_calibration_btn.toolTip())

    def test_unsigned_fvb_calibration_explains_open_reference_role(self):
        calibration = RacerCalibration(
            read_mode="fvb",
            exposure_ms=100.0,
            detector_shape=(4,),
            detector_columns=(0, 1, 2, 3),
            detector_rows=None,
            gain_e_per_adu=0.7,
            saturation_adu=65535.0,
            detector_signature=None,
            metadata_warnings=(),
            closed_shutter_path=Path("closed.zarr"),
            open_reference_path=Path("reference.zarr"),
            _calibrator=Mock(),
        )
        with patch.object(
            racer_ui.QFileDialog,
            "getExistingDirectory",
            side_effect=["closed.zarr", "reference.zarr"],
        ), patch.object(
            racer_ui,
            "load_racer_calibration",
            return_value=calibration,
        ):
            result = racer_ui.load_racer_calibration_from_dialogs(self.owner)

        self.assertIsNone(result)
        message = self.owner.status.text()
        self.assertIn("RACER-1D", message)
        self.assertIn("illuminated reference spectrum", message)
        self.assertIn("variance at its signal level", message)
        self.assertIn("does not use a 2D spatial profile", message)

    def test_unsigned_dark_only_fvb_requests_a_new_closed_stack(self):
        calibration = RacerCalibration(
            read_mode="fvb",
            exposure_ms=100.0,
            detector_shape=(4,),
            detector_columns=(0, 1, 2, 3),
            detector_rows=None,
            gain_e_per_adu=0.7,
            saturation_adu=65535.0,
            detector_signature=None,
            metadata_warnings=("closed-shutter variance only",),
            closed_shutter_path=Path("closed.zarr"),
            open_reference_path=None,
            _calibrator=Mock(),
        )
        with patch.object(
            racer_ui.QFileDialog,
            "getExistingDirectory",
            side_effect=["closed.zarr", ""],
        ), patch.object(
            racer_ui,
            "load_racer_calibration",
            return_value=calibration,
        ):
            result = racer_ui.load_racer_calibration_from_dialogs(self.owner)

        self.assertIsNone(result)
        message = self.owner.status.text()
        self.assertIn("closed-shutter store", message)
        self.assertIn("recollect the closed-shutter stack", message)
        self.assertNotIn("requires an illuminated reference", message)

    def test_matching_helper_flattens_only_leading_acquisition_dimensions(self):
        cases = (
            ("fvb", np.zeros((2, 3, 4)), (6, 4)),
            ("image", np.zeros((2, 3, 4, 5)), (6, 4, 5)),
        )
        for read_mode, values, expected_shape in cases:
            with self.subTest(read_mode=read_mode):
                calibration = _Calibration(read_mode=read_mode)
                self.owner.racer_calibration = calibration
                result = racer_ui.matching_racer_calibration(
                    self.owner,
                    values,
                    100.0,
                    read_mode,
                )
                self.assertIs(result, calibration)
                self.assertEqual(
                    calibration.calls,
                    [(expected_shape, 100.0, read_mode)],
                )

    def test_incompatible_calibration_is_not_attached_to_a_new_plot(self):
        calibration = _Calibration(compatible=False, reason="exposure mismatch")
        self.owner.racer_calibration = calibration

        result = racer_ui.matching_racer_calibration(
            self.owner,
            np.zeros((2, 4)),
            250.0,
            "fvb",
        )

        self.assertIsNone(result)
        self.assertIn(
            "exposure mismatch",
            self.owner.load_racer_calibration_btn.toolTip(),
        )

    def test_load_and_clear_refresh_already_open_compatible_plots(self):
        panel = SpectrumWindow(np.arange(4.0), exposure_ms=100.0)
        self.addCleanup(panel.close)
        self.owner._plot_windows.append(panel)
        self.assertFalse(panel.cosmic_ray_removal_check.isEnabled())

        calibration = _Calibration()
        with patch.object(
            racer_ui.QFileDialog,
            "getExistingDirectory",
            side_effect=["closed.zarr", "reference.zarr"],
        ), patch.object(
            racer_ui,
            "load_racer_calibration",
            return_value=calibration,
        ):
            result = racer_ui.load_racer_calibration_from_dialogs(self.owner)

        self.assertIs(result, calibration)
        self.assertIs(panel.cosmic_ray_calibration, calibration)
        self.assertTrue(panel.cosmic_ray_removal_check.isEnabled())
        self.assertIn("1/1 open spectral plots", self.owner.status.text())

        racer_ui.clear_racer_calibration(self.owner)
        self.assertIsNone(panel.cosmic_ray_calibration)
        self.assertFalse(panel.cosmic_ray_removal_check.isEnabled())
        self.assertFalse(panel.cosmic_ray_removal_check.isChecked())


class RacerPlotWiringTests(unittest.TestCase):
    @staticmethod
    def calls(filename, names):
        tree = ast.parse((PACKAGE / filename).read_text(encoding="utf-8"))
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in names
        ]

    def test_collect_widgets_pass_session_calibration_to_direct_plots_only(self):
        direct_names = {"DetectorImageWindow", "SpectrumWindow"}
        for filename in ("hardware_widget.py", "demo_widget.py"):
            with self.subTest(filename=filename):
                source = (PACKAGE / filename).read_text(encoding="utf-8")
                self.assertIn(
                    "add_racer_calibration_controls(self, loading_layout)", source
                )
                direct_calls = self.calls(filename, direct_names)
                wired_calls = [
                    call
                    for call in direct_calls
                    if "cosmic_ray_calibration"
                    in {item.arg for item in call.keywords}
                ]
                self.assertEqual(len(wired_calls), 4)
                for call in wired_calls:
                    keywords = {item.arg for item in call.keywords}
                    self.assertIn("exposure_ms", keywords)

                arbitrary_calls = self.calls(filename, {"DatasetViewerWindow"})
                self.assertTrue(arbitrary_calls)
                for call in arbitrary_calls:
                    keywords = {item.arg for item in call.keywords}
                    self.assertNotIn("cosmic_ray_calibration", keywords)
                    self.assertNotIn("exposure_ms", keywords)

    def test_scan_result_plots_receive_known_exposure_and_calibration(self):
        calls = self.calls(
            "scan_workflows.py",
            {
                "CalibrationPlotWindow",
                "ReferenceSpectraWindow",
                "GridScanPlotWindow",
            },
        )
        self.assertEqual(len(calls), 3)
        for call in calls:
            keywords = {item.arg for item in call.keywords}
            self.assertIn("cosmic_ray_calibration", keywords)
            self.assertIn("exposure_ms", keywords)


if __name__ == "__main__":
    unittest.main()
