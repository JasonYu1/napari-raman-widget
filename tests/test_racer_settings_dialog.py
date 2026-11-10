"""The advanced dialog edits fit inputs without touching active calibrations."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QDialog

from napari_raman_widget.racer_settings_dialog import RacerCalibrationSettingsDialog


def _image(dark, reference, background, *, trace_half_width_rows=12,
           minimum_row_probability=1e-10, maximum_row_probability=0.05,
           event_seed_sigma=8.0, event_grow_sigma=3.0):
    pass


def _dark(stack, *, exposure_ms, temporal_iterations=2, seed_sigma=8.0,
          growth_sigma=3.0):
    pass


def _reference(stack, dark, *, minimum_sigma_adu=0.5):
    pass


class RacerSettingsDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        racer = SimpleNamespace(
            calibrate_image=_image,
            calibrate_spectrum_dark=_dark,
            calibrate_spectrum_reference=_reference,
        )
        patcher = patch(
            "napari_raman_widget.cosmic_ray_removal._import_racer", return_value=racer
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def dialog(self, options=None, **kwargs):
        dialog = RacerCalibrationSettingsDialog(options, **kwargs)
        self.addCleanup(dialog.close)
        return dialog

    def test_defaults_preserve_small_numbers_and_save_no_overrides(self):
        dialog = self.dialog(read_mode="image")
        self.assertEqual(dialog.tabs.currentIndex(), dialog._group_tabs["image"])
        field = dialog.editors[("image", "minimum_row_probability")]
        self.assertEqual(float(field.text()), 1e-10)
        dialog.accept()
        self.assertEqual(dialog.result(), QDialog.Accepted)
        self.assertEqual(dialog.calibration_options, {})

    def test_save_collects_independent_groups_without_mutating_input(self):
        original = {"image": {"trace_half_width_rows": 20}}
        dialog = self.dialog(original)
        dialog.editors[("image", "trace_half_width_rows")].setValue(64)
        dialog.editors[("image", "minimum_row_probability")].setText("1e-11")
        dialog.editors[("spectrum_dark", "temporal_iterations")].setValue(3)
        dialog.accept()
        self.assertEqual(dialog.result(), QDialog.Accepted)
        saved = dialog.calibration_options
        self.assertEqual(saved, {
            "image": {"trace_half_width_rows": 64, "minimum_row_probability": 1e-11},
            "spectrum_dark": {"temporal_iterations": 3},
        })
        self.assertEqual(original, {"image": {"trace_half_width_rows": 20}})
        saved["image"]["trace_half_width_rows"] = 1
        self.assertEqual(dialog.calibration_options["image"]["trace_half_width_rows"], 64)

    def test_restore_defaults_applies_only_after_save(self):
        original = {"image": {"trace_half_width_rows": 64}}
        cancelled = self.dialog(original)
        cancelled.restore_defaults()
        cancelled.reject()
        self.assertEqual(cancelled.calibration_options, original)
        accepted = self.dialog(original)
        accepted.restore_defaults()
        accepted.accept()
        self.assertEqual(accepted.calibration_options, {})

    def test_invalid_text_keeps_dialog_open_and_selects_affected_tab(self):
        dialog = self.dialog()
        dialog.editors[("image", "minimum_row_probability")].setText("NaN")
        dialog.accept()
        self.assertNotEqual(dialog.result(), QDialog.Accepted)
        self.assertIn("finite number", dialog.error_label.text())
        self.assertEqual(dialog.tabs.currentIndex(), dialog._group_tabs["image"])

    def test_invalid_threshold_relationship_does_not_save(self):
        dialog = self.dialog()
        dialog.editors[("image", "event_grow_sigma")].setText("9")
        dialog.accept()
        self.assertNotEqual(dialog.result(), QDialog.Accepted)
        self.assertIn("greater than", dialog.error_label.text())
        self.assertEqual(dialog.calibration_options, {})

    def test_obsolete_preferences_can_be_reset_after_racer_upgrade(self):
        original = {"image": {"removed_parameter": 123}}
        dialog = self.dialog(original)
        self.assertIn("incompatible", dialog.error_label.text())
        self.assertEqual(dialog.editors[("image", "trace_half_width_rows")].value(), 12)
        dialog.restore_defaults()
        dialog.accept()
        self.assertEqual(dialog.result(), QDialog.Accepted)
        self.assertEqual(dialog.calibration_options, {})
        self.assertEqual(original, {"image": {"removed_parameter": 123}})

    def test_preferences_read_error_is_visible_in_dialog(self):
        dialog = self.dialog(settings_warning="Saved settings could not be read.")
        self.assertIn("could not be read", dialog.error_label.text())
        dialog.reject()
        self.assertEqual(dialog.result(), QDialog.Rejected)


if __name__ == "__main__":
    unittest.main()
