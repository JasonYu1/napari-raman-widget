"""Remember calibration file choices while keeping calibration loading manual."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication

from napari_raman_widget import racer_preferences, racer_ui
from test_racer_ui import _Calibration, _Owner


class RacerRememberedPathsUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        for attribute, return_value in (
            ("load_calibration_options", {}),
            ("save_calibration_options", None),
            ("load_calibration_paths", {}),
            ("save_calibration_paths", None),
        ):
            patcher = patch.object(racer_ui, attribute, return_value=return_value)
            setattr(self, attribute, patcher.start())
            self.addCleanup(patcher.stop)

    def owner(self):
        owner = _Owner()
        self.addCleanup(owner.close)
        return owner

    def remembered(self, *, reference=True):
        closed = str(Path("remembered-dark.zarr").resolve())
        opened = str(Path("remembered-reference.zarr").resolve()) if reference else None
        self.load_calibration_paths.return_value = {
            "closed_shutter": closed, "open_reference": opened,
        }
        return closed, opened

    def test_startup_restores_fields_without_fitting_or_opening_file_choosers(self):
        closed, opened = self.remembered()
        with patch.object(racer_ui, "load_racer_calibration") as fit, patch.object(
            racer_ui.QFileDialog, "getExistingDirectory"
        ) as choose, patch(
            "napari_raman_widget.cosmic_ray_removal._import_racer"
        ) as import_racer:
            owner = self.owner()
        self.assertEqual(owner.racer_closed_shutter_path.text(), closed)
        self.assertEqual(owner.racer_open_reference_path.text(), opened)
        self.assertIsNone(owner.racer_calibration)
        self.assertFalse(owner.clear_racer_calibration_btn.isEnabled())
        fit.assert_not_called()
        choose.assert_not_called()
        import_racer.assert_not_called()
        self.save_calibration_paths.assert_not_called()

    def test_manual_load_reuses_both_remembered_paths_without_choosers(self):
        closed, opened = self.remembered()
        owner = self.owner()
        calibration = _Calibration()
        with patch.object(racer_ui, "load_racer_calibration", return_value=calibration) as fit, patch.object(
            racer_ui.QFileDialog, "getExistingDirectory"
        ) as choose:
            owner.load_racer_calibration_btn.click()
        fit.assert_called_once_with(Path(closed), Path(opened))
        choose.assert_not_called()
        self.assertIs(owner.racer_calibration, calibration)

    def test_manual_load_preserves_remembered_dark_only_choice(self):
        closed, _ = self.remembered(reference=False)
        owner = self.owner()
        calibration = _Calibration(reference_path=None)
        with patch.object(racer_ui, "load_racer_calibration", return_value=calibration) as fit, patch.object(
            racer_ui.QFileDialog, "getExistingDirectory"
        ) as choose:
            owner.load_racer_calibration_btn.click()
        fit.assert_called_once_with(Path(closed), None)
        choose.assert_not_called()
        self.assertEqual(owner.racer_open_reference_path.text(), "")

    def test_browsing_each_path_remembers_choice_without_replacing_active_fit(self):
        owner = self.owner()
        active = _Calibration()
        owner.racer_calibration = active
        closed = str(Path("new-dark.zarr").resolve())
        opened = str(Path("new-reference.zarr").resolve())
        with patch.object(racer_ui, "load_racer_calibration") as fit, patch.object(
            racer_ui.QFileDialog, "getExistingDirectory", side_effect=[closed, opened]
        ):
            owner.browse_closed_shutter_racer_btn.click()
            self.save_calibration_paths.assert_called_with(closed, None)
            owner.browse_open_reference_racer_btn.click()
        self.save_calibration_paths.assert_called_with(closed, opened)
        self.assertEqual(owner.racer_closed_shutter_path.text(), closed)
        self.assertEqual(owner.racer_open_reference_path.text(), opened)
        self.assertIs(owner.racer_calibration, active)
        fit.assert_not_called()

    def test_browse_cancel_preserves_remembered_selection_and_does_not_save(self):
        closed, opened = self.remembered()
        owner = self.owner()
        with patch.object(racer_ui.QFileDialog, "getExistingDirectory", return_value=""), patch.object(
            racer_ui, "load_racer_calibration"
        ) as fit:
            owner.browse_closed_shutter_racer_btn.click()
            owner.browse_open_reference_racer_btn.click()
        self.assertEqual(owner.racer_closed_shutter_path.text(), closed)
        self.assertEqual(owner.racer_open_reference_path.text(), opened)
        self.save_calibration_paths.assert_not_called()
        fit.assert_not_called()

    def test_editing_or_clearing_reference_is_saved_without_loading(self):
        closed, _ = self.remembered()
        owner = self.owner()
        with patch.object(racer_ui, "load_racer_calibration") as fit:
            owner.racer_open_reference_path.clear()
            owner.racer_open_reference_path.editingFinished.emit()
            self.save_calibration_paths.assert_called_with(closed, None)
            owner.racer_closed_shutter_path.clear()
            owner.racer_closed_shutter_path.editingFinished.emit()
        self.save_calibration_paths.assert_called_with(None, None)
        fit.assert_not_called()

    def test_saved_choices_survive_reopening_and_settings_save_keeps_them(self):
        with tempfile.TemporaryDirectory() as directory:
            files_path = Path(directory) / "files.json"
            settings_path = Path(directory) / "settings.json"
            self.load_calibration_paths.side_effect = (
                lambda: racer_preferences.load_calibration_paths(files_path)
            )
            self.save_calibration_paths.side_effect = (
                lambda closed, opened: racer_preferences.save_calibration_paths(
                    closed, opened, files_path
                )
            )
            self.save_calibration_options.side_effect = (
                lambda options: racer_preferences.save_calibration_options(options, settings_path)
            )
            owner = self.owner()
            closed = str(Path(directory) / "dark.zarr")
            opened = str(Path(directory) / "reference.zarr")
            owner.racer_closed_shutter_path.setText(closed)
            owner.racer_open_reference_path.setText(opened)
            owner.racer_open_reference_path.editingFinished.emit()
            with patch.object(racer_ui, "RacerCalibrationSettingsDialog") as make:
                make.return_value.exec_.return_value = racer_ui.QDialog.Accepted
                make.return_value.calibration_options = {"image": {"trace_half_width_rows": 64}}
                racer_ui.show_racer_calibration_settings(owner)
            with patch.object(racer_ui, "load_racer_calibration") as fit:
                reopened = self.owner()
            self.assertEqual(reopened.racer_closed_shutter_path.text(), closed)
            self.assertEqual(reopened.racer_open_reference_path.text(), opened)
            self.assertIsNone(reopened.racer_calibration)
            fit.assert_not_called()
            self.assertEqual(racer_preferences.load_calibration_options(settings_path), {
                "image": {"trace_half_width_rows": 64},
            })

    def test_clear_active_calibration_keeps_paths_for_next_manual_load(self):
        closed, opened = self.remembered()
        owner = self.owner()
        owner.racer_calibration = _Calibration()
        owner.clear_racer_calibration_btn.setEnabled(True)
        owner.clear_racer_calibration_btn.click()
        self.assertIsNone(owner.racer_calibration)
        self.assertEqual(owner.racer_closed_shutter_path.text(), closed)
        self.assertEqual(owner.racer_open_reference_path.text(), opened)
        self.save_calibration_paths.assert_not_called()

    def test_missing_remembered_store_fails_only_on_manual_load_without_rechoosing(self):
        closed, opened = self.remembered()
        with patch.object(racer_ui, "load_racer_calibration") as fit, patch.object(
            racer_ui.QFileDialog, "getExistingDirectory"
        ) as choose:
            owner = self.owner()
            fit.assert_not_called()
            previous = _Calibration()
            owner.racer_calibration = previous
            fit.side_effect = FileNotFoundError("remembered reference moved")
            owner.load_racer_calibration_btn.click()
        fit.assert_called_once_with(Path(closed), Path(opened))
        choose.assert_not_called()
        self.assertIs(owner.racer_calibration, previous)
        self.assertEqual(owner.racer_open_reference_path.text(), opened)
        self.assertIn("remembered reference moved", owner.status.text())

    def test_initial_load_cancel_does_not_save_or_load_anything(self):
        owner = self.owner()
        with patch.object(racer_ui.QFileDialog, "getExistingDirectory", return_value=""), patch.object(
            racer_ui, "load_racer_calibration"
        ) as fit:
            owner.load_racer_calibration_btn.click()
        fit.assert_not_called()
        self.save_calibration_paths.assert_not_called()

    def test_first_manual_load_chooses_missing_closed_path_and_keeps_existing_reference(self):
        opened = str(Path("reference-first.zarr").resolve())
        self.load_calibration_paths.return_value = {
            "closed_shutter": None, "open_reference": opened,
        }
        owner = self.owner()
        closed = str(Path("then-dark.zarr").resolve())
        with patch.object(racer_ui.QFileDialog, "getExistingDirectory", return_value=closed) as choose, patch.object(
            racer_ui, "load_racer_calibration", return_value=_Calibration()
        ) as fit:
            owner.load_racer_calibration_btn.click()
        choose.assert_called_once()
        fit.assert_called_once_with(Path(closed), Path(opened))
        self.save_calibration_paths.assert_called_with(closed, opened)


if __name__ == "__main__":
    unittest.main()
