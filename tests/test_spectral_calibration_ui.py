"""Optional, explicitly selected spectral calibration UI."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from napari_raman_widget.spectral_calibration_ui import (
    add_spectral_calibration_loader,
    load_spectral_calibration,
)


class SpectralCalibrationUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.owner = QWidget()
        self.owner.status = QLabel()
        self.owner.browse_spectral_calibration = lambda: None
        add_spectral_calibration_loader(self.owner, QVBoxLayout(self.owner))

    def tearDown(self):
        self.owner.status.deleteLater()
        self.owner.deleteLater()

    def test_calibration_starts_unselected_and_optional(self):
        self.assertIsNone(self.owner.spectral_calibration)
        self.assertEqual(self.owner.spectral_calibration_path.text(), "")
        self.assertIn("None", self.owner.spectral_calibration_path.placeholderText())

    def test_clear_removes_selected_calibration(self):
        self.owner.spectral_calibration = object()
        self.owner.spectral_calibration_path.setText("selected.json")

        self.owner.clear_spectral_calibration_btn.click()

        self.assertIsNone(self.owner.spectral_calibration)
        self.assertEqual(self.owner.spectral_calibration_path.text(), "")

    def test_explicit_load_explains_pixel_default_and_opt_in(self):
        self.owner.spectral_calibration_path.setText("selected.json")
        calibration = SimpleNamespace(degree=1, pixel_positions=[0, 10])
        with patch(
            "napari_raman_widget.spectral_calibration_ui.load_pixel_to_wavenumber_calibration",
            return_value=calibration,
        ) as loader, patch(
            "napari_raman_widget.spectral_calibration_ui.QMessageBox.information"
        ) as information:
            result = load_spectral_calibration(self.owner)

        loader.assert_called_once_with("selected.json")
        self.assertIs(result, calibration)
        self.assertIs(self.owner.spectral_calibration, calibration)
        message = information.call_args.args[2]
        self.assertIn("start in pixels", message)
        self.assertIn("Show wavenumber", message)


if __name__ == "__main__":
    unittest.main()
