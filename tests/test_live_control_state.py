"""Live acquisition restores controls hidden inside collapsed setup groups."""

import ast
import os
from pathlib import Path
from types import MethodType
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtWidgets import QApplication, QLineEdit, QPushButton, QVBoxLayout, QWidget

from napari_raman_widget.ui_helpers import make_collapsible


PACKAGE = Path(__file__).resolve().parents[1] / "napari_raman_widget"


class LiveControlStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def tearDown(self):
        self.doCleanups()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_collapsed_loading_controls_recover_after_live_acquisition(self):
        for filename in ("hardware_widget.py", "demo_widget.py"):
            with self.subTest(filename=filename):
                owner = QWidget()
                self.addCleanup(owner.deleteLater)
                self.addCleanup(owner.close)
                root_layout = QVBoxLayout(owner)
                owner.loading_box = make_collapsible("Loading")
                root_layout.addWidget(owner.loading_box)
                loading_layout = QVBoxLayout()
                configuration = make_collapsible("Configuration", expanded=False)
                loading_layout.addWidget(configuration)
                owner.loading_box.setLayout(loading_layout)
                configuration_layout = QVBoxLayout()
                configuration.setLayout(configuration_layout)

                for name in (
                    "load_racer_calibration_btn",
                    "clear_racer_calibration_btn",
                    "racer_calibration_settings_btn",
                ):
                    control = QPushButton(name)
                    setattr(owner, name, control)
                    configuration_layout.addWidget(control)
                owner.clear_racer_calibration_btn.setEnabled(False)
                owner.racer_calibration_path_controls = QWidget()
                paths_layout = QVBoxLayout(owner.racer_calibration_path_controls)
                path_field = QLineEdit("remembered-dark.zarr")
                paths_layout.addWidget(path_field)
                configuration_layout.addWidget(owner.racer_calibration_path_controls)

                for name in (
                    "calib_box", "grid_box", "sel_box", "exposure_input",
                    "n_input", "live_collect_check", "collect_read_mode_combo",
                    "collect_single_track_controls", "collect_save_input",
                ):
                    control = QWidget()
                    setattr(owner, name, control)
                    root_layout.addWidget(control)

                # Exercise the production method without initializing hardware.
                source = (PACKAGE / filename).read_text(encoding="utf-8")
                method = next(
                    node for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "_set_live_raman_controls"
                )
                namespace = {}
                exec(compile(ast.Module(body=[method], type_ignores=[]),
                             filename, "exec"), namespace)
                set_running = MethodType(namespace[method.name], owner)
                owner.show()
                self.app.processEvents()

                self.assertFalse(owner.load_racer_calibration_btn.isEnabled())
                set_running(True)
                self.assertFalse(owner.loading_box.isEnabled())
                set_running(False)
                self.assertFalse(configuration.isChecked())
                configuration.setChecked(True)
                self.app.processEvents()

                for control in (
                    owner.load_racer_calibration_btn,
                    owner.racer_calibration_settings_btn,
                    owner.racer_calibration_path_controls,
                    path_field,
                ):
                    self.assertTrue(control.isEnabled())
                self.assertFalse(owner.clear_racer_calibration_btn.isEnabled())
                self.assertEqual(path_field.text(), "remembered-dark.zarr")
                self.assertEqual(owner._live_control_states, [])


if __name__ == "__main__":
    unittest.main()
