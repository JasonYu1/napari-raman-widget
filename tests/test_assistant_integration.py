"""Assistant dispatch through real Qt controls, without an API or hardware."""

import json
import os
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from qtpy.QtCore import QCoreApplication, QEvent, Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import (
    QApplication, QDockWidget, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QSpinBox, QWidget,
)

from napari_raman_widget.assistant_plot_tools import get_plot_state
from napari_raman_widget.assistant_history import HistoryStore
from napari_raman_widget.chat_panel import ChatPanel
from napari_raman_widget.plot_windows import SpectrumWindow
from napari_raman_widget.plot_workspace import show_plot
from napari_raman_widget.spectral_calibration import PixelToWavenumberCalibration


class _ViewerWindow:
    def __init__(self):
        self.main = QMainWindow()

    def add_dock_widget(self, widget, **kwargs):
        dock = QDockWidget(kwargs["name"], self.main)
        dock.setWidget(widget)
        self.main.addDockWidget(Qt.BottomDockWidgetArea, dock)
        return dock


class AssistantIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.owner = QWidget()
        self.owner.core = None
        self.owner.status = QLabel("Ready", self.owner)
        self.owner.spectral_calibration_path = QLineEdit(self.owner)
        self.owner.spectral_calibration = None
        self.owner.exposure_input = QSpinBox(self.owner)
        self.owner.exposure_input.setRange(1, 1000)
        self.owner.exposure_input.setValue(10)
        self.owner.collect_dark_noise = Mock()
        self.owner._stop_live_raman = Mock()
        self.owner._plot_windows = []
        self.owner.viewer = SimpleNamespace(window=_ViewerWindow())
        history_dir = tempfile.TemporaryDirectory()
        self.addCleanup(history_dir.cleanup)
        self.chat = ChatPanel(self.owner, history_store=HistoryStore(history_dir.name))

    def tearDown(self):
        self.chat.deleteLater()
        self.owner.viewer.window.main.deleteLater()
        self.owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def execute(self, name, **inputs):
        return self.chat._execute_tool(name, inputs)

    @staticmethod
    def calibration():
        return PixelToWavenumberCalibration([0, 31], [200, 1800], degree=1)

    def test_session_load_does_not_change_existing_plot_without_explicit_id(self):
        plot = show_plot(self.owner, SpectrumWindow(np.arange(32)))
        with patch(
            "napari_raman_widget.spectral_calibration_ui."
            "load_pixel_to_wavenumber_calibration",
            return_value=self.calibration(),
        ):
            result = json.loads(self.execute("load_wavenumber_calibration", file="chosen.json"))
        self.assertTrue(result["loaded"])
        self.assertIsNone(plot.spectral_calibration)
        self.assertFalse(plot.show_wavenumber_check.isChecked())
        self.assertFalse(plot.show_wavenumber_check.isEnabled())
        self.assertIn('"loaded": true', self.execute("get_state"))

    def test_load_attach_opt_in_then_clear_keeps_the_existing_plot(self):
        plot = show_plot(self.owner, SpectrumWindow(np.arange(32)))
        panel_id = get_plot_state(self.owner)[0]["panel_id"]
        calibration = self.calibration()
        with patch(
            "napari_raman_widget.spectral_calibration_ui."
            "load_pixel_to_wavenumber_calibration", return_value=calibration,
        ):
            self.execute("load_wavenumber_calibration", file="chosen.json", panel_id=panel_id)
        self.assertIs(plot.spectral_calibration, calibration)
        self.assertFalse(plot.show_wavenumber_check.isChecked())
        self.execute("configure_plot", panel_id=panel_id, show_wavenumber=True, white_background=True)
        self.assertTrue(plot.show_wavenumber_check.isChecked())
        self.assertTrue(plot.white_background_check.isChecked())
        np.testing.assert_allclose(plot.ax.lines[0].get_xdata(), calibration.transform(np.arange(32)))
        self.execute("clear_wavenumber_calibration")
        self.assertIsNone(self.owner.spectral_calibration)
        self.assertEqual(self.owner.spectral_calibration_path.text(), "")
        self.assertIs(plot.spectral_calibration, calibration)

    def test_invalid_load_and_unknown_target_preserve_session(self):
        calibration = self.calibration()
        self.owner.spectral_calibration = calibration
        self.owner.spectral_calibration_path.setText("old.json")
        with patch(
            "napari_raman_widget.spectral_calibration_ui."
            "load_pixel_to_wavenumber_calibration", side_effect=ValueError("bad JSON"),
        ) as loader:
            self.assertIn("bad JSON", self.execute("load_wavenumber_calibration", file="bad.json"))
            self.assertIs(self.owner.spectral_calibration, calibration)
            self.assertEqual(self.owner.spectral_calibration_path.text(), "old.json")
            loader.reset_mock()
            self.execute("load_wavenumber_calibration", file="new.json", panel_id="missing")
            loader.assert_not_called()

    def test_declining_hardware_action_keeps_fields_and_does_not_acquire(self):
        with patch("napari_raman_widget.chat_panel.QMessageBox.question", return_value=QMessageBox.No):
            result = self.execute("collect_dark_noise", exposure_ms=50)
        self.assertIn("declined", result)
        self.assertEqual(self.owner.exposure_input.value(), 10)
        self.owner.collect_dark_noise.assert_not_called()

    def test_stop_bypasses_confirmation_and_unknown_arguments_are_rejected(self):
        with patch("napari_raman_widget.chat_panel.QMessageBox.question") as confirm:
            self.execute("stop_live_spectra")
        confirm.assert_not_called()
        self.owner._stop_live_raman.assert_called_once()
        result = self.execute("collect_dark_noise", unknown_option=1)
        self.assertIn("Unknown parameters", result)
        self.owner.collect_dark_noise.assert_not_called()

    def test_capabilities_are_generated_from_registered_tools(self):
        result = json.loads(self.execute("get_assistant_capabilities"))
        names = {tool["name"] for tool in result["tools"]}
        self.assertIn("configure_plot", names)
        self.assertIn("load_wavenumber_calibration", names)
        self.assertIn("snapshots", result["limits"])

    def test_assistant_is_one_console_without_extra_input_or_send_button(self):
        self.assertEqual(self.chat.findChildren(QLineEdit), [])
        self.assertEqual(self.chat.findChildren(QPushButton), [])
        self.assertEqual(self.chat.findChildren(QPlainTextEdit), [self.chat.console])
        self.assertIs(self.chat.log, self.chat.console)
        self.assertIs(self.chat.focusProxy(), self.chat.console)

    def test_blank_prompt_does_not_start_a_model_request(self):
        with patch("napari_raman_widget.chat_panel.threading.Thread") as worker:
            self.chat.console.set_command_text("  ")
            self.chat.console.submit_command()
        worker.assert_not_called()
        self.assertEqual(self.chat._messages, [])

    def test_model_failure_restores_a_usable_prompt(self):
        api = SimpleNamespace(Anthropic=Mock(side_effect=RuntimeError("Mock API unavailable")))
        with patch.dict(sys.modules, {"anthropic": api}):
            self.chat.console.set_command_text("List my plots")
            QTest.keyClick(self.chat.console, Qt.Key_Return)
            deadline = time.monotonic() + 10
            while self.chat._busy and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)
            self.app.processEvents()
        self.assertFalse(self.chat._busy)
        self.assertIn("Mock API unavailable", self.chat.console.toPlainText())
        self.assertFalse(self.chat.console.isReadOnly())
        QTest.keyClicks(self.chat.console, "Try again")
        self.assertEqual(self.chat.console.command_text(), "Try again")

    def test_mock_model_tool_loop_dispatches_on_gui_thread(self):
        plot = show_plot(self.owner, SpectrumWindow(np.arange(32)))
        panel_id = get_plot_state(self.owner)[0]["panel_id"]
        use = SimpleNamespace(
            type="tool_use", id="test-tool-1", name="configure_plot",
            input={"panel_id": panel_id, "white_background": True},
        )
        done = SimpleNamespace(type="text", text="Background updated.")
        create = Mock(side_effect=[
            SimpleNamespace(content=[use], stop_reason="tool_use"),
            SimpleNamespace(content=[done], stop_reason="end_turn"),
        ])
        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        api = SimpleNamespace(Anthropic=Mock(return_value=client))
        with patch.dict(sys.modules, {"anthropic": api}):
            self.chat.console.set_command_text("Make this plot background white")
            QTest.keyClick(self.chat.console, Qt.Key_Return)
            deadline = time.monotonic() + 10
            while self.chat._busy and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)
            self.app.processEvents()
        self.assertFalse(self.chat._busy, "Mock model/tool loop did not complete")
        self.assertTrue(plot.white_background_check.isChecked())
        self.assertEqual(create.call_count, 2)
        request = create.call_args_list[0].kwargs
        self.assertIn("list_plots", {tool["name"] for tool in request["tools"]})
        self.assertIn("snapshots", request["system"])
        self.assertIn("Background updated.", self.chat.log.toPlainText())
        self.assertEqual(self.chat.log.toPlainText().count("Make this plot background white"), 1)
        self.assertEqual(self.chat.console.command_text(), "")
        self.assertFalse(self.chat.console.isReadOnly())


if __name__ == "__main__":
    unittest.main()
