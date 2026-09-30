"""Live imaging must stop before spatial acquisition, never on preview cancel."""

import ast
import os
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication, QCheckBox

from napari_raman_widget.scan_workflows import start_grid_scan


class _Shapes:
    def __init__(self):
        self.data = [np.array([[10, 20], [30, 40]])]
        self.selected_data = {0}
        self.shape_type = ["rectangle"]
        self.name = "Scan ROI"
        self.mode = "select"


class _LiveCore:
    def __init__(self, events, running=True, *, failure=None, stuck=False):
        self.events = events
        self.running = running
        self.failure = failure
        self.stuck = stuck

    def stopSequenceAcquisition(self):
        self.events.append("stop live")
        if self.failure:
            raise RuntimeError(self.failure)
        if not self.stuck:
            self.running = False

    def isSequenceRunning(self):
        self.events.append("verify stopped")
        return self.running


class ScanLiveStopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_owner(self, *, running=True, failure=None, stuck=False, demo=False):
        self.events = []
        self.shapes = _Shapes()
        owner_type = type("DemoWidget", (SimpleNamespace,), {}) if demo else SimpleNamespace
        owner = owner_type(
            core=_LiveCore(self.events, running, failure=failure, stuck=stuck),
            daq=Mock(), collector=Mock(), transformer=Mock(), status=Mock(),
            _acquisition_jobs=Mock(), _get_image_xy=Mock(return_value=(100, 100)),
            viewer=SimpleNamespace(layers=SimpleNamespace(
                selection=SimpleNamespace(active=self.shapes))),
            scan_name_input=Mock(text=lambda: "live-stop"),
            scan_zscan_check=Mock(isChecked=lambda: False),
            scan_sampling_mode_combo=Mock(currentData=lambda: "count"),
            scan_total_points_input=Mock(value=lambda: 4),
            scan_spacing_input=Mock(value=lambda: 10),
            scan_exp_input=Mock(value=lambda: 100),
            scan_z_input=Mock(value=lambda: 0), channel_rows=[],
            _set_scan_imaging_channel=Mock(), _set_scan_raman_mode=Mock(),
            spectral_calibration=None,
        )
        owner._acquisition_jobs.available.return_value = True

        def convert(points, **kwargs):
            self.assertFalse(owner.core.running)
            self.events.append("convert points")
            return points

        def start(*args):
            self.assertFalse(owner.core.running)
            if demo:
                self.assertFalse(owner.demo_live_timer.isActive())
                self.assertFalse(owner.demo_live_check.isChecked())
            self.events.append("start worker")
            return True

        owner.transformer.BF_to_volts.side_effect = convert
        owner._acquisition_jobs.start.side_effect = start
        if demo:
            # Exercise the real GUI-only stop method without importing or
            # constructing a microscope widget or its hardware dependencies.
            source = Path(__file__).resolve().parents[1] / "napari_raman_widget" / "demo_widget.py"
            tree = ast.parse(source.read_text(encoding="utf-8"))
            method = next(node for node in ast.walk(tree)
                          if isinstance(node, ast.FunctionDef) and node.name == "_toggle_demo_live")
            namespace = {}
            exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
            owner._toggle_demo_live = MethodType(namespace[method.name], owner)
            owner.demo_backend = object()
            owner.demo_live_timer = QTimer()
            owner.demo_live_timer.start(150)
            owner.demo_live_check = QCheckBox()
            owner.demo_live_check.setChecked(True)
            self.addCleanup(owner.demo_live_timer.stop)
            self.addCleanup(owner.demo_live_check.close)
        return owner

    def run_preview(self, owner, *, accepted=True):
        def confirm(*args):
            self.events.append("confirm preview")
            return accepted

        with patch.dict("sys.modules", {"napari.layers": SimpleNamespace(Shapes=_Shapes)}), patch(
            "napari_raman_widget.scan_workflows.scan_image_reference", return_value=None
        ), patch("napari_raman_widget.scan_workflows._confirm", side_effect=confirm):
            start_grid_scan(owner)

    def assert_not_started(self, owner):
        owner._acquisition_jobs.start.assert_not_called()
        owner.transformer.BF_to_volts.assert_not_called()
        owner._set_scan_imaging_channel.assert_not_called()
        owner._set_scan_raman_mode.assert_not_called()
        self.assertEqual(owner.daq.mock_calls, [])
        self.assertEqual(owner.collector.mock_calls, [])
        self.assertEqual(self.shapes.mode, "select")
        self.assertEqual(self.shapes.selected_data, {0})

    def test_live_stops_and_is_verified_before_point_conversion_and_worker(self):
        owner = self.make_owner()
        self.run_preview(owner)
        self.assertEqual(self.events, ["confirm preview", "stop live", "verify stopped",
                                       "convert points", "start worker"])
        owner._acquisition_jobs.start.assert_called_once()
        self.assertFalse(owner.core.running)

    def test_already_stopped_camera_can_start_scan(self):
        owner = self.make_owner(running=False)
        self.run_preview(owner)
        owner._acquisition_jobs.start.assert_called_once()
        self.assertFalse(owner.core.running)

    def test_cancelled_preview_leaves_live_camera_running(self):
        owner = self.make_owner()
        self.run_preview(owner, accepted=False)
        self.assertEqual(self.events, ["confirm preview"])
        self.assertTrue(owner.core.running)
        self.assert_not_started(owner)

    def test_stop_error_aborts_before_acquisition_or_shape_changes(self):
        owner = self.make_owner(failure="camera could not stop")
        self.run_preview(owner)
        self.assert_not_started(owner)
        self.assertIn("camera could not stop", owner.status.setText.call_args.args[0])

    def test_camera_still_running_after_stop_aborts(self):
        owner = self.make_owner(stuck=True)
        self.run_preview(owner)
        self.assert_not_started(owner)
        self.assertIn("Live imaging is still running", owner.status.setText.call_args.args[0])

    def test_new_acquisition_during_preview_is_not_interrupted(self):
        owner = self.make_owner()
        owner._acquisition_jobs.available.side_effect = [True, False]
        self.run_preview(owner)
        self.assertEqual(self.events, ["confirm preview"])
        self.assertTrue(owner.core.running)
        self.assert_not_started(owner)

    def test_demo_timer_checkbox_and_camera_stop_before_worker(self):
        owner = self.make_owner(demo=True)
        self.run_preview(owner)
        self.assertEqual(self.events, ["confirm preview", "stop live", "verify stopped", "start worker"])
        owner._acquisition_jobs.start.assert_called_once()

    def test_demo_preview_cancel_keeps_timer_checkbox_and_camera_live(self):
        owner = self.make_owner(demo=True)
        self.run_preview(owner, accepted=False)
        self.assertTrue(owner.demo_live_timer.isActive())
        self.assertTrue(owner.demo_live_check.isChecked())
        self.assertTrue(owner.core.running)
        self.assert_not_started(owner)


if __name__ == "__main__":
    unittest.main()
