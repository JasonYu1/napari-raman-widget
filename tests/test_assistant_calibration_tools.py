"""Read-only assistant access to real calibration result and log panels."""

import json
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import xarray as xr
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QDockWidget, QMainWindow, QWidget

from napari_raman_widget.assistant_calibration_tools import (
    CALIBRATION_ACTIONS,
    control_spectral_axis_calibration,
    get_calibration_state,
    inspect_calibration_result,
    query_calibration_progress,
)
from napari_raman_widget.log_window import LogWindow
from napari_raman_widget.plot_windows import (
    CalibrationPlotWindow,
    SpectrumWindow,
)
from napari_raman_widget.plot_workspace import show_plot


class _Window:
    def __init__(self):
        self.main = QMainWindow()
        self.docks = []

    def add_dock_widget(self, panel, **kwargs):
        dock = QDockWidget(kwargs["name"], self.main)
        dock.setWidget(panel)
        self.main.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.docks.append(dock)
        return dock


class AssistantCalibrationToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = _Window()
        self.owner = QWidget()
        self.owner.viewer = SimpleNamespace(window=self.window)
        self.owner._plot_windows = []
        self.owner._plot_workspace = None
        self.owner._live_raman_window = None

    def tearDown(self):
        self.window.main.close()
        self.owner.close()

    @staticmethod
    def calibration_panel():
        spectra = np.array(
            [
                [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
                [[5, 6, 7, 8, 9], [5, 6, 7, 8, 9]],
                [[10, 11, 12, 13, 14], [10, 11, 12, 13, 14]],
            ],
            dtype=float,
        )
        dataset = xr.Dataset(
            {
                "imgs": (
                    ("point", "Y", "X"),
                    np.zeros((3, 12, 12), dtype=float),
                ),
                "rel_BF_pos": (
                    ("point", "coordinate"),
                    np.array([[1, 2], [np.nan, 4], [7, 8]], dtype=float),
                ),
                "laser_pos": (
                    ("point", "volt"),
                    np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
                ),
                "specs": (("point", "repeat", "pixel"), spectra),
            }
        )
        return CalibrationPlotWindow(dataset)

    def test_inspection_preserves_selection_and_summarizes_stored_data(self):
        panel = show_plot(self.owner, self.calibration_panel())
        original_index = panel.selected_index

        result = json.loads(inspect_calibration_result(self.owner, {}))

        self.assertEqual(panel.selected_index, original_index)
        self.assertFalse(result["selection_changed"])
        self.assertEqual(result["selected_point_index"], 1)
        self.assertEqual(result["stored_spectra"]["shape"], [2, 5])
        self.assertEqual(result["stored_spectra"]["finite_count"], 10)
        self.assertEqual(result["stored_spectra"]["mean"], 2.0)
        self.assertEqual(result["displayed_spectrum"]["shape"], [5])
        self.assertNotIn("trace", result)

    def test_one_based_selection_inventory_and_bounded_explicit_trace(self):
        panel = show_plot(self.owner, self.calibration_panel())

        result = json.loads(
            inspect_calibration_result(
                self.owner,
                {
                    "point_index": 3,
                    "include_trace": True,
                    "trace_limit": 3,
                    "include_point_inventory": True,
                    "point_limit": 2,
                },
            )
        )

        self.assertEqual(panel.selected_index, 2)
        self.assertEqual(result["selected_point_index"], 3)
        self.assertEqual(result["image_x"], 7.0)
        self.assertEqual(result["galvo_y_v"], 0.6)
        self.assertEqual(result["trace"]["total_length"], 5)
        self.assertEqual(result["trace"]["returned_length"], 3)
        self.assertTrue(result["trace"]["truncated"])
        self.assertEqual(result["trace"]["intensity"], [10.0, 11.0, 12.0])
        inventory = result["point_inventory"]
        self.assertEqual(inventory["total_count"], 3)
        self.assertEqual(inventory["returned_count"], 2)
        self.assertTrue(inventory["truncated"])
        self.assertFalse(inventory["points"][1]["selectable"])

        with self.assertRaisesRegex(ValueError, "not a finite selectable"):
            inspect_calibration_result(self.owner, {"point_index": 2})
        self.assertEqual(panel.selected_index, 2)

    def test_explicit_stable_id_can_inspect_a_noncurrent_result(self):
        panel = show_plot(self.owner, self.calibration_panel())
        panel_id = panel._plot_panel_id
        show_plot(self.owner, LogWindow("Calibration log", show_progress=True))

        result = json.loads(
            inspect_calibration_result(self.owner, {"panel_id": panel_id})
        )

        self.assertEqual(result["panel_id"], panel_id)
        self.assertEqual(result["selected_point_index"], 1)

    def test_invalid_options_do_not_change_the_selected_point(self):
        panel = show_plot(self.owner, self.calibration_panel())
        self.assertEqual(panel.selected_index, 0)

        for invalid_input in (
            {
                "point_index": 3,
                "include_trace": True,
                "trace_limit": 1.5,
            },
            {"point_index": 3, "include_trace": "true"},
            {"point_index": 1.9},
        ):
            with self.subTest(invalid_input=invalid_input):
                with self.assertRaises(ValueError):
                    inspect_calibration_result(self.owner, invalid_input)
                self.assertEqual(panel.selected_index, 0)

        with self.assertRaises(ValueError):
            inspect_calibration_result(self.owner, {"panel_id": ""})

    def test_progress_uses_real_bar_values_and_latest_log_for_current_result(self):
        log = show_plot(
            self.owner, LogWindow("Calibration log", show_progress=True)
        )
        log.start_progress("Preparing calibration")
        log.update_progress(2, 5, "Collecting calibration points (2/3)")
        log.append("0123456789")
        result_panel = show_plot(self.owner, self.calibration_panel())
        self.assertIs(
            self.owner._plot_workspace.tabs.currentWidget(), result_panel
        )

        result = json.loads(
            query_calibration_progress(self.owner, {"tail_chars": 4})
        )

        self.assertEqual(result["panel_id"], log._plot_panel_id)
        self.assertEqual(result["progress"]["state"], "busy")
        self.assertEqual(result["progress"]["completed"], 2)
        self.assertEqual(result["progress"]["total"], 5)
        self.assertEqual(
            result["progress"]["label"],
            "Collecting calibration points (2/3)",
        )
        self.assertEqual(result["log"]["recent_text"], "6789")
        self.assertTrue(result["log"]["truncated"])
        self.assertIn("Qt main thread", result["synchronous_constraint"])

        show_plot(
            self.owner,
            SpectrumWindow(np.array([[0, 1, 2]], dtype=float)),
        )
        newest = json.loads(query_calibration_progress(self.owner, {}))
        self.assertEqual(newest["panel_id"], log._plot_panel_id)

        log.fail_progress("Calibration failed")
        failed = json.loads(
            query_calibration_progress(
                self.owner, {"panel_id": log._plot_panel_id, "tail_chars": 0}
            )
        )
        self.assertEqual(failed["progress"]["state"], "failed")

        log.finish_progress("Calibration complete")
        complete = json.loads(
            query_calibration_progress(
                self.owner, {"panel_id": log._plot_panel_id, "tail_chars": 0}
            )
        )
        self.assertEqual(complete["progress"]["state"], "complete")

    def test_progress_query_does_not_invent_values_for_arbitrary_logs(self):
        calibration_log = show_plot(self.owner, LogWindow("Calibration log"))
        unavailable = json.loads(query_calibration_progress(self.owner, {}))
        self.assertEqual(
            unavailable["panel_id"], calibration_log._plot_panel_id
        )
        self.assertFalse(unavailable["progress_available"])
        self.assertNotIn("progress", unavailable)

        other_log = show_plot(
            self.owner, LogWindow("Acquisition log", show_progress=True)
        )
        other_log.start_progress("Collecting")
        with self.assertRaisesRegex(ValueError, "not a calibration log"):
            query_calibration_progress(
                self.owner, {"panel_id": other_log._plot_panel_id}
            )

    def test_spectral_axis_tool_starts_reports_and_cancels_manual_workflow(self):
        panel = show_plot(
            self.owner,
            SpectrumWindow(np.array([[0, 1, 4, 1, 0]], dtype=float)),
        )

        status = json.loads(
            control_spectral_axis_calibration(
                self.owner, {"operation": "status"}
            )
        )
        self.assertEqual(status["state"], "inactive")
        self.assertEqual(status["accepted_peak_count"], 0)

        started = json.loads(
            control_spectral_axis_calibration(
                self.owner, {"operation": "start", "degree": 1}
            )
        )
        self.assertTrue(panel._calibrating)
        self.assertEqual(started["state"], "active")
        self.assertEqual(started["degree"], 1)
        self.assertEqual(started["required_peak_count"], 2)
        self.assertIn("never saves", started["manual_finish_required"])

        panel._calibration_pixels = [2]
        panel._known_shifts = [520.7]
        panel._pending_pixel = 4
        panel._update_calibration_progress()
        progress = json.loads(
            control_spectral_axis_calibration(
                self.owner, {"operation": "status"}
            )
        )
        self.assertEqual(progress["accepted_peak_count"], 1)
        self.assertEqual(
            progress["accepted_peaks"],
            [{"known_shift_cm-1": 520.7, "pixel": 2}],
        )
        self.assertEqual(progress["pending_pixel"], 4)

        with self.assertRaisesRegex(ValueError, "already active"):
            control_spectral_axis_calibration(
                self.owner, {"operation": "start", "degree": 2}
            )
        self.assertEqual(panel._calibration_pixels, [2])

        cancelled = json.loads(
            control_spectral_axis_calibration(
                self.owner, {"operation": "cancel"}
            )
        )
        self.assertFalse(panel._calibrating)
        self.assertEqual(cancelled["state"], "inactive")

        with self.assertRaises(ValueError):
            control_spectral_axis_calibration(
                self.owner, {"operation": "start", "degree": 1.5}
            )
        self.assertFalse(panel._calibrating)

    def test_exported_actions_are_readonly_and_state_helper_is_compact(self):
        names = {action["name"] for action in CALIBRATION_ACTIONS}
        self.assertEqual(
            names,
            {
                "inspect_calibration_result",
                "query_calibration_progress",
                "control_spectral_axis_calibration",
            },
        )
        self.assertTrue(all(action["readonly"] for action in CALIBRATION_ACTIONS))
        progress_action = next(
            action
            for action in CALIBRATION_ACTIONS
            if action["name"] == "query_calibration_progress"
        )
        self.assertIn("synchronous", progress_action["description"])

        panel = show_plot(self.owner, self.calibration_panel())
        state = get_calibration_state(self.owner)
        self.assertEqual(
            state["latest_calibration_result"]["panel_id"],
            panel._plot_panel_id,
        )
        self.assertEqual(
            state["latest_calibration_result"]["selected_point_index"], 1
        )


if __name__ == "__main__":
    unittest.main()
