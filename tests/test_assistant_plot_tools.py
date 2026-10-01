"""Assistant plot actions drive the same offscreen Qt controls as users."""

import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
from qtpy.QtCore import QCoreApplication, QEvent, Qt
from qtpy.QtWidgets import QApplication, QDockWidget, QMainWindow, QWidget
import xarray as xr

from napari_raman_widget.assistant_plot_tools import (
    PLOT_ACTIONS,
    get_plot_state,
    get_plot_workspace_state,
    resolve_plot_panel,
)
from napari_raman_widget.plot_windows import (
    DatasetViewerWindow,
    DetectorImageWindow,
    GridScanPlotWindow,
    SpectrumWindow,
)
from napari_raman_widget.plot_workspace import show_plot
from napari_raman_widget.spectral_calibration import (
    PixelToWavenumberCalibration,
)


_ACTIONS = {action["name"]: action for action in PLOT_ACTIONS}


class _ViewerWindow:
    def __init__(self):
        self.main = QMainWindow()
        self.docks = []

    def add_dock_widget(self, panel, **kwargs):
        dock = QDockWidget(kwargs["name"], self.main)
        dock.setWidget(panel)
        self.main.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.docks.append(dock)
        return dock


def _grid_dataset(*, z_scan=False):
    data = {
        "grid_pos": (
            ("point", "coord"),
            np.array([[1.0, 1.0], [2.0, 2.0]]),
        ),
    }
    if z_scan:
        data.update(
            {
                "BF_z": (("z", "Y", "X"), np.zeros((2, 4, 4))),
                "z_range": (("z",), np.array([-1.0, 1.0])),
                "specs": (
                    ("z", "point", "pixel"),
                    np.ones((2, 2, 11)),
                ),
            }
        )
    else:
        data.update(
            {
                "BF": (("Y", "X"), np.zeros((4, 4))),
                "specs": (("point", "pixel"), np.ones((2, 11))),
            }
        )
    return xr.Dataset(data)


def _dataset_viewer_data():
    index = pd.MultiIndex.from_tuples(
        [(0, 0, 0, 0), (1, 0, 0, 0)],
        names=["t", "p", "z", "pt"],
    )
    frame = pd.DataFrame(
        [
            [0.0, 1.0, 2.0, 1.0, 2.0, 0.0],
            [3.0, 4.0, 5.0, 1.0, 2.0, 1.0],
        ],
        index=index,
        columns=[0, 1, 2, "X", "Y", "time"],
    )
    images = xr.DataArray(
        np.zeros((2, 1, 1, 1, 4, 4)),
        dims=["t", "p", "c", "z", "y", "x"],
        coords={"t": [0, 1], "p": [0], "c": [0], "z": [0]},
    )
    return frame, images


class AssistantPlotToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = _ViewerWindow()
        self.owner = QWidget()
        self.owner.viewer = SimpleNamespace(window=self.window)
        self.owner._plot_windows = []
        self.owner._live_raman_window = None
        self.owner._live_raman_context = None
        self.owner._stop_live_raman = Mock()

    def tearDown(self):
        self.window.main.close()
        self.window.main.deleteLater()
        self.owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def call(self, name, **tool_input):
        return _ACTIONS[name]["handler"](self.owner, tool_input)

    def test_listing_does_not_create_a_workspace(self):
        payload = json.loads(self.call("list_plots"))

        self.assertFalse(hasattr(self.owner, "_plot_workspace"))
        self.assertEqual(payload["plots"], [])
        self.assertEqual(
            payload["workspace"],
            {
                "current_panel_id": None,
                "exists": False,
                "floating": None,
                "panel_count": 0,
                "visible": False,
            },
        )
        with self.assertRaisesRegex(ValueError, "No plot workspace"):
            resolve_plot_panel(self.owner)

    def test_panel_ids_survive_titles_and_reordering_and_are_not_reused(self):
        first = show_plot(self.owner, SpectrumWindow(np.arange(11.0)))
        second = show_plot(self.owner, SpectrumWindow(np.arange(11.0)))
        workspace = self.owner._plot_workspace
        first_id = first._plot_panel_id
        second_id = second._plot_panel_id

        second.setWindowTitle("Live spectrum 42")
        workspace.tabs.tabBar().moveTab(1, 0)
        states = get_plot_state(self.owner)

        self.assertEqual(first_id, "spectrum-1")
        self.assertEqual(second_id, "spectrum-2")
        self.assertEqual([item["panel_id"] for item in states], [second_id, first_id])
        self.assertTrue(states[0]["current"])
        self.assertIs(resolve_plot_panel(self.owner, second_id), second)

        workspace.close_panel(workspace.tabs.indexOf(first))
        with self.assertRaisesRegex(ValueError, "Unknown plot panel_id"):
            resolve_plot_panel(self.owner, first_id)
        third = show_plot(self.owner, SpectrumWindow(np.arange(11.0)))
        self.assertEqual(third._plot_panel_id, "spectrum-3")
        self.assertEqual(first._plot_panel_id, first_id)

    def test_spectrum_configuration_is_transactional_and_uses_controls(self):
        calibration = PixelToWavenumberCalibration(
            [0.0, 20.0], [100.0, 300.0], degree=1
        )
        panel = show_plot(
            self.owner,
            SpectrumWindow(
                np.vstack([np.arange(21.0), np.arange(21.0) + 1]),
                spectral_calibration=calibration,
            ),
        )

        with self.assertRaisesRegex(ValueError, "odd integer"):
            self.call(
                "configure_plot",
                panel_id=panel._plot_panel_id,
                white_background=True,
                smoothing_window=8,
            )
        self.assertFalse(panel.white_background_check.isChecked())
        self.assertEqual(panel.smoothing_window_input.value(), 5)

        state = json.loads(
            self.call(
                "configure_plot",
                panel_id=panel._plot_panel_id,
                white_background=True,
                fix_y_scale=True,
                show_wavenumber=True,
                smoothing=True,
                smoothing_window=7,
                baseline_subtraction=True,
                baseline_lambda=1e5,
                spectrum_view="all",
            )
        )

        self.assertTrue(panel.white_background_check.isChecked())
        self.assertTrue(panel.fix_y_scale_check.isChecked())
        self.assertTrue(panel.show_wavenumber_check.isChecked())
        self.assertTrue(panel.smoothing_check.isChecked())
        self.assertEqual(panel.smoothing_window_input.value(), 7)
        self.assertTrue(panel.baseline_subtraction_check.isChecked())
        self.assertEqual(panel.baseline_lambda_input.value(), 1e5)
        self.assertFalse(panel._show_mean)
        self.assertEqual(state["values"]["spectrum_view"], "all")
        self.assertTrue(state["spectral_calibration_available"])
        self.assertNotIn("selected_dataset_point_index", state["supported_controls"])

    def test_missing_calibration_fails_before_other_mutations(self):
        panel = show_plot(self.owner, SpectrumWindow(np.arange(11.0)))

        with self.assertRaisesRegex(ValueError, "requires a loaded"):
            self.call(
                "configure_plot",
                panel_id=panel._plot_panel_id,
                white_background=True,
                show_wavenumber=True,
            )

        self.assertFalse(panel.white_background_check.isChecked())
        self.assertFalse(panel.show_wavenumber_check.isChecked())

    def test_detector_can_switch_view_and_configure_processing_atomically(self):
        calibration = PixelToWavenumberCalibration(
            [0.0, 10.0], [100.0, 200.0], degree=1
        )
        panel = show_plot(
            self.owner,
            DetectorImageWindow(
                np.ones((1, 4, 11)),
                spectral_calibration=calibration,
            ),
        )
        before = get_plot_state(self.owner)[0]
        self.assertFalse(before["enabled"]["smoothing"])
        self.assertFalse(before["enabled"]["show_wavenumber"])
        with self.assertRaisesRegex(ValueError, "detector_view='spectrum'"):
            self.call(
                "configure_plot",
                panel_id=panel._plot_panel_id,
                show_wavenumber=True,
            )

        self.call(
            "configure_plot",
            panel_id=panel._plot_panel_id,
            detector_view="spectrum",
            detector_start_row=1,
            detector_end_row=2,
            smoothing=True,
            smoothing_window=7,
            fix_y_scale=True,
            show_wavenumber=True,
        )

        self.assertTrue(panel._show_spectrum)
        self.assertEqual(panel.start_row_input.value(), 1)
        self.assertEqual(panel.end_row_input.value(), 2)
        self.assertTrue(panel.smoothing_check.isChecked())
        self.assertEqual(panel.smoothing_window_input.value(), 7)
        self.assertTrue(panel.fix_y_scale_check.isChecked())
        self.assertTrue(panel.show_wavenumber_check.isChecked())
        after = get_plot_state(self.owner)[0]
        self.assertTrue(after["enabled"]["smoothing"])

    def test_short_detector_rejects_processing_before_switching_view(self):
        panel = show_plot(
            self.owner, DetectorImageWindow(np.ones((1, 4, 2)))
        )

        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.call(
                "configure_plot",
                panel_id=panel._plot_panel_id,
                detector_view="spectrum",
                white_background=True,
                smoothing=True,
                baseline_subtraction=True,
            )

        self.assertFalse(panel._show_spectrum)
        self.assertFalse(panel.white_background_check.isChecked())
        self.assertFalse(panel.smoothing_check.isChecked())
        self.assertFalse(panel.baseline_subtraction_check.isChecked())

    def test_grid_and_dataset_navigation_use_their_numeric_controls(self):
        grid = show_plot(
            self.owner, GridScanPlotWindow(_grid_dataset(z_scan=True))
        )
        frame, images = _dataset_viewer_data()
        dataset = show_plot(self.owner, DatasetViewerWindow(frame, images))

        self.call(
            "configure_plot",
            panel_id=grid._plot_panel_id,
            grid_view="point",
            grid_z_index=1,
        )
        self.assertFalse(grid._show_average)
        self.assertEqual(grid._z_input.value(), 1)
        self.assertEqual(grid._z_slider.value(), 1)

        state = json.loads(
            self.call(
                "configure_plot",
                panel_id=dataset._plot_panel_id,
                dataset_t_index=1,
            )
        )
        self.assertEqual(dataset.t_input.value(), 1)
        self.assertEqual(dataset.t_slider.value(), 1)
        self.assertEqual(state["values"]["dataset_t_index"], 1)
        self.assertNotIn(
            "selected_dataset_point_index", state["supported_controls"]
        )

    def test_show_focus_float_and_hide_preserve_tabs(self):
        first = show_plot(self.owner, SpectrumWindow(np.arange(11.0)))
        second = show_plot(self.owner, SpectrumWindow(np.arange(11.0)))
        workspace = self.owner._plot_workspace

        shown = json.loads(
            self.call(
                "show_plot_workspace",
                panel_id=first._plot_panel_id,
                floating=False,
            )
        )
        self.assertIs(workspace.tabs.currentWidget(), first)
        self.assertFalse(shown["floating"])
        self.assertEqual(shown["current_panel_id"], first._plot_panel_id)

        self.assertEqual(self.call("hide_plot_workspace"),
                         "Plot workspace hidden; its tabs remain open.")
        self.assertTrue(workspace._dock.isHidden())
        self.assertEqual(workspace.tabs.count(), 2)
        self.assertFalse(get_plot_workspace_state(self.owner)["visible"])
        self.assertIs(resolve_plot_panel(self.owner, second._plot_panel_id), second)

    def test_grid_map_state_is_read_only_and_reports_applied_values(self):
        dataset = _grid_dataset()
        dataset["specs"].values[1] = np.nan
        grid = show_plot(self.owner, GridScanPlotWindow(dataset))
        controls = grid.band_map_controls
        original = dataset["specs"].values.copy()

        initial = get_plot_state(self.owner)[0]
        self.assertEqual(initial["raman_map"]["mode"], "points")
        self.assertEqual(initial["raman_map"]["unit"], "pixel")
        self.assertEqual(initial["raman_map"]["colormap"], "viridis")
        self.assertFalse(initial["raman_map"]["reverse_colormap"])
        self.assertFalse(initial["raman_map"]["applied"])
        self.assertIsNone(initial["raman_map"]["valid_points"])
        self.assertIsNone(initial["raman_map"]["invalid_points"])
        self.assertNotIn("raman_map", initial["supported_controls"])
        self.assertIsNone(controls.values)

        controls.mode_combo.setCurrentIndex(controls.mode_combo.findData("ratio"))
        controls.a_low.setValue(1)
        controls.a_high.setValue(4)
        controls.b_low.setValue(5)
        controls.b_high.setValue(7)
        controls.apply_button.click()
        values_before = controls.values.copy()
        applied = json.loads(self.call("list_plots"))["plots"][0]["raman_map"]

        self.assertEqual(applied, {
            "mode": "ratio",
            "unit": "pixel",
            "colormap": "viridis",
            "reverse_colormap": False,
            "band_a": [1.0, 4.0],
            "band_b": [5.0, 7.0],
            "applied": True,
            "valid_points": 1,
            "invalid_points": 1,
            "uses_raw_spectra": True,
        })
        np.testing.assert_array_equal(controls.values, values_before)
        np.testing.assert_array_equal(dataset["specs"].values, original)

        controls.colormap_combo.setCurrentText("plasma")
        controls.reverse_check.setChecked(True)
        recolored = get_plot_state(self.owner)[0]
        self.assertEqual(recolored["raman_map"]["colormap"], "plasma")
        self.assertTrue(recolored["raman_map"]["reverse_colormap"])
        self.assertTrue(recolored["raman_map"]["applied"])
        self.assertNotIn("colormap", recolored["supported_controls"])
        self.assertNotIn("reverse_colormap", recolored["supported_controls"])
        np.testing.assert_array_equal(controls.values, values_before)
        np.testing.assert_array_equal(dataset["specs"].values, original)

        controls.a_high.setValue(3)
        changed = get_plot_state(self.owner)[0]["raman_map"]
        self.assertFalse(changed["applied"])
        self.assertEqual(changed["band_a"], [1.0, 3.0])
        self.assertIsNone(changed["valid_points"])

    def test_grid_map_state_follows_optional_wavenumber_calibration(self):
        calibration = PixelToWavenumberCalibration(
            [0.0, 10.0], [100.0, 200.0], degree=1
        )
        grid = show_plot(
            self.owner,
            GridScanPlotWindow(_grid_dataset(), spectral_calibration=calibration),
        )
        grid.show_wavenumber_check.setChecked(True)
        mapped = get_plot_state(self.owner)[0]["raman_map"]
        self.assertEqual(mapped["unit"], "cm⁻¹")
        np.testing.assert_allclose(mapped["band_a"], [100.0, 200.0])
        self.assertFalse(mapped["applied"])

    def test_unknown_or_wrong_panel_fields_fail_without_changes(self):
        panel = show_plot(self.owner, SpectrumWindow(np.arange(11.0)))

        with self.assertRaisesRegex(ValueError, "Unknown plot argument"):
            self.call(
                "configure_plot",
                panel_id=panel._plot_panel_id,
                white_background=True,
                mystery=True,
            )
        self.assertFalse(panel.white_background_check.isChecked())

        with self.assertRaisesRegex(ValueError, "detector_view is not supported"):
            self.call(
                "configure_plot",
                panel_id=panel._plot_panel_id,
                white_background=True,
                detector_view="spectrum",
            )
        self.assertFalse(panel.white_background_check.isChecked())


if __name__ == "__main__":
    unittest.main()
