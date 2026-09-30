"""Labels-to-Points generation uses the exact confirmed acquisition snapshot."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from napari_raman_widget.scan_workflows import start_grid_scan


class _Labels:
    def __init__(self):
        self.data = np.zeros((20, 24), dtype=np.int32)
        self.data[1:8, 2:9] = 3
        self.data[3:5, 4:6] = 0
        self.data[12:18, 14:22] = 11
        self.ndim = 2
        self.name = "Cell labels"
        self.selected_label = 3
        self.show_selected_label = True
        self.mode = "paint"

    def data_to_world(self, point):
        return np.array([[2, .4], [.3, 1.5]]) @ point + [100, 200]


class LabelScanWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.labels = _Labels()
        self.original = self.labels.data.copy()
        self.events = []
        self.image = SimpleNamespace(
            name="Camera", ndim=2,
            world_to_data=lambda p: np.linalg.solve([[2, .4], [.3, 1.5]], np.asarray(p) - [100, 200]),
        )
        self.points_layer = SimpleNamespace(editable=True)
        self.owner = SimpleNamespace(
            core=Mock(), daq=Mock(), collector=Mock(), transformer=Mock(), status=Mock(),
            _acquisition_jobs=Mock(), _get_image_xy=Mock(return_value=(24, 20)),
            viewer=SimpleNamespace(
                layers=SimpleNamespace(selection=SimpleNamespace(active=self.labels)),
                add_points=Mock(return_value=self.points_layer)),
            scan_name_input=Mock(text=lambda: "label-grid"),
            scan_zscan_check=Mock(isChecked=lambda: False),
            scan_sampling_mode_combo=Mock(currentData=lambda: "spacing"),
            scan_total_points_input=Mock(value=lambda: 400),
            scan_spacing_input=Mock(value=lambda: 2),
            scan_exp_input=Mock(value=lambda: 100), scan_z_input=Mock(value=lambda: 0),
            channel_rows=[], spectral_calibration=None,
            _set_scan_imaging_channel=Mock(), _set_scan_raman_mode=Mock(),
        )
        self.owner.core.isSequenceRunning.return_value = False
        self.owner.core.stopSequenceAcquisition.side_effect = lambda: self.events.append("stop live")
        self.owner._acquisition_jobs.available.return_value = True
        self.owner.transformer.BF_to_volts.side_effect = lambda points, **kwargs: points
        self.owner._acquisition_jobs.start.side_effect = lambda *args: self.events.append("start worker")

    def run_scan(self, accepted=True, mutate_after_preview=False):
        def confirm(owner, plan):
            self.plan = plan
            self.events.append("preview")
            if mutate_after_preview:
                self.labels.data[:] = 0
            return accepted

        with patch.dict("sys.modules", {"napari.layers": SimpleNamespace(Shapes=type("Shapes", (), {}), Labels=_Labels)}), patch(
            "napari_raman_widget.scan_workflows.scan_image_reference", return_value=self.image
        ), patch("napari_raman_widget.scan_workflows._confirm", side_effect=confirm):
            start_grid_scan(self.owner)

    def test_start_creates_readonly_transformed_points_for_all_ids_without_changing_labels(self):
        self.run_scan()
        self.assertEqual(self.events, ["preview", "stop live", "start worker"])
        self.owner.viewer.add_points.assert_called_once()
        call = self.owner.viewer.add_points.call_args
        grid = np.asarray(self.plan.points_yx)
        np.testing.assert_array_equal(call.args[0], grid)
        np.testing.assert_allclose(call.kwargs["affine"], [[2, .4, 100], [.3, 1.5, 200], [0, 0, 1]])
        self.assertEqual(set(call.kwargs["features"]["label_id"]), {3, 11})
        self.assertTrue(call.kwargs["metadata"]["all_nonzero_labels"])
        self.assertFalse(self.points_layer.editable)
        np.testing.assert_allclose(self.owner.transformer.BF_to_volts.call_args.args[0], grid / [20, 24])
        np.testing.assert_array_equal(self.labels.data, self.original)
        self.assertEqual(self.labels.mode, "paint")
        self.assertEqual(self.labels.selected_label, 3)
        self.assertIs(self.owner.viewer.layers.selection.active, self.labels)

    def test_cancel_preserves_source_live_mode_and_layer_list(self):
        self.run_scan(accepted=False)
        self.assertEqual(self.events, ["preview"])
        self.owner.viewer.add_points.assert_not_called()
        self.owner.core.stopSequenceAcquisition.assert_not_called()
        self.owner._acquisition_jobs.start.assert_not_called()
        np.testing.assert_array_equal(self.labels.data, self.original)

    def test_later_mask_edits_cannot_change_confirmed_scan_points_or_label_ids(self):
        self.run_scan(mutate_after_preview=True)
        np.testing.assert_array_equal(self.plan.label_roi.labels, self.original)
        self.assertEqual(set(self.owner.viewer.add_points.call_args.kwargs["features"]["label_id"]), {3, 11})
        self.owner._acquisition_jobs.start.assert_called_once()
        self.assertFalse(self.labels.data.any())  # No restoration/overwrite of user edits.

    def test_empty_labels_fail_before_live_stop_or_points_generation(self):
        self.labels.data[:] = 0
        self.run_scan()
        self.assertEqual(self.events, [])
        self.owner.viewer.add_points.assert_not_called()
        self.owner._acquisition_jobs.start.assert_not_called()
        self.assertIn("no non-zero labels", self.owner.status.setText.call_args.args[0])

    def test_live_stop_error_prevents_points_generation_and_acquisition(self):
        self.owner.core.stopSequenceAcquisition.side_effect = RuntimeError("camera failed to stop")
        self.run_scan()
        self.owner.viewer.add_points.assert_not_called()
        self.owner._acquisition_jobs.start.assert_not_called()
        np.testing.assert_array_equal(self.labels.data, self.original)
        self.assertIn("camera failed to stop", self.owner.status.setText.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
