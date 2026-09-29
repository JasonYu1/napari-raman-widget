"""Progress and safe cooperative stops for reference axial acquisition."""

import unittest
from unittest.mock import Mock, call, patch

import numpy as np

from napari_raman_widget.acquisition.autofocus import autofocus_w_bkd
from napari_raman_widget.acquisition.control import AcquisitionCancelled


class ReferenceAcquisitionProgressTests(unittest.TestCase):
    def setUp(self):
        self.core = Mock()
        self.core.getPosition.return_value = 10.0
        self.daq = Mock()
        self.collector = Mock()
        self.collector.collect_spectra_pts.return_value = np.ones((2, 5))
        self.volts = np.zeros((2, 2))
        self.sleep = patch("napari_raman_widget.acquisition.autofocus.time.sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def run_scan(self, **kwargs):
        return autofocus_w_bkd(
            self.core, self.daq, self.collector, self.volts,
            search_range=2, search_pts=3, exposure=0.1, **kwargs,
        )

    def test_reports_measured_z_positions_and_preserves_success_semantics(self):
        events = []
        initial_z, means, spectra = self.run_scan(
            progress_callback=lambda *event: events.append(event),
        )

        self.assertEqual(initial_z, 10)
        self.assertEqual(means.shape, (3, 5))
        self.assertEqual(spectra.shape, (3, 2, 5))
        self.assertEqual([event[:2] for event in events], [(0, 3), (1, 3), (2, 3), (3, 3)])
        self.assertEqual(self.core.setZPosition.call_args_list, [call(8), call(10), call(12)])
        self.core.setShutterOpen.assert_has_calls([
            call("Fluoshutter", True), call("Fluoshutter", False),
        ])

    def test_stop_after_one_batch_closes_shutter_and_restores_z(self):
        events = []

        def progress(*event):
            events.append(event)

        with self.assertRaises(AcquisitionCancelled) as caught:
            self.run_scan(
                progress_callback=progress,
                cancel_check=lambda: bool(events and events[-1][0] == 1),
            )

        self.assertEqual(self.collector.collect_spectra_pts.call_count, 1)
        self.assertEqual(self.core.setZPosition.call_args_list, [call(8), call(10)])
        self.core.setShutterOpen.assert_called_with("Fluoshutter", False)
        self.assertEqual([event[0] for event in events], [0, 1])
        partial = caught.exception.partial_result
        self.assertEqual(partial["initial_z"], 10)
        np.testing.assert_array_equal(partial["z_offsets"], [-2])
        self.assertEqual(partial["all_spectra"].shape, (1, 2, 5))

    def test_already_stopped_performs_no_hardware_actions(self):
        with self.assertRaises(AcquisitionCancelled):
            self.run_scan(cancel_check=lambda: True)
        self.assertEqual(self.core.mock_calls, [])
        self.assertEqual(self.daq.mock_calls, [])
        self.collector.collect_spectra_pts.assert_not_called()

    def test_camera_failure_closes_shutter_and_restores_z(self):
        self.collector.collect_spectra_pts.side_effect = RuntimeError("camera failed")
        with self.assertRaisesRegex(RuntimeError, "camera failed"):
            self.run_scan(progress_callback=Mock())
        self.core.setShutterOpen.assert_called_with("Fluoshutter", False)
        self.core.setZPosition.assert_called_with(10)

    def test_failed_shutter_close_still_restores_z(self):
        def set_shutter(_name, is_open):
            if not is_open:
                raise RuntimeError("shutter failed")

        self.core.setShutterOpen.side_effect = set_shutter
        with self.assertRaisesRegex(RuntimeError, "shutter closed failed"):
            self.run_scan(progress_callback=Mock())
        self.core.setZPosition.assert_called_with(10)


if __name__ == "__main__":
    unittest.main()
