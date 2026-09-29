"""Truthful progress reporting for calibration acquisition and saving."""

import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import xarray as xr

from napari_raman_widget.calibration.calibrator import Calibrator
from napari_raman_widget.acquisition.control import AcquisitionCancelled


class _Timing:
    def cfg_samp_clk_timing(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _Core:
    def __init__(self):
        self.auto_shutter = []
        self.snap_count = 0

    def setAutoShutter(self, enabled):
        self.auto_shutter.append(enabled)

    def snap(self):
        self.snap_count += 1
        return np.full((3, 4), self.snap_count, dtype=float)


class _Collector:
    def __init__(self, repeats):
        self.repeats = repeats
        self.voltages = []

    def collect_spectra_pts(self, voltages, exposure):
        self.voltages.append((voltages.copy(), exposure))
        return np.ones((self.repeats, 5), dtype=float)


class CalibratorProgressTests(unittest.TestCase):
    def make_calibrator(self):
        core = _Core()
        task = SimpleNamespace(
            out_stream=SimpleNamespace(output_buf_size=None),
            timing=_Timing(),
        )
        daq = SimpleNamespace(_galvo=task)
        collector = _Collector(repeats=2)
        calibrator = Calibrator(
            core,
            daq,
            transformer=object(),
            collector=collector,
            repeats=2,
            exposure=0.25,
        )
        return calibrator, core, collector

    def test_counts_only_accepted_points_then_processing_and_save(self):
        calibrator, core, collector = self.make_calibrator()
        saved = False
        events = []

        def fake_to_zarr(_dataset, _path):
            nonlocal saved
            saved = True

        def on_progress(completed, total, stage):
            events.append((completed, total, stage, saved))

        volts = np.array([[0.0, 0.0], [1.2, 0.0], [0.2, -0.3]])
        positions = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "napari_raman_widget.calibration.calibrator.time.sleep"
        ), patch.object(xr.Dataset, "to_zarr", new=fake_to_zarr):
            dataset = calibrator.collect_calibration_images(
                volts,
                threshold=1.0,
                relative_positions=positions,
                save_directory=temp_dir,
                progress_callback=on_progress,
            )

        self.assertEqual(dataset.sizes["idx"], 2)
        self.assertEqual(len(collector.voltages), 2)
        self.assertEqual(core.snap_count, 3)
        self.assertEqual(core.auto_shutter, [False, True])
        self.assertEqual(events[0][:2], (0, 4))
        self.assertEqual(
            [event[0] for event in events if event[2].startswith("Collecting")],
            [0, 1, 2],
        )
        self.assertIn((2, 4, "Processing calibration data", False), events)
        self.assertIn((3, 4, "Saving calibration dataset", False), events)
        self.assertEqual(events[-1], (4, 4, "Calibration dataset saved", True))
        self.assertFalse(
            any(completed == total for completed, total, _, _ in events[:-1])
        )

    def test_calibrate_forwards_progress_callback(self):
        core = Mock()
        core.getImageWidth.return_value = 20
        core.getImageHeight.return_value = 10
        daq = SimpleNamespace(galvo=Mock())
        transformer = Mock()
        transformer.BF_to_volts.return_value = np.zeros((1, 2))
        calibrator = Calibrator(
            core,
            daq,
            transformer,
            collector=Mock(),
            repeats=1,
            exposure=0.1,
        )
        result = xr.Dataset()
        callback = Mock()
        cancel_check = Mock(return_value=False)
        calibrator.collect_calibration_images = Mock(return_value=result)

        aiming = types.ModuleType("raman_mda_engine.aiming")
        grid = Mock()
        grid.get_current_points.return_value = np.array([[0.5, 0.5]])
        aiming.SimpleGridSource = Mock(return_value=grid)
        package = types.ModuleType("raman_mda_engine")
        package.aiming = aiming

        with patch.dict(
            sys.modules,
            {
                "raman_mda_engine": package,
                "raman_mda_engine.aiming": aiming,
            },
        ):
            self.assertIs(
                calibrator.calibrate(
                    N=1,
                    plot=False,
                    progress_callback=callback,
                    cancel_check=cancel_check,
                ),
                result,
            )

        self.assertIs(
            calibrator.collect_calibration_images.call_args.kwargs[
                "progress_callback"
            ],
            callback,
        )
        self.assertIs(
            calibrator.collect_calibration_images.call_args.kwargs["cancel_check"],
            cancel_check,
        )

    def test_cancelled_calibration_restores_auto_shutter_without_saving(self):
        calibrator, core, collector = self.make_calibrator()
        events = []

        def on_progress(*event):
            events.append(event)

        with patch(
            "napari_raman_widget.calibration.calibrator.time.sleep"
        ), patch.object(xr.Dataset, "to_zarr") as save:
            with self.assertRaises(AcquisitionCancelled):
                calibrator.collect_calibration_images(
                    np.zeros((3, 2)), threshold=1,
                    progress_callback=on_progress,
                    cancel_check=lambda: bool(events and events[-1][0] == 1),
                )
        self.assertEqual(len(collector.voltages), 1)
        self.assertEqual(core.snap_count, 1)
        self.assertEqual(core.auto_shutter, [False, True])
        self.assertEqual([event[0] for event in events], [0, 1])
        save.assert_not_called()

    def test_stop_before_save_does_not_write_or_report_complete(self):
        calibrator, core, _ = self.make_calibrator()
        events = []

        def on_progress(*event):
            events.append(event)

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "napari_raman_widget.calibration.calibrator.time.sleep"
        ), patch.object(xr.Dataset, "to_zarr") as save:
            with self.assertRaises(AcquisitionCancelled):
                calibrator.collect_calibration_images(
                    np.zeros((2, 2)), threshold=1,
                    save_directory=temp_dir,
                    progress_callback=on_progress,
                    cancel_check=lambda: bool(
                        events and events[-1][2] == "Saving calibration dataset"
                    ),
                )
        self.assertEqual(core.auto_shutter, [False, True])
        self.assertFalse(any(completed == total for completed, total, _ in events))
        save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
