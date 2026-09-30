"""Hardware-free checks for confirmed, stoppable grid and axial scans."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import xarray as xr
from qtpy.QtWidgets import QDialog

from napari_raman_widget.acquisition.control import AcquisitionCancelled
from napari_raman_widget.label_sampling import make_label_roi
from napari_raman_widget.scan_preview import build_axial_preview, build_grid_preview
from napari_raman_widget.scan_workflows import (
    acquire_grid,
    acquire_reference,
    start_grid_scan,
    start_reference,
)

_ORIGINAL_TO_ZARR = xr.Dataset.to_zarr


class _Core:
    def __init__(self):
        self.z = 100.0
        self.moves = []
        self.shutter = False
        self.shutter_calls = []
        self.actions = []
        self.snaps = 0

    def getPosition(self):
        return self.z

    def setPosition(self, z):
        self.z = z
        self.moves.append(z)
        self.actions.append(("move", z))

    setZPosition = setPosition

    def waitForSystem(self):
        pass

    def setExposure(self, exposure):
        self.actions.append(("exposure", exposure))

    def setConfig(self, group, channel):
        self.actions.append(("channel", group, channel))

    def setShutterOpen(self, name, enabled):
        self.shutter = enabled
        self.shutter_calls.append((name, enabled))
        self.actions.append(("shutter", enabled))

    def stopSequenceAcquisition(self):
        self.actions.append(("stop_sequence",))

    def snap(self):
        self.snaps += 1
        self.actions.append(("snap",))
        return np.full((2, 3), self.snaps, dtype=float)


class _Collector:
    def __init__(self):
        self.calls = []

    def collect_spectra_pts(self, points, exposure):
        points = np.asarray(points)
        self.calls.append((points.copy(), exposure))
        return np.column_stack((points, np.full(len(points), len(self.calls))))

    collect_spectra_image_points = collect_spectra_pts


class ScanWorkflowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.core = _Core()
        self.daq = SimpleNamespace(galvo=Mock())
        self.collector = _Collector()
        self.events = []
        self.stop = False
        self.imaging = Mock()
        self.raman = Mock(side_effect=lambda enabled: self.core.setShutterOpen("Fluoshutter", enabled))
        save_patch = patch.object(xr.Dataset, "to_zarr", autospec=True)
        self.save = save_patch.start()
        self.addCleanup(save_patch.stop)
        sleep_patch = patch("napari_raman_widget.acquisition.autofocus.time.sleep")
        sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    def progress(self, *event):
        self.events.append(event)

    def grid_plan(self):
        return build_grid_preview(
            [[10, 20], [30, 40]], total_points=4,
            z_offsets_um=[-1, 1], exposure_ms=2500,
            output_path=self.directory / "grid.zarr", z_offset_um=3,
        )

    def axial_plan(self):
        return build_axial_preview(
            [12, 34], repeats=2, search_range_um=2, search_points=3,
            exposure_ms=100, output_path=self.directory / "axial.zarr",
        )

    def run_grid(self, progress=None):
        plan = self.grid_plan()
        volts = np.asarray(plan.points_yx) / 100
        return acquire_grid(
            plan, self.core, self.daq, self.collector, volts,
            self.imaging, self.raman, progress or self.progress, lambda: self.stop,
        )

    def run_reference(self, progress=None, *, is_demo=False):
        return acquire_reference(
            self.axial_plan(), self.core, self.daq, self.collector,
            np.tile([0.1, 0.2], (2, 1)),
            progress or self.progress, lambda: self.stop, is_demo=is_demo,
        )

    def assert_safe(self):
        self.assertEqual(self.core.z, 100)
        self.assertFalse(self.core.shutter)

    def test_grid_preserves_coordinate_order_and_counts_completed_batches(self):
        result = self.run_grid()
        dataset = result["dataset"]
        expected_points = np.asarray(self.grid_plan().points_yx)
        np.testing.assert_array_equal(dataset["grid_pos"], expected_points)
        np.testing.assert_array_equal(dataset["laser_pos"], expected_points / 100)
        measured = np.concatenate([points for points, _ in self.collector.calls])
        np.testing.assert_array_equal(measured, np.tile(expected_points / 100, (2, 1)))
        self.assertEqual([len(points) for points, _ in self.collector.calls], [2] * 4)
        self.assertEqual([exposure for _, exposure in self.collector.calls], [2500] * 4)
        self.assertEqual(dataset["specs"].shape, (2, 4, 3))
        self.assertEqual(dataset["BF_z"].shape, (2, 2, 3))
        self.assertEqual([event[:2] for event in self.events], [(0, 8), (2, 8), (4, 8), (6, 8), (8, 8), (8, 8)])
        self.assertEqual(self.core.moves, [96, 98, 100])
        self.assertEqual(self.core.snaps, 4)
        self.assertFalse(result["stopped"])
        self.assertEqual(dataset.attrs["acquisition_status"], "complete")
        self.assertEqual(self.save.call_args.kwargs, {"mode": "w-"})
        self.assert_safe()

    def test_grid_stop_saves_only_real_samples_with_correct_plane_indices(self):
        def progress(*event):
            self.progress(*event)
            if event[0] == 6:
                self.stop = True

        result = self.run_grid(progress)
        dataset = result["dataset"]
        self.assertTrue(result["stopped"])
        self.assertEqual(dataset["specs"].shape, (6, 3))
        self.assertEqual(dataset["specs"].dims, ("sample", "spec_dim"))
        np.testing.assert_array_equal(dataset["sample_grid_index"], [0, 1, 2, 3, 0, 1])
        np.testing.assert_array_equal(dataset["sample_z_index"], [0, 0, 0, 0, 1, 1])
        self.assertEqual(dataset.attrs["completed_spectra"], 6)
        self.assertEqual(dataset.attrs["planned_spectra"], 8)
        self.assertEqual(dataset.attrs["acquisition_status"], "stopped")
        self.assertNotIn("end_BF", dataset)
        self.assertNotIn("BF_z", dataset)
        self.assertEqual(len(self.collector.calls), 3)
        self.save.assert_called_once()
        self.assert_safe()

    def assert_grid_batches(self, points_per_axis, exposure, expected_sizes):
        plan = build_grid_preview(
            [[10, 20], [30, 40]], total_points=points_per_axis ** 2,
            z_offsets_um=[0], exposure_ms=exposure,
            output_path=self.directory / "batch-grid.zarr",
        )
        volts = np.asarray(plan.points_yx) / 100
        result = acquire_grid(
            plan, self.core, self.daq, self.collector, volts,
            self.imaging, self.raman, self.progress, lambda: False,
        )
        sizes = [len(points) for points, _ in self.collector.calls]
        self.assertEqual(sizes, expected_sizes)
        self.assertTrue(all(size >= 2 for size in sizes))
        # Joining small/last batches must neither repeat nor drop grid points.
        measured = np.concatenate([points for points, _ in self.collector.calls])
        np.testing.assert_array_equal(measured, volts)
        self.assertEqual(result["dataset"].attrs["completed_spectra"], len(volts))
        self.assertEqual(result["dataset"]["specs"].shape[0], len(volts))
        self.assert_safe()

    def test_grid_long_exposure_still_uses_two_daq_samples_per_batch(self):
        self.assert_grid_batches(points_per_axis=2, exposure=6000, expected_sizes=[2, 2])

    def test_grid_trailing_singleton_merges_without_duplicate_measurements(self):
        self.assert_grid_batches(points_per_axis=3, exposure=1250, expected_sizes=[4, 5])

    def test_grid_already_stopped_does_not_touch_hardware_or_save(self):
        self.stop = True
        with self.assertRaises(AcquisitionCancelled):
            self.run_grid()
        self.assertEqual(self.core.actions, [])
        self.assertEqual(self.daq.galvo.mock_calls, [])
        self.imaging.assert_not_called()
        self.save.assert_not_called()

    def test_grid_camera_error_closes_shutter_restores_z_and_does_not_save(self):
        self.collector.collect_spectra_pts = Mock(side_effect=RuntimeError("camera failed"))
        with self.assertRaisesRegex(RuntimeError, "camera failed"):
            self.run_grid()
        self.assert_safe()
        self.save.assert_not_called()

    def test_grid_invalid_detector_count_does_not_save_fabricated_samples(self):
        self.collector.collect_spectra_pts = Mock(return_value=np.zeros((1, 3)))
        with self.assertRaisesRegex(ValueError, "unexpected number of spectra"):
            self.run_grid()
        self.assert_safe()
        self.save.assert_not_called()

    def test_grid_shutter_close_error_does_not_skip_z_restore(self):
        def raman(enabled):
            if not enabled:
                raise RuntimeError("shutter failed")
            self.core.setShutterOpen("Fluoshutter", enabled)

        self.raman.side_effect = raman
        with self.assertRaisesRegex(RuntimeError, "shutter failed"):
            self.run_grid()
        self.assertEqual(self.core.z, 100)
        self.save.assert_not_called()

    def test_grid_save_failure_occurs_after_hardware_cleanup(self):
        self.save.side_effect = OSError("disk full")
        with self.assertRaisesRegex(OSError, "disk full"):
            self.run_grid()
        self.assert_safe()

    def test_reference_complete_preserves_all_planes_and_restores_z(self):
        result = self.run_reference()
        dataset = result["dataset"]
        np.testing.assert_array_equal(result["zs"], [-2, 0, 2])
        self.assertEqual(dataset["spec"].shape, (3, 2, 3))
        self.assertEqual(dataset.coords["x"].item(), 34)
        self.assertEqual(dataset.coords["y"].item(), 12)
        self.assertEqual(dataset.attrs["planned_z_planes"], 3)
        self.assertEqual(dataset.attrs["acquisition_status"], "complete")
        self.assertEqual(self.core.moves, [98, 100, 102, 100])
        self.assert_safe()
        self.save.assert_called_once()

    def test_reference_stop_preserves_only_measured_planes(self):
        def progress(*event):
            self.progress(*event)
            if event[0] == 1:
                self.stop = True

        result = self.run_reference(progress)
        self.assertTrue(result["stopped"])
        self.assertEqual(result["spectra"].shape, (1, 2, 3))
        np.testing.assert_array_equal(result["zs"], [-2])
        self.assertEqual(result["dataset"].attrs["acquisition_status"], "stopped")
        self.assertEqual(result["dataset"].attrs["planned_z_planes"], 3)
        self.assertEqual(len(self.collector.calls), 1)
        self.assert_safe()
        self.save.assert_called_once()

    def test_reference_camera_error_restores_hardware_without_saving(self):
        self.collector.collect_spectra_pts = Mock(side_effect=RuntimeError("camera failed"))
        with self.assertRaisesRegex(RuntimeError, "camera failed"):
            self.run_reference()
        self.assert_safe()
        self.save.assert_not_called()

    def test_demo_reference_stop_preserves_measurements_and_restores_z(self):
        def progress(*event):
            self.progress(*event)
            if event[0] == 1:
                self.stop = True

        result = self.run_reference(progress, is_demo=True)
        self.assertTrue(result["stopped"])
        self.assertEqual(result["spectra"].shape, (1, 2, 3))
        np.testing.assert_array_equal(self.collector.calls[0][0], [[12, 34], [12, 34]])
        self.assert_safe()

    def test_complete_multiz_grid_round_trips_through_real_zarr(self):
        self.save.side_effect = _ORIGINAL_TO_ZARR
        result = self.run_grid()
        with xr.open_zarr(result["path"]) as saved:
            saved.load()
            xr.testing.assert_identical(saved, result["dataset"])
            self.assertEqual(saved["specs"].dims, ("z", "idx", "spec_dim"))
            self.assertEqual(saved["specs"].shape, (2, 4, 3))
            self.assertEqual(saved["BF_z"].dims, ("z", "Y", "X"))
            np.testing.assert_array_equal(saved["z_range"], [-1, 1])
            self.assertEqual(saved.attrs["acquisition_status"], "complete")
            self.assertEqual(saved.attrs["completed_spectra"], 8)
            self.assertEqual(saved.attrs["channel_exposures_ms"], {})
            self.assertEqual(saved.attrs["roi"]["shape_type"], "rectangle")
            self.assertTrue(saved.attrs["roi"]["inside_only"])
            self.assertEqual(saved.attrs["sampling_mode"], "count")
            self.assertEqual(saved.attrs["sampling_pattern"], "square_grid")
            self.assertEqual(saved.attrs["points_per_z"], 4)
            self.assertEqual(saved.attrs["requested_points_per_z"], 4)
            pitch = self.grid_plan().spacing_px
            self.assertGreater(pitch, 0)
            self.assertEqual(saved.attrs["point_spacing_px"], pitch)
            for axis in (0, 1):
                np.testing.assert_allclose(np.diff(np.unique(saved["grid_pos"].values[:, axis])), pitch)
        self.assert_safe()

    def test_count_target_round_trip_keeps_complete_regular_grid_and_actual_count(self):
        self.save.side_effect = _ORIGINAL_TO_ZARR
        plan = build_grid_preview(
            [[10, 20], [30, 40]], shape_type="rectangle",
            sampling_mode="count", total_points=7,
            z_offsets_um=[0], exposure_ms=1000,
            output_path=self.directory / "approximate-count-grid.zarr",
        )
        points = np.asarray(plan.points_yx)
        rows, columns = np.unique(points[:, 0]), np.unique(points[:, 1])
        # A full square lattice cannot have seven points in a square ROI:
        # choose the nearest attainable count, without trimming or padding.
        self.assertEqual(len(points), 9)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(columns), 3)
        self.assertEqual(len(np.unique(points, axis=0)), len(points))
        expected_grid = np.array([[row, column] for column in columns for row in rows])
        np.testing.assert_array_equal(points, expected_grid)
        np.testing.assert_allclose(np.diff(rows), plan.spacing_px)
        np.testing.assert_allclose(np.diff(columns), plan.spacing_px)
        self.assertEqual(plan.requested_point_count, 7)
        self.assertEqual(plan.spectrum_count, 9)
        volts = points / 100

        result = acquire_grid(
            plan, self.core, self.daq, self.collector, volts,
            self.imaging, self.raman, self.progress, lambda: False,
        )

        measured = np.concatenate([batch for batch, _ in self.collector.calls])
        np.testing.assert_array_equal(measured, volts)
        self.assertEqual(self.events[-1][:2], (9, 9))
        with xr.open_zarr(result["path"]) as saved:
            saved.load()
            xr.testing.assert_identical(saved, result["dataset"])
            np.testing.assert_array_equal(saved["grid_pos"], points)
            np.testing.assert_array_equal(saved["specs"].values[:, :2], volts)
            self.assertEqual(saved.attrs["sampling_mode"], "count")
            self.assertEqual(saved.attrs["sampling_pattern"], "square_grid")
            self.assertEqual(saved.attrs["requested_points_per_z"], 7)
            self.assertEqual(saved.attrs["points_per_z"], 9)
            self.assertEqual(saved.attrs["completed_spectra"], 9)
            self.assertEqual(saved.attrs["planned_spectra"], 9)
            self.assertEqual(saved.attrs["point_spacing_px"], plan.spacing_px)
        self.assert_safe()

    def test_count_target_concave_roi_keeps_every_lattice_point_inside_shape(self):
        vertices = [[10, 20], [10, 26], [12, 26], [12, 22], [16, 22], [16, 20]]
        plan = build_grid_preview(
            vertices, shape_type="polygon", sampling_mode="count", total_points=14,
            z_offsets_um=[0], exposure_ms=1000,
            output_path=self.directory / "concave-count.zarr",
        )
        pitch = plan.spacing_px
        axis_steps = np.arange(int(np.floor(6 / pitch + 1e-10)) + 1) * pitch
        expected_points = np.array([
            [row, column]
            for column in 20 + axis_steps
            for row in 10 + axis_steps
            if row <= 12 + 1e-10 or column <= 22 + 1e-10
        ])
        np.testing.assert_allclose(plan.points_yx, expected_points, atol=1e-10)
        self.assertEqual(plan.requested_point_count, 14)
        self.assertEqual(plan.spectrum_count, len(expected_points))
        volts = expected_points / 100

        result = acquire_grid(
            plan, self.core, self.daq, self.collector, volts,
            self.imaging, self.raman, self.progress, lambda: False,
        )

        measured = np.concatenate([batch for batch, _ in self.collector.calls])
        np.testing.assert_allclose(measured, volts)
        np.testing.assert_allclose(result["dataset"]["grid_pos"], expected_points)
        self.assertEqual(result["dataset"].attrs["points_per_z"], len(expected_points))
        self.assertEqual(result["dataset"].attrs["requested_points_per_z"], 14)
        self.assertEqual(result["dataset"].attrs["point_spacing_px"], pitch)
        self.assert_safe()

    def test_concave_roi_spacing_acquires_exact_points_and_round_trips_metadata(self):
        self.save.side_effect = _ORIGINAL_TO_ZARR
        vertices = [[10, 20], [10, 26], [12, 26], [12, 22], [16, 22], [16, 20]]
        # Independent lattice oracle: the concave L excludes the lower-right
        # four candidates of its 4-by-4 bounding-box grid.
        expected_points = np.array([
            [row, column]
            for column in (20, 22, 24, 26)
            for row in (10, 12, 14, 16)
            if row <= 12 or column <= 22
        ], dtype=float)
        self.assertEqual(len(expected_points), 12)
        plan = build_grid_preview(
            vertices, shape_type="polygon", sampling_mode="spacing", spacing_px=2,
            z_offsets_um=[-1, 1], exposure_ms=1000,
            output_path=self.directory / "concave-spacing.zarr",
            layer_name="Cell-shaped ROI", image_layer_name="Live camera",
        )
        np.testing.assert_array_equal(plan.points_yx, expected_points)
        volts = expected_points / 100

        result = acquire_grid(
            plan, self.core, self.daq, self.collector, volts,
            self.imaging, self.raman, self.progress, lambda: False,
        )

        measured = np.concatenate([points for points, _ in self.collector.calls])
        np.testing.assert_array_equal(measured, np.tile(volts, (2, 1)))
        self.assertEqual([len(points) for points, _ in self.collector.calls], [5, 5, 2] * 2)
        self.assertEqual(self.events[-1][:2], (24, 24))
        self.assertFalse(result["stopped"])
        with xr.open_zarr(result["path"]) as saved:
            saved.load()
            xr.testing.assert_identical(saved, result["dataset"])
            np.testing.assert_array_equal(saved["grid_pos"], expected_points)
            np.testing.assert_array_equal(saved["laser_pos"], volts)
            self.assertEqual(saved["specs"].shape, (2, 12, 3))
            # This fake detector encodes the exact addressed voltage into the
            # first two spectral values, verifying point order in each plane.
            np.testing.assert_array_equal(saved["specs"].values[:, :, :2], np.tile(volts, (2, 1, 1)))
            self.assertEqual(saved.attrs["sampling_mode"], "spacing")
            self.assertEqual(saved.attrs["sampling_pattern"], "square_grid")
            self.assertEqual(saved.attrs["point_spacing_px"], 2.0)
            self.assertIsNone(saved.attrs["requested_points_per_z"])
            self.assertEqual(saved.attrs["points_per_z"], 12)
            self.assertEqual(saved.attrs["completed_spectra"], 24)
            self.assertEqual(saved.attrs["planned_spectra"], 24)
            self.assertEqual(saved.attrs["roi"], {
                "shape_type": "polygon", "vertices_yx": vertices,
                "image_layer": "Live camera", "source_layer": "Cell-shaped ROI",
                "inside_only": True,
            })
        self.assert_safe()

    def test_partial_grid_round_trip_retains_real_indices_and_stop_metadata(self):
        self.save.side_effect = _ORIGINAL_TO_ZARR

        def progress(*event):
            self.progress(*event)
            if event[0] == 6:
                self.stop = True

        result = self.run_grid(progress)
        with xr.open_zarr(result["path"]) as saved:
            saved.load()
            xr.testing.assert_identical(saved, result["dataset"])
            self.assertEqual(saved["specs"].dims, ("sample", "spec_dim"))
            self.assertEqual(saved.sizes["sample"], 6)
            np.testing.assert_array_equal(saved["sample_grid_index"], [0, 1, 2, 3, 0, 1])
            np.testing.assert_array_equal(saved["sample_z_index"], [0, 0, 0, 0, 1, 1])
            self.assertEqual(saved.attrs["acquisition_status"], "stopped")
            self.assertEqual(saved.attrs["completed_spectra"], 6)
            self.assertEqual(saved.attrs["planned_spectra"], 8)
            self.assertNotIn("end_BF", saved)
            self.assertNotIn("BF_z", saved)
        self.assert_safe()

    def test_all_labels_scan_saves_per_point_ids_and_source_metadata_in_real_zarr(self):
        self.save.side_effect = _ORIGINAL_TO_ZARR
        labels = np.zeros((20, 24), dtype=np.uint64)
        labels[2:8, 3:10] = 7
        large_id = 2**63 + 29
        labels[12:18, 14:22] = large_id
        labels[3:5, 5:7] = 0
        original = labels.copy()
        roi = make_label_roi(labels)
        plan = build_grid_preview(
            None, label_roi=roi, sampling_mode="spacing", spacing_px=2,
            z_offsets_um=[-1, 1], exposure_ms=1000,
            output_path=self.directory / "all-labels.zarr", layer_name="Cells",
        )
        grid = np.asarray(plan.points_yx)
        volts = grid / 100
        result = acquire_grid(
            plan, self.core, self.daq, self.collector, volts,
            self.imaging, self.raman, self.progress, lambda: False,
        )
        measured = np.concatenate([batch for batch, _ in self.collector.calls])
        np.testing.assert_array_equal(measured, np.tile(volts, (2, 1)))
        with xr.open_zarr(result["path"]) as saved:
            saved.load()
            xr.testing.assert_identical(saved, result["dataset"])
            np.testing.assert_array_equal(saved["grid_pos"], grid)
            np.testing.assert_array_equal(saved["label_id"], roi.labels_at(grid))
            self.assertEqual(saved["label_id"].dtype, np.uint64)
            self.assertEqual(set(int(v) for v in saved["label_id"].values), {7, large_id})
            self.assertEqual(saved["label_id"].dims, ("idx",))
            self.assertEqual(saved.attrs["roi"]["selection"], "all_nonzero_labels")
            self.assertEqual(saved.attrs["roi"]["label_values"], [7, large_id])
            self.assertEqual(saved.attrs["roi"]["sampled_label_count"], 2)
            self.assertEqual(saved.attrs["planned_spectra"], 2 * len(grid))
        np.testing.assert_array_equal(labels, original)
        self.assert_safe()

    def test_partial_axial_round_trip_retains_only_completed_z_planes(self):
        self.save.side_effect = _ORIGINAL_TO_ZARR

        def progress(*event):
            self.progress(*event)
            if event[0] == 1:
                self.stop = True

        result = self.run_reference(progress)
        with xr.open_zarr(result["path"]) as saved:
            saved.load()
            xr.testing.assert_identical(saved, result["dataset"])
            self.assertEqual(saved["spec"].dims, ("z", "n", "pixel"))
            self.assertEqual(saved["spec"].shape, (1, 2, 3))
            np.testing.assert_array_equal(saved.coords["z"], [-2])
            self.assertEqual(saved.attrs["acquisition_status"], "stopped")
            self.assertEqual(saved.attrs["planned_z_planes"], 3)
            self.assertEqual(saved.attrs["initial_z_um"], 100)
        self.assert_safe()


class RejectedScanPreviewTests(unittest.TestCase):
    def make_owner(self, layer):
        jobs = Mock()
        jobs.available.return_value = True
        return SimpleNamespace(
            _acquisition_jobs=jobs, status=Mock(), core=Mock(), daq=Mock(),
            collector=Mock(), transformer=Mock(),
            viewer=SimpleNamespace(layers=SimpleNamespace(selection=SimpleNamespace(active=layer))),
            _get_image_xy=Mock(return_value=(100, 100)), _pt_to_volts=Mock(),
            _raman_point_from_active_layer=Mock(return_value=(np.array([12, 34]), 0)),
        )

    def spacing_owner(self, layer, *, dimensions=100, spacing=1):
        owner = self.make_owner(layer)
        owner._get_image_xy.return_value = (dimensions, dimensions)
        owner.scan_name_input = Mock(text=lambda: "boundary-grid")
        owner.scan_zscan_check = Mock(isChecked=lambda: False)
        owner.scan_sampling_mode_combo = Mock(currentData=lambda: "spacing")
        owner.scan_total_points_input = Mock(value=lambda: 4)
        owner.scan_spacing_input = Mock(value=lambda: spacing)
        owner.scan_exp_input = Mock(value=lambda: 100)
        owner.scan_z_input = Mock(value=lambda: 0)
        owner.channel_rows = []
        return owner

    def assert_untouched(self, owner, *, image_size_read=False):
        self.assertEqual(owner.core.mock_calls, [])
        self.assertEqual(owner.daq.mock_calls, [])
        self.assertEqual(owner.collector.mock_calls, [])
        self.assertEqual(owner.transformer.mock_calls, [])
        owner._acquisition_jobs.start.assert_not_called()
        if image_size_read:
            owner._get_image_xy.assert_called_once()
        else:
            owner._get_image_xy.assert_not_called()
        owner._pt_to_volts.assert_not_called()
        self.assertIn("preview cancelled", owner.status.setText.call_args.args[0])

    def test_rejected_reference_preview_never_starts_acquisition(self):
        owner = self.make_owner(SimpleNamespace(name="Cells"))
        owner.ref_name_input = Mock(text=lambda: "reference")
        owner.ref_n_input = Mock(value=lambda: 2)
        owner.ref_range_input = Mock(value=lambda: 10)
        owner.ref_pts_input = Mock(value=lambda: 3)
        owner.ref_exp_input = Mock(value=lambda: 100)
        with patch("napari_raman_widget.scan_workflows.ScanPreviewDialog") as dialog:
            dialog.return_value.exec.return_value = QDialog.Rejected
            start_reference(owner)
            dialog.return_value.deleteLater.assert_called_once()
        self.assert_untouched(owner)

    def test_rejected_grid_preview_never_starts_acquisition(self):
        class Shapes:
            data = [np.array([[10, 20], [30, 40]])]
            selected_data = {0}
            shape_type = ["rectangle"]
            name = "Scan region"
            mode = "select"

        owner = self.make_owner(Shapes())
        owner.scan_name_input = Mock(text=lambda: "grid")
        owner.scan_zscan_check = Mock(isChecked=lambda: True)
        owner.scan_zrange_input = Mock(value=lambda: 2)
        owner.scan_zsteps_input = Mock(value=lambda: 3)
        owner.scan_sampling_mode_combo = Mock(currentData=lambda: "count")
        owner.scan_total_points_input = Mock(value=lambda: 4)
        owner.scan_spacing_input = Mock(value=lambda: 10)
        owner.scan_exp_input = Mock(value=lambda: 100)
        owner.scan_z_input = Mock(value=lambda: 0)
        owner.channel_rows = []
        with patch.dict("sys.modules", {"napari.layers": SimpleNamespace(Shapes=Shapes)}), patch(
            "napari_raman_widget.scan_workflows.ScanPreviewDialog"
        ) as dialog, patch("napari_raman_widget.scan_workflows.scan_image_reference", return_value=None):
            dialog.return_value.exec.return_value = QDialog.Rejected
            start_grid_scan(owner)
            dialog.return_value.deleteLater.assert_called_once()
        self.assert_untouched(owner, image_size_read=True)
        self.assertEqual(owner.viewer.layers.selection.active.mode, "select")
        self.assertEqual(owner.viewer.layers.selection.active.selected_data, {0})

    def test_camera_edge_affine_roundtrip_residue_reaches_preview_without_clipping(self):
        matrix = np.array([[1.0, 0.3], [0.2, 1.0]])
        translate = np.array([100.3, 200.7])

        class Shapes:
            data = [np.array([[0, 0], [0, 127], [127, 127], [127, 0]], dtype=float)]
            selected_data = {0}
            shape_type = ["rectangle"]
            name = "Full frame ROI"
            mode = "select"

            def data_to_world(self, point):
                return matrix @ point + translate

        image = SimpleNamespace(
            ndim=2, name="Transformed camera",
            world_to_data=lambda point: np.linalg.solve(matrix, np.asarray(point) - translate),
        )
        owner = self.spacing_owner(Shapes(), dimensions=128)
        with patch.dict("sys.modules", {"napari.layers": SimpleNamespace(Shapes=Shapes)}), patch(
            "napari_raman_widget.scan_workflows.ScanPreviewDialog"
        ) as dialog, patch("napari_raman_widget.scan_workflows.scan_image_reference", return_value=image):
            dialog.return_value.exec.return_value = QDialog.Rejected
            start_grid_scan(owner)
            dialog.assert_called_once()
            plan = dialog.call_args.args[0]

        self.assertEqual(len(plan.points_yx), 128 * 128)
        roi_vertices = np.asarray(plan.roi_vertices_yx)
        # The exact sign of a few-ULP affine residue depends on BLAS; explicit
        # positive/negative sub-tolerance residues are covered below as well.
        self.assertLess(roi_vertices.max(), 127.0 + 1e-7)
        np.testing.assert_allclose(roi_vertices, Shapes.data[0], atol=1e-12)
        self.assert_untouched(owner, image_size_read=True)
        self.assertEqual(owner.viewer.layers.selection.active.mode, "select")
        self.assertEqual(owner.viewer.layers.selection.active.selected_data, {0})

    def test_subtolerance_camera_edge_residue_is_not_clipped_or_resampled(self):
        class Shapes:
            selected_data = {0}
            shape_type = ["rectangle"]
            name = "Edge ROI"
            mode = "select"

        for residue in (-5e-8, 5e-8):
            with self.subTest(residue=residue):
                layer = Shapes()
                layer.data = [np.array([[0, 0], [0, 99], [99, 99], [99, 0]], dtype=float) + residue]
                owner = self.spacing_owner(layer, spacing=33)
                with patch.dict("sys.modules", {"napari.layers": SimpleNamespace(Shapes=Shapes)}), patch(
                    "napari_raman_widget.scan_workflows.ScanPreviewDialog"
                ) as dialog, patch("napari_raman_widget.scan_workflows.scan_image_reference", return_value=None):
                    dialog.return_value.exec.return_value = QDialog.Rejected
                    start_grid_scan(owner)
                    dialog.assert_called_once()
                    plan = dialog.call_args.args[0]

                points = np.asarray(plan.points_yx)
                self.assertEqual(len(points), 16)
                self.assertAlmostEqual(points.min(), residue, places=12)
                self.assertAlmostEqual(points.max(), 99 + residue, places=12)
                self.assert_untouched(owner, image_size_read=True)

    def test_real_outside_camera_coordinate_is_rejected_before_preview(self):
        class Shapes:
            selected_data = {0}
            shape_type = ["rectangle"]
            name = "Outside ROI"
            mode = "select"

        for offset in (0.001, 1.001):
            with self.subTest(offset=offset):
                layer = Shapes()
                layer.data = [np.array([[0, 0], [0, 99], [99, 99], [99, 0]], dtype=float) + offset]
                owner = self.spacing_owner(layer, spacing=33)
                with patch.dict("sys.modules", {"napari.layers": SimpleNamespace(Shapes=Shapes)}), patch(
                    "napari_raman_widget.scan_workflows.ScanPreviewDialog"
                ) as dialog, patch("napari_raman_widget.scan_workflows.scan_image_reference", return_value=None):
                    start_grid_scan(owner)
                    dialog.assert_not_called()

                self.assertIn("outside the camera image", owner.status.setText.call_args.args[0])
                owner._acquisition_jobs.start.assert_not_called()
                self.assertEqual(owner.core.mock_calls, [])
                self.assertEqual(owner.daq.mock_calls, [])
                self.assertEqual(owner.collector.mock_calls, [])
                self.assertEqual(owner.transformer.mock_calls, [])
                self.assertEqual(layer.mode, "select")
                self.assertEqual(layer.selected_data, {0})


if __name__ == "__main__":
    unittest.main()
