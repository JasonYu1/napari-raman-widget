"""Previewed scan workflows: GUI snapshots first, hardware work in a worker."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from qtpy.QtWidgets import QDialog

from .acquisition import autofocus_w_bkd
from .acquisition.control import AcquisitionCancelled, check_cancelled
from .calibration import Calibrator
from .plot_windows import CalibrationPlotWindow, GridScanPlotWindow, ReferenceSpectraWindow
from .scan_preview import ScanPreviewDialog, build_axial_preview, build_grid_preview
from .spatial_mapping import scan_image_reference, snapshot_scan_labels, snapshot_scan_roi, snapshot_scan_shape


def _ready(owner):
    if not owner._acquisition_jobs.available():
        return False
    if any(getattr(owner, name, None) is None for name in ("core", "daq", "collector")):
        owner.status.setText("Status: not connected")
        return False
    if owner.transformer is None:
        owner.status.setText("Status: no transformer loaded")
        return False
    return True


def _filename(label, prefix, directory=Path(".")):
    label = str(label).strip()
    if not label or any(character in label for character in '<>:"/\\|?*') or label in {".", ".."}:
        raise ValueError("Enter a file label without path separators or reserved characters.")
    return (directory / f"{prefix}{label}_{uuid.uuid4().hex[:8]}.zarr").resolve()


def _confirm(owner, plan):
    dialog = ScanPreviewDialog(plan, owner)
    try:
        accepted = dialog.exec() == QDialog.Accepted
    finally:
        dialog.deleteLater()
    if not accepted:
        owner.status.setText("Status: preview cancelled — no acquisition started")
    return accepted


def start_calibration(owner):
    if not _ready(owner):
        return
    calibrator = Calibrator(
        owner.core, owner.daq, owner.transformer, owner.collector,
        repeats=int(owner.cal_n_input.value()),
        exposure=float(owner.cal_exp_input.value()),
        max_volts=float(owner.cal_volts_input.value()),
    )
    grid = int(owner.cal_grid_input.value())
    threshold = float(owner.cal_thres_input.value())
    directory = str(Path.cwd())
    spectral_calibration = owner.spectral_calibration

    def operation(progress, cancelled):
        dataset = calibrator.calibrate(
            grid, threshold=threshold, plot=False, save_directory=directory,
            progress_callback=progress, cancel_check=cancelled,
        )
        return {"dataset": dataset}

    def complete(result):
        owner.calibrator = calibrator
        owner.calibration_ds = result["dataset"]
        owner._show_plot(CalibrationPlotWindow(
            result["dataset"], title="Calibration result",
            spectral_calibration=spectral_calibration,
        ))
        owner.status.setText("Status: calibration done OK")

    owner._acquisition_jobs.start("Calibration", operation, complete)


def start_reference(owner):
    if not _ready(owner):
        return
    try:
        name = owner.ref_name_input.text().strip()
        output_path = _filename(name, "", Path("reference"))
        point, _point_index = owner._raman_point_from_active_layer()
        plan = build_axial_preview(
            point, repeats=int(owner.ref_n_input.value()),
            search_range_um=float(owner.ref_range_input.value()),
            search_points=int(owner.ref_pts_input.value()),
            exposure_ms=float(owner.ref_exp_input.value()), output_path=output_path,
            layer_name=owner.viewer.layers.selection.active.name,
        )
        if not _confirm(owner, plan):
            return
        core, daq, collector = owner.core, owner.daq, owner.collector
        is_demo = type(owner).__name__ == "DemoWidget"
        frozen_point = np.asarray(plan.points_yx[0])
        voltages = None if is_demo else np.repeat(owner._pt_to_volts(frozen_point), plan.repeats, axis=0)
        spectral_calibration = owner.spectral_calibration

        def operation(progress, cancelled):
            return acquire_reference(
                plan, core, daq, collector, voltages, progress, cancelled,
                is_demo=is_demo,
            )

        def complete(result):
            label = "Partial axial scan" if result["stopped"] else "Reference spectra"
            owner._show_plot(ReferenceSpectraWindow(
                result["spectra"], result["zs"], title=f"{label}: {name}",
                spectral_calibration=spectral_calibration,
            ))
            owner.status.setText(f"Status: {label.lower()} saved → {result['path']}")

        owner._acquisition_jobs.start("Axial background scan", operation, complete)
    except Exception as error:
        owner.status.setText(f"Status: cannot prepare axial scan — {error}")


def acquire_reference(plan, core, daq, collector, voltages, progress, cancelled, *, is_demo=False):
    """Return complete or explicitly partial measured Z planes, never guessed data."""
    stopped = False
    zs = np.asarray(plan.z_offsets_um)
    if is_demo:
        check_cancelled(cancelled)
        initial_z = core.getPosition()
        spectra = []
        try:
            progress(0, len(zs), "Starting axial background scan")
            for index, offset in enumerate(zs):
                check_cancelled(cancelled)
                core.setZPosition(initial_z + offset)
                spectra.append(collector.collect_spectra_image_points(
                    np.tile(plan.points_yx[0], (plan.repeats, 1)), plan.exposure_ms,
                ))
                progress(index + 1, len(zs), f"Z position {index + 1}/{len(zs)} ({offset:+g} µm)")
                check_cancelled(cancelled)
        except AcquisitionCancelled:
            stopped = True
            if not spectra:
                raise
        finally:
            core.setZPosition(initial_z)
        spectra = np.asarray(spectra)
        zs = zs[:len(spectra)]
    else:
        try:
            initial_z, _mean, spectra = autofocus_w_bkd(
                core, daq, collector, voltages,
                search_range=max(abs(zs)), search_pts=len(zs),
                exposure=plan.exposure_ms,
                progress_callback=progress, cancel_check=cancelled,
            )
            core.setZPosition(initial_z)
        except AcquisitionCancelled as error:
            partial = error.partial_result
            if not partial:
                raise
            stopped = True
            initial_z = partial["initial_z"]
            spectra = partial["all_spectra"]
            zs = partial["z_offsets"]
    # An exposure that finished just as Stop arrived is still worth preserving.
    stopped = stopped or cancelled()
    dataset = xr.Dataset(
        {"spec": (["z", "n", "pixel"], spectra)},
        coords={"z": zs, "x": plan.points_yx[0][1], "y": plan.points_yx[0][0],
                "exposure": plan.exposure_ms},
        attrs={"acquisition_status": "stopped" if stopped else "complete",
               "initial_z_um": float(initial_z), "planned_z_planes": len(plan.z_offsets_um)},
    )
    progress(len(zs), len(plan.z_offsets_um), "Saving completed axial spectra")
    Path(plan.output_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_zarr(plan.output_path, mode="w-")
    print(f"Saved {'partial ' if stopped else ''}axial scan to {plan.output_path}")
    return {"dataset": dataset, "spectra": spectra, "zs": zs,
            "path": plan.output_path, "stopped": stopped}


def _stop_grid_live_imaging(owner):
    """Stop live imaging on the GUI thread before any grid worker starts.

    CMMCorePlus emits sequenceAcquisitionStopped here, allowing napari's live
    timer and Live buttons to update before exposure/channel changes. The
    simulator additionally owns a GUI timer that must be explicitly stopped.
    A stop failure must abort preparation, not race live imaging with a scan.
    """
    stop_demo_live = getattr(owner, "_toggle_demo_live", None)
    if callable(stop_demo_live):
        stop_demo_live(False)
    else:
        owner.core.stopSequenceAcquisition()
    is_running = getattr(owner.core, "isSequenceRunning", None)
    if callable(is_running) and is_running():
        raise RuntimeError("Live imaging is still running; cannot start spatial mapping.")


def start_grid_scan(owner):
    if not _ready(owner):
        return
    try:
        from napari.layers import Shapes

        shapes = owner.viewer.layers.selection.active
        is_shapes = isinstance(shapes, Shapes)
        if not is_shapes:
            from napari.layers import Labels

            if not isinstance(shapes, Labels):
                raise ValueError("Select a Shapes ROI or a 2D Labels layer (all non-zero labels).")
        width, height = owner._get_image_xy()
        image_layer = scan_image_reference(owner.viewer, width, height)
        label_roi = None
        if is_shapes:
            roi = snapshot_scan_roi(shapes, image_layer=image_layer)
            roi_options = dict(shape_type=roi.shape_type, image_layer_name=roi.image_layer_name)
            shape_yx = roi.vertices_yx
        else:
            label_roi = snapshot_scan_labels(shapes, image_layer=image_layer, width=width, height=height)
            roi_options = dict(label_roi=label_roi)
            shape_yx = None
        name = owner.scan_name_input.text().strip()
        multi_z = owner.scan_zscan_check.isChecked()
        zs = np.linspace(-owner.scan_zrange_input.value(), owner.scan_zrange_input.value(),
                         owner.scan_zsteps_input.value()) if multi_z else np.array([0.0])
        channels = []
        for row in owner.channel_rows:
            channel = row["combo"].currentText()
            if row["combo"].isEnabled() and channel and channel not in dict(channels):
                channels.append((channel, float(row["exp"].value())))
        plan = build_grid_preview(
            shape_yx,
            sampling_mode=owner.scan_sampling_mode_combo.currentData(),
            total_points=owner.scan_total_points_input.value(),
            spacing_px=owner.scan_spacing_input.value(),
            z_offsets_um=zs, exposure_ms=owner.scan_exp_input.value(),
            output_path=_filename(name, "grid_scan_z_" if multi_z else "grid_scan_data_"),
            extra_channels=channels, z_offset_um=owner.scan_z_input.value(), layer_name=shapes.name,
            **roi_options,
        )
        grid = np.asarray(plan.points_yx)
        # Layer affine round-trips can put an edge pixel a few ulps outside
        # the image. Permit numerical residue without clipping/resampling the
        # frozen plan, while still rejecting genuinely out-of-frame points.
        pixel_tolerance = 1e-7
        if (np.any(grid < -pixel_tolerance)
                or np.any(grid[:, 0] > height - 1 + pixel_tolerance)
                or np.any(grid[:, 1] > width - 1 + pixel_tolerance)):
            raise ValueError("Some scan points are outside the camera image. Move or resize the ROI and preview again.")
        if not _confirm(owner, plan):
            return
        # The modal preview may have allowed another acquisition to start.
        # Recheck before stopping a sequence that could now belong to an MDA.
        if not _ready(owner):
            return
        _stop_grid_live_imaging(owner)
        # Only finish drawing after Start; Cancel leaves selection unchanged.
        if is_shapes:
            snapshot_scan_shape(shapes)
        core, daq, collector = owner.core, owner.daq, owner.collector
        transformer = owner.transformer
        is_demo = type(owner).__name__ == "DemoWidget"
        volts = np.full_like(grid, np.nan) if is_demo else transformer.BF_to_volts(
            grid / [height, width], max_volts=1.8,
        )
        imaging_channel, raman_mode = owner._set_scan_imaging_channel, owner._set_scan_raman_mode
        spectral_calibration = owner.spectral_calibration

        def operation(progress, cancelled):
            return acquire_grid(plan, core, daq, collector, volts, imaging_channel,
                                raman_mode, progress, cancelled, is_demo=is_demo)

        def complete(result):
            owner.scan_ds = result["dataset"]
            if not result["stopped"]:
                owner._show_plot(GridScanPlotWindow(
                    result["dataset"], title=f"Grid scan: {name}",
                    spectral_calibration=spectral_calibration,
                ))
            state = "partial grid scan" if result["stopped"] else "grid scan"
            owner.status.setText(f"Status: {state} saved → {result['path']}")

        if label_roi is not None:
            # Use the same frozen points as acquisition, not cell centroids.
            # Keep the source Labels layer and selection intact for repeat scans.
            points_layer = owner.viewer.add_points(
                grid.copy(), name=f"Spatial map points — {plan.layer_name}",
                affine=np.asarray(label_roi.image_to_world_yx), size=4,
                face_color="#39a9dc",
                features={"label_id": np.asarray(plan.point_label_ids, dtype=label_roi.dtype)},
                metadata={"sampling_mode": plan.sampling_mode, "spacing_px": plan.spacing_px,
                          "source_labels": plan.layer_name, "all_nonzero_labels": True},
            )
            points_layer.editable = False
            owner.viewer.layers.selection.active = shapes
        owner._acquisition_jobs.start("Grid scan", operation, complete)
    except Exception as error:
        owner.status.setText(f"Status: cannot prepare grid scan — {error}")


def acquire_grid(plan, core, daq, collector, volts, imaging_channel, raman_mode,
                 progress, cancelled, *, is_demo=False):
    """Collect bounded point batches, restoring shutter/Z even on an error."""
    check_cancelled(cancelled)
    grid = np.asarray(plan.points_yx)
    zs = np.asarray(plan.z_offsets_um)
    initial_z = core.getPosition()
    batches, sample_points, sample_zs, bf_planes = [], [], [], []
    extra_images = {}
    before = after = None
    stopped = False
    completed = 0
    # Target about five seconds of exposure, but the finite DAQ waveform needs
    # at least two samples. Merge a final singleton without duplicate spectra.
    # No SDK aborts or fabricated per-frame progress are used.
    batch_size = max(2, min(len(grid), 32, int(5000 / plan.exposure_ms)))
    try:
        progress(0, plan.spectrum_count, "Preparing grid scan")
        check_cancelled(cancelled)
        imaging_channel()
        core.setExposure(10)
        before = np.asarray(core.snap()).copy()
        for channel, exposure in plan.extra_channels:
            check_cancelled(cancelled)
            core.setConfig("Channel", channel)
            core.setExposure(exposure)
            extra_images[channel] = np.asarray(core.snap()).copy()
        daq.galvo.stop()
        daq.galvo.start()
        check_cancelled(cancelled)
        raman_mode(True)
        core.stopSequenceAcquisition()
        core.setExposure(1)
        for zi, offset in enumerate(zs):
            check_cancelled(cancelled)
            core.setPosition(initial_z - plan.z_offset_um + offset)
            core.waitForSystem()
            start = 0
            while start < len(grid):
                check_cancelled(cancelled)
                end = min(len(grid), start + batch_size)
                if len(grid) - end == 1:
                    end += 1
                if is_demo:
                    spectra = collector.collect_spectra_image_points(grid[start:end], plan.exposure_ms)
                else:
                    spectra = collector.collect_spectra_pts(volts[start:end], plan.exposure_ms)
                spectra = np.asarray(spectra)
                if spectra.ndim != 2 or spectra.shape[0] != end - start:
                    raise ValueError("Detector returned an unexpected number of spectra.")
                batches.append(spectra.copy())
                sample_points.extend(range(start, end))
                sample_zs.extend([zi] * (end - start))
                completed += end - start
                progress(completed, plan.spectrum_count,
                         f"Z plane {zi + 1}/{len(zs)} — {completed}/{plan.spectrum_count} spectra")
                check_cancelled(cancelled)
                start = end
            if len(zs) > 1:
                raman_mode(False)
                imaging_channel()
                core.setExposure(10)
                bf_planes.append(np.asarray(core.snap()).copy())
                check_cancelled(cancelled)
                raman_mode(True)
                core.setExposure(1)
    except AcquisitionCancelled:
        stopped = True
        if not batches:
            raise
    finally:
        # Nested finally means a shutter failure cannot skip the Z restore.
        try:
            raman_mode(False)
        finally:
            core.setPosition(initial_z)
            core.waitForSystem()
    imaging_channel()
    core.setExposure(10)
    if not stopped:
        after = np.asarray(core.snap()).copy()
    stopped = stopped or cancelled()
    spectra = np.concatenate(batches, axis=0)
    data = {"laser_pos": (("idx", "volt"), volts),
            "grid_pos": (("idx", "xy"), grid), "BF": (("Y", "X"), before)}
    if plan.label_roi is not None:
        data["label_id"] = ("idx", np.asarray(plan.point_label_ids, dtype=plan.label_roi.dtype))
    if stopped:
        # Ragged partial scans contain only real measured samples. No NaN
        # placeholders or unmeasured images are presented as acquired data.
        data.update({"specs": (("sample", "spec_dim"), spectra),
                     "sample_grid_index": ("sample", np.asarray(sample_points)),
                     "sample_z_index": ("sample", np.asarray(sample_zs)),
                     "z_range": ("z", zs)})
    elif len(zs) > 1:
        data.update({"specs": (("z", "idx", "spec_dim"), spectra.reshape(len(zs), len(grid), -1)),
                     "z_range": ("z", zs), "BF_z": (("z", "Y", "X"), np.stack(bf_planes))})
    else:
        data["specs"] = (("N", "spec_dim"), spectra)
    if after is not None:
        data["end_BF"] = (("Y", "X"), after)
    data.update({channel: (("Y", "X"), image) for channel, image in extra_images.items()})
    dataset = xr.Dataset(data, attrs={
        "time": str(datetime.now()), "raman_exposure_ms": plan.exposure_ms,
        "z_offset": plan.z_offset_um, "initial_z_um": float(initial_z),
        "z_range_min": float(zs.min()), "z_range_max": float(zs.max()), "z_steps": len(zs),
        "channel_exposures_ms": dict(plan.extra_channels),
        "acquisition_status": "stopped" if stopped else "complete",
        "completed_spectra": completed, "planned_spectra": plan.spectrum_count,
        "sampling_mode": plan.sampling_mode,
        "sampling_pattern": "square_grid",
        "points_per_z": len(plan.points_yx),
        "requested_points_per_z": plan.requested_point_count,
        "point_spacing_px": plan.spacing_px,
        "roi": {"shape_type": plan.shape_type,
                "vertices_yx": [list(point) for point in plan.roi_vertices_yx],
                "image_layer": plan.image_layer_name,
                "source_layer": plan.layer_name, "inside_only": True},
    })
    if plan.label_roi is not None:
        dataset.attrs["roi"].update({
            "selection": "all_nonzero_labels", "background_label": 0,
            "label_values": list(plan.label_roi.label_values),
            "label_image_shape": list(plan.label_roi.shape),
            "sampled_label_count": len(set(plan.point_label_ids)),
        })
    progress(completed, plan.spectrum_count, "Saving completed grid spectra")
    dataset.to_zarr(plan.output_path, mode="w-")
    print(f"Saved {'partial ' if stopped else ''}grid scan to {plan.output_path}")
    return {"dataset": dataset, "path": plan.output_path, "stopped": stopped}
