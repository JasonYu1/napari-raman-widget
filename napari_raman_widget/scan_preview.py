"""Hardware-free acquisition plans and a confirmation preview dialog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
)

from .figure_panel import FigurePanel
from .label_sampling import LabelROI, sample_label_plan
from .roi_sampling import roi_outline, sample_roi_plan


_MAX_AXIS_POINTS = 500
_MAX_REPEATS = 1000
_MAX_DISPLAY_POINTS = 2500


def _positive_number(value, name):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return number


def _count(value, name, *, minimum=1, maximum=_MAX_AXIS_POINTS):
    number = float(value)
    if (
        not np.isfinite(number)
        or not number.is_integer()
        or not minimum <= number <= maximum
    ):
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}.")
    return int(number)


def _output_path(value):
    if not str(value).strip():
        raise ValueError("An output path is required.")
    return str(Path(value).expanduser().resolve())


@dataclass(frozen=True)
class ScanPreview:
    """Copied, immutable settings for a single confirmed acquisition.

    Points use image ``(row, column)`` order. Grid Z offsets are relative to
    ``initial Z - z_offset_um``; axial offsets are relative to initial Z.
    No stage position is read and no output file or directory is created.
    """

    kind: str
    points_yx: tuple[tuple[float, float], ...]
    z_offsets_um: tuple[float, ...]
    repeats: int
    exposure_ms: float
    output_path: str
    layer_name: str = ""
    shape_type: str | None = None
    roi_vertices_yx: tuple[tuple[float, float], ...] = ()
    roi_boundary_yx: tuple[tuple[float, float], ...] = ()
    sampling_mode: str | None = None
    requested_point_count: int | None = None
    spacing_px: float | None = None
    image_layer_name: str = "Camera pixels"
    z_offset_um: float = 0.0
    extra_channels: tuple[tuple[str, float], ...] = ()
    brightfield_frames: int = 0
    label_roi: LabelROI | None = None
    point_label_ids: tuple[int, ...] = ()

    @property
    def grid_points_yx(self):
        """Grid-compatible alias for the copied image coordinates."""
        return self.points_yx

    @property
    def spectrum_count(self):
        return len(self.points_yx) * len(self.z_offsets_um) * self.repeats

    @property
    def minimum_duration_seconds(self):
        """Exposure-only lower bound, not a predicted completion time."""
        exposure_ms = self.spectrum_count * self.exposure_ms
        exposure_ms += sum(exposure for _, exposure in self.extra_channels)
        exposure_ms += self.brightfield_frames * 10.0
        return exposure_ms / 1000.0

    @property
    def summary_text(self):
        lines = [
            "Spatial grid scan" if self.kind == "grid" else "Axial background scan",
            f"Source layer: {self.layer_name or '(unnamed)'}",
        ]
        rows, columns = zip(*self.points_yx)
        if self.kind == "grid":
            roi_rows, roi_columns = zip(*self.roi_boundary_yx)
            lines.extend([
                ("ROI: all non-zero labels — background 0 excluded" if self.label_roi is not None
                 else f"ROI: {self.shape_type} — inside shape only"),
                f"Pixel reference: {self.image_layer_name}",
                f"ROI extent (pixels): X {min(roi_columns):g} to "
                f"{max(roi_columns):g}; Y {min(roi_rows):g} to {max(roi_rows):g}",
                (f"Sampling: target {self.requested_point_count:,} points per Z plane "
                 "(approximate; automatic spacing)" if self.sampling_mode == "count" else
                 "Sampling: specified pixel spacing"),
                f"Grid spacing: {self.spacing_px:g} px spacing in X and Y (clipped square grid)",
                f"Actual points per Z plane: {len(self.points_yx):,}",
            ])
            if self.label_roi is not None:
                sampled = len(set(self.point_label_ids))
                available = len(self.label_roi.label_values)
                lines.append(f"Label IDs sampled: {sampled:,} / {available:,}")
                if sampled < available:
                    lines.append("Some small labels have no grid point; reduce spacing or increase the target to sample them.")
        else:
            lines.append(f"Selected point (pixels): X {columns[0]:g}, Y {rows[0]:g}")
        z_min, z_max = min(self.z_offsets_um), max(self.z_offsets_um)
        lines.extend([
            f"Z: {len(self.z_offsets_um)} planes; offsets {z_min:g} to {z_max:g} µm",
            (
                f"Z reference: initial Z minus {self.z_offset_um:g} µm"
                if self.kind == "grid" else "Z reference: initial Z"
            ),
            f"Repeats: {self.repeats} spectrum/spectra per point per Z plane",
            f"Raman exposure: {self.exposure_ms:g} ms per spectrum",
            f"Total Raman spectra: {self.spectrum_count:,}",
        ])
        if self.brightfield_frames:
            lines.append(f"Brightfield: {self.brightfield_frames} frames at 10 ms")
        if self.extra_channels:
            channels = ", ".join(
                f"{name} ({exposure:g} ms)" for name, exposure in self.extra_channels
            )
            lines.append(f"Extra channels (one image each): {channels}")
        duration = self.minimum_duration_seconds
        lines.extend([
            f"Approximate duration: at least {duration:,.2f} s "
            f"({duration / 60:,.2f} min), exposure only.",
            "Excludes movement, settling, readout, autofocus, processing and saving; "
            "actual duration will be longer.",
            f"Output: {self.output_path}",
        ])
        return "\n".join(lines)


def build_grid_preview(
    shape_yx,
    *,
    z_offsets_um,
    exposure_ms,
    output_path,
    shape_type="rectangle",
    sampling_mode="count",
    total_points=400,
    spacing_px=10.0,
    extra_channels=(),
    z_offset_um=0.0,
    layer_name="",
    image_layer_name="Camera pixels",
    label_roi=None,
):
    """Freeze shape-clipped sampling; the exact same points drive acquisition."""
    if label_roi is None:
        shape = np.array(shape_yx, dtype=float, copy=True)
        if shape.ndim != 2 or shape.shape[0] < 2 or shape.shape[1] != 2:
            raise ValueError("The scan region must contain image (row, column) pairs.")
        if not np.isfinite(shape).all():
            raise ValueError("The scan region must contain finite coordinates.")
    else:
        shape = np.empty((0, 2))
        shape_type = "labels"
        image_layer_name = label_roi.image_layer_name
    z_values = np.array(z_offsets_um, dtype=float, copy=True)
    if (
        z_values.ndim != 1
        or not 1 <= len(z_values) <= _MAX_AXIS_POINTS
        or not np.isfinite(z_values).all()
    ):
        raise ValueError("Z offsets must contain 1 to 500 finite values.")
    z_offset = float(z_offset_um)
    if not np.isfinite(z_offset):
        raise ValueError("Z offset must be finite.")
    channels = tuple(
        (str(name), _positive_number(exposure, "Channel exposure"))
        for name, exposure in extra_channels
    )
    if any(not name.strip() for name, _ in channels):
        raise ValueError("Extra channels must have a name.")
    exposure = _positive_number(exposure_ms, "Raman exposure")
    if label_roi is None:
        points, effective_spacing = sample_roi_plan(
            shape, shape_type, mode=sampling_mode,
            total_points=total_points, spacing_px=spacing_px,
        )
        boundary = roi_outline(shape, shape_type)
        point_label_ids = ()
    else:
        points, effective_spacing = sample_label_plan(
            label_roi, mode=sampling_mode, total_points=total_points, spacing_px=spacing_px,
        )
        low, high = label_roi.bounds()
        boundary = [low, [low[0], high[1]], high, [high[0], low[1]], low]
        point_label_ids = tuple(int(value) for value in label_roi.labels_at(points))
    return ScanPreview(
        kind="grid",
        points_yx=tuple(tuple(float(value) for value in point) for point in points),
        z_offsets_um=tuple(float(z) for z in z_values),
        repeats=1,
        exposure_ms=exposure,
        output_path=_output_path(output_path),
        layer_name=str(layer_name),
        shape_type=str(shape_type),
        roi_vertices_yx=tuple(tuple(float(value) for value in point) for point in shape),
        roi_boundary_yx=tuple(tuple(float(value) for value in point) for point in boundary),
        sampling_mode=sampling_mode,
        requested_point_count=int(total_points) if sampling_mode == "count" else None,
        spacing_px=float(effective_spacing),
        image_layer_name=str(image_layer_name),
        z_offset_um=z_offset,
        extra_channels=channels,
        brightfield_frames=2 + (len(z_values) if len(z_values) > 1 else 0),
        label_roi=label_roi,
        point_label_ids=point_label_ids,
    )


def build_axial_preview(
    point_yx,
    *,
    repeats,
    search_range_um,
    search_points,
    exposure_ms,
    output_path,
    layer_name="",
):
    """Describe ``autofocus_w_bkd``: one evenly spaced Z sweep, with repeats.

    Despite the acquisition function's name, this reference scan has no
    measured fine-search stage and does not choose an optimal focus.
    """
    count = _count(search_points, "Z search points", minimum=2)
    repeat_count = _count(repeats, "Repeats", maximum=_MAX_REPEATS)
    span = _positive_number(search_range_um, "Z search range")
    exposure = _positive_number(exposure_ms, "Raman exposure")
    point = np.array(point_yx, dtype=float, copy=True)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("Select one finite image (row, column) point.")
    return ScanPreview(
        kind="axial",
        points_yx=((float(point[0]), float(point[1])),),
        z_offsets_um=tuple(float(z) for z in np.linspace(-span, span, count)),
        repeats=repeat_count,
        exposure_ms=exposure,
        output_path=_output_path(output_path),
        layer_name=str(layer_name),
    )


class ScanPreviewDialog(QDialog):
    """Show a plan only; accepting never starts hardware by itself."""

    def __init__(self, preview: ScanPreview, parent=None):
        super().__init__(parent)
        self.preview = preview
        self.setWindowTitle("Review acquisition")
        self.resize(820, 760)
        layout = QVBoxLayout(self)
        self.summary_label = QLabel(preview.summary_text, self)
        self.summary_label.setTextFormat(Qt.PlainText)
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.summary_label)
        self.plot_panel = FigurePanel(parent=self, figsize=(7, 3))
        self.plot_panel.toolbar.hide()
        layout.addWidget(self.plot_panel, 1)
        self._draw_plan()
        self.buttons = QDialogButtonBox(self)
        self.start_button = self.buttons.addButton(
            "Start scan", QDialogButtonBox.AcceptRole
        )
        self.cancel_button = self.buttons.addButton(QDialogButtonBox.Cancel)
        self.start_button.setAutoDefault(False)
        self.start_button.setDefault(False)
        self.cancel_button.setDefault(True)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.cancel_button.setFocus()

    def _draw_plan(self):
        from matplotlib.ticker import MaxNLocator

        preview = self.preview
        if preview.kind == "grid":
            xy_axis, z_axis = self.plot_panel.figure.subplots(1, 2)
            if preview.label_roi is not None:
                from matplotlib.colors import ListedColormap

                roi = preview.label_roi
                (r0, c0), (r1, c1) = roi.limits_yx
                mask = roi.labels[r0:r1 + 1, c0:c1 + 1] != 0
                xy_axis.imshow(
                    np.ma.masked_where(~mask, mask), origin="lower",
                    extent=(c0 - .5, c1 + .5, r0 - .5, r1 + .5),
                    cmap=ListedColormap(["#e5a44d"]), alpha=.35,
                    interpolation="nearest", zorder=0,
                )
            points = np.asarray(preview.points_yx)
            indices = np.linspace(
                0, len(points) - 1, min(len(points), _MAX_DISPLAY_POINTS), dtype=int
            )
            shown = points[indices]
            xy_axis.scatter(shown[:, 1], shown[:, 0], s=8, color="#39a9dc")
            if preview.label_roi is None:
                outline = np.asarray(preview.roi_boundary_yx)
                xy_axis.plot(
                    outline[:, 1], outline[:, 0],
                    color="#e5a44d",
                )
            xy_axis.set_xlabel("X / column (pixels)")
            xy_axis.set_ylabel("Y / row (pixels)")
            xy_axis.invert_yaxis()
            xy_axis.set_aspect("equal", adjustable="datalim")
            xy_axis.set_title(
                "ROI points (sampled display)" if len(shown) < len(points) else "Inside-ROI scan points"
            )
        else:
            z_axis = self.plot_panel.figure.subplots()
        z_axis.scatter(
            preview.z_offsets_um, np.arange(1, len(preview.z_offsets_um) + 1),
            color="#39a9dc",
        )
        z_axis.set_xlabel("Z offset from reference (µm)")
        z_axis.set_ylabel("Acquisition plane")
        z_axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        z_axis.set_title("Planned Z positions")
        self.plot_panel.canvas.draw_idle()

    def done(self, result):
        self.plot_panel.close()
        super().done(result)
