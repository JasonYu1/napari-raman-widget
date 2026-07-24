"""One-shot refinement of Raman cell targets from fresh segmentations."""

from __future__ import annotations

import time
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.ndimage import center_of_mass
from skimage.transform import rescale, resize
from tqdm.auto import tqdm

__all__ = [
    "refine_cell_source_points",
    "refine_points_to_label_centers",
]


class _ReusableCellposeSegmenter:
    """Match ``segment_single_img`` while reusing one loaded model."""

    def __init__(self, cellpose_model: str) -> None:
        from cellpose.models import Cellpose

        self.cellpose_model = str(cellpose_model)
        self.model = Cellpose(model_type=self.cellpose_model, gpu=False)

    def __call__(
        self,
        image: np.ndarray,
        *,
        scale: int = 4,
        crop: bool = True,
        circle_center: tuple[float, float] | None = None,
        circle_radius: float = 100,
        **_: Any,
    ) -> np.ndarray:
        image = np.asarray(image)
        scale = max(1, int(scale))
        if crop:
            if circle_center is None:
                circle_center = tuple(np.asarray(image.shape) / 2)
            yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
            cy, cx = circle_center
            inside = (yy - cy) ** 2 + (xx - cx) ** 2 <= circle_radius**2
            image = image.copy()
            image[~inside] = image.min()

        scaled = rescale(image, 1 / scale, anti_aliasing=True)
        value_range = float(scaled.max() - scaled.min())
        if value_range > 0:
            scaled = (scaled - scaled.min()) / value_range
        else:
            scaled = np.zeros_like(scaled, dtype=float)

        masks, _, _ = self.model.cp.eval(
            scaled,
            batch_size=1024,
            channels=[[0, 0]],
            diameter=50 / scale,
            flow_threshold=0.6,
            cellprob_threshold=-2,
            normalize=False,
        )
        return np.asarray(masks)


_CELLPOSE_SEGMENTERS: dict[str, _ReusableCellposeSegmenter] = {}


def _reusable_cellpose_segmenter(model_name: str) -> _ReusableCellposeSegmenter:
    """Load each requested Cellpose model at most once per process."""
    normalized = str(model_name or "cyto2")
    segmenter = _CELLPOSE_SEGMENTERS.get(normalized)
    if segmenter is None:
        print(f"Loading Cellpose model {normalized!r} (cached after this run)...")
        segmenter = _ReusableCellposeSegmenter(normalized)
        _CELLPOSE_SEGMENTERS[normalized] = segmenter
    return segmenter


def refine_points_to_label_centers(
    points_yx: np.ndarray,
    label_mask: np.ndarray,
    *,
    max_distance: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Snap points to the center of their cell, or a nearby cell.

    A point inside a labeled object is moved to that object's center of mass.
    A background point is moved to the nearest labeled-object center only when
    it is within ``max_distance`` pixels. Points without a suitable match are
    left unchanged.

    Returns the refined ``(y, x)`` coordinates and a Boolean match mask.
    """
    points = np.asarray(points_yx, dtype=float)
    mask = np.asarray(label_mask)

    if points.size == 0:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=bool)
    points = np.atleast_2d(points)
    if points.shape[1] != 2:
        raise ValueError("points_yx must have shape (N, 2)")
    if mask.ndim != 2:
        raise ValueError("label_mask must be two-dimensional")
    if max_distance < 0:
        raise ValueError("max_distance cannot be negative")

    label_values = np.unique(mask)
    label_values = label_values[label_values != 0]
    if len(label_values) == 0:
        return points.copy(), np.zeros(len(points), dtype=bool)

    centers = np.asarray(
        [
            center_of_mass(
                np.ones_like(mask, dtype=float),
                labels=mask,
                index=label_value,
            )
            for label_value in label_values
        ],
        dtype=float,
    )
    finite = np.isfinite(centers).all(axis=1)
    label_values = label_values[finite]
    centers = centers[finite]
    if len(centers) == 0:
        return points.copy(), np.zeros(len(points), dtype=bool)

    refined = points.copy()
    matched = np.zeros(len(points), dtype=bool)
    height, width = mask.shape

    for point_index, point_yx in enumerate(points):
        pixel_yx = np.rint(point_yx).astype(int)
        point_label = 0
        if (
            0 <= pixel_yx[0] < height
            and 0 <= pixel_yx[1] < width
        ):
            point_label = mask[pixel_yx[0], pixel_yx[1]]

        label_matches = np.flatnonzero(label_values == point_label)
        if point_label != 0 and len(label_matches):
            center_index = int(label_matches[0])
        else:
            distances = np.linalg.norm(centers - point_yx, axis=1)
            center_index = int(np.argmin(distances))
            if distances[center_index] > max_distance:
                continue

        refined[point_index] = centers[center_index]
        matched[point_index] = True

    return refined, matched


def _acquire_mask(
    core: Any,
    *,
    center_yx: tuple[float, float],
    radius: float,
    cellpose_model: str,
    segmentation_scale: int,
    cellpose_upsample: int,
    segmenter: Callable[..., np.ndarray],
) -> np.ndarray:
    core.snapImage()
    core.waitForSystem()
    image = np.asarray(core.getImage())

    def scaled_segmenter(segment_image: np.ndarray, **kwargs: Any) -> np.ndarray:
        kwargs["scale"] = int(segmentation_scale)
        scaled_mask = np.asarray(segmenter(segment_image, **kwargs))
        if scaled_mask.shape == segment_image.shape:
            return scaled_mask
        return resize(
            scaled_mask,
            segment_image.shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(scaled_mask.dtype, copy=False)

    # ``segment_single_img`` only masks the outside of its circle; it still
    # sends the entire camera frame through Cellpose. Crop the actual array so
    # a small refinement radius results in proportionally less inference work.
    factor = max(1, int(cellpose_upsample))
    center = np.asarray(center_yx, dtype=float)
    crop_radius = max(float(radius), 1.0) + 4.0
    y0 = max(0, int(np.floor(center[0] - crop_radius)))
    y1 = min(image.shape[0], int(np.ceil(center[0] + crop_radius + 1)))
    x0 = max(0, int(np.floor(center[1] - crop_radius)))
    x1 = min(image.shape[1], int(np.ceil(center[1] + crop_radius + 1)))
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError("The refinement circle does not overlap the image")

    enlarged = np.repeat(np.repeat(crop, factor, axis=0), factor, axis=1)
    local_center = (
        float((center[0] - y0) * factor),
        float((center[1] - x0) * factor),
    )
    enlarged_mask = scaled_segmenter(
        enlarged,
        cellpose_model=cellpose_model,
        circle_center=local_center,
        circle_radius=float(radius) * factor,
    )
    restored_crop = enlarged_mask[::factor, ::factor]
    restored_crop = restored_crop[: crop.shape[0], : crop.shape[1]]
    restored = np.zeros(image.shape, dtype=restored_crop.dtype)
    restored[y0:y1, x0:x1] = restored_crop
    return restored


def refine_cell_source_points(
    core: Any,
    source: Any,
    sequence: Any,
    *,
    center_yx: tuple[float, float],
    radius: float,
    max_distance: float = 80.0,
    cellpose_model: str = "cyto2",
    channel_group: str | None = "Channel",
    channel: str | None = "BF",
    stage_settle_time: float = 0.5,
    segmentation_scale: int = 4,
    cellpose_upsample: int = 1,
    show_progress: bool = True,
    segmenter: Callable[..., np.ndarray] | None = None,
    point_refiner: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, int]:
    """Acquire each selected FOV and refine its cell-layer points in place.

    Only ``source`` is changed; autofocus/background sources and stage
    positions are untouched. Repeated positions share one acquired mask. The
    starting XY position and channel are restored before returning.
    """
    if int(segmentation_scale) < 1:
        raise ValueError("segmentation_scale must be at least one")

    if segmenter is None:
        segmenter = _reusable_cellpose_segmenter(cellpose_model)

    point_data = np.asarray(source._points.data, dtype=float)
    if point_data.size == 0:
        raise ValueError("The cells layer does not contain any points")
    point_data = np.atleast_2d(point_data)
    if point_data.shape[1] < 3:
        raise ValueError(
            "The cells layer must contain position, y, and x coordinates"
        )

    stage_positions = list(sequence.stage_positions or ())
    if not stage_positions:
        raise ValueError("The selection does not contain stage positions")

    position_indices = point_data[:, 1].astype(int)
    invalid = sorted(
        {
            int(index)
            for index in position_indices
            if index < 0 or index >= len(stage_positions)
        }
    )
    if invalid:
        raise ValueError(
            "Cell points refer to positions outside the selection: "
            + ", ".join(map(str, invalid))
        )

    original_xy = None
    original_channel = None
    try:
        original_xy = tuple(core.getXYPosition())
    except Exception:
        pass
    if channel_group and channel:
        try:
            original_channel = core.getCurrentConfig(channel_group)
        except Exception:
            pass

    updated = point_data.copy()
    matched_total = 0
    masks_by_xy: dict[tuple[float, float], np.ndarray] = {}

    try:
        try:
            core.stopSequenceAcquisition()
        except Exception:
            pass

        if channel_group and channel:
            core.setConfig(channel_group, channel)
            core.waitForSystem()

        selected_positions = sorted(set(position_indices))
        for position_index in tqdm(
            selected_positions,
            desc="Refining cell targets",
            unit="position",
            file=sys.stdout,
            disable=not show_progress,
        ):
            stage_position = stage_positions[position_index]
            stage_xy = (float(stage_position.x), float(stage_position.y))

            if stage_xy not in masks_by_xy:
                core.setXYPosition(*stage_xy)
                core.waitForSystem()
                if stage_settle_time > 0:
                    time.sleep(stage_settle_time)
                masks_by_xy[stage_xy] = _acquire_mask(
                    core,
                    center_yx=center_yx,
                    radius=radius,
                    cellpose_model=cellpose_model,
                    segmentation_scale=int(segmentation_scale),
                    cellpose_upsample=cellpose_upsample,
                    segmenter=segmenter,
                )

            rows = np.flatnonzero(position_indices == position_index)
            refined, matched = refine_points_to_label_centers(
                updated[rows, -2:],
                masks_by_xy[stage_xy],
                max_distance=max_distance,
            )
            if point_refiner is not None and np.any(matched):
                extra_refined = np.asarray(
                    point_refiner(refined[matched]),
                    dtype=float,
                )
                if extra_refined.shape != refined[matched].shape:
                    raise ValueError("point_refiner must preserve point shape")
                refined[matched] = extra_refined
            updated[rows, -2:] = refined
            matched_total += int(np.count_nonzero(matched))
    finally:
        if channel_group and original_channel:
            try:
                core.setConfig(channel_group, original_channel)
                core.waitForSystem()
            except Exception:
                pass
        if original_xy is not None:
            try:
                core.setXYPosition(*original_xy)
                core.waitForSystem()
            except Exception:
                pass

    moved = int(
        np.count_nonzero(
            np.linalg.norm(updated[:, -2:] - point_data[:, -2:], axis=1)
            > 1e-6
        )
    )
    source._points.data = updated
    return {
        "total": int(len(updated)),
        "matched": matched_total,
        "moved": moved,
        "unmatched": int(len(updated) - matched_total),
    }
