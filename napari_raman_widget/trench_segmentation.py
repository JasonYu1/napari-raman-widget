"""Cellpose input preparation for learned microfluidic trenches.

The upstream Raman engine normally crops Cellpose input to a circle.  In
trench-constrained mode that circle is both redundant and harmful: it can cut
through the closed head or include the walls between adjacent trenches.  This
adapter keeps the upstream segmenter's public ``segment`` contract while
using the circle's center to select one trench and replacing the circular crop
with that trench's complete registered, cap-aware interior.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .trench_tracking import (
    DEFAULT_TRENCH_CONFIG,
    TrenchTrackingConfig,
    nearest_trench_index,
    registered_trench_geometry,
)

__all__ = ["TrenchCellposeSegmenter"]


def _scaled_shape(shape: tuple[int, int], scale: int) -> tuple[int, int]:
    return tuple(
        np.maximum(1, np.round(np.asarray(shape) / scale).astype(int))
    )


def _sample_mask(mask: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Sample a full-resolution boolean mask at output-pixel centers."""

    source = np.asarray(mask, dtype=bool)
    if source.shape == output_shape:
        return source.copy()
    rows = np.floor(
        (np.arange(output_shape[0]) + 0.5)
        * source.shape[0]
        / output_shape[0]
    ).astype(int)
    columns = np.floor(
        (np.arange(output_shape[1]) + 0.5)
        * source.shape[1]
        / output_shape[1]
    ).astype(int)
    rows = np.clip(rows, 0, source.shape[0] - 1)
    columns = np.clip(columns, 0, source.shape[1] - 1)
    return source[np.ix_(rows, columns)]


def _aligned_bounds(
    mask: np.ndarray,
    scale: int,
    padding: int,
) -> tuple[int, int, int, int]:
    pixels = np.argwhere(mask)
    if not len(pixels):
        raise ValueError("registered trench contains no safe pixels")
    lower = np.maximum(pixels.min(axis=0) - padding, 0)
    upper = np.minimum(pixels.max(axis=0) + padding + 1, mask.shape)
    y0 = int(np.floor(lower[0] / scale) * scale)
    x0 = int(np.floor(lower[1] / scale) * scale)
    y1 = int(np.ceil(upper[0] / scale) * scale)
    x1 = int(np.ceil(upper[1] / scale) * scale)
    y1 = min(mask.shape[0], max(y0 + scale, y1))
    x1 = min(mask.shape[1], max(x0 + scale, x1))
    return y0, y1, x0, x1


class TrenchCellposeSegmenter:
    """Replace a circular Cellpose ROI with the single central trench.

    ``delegate`` is the reusable upstream Cellpose segmenter. The circle's
    center selects the containing or nearest trench; its radius never clips
    the input. Geometry failures return an empty label image, allowing the
    downstream trench tracker to hold the previous safe point.
    """

    def __init__(
        self,
        delegate: Any,
        config: TrenchTrackingConfig = DEFAULT_TRENCH_CONFIG,
    ) -> None:
        if not callable(getattr(delegate, "segment", None)):
            raise TypeError("delegate must provide a callable segment method")
        self.delegate = delegate
        self.config = config

    def segment(
        self,
        img: np.ndarray,
        scale: int = 4,
        crop: bool = True,
        circle_center: tuple[float, float] | None = None,
        circle_radius: float = 100,
        crop_padding: int = 4,
    ) -> np.ndarray:
        """Segment the trench nearest the circle center without circular clipping."""

        del crop, circle_radius
        frame = np.asarray(img)
        if frame.ndim != 2:
            raise ValueError("img must be a two-dimensional image")
        scale = int(scale)
        if scale < 1:
            raise ValueError("scale must be at least one")
        if crop_padding < 0:
            raise ValueError("crop_padding cannot be negative")

        output = np.zeros(_scaled_shape(frame.shape, scale), dtype=np.uint32)
        geometry = registered_trench_geometry(frame, self.config)
        if not geometry.geometry_valid:
            return output
        center = (
            (np.asarray(frame.shape) - 1) / 2
            if circle_center is None
            else circle_center
        )
        index = nearest_trench_index(
            geometry.trench_masks,
            center,
        )
        if index is None or not np.any(geometry.safe_masks[index]):
            return output

        next_label = 1
        padding = max(int(crop_padding), 2 * scale)
        safe = geometry.safe_masks[index]
        y0, y1, x0, x1 = _aligned_bounds(safe, scale, padding)
        safe_crop = safe[y0:y1, x0:x1]
        image_crop = frame[y0:y1, x0:x1].astype(np.float32, copy=True)
        neutral = float(np.median(image_crop[safe_crop]))
        image_crop[~safe_crop] = neutral

        labels = np.asarray(
            self.delegate.segment(
                image_crop,
                scale=scale,
                crop=False,
                circle_center=None,
                circle_radius=np.inf,
                crop_padding=crop_padding,
            )
        )
        if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Cellpose delegate returned an invalid label image")
        if np.any(labels < 0):
            raise ValueError("Cellpose labels cannot be negative")

        allowed = _sample_mask(safe_crop, labels.shape)
        labels = np.where(allowed, labels, 0)
        relabeled = np.zeros(labels.shape, dtype=np.uint32)
        for value in np.unique(labels):
            if value == 0:
                continue
            relabeled[labels == value] = next_label
            next_label += 1

        target_y0 = y0 // scale
        target_x0 = x0 // scale
        target_y1 = min(output.shape[0], target_y0 + relabeled.shape[0])
        target_x1 = min(output.shape[1], target_x0 + relabeled.shape[1])
        source_y1 = target_y1 - target_y0
        source_x1 = target_x1 - target_x0
        destination = output[target_y0:target_y1, target_x0:target_x1]
        source = relabeled[:source_y1, :source_x1]
        destination[source != 0] = source[source != 0]

        return output
