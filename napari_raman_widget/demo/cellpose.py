"""Cellpose preprocessing for small objects in simulated BF frames."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def segment_upsampled_demo_region(
    image: np.ndarray,
    *,
    center_yx: tuple[float, float],
    radius: float,
    upsample: int,
    cellpose_model: str,
    segmenter: Callable,
) -> np.ndarray:
    """Upsample the mask region for Cellpose and restore image coordinates."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("Cellpose demo image must be two-dimensional.")
    factor = max(1, int(upsample))
    center = np.asarray(center_yx, dtype=float).reshape(2)
    crop_radius = max(float(radius), 1.0) + 4.0
    y0 = max(0, int(np.floor(center[0] - crop_radius)))
    y1 = min(image.shape[0], int(np.ceil(center[0] + crop_radius + 1)))
    x0 = max(0, int(np.floor(center[1] - crop_radius)))
    x1 = min(image.shape[1], int(np.ceil(center[1] + crop_radius + 1)))
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError("The Cellpose selection mask does not overlap the image.")

    enlarged = np.repeat(np.repeat(crop, factor, axis=0), factor, axis=1)
    local_center = (
        float((center[0] - y0) * factor),
        float((center[1] - x0) * factor),
    )
    enlarged_mask = np.asarray(
        segmenter(
            enlarged,
            scale=1,
            cellpose_model=cellpose_model,
            circle_center=local_center,
            circle_radius=float(radius) * factor,
        )
    )
    restored_crop = enlarged_mask[::factor, ::factor]
    restored_crop = restored_crop[: crop.shape[0], : crop.shape[1]]
    restored = np.zeros(image.shape, dtype=restored_crop.dtype)
    restored[y0:y1, x0:x1] = restored_crop
    return restored


__all__ = ["segment_upsampled_demo_region"]
