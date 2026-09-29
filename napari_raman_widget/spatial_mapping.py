"""Helpers shared by hardware and simulated spatial mapping."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["snapshot_scan_shape"]


def snapshot_scan_shape(shapes_layer: Any, *, finish_interaction=True) -> np.ndarray:
    """Copy the selected scan shape and leave its layer safe to interact with.

    Napari 0.5.0 can transiently leave a selected Shapes layer with
    ``_value is None``. Toggling that layer's visibility then refreshes its
    highlight and raises ``TypeError: 'NoneType' object is not subscriptable``.
    A spatial scan only needs a fixed copy of the rectangle, so finish the
    drawing interaction and deselect it before the long acquisition begins.

    Only public layer properties are changed here. The returned coordinates
    do not share storage with the napari layer.
    """
    data = shapes_layer.data
    if len(data) == 0:
        raise RuntimeError("The active Shapes layer is empty.")

    selected = sorted(int(index) for index in shapes_layer.selected_data)
    shape_index = selected[-1] if selected else len(data) - 1
    shape = np.array(data[shape_index], dtype=float, copy=True)

    if shape.ndim != 2 or shape.shape[0] == 0 or shape.shape[1] < 2:
        raise RuntimeError("The selected scan shape has no usable coordinates.")

    # Changing mode asks napari to finish an in-progress drawing gesture.
    # Clearing selected_data prevents visibility refreshes from entering the
    # buggy selected-shape highlight path in napari 0.5.0.
    if finish_interaction:
        shapes_layer.mode = "pan_zoom"
        shapes_layer.selected_data = set()
    return shape
