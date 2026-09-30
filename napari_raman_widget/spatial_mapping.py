"""Helpers shared by hardware and simulated spatial mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .label_sampling import make_label_roi

__all__ = ["snapshot_scan_shape", "snapshot_scan_roi", "snapshot_scan_labels", "scan_image_reference"]


def snapshot_scan_labels(labels_layer, *, image_layer=None, width, height):
    """Snapshot all non-zero IDs, requiring alignment with camera pixels.

    Shared image/label scale, rotation, shear and translation are supported.
    A different label pixel frame must be resampled/aligned first; never guess
    how label IDs should map to camera scan coordinates.
    """
    if (labels_layer.ndim != 2 or getattr(labels_layer, "multiscale", False)
            or tuple(labels_layer.data.shape) != (height, width)):
        raise ValueError("Use a 2D Labels layer with the same pixel dimensions as the camera image.")
    basis = np.array([[0., 0.], [1., 0.], [0., 1.],
                      [height - 1., width - 1.], [0., width - 1.], [height - 1., 0.]])
    world = np.asarray([labels_layer.data_to_world(point) for point in basis], dtype=float)
    if image_layer is None:
        pixels = world
        name = "Camera pixels (untransformed)"
    else:
        pixels = np.asarray([image_layer.world_to_data(point) for point in world], dtype=float)
        name = image_layer.name
    if pixels.shape != basis.shape or not np.isfinite(pixels).all() or not np.allclose(
        pixels, basis, rtol=0, atol=1e-7
    ):
        raise ValueError("Align the Labels layer with the camera image: their pixel grids and transforms must match.")
    matrix = np.eye(3)
    matrix[:2, :2] = (world[1:3] - world[0]).T
    matrix[:2, 2] = world[0]
    return make_label_roi(labels_layer.data, image_layer_name=name, image_to_world_yx=matrix)


@dataclass(frozen=True)
class ScanROI:
    """One closed shape copied into the reference camera's pixel coordinates."""

    vertices_yx: tuple[tuple[float, float], ...]
    shape_type: str
    shape_index: int
    image_layer_name: str = "Camera pixels (untransformed)"


def _selected_index(shapes_layer):
    selected = sorted(int(index) for index in shapes_layer.selected_data)
    if len(selected) > 1:
        raise ValueError("Select only one shape for the spatial scan.")
    return selected[-1] if selected else len(shapes_layer.data) - 1


def snapshot_scan_roi(shapes_layer, *, image_layer=None):
    """Snapshot shape type and geometry without changing drawing/selection.

    Public napari transforms convert Shapes-local coordinates through world
    into image-local pixels. Without a reference image, only an identity
    Shapes-to-world transform is safe to interpret as camera pixels.
    """
    if len(shapes_layer.data) == 0:
        raise ValueError("The active Shapes layer is empty.")
    index = _selected_index(shapes_layer)
    vertices = np.array(shapes_layer.data[index], dtype=float, copy=True)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("Use a 2D Shapes layer for spatial mapping.")
    types = shapes_layer.shape_type
    kind = types if isinstance(types, str) else types[index]
    kind = str(getattr(kind, "value", kind)).lower()
    if kind not in {"rectangle", "polygon", "ellipse"}:
        raise ValueError("Draw a closed rectangle, ellipse, or polygon; lines and paths cannot be scanned.")
    to_world = getattr(shapes_layer, "data_to_world", None)
    world = np.array([to_world(vertex) for vertex in vertices]) if callable(to_world) else vertices
    if image_layer is not None:
        if image_layer.ndim != 2:
            raise ValueError("The scan reference image must be two-dimensional.")
        pixels = np.array([image_layer.world_to_data(vertex) for vertex in world])
        name = image_layer.name
    else:
        if not np.allclose(world, vertices, rtol=1e-12, atol=1e-9):
            raise ValueError("Show a matching 2D camera image to interpret a transformed ROI in pixels.")
        pixels = vertices
        name = "Camera pixels (untransformed)"
    if pixels.shape != vertices.shape or not np.isfinite(pixels).all():
        raise ValueError("The ROI cannot be mapped to finite 2D image-pixel coordinates.")
    return ScanROI(tuple(tuple(float(value) for value in point) for point in pixels),
                   kind, index, str(name))


def scan_image_reference(viewer, width, height):
    """Find an unambiguous visible 2D camera-pixel frame, never guess a transform."""
    from napari.layers import Image

    visible_images = [layer for layer in viewer.layers
                      if isinstance(layer, Image) and layer.visible]
    candidates = []
    for layer in visible_images:
        if layer.ndim != 2 or getattr(layer, "multiscale", False):
            continue
        image_shape = layer.data.shape[:2] if layer.rgb else layer.data.shape[-2:]
        if tuple(image_shape) == (height, width):
            candidates.append(layer)
    if not candidates:
        if visible_images:
            raise ValueError("Show a 2D image matching the camera dimensions before spatial mapping.")
        return None
    basis = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    reference = np.array([candidates[0].data_to_world(point) for point in basis])
    for layer in candidates[1:]:
        transform = np.array([layer.data_to_world(point) for point in basis])
        if not np.allclose(reference, transform, rtol=1e-12, atol=1e-9):
            raise ValueError("Visible camera-sized images have different transforms. Hide unrelated images and retry.")
    return candidates[0]


def snapshot_scan_shape(shapes_layer: Any, *, finish_interaction=True) -> np.ndarray:
    """Copy the selected scan shape and leave its layer safe to interact with.

    Napari 0.5.0 can transiently leave a selected Shapes layer with
    ``_value is None``. Toggling that layer's visibility then refreshes its
    highlight and raises ``TypeError: 'NoneType' object is not subscriptable``.
    A spatial scan only needs a fixed copy of the ROI, so finish the
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
