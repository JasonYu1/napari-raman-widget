"""Mask visualization and center-point selection tools."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.ndimage import center_of_mass, distance_transform_edt
from skimage.draw import disk

__all__ = [
    "add_mask_with_hole",
    "find_clear_center_point",
    "get_n_most_centered_coms",
]


_AUTOFOCUS_TARGETS = {
    "glass",
    "quartz",
    "laser",
    "software",
}


def _validate_rgba_value(
    value: int,
    name: str,
) -> int:
    """Validate an eight-bit alpha value."""
    value = int(value)

    if not 0 <= value <= 255:
        raise ValueError(
            f"{name} must be between 0 and 255."
        )

    return value


def _validate_rgb_color(
    color: Sequence[int],
    name: str,
) -> tuple[int, int, int]:
    """Validate an eight-bit RGB color."""
    if len(color) != 3:
        raise ValueError(
            f"{name} must contain three values."
        )

    validated = tuple(int(value) for value in color)

    if any(
        value < 0 or value > 255
        for value in validated
    ):
        raise ValueError(
            f"Every {name} value must be between 0 and 255."
        )

    return validated


def add_mask_with_hole(
    viewer: Any,
    image_size: tuple[int, int],
    circle_radius: float = 200,
    color: Sequence[int] = (255, 0, 0),
    alpha: int = 50,
    circle_center: tuple[float, float] | None = None,
    small_circle_radius: float = 10,
    small_circle_color: Sequence[int] = (0, 255, 0),
    small_circle_alpha: int = 255,
):
    """Add a targeting overlay to a napari viewer.

    The overlay contains a translucent colored region, a transparent
    circular viewing area, and a small central targeting marker.

    Parameters
    ----------
    viewer
        Napari viewer receiving the overlay.
    image_size
        Image dimensions in ``(height, width)`` order.
    circle_radius
        Radius of the transparent viewing area, in pixels.
    color
        RGB color of the surrounding overlay.
    alpha
        Alpha value of the surrounding overlay.
    circle_center
        Center in ``(y, x)`` order. The image center is used when this
        value is omitted.
    small_circle_radius
        Radius of the central targeting marker.
    small_circle_color
        RGB color of the central targeting marker.
    small_circle_alpha
        Alpha value of the central targeting marker.

    Returns
    -------
    napari.layers.Image
        The image layer added to the viewer.
    """
    if len(image_size) != 2:
        raise ValueError(
            "image_size must contain height and width."
        )

    height = int(image_size[0])
    width = int(image_size[1])

    if height <= 0 or width <= 0:
        raise ValueError(
            "Image dimensions must be greater than zero."
        )

    if circle_radius < 0:
        raise ValueError(
            "circle_radius cannot be negative."
        )

    if small_circle_radius < 0:
        raise ValueError(
            "small_circle_radius cannot be negative."
        )

    color = _validate_rgb_color(
        color,
        "color",
    )
    small_circle_color = _validate_rgb_color(
        small_circle_color,
        "small_circle_color",
    )
    alpha = _validate_rgba_value(
        alpha,
        "alpha",
    )
    small_circle_alpha = _validate_rgba_value(
        small_circle_alpha,
        "small_circle_alpha",
    )

    if circle_center is None:
        circle_center = (
            height / 2,
            width / 2,
        )

    rgba_image = np.zeros(
        (height, width, 4),
        dtype=np.uint8,
    )
    rgba_image[:, :, :3] = color
    rgba_image[:, :, 3] = alpha

    main_rows, main_columns = disk(
        circle_center,
        circle_radius,
        shape=(height, width),
    )
    rgba_image[
        main_rows,
        main_columns,
        :,
    ] = 0

    marker_rows, marker_columns = disk(
        circle_center,
        small_circle_radius,
        shape=(height, width),
    )
    rgba_image[
        marker_rows,
        marker_columns,
        :3,
    ] = small_circle_color
    rgba_image[
        marker_rows,
        marker_columns,
        3,
    ] = small_circle_alpha

    return viewer.add_image(
        rgba_image,
        rgb=True,
        name="Targeting guide",
    )


def find_clear_center_point(
    mask: np.ndarray,
    threshold: float = 20,
) -> np.ndarray:
    """Find a central background point away from labeled objects.

    Parameters
    ----------
    mask
        Two-dimensional mask. Zero-valued pixels are treated as
        background.
    threshold
        Minimum required distance from a labeled object, in pixels.

    Returns
    -------
    numpy.ndarray
        Selected coordinate in ``(y, x)`` order.
    """
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError(
            "mask must be two-dimensional."
        )

    if threshold < 0:
        raise ValueError(
            "threshold cannot be negative."
        )

    background = mask == 0
    distance_map = distance_transform_edt(
        background
    )
    valid_points = np.argwhere(
        distance_map >= threshold
    )

    if len(valid_points) == 0:
        raise ValueError(
            "No background point meets the requested clearance."
        )

    image_center = (
        np.asarray(mask.shape, dtype=float)
        / 2
    )
    distances_to_center = np.linalg.norm(
        valid_points - image_center,
        axis=1,
    )
    best_index = int(
        np.argmin(distances_to_center)
    )

    return valid_points[best_index].astype(float)


def get_n_most_centered_coms(
    label_mask: np.ndarray,
    N: int = 10,
    center: tuple[float, float] | None = None,
    radius: float = 250,
    autofocus_object: str | None = "glass",
    bkd_threshold: float = 50,
) -> np.ndarray:
    """Return labeled-object centers closest to an image center.

    When an autofocus target is requested, the first returned coordinate
    is a clear background point. The remaining coordinates are centers
    of labeled objects.

    Parameters
    ----------
    label_mask
        Two-dimensional labeled mask where zero represents background.
    N
        Maximum total number of returned points.
    center
        Reference center in ``(y, x)`` order. The mask center is used
        when omitted.
    radius
        Maximum allowed distance from the reference center.
    autofocus_object
        Autofocus target. Supported autofocus values are ``glass``,
        ``quartz``, ``laser``, and ``software``. Use ``None`` or
        ``"None"`` to return only labeled-object centers.
    bkd_threshold
        Minimum background clearance used for the autofocus point.

    Returns
    -------
    numpy.ndarray
        Coordinates with shape ``(number_of_points, 2)`` in ``(y, x)``
        order.
    """
    label_mask = np.asarray(label_mask)

    if label_mask.ndim != 2:
        raise ValueError(
            "label_mask must be two-dimensional."
        )

    if N < 1:
        return np.empty(
            (0, 2),
            dtype=float,
        )

    if radius < 0:
        raise ValueError(
            "radius cannot be negative."
        )

    if center is None:
        reference_center = (
            np.asarray(
                label_mask.shape,
                dtype=float,
            )
            / 2
        )
    else:
        reference_center = np.asarray(
            center,
            dtype=float,
        )

        if reference_center.shape != (2,):
            raise ValueError(
                "center must contain one (y, x) coordinate."
            )

    labels = np.unique(label_mask)
    labels = labels[labels != 0]

    centers_with_distances = []

    for label_value in labels:
        object_center = np.asarray(
            center_of_mass(
                np.ones_like(
                    label_mask,
                    dtype=float,
                ),
                labels=label_mask,
                index=label_value,
            ),
            dtype=float,
        )

        if not np.isfinite(object_center).all():
            continue

        distance = float(
            np.linalg.norm(
                object_center
                - reference_center
            )
        )

        if distance <= radius:
            centers_with_distances.append(
                (distance, object_center)
            )

    centers_with_distances.sort(
        key=lambda item: item[0]
    )

    selected_points = [
        object_center
        for _, object_center
        in centers_with_distances
    ]

    normalized_autofocus = (
        autofocus_object.strip().lower()
        if isinstance(autofocus_object, str)
        else autofocus_object
    )

    if normalized_autofocus in _AUTOFOCUS_TARGETS:
        autofocus_point = find_clear_center_point(
            label_mask,
            threshold=bkd_threshold,
        )
        selected_points.insert(
            0,
            autofocus_point,
        )

    selected_points = selected_points[:N]

    if not selected_points:
        return np.empty(
            (0, 2),
            dtype=float,
        )

    return np.asarray(
        selected_points,
        dtype=float,
    )