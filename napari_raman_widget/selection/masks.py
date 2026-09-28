"""Mask visualization and center-point selection tools."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    center_of_mass,
    distance_transform_edt,
    label as connected_components,
)
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
    center: tuple[float, float] | None = None,
    exclusion_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Find a central background point away from labeled objects.

    Parameters
    ----------
    mask
        Two-dimensional mask. Zero-valued pixels are treated as
        background.
    threshold
        Minimum required distance from a labeled object, in pixels.
    center
        Preferred point in ``(y, x)`` order. The image center is used when
        omitted.
    exclusion_mask
        Optional boolean mask of pixels that must never be selected. These
        pixels are treated like labeled foreground when measuring clearance.

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

    excluded = _prepare_exclusion_mask(
        exclusion_mask,
        mask.shape,
    )
    background = (mask == 0) & ~excluded
    distance_map = distance_transform_edt(
        background
    )
    valid_points = np.argwhere(
        background & (distance_map >= threshold)
    )

    if len(valid_points) == 0:
        raise ValueError(
            "No background point meets the requested clearance."
        )

    if center is None:
        image_center = (
            np.asarray(mask.shape, dtype=float)
            / 2
        )
    else:
        image_center = np.asarray(center, dtype=float)
        if image_center.shape != (2,):
            raise ValueError(
                "center must contain one (y, x) coordinate."
            )
    distances_to_center = np.linalg.norm(
        valid_points - image_center,
        axis=1,
    )
    best_index = int(
        np.argmin(distances_to_center)
    )

    return valid_points[best_index].astype(float)


def _prepare_exclusion_mask(
    exclusion_mask: np.ndarray | None,
    shape: tuple[int, int],
    margin: int = 0,
) -> np.ndarray:
    """Validate and optionally expand a hard no-target mask."""
    if margin < 0:
        raise ValueError(
            "exclusion_margin cannot be negative."
        )

    if exclusion_mask is None:
        return np.zeros(shape, dtype=bool)

    excluded = np.asarray(exclusion_mask, dtype=bool)
    if excluded.ndim != 2 or excluded.shape != shape:
        raise ValueError(
            "exclusion_mask must be two-dimensional and match label_mask."
        )
    if margin:
        excluded = binary_dilation(
            excluded,
            iterations=int(margin),
        )
    return excluded


def _safe_label_target(
    label_mask: np.ndarray,
    label_value: int,
    excluded: np.ndarray,
    minimum_safe_area: int = 16,
    minimum_safe_radius: float = 2.0,
) -> np.ndarray | None:
    """Return a deep pixel in a viable cell body outside exclusion."""
    object_pixels = label_mask == label_value
    safe_pixels = object_pixels & ~excluded
    if not np.any(safe_pixels):
        return None

    object_center = np.asarray(
        center_of_mass(object_pixels),
        dtype=float,
    )
    components, component_count = connected_components(
        safe_pixels,
        structure=np.ones((3, 3), dtype=bool),
    )
    viable_targets = []
    for component_value in range(1, component_count + 1):
        component = components == component_value
        area = int(np.count_nonzero(component))
        if area < int(minimum_safe_area):
            continue
        interior_distance = distance_transform_edt(
            np.pad(component, 1, mode="constant", constant_values=False)
        )[1:-1, 1:-1]
        maximum_distance = float(interior_distance.max())
        if maximum_distance < float(minimum_safe_radius):
            continue
        candidates = np.argwhere(interior_distance == maximum_distance)
        if np.isfinite(object_center).all():
            closest_index = int(
                np.argmin(
                    np.linalg.norm(
                        candidates - object_center,
                        axis=1,
                    )
                )
            )
            target = candidates[closest_index]
        else:
            target = candidates[0]
        viable_targets.append(
            (maximum_distance, area, target.astype(float))
        )

    if not viable_targets:
        return None
    viable_targets.sort(
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )
    return viable_targets[0][2]


def get_n_most_centered_coms(
    label_mask: np.ndarray,
    N: int = 10,
    center: tuple[float, float] | None = None,
    radius: float = 250,
    autofocus_object: str | None = "glass",
    bkd_threshold: float = 50,
    exclusion_mask: np.ndarray | None = None,
    exclusion_margin: int = 0,
    maximum_exclusion_overlap: float = 0.25,
    maximum_axis_ratio_near_exclusion: float = 1.6,
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
    exclusion_mask
        Optional hard no-target mask, such as the registered pillar mask.
    exclusion_margin
        Number of pixels by which to expand ``exclusion_mask`` before cell
        targets are chosen.
    maximum_exclusion_overlap
        Retained for compatibility with the previous whole-label filter.
    maximum_axis_ratio_near_exclusion
        Retained for compatibility with the previous whole-label filter.

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
    if not 0 <= maximum_exclusion_overlap <= 1:
        raise ValueError(
            "maximum_exclusion_overlap must be between zero and one."
        )
    if maximum_axis_ratio_near_exclusion < 1:
        raise ValueError(
            "maximum_axis_ratio_near_exclusion must be at least one."
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
    has_exclusion = exclusion_mask is not None
    excluded = _prepare_exclusion_mask(
        exclusion_mask,
        label_mask.shape,
        margin=int(exclusion_margin),
    )
    _ = maximum_exclusion_overlap, maximum_axis_ratio_near_exclusion

    centers_with_distances = []

    for label_value in labels:
        object_pixels = label_mask == label_value
        if has_exclusion and np.any(object_pixels & excluded):
            object_center = _safe_label_target(
                label_mask,
                label_value,
                excluded,
            )
            if object_center is None:
                continue
        else:
            object_center = np.asarray(
                center_of_mass(object_pixels),
                dtype=float,
            )
            if not np.isfinite(object_center).all():
                continue
            if has_exclusion:
                center_pixel = np.clip(
                    np.rint(object_center).astype(int),
                    (0, 0),
                    np.asarray(label_mask.shape) - 1,
                )
                if excluded[tuple(center_pixel)]:
                    object_center = _safe_label_target(
                        label_mask,
                        label_value,
                        excluded,
                    )
                    if object_center is None:
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
            center=(tuple(reference_center) if has_exclusion else None),
            exclusion_mask=(excluded if has_exclusion else None),
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
