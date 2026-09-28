"""Find safe Raman targets for cells that newly appear during an MDA."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

__all__ = ["find_new_cell_targets"]


def _validate_label_image(labels: np.ndarray, name: str) -> np.ndarray:
    """Return a validated two-dimensional, nonnegative label image."""
    result = np.asarray(labels)
    if result.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if not np.issubdtype(result.dtype, np.integer):
        raise TypeError(f"{name} must contain integer labels")
    if np.any(result < 0):
        raise ValueError(f"{name} cannot contain negative labels")
    return result


def _as_points(points_yx: np.ndarray) -> np.ndarray:
    """Normalize a possibly empty collection to an ``(N, 2)`` array."""
    points = np.asarray(points_yx, dtype=float)
    if points.size == 0:
        return np.empty((0, 2), dtype=float)
    points = np.atleast_2d(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("existing_points_yx must have shape (N, 2)")
    return points


def _nonzero_labels(labels: np.ndarray) -> np.ndarray:
    """Return sorted foreground label values as signed integers."""
    values = np.unique(labels)
    return values[values != 0].astype(np.int64, copy=False)


def _labels_at_points(
    labels: np.ndarray,
    points_yx: np.ndarray,
    scale: float,
) -> set[int]:
    """Return foreground labels containing full-resolution points."""
    if len(points_yx) == 0:
        return set()

    finite = np.isfinite(points_yx).all(axis=1)
    pixels = np.zeros((len(points_yx), 2), dtype=np.int64)
    pixels[finite] = np.floor(points_yx[finite] / scale).astype(np.int64)
    in_bounds = (
        finite
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < labels.shape[0])
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < labels.shape[1])
    )
    if not np.any(in_bounds):
        return set()

    values = labels[pixels[in_bounds, 0], pixels[in_bounds, 1]]
    return {int(value) for value in values if value != 0}


def _deepest_interior_point(
    labels: np.ndarray,
    label_value: int,
) -> tuple[np.ndarray, float]:
    """Return a deterministic deepest-inside point and its safe radius."""
    object_pixels = labels == label_value
    interior_distance = ndi.distance_transform_edt(
        np.pad(object_pixels, 1, mode="constant", constant_values=False)
    )[1:-1, 1:-1]
    safe_radius = float(interior_distance.max(initial=0.0))
    candidates = np.argwhere(interior_distance == safe_radius)
    if len(candidates) == 0:
        return np.asarray((np.nan, np.nan), dtype=float), safe_radius

    object_center = np.asarray(ndi.center_of_mass(object_pixels), dtype=float)
    if not np.isfinite(object_center).all():
        return candidates[0].astype(float), safe_radius

    closest = int(
        np.argmin(np.linalg.norm(candidates - object_center, axis=1))
    )
    return candidates[closest].astype(float), safe_radius


def _proximity_new_labels(
    previous_labels: np.ndarray,
    current_labels: np.ndarray,
    *,
    scale: float,
    maximum_distance: float,
) -> np.ndarray:
    """Find conservative new objects without comparing raw label numbers.

    A current object is considered a continuation when any of its pixels is
    sufficiently close to any previous foreground pixel.  This intentionally
    errs toward missing an addition in a crowded field rather than adding a
    duplicate target after btrack has failed.
    """
    current_ids = _nonzero_labels(current_labels)
    if len(current_ids) == 0:
        return current_ids

    previous_foreground = previous_labels != 0
    if not np.any(previous_foreground):
        return current_ids

    distance_to_previous = ndi.distance_transform_edt(~previous_foreground)
    maximum_distance_in_mask = maximum_distance / scale
    new_ids = [
        int(label_value)
        for label_value in current_ids
        if float(distance_to_previous[current_labels == label_value].min())
        > maximum_distance_in_mask
    ]
    return np.asarray(new_ids, dtype=np.int64)


def find_new_cell_targets(
    previous_labels: np.ndarray,
    current_labels: np.ndarray,
    existing_points_yx: np.ndarray,
    *,
    scale: float = 1.0,
    tracked_labels: np.ndarray | None = None,
    fallback_match_distance: float = 60.0,
    minimum_area: int = 16,
    minimum_interior_radius: float = 2.0,
    center_yx: tuple[float, float] | np.ndarray | None = None,
    max_new_cells: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return IDs and safe target points for newly appearing cells.

    Parameters
    ----------
    previous_labels, current_labels
        Consecutive raw segmentation masks.  Raw label values are local to a
        frame and are never compared with each other.
    existing_points_yx
        Current Raman cell targets in full-resolution ``(y, x)`` pixels.  A
        new object that already contains a target is not returned.  This is
        what prevents a manually added point from receiving an automatic
        duplicate.
    scale
        Ratio of full-resolution camera pixels to segmentation-mask pixels.
    tracked_labels
        Optional trustworthy btrack output with shape ``(2, Y, X)``.  When it
        is supplied, track IDs present in the current frame but absent from
        the previous frame are new.  Pass ``None`` after a tracking failure;
        the conservative, label-ID-independent proximity fallback is used.
    fallback_match_distance
        Full-resolution distance from previous foreground within which a raw
        current object is conservatively treated as a continuation.
    minimum_area, minimum_interior_radius
        Minimum object area and deepest-inside radius, both measured in
        segmentation-mask pixels.
    center_yx
        Optional full-resolution reference point used to order results from
        nearest to farthest.  The image center is used by default.
    max_new_cells
        Optional maximum number of returned targets after filtering.

    Returns
    -------
    label_ids, points_yx
        One-dimensional label IDs and full-resolution deepest-interior target
        coordinates with shape ``(N, 2)``.  IDs refer to the btrack current
        frame when ``tracked_labels`` is supplied, otherwise to the current
        raw mask and should not be persisted across frames.

    Notes
    -----
    The caller should use the first segmentation as a baseline and invoke this
    helper only for later time points.  It should also skip mask-reuse events
    representing another position index for the same physical field of view.
    """
    previous = _validate_label_image(previous_labels, "previous_labels")
    current = _validate_label_image(current_labels, "current_labels")
    if previous.shape != current.shape:
        raise ValueError("previous_labels and current_labels must have the same shape")

    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite number")
    fallback_match_distance = float(fallback_match_distance)
    if (
        not np.isfinite(fallback_match_distance)
        or fallback_match_distance < 0
    ):
        raise ValueError("fallback_match_distance must be nonnegative and finite")
    if int(minimum_area) != minimum_area or minimum_area < 1:
        raise ValueError("minimum_area must be a positive integer")
    minimum_area = int(minimum_area)
    minimum_interior_radius = float(minimum_interior_radius)
    if (
        not np.isfinite(minimum_interior_radius)
        or minimum_interior_radius <= 0
    ):
        raise ValueError("minimum_interior_radius must be positive and finite")
    if max_new_cells is not None:
        if int(max_new_cells) != max_new_cells or max_new_cells < 0:
            raise ValueError("max_new_cells must be a nonnegative integer or None")
        max_new_cells = int(max_new_cells)

    points = _as_points(existing_points_yx)
    if tracked_labels is None:
        target_mask = current
        candidate_ids = _proximity_new_labels(
            previous,
            current,
            scale=scale,
            maximum_distance=fallback_match_distance,
        )
    else:
        tracked = np.asarray(tracked_labels)
        if tracked.ndim != 3 or tracked.shape[0] != 2:
            raise ValueError("tracked_labels must have shape (2, Y, X)")
        if tracked.shape[1:] != current.shape:
            raise ValueError("tracked_labels must match the raw label image shape")
        if not np.issubdtype(tracked.dtype, np.integer):
            raise TypeError("tracked_labels must contain integer labels")
        if np.any(tracked < 0):
            raise ValueError("tracked_labels cannot contain negative labels")
        target_mask = tracked[1]
        candidate_ids = np.setdiff1d(
            _nonzero_labels(tracked[1]),
            _nonzero_labels(tracked[0]),
            assume_unique=True,
        )

    represented = _labels_at_points(target_mask, points, scale)
    candidate_ids = np.asarray(
        [value for value in candidate_ids if int(value) not in represented],
        dtype=np.int64,
    )

    if center_yx is None:
        reference_center = np.asarray(target_mask.shape, dtype=float) * scale / 2
    else:
        reference_center = np.asarray(center_yx, dtype=float)
        if reference_center.shape != (2,) or not np.isfinite(reference_center).all():
            raise ValueError("center_yx must contain one finite (y, x) coordinate")

    accepted: list[tuple[float, int, np.ndarray]] = []
    for raw_label in candidate_ids:
        label_value = int(raw_label)
        object_area = int(np.count_nonzero(target_mask == label_value))
        if object_area < minimum_area:
            continue
        interior_point, safe_radius = _deepest_interior_point(
            target_mask,
            label_value,
        )
        if (
            safe_radius < minimum_interior_radius
            or not np.isfinite(interior_point).all()
        ):
            continue
        point_yx = interior_point * scale
        distance = float(np.linalg.norm(point_yx - reference_center))
        accepted.append((distance, label_value, point_yx))

    accepted.sort(key=lambda item: (item[0], item[1]))
    if max_new_cells is not None:
        accepted = accepted[:max_new_cells]
    if not accepted:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 2), dtype=float),
        )

    return (
        np.asarray([item[1] for item in accepted], dtype=np.int64),
        np.asarray([item[2] for item in accepted], dtype=float),
    )
