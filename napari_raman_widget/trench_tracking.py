"""Conservative tracking inside the learned run-35 trench lattice.

Every trench uses one fixed closed rectangle-with-trapezoid-head primitive.
Runtime registration fits only a single rotation and translation shared by the
whole lattice.  No wall or trench is learned independently during an
experiment.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import line, polygon
from skimage.feature import match_template
from skimage.filters import sobel
from skimage.morphology import (
    binary_closing,
    binary_erosion,
    disk,
    remove_small_holes,
)
from skimage.registration import phase_cross_correlation
from skimage.segmentation import watershed
from skimage.transform import probabilistic_hough_line

from .cell_identity import CellReference, cell_reference, match_cell

__all__ = [
    "DEFAULT_TRENCH_CONFIG",
    "ConstrainedTrenchTracker",
    "TrenchConstraintTracker",
    "RegisteredTrenchGeometry",
    "TrenchPose",
    "TrenchSelectionResult",
    "TrenchTarget",
    "TrenchTargetResult",
    "TrenchTrackingConfig",
    "TrenchUpdate",
    "filter_labels_to_trenches",
    "nearest_trench_index",
    "registered_trench_geometry",
    "reference_lattice_vertices",
]


@dataclass(frozen=True)
class TrenchTrackingConfig:
    """Fixed learned geometry and conservative acceptance thresholds."""

    reference_image_shape_yx: tuple[int, int] = (1024, 1344)
    canonical_patch_shape_yx: tuple[int, int] = (93, 660)
    canonical_polygon_yx: tuple[tuple[float, float], ...] = (
        (25.25, 188.5),
        (5.5, 306.0),
        (5.5, 622.0),
        (62.0, 622.0),
        (62.0, 306.0),
        (42.25, 188.5),
        (25.25, 188.5),
    )
    reference_anchor_yx: tuple[float, float] = (470.0, 500.0)
    lattice_indices: tuple[int, ...] = tuple(range(-4, 6))
    pitch_px: float = 92.5
    default_angle_deg: float = 4.215
    # An ideal closed-outline reference scores about 0.54 on good run-35
    # images; the known bad p003 geometry fails registration.
    minimum_geometry_score: float = 0.45
    maximum_step_px: float = 45.0
    # Identity comes from the brightfield image, not mutable Cellpose IDs or
    # overlap with a mask that can split, merge, or disappear between frames.
    track_cell_appearance: bool = True
    wall_clearance_px: int = 4
    # The closed-end wall and its phase-contrast halo cannot contain a yeast
    # cell.  Exclude a cap extending this many flat-tip lengths inward toward
    # the trapezoid shoulder.  The learned short edge is 17 px, so the default
    # no-cell depth is 34 full-resolution pixels.
    tip_exclusion_depth_multiplier: float = 2.0
    minimum_cell_area_px: float = 300.0
    # Full Cellpose mother-cell outlines in run_40 exceed the old 2600 px
    # cap. Areas are measured in full-resolution pixels at every scale.
    maximum_cell_area_px: float = 4000.0
    minimum_area_ratio: float = 0.40
    maximum_area_ratio: float = 2.50
    # Legacy overlap gate, used only with track_cell_appearance=False.
    # The default image matcher permits real cell movement without requiring
    # its old and new segmentation footprints to overlap.
    minimum_identity_overlap: float = 0.25
    minimum_template_score: float = 0.30
    maximum_cell_radius_px: int = 27
    template_radius_px: int = 18
    maximum_seed_projection_px: float = 5.0
    maximum_incremental_shift_yx: tuple[float, float] = (25.0, 35.0)
    # Center-cell selection can translate the device laterally by hundreds of
    # pixels.  The closed-end geometry and confidence gate still decide
    # whether that wider initial registration is trustworthy.
    maximum_global_shift_yx: tuple[float, float] = (25.0, 300.0)
    match_roi_yx: tuple[int, int, int, int] = (0, 1024, 480, 1180)

    def __post_init__(self) -> None:
        vertices = np.asarray(self.canonical_polygon_yx, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError("canonical_polygon_yx must have shape (N, 2)")
        if len(vertices) < 4 or not np.allclose(vertices[0], vertices[-1]):
            raise ValueError("canonical trench polygon must be explicitly closed")
        if any(value <= 0 for value in self.reference_image_shape_yx):
            raise ValueError("reference image dimensions must be positive")
        if not self.lattice_indices or self.pitch_px <= 0:
            raise ValueError("the trench lattice must have a positive pitch")
        if not 0 < self.minimum_geometry_score <= 1:
            raise ValueError("minimum_geometry_score must be in (0, 1]")
        if self.maximum_step_px <= 0 or self.wall_clearance_px < 0:
            raise ValueError("step and wall-clearance limits are invalid")
        if self.tip_exclusion_depth_multiplier < 0:
            raise ValueError("tip_exclusion_depth_multiplier cannot be negative")
        if not 0 < self.minimum_cell_area_px <= self.maximum_cell_area_px:
            raise ValueError("cell area bounds are invalid")
        if not 0 < self.minimum_area_ratio <= self.maximum_area_ratio:
            raise ValueError("cell area-ratio bounds are invalid")
        if not 0 < self.minimum_identity_overlap <= 1:
            raise ValueError("minimum_identity_overlap must be in (0, 1]")


DEFAULT_TRENCH_CONFIG = TrenchTrackingConfig()


def _point(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain two finite YX coordinates")
    return result


def _rotate_points(
    points_yx: np.ndarray,
    image_shape_yx: tuple[int, int],
    angle_deg: float,
) -> np.ndarray:
    """Apply scipy.ndimage.rotate's content transform to YX coordinates."""

    points = np.atleast_2d(np.asarray(points_yx, dtype=float))
    center_y = (image_shape_yx[0] - 1) / 2.0
    center_x = (image_shape_yx[1] - 1) / 2.0
    delta_y = points[:, 0] - center_y
    delta_x = points[:, 1] - center_x
    angle = np.deg2rad(angle_deg)
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    return np.column_stack(
        (
            center_y + cosine * delta_y - sine * delta_x,
            center_x + sine * delta_y + cosine * delta_x,
        )
    )


@dataclass(frozen=True)
class TrenchPose:
    """One global transform from the learned lattice to the current frame."""

    angle_deg: float = 0.0
    shift_yx: tuple[float, float] = (0.0, 0.0)
    score: float = 1.0

    def apply(
        self,
        points_yx: np.ndarray,
        image_shape_yx: tuple[int, int] = (1024, 1344),
    ) -> np.ndarray:
        points = np.asarray(points_yx, dtype=float)
        shape = points.shape
        transformed = _rotate_points(
            points.reshape(-1, 2), image_shape_yx, self.angle_deg
        )
        transformed += np.asarray(self.shift_yx, dtype=float)
        return transformed.reshape(shape)

    def inverse(
        self,
        points_yx: np.ndarray,
        image_shape_yx: tuple[int, int] = (1024, 1344),
    ) -> np.ndarray:
        points = np.asarray(points_yx, dtype=float)
        shape = points.shape
        unshifted = points.reshape(-1, 2) - np.asarray(self.shift_yx, dtype=float)
        return _rotate_points(unshifted, image_shape_yx, -self.angle_deg).reshape(shape)


@dataclass(frozen=True)
class RegisteredTrenchGeometry:
    """Registered raw and cell-safe trench masks for one camera frame."""

    trench_masks: tuple[np.ndarray, ...]
    safe_masks: tuple[np.ndarray, ...]
    geometry_valid: bool
    geometry_score: float
    pose: TrenchPose | None

    @property
    def safe_union(self) -> np.ndarray:
        """Return the union of all cap-aware, wall-cleared trench interiors."""

        if not self.safe_masks:
            return np.zeros((0, 0), dtype=bool)
        return np.logical_or.reduce(self.safe_masks)


@dataclass(frozen=True)
class TrenchTarget:
    """One proposed live point update; ``token`` must remain stable."""

    token: Hashable
    previous_yx: tuple[float, float] | np.ndarray
    proposed_yx: tuple[float, float] | np.ndarray | None = None


@dataclass(frozen=True)
class TrenchTargetResult:
    token: Hashable
    point_yx: np.ndarray
    accepted: bool
    reason: str
    trench_index: int | None
    step_px: float
    area_px: float
    template_score: float


@dataclass(frozen=True)
class TrenchUpdate:
    """Corrected points and segmentation safe for downstream consumers."""

    points_yx: np.ndarray
    labels: np.ndarray
    results: tuple[TrenchTargetResult, ...]
    geometry_valid: bool
    geometry_score: float
    pose: TrenchPose | None

    @property
    def accepted(self) -> np.ndarray:
        return np.asarray([result.accepted for result in self.results], dtype=bool)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(result.reason for result in self.results)


@dataclass(frozen=True)
class TrenchSelectionResult:
    """Cellpose labels restricted to one registered safe trench lattice."""

    filtered_labels: np.ndarray
    safe_interior_mask: np.ndarray
    geometry_valid: bool
    geometry_score: float
    pose: TrenchPose | None

    @property
    def exclusion_mask(self) -> np.ndarray:
        """Return pixels that automated cell targets must not occupy."""

        return ~np.asarray(self.safe_interior_mask, dtype=bool)


@dataclass
class _TargetState:
    trench_index: int
    canonical_point_yx: np.ndarray
    previous_template: np.ndarray
    previous_area_px: float
    canonical_cell_yx: np.ndarray


@dataclass
class _AppearanceState:
    trench_index: int
    # The image feature used for identity stays independent of mask centering.
    canonical_point_yx: np.ndarray
    reference: CellReference
    canonical_aim_yx: np.ndarray
    previous_area_px: float = 0.0


@dataclass
class _PositionState:
    pose: TrenchPose | None = None
    response: np.ndarray | None = None
    targets: dict[Hashable, _TargetState] = field(default_factory=dict)
    appearance_targets: dict[Hashable, _AppearanceState] = field(default_factory=dict)


def _canonical_to_raw_patch(
    points_yx: np.ndarray,
    patch_shape_yx: tuple[int, int],
    angle_deg: float,
) -> np.ndarray:
    center_y = (patch_shape_yx[0] - 1) / 2.0
    center_x = (patch_shape_yx[1] - 1) / 2.0
    points = np.asarray(points_yx, dtype=float)
    delta_y, delta_x = points[:, 0] - center_y, points[:, 1] - center_x
    angle = np.deg2rad(angle_deg)
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    return np.column_stack(
        (
            center_y + cosine * delta_y + sine * delta_x,
            center_x - sine * delta_y + cosine * delta_x,
        )
    )


@lru_cache(maxsize=None)
def reference_lattice_vertices(
    config: TrenchTrackingConfig = DEFAULT_TRENCH_CONFIG,
) -> np.ndarray:
    """Return the ten exact run-35 reference polygons in full-image YX."""

    raw_patch = _canonical_to_raw_patch(
        np.asarray(config.canonical_polygon_yx, dtype=float),
        config.canonical_patch_shape_yx,
        config.default_angle_deg,
    )
    origin = np.asarray(
        (
            config.reference_anchor_yx[0] - config.canonical_patch_shape_yx[0] // 2,
            config.reference_anchor_yx[1],
        )
    )
    base = raw_patch + origin
    angle = np.deg2rad(config.default_angle_deg)
    repeat = np.asarray(
        (config.pitch_px * np.cos(angle), -config.pitch_px * np.sin(angle))
    )
    vertices = np.stack(
        [base + float(index) * repeat for index in config.lattice_indices]
    )
    vertices.setflags(write=False)
    return vertices


@lru_cache(maxsize=None)
def _reference_response(config: TrenchTrackingConfig) -> np.ndarray:
    response = np.zeros(config.reference_image_shape_yx, dtype=np.float32)
    for vertices in reference_lattice_vertices(config):
        for start, stop in zip(vertices[:-1], vertices[1:], strict=True):
            rows, columns = line(
                *np.rint(start).astype(int), *np.rint(stop).astype(int)
            )
            valid = (
                (rows >= 0)
                & (rows < response.shape[0])
                & (columns >= 0)
                & (columns < response.shape[1])
            )
            response[rows[valid], columns[valid]] = 1.0
    response = ndi.gaussian_filter(response, sigma=1.4)
    response /= float(np.max(response))
    response.setflags(write=False)
    return response


def _wall_response(image: np.ndarray) -> np.ndarray:
    work = np.asarray(image, dtype=np.float32)
    smooth = ndi.gaussian_filter(work, sigma=(1.2, 1.8))
    vertical = np.abs(ndi.sobel(smooth, axis=0))
    horizontal = np.abs(ndi.sobel(smooth, axis=1))
    scale = float(np.percentile(vertical, 99.5))
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros(work.shape, dtype=np.float32)
    strength = np.clip(vertical / scale, 0.0, 1.0)
    orientation = vertical / (vertical + horizontal + np.finfo(np.float32).eps)
    return np.asarray(strength * orientation**2, dtype=np.float32)


def _angle(response: np.ndarray) -> float:
    lines = probabilistic_hough_line(
        response > 0.18,
        threshold=12,
        line_length=80,
        line_gap=12,
        rng=0,
    )
    values, weights = [], []
    for (x0, y0), (x1, y1) in lines:
        dx, dy = float(x1 - x0), float(y1 - y0)
        if dx == 0:
            continue
        value = float(np.degrees(np.arctan2(dy, dx)))
        value = value - 180 if value > 90 else value + 180 if value < -90 else value
        length = float(np.hypot(dx, dy))
        if abs(value) <= 12 and length >= 80:
            values.append(value)
            weights.append(length)
    if not values:
        raise RuntimeError("No sufficiently long trench-wall lines were detected")
    order = np.argsort(values)
    ordered = np.asarray(values)[order]
    cumulative = np.cumsum(np.asarray(weights)[order])
    return float(ordered[np.searchsorted(cumulative, cumulative[-1] / 2)])


def _roi(array: np.ndarray, config: TrenchTrackingConfig) -> np.ndarray:
    y0, y1, x0, x1 = config.match_roi_yx
    return array[y0:y1, x0:x1]


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left).ravel(), np.asarray(right).ravel()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator == 0 else float(np.dot(left, right) / denominator)


def _global_pose(response: np.ndarray, config: TrenchTrackingConfig) -> TrenchPose:
    reference = _reference_response(config)
    initial = float(np.clip(_angle(reference) - _angle(response), -8.0, 8.0))
    best: tuple[float, np.ndarray, float] | None = None
    for angle in np.linspace(initial - 1.25, initial + 1.25, 11):
        if abs(angle) > 8:
            continue
        rotated = ndi.rotate(reference, angle, reshape=False, order=1, mode="constant")
        shift, _, _ = phase_cross_correlation(
            _roi(response, config),
            _roi(rotated, config),
            upsample_factor=2,
            normalization=None,
        )
        shift = np.asarray(shift, dtype=float)
        shift[0] -= np.round(shift[0] / config.pitch_px) * config.pitch_px
        if np.any(np.abs(shift) > config.maximum_global_shift_yx):
            continue
        aligned = ndi.shift(rotated, shift, order=1, mode="constant")
        score = _cosine(_roi(response, config), _roi(aligned, config))
        if best is None or score > best[0]:
            best = score, shift, float(angle)
    if best is None:
        raise RuntimeError("Rigid trench-template registration failed")
    score, shift, angle = best
    return TrenchPose(angle, (float(shift[0]), float(shift[1])), float(score))


def _incremental_pose(
    response: np.ndarray,
    previous_response: np.ndarray,
    previous_pose: TrenchPose,
    config: TrenchTrackingConfig,
) -> TrenchPose | None:
    delta, _, _ = phase_cross_correlation(
        _roi(response, config),
        _roi(previous_response, config),
        upsample_factor=2,
        normalization=None,
    )
    delta = np.asarray(delta, dtype=float)
    delta[0] -= np.round(delta[0] / config.pitch_px) * config.pitch_px
    if np.any(np.abs(delta) > config.maximum_incremental_shift_yx):
        return None
    shift = np.asarray(previous_pose.shift_yx) + delta
    rotated = ndi.rotate(
        _reference_response(config),
        previous_pose.angle_deg,
        reshape=False,
        order=1,
        mode="constant",
    )
    aligned = ndi.shift(rotated, shift, order=1, mode="constant")
    score = _cosine(_roi(response, config), _roi(aligned, config))
    if score < config.minimum_geometry_score:
        return None
    return TrenchPose(
        previous_pose.angle_deg,
        (float(shift[0]), float(shift[1])),
        score,
    )


def _polygon_mask(shape: tuple[int, int], vertices_yx: np.ndarray) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    rows, columns = polygon(vertices_yx[:-1, 0], vertices_yx[:-1, 1], shape=shape)
    result[rows, columns] = True
    return result


def _pose_trench_masks(
    pose: TrenchPose,
    config: TrenchTrackingConfig,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Return raw and cap-aware safe masks for one registered lattice pose."""

    shape = config.reference_image_shape_yx
    vertices = pose.apply(reference_lattice_vertices(config), shape)
    row_grid, column_grid = np.ogrid[: shape[0], : shape[1]]
    trench_masks: list[np.ndarray] = []
    safe_masks: list[np.ndarray] = []
    for polygon_yx in vertices:
        trench_mask = _polygon_mask(shape, polygon_yx)
        safe = (
            binary_erosion(
                trench_mask,
                footprint=disk(config.wall_clearance_px),
            )
            if config.wall_clearance_px
            else trench_mask.copy()
        )

        tip_midpoint = (polygon_yx[0] + polygon_yx[5]) / 2.0
        shoulder_midpoint = (polygon_yx[1] + polygon_yx[4]) / 2.0
        inward_axis = shoulder_midpoint - tip_midpoint
        axis_length = float(np.linalg.norm(inward_axis))
        tip_length = float(np.linalg.norm(polygon_yx[5] - polygon_yx[0]))
        if axis_length <= 0 or tip_length <= 0:
            raise ValueError("trench tip and shoulder geometry must be non-degenerate")
        inward_axis /= axis_length
        exclusion_depth = config.tip_exclusion_depth_multiplier * tip_length
        axial_depth = (
            (row_grid - tip_midpoint[0]) * inward_axis[0]
            + (column_grid - tip_midpoint[1]) * inward_axis[1]
        )
        safe &= axial_depth >= exclusion_depth
        trench_masks.append(trench_mask)
        safe_masks.append(safe)
    return tuple(trench_masks), tuple(safe_masks)


def registered_trench_geometry(
    image: np.ndarray,
    config: TrenchTrackingConfig = DEFAULT_TRENCH_CONFIG,
    *,
    pose_override: TrenchPose | None = None,
) -> RegisteredTrenchGeometry:
    """Register the fixed lattice and return cap-aware masks for ``image``.

    Geometry failures are represented by ``geometry_valid=False`` and empty
    mask tuples.  This lets Cellpose preprocessing and downstream filtering
    share exactly the same fail-closed geometry decision.
    """

    frame = np.asarray(image)
    if frame.ndim != 2:
        raise ValueError("image must be two-dimensional")
    if frame.shape != config.reference_image_shape_yx:
        return RegisteredTrenchGeometry((), (), False, 0.0, None)

    pose = pose_override
    if pose is None:
        try:
            pose = _global_pose(_wall_response(frame), config)
        except RuntimeError:
            pose = None
    geometry_valid = bool(
        pose is not None
        and np.isfinite(pose.score)
        and pose.score >= config.minimum_geometry_score
    )
    if not geometry_valid or pose is None:
        return RegisteredTrenchGeometry(
            (),
            (),
            False,
            0.0 if pose is None else float(pose.score),
            pose,
        )

    trench_masks, safe_masks = _pose_trench_masks(pose, config)
    return RegisteredTrenchGeometry(
        trench_masks,
        safe_masks,
        True,
        float(pose.score),
        pose,
    )


def nearest_trench_index(
    trench_masks: Sequence[np.ndarray],
    center_yx: tuple[float, float] | np.ndarray,
) -> int | None:
    """Choose the containing trench, or the nearest interior across a wall.

    Use raw trench interiors so a center in the excluded tip cap still selects
    that trench. Circular radius and existing cell targets play no part.
    """

    center = _point(center_yx, "trench selection center")
    pixel = np.rint(center).astype(int)
    nearest = None
    minimum_distance = np.inf
    for index, mask in enumerate(trench_masks):
        if (
            0 <= pixel[0] < mask.shape[0]
            and 0 <= pixel[1] < mask.shape[1]
            and mask[tuple(pixel)]
        ):
            return index
        coordinates = np.argwhere(mask)
        if not len(coordinates):
            continue
        distance = float(np.min(np.sum((coordinates - center) ** 2, axis=1)))
        if distance < minimum_distance:
            nearest = index
            minimum_distance = distance
    return nearest


@lru_cache(maxsize=None)
def _reference_polygon_masks(config: TrenchTrackingConfig) -> tuple[np.ndarray, ...]:
    masks = tuple(
        _polygon_mask(config.reference_image_shape_yx, vertices)
        for vertices in reference_lattice_vertices(config)
    )
    for mask in masks:
        mask.setflags(write=False)
    return masks


def _nearest_allowed(
    point_yx: np.ndarray,
    allowed: np.ndarray,
    maximum_distance: float,
) -> tuple[np.ndarray, bool]:
    row, column = np.rint(point_yx).astype(int)
    if (
        0 <= row < allowed.shape[0]
        and 0 <= column < allowed.shape[1]
        and allowed[row, column]
    ):
        return np.asarray(point_yx, dtype=float), True
    coordinates = np.argwhere(allowed)
    if len(coordinates) == 0:
        return np.asarray(point_yx, dtype=float), False
    distances = np.linalg.norm(coordinates - point_yx, axis=1)
    nearest = int(np.argmin(distances))
    return (
        coordinates[nearest].astype(float),
        bool(distances[nearest] <= maximum_distance),
    )


def _feature(image: np.ndarray) -> np.ndarray:
    # Rejected updates must not progressively high-pass the stored template.
    work = np.array(image, dtype=np.float32, copy=True)
    work -= ndi.gaussian_filter(work, sigma=7.0)
    scale = float(np.percentile(np.abs(work), 99.0))
    return np.clip(work / scale, -1.0, 1.0) if scale > 0 else work


def _extract_template(
    image: np.ndarray,
    point_yx: np.ndarray,
    radius: int,
) -> np.ndarray | None:
    center = np.rint(point_yx).astype(int)
    y0, y1 = center[0] - radius, center[0] + radius + 1
    x0, x1 = center[1] - radius, center[1] + radius + 1
    if y0 < 0 or x0 < 0 or y1 > image.shape[0] or x1 > image.shape[1]:
        return None
    return np.asarray(image[y0:y1, x0:x1], dtype=np.float32).copy()


def _template_candidate(
    previous_template: np.ndarray,
    image: np.ndarray,
    predicted_yx: np.ndarray,
    config: TrenchTrackingConfig,
) -> tuple[np.ndarray, float, bool]:
    radius = config.template_radius_px
    search = int(np.ceil(config.maximum_step_px))
    center = np.rint(predicted_yx).astype(int)
    y0, y1 = center[0] - radius - search, center[0] + radius + search + 1
    x0, x1 = center[1] - radius - search, center[1] + radius + search + 1
    if y0 < 0 or x0 < 0 or y1 > image.shape[0] or x1 > image.shape[1]:
        return predicted_yx.copy(), 0.0, False
    response = match_template(
        _feature(image[y0:y1, x0:x1]),
        _feature(previous_template),
        pad_input=False,
    )
    if response.size == 0 or not np.any(np.isfinite(response)):
        return predicted_yx.copy(), 0.0, False
    peak = np.unravel_index(int(np.nanargmax(response)), response.shape)
    score = float(response[peak])
    candidate = center.astype(float) + np.asarray(peak, dtype=float) - search
    return candidate, score, bool(score >= config.minimum_template_score)


def _cell_watershed(
    image: np.ndarray,
    seed_yx: np.ndarray,
    allowed: np.ndarray,
    config: TrenchTrackingConfig,
) -> np.ndarray:
    seed, valid_seed = _nearest_allowed(
        seed_yx, allowed, config.maximum_seed_projection_px
    )
    if not valid_seed:
        return np.zeros(image.shape, dtype=bool)
    center = np.rint(seed).astype(int)
    crop_radius = config.maximum_cell_radius_px + 6
    y0, y1 = (
        max(0, center[0] - crop_radius),
        min(image.shape[0], center[0] + crop_radius + 1),
    )
    x0, x1 = (
        max(0, center[1] - crop_radius),
        min(image.shape[1], center[1] + crop_radius + 1),
    )
    crop = np.asarray(image[y0:y1, x0:x1], dtype=np.float32)
    allowed_crop = allowed[y0:y1, x0:x1]
    yy, xx = np.indices(crop.shape)
    local_y, local_x = float(seed[0] - y0), float(seed[1] - x0)
    radius = np.hypot(yy - local_y, xx - local_x)
    valid = allowed_crop & (radius <= config.maximum_cell_radius_px)
    if not np.any(valid):
        return np.zeros(image.shape, dtype=bool)
    gradient = sobel(ndi.gaussian_filter(crop, sigma=1.2))
    scale = float(np.percentile(gradient[valid], 99.0))
    if scale > 0:
        gradient = np.clip(gradient / scale, 0.0, 1.0)
    cost = gradient + 0.35 * (radius / config.maximum_cell_radius_px) ** 3
    markers = np.zeros(crop.shape, dtype=np.uint8)
    boundary = valid & ~binary_erosion(valid, footprint=disk(2))
    markers[boundary | ((radius >= config.maximum_cell_radius_px - 2) & valid)] = 2
    # Write the seed last, so a near-wall cell is not erased by the boundary.
    markers[(radius <= 3.5) & valid] = 1
    if not np.any(markers == 1) or not np.any(markers == 2):
        return np.zeros(image.shape, dtype=bool)
    segmented = watershed(cost, markers=markers, mask=valid) == 1
    segmented = binary_closing(segmented, footprint=disk(2)) & valid
    segmented = remove_small_holes(segmented, area_threshold=64) & valid
    components, count = ndi.label(segmented)
    if count == 0:
        return np.zeros(image.shape, dtype=bool)
    row = int(np.clip(round(local_y), 0, components.shape[0] - 1))
    column = int(np.clip(round(local_x), 0, components.shape[1] - 1))
    component = int(components[row, column])
    if component == 0:
        return np.zeros(image.shape, dtype=bool)
    result = np.zeros(image.shape, dtype=bool)
    result[y0:y1, x0:x1] = components == component
    return result


def _sample_mask(mask: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == output_shape:
        return np.asarray(mask, dtype=bool).copy()
    rows = np.floor(
        (np.arange(output_shape[0]) + 0.5) * mask.shape[0] / output_shape[0]
    ).astype(int)
    columns = np.floor(
        (np.arange(output_shape[1]) + 0.5) * mask.shape[1] / output_shape[1]
    ).astype(int)
    rows = np.clip(rows, 0, mask.shape[0] - 1)
    columns = np.clip(columns, 0, mask.shape[1] - 1)
    return np.asarray(mask[np.ix_(rows, columns)], dtype=bool)


def _point_in_mask(point: np.ndarray, mask: np.ndarray) -> bool:
    pixel = np.floor(point).astype(int)
    return bool(
        np.all(pixel >= 0)
        and np.all(pixel < mask.shape)
        and mask[tuple(pixel)]
    )


def _proposed_component(
    labels: np.ndarray,
    point: np.ndarray,
    allowed: np.ndarray,
) -> np.ndarray:
    """Resolve a proposal to one connected cell, without snapping to a neighbor.

    Label IDs may change at every frame. Coordinate conversion follows the
    upstream tracker's floor convention, including non-integer image scaling.
    Split components after clipping to the safe trench so an attached wall or
    disconnected island cannot connect the target to another cell.
    """

    pixel = np.floor(point * np.asarray(labels.shape) / allowed.shape).astype(int)
    empty = np.zeros(allowed.shape, dtype=bool)
    if np.any(pixel < 0) or np.any(pixel >= labels.shape):
        return empty
    value = int(labels[tuple(pixel)])
    if value == 0:
        return empty
    # Full-resolution pixels belong to their floor-indexed segmentation pixel.
    ys = np.floor(
        np.arange(allowed.shape[0]) * labels.shape[0] / allowed.shape[0]
    ).astype(int)
    xs = np.floor(
        np.arange(allowed.shape[1]) * labels.shape[1] / allowed.shape[1]
    ).astype(int)
    foreground = (labels[np.ix_(ys, xs)] == value) & allowed
    components, _ = ndi.label(foreground)
    if not _point_in_mask(point, foreground):
        return empty
    component = components[tuple(np.floor(point).astype(int))]
    return components == component


def _safe_labels(
    image_shape: tuple[int, int],
    safe_union: np.ndarray,
    accepted_masks: Sequence[np.ndarray],
    current_labels: np.ndarray | None,
    config: TrenchTrackingConfig,
) -> np.ndarray:
    base = None if current_labels is None else np.asarray(current_labels)
    if base is not None:
        if base.ndim != 2 or not np.issubdtype(base.dtype, np.integer):
            raise ValueError("current_labels must be a two-dimensional integer mask")
        if np.any(base < 0):
            raise ValueError("current_labels cannot contain negative values")
    output_shape = image_shape if base is None else base.shape
    labels = np.zeros(output_shape, dtype=np.uint32)
    next_label = 1
    for mask in accepted_masks:
        sampled = _sample_mask(mask, output_shape) & (labels == 0)
        if np.any(sampled):
            labels[sampled] = next_label
            next_label += 1
    if base is None:
        return labels

    allowed = _sample_mask(safe_union, output_shape)
    represented_base_labels = {
        int(value) for value in np.unique(base[labels != 0]) if value != 0
    }
    area_factor = (image_shape[0] / output_shape[0]) * (
        image_shape[1] / output_shape[1]
    )
    for raw_label in np.unique(base):
        if raw_label == 0 or int(raw_label) in represented_base_labels:
            continue
        components, count = ndi.label((base == raw_label) & allowed & (labels == 0))
        for index in range(1, count + 1):
            component = components == index
            area = float(np.count_nonzero(component) * area_factor)
            if config.minimum_cell_area_px <= area <= config.maximum_cell_area_px:
                labels[component] = next_label
                next_label += 1
    return labels


def filter_labels_to_trenches(
    image: np.ndarray,
    label_mask: np.ndarray,
    config: TrenchTrackingConfig = DEFAULT_TRENCH_CONFIG,
    *,
    pose_override: TrenchPose | None = None,
) -> TrenchSelectionResult:
    """Keep only cell-sized label components inside safe learned trenches.

    A single rotation and translation registers the complete fixed run-35
    lattice.  Every Cellpose label is intersected with the union of the
    registered trapezoids eroded by ``wall_clearance_px`` and with the learned
    closed-tip no-cell cap removed.  Connected components are then relabeled
    independently and retained only when their full-resolution area satisfies
    the configured cell bounds.

    Geometry failures are fail-closed: the returned label image and safe mask
    are empty, so no automated cell target can be selected from that frame.
    """

    frame = np.asarray(image)
    labels = np.asarray(label_mask)
    if frame.ndim != 2:
        raise ValueError("image must be two-dimensional")
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("label_mask must be a two-dimensional integer mask")
    if np.any(labels < 0):
        raise ValueError("label_mask cannot contain negative values")

    empty_labels = np.zeros(labels.shape, dtype=np.uint32)
    empty_safe = np.zeros(labels.shape, dtype=bool)
    if frame.shape != config.reference_image_shape_yx:
        return TrenchSelectionResult(
            empty_labels,
            empty_safe,
            False,
            0.0,
            None,
        )

    geometry = registered_trench_geometry(
        frame,
        config,
        pose_override=pose_override,
    )
    if not geometry.geometry_valid or geometry.pose is None:
        return TrenchSelectionResult(
            empty_labels,
            empty_safe,
            False,
            geometry.geometry_score,
            geometry.pose,
        )

    safe_union = geometry.safe_union
    filtered = _safe_labels(
        config.reference_image_shape_yx,
        safe_union,
        (),
        labels,
        config,
    )
    safe_at_label_scale = _sample_mask(safe_union, labels.shape)
    return TrenchSelectionResult(
        filtered,
        safe_at_label_scale,
        True,
        geometry.geometry_score,
        geometry.pose,
    )


class TrenchConstraintTracker:
    """Apply fixed-trench segmentation and motion gates to live targets."""

    def __init__(
        self,
        config: TrenchTrackingConfig = DEFAULT_TRENCH_CONFIG,
    ) -> None:
        self.config = config
        self._positions: dict[int, _PositionState] = {}

    def reset(self) -> None:
        """Forget pose and target state at the start of a new acquisition."""

        self._positions.clear()

    def _pose(
        self,
        position: int,
        image: np.ndarray,
        override: TrenchPose | None,
    ) -> tuple[_PositionState, TrenchPose | None, bool]:
        state = self._positions.setdefault(int(position), _PositionState())
        response = _wall_response(image)
        pose = override
        if pose is None and state.pose is not None and state.response is not None:
            pose = _incremental_pose(response, state.response, state.pose, self.config)
        if pose is None:
            try:
                pose = _global_pose(response, self.config)
            except RuntimeError:
                pose = None
        valid = bool(
            pose is not None
            and np.isfinite(pose.score)
            and pose.score >= self.config.minimum_geometry_score
        )
        if valid:
            state.pose = pose
            state.response = response
        return state, pose, valid

    @staticmethod
    def _held_result(
        target: TrenchTarget,
        point: np.ndarray,
        reason: str,
        trench_index: int | None,
        template_score: float = 0.0,
    ) -> TrenchTargetResult:
        # A rejected update must not move the microscope target, including
        # when the proposed wall-registration correction is itself uncertain.
        del point
        return TrenchTargetResult(
            token=target.token,
            point_yx=np.asarray(target.previous_yx, dtype=float).copy(),
            accepted=False,
            reason=reason,
            trench_index=trench_index,
            step_px=0.0,
            area_px=0.0,
            template_score=template_score,
        )

    def _appearance_update(
        self, position_state, target, target_state, previous, predicted,
        frame, allowed, trench_index, pose, base_labels,
    ):
        """Identify the cell from BF, then aim at its Cellpose centroid."""
        score = 1.0
        seed = previous.copy()
        if target_state is not None:
            match = match_cell(
                target_state.reference, frame, predicted,
                self.config.maximum_step_px,
            )
            score = match.score
            if not match.accepted:
                return self._held_result(
                    target, previous, "cell_appearance", trench_index, score,
                ), None
            seed = match.point_yx
        if not _point_in_mask(seed, allowed):
            return self._held_result(
                target, previous, "same_trench_interior", trench_index, score,
            ), None
        reference = cell_reference(frame, seed, self.config.maximum_step_px)
        if reference is None:
            return self._held_result(
                target, previous, "template_bounds", trench_index, score,
            ), None
        if target_state is None:
            # The selected first-frame anchor is trusted even if Cellpose
            # misses it. Preserve its appearance so later frames can recover.
            target_state = _AppearanceState(
                trench_index, pose.inverse(seed, frame.shape), reference,
                pose.inverse(previous, frame.shape),
            )
            position_state.appearance_targets[target.token] = target_state

        # Cellpose already supplies the cell boundary. Re-segmenting it around
        # the aim can cut a mother cell into an artificial seed-centered lobe.
        # Keep its connected outline, clipped only to the safe trench. A
        # missing mask must hold the previous aim, never invent a replacement
        # mask or snap to a neighboring cell.
        mask = _proposed_component(base_labels, seed, allowed)
        if not np.any(mask):
            return self._held_result(
                target, previous, "missing_cellpose_mask", trench_index, score,
            ), None
        # Center the exact mask that will be emitted at Cellpose's resolution,
        # so scaling and clipping cannot leave the displayed aim off-center.
        # Pixel centers transform with the half-pixel offset, including for
        # non-integer rescaling. Do not fill holes or reshape the outline.
        sampled = _sample_mask(mask, base_labels.shape)
        if not np.any(sampled):
            return self._held_result(
                target, previous, "missing_cellpose_mask", trench_index, score,
            ), None
        center = (
            (np.asarray(ndi.center_of_mass(sampled)) + 0.5)
            * np.asarray(frame.shape) / np.asarray(base_labels.shape) - 0.5
        )
        predicted_aim = pose.apply(target_state.canonical_aim_yx, frame.shape)
        step = float(np.linalg.norm(center - predicted_aim))
        area = float(np.count_nonzero(mask))
        prior_area = target_state.previous_area_px
        ratio = area / prior_area if prior_area else 1.0
        gates = {
            "same_cell": _point_in_mask(seed, mask),
            # A concave mask's centroid can lie in a hole. Hold rather than
            # projecting to an arbitrary boundary or switching cell labels.
            "centroid_in_cell": _point_in_mask(center, mask)
            and _point_in_mask(
                center * np.asarray(base_labels.shape) / frame.shape, sampled,
            ),
            "same_trench_interior": _point_in_mask(center, allowed),
            "cell_area": self.config.minimum_cell_area_px <= area
            <= self.config.maximum_cell_area_px,
            "area_continuity": self.config.minimum_area_ratio <= ratio
            <= self.config.maximum_area_ratio,
            "maximum_step": step <= self.config.maximum_step_px,
        }
        failed = [name for name, passed in gates.items() if not passed]
        if failed:
            return self._held_result(
                target, previous, "+".join(failed), trench_index, score,
            ), None
        # Keep the reference on the matched image feature, not on the moving
        # mask centroid. Commit both histories only after all final-aim checks.
        position_state.appearance_targets[target.token] = _AppearanceState(
            trench_index, pose.inverse(seed, frame.shape), reference,
            pose.inverse(center, frame.shape), area,
        )
        return TrenchTargetResult(
            target.token, center, True, "accepted", trench_index, step, area, score,
        ), mask

    def update(
        self,
        position: int,
        time_index: int,
        image: np.ndarray,
        targets: Sequence[TrenchTarget],
        current_labels: np.ndarray | None = None,
        label_scale: float = 1.0,
        *,
        pose_override: TrenchPose | None = None,
        trench_center_yx: tuple[float, float] | np.ndarray | None = None,
    ) -> TrenchUpdate:
        """Return safe point updates and a filtered current segmentation.

        ``label_scale`` is validated for compatibility with the engine API;
        exact full-resolution area conversion uses the observed image and mask
        shapes, so non-integer rescaling remains safe.

        When ``trench_center_yx`` is provided, only the trench nearest that
        point can contribute segmentation or point updates. Targets in other
        trenches are held without running cell segmentation around them.

        With current labels, brightfield appearance establishes identity using
        a reciprocal local match. The connected Cellpose outline is retained
        without watershed refinement; its centroid becomes the aim only if
        it is inside the same cell and safe trench and within the movement
        limit. Identity continues following its independent image feature.
        Rejected updates hold the exact previous aim and trusted reference.
        """

        frame = np.asarray(image)
        if frame.ndim != 2:
            raise ValueError("image must be two-dimensional")
        if not np.isscalar(label_scale) or float(label_scale) <= 0:
            raise ValueError("label_scale must be positive")
        target_list = list(targets)
        previous_points = [
            _point(target.previous_yx, "previous_yx") for target in target_list
        ]
        for target in target_list:
            if target.proposed_yx is not None:
                _point(target.proposed_yx, "proposed_yx")

        expected_shape = self.config.reference_image_shape_yx
        base_labels = None if current_labels is None else np.asarray(current_labels)
        if base_labels is not None and (
            base_labels.ndim != 2
            or not np.issubdtype(base_labels.dtype, np.integer)
            or np.any(base_labels < 0)
        ):
            raise ValueError(
                "current_labels must be a two-dimensional nonnegative integer mask"
            )
        output_shape = (
            frame.shape if current_labels is None else np.asarray(current_labels).shape
        )
        if frame.shape != expected_shape:
            results = tuple(
                self._held_result(target, point, "geometry_shape", None)
                for target, point in zip(target_list, previous_points, strict=True)
            )
            return TrenchUpdate(
                np.asarray(previous_points, dtype=float).reshape(-1, 2),
                np.zeros(output_shape, dtype=np.uint32),
                results,
                False,
                0.0,
                None,
            )

        position_state, pose, geometry_valid = self._pose(
            position, frame, pose_override
        )
        use_appearance = self.config.track_cell_appearance and base_labels is not None
        target_states = (
            position_state.appearance_targets if use_appearance
            else position_state.targets
        )
        if not geometry_valid or pose is None:
            results = []
            for target, previous in zip(target_list, previous_points, strict=True):
                target_state = target_states.get(target.token)
                if target_state is None or position_state.pose is None:
                    held = previous
                    trench_index = None
                else:
                    held = position_state.pose.apply(
                        target_state.canonical_point_yx, expected_shape
                    )
                    trench_index = target_state.trench_index
                results.append(
                    self._held_result(
                        target,
                        np.asarray(held),
                        "geometry_confidence",
                        trench_index,
                    )
                )
            return TrenchUpdate(
                np.asarray([result.point_yx for result in results]).reshape(-1, 2),
                np.zeros(output_shape, dtype=np.uint32),
                tuple(results),
                False,
                0.0 if pose is None else float(pose.score),
                pose,
            )

        trench_masks, safe_masks = _pose_trench_masks(pose, self.config)
        selected_trench = (
            None
            if trench_center_yx is None
            else nearest_trench_index(trench_masks, trench_center_yx)
        )
        if trench_center_yx is None:
            safe_union = np.logical_or.reduce(safe_masks)
        elif selected_trench is None:
            safe_union = np.zeros(expected_shape, dtype=bool)
        else:
            safe_union = safe_masks[selected_trench]
        reference_masks = _reference_polygon_masks(self.config)

        results: list[TrenchTargetResult] = []
        accepted_masks: list[np.ndarray] = []
        for target, previous in zip(target_list, previous_points, strict=True):
            target_state = target_states.get(target.token)
            if target_state is None:
                canonical = pose.inverse(previous, expected_shape)
                pixel = np.rint(canonical).astype(int)
                trench_index = next(
                    (
                        index
                        for index, mask in enumerate(reference_masks)
                        if 0 <= pixel[0] < expected_shape[0]
                        and 0 <= pixel[1] < expected_shape[1]
                        and mask[tuple(pixel)]
                    ),
                    None,
                )
                predicted = previous.copy()
            else:
                trench_index = target_state.trench_index
                predicted = np.asarray(
                    pose.apply(target_state.canonical_point_yx, expected_shape)
                )

            if trench_index is None:
                results.append(
                    self._held_result(target, predicted, "outside_trench", None)
                )
                continue

            if trench_center_yx is not None and trench_index != selected_trench:
                results.append(
                    self._held_result(
                        target, previous, "outside_selected_trench", trench_index
                    )
                )
                continue

            allowed = safe_masks[trench_index]
            if use_appearance:
                result, mask = self._appearance_update(
                    position_state, target, target_state, previous, predicted,
                    frame, allowed, trench_index, pose, base_labels,
                )
                results.append(result)
                if mask is not None:
                    accepted_masks.append(mask)
                continue
            matched_cell = None
            prior_pixels = None
            visible_pixels = None
            if base_labels is not None:
                # The particle tracker already chose a cell. Refine its
                # interior, rather than starting a second unrestricted tracker.
                anchor = (
                    predicted
                    if target.proposed_yx is None
                    else np.asarray(target.proposed_yx, dtype=float)
                )
                # At initialization the user-selected cell is authoritative.
                if target_state is None:
                    anchor = previous
                matched_cell = _proposed_component(base_labels, anchor, allowed)
                if not np.any(matched_cell):
                    results.append(
                        self._held_result(
                            target, predicted, "missing_proposed_cell", trench_index,
                        )
                    )
                    continue
                if target_state is not None:
                    # Compare geometry, not numeric label IDs. Keep the last
                    # accepted footprint across missed/ambiguous frames.
                    prior_pixels = np.rint(pose.apply(
                        target_state.canonical_cell_yx, expected_shape,
                    )).astype(int)
                    in_frame = np.all(
                        (prior_pixels >= 0) & (prior_pixels < expected_shape), axis=1,
                    )
                    visible_pixels = prior_pixels[in_frame]
                    overlap = np.count_nonzero(matched_cell[
                        visible_pixels[:, 0], visible_pixels[:, 1],
                    ]) / len(prior_pixels)
                    # If the old cell is still segmented at its registered
                    # location, an assignment to another cell is ambiguous.
                    prior_cell = _proposed_component(base_labels, predicted, allowed)
                    prior_radius = np.sqrt(target_state.previous_area_px / np.pi)
                    identity_conflict = (
                        overlap < self.config.minimum_identity_overlap
                        or (
                            np.any(prior_cell)
                            and (
                                not _point_in_mask(predicted, matched_cell)
                                or np.linalg.norm(anchor - predicted) > prior_radius
                            )
                        )
                    )
                    if identity_conflict:
                        results.append(
                            self._held_result(
                                target, predicted, "cell_identity", trench_index,
                            )
                        )
                        continue
                seed, template_valid = anchor.copy(), True
                # No independent template search in this path.
                template_score = 0.0
            elif target_state is None:
                seed, template_valid = _nearest_allowed(
                    predicted, allowed, self.config.maximum_seed_projection_px
                )
                template_score = 1.0 if template_valid else 0.0
            else:
                seed, template_score, template_valid = _template_candidate(
                    target_state.previous_template,
                    frame,
                    predicted,
                    self.config,
                )
            if not template_valid:
                results.append(
                    self._held_result(
                        target,
                        predicted,
                        "template_confidence",
                        trench_index,
                        template_score,
                    )
                )
                continue

            cell_mask = _cell_watershed(
                frame,
                seed,
                allowed if matched_cell is None else matched_cell,
                self.config,
            )
            if not np.any(cell_mask):
                results.append(
                    self._held_result(
                        target,
                        predicted,
                        "empty_cell_mask",
                        trench_index,
                        template_score,
                    )
                )
                continue

            center = np.asarray(ndi.center_of_mass(cell_mask), dtype=float)
            area = float(np.count_nonzero(cell_mask))
            step = float(np.linalg.norm(center - predicted))
            row, column = np.rint(center).astype(int)
            center_inside = bool(
                0 <= row < frame.shape[0]
                and 0 <= column < frame.shape[1]
                and allowed[row, column]
            )
            clearance = (
                float(
                    ndi.distance_transform_edt(trench_masks[trench_index])[row, column]
                )
                if center_inside
                else 0.0
            )
            previous_area = (
                None if target_state is None else target_state.previous_area_px
            )
            ratio = 1.0 if previous_area is None else area / previous_area
            gates = {
                "maximum_step": step <= self.config.maximum_step_px,
                "same_trench_interior": center_inside,
                "between_walls": clearance >= self.config.wall_clearance_px,
                "cell_area": (
                    self.config.minimum_cell_area_px
                    <= area
                    <= self.config.maximum_cell_area_px
                ),
                "area_continuity": (
                    previous_area is None
                    or self.config.minimum_area_ratio
                    <= ratio
                    <= self.config.maximum_area_ratio
                ),
            }
            if matched_cell is not None:
                # A merged/concave raw mask must not let refinement move the
                # aim into a different lobe or put its COM outside the cell.
                refinement_radius = np.sqrt(area / np.pi)
                gates["same_cell"] = bool(
                    _point_in_mask(seed, cell_mask)
                    and _point_in_mask(center, cell_mask)
                    and _point_in_mask(center, matched_cell)
                    and np.linalg.norm(center - seed) <= refinement_radius
                    and not np.any(cell_mask & ~matched_cell)
                )
                if prior_pixels is not None and visible_pixels is not None:
                    # A merged Cellpose object may overlap the old cell even
                    # when its refined sub-region belongs to a neighbor.
                    retained = np.count_nonzero(cell_mask[
                        visible_pixels[:, 0], visible_pixels[:, 1],
                    ]) / len(prior_pixels)
                    gates["cell_identity"] = (
                        retained >= self.config.minimum_identity_overlap
                    )
            failed = [name for name, passed in gates.items() if not passed]
            accepted = not failed
            if accepted:
                # Preserve the trusted user-selected anchor at t0.  Later
                # frames use the center of the constrained segmentation.
                point = (
                    previous.copy()
                    if target_state is None and int(time_index) == 0
                    else center
                )
                template = _extract_template(
                    frame, point, self.config.template_radius_px
                )
                if template is None:
                    accepted = False
                    failed = ["template_bounds"]
                else:
                    position_state.targets[target.token] = _TargetState(
                        trench_index,
                        np.asarray(pose.inverse(point, expected_shape)),
                        template,
                        area,
                        np.asarray(pose.inverse(np.argwhere(cell_mask), expected_shape)),
                    )
                    accepted_masks.append(cell_mask)
            if not accepted:
                point = previous.copy()
            results.append(
                TrenchTargetResult(
                    target.token,
                    np.asarray(point, dtype=float).copy(),
                    accepted,
                    "accepted" if accepted else "+".join(failed),
                    trench_index,
                    step if accepted else 0.0,
                    area if accepted else 0.0,
                    template_score,
                )
            )

        labels = _safe_labels(
            expected_shape,
            safe_union,
            accepted_masks,
            current_labels,
            self.config,
        )
        return TrenchUpdate(
            np.asarray([result.point_yx for result in results], dtype=float).reshape(
                -1, 2
            ),
            labels,
            tuple(results),
            True,
            float(pose.score),
            pose,
        )


# A descriptive alias retained for callers that used the prototype name.
ConstrainedTrenchTracker = TrenchConstraintTracker
