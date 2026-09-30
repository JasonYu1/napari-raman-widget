"""Deterministic, bounded sampling inside image-space regions of interest.

Both modes clip a square lattice in image pixel coordinates. Count mode
chooses a nearby pitch to approximate the requested total; it never removes,
adds, or jitters points to force an exact count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MAX_ROI_POINTS = 250_000
_MAX_VERTICES = 2048
_MAX_SCANLINES = 1_000_000
_MAX_SCANLINE_EDGE_TESTS = 8_000_000
_MAX_TARGET_EVALUATIONS = 24
_MAX_TARGET_SCANLINES = 30_000
_MAX_TARGET_EDGE_TESTS = 2_000_000
_EPS = 64 * np.finfo(float).eps


@dataclass(frozen=True)
class _Region:
    kind: str
    vertices: np.ndarray
    origin: np.ndarray
    scale: float
    center: np.ndarray | None = None
    axes: np.ndarray | None = None

    def bounds(self):
        if self.kind == "ellipse":
            radius = np.sqrt(np.sum(self.axes**2, axis=1))
            return self.center - radius, self.center + radius
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def world(self, points):
        return self.origin + np.asarray(points) * self.scale


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _signed_area(vertices):
    terms = _cross(vertices, np.roll(vertices, -1, axis=0))
    return math.fsum(float(term) for term in terms) / 2


def _has_self_intersection(vertices):
    """Reject crossings and non-adjacent touches, including repeated corners."""
    count = len(vertices)
    for index in range(count - 2):
        others = np.arange(index + 2, count)
        if index == 0:
            others = others[others != count - 1]
        if not len(others):
            continue
        p = vertices[index]
        p_end = vertices[index + 1]
        q = vertices[others]
        q_end = vertices[(others + 1) % count]
        first = _cross(p_end - p, q - p)
        second = _cross(p_end - p, q_end - p)
        third = _cross(q_end - q, p - q)
        fourth = _cross(q_end - q, p_end - q)
        crosses = (
            (first > _EPS) & (second < -_EPS) | (first < -_EPS) & (second > _EPS)
        ) & ((third > _EPS) & (fourth < -_EPS) | (third < -_EPS) & (fourth > _EPS))
        touches = (
            (np.abs(first) <= _EPS)
            & np.all(
                (q >= np.minimum(p, p_end) - _EPS) & (q <= np.maximum(p, p_end) + _EPS),
                axis=1,
            )
            | (np.abs(second) <= _EPS)
            & np.all(
                (q_end >= np.minimum(p, p_end) - _EPS)
                & (q_end <= np.maximum(p, p_end) + _EPS),
                axis=1,
            )
            | (np.abs(third) <= _EPS)
            & np.all(
                (p >= np.minimum(q, q_end) - _EPS) & (p <= np.maximum(q, q_end) + _EPS),
                axis=1,
            )
            | (np.abs(fourth) <= _EPS)
            & np.all(
                (p_end >= np.minimum(q, q_end) - _EPS)
                & (p_end <= np.maximum(q, q_end) + _EPS),
                axis=1,
            )
        )
        if np.any(crosses | touches):
            return True
    return False


def _remove_collinear(vertices):
    """Remove only forward collinear corners; backtracking is not a valid ROI."""
    while len(vertices) > 3:
        incoming = vertices - np.roll(vertices, 1, axis=0)
        outgoing = np.roll(vertices, -1, axis=0) - vertices
        threshold = (
            _EPS * np.linalg.norm(incoming, axis=1) * np.linalg.norm(outgoing, axis=1)
        )
        collinear = np.abs(_cross(incoming, outgoing)) <= threshold
        if np.any(collinear & (np.sum(incoming * outgoing, axis=1) <= 0)):
            raise ValueError(
                "The ROI doubles back along an edge; draw a simple closed shape."
            )
        if not collinear.any():
            return vertices
        vertices = vertices[~collinear]
    return vertices


def _region(vertices_yx, shape_type):
    kind = str(getattr(shape_type, "value", shape_type)).strip().lower()
    if kind not in {"rectangle", "polygon", "ellipse"}:
        raise ValueError(
            "Choose a filled rectangle, polygon, or ellipse ROI; lines and paths cannot be scanned."
        )
    try:
        vertices = np.array(vertices_yx, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "ROI vertices must be finite image (row, column) coordinates."
        ) from error
    if vertices.ndim != 2 or vertices.shape[1] != 2 or not np.isfinite(vertices).all():
        raise ValueError("ROI vertices must be finite image (row, column) coordinates.")
    if len(vertices) > 1 and np.array_equal(vertices[0], vertices[-1]):
        vertices = vertices[:-1]
    if kind != "polygon" and len(vertices) == 2:
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        vertices = np.array([low, [low[0], high[1]], high, [high[0], low[1]]])
    if (kind != "polygon" and len(vertices) != 4) or len(vertices) < 3:
        raise ValueError(
            "A polygon needs at least three corners; rectangles and ellipses need four ordered corners."
        )
    if len(vertices) > _MAX_VERTICES:
        raise ValueError(
            f"The ROI is too complex; use at most {_MAX_VERTICES} vertices."
        )
    origin = vertices.min(axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        extent = vertices.max(axis=0) - origin
    scale = float(extent.max())
    if not np.isfinite(scale) or scale <= 0 or np.any(extent <= 0):
        raise ValueError("The ROI is degenerate; draw a region with nonzero area.")
    vertices = (vertices - origin) / scale
    if np.any(np.linalg.norm(vertices - np.roll(vertices, 1, axis=0), axis=1) <= _EPS):
        raise ValueError(
            "The ROI has coincident corners or an edge too small to resolve."
        )
    if _has_self_intersection(vertices):
        raise ValueError(
            "The ROI intersects or touches itself; draw a simple closed shape."
        )
    if abs(_signed_area(vertices)) <= _EPS:
        raise ValueError("The ROI is degenerate or too thin to resolve reliably.")
    if kind in {"rectangle", "ellipse"}:
        # Napari corners can be rotated/sheared by a layer transform. A full
        # rank parallelogram is valid; perpendicular edges are not required.
        if not np.allclose(
            vertices[0] + vertices[2], vertices[1] + vertices[3], rtol=0, atol=1e-6
        ):
            raise ValueError(
                "Rectangle and ellipse corners must form an ordered parallelogram."
            )
        if kind == "ellipse":
            axes = np.column_stack(
                ((vertices[1] - vertices[0]) / 2, (vertices[2] - vertices[1]) / 2)
            )
            return _Region(kind, vertices, origin, scale, vertices.mean(axis=0), axes)
    vertices = _remove_collinear(vertices)
    if len(vertices) < 3:
        raise ValueError("The ROI is degenerate; draw a region with nonzero area.")
    return _Region(kind, vertices, origin, scale)


def roi_outline(vertices_yx, shape_type, *, samples=128):
    """Return a closed plotting outline of the actual ROI in image (y, x)."""
    region = _region(vertices_yx, shape_type)
    if region.kind == "ellipse":
        count = _integer(samples, "Outline samples", minimum=8, maximum=4096)
        theta = np.linspace(0, 2 * np.pi, count, endpoint=False)
        points = (
            region.center
            + np.column_stack((np.cos(theta), np.sin(theta))) @ region.axes.T
        )
    else:
        points = region.vertices
    return region.world(np.vstack((points, points[0])))


def _integer(value, label, *, minimum=2, maximum=MAX_ROI_POINTS):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{label} must be an integer from {minimum} to {maximum:,}."
        ) from error
    if (
        isinstance(value, (bool, np.bool_))
        or not np.isfinite(number)
        or not number.is_integer()
        or not minimum <= number <= maximum
    ):
        raise ValueError(f"{label} must be an integer from {minimum} to {maximum:,}.")
    return int(number)


def _polygon_intervals(vertices, y):
    first = vertices
    second = np.roll(vertices, -1, axis=0)
    crosses = ((first[:, 0] <= y) & (y < second[:, 0])) | (
        (second[:, 0] <= y) & (y < first[:, 0])
    )
    edge_a, edge_b = first[crosses], second[crosses]
    xs = np.sort(
        edge_a[:, 1]
        + (y - edge_a[:, 0])
        * (edge_b[:, 1] - edge_a[:, 1])
        / (edge_b[:, 0] - edge_a[:, 0])
    )
    intervals = list(zip(xs[::2], xs[1::2]))
    # Horizontal edges and tangent vertices are part of the ROI boundary too.
    horizontal = (np.abs(first[:, 0] - y) <= _EPS) & (np.abs(second[:, 0] - y) <= _EPS)
    intervals.extend(
        (min(a[1], b[1]), max(a[1], b[1]))
        for a, b in zip(first[horizontal], second[horizontal])
    )
    intervals.extend((x, x) for x in vertices[np.abs(vertices[:, 0] - y) <= _EPS, 1])
    merged = []
    for low, high in sorted(intervals):
        if merged and low <= merged[-1][1] + _EPS:
            merged[-1] = (merged[-1][0], max(high, merged[-1][1]))
        else:
            merged.append((low, high))
    return merged


def _ellipse_intervals(region, y):
    a, b = region.axes[:, 0], region.axes[:, 1]
    y_radius = math.hypot(a[0], b[0])
    dy = y - region.center[0]
    remaining = 1 - (dy / y_radius) ** 2
    if remaining < -_EPS:
        return []
    center_x = region.center[1] + (a[0] * a[1] + b[0] * b[1]) / (y_radius**2) * dy
    half_width = abs(_cross(a, b)) / y_radius * math.sqrt(max(0, remaining))
    return [(center_x - half_width, center_x + half_width)]


def _index_tolerance(value):
    return _EPS * max(1.0, abs(value))


def _lattice(region, spacing, *, collect=True, budget=None):
    """Count without allocating points, or generate the same complete lattice."""
    low, high = region.bounds()
    step = spacing / region.scale
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratios = (high - low) / step
    if not np.isfinite(ratios).all() or np.any(ratios > 2**52):
        raise ValueError(
            "Spacing is too small relative to the ROI size; increase pixel spacing."
        )
    rows = math.floor(ratios[0] + _index_tolerance(ratios[0])) + 1
    columns = math.floor(ratios[1] + _index_tolerance(ratios[1])) + 1
    axis_box = (
        region.kind != "ellipse"
        and len(region.vertices) == 4
        and np.all(
            (np.abs(region.vertices - low) <= _EPS)
            | (np.abs(region.vertices - high) <= _EPS)
        )
    )
    if axis_box:
        total = rows * columns
        if not collect:
            return min(total, MAX_ROI_POINTS + 1)
        if total > MAX_ROI_POINTS:
            raise ValueError(
                f"The ROI contains more than {MAX_ROI_POINTS:,} sampling points; increase pixel spacing."
            )
        if total < 2:
            raise ValueError(
                "The selected ROI and spacing contain fewer than two sample points; reduce pixel spacing or enlarge the ROI."
            )
        ys = low[0] + np.arange(rows) * step
        xs = low[1] + np.arange(columns) * step
        return np.column_stack((np.tile(ys, columns), np.repeat(xs, rows)))
    if rows > _MAX_SCANLINES or (
        region.kind != "ellipse"
        and rows * len(region.vertices) > _MAX_SCANLINE_EDGE_TESTS
    ):
        raise ValueError(
            "This ROI and spacing require too many scan lines to evaluate; increase spacing or simplify the ROI."
        )
    if budget is not None:
        work = rows * (1 if region.kind == "ellipse" else len(region.vertices))
        if rows > budget[0] or work > budget[1]:
            raise ValueError("Target-count spacing search reached its safe work limit.")
        budget[0] -= rows
        budget[1] -= work
    chunks = []
    total = 0
    for row in range(rows):
        y = low[0] + row * step
        intervals = (
            _ellipse_intervals(region, y)
            if region.kind == "ellipse"
            else _polygon_intervals(region.vertices, y)
        )
        for start, stop in intervals:
            first_value = (start - low[1]) / step
            last_value = (stop - low[1]) / step
            first = max(0, math.ceil(first_value - _index_tolerance(first_value)))
            last = math.floor(last_value + _index_tolerance(last_value))
            size = last - first + 1
            if size <= 0:
                continue
            total += size
            if total > MAX_ROI_POINTS:
                if not collect:
                    return MAX_ROI_POINTS + 1
                raise ValueError(
                    f"The ROI contains more than {MAX_ROI_POINTS:,} sampling points; increase pixel spacing."
                )
            # Evaluate each lattice column from its global integer index.
            # Different scanline interval starts must not round a shared X
            # column differently and scramble X-major acquisition ordering.
            if collect:
                xs = low[1] + (first + np.arange(size)) * step
                chunks.append(np.column_stack((np.full(size, y), xs)))
    if not collect:
        return total
    if total < 2:
        raise ValueError(
            "The selected ROI and spacing contain fewer than two sample points; reduce pixel spacing or enlarge the ROI."
        )
    return np.concatenate(chunks)


def _target_spacing(region, target, *, area=None, lattice_counter=None):
    """Choose the nearest valid evaluated count, not a claimed global optimum."""
    low, high = region.bounds()
    height, width = high - low
    if area is None:
        area = (
            np.pi * abs(np.linalg.det(region.axes))
            if region.kind == "ellipse"
            else abs(_signed_area(region.vertices))
        )
    counter = _lattice if lattice_counter is None else lattice_counter
    area_pitch = math.sqrt(area / target) * region.scale
    # The boundary term makes a square target400 start at20×20, not21×21.
    boundary = height + width
    corrected = (
        (boundary + math.sqrt(boundary**2 + 4 * (target - 1) * area))
        / (2 * (target - 1))
        * region.scale
    )
    budget = [_MAX_TARGET_SCANLINES, _MAX_TARGET_EDGE_TESTS]
    evaluated = {}
    attempts = set()

    def evaluate(pitch):
        pitch = float(pitch)
        if (
            pitch in attempts
            or len(attempts) >= _MAX_TARGET_EVALUATIONS
            or not np.isfinite(pitch)
            or pitch <= 0
        ):
            return
        attempts.add(pitch)
        try:
            evaluated[pitch] = counter(region, pitch, collect=False, budget=budget)
        except ValueError:
            # An impractically dense candidate does not allocate a large mesh.
            return

    def best():
        valid = [
            (pitch, count)
            for pitch, count in evaluated.items()
            if 2 <= count <= MAX_ROI_POINTS
        ]
        return (
            min(
                valid,
                key=lambda item: (abs(item[1] - target), item[1] > target, item[0]),
            )
            if valid
            else None
        )

    for pitch in (
        corrected,
        area_pitch,
        max(height, width) / (target - 1) * region.scale,
    ):
        evaluate(pitch)
        if best() is not None and best()[1] == target:
            return best()[0]
    for factor in (0.75, 1.5, 0.5, 2.0, 0.25, 4.0):
        if any(count >= target for count in evaluated.values()) and any(
            count < target for count in evaluated.values()
        ):
            break
        evaluate(area_pitch * factor)
    while len(attempts) < _MAX_TARGET_EVALUATIONS:
        if best() is not None and best()[1] == target:
            break
        ordered = sorted(evaluated.items())
        brackets = [
            (a[0], b[0])
            for a, b in zip(ordered, ordered[1:])
            if (a[1] >= target) != (b[1] >= target)
        ]
        if not brackets:
            break
        lower, upper = min(brackets, key=lambda pair: pair[1] / pair[0])
        midpoint = math.sqrt(lower) * math.sqrt(upper)
        if midpoint in attempts or upper / lower - 1 < 1e-8:
            break
        before = len(evaluated)
        evaluate(midpoint)
        if len(evaluated) == before:
            break
    selected = best()
    if selected is None:
        raise ValueError(
            "Cannot find a safe equally spaced lattice for this ROI and target. Enlarge/simplify the ROI or choose an explicit pixel spacing."
        )
    return selected[0]


def sample_roi_plan(
    vertices_yx, shape_type, *, mode="count", total_points=400, spacing_px=10.0
):
    """Return ``(points_yx, effective_spacing_px)`` for a complete clipped grid.

    ``count`` treats ``total_points`` as an approximate target and chooses a
    nearby pitch using a bounded search. ``spacing`` uses the supplied pitch
    unchanged. Both return every point of one image-axis-aligned square lattice
    anchored at the actual ROI's minimum Y/X bounds and clipped to its boundary.
    There is no jitter, dropping, padding, or promise of a globally closest count.
    The actual output always contains 2 to 250,000 distinct acquired points.

    Rectangles/ellipses accept four ordered parallelogram corners, including
    rotation and shear, or two axis-aligned opposite corners. A polygon must
    be simple and nondegenerate; an optional repeated closing corner is fine.
    """
    region = _region(vertices_yx, shape_type)
    if mode == "count":
        spacing = _target_spacing(region, _integer(total_points, "Target total points"))
    elif mode == "spacing":
        try:
            spacing = float(spacing_px)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "Pixel spacing must be finite and greater than zero."
            ) from error
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("Pixel spacing must be finite and greater than zero.")
    else:
        raise ValueError("Sampling mode must be 'count' or 'spacing'.")
    points = region.world(_lattice(region, spacing))
    if not np.isfinite(points).all() or len(np.unique(points, axis=0)) != len(points):
        raise ValueError(
            "Some sample points cannot be distinguished at this coordinate precision; increase spacing or reduce the total count."
        )
    # Match the established X-major/Y-minor acquisition order deterministically.
    return points[np.lexsort((points[:, 0], points[:, 1]))], spacing


def sample_roi(
    vertices_yx, shape_type, *, mode="count", total_points=400, spacing_px=10.0
):
    """Return the complete equally spaced ROI grid; count mode is approximate."""
    return sample_roi_plan(
        vertices_yx,
        shape_type,
        mode=mode,
        total_points=total_points,
        spacing_px=spacing_px,
    )[0]
