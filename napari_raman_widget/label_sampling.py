"""Uniform camera-pixel lattices clipped to all non-zero label pixels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .roi_sampling import MAX_ROI_POINTS, _integer, _target_spacing


MAX_LABEL_PIXELS = 16_000_000
_MAX_CANDIDATES = 8_000_000
_CHUNK_SIZE = 65_536


@dataclass(frozen=True)
class LabelROI:
    """An immutable copy of a 2D label raster aligned with camera pixels."""

    data_bytes: bytes
    shape: tuple[int, int]
    dtype: str
    limits_yx: tuple[tuple[int, int], tuple[int, int]]
    label_values: tuple[int, ...]
    area: int
    image_layer_name: str
    image_to_world_yx: tuple[tuple[float, ...], ...]
    scale: float = 1.0

    @property
    def labels(self):
        # A bytes-backed view cannot be made writable, unlike a frozen ndarray.
        return np.frombuffer(self.data_bytes, dtype=self.dtype).reshape(self.shape)

    def bounds(self):
        return tuple(np.asarray(point, dtype=float) for point in self.limits_yx)

    def labels_at(self, points):
        """Nearest pixel, with half-pixel ties assigned to the next pixel."""
        indices = np.floor(np.asarray(points) + 0.5).astype(np.int64)
        inside = np.all((indices >= 0) & (indices < self.shape), axis=1)
        result = np.zeros(len(indices), dtype=self.dtype)
        result[inside] = self.labels[indices[inside, 0], indices[inside, 1]]
        return result


def make_label_roi(data, *, image_layer_name="Camera pixels", image_to_world_yx=None):
    """Copy integer labels; zero is background, all other IDs are included."""
    shape = getattr(data, "shape", ())
    if len(shape) != 2 or not all(shape) or int(shape[0]) * int(shape[1]) > MAX_LABEL_PIXELS:
        raise ValueError("Use a nonempty 2D Labels layer with at most 16 million pixels.")
    labels = np.asarray(data)
    if labels.dtype.kind not in "biu":
        raise ValueError("The Labels layer must contain integer label IDs.")
    mask = labels != 0
    if not np.any(mask):
        raise ValueError("The Labels layer has no non-zero labels to scan.")
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    matrix = np.eye(3) if image_to_world_yx is None else np.asarray(image_to_world_yx, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("The label-to-world transform must be a finite 2D affine.")
    return LabelROI(
        labels.tobytes(), tuple(int(n) for n in labels.shape), labels.dtype.str,
        ((int(rows[0]), int(columns[0])), (int(rows[-1]), int(columns[-1]))),
        tuple(int(value) for value in np.unique(labels[mask])), int(mask.sum()),
        str(image_layer_name), tuple(tuple(float(v) for v in row) for row in matrix),
    )


def _label_lattice(roi, spacing, *, collect=True, budget=None):
    low, high = roi.bounds()
    with np.errstate(over="ignore", divide="ignore"):
        ratios = (high - low) / spacing
    if not np.isfinite(ratios).all() or np.any(ratios > _MAX_CANDIDATES):
        raise ValueError("Label grid is too dense; increase pixel spacing.")
    counts = np.floor(ratios + 1e-10).astype(np.int64) + 1
    candidates = int(counts[0]) * int(counts[1])
    if candidates > _MAX_CANDIDATES:
        raise ValueError("Label grid requires too many candidate points; increase pixel spacing.")
    if budget is not None:
        if counts[0] > budget[0] or candidates > budget[1]:
            raise ValueError("Label target-count search reached its safe work limit.")
        budget[0] -= int(counts[0])
        budget[1] -= candidates
    total, chunks = 0, []
    for start in range(0, candidates, _CHUNK_SIZE):
        indices = np.arange(start, min(start + _CHUNK_SIZE, candidates))
        # X-major / Y-minor order, shared pitch in both image axes.
        points = low + np.column_stack((indices % counts[0], indices // counts[0])) * spacing
        inside = roi.labels_at(points) != 0
        total += int(inside.sum())
        if total > MAX_ROI_POINTS:
            if not collect:
                return MAX_ROI_POINTS + 1
            raise ValueError("The labels contain more than 250,000 grid points; increase pixel spacing.")
        if collect:
            chunks.append(points[inside])
    if not collect:
        return total
    if total < 2:
        raise ValueError("Fewer than two grid points fall in non-zero labels; reduce spacing or enlarge the labels.")
    return np.concatenate(chunks)


def sample_label_plan(roi, *, mode="count", total_points=400, spacing_px=10.0):
    """Return every masked grid point plus the shared X/Y pitch in pixels."""
    if mode == "count":
        target = _integer(total_points, "Target total points")
        spacing = _target_spacing(roi, target, area=roi.area, lattice_counter=_label_lattice)
    elif mode == "spacing":
        spacing = float(spacing_px)
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("Pixel spacing must be finite and greater than zero.")
    else:
        raise ValueError("Sampling mode must be 'count' or 'spacing'.")
    points = _label_lattice(roi, spacing)
    if not np.isfinite(points).all() or len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Label grid points cannot be resolved at this spacing.")
    return points, spacing
