"""Local brightfield identity matching, independent of segmentation label IDs."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import correlate


@dataclass(frozen=True)
class CellReference:
    image: np.ndarray
    point_yx: np.ndarray


@dataclass(frozen=True)
class CellMatch:
    point_yx: np.ndarray
    accepted: bool
    score: float
    reverse_error: float
    margin: float


def _feature(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32).copy()
    image -= ndi.gaussian_filter(image, 7.0)
    scale = float(np.percentile(np.abs(image), 99))
    if scale > 0:
        image /= scale
    return np.clip(image, -1, 1)


def cell_reference(image, point_yx, maximum_step_px) -> CellReference | None:
    """Copy a local reference; never retain the acquisition image stack."""
    point = np.asarray(point_yx, dtype=float)
    radius = 40 + int(np.ceil(maximum_step_px))
    center = np.rint(point).astype(int)
    low, high = center - radius, center + radius + 1
    if np.any(low < 0) or np.any(high > image.shape):
        return None
    patch = np.asarray(image[low[0]:high[0], low[1]:high[1]], dtype=np.float32)
    if not np.isfinite(patch).all():
        return None
    return CellReference(_feature(patch), point.copy())


def _locate(reference, search_image, maximum_step_px):
    search = int(np.ceil(maximum_step_px))
    template = reference[search:search + 81, search:search + 81].copy()
    yy, xx = np.indices(template.shape)
    # Include the cell and its budding context without the trench walls.
    ellipse = ((yy - 40) / 22) ** 2 + ((xx - 40) / 38) ** 2
    weight = np.exp(-ellipse / 2) * (ellipse <= 1)
    count = weight.sum()
    template -= np.sum(weight * template) / count
    energy = float(np.sum(weight * template ** 2))
    if energy <= 1e-9:
        return np.zeros(2), 0.0, 0.0
    total = correlate(search_image, weight, mode="valid", method="fft")
    variance = correlate(search_image ** 2, weight, mode="valid", method="fft")
    variance -= total ** 2 / count
    cross = correlate(search_image, weight * template, mode="valid", method="fft")
    scores = cross / np.sqrt(np.maximum(variance, 1e-9) * energy)
    yy, xx = np.indices(scores.shape)
    distance = np.hypot(yy - search, xx - search)
    adjusted = scores - 0.00010 * distance ** 2
    adjusted[distance > maximum_step_px] = -np.inf
    peak = np.unravel_index(adjusted.argmax(), adjusted.shape)
    competing = adjusted[np.hypot(yy - peak[0], xx - peak[1]) > 20]
    margin = (
        float(adjusted[peak] - np.max(competing))
        if competing.size and np.isfinite(competing).any() else 1.0
    )
    return np.asarray(peak) - search, float(scores[peak]), margin


def match_cell(reference, image, predicted_yx, maximum_step_px) -> CellMatch:
    """Require a distinct forward match that returns to the original cell.

    The caller commits a new reference only after accepting the final aim.
    A missing Cellpose object therefore cannot teach a neighbor as the target.
    """
    predicted = np.asarray(predicted_yx, dtype=float)
    current = cell_reference(image, predicted, maximum_step_px)
    failure = CellMatch(predicted.copy(), False, 0.0, float("inf"), 0.0)
    if current is None or current.image.shape != reference.image.shape:
        return failure
    def compare(transform):
        old = transform(reference.image)
        offset, score, margin = _locate(old, transform(current.image), maximum_step_px)
        fractional = reference.point_yx - np.rint(reference.point_yx)
        candidate = np.rint(predicted) + offset + fractional
        candidate_reference = cell_reference(image, candidate, maximum_step_px)
        if candidate_reference is None:
            return failure
        reverse, _, _ = _locate(
            transform(candidate_reference.image), old, maximum_step_px,
        )
        reverse_error = float(np.linalg.norm(reverse))
        step = float(np.linalg.norm(candidate - predicted))
        minimum_margin = 0.05 if step > 20 else 0.025
        accepted = bool(
            np.isfinite(score) and score >= 0.30
            and reverse_error <= 5 and margin >= minimum_margin
            and step <= maximum_step_px
        )
        return CellMatch(candidate, accepted, score, reverse_error, margin)

    direct = compare(lambda patch: patch)
    if direct.accepted:
        return direct
    # Focus/phase changes can reverse fine brightfield texture. A recovery
    # must agree in location in both a polarity-independent and a smoothed
    # representation, and both must return to the original cell. One weak
    # alternate match alone is never enough to replace the target.
    magnitude = compare(np.abs)
    smooth = compare(lambda patch: ndi.gaussian_filter(patch, 2.0))
    recovered = bool(
        magnitude.score >= 0.30 and smooth.score >= 0.30
        and magnitude.margin >= 0.025
        and magnitude.reverse_error <= 3 and smooth.reverse_error <= 3
        and np.linalg.norm(magnitude.point_yx - smooth.point_yx) <= 3
        and np.linalg.norm(magnitude.point_yx - predicted) <= maximum_step_px
    )
    if recovered:
        return CellMatch(
            magnitude.point_yx, True, magnitude.score,
            magnitude.reverse_error, magnitude.margin,
        )
    return direct
