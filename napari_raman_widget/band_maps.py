"""Numerical integration for spatial Raman intensity and band-ratio maps."""

from __future__ import annotations

import numpy as np

__all__ = ["band_map_values"]

# Limit temporary float conversion and finite masks for large spatial scans.
_MAX_INTEGRATION_SAMPLES = 1_000_000


def _band_integrals(spectra, axis, band):
    """Integrate a band using only the samples needed for its interpolation."""
    try:
        bounds = np.asarray(band)
        if np.iscomplexobj(bounds):
            raise ValueError
        bounds = np.asarray(bounds, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("A band must contain two finite real bounds (low, high).") from exc
    if bounds.shape != (2,) or not np.isfinite(bounds).all():
        raise ValueError("A band must contain two finite real bounds (low, high).")
    low, high = bounds
    if low >= high:
        raise ValueError("The band lower bound must be less than its upper bound.")
    if low < axis[0] or high > axis[-1]:
        raise ValueError("Band bounds must lie within the spectral axis range.")

    # Include neighboring samples only when an edge needs interpolation. In
    # particular, a bad sample just outside an exact bin edge is irrelevant.
    start = int(np.searchsorted(axis, low, side="right")) - 1
    stop = int(np.searchsorted(axis, high, side="left")) + 1
    x = axis[start:stop]
    # Integrate each clipped linear segment trapezoidally. Expressing the
    # interpolated edge trapezoids as weights avoids copying the full scan or
    # constructing an extra points-by-bins interpolated array.
    left = np.maximum(x[:-1], low)
    right = np.minimum(x[1:], high)
    width = right - left
    midpoint_fraction = ((left - x[:-1]) / (x[1:] - x[:-1])
                         + (right - x[:-1]) / (x[1:] - x[:-1])) * 0.5
    weights = np.zeros(x.size, dtype=float)
    weights[:-1] += width * (1.0 - midpoint_fraction)
    weights[1:] += width * midpoint_fraction
    areas = np.empty(spectra.shape[0], dtype=float)
    rows_per_chunk = max(1, _MAX_INTEGRATION_SAMPLES // x.size)
    for row_start in range(0, len(areas), rows_per_chunk):
        row_stop = min(row_start + rows_per_chunk, len(areas))
        try:
            y = np.asarray(spectra[row_start:row_stop, start:stop], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Spectra must contain real numeric samples.") from exc
        with np.errstate(invalid="ignore", over="ignore"):
            chunk = y @ weights
        chunk[~np.isfinite(y).all(axis=1) | ~np.isfinite(chunk)] = np.nan
        areas[row_start:row_stop] = chunk
    return areas


def band_map_values(spectra, axis, band_a, band_b=None, *, denominator_floor=1e-12):
    """Return an integrated intensity or band-area ratio for every spectrum.

    Parameters
    ----------
    spectra : array-like, shape (points, bins)
        Real spectral samples. Inputs are never modified. Only the slices
        needed by the requested bands are converted to floating point, in
        bounded row chunks to avoid allocating a full floating-point scan.
    axis : array-like, shape (bins,)
        At least two finite, strictly increasing or strictly decreasing real
        coordinates, in pixels or calibrated wavenumbers. Descending axes are
        reversed with their spectra so the integral keeps the intensity sign.
    band_a, band_b : pair of float
        Finite ``(low, high)`` bounds with ``low < high``, entirely within the
        axis range. Exact edges are linearly interpolated between samples, and
        areas use the trapezoidal rule. Omit ``band_b`` for intensity; provide
        it for the ratio of integrated area A to integrated area B.
    denominator_floor : float, optional
        Finite nonnegative threshold. Ratios with denominator areas less than
        or equal to this value are undefined and returned as NaN. The default
        is 1e-12 in the integrated intensity units.

    Returns
    -------
    numpy.ndarray, shape (points,)
        Signed area or ratio. A nonfinite sample needed within a band or for
        an interpolated edge makes that spectrum's result NaN; samples outside
        the bands have no effect. Nonfinite computed results also become NaN.
        No baseline correction, normalization, or clipping is applied.

    Raises
    ------
    ValueError
        If array shapes, the spectral axis, bounds, or threshold are invalid.
    """
    try:
        floor = float(denominator_floor)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("The denominator floor must be finite and nonnegative.") from exc
    if not np.isfinite(floor) or floor < 0:
        raise ValueError("The denominator floor must be finite and nonnegative.")

    try:
        x = np.asarray(axis)
        if np.iscomplexobj(x):
            raise ValueError
        x = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("The spectral axis must contain finite real coordinates.") from exc
    if x.ndim != 1 or x.size < 2:
        raise ValueError("The spectral axis must be one-dimensional with at least two bins.")
    if not np.isfinite(x).all():
        raise ValueError("The spectral axis must contain finite real coordinates.")
    increasing = bool(np.all(x[1:] > x[:-1]))
    decreasing = bool(np.all(x[1:] < x[:-1]))
    if not increasing and not decreasing:
        raise ValueError("The spectral axis must be strictly monotonic without duplicate bins.")

    try:
        data = np.asarray(spectra)
    except (TypeError, ValueError) as exc:
        raise ValueError("Spectra must have shape (points, spectral bins).") from exc
    if data.ndim != 2 or data.shape[1] != x.size:
        raise ValueError("Spectra must have shape (points, spectral bins) matching the axis.")
    if np.iscomplexobj(data):
        raise ValueError("Spectra must contain real numeric samples.")
    if decreasing:
        x = x[::-1]
        data = data[:, ::-1]

    numerator = _band_integrals(data, x, band_a)
    if band_b is None:
        return numerator
    denominator = _band_integrals(data, x, band_b)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > floor)
    values = np.full(numerator.shape, np.nan, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        np.divide(numerator, denominator, out=values, where=valid)
    values[~np.isfinite(values)] = np.nan
    return values
