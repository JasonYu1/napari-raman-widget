"""Spectral processing utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

__all__ = ["filter_mean", "save_collection_record"]


def save_collection_record(
    filename: str | Path,
    data: np.ndarray,
    metadata: dict,
) -> Path:
    """Save acquired detector data and metadata in one xarray/Zarr store."""
    requested_path = Path(filename)
    if requested_path.suffix.lower() in {".npy", ".npz", ".json", ".zarr"}:
        base_path = requested_path.with_suffix("")
    else:
        base_path = requested_path
    store_path = Path(f"{base_path}.zarr")

    array = np.asarray(data)
    if array.ndim == 2:
        dimensions = ("repeat", "detector_x")
    elif array.ndim == 3:
        dimensions = ("repeat", "detector_y", "detector_x")
    else:
        dimensions = tuple(f"dimension_{index}" for index in range(array.ndim))

    saved_metadata = {
        key: value for key, value in metadata.items() if value is not None
    }
    saved_metadata.update(
        data_variable="signal",
        data_shape=list(array.shape),
        data_dtype=str(array.dtype),
    )
    dataset = xr.Dataset(
        data_vars={"signal": (dimensions, array)},
        attrs=saved_metadata,
    )
    dataset.to_zarr(store_path, mode="w", consolidated=True)
    return store_path


def filter_mean(
    spectra: np.ndarray,
    f: float = 2,
) -> np.ndarray:
    """Calculate a mean spectrum after excluding outlying values.

    For each spectral pixel, measurements farther than ``f`` standard
    deviations from the initial mean are excluded.

    Parameters
    ----------
    spectra
        Spectral measurements with shape
        ``(number_of_measurements, number_of_pixels)``.
    f
        Number of standard deviations retained around the initial mean.

    Returns
    -------
    numpy.ndarray
        Filtered mean spectrum with shape ``(number_of_pixels,)``.
    """
    spectra = np.asarray(
        spectra,
        dtype=float,
    )

    if spectra.ndim != 2:
        raise ValueError(
            "spectra must have shape "
            "(number_of_measurements, number_of_pixels)."
        )

    if spectra.shape[0] == 0:
        raise ValueError(
            "spectra must contain at least one measurement."
        )

    if f < 0:
        raise ValueError(
            "f cannot be negative."
        )

    mean_spectrum = np.nanmean(
        spectra,
        axis=0,
    )
    standard_deviation = np.nanstd(
        spectra,
        axis=0,
    )

    lower_limit = (
        mean_spectrum
        - f * standard_deviation
    )
    upper_limit = (
        mean_spectrum
        + f * standard_deviation
    )

    accepted = (
        np.isfinite(spectra)
        & (spectra >= lower_limit)
        & (spectra <= upper_limit)
    )

    accepted_counts = np.sum(
        accepted,
        axis=0,
    )
    accepted_sums = np.sum(
        np.where(
            accepted,
            spectra,
            0,
        ),
        axis=0,
    )

    filtered_mean = np.divide(
        accepted_sums,
        accepted_counts,
        out=mean_spectrum.copy(),
        where=accepted_counts > 0,
    )

    return filtered_mean
