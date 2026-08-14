"""Spectral processing utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

__all__ = [
    "filter_mean",
    "save_collection_record",
    "spectral_bias_from_dark_noise",
    "subtract_spectral_bias",
    "sum_detector_rows",
]


def sum_detector_rows(
    image: np.ndarray,
    start_row: int,
    end_row: int,
) -> np.ndarray:
    """Sum an inclusive detector-row range into one spectrum."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("detector image must have shape (rows, spectral_pixels)")

    if isinstance(start_row, (bool, np.bool_)) or not isinstance(
        start_row, (int, np.integer)
    ):
        raise TypeError("start_row must be an integer")
    if isinstance(end_row, (bool, np.bool_)) or not isinstance(
        end_row, (int, np.integer)
    ):
        raise TypeError("end_row must be an integer")

    start_row = int(start_row)
    end_row = int(end_row)
    row_count = image.shape[0]
    if not 0 <= start_row < row_count:
        raise ValueError(f"start_row must be between 0 and {row_count - 1}")
    if not start_row <= end_row < row_count:
        raise ValueError(
            f"end_row must be between start_row ({start_row}) and "
            f"{row_count - 1}"
        )

    return np.sum(image[start_row : end_row + 1], axis=0, dtype=float)


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


def spectral_bias_from_dark_noise(dark_noise: np.ndarray) -> np.ndarray:
    """Return the filtered mean spectrum represented by dark measurements."""
    dark_noise = np.asarray(dark_noise)
    if dark_noise.ndim != 2:
        raise ValueError(
            "dark noise must have shape "
            "(number_of_measurements, number_of_spectral_pixels)"
        )
    return filter_mean(dark_noise)


def subtract_spectral_bias(
    spectra: np.ndarray,
    spectral_bias: np.ndarray,
) -> np.ndarray:
    """Subtract one spectral bias vector while preserving negative values."""
    spectra = np.asarray(spectra, dtype=float)
    spectral_bias = np.asarray(spectral_bias, dtype=float)

    if spectra.ndim != 2:
        raise ValueError(
            "spectra must have shape "
            "(number_of_measurements, number_of_spectral_pixels)"
        )
    if spectral_bias.ndim != 1:
        raise ValueError("spectral bias must be a one-dimensional array")
    if spectra.shape[-1] != spectral_bias.shape[0]:
        raise ValueError(
            "dark-noise spectral length does not match acquired spectra: "
            f"{spectral_bias.shape[0]} != {spectra.shape[-1]}"
        )

    return spectra - spectral_bias
