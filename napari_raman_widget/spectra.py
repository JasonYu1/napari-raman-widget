"""Spectral processing utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.linalg import solveh_banded
from scipy.signal import savgol_filter
from scipy.sparse import diags

__all__ = [
    "asls_baseline",
    "filter_mean",
    "save_collection_record",
    "smooth_spectra",
    "spectral_bias_from_dark_noise",
    "subtract_spectral_bias",
    "sum_detector_rows",
]


def asls_baseline(
    spectra: np.ndarray,
    lam: float = 1e6,
    asymmetry: float = 0.01,
    iterations: int = 10,
) -> np.ndarray:
    """Estimate spectral baselines with asymmetric least squares.

    The Whittaker penalty suppresses baseline curvature, while asymmetric
    weights keep positive spectral peaks from pulling the estimate upward.
    One- and two-dimensional inputs are supported; rows in a two-dimensional
    input are fitted independently. Non-finite samples are ignored by the fit
    and returned as ``nan``. If a row has fewer than three finite samples, its
    finite values receive a zero baseline so they remain visible.
    """
    spectra = np.asarray(spectra, dtype=float)
    if spectra.ndim not in (1, 2):
        raise ValueError("spectra must be a one- or two-dimensional array")
    if spectra.shape[-1] < 3:
        raise ValueError("AsLS baseline fitting requires at least 3 pixels")
    if isinstance(lam, (bool, np.bool_)) or not np.isfinite(lam) or lam <= 0:
        raise ValueError("baseline lambda must be a finite positive number")
    if not np.isfinite(asymmetry) or not 0 < asymmetry < 1:
        raise ValueError("baseline asymmetry must be between 0 and 1")
    if isinstance(iterations, (bool, np.bool_)) or not isinstance(
        iterations, (int, np.integer)
    ) or iterations < 1:
        raise ValueError("baseline iterations must be a positive integer")

    rows = spectra[np.newaxis, :] if spectra.ndim == 1 else spectra
    pixel_count = rows.shape[-1]
    difference = diags(
        (
            np.ones(pixel_count - 2),
            -2 * np.ones(pixel_count - 2),
            np.ones(pixel_count - 2),
        ),
        (0, 1, 2),
        shape=(pixel_count - 2, pixel_count),
        format="csc",
    )
    penalty = float(lam) * (difference.T @ difference)
    penalty_bands = np.zeros((3, pixel_count))
    penalty_bands[0] = penalty.diagonal()
    penalty_bands[1, :-1] = penalty.diagonal(-1)
    penalty_bands[2, :-2] = penalty.diagonal(-2)
    baselines = np.empty_like(rows, dtype=float)
    pixel_indices = np.arange(pixel_count)

    for index, spectrum in enumerate(rows):
        finite = np.isfinite(spectrum)
        if np.count_nonzero(finite) < 3:
            baseline = np.zeros(pixel_count)
            baseline[~finite] = np.nan
            baselines[index] = baseline
            continue
        fitted_spectrum = np.interp(
            pixel_indices,
            pixel_indices[finite],
            spectrum[finite],
        )
        weights = np.ones(pixel_count)
        baseline = fitted_spectrum.copy()
        for _ in range(int(iterations)):
            system_bands = penalty_bands.copy()
            system_bands[0] += weights
            baseline = solveh_banded(
                system_bands,
                weights * fitted_spectrum,
                lower=True,
                check_finite=False,
            )
            updated_weights = np.where(
                fitted_spectrum > baseline,
                asymmetry,
                1 - asymmetry,
            )
            if np.array_equal(updated_weights, weights):
                break
            weights = updated_weights
        baseline[~finite] = np.nan
        baselines[index] = baseline

    return baselines[0] if spectra.ndim == 1 else baselines


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


def smooth_spectra(
    spectra: np.ndarray,
    window_length: int,
) -> np.ndarray:
    """Savitzky-Golay smooth along spectral pixels with polynomial order 3."""
    spectra = np.asarray(spectra, dtype=float)
    if spectra.ndim not in (1, 2):
        raise ValueError("spectra must be a one- or two-dimensional array")
    if isinstance(window_length, bool) or not isinstance(
        window_length, (int, np.integer)
    ):
        raise ValueError("smoothing window must be an integer")
    window_length = int(window_length)
    if window_length <= 3 or window_length % 2 == 0:
        raise ValueError(
            "smoothing window must be an odd integer greater than 3"
        )
    if window_length > spectra.shape[-1]:
        raise ValueError(
            "smoothing window cannot exceed the spectral pixel count"
        )
    return savgol_filter(
        spectra,
        window_length=window_length,
        polyorder=3,
        axis=-1,
    )
