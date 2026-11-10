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
    "racer_detector_signature",
    "save_collection_record",
    "save_dark_noise_artifacts",
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


def racer_detector_signature(
    collector: object,
    data: np.ndarray,
    read_mode: str,
    exposure_ms: float | None = None,
    *,
    include_acquisition_timing: bool = True,
) -> dict:
    """Describe the fixed detector settings used by RACER acquisitions."""
    read_mode = str(read_mode).lower()
    detector_data = np.asarray(data)
    if read_mode == "image":
        if detector_data.ndim < 2:
            raise ValueError("image data must include detector Y and X axes")
        detector_shape = [
            int(detector_data.shape[-2]),
            int(detector_data.shape[-1]),
        ]
        detector_y_stop: int | None = int(detector_data.shape[-2])
        detector_y_bin: int | str = 1
    elif read_mode == "fvb":
        if detector_data.ndim < 1:
            raise ValueError("FVB data must include a detector X axis")
        detector_shape = [int(detector_data.shape[-1])]
        detector_y_stop = None
        detector_rows = getattr(collector, "detector_rows", None)
        if detector_rows is not None:
            try:
                detector_y_stop = int(detector_rows)
            except (TypeError, ValueError):
                detector_y_stop = None
        if detector_y_stop is None:
            sdk = getattr(collector, "_sdk", None)
            if sdk is not None and hasattr(sdk, "GetDetector"):
                try:
                    detector_result = sdk.GetDetector()
                    if len(detector_result) >= 3:
                        detector_y_stop = int(detector_result[2])
                except Exception:
                    detector_y_stop = None
        detector_y_bin = "full"
    else:
        raise ValueError("RACER detector signatures support FVB or image")

    collector_type = type(collector)
    sdk = getattr(collector, "_sdk", None)

    def sdk_result(method_name, *arguments):
        """Return one best-effort Andor SDK value after its status code."""
        if sdk is None or not hasattr(sdk, method_name):
            return None
        try:
            result = getattr(sdk, method_name)(*arguments)
        except Exception:
            return None
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            return None
        return result[1] if len(result) == 2 else tuple(result[1:])

    camera_serial = sdk_result("GetCameraSerialNumber")
    bit_depth = sdk_result("GetBitDepth", 0)
    preamp_gain = sdk_result("GetPreAmpGain", 0)
    horizontal_shift_speed = sdk_result("GetHSSpeed", 0, 0, 2)
    vertical_shift_speed = sdk_result("GetVSSpeed", 0)
    acquisition_timings = (
        sdk_result("GetAcquisitionTimings")
        if include_acquisition_timing
        else None
    )
    actual_exposure_ms = None
    if acquisition_timings is not None:
        timings = (
            acquisition_timings
            if isinstance(acquisition_timings, tuple)
            else (acquisition_timings,)
        )
        if timings:
            try:
                actual_exposure_ms = round(float(timings[0]) * 1000.0, 6)
            except (TypeError, ValueError):
                actual_exposure_ms = None

    return {
        "schema_version": 2,
        "collector_type": (
            f"{collector_type.__module__}.{collector_type.__qualname__}"
        ),
        "camera_serial": (
            None if camera_serial is None else str(camera_serial)
        ),
        "camera_identity_source": (
            "andor_sdk" if camera_serial is not None else "unavailable"
        ),
        "read_mode": read_mode,
        "detector_shape": detector_shape,
        "detector_x_start": 0,
        "detector_x_stop_exclusive": int(detector_data.shape[-1]),
        "detector_x_bin": 1,
        "detector_y_start": 0,
        "detector_y_stop_exclusive": detector_y_stop,
        "detector_y_bin": detector_y_bin,
        "baseline_clamp": 0,
        "ad_channel_index": 0,
        "bit_depth": None if bit_depth is None else int(bit_depth),
        "preamp_gain_index": 0,
        "preamp_gain": (
            None if preamp_gain is None else float(preamp_gain)
        ),
        "horizontal_shift_speed_index": 2,
        "horizontal_shift_speed": (
            None
            if horizontal_shift_speed is None
            else float(horizontal_shift_speed)
        ),
        "vertical_shift_speed_index": 0,
        "vertical_shift_speed": (
            None
            if vertical_shift_speed is None
            else float(vertical_shift_speed)
        ),
        "number_accumulations": 1,
        "output_amplifier_index": 0,
        "em_gain_mode": 0,
        "requested_exposure_ms": (
            None if exposure_ms is None else float(exposure_ms)
        ),
        "actual_exposure_ms": actual_exposure_ms,
        "gain_source": "racer_default_assumption",
    }


def save_dark_noise_artifacts(
    filename: str | Path,
    data: np.ndarray,
    *,
    collector: object,
    read_mode: str,
    exposure_ms: float,
    metadata: dict | None = None,
) -> tuple[Path | None, Path | None]:
    """Save dark data for bias subtraction and/or RACER calibration.

    FVB acquisitions produce both the existing ``.npy`` bias file and a
    RACER-compatible closed-shutter Zarr store. Image acquisitions produce
    only the Zarr store because row-wise bias subtraction is not defined for
    that view. Single-track remains bias-only because RACER does not validate
    that readout mode.
    """
    read_mode = str(read_mode).lower()
    if read_mode not in {"fvb", "image", "single_track"}:
        raise ValueError(
            "read_mode must be 'fvb', 'image', or 'single_track'"
        )

    dark_data = np.asarray(data)
    expected_dimensions = 3 if read_mode == "image" else 2
    if dark_data.ndim != expected_dimensions or dark_data.shape[0] == 0:
        expected = (
            "(repeats, detector_y, detector_x)"
            if read_mode == "image"
            else "(repeats, detector_x)"
        )
        raise ValueError(f"{read_mode} dark data must have shape {expected}")

    requested_path = Path(filename)
    npy_path = requested_path.with_suffix(".npy")
    saved_npy: Path | None = None
    if read_mode in {"fvb", "single_track"}:
        np.save(npy_path, dark_data, allow_pickle=False)
        saved_npy = npy_path

    racer_store: Path | None = None
    if read_mode in {"fvb", "image"}:
        saved_metadata = dict(metadata or {})
        saved_metadata.update(
            record_type="racer_closed_shutter_calibration",
            format_version=1,
            racer_role="closed_shutter",
            units="ADU",
            gain_e_per_adu=0.7,
            saturation_adu=65535.0,
            gain_source="racer_default_assumption",
            racer_detector_signature=racer_detector_signature(
                collector, dark_data, read_mode, exposure_ms
            ),
            read_mode=read_mode,
            exposure_ms=float(exposure_ms),
            repeats=int(dark_data.shape[0]),
        )
        racer_store = save_collection_record(
            npy_path,
            dark_data,
            saved_metadata,
        )

    return saved_npy, racer_store


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
