"""Autofocus routines for Raman acquisition."""

from __future__ import annotations

import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from tqdm.auto import tqdm

__all__ = [
    "autofocus_w_bkd",
    "autofocus_w_raman",
    "gaussian",
    "remove_outliers",
    "rescale",
    "try_set_z_position",
]


def gaussian(
    x: np.ndarray,
    amplitude: float,
    center: float,
    sigma: float,
) -> np.ndarray:
    """Evaluate a Gaussian curve."""
    x = np.asarray(x, dtype=float)

    return amplitude * np.exp(
        -((x - center) ** 2) / (2 * sigma**2)
    )


def rescale(data: np.ndarray) -> np.ndarray:
    """Rescale data to the interval from zero to one."""
    data = np.asarray(data, dtype=float)
    minimum = np.nanmin(data)
    maximum = np.nanmax(data)
    span = maximum - minimum

    if not np.isfinite(span) or span == 0:
        return np.zeros_like(data, dtype=float)

    return (data - minimum) / span


def remove_outliers(
    data: np.ndarray,
    standard_deviations: float = 4,
) -> np.ndarray:
    """Replace unusually high values with zero."""
    data = np.asarray(data, dtype=float)

    if data.ndim < 2:
        raise ValueError(
            "data must contain at least two dimensions."
        )

    mean = np.mean(
        data,
        axis=1,
        keepdims=True,
    )
    standard_deviation = np.std(
        data,
        axis=1,
        keepdims=True,
    )

    threshold = (
        mean
        + standard_deviations * standard_deviation
    )

    return np.where(
        data < threshold,
        data,
        0,
    )


def try_set_z_position(
    core: Any,
    position: float,
    attempts: int = 20,
    initial_delay: float = 0.1,
    delay_increment: float = 1.0,
) -> None:
    """Set the Z position, retrying temporary device failures."""
    if attempts < 1:
        raise ValueError("attempts must be at least one.")

    delay = float(initial_delay)
    last_error: RuntimeError | None = None

    for _ in range(attempts):
        try:
            time.sleep(delay)
            core.setZPosition(position)
            core.waitForSystem()
            return
        except RuntimeError as error:
            last_error = error
            delay += delay_increment

    raise RuntimeError(
        f"Could not set the Z position to {position} "
        f"after {attempts} attempts."
    ) from last_error


def autofocus_w_bkd(
    core: Any,
    daq: Any,
    collector: Any,
    volts: np.ndarray,
    search_range: float = 20,
    search_pts: int = 15,
    exposure: float = 1000,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Collect Raman reference spectra across a Z-position range.

    This routine records spectra across the requested range. It does not
    select or move to a best-focus position. The caller can analyze the
    returned spectra and restore or update the stage position.

    Returns
    -------
    initial_z
        Z position before the scan.
    mean_spectra
        Mean spectrum at each Z position.
    all_spectra
        Complete repeated spectra at each Z position.
    """
    if search_pts < 2:
        raise ValueError(
            "search_pts must be at least two."
        )

    volts = np.asarray(volts, dtype=float)

    if volts.ndim != 2 or volts.shape[1] != 2:
        raise ValueError(
            "volts must have shape (N, 2)."
        )

    initial_z = float(core.getPosition())
    z_offsets = np.linspace(
        -search_range,
        search_range,
        search_pts,
    )

    mean_spectra = []
    all_spectra = []

    core.stopSequenceAcquisition()
    daq.galvo.stop()
    core.setConfig("Channel", "RM")
    core.setShutterOpen("Fluoshutter", True)

    try:
        for z_offset in tqdm(
            z_offsets,
            desc="Collecting reference spectra",
        ):
            core.setPosition(
                initial_z + z_offset
            )
            core.waitForSystem()

            spectra = (
                collector.collect_spectra_pts(
                    volts,
                    exposure,
                )
            )
            spectra = np.asarray(spectra)

            mean_spectra.append(
                np.mean(spectra, axis=0)
            )
            all_spectra.append(spectra)
    finally:
        core.setShutterOpen(
            "Fluoshutter",
            False,
        )

    return (
        initial_z,
        np.asarray(mean_spectra),
        np.asarray(all_spectra),
    )


def autofocus_w_raman(
    core: Any,
    daq: Any,
    collector: Any,
    transformer: Any,
    point: np.ndarray,
    start: int = 500,
    end: int = 2000,
    search_range: float = 10,
    search_pts: int = 10,
    max_volt: float = 1.5,
    plot: bool = True,
    exposure: float = 10,
    image_shape: tuple[int, int] = (1024, 1344),
) -> tuple[float, float, np.ndarray]:
    """Find a focus position using integrated Raman intensity.

    Parameters
    ----------
    point
        Selected image point in ``(y, x)`` order.
    start, end
        Spectral pixel interval used to calculate focus intensity.
    image_shape
        Camera image shape in ``(height, width)`` order.

    Returns
    -------
    initial_z
        Z position before autofocus.
    best_z
        Estimated Z position at maximum Raman intensity.
    spectra
        Mean spectrum acquired at each tested Z position.
    """
    if search_pts < 4:
        raise ValueError(
            "search_pts must be at least four for cubic interpolation."
        )

    if start < 0 or end <= start:
        raise ValueError(
            "The spectral interval must satisfy 0 <= start < end."
        )

    point = np.asarray(
        point,
        dtype=float,
    ).reshape(-1)

    if point.shape != (2,):
        raise ValueError(
            "point must contain one (y, x) coordinate."
        )

    height, width = image_shape
    normalized_point = (
        point.reshape(1, 2)
        * np.array([width, height])
        / np.array([height, width])
    )

    volts = transformer.BF_to_volts(
        normalized_point,
        max_volts=max_volt,
    )

    repeated_volts = np.repeat(
        volts,
        repeats=2,
        axis=0,
    )

    initial_z = float(core.getPosition())
    z_offsets = np.linspace(
        -search_range,
        search_range,
        search_pts,
    )
    mean_spectra = []

    core.stopSequenceAcquisition()
    daq.galvo.stop()
    core.setConfig("Channel", "RM")
    core.setShutterOpen("Fluoshutter", True)
    core.waitForSystem()

    try:
        for z_offset in tqdm(
            z_offsets,
            desc="Running Raman autofocus",
        ):
            try_set_z_position(
                core,
                initial_z + z_offset,
            )

            spectra = (
                collector.collect_spectra_pts(
                    repeated_volts,
                    exposure,
                )
            )

            mean_spectra.append(
                np.mean(
                    np.asarray(spectra),
                    axis=0,
                )
            )
    finally:
        core.setShutterOpen(
            "Fluoshutter",
            False,
        )
        core.waitForSystem()

    mean_spectra = np.asarray(mean_spectra)

    if end > mean_spectra.shape[1]:
        raise ValueError(
            f"The spectral end index {end} exceeds the "
            f"available length {mean_spectra.shape[1]}."
        )

    integrated_intensity = mean_spectra[
        :,
        start:end,
    ].sum(axis=1)

    median_spectrum = np.median(mean_spectra)

    if median_spectrum != 0:
        integrated_intensity = (
            integrated_intensity
            / median_spectrum
        )

    focus_intensity = rescale(
        integrated_intensity
    )

    interpolation = interp1d(
        z_offsets,
        focus_intensity,
        kind="cubic",
    )

    fine_offsets = np.linspace(
        z_offsets.min(),
        z_offsets.max(),
        1000,
    )
    fine_intensity = interpolation(
        fine_offsets
    )

    peak_index = int(
        np.nanargmax(fine_intensity)
    )
    best_offset = float(
        fine_offsets[peak_index]
    )
    peak_intensity = float(
        fine_intensity[peak_index]
    )

    if abs(best_offset) >= search_range:
        best_offset = 0.0

    best_z = initial_z + best_offset

    if plot:
        figure, axes = plt.subplots()
        axes.scatter(
            z_offsets,
            focus_intensity,
            label="Measured",
        )
        axes.plot(
            fine_offsets,
            fine_intensity,
            label="Interpolated",
        )
        axes.plot(
            best_offset,
            peak_intensity,
            "r*",
            markersize=10,
            label="Estimated focus",
        )
        axes.set_xlabel("Z offset")
        axes.set_ylabel("Scaled Raman intensity")
        axes.legend()
        figure.tight_layout()

    return (
        initial_z,
        best_z,
        mean_spectra,
    )


# Compatibility with the names used by the previous control package.
remove_outlier = remove_outliers
try_set_ZPosition = try_set_z_position