"""Autofocus routines and retry-aware microscope operations."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

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
    "try_get_image",
    "try_get_xy_position",
    "try_get_z_position",
    "try_set_auto_shutter",
    "try_set_config",
    "try_set_exposure",
    "try_set_shutter_open",
    "try_set_xy_position",
    "try_set_z_position",
    "try_snap",
    "try_snap_image",
    "try_stop_sequence_acquisition",
    "try_wait_for_system",
]


_Result = TypeVar("_Result")


# ---------------------------------------------------------------------------
# Retry-aware microscope operations
# ---------------------------------------------------------------------------


def _retry_core_operation(
    operation_name: str,
    operation: Callable[[], _Result],
    attempts: int = 3,
    initial_delay: float = 0.1,
    delay_increment: float = 0.5,
) -> _Result:
    """Run a microscope operation with retry handling."""
    if attempts < 1:
        raise ValueError(
            "attempts must be at least one."
        )

    if initial_delay < 0:
        raise ValueError(
            "initial_delay cannot be negative."
        )

    if delay_increment < 0:
        raise ValueError(
            "delay_increment cannot be negative."
        )

    delay = float(initial_delay)
    last_error: RuntimeError | None = None

    for attempt in range(1, attempts + 1):
        try:
            if delay > 0:
                time.sleep(delay)

            return operation()

        except RuntimeError as error:
            last_error = error

            if attempt < attempts:
                delay += delay_increment

    raise RuntimeError(
        f"{operation_name} failed after "
        f"{attempts} attempts."
    ) from last_error


def try_wait_for_system(
    core: Any,
    attempts: int = 3,
) -> None:
    """Wait for all microscope devices."""

    def operation() -> None:
        core.waitForSystem()

    _retry_core_operation(
        "Wait for microscope system",
        operation,
        attempts=attempts,
    )


def try_get_xy_position(
    core: Any,
    attempts: int = 3,
) -> tuple[float, float]:
    """Read the current XY stage position."""

    def operation() -> tuple[float, float]:
        x_position, y_position = (
            core.getXYPosition()
        )

        return (
            float(x_position),
            float(y_position),
        )

    return _retry_core_operation(
        "Get XY position",
        operation,
        attempts=attempts,
    )


def try_set_xy_position(
    core: Any,
    x_position: float,
    y_position: float,
    attempts: int = 3,
) -> None:
    """Set the XY stage position."""

    def operation() -> None:
        core.setXYPosition(
            float(x_position),
            float(y_position),
        )
        core.waitForSystem()

    _retry_core_operation(
        (
            "Set XY position to "
            f"({x_position}, {y_position})"
        ),
        operation,
        attempts=attempts,
    )


def try_get_z_position(
    core: Any,
    attempts: int = 3,
) -> float:
    """Read the current Z position."""

    def operation() -> float:
        return float(
            core.getPosition()
        )

    return _retry_core_operation(
        "Get Z position",
        operation,
        attempts=attempts,
    )


def try_set_z_position(
    core: Any,
    position: float,
    attempts: int = 3,
) -> None:
    """Set the Z position."""

    def operation() -> None:
        core.setZPosition(
            float(position)
        )
        core.waitForSystem()

    _retry_core_operation(
        f"Set Z position to {position}",
        operation,
        attempts=attempts,
    )


def try_stop_sequence_acquisition(
    core: Any,
    attempts: int = 3,
) -> None:
    """Stop the active camera sequence."""

    def operation() -> None:
        core.stopSequenceAcquisition()
        core.waitForSystem()

    _retry_core_operation(
        "Stop sequence acquisition",
        operation,
        attempts=attempts,
    )


def try_set_config(
    core: Any,
    group: str,
    configuration: str,
    attempts: int = 3,
) -> None:
    """Apply a Micro-Manager configuration."""

    def operation() -> None:
        core.setConfig(
            group,
            configuration,
        )
        core.waitForSystem()

    _retry_core_operation(
        (
            f"Set configuration group {group} "
            f"to {configuration}"
        ),
        operation,
        attempts=attempts,
    )


def try_set_shutter_open(
    core: Any,
    shutter: str,
    is_open: bool,
    attempts: int = 3,
) -> None:
    """Open or close a named shutter."""

    def operation() -> None:
        core.setShutterOpen(
            shutter,
            bool(is_open),
        )
        core.waitForSystem()

    state = (
        "open"
        if is_open
        else "closed"
    )

    _retry_core_operation(
        f"Set {shutter} shutter {state}",
        operation,
        attempts=attempts,
    )


def try_set_exposure(
    core: Any,
    exposure: float,
    attempts: int = 3,
) -> None:
    """Set the camera exposure."""

    def operation() -> None:
        core.setExposure(
            float(exposure)
        )
        core.waitForSystem()

    _retry_core_operation(
        f"Set exposure to {exposure}",
        operation,
        attempts=attempts,
    )


def try_set_auto_shutter(
    core: Any,
    enabled: bool,
    attempts: int = 3,
) -> None:
    """Enable or disable automatic shutter control."""

    def operation() -> None:
        core.setAutoShutter(
            bool(enabled)
        )
        core.waitForSystem()

    _retry_core_operation(
        (
            "Enable automatic shutter"
            if enabled
            else "Disable automatic shutter"
        ),
        operation,
        attempts=attempts,
    )


def try_snap_image(
    core: Any,
    attempts: int = 3,
) -> None:
    """Acquire an image into the core image buffer."""

    def operation() -> None:
        core.snapImage()
        core.waitForSystem()

    _retry_core_operation(
        "Acquire image",
        operation,
        attempts=attempts,
    )


def try_get_image(
    core: Any,
    attempts: int = 3,
) -> np.ndarray:
    """Return the image in the core image buffer."""

    def operation() -> np.ndarray:
        image = core.getImage()
        core.waitForSystem()

        return np.asarray(image)

    return _retry_core_operation(
        "Get image",
        operation,
        attempts=attempts,
    )


def try_snap(
    core: Any,
    attempts: int = 3,
) -> np.ndarray:
    """Acquire and return one image."""

    def operation() -> np.ndarray:
        image = core.snap()
        core.waitForSystem()

        return np.asarray(image)

    return _retry_core_operation(
        "Acquire and return image",
        operation,
        attempts=attempts,
    )


# ---------------------------------------------------------------------------
# Spectral processing
# ---------------------------------------------------------------------------


def gaussian(
    x: np.ndarray,
    amplitude: float,
    center: float,
    sigma: float,
) -> np.ndarray:
    """Evaluate a Gaussian curve."""
    x = np.asarray(
        x,
        dtype=float,
    )

    return amplitude * np.exp(
        -(
            (x - center) ** 2
            / (2 * sigma**2)
        )
    )


def rescale(
    data: np.ndarray,
) -> np.ndarray:
    """Rescale data to the interval from zero to one."""
    data = np.asarray(
        data,
        dtype=float,
    )

    minimum = np.nanmin(data)
    maximum = np.nanmax(data)
    span = maximum - minimum

    if (
        not np.isfinite(span)
        or span == 0
    ):
        return np.zeros_like(
            data,
            dtype=float,
        )

    return (
        data - minimum
    ) / span


def remove_outliers(
    data: np.ndarray,
    standard_deviations: float = 4,
) -> np.ndarray:
    """Replace unusually high values with zero."""
    data = np.asarray(
        data,
        dtype=float,
    )

    if data.ndim < 2:
        raise ValueError(
            "data must contain at least "
            "two dimensions."
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
        + standard_deviations
        * standard_deviation
    )

    return np.where(
        data < threshold,
        data,
        0,
    )


# ---------------------------------------------------------------------------
# Autofocus routines
# ---------------------------------------------------------------------------


def autofocus_w_bkd(
    core: Any,
    daq: Any,
    collector: Any,
    volts: np.ndarray,
    search_range: float = 20,
    search_pts: int = 15,
    exposure: float = 1000,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
]:
    """Collect Raman reference spectra over a Z range.

    This routine records spectra but does not choose a final focus
    position. The caller may analyze the returned data and then restore
    or update the Z position.

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

    if search_range <= 0:
        raise ValueError(
            "search_range must be greater than zero."
        )

    if exposure <= 0:
        raise ValueError(
            "exposure must be greater than zero."
        )

    volts = np.asarray(
        volts,
        dtype=float,
    )

    if (
        volts.ndim != 2
        or volts.shape[1] != 2
    ):
        raise ValueError(
            "volts must have shape (N, 2)."
        )

    initial_z = try_get_z_position(
        core
    )
    z_offsets = np.linspace(
        -search_range,
        search_range,
        search_pts,
    )

    mean_spectra = []
    all_spectra = []
    shutter_opened = False

    try_stop_sequence_acquisition(
        core
    )
    daq.galvo.stop()
    try_set_config(
        core,
        "Channel",
        "RM",
    )

    try:
        try_set_shutter_open(
            core,
            "Fluoshutter",
            True,
        )
        shutter_opened = True

        for z_offset in tqdm(
            z_offsets,
            desc="Collecting reference spectra",
        ):
            try_set_z_position(
                core,
                initial_z + z_offset,
            )

            spectra = (
                collector.collect_spectra_pts(
                    volts,
                    exposure,
                )
            )
            spectra = np.asarray(
                spectra
            )

            mean_spectra.append(
                np.mean(
                    spectra,
                    axis=0,
                )
            )
            all_spectra.append(
                spectra
            )

    finally:
        if shutter_opened:
            try_set_shutter_open(
                core,
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
    image_shape: tuple[int, int] = (
        1024,
        1344,
    ),
) -> tuple[
    float,
    float,
    np.ndarray,
]:
    """Find a focus position from integrated Raman intensity.

    Parameters
    ----------
    point
        Selected image point in ``(y, x)`` order.
    start, end
        Spectral interval used to calculate focus intensity.
    search_range
        Z distance searched on either side of the initial position.
    search_pts
        Number of measured Z positions.
    max_volt
        Maximum galvo voltage used by the coordinate transformer.
    plot
        Display the measured and interpolated focus response.
    exposure
        Raman exposure used at each Z position.
    image_shape
        Camera shape in ``(height, width)`` order.

    Returns
    -------
    initial_z
        Z position before autofocus.
    best_z
        Estimated Z position at maximum Raman intensity.
    mean_spectra
        Mean spectrum acquired at each tested Z position.
    """
    if search_pts < 4:
        raise ValueError(
            "search_pts must be at least four "
            "for cubic interpolation."
        )

    if search_range <= 0:
        raise ValueError(
            "search_range must be greater than zero."
        )

    if exposure <= 0:
        raise ValueError(
            "exposure must be greater than zero."
        )

    if start < 0 or end <= start:
        raise ValueError(
            "The spectral interval must satisfy "
            "0 <= start < end."
        )

    point = np.asarray(
        point,
        dtype=float,
    ).reshape(-1)

    if point.shape != (2,):
        raise ValueError(
            "point must contain one "
            "(y, x) coordinate."
        )

    height, width = image_shape

    if height <= 0 or width <= 0:
        raise ValueError(
            "image_shape values must be "
            "greater than zero."
        )

    normalized_point = (
        point.reshape(1, 2)
        * np.array(
            [width, height]
        )
        / np.array(
            [height, width]
        )
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

    initial_z = try_get_z_position(
        core
    )
    z_offsets = np.linspace(
        -search_range,
        search_range,
        search_pts,
    )
    mean_spectra = []
    shutter_opened = False

    try_stop_sequence_acquisition(
        core
    )
    daq.galvo.stop()
    try_set_config(
        core,
        "Channel",
        "RM",
    )

    try:
        try_set_shutter_open(
            core,
            "Fluoshutter",
            True,
        )
        shutter_opened = True

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
        if shutter_opened:
            try_set_shutter_open(
                core,
                "Fluoshutter",
                False,
            )

    mean_spectra = np.asarray(
        mean_spectra
    )

    if mean_spectra.ndim != 2:
        raise ValueError(
            "The collected spectra do not have "
            "the expected two-dimensional shape."
        )

    if end > mean_spectra.shape[1]:
        raise ValueError(
            f"The spectral end index {end} "
            "exceeds the available spectrum "
            f"length {mean_spectra.shape[1]}."
        )

    integrated_intensity = (
        mean_spectra[
            :,
            start:end,
        ].sum(axis=1)
    )
    median_spectrum = np.median(
        mean_spectra
    )

    if (
        np.isfinite(median_spectrum)
        and median_spectrum != 0
    ):
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
        np.nanargmax(
            fine_intensity
        )
    )
    best_offset = float(
        fine_offsets[peak_index]
    )
    peak_intensity = float(
        fine_intensity[peak_index]
    )

    if abs(best_offset) >= search_range:
        best_offset = 0.0

    best_z = (
        initial_z + best_offset
    )

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
        axes.set_xlabel(
            "Z offset"
        )
        axes.set_ylabel(
            "Scaled Raman intensity"
        )
        axes.legend()
        figure.tight_layout()

    return (
        initial_z,
        best_z,
        mean_spectra,
    )