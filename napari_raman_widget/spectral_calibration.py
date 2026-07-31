"""Pixel-to-Raman-shift calibration models and JSON persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def pixel_to_shift(pixels, pixel_positions, known_shifts, degree=2):
    """Convert detector pixels to Raman shifts with a polynomial fit."""
    coeffs = np.polyfit(pixel_positions, known_shifts, deg=degree)
    return np.polyval(coeffs, pixels)


@dataclass(frozen=True)
class PixelToWavenumberCalibration:
    """Known detector-pixel and Raman-shift pairs for a polynomial fit."""

    pixel_positions: np.ndarray
    known_shifts: np.ndarray
    degree: int = 2

    def __post_init__(self) -> None:
        pixels = np.asarray(self.pixel_positions, dtype=float)
        shifts = np.asarray(self.known_shifts, dtype=float)
        try:
            degree = int(self.degree)
        except (TypeError, ValueError) as error:
            raise ValueError("polynomial degree must be an integer") from error
        if isinstance(self.degree, bool) or degree != self.degree:
            raise ValueError("polynomial degree must be an integer")
        if degree < 1:
            raise ValueError("polynomial degree must be at least one")
        if pixels.ndim != 1 or shifts.ndim != 1:
            raise ValueError("calibration values must be one-dimensional")
        if len(pixels) != len(shifts):
            raise ValueError(
                "pixel_positions and known_shifts must have the same length"
            )
        required_points = degree + 1
        if len(pixels) < required_points:
            raise ValueError(
                f"a degree-{degree} calibration requires at least "
                f"{required_points} points"
            )
        if not np.all(np.isfinite(pixels)) or not np.all(np.isfinite(shifts)):
            raise ValueError("calibration values must all be finite")
        if len(np.unique(pixels)) != len(pixels):
            raise ValueError("calibration pixel positions must be unique")

        order = np.argsort(pixels)
        pixels = pixels[order]
        shifts = shifts[order]
        # Fit once while validating so invalid/rank-deficient input fails at load.
        np.polyfit(pixels, shifts, deg=degree)
        object.__setattr__(self, "pixel_positions", pixels)
        object.__setattr__(self, "known_shifts", shifts)
        object.__setattr__(self, "degree", degree)

    @property
    def coefficients(self) -> np.ndarray:
        """Polynomial coefficients in descending-power order."""
        return np.polyfit(
            self.pixel_positions, self.known_shifts, deg=self.degree
        )

    def transform(self, pixels) -> np.ndarray:
        """Evaluate the calibration at detector pixel indices."""
        return pixel_to_shift(
            pixels,
            self.pixel_positions,
            self.known_shifts,
            degree=self.degree,
        )

    def to_dict(self) -> dict:
        """Return the stable, human-readable JSON representation."""
        return {
            "format_version": 1,
            "calibration_type": "pixel_to_wavenumber",
            "polynomial_degree": self.degree,
            "pixel_positions": self.pixel_positions.tolist(),
            "known_shifts": self.known_shifts.tolist(),
            "units": "cm^-1",
            "coefficients": self.coefficients.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PixelToWavenumberCalibration":
        """Build a calibration from its JSON object representation."""
        if not isinstance(data, dict):
            raise ValueError("calibration JSON must contain an object")
        degree = data.get("polynomial_degree", 2)
        try:
            pixels = data["pixel_positions"]
            shifts = data["known_shifts"]
        except KeyError as error:
            raise ValueError(
                f"calibration JSON is missing {error.args[0]!r}"
            ) from error
        return cls(pixels, shifts, degree=degree)


def save_pixel_to_wavenumber_calibration(
    path: str | Path,
    calibration: PixelToWavenumberCalibration,
) -> Path:
    """Save *calibration* as JSON and return the resolved output path."""
    output = Path(path).expanduser()
    if output.suffix.lower() != ".json":
        output = output.with_suffix(".json")
    with output.open("w", encoding="utf-8") as stream:
        json.dump(calibration.to_dict(), stream, indent=2)
        stream.write("\n")
    return output.resolve()


def load_pixel_to_wavenumber_calibration(
    path: str | Path,
) -> PixelToWavenumberCalibration:
    """Load and validate a pixel-to-wavenumber calibration JSON file."""
    with Path(path).expanduser().open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    return PixelToWavenumberCalibration.from_dict(data)
