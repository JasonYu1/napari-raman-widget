"""Shared state and deterministic signal generation for demonstration mode."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DemoWorld:
    """A virtual Raman microscope sample shared by all demo devices."""

    image_height: int = 1024
    image_width: int = 1344
    spectrum_pixels: int = 2048
    microns_per_pixel: float = 0.5
    maximum_galvo_voltage: float = 1.5
    stage_x: float = 0.0
    stage_y: float = 0.0
    stage_z: float = 0.0
    focus_z: float = 0.0
    galvo_x: float = 0.0
    galvo_y: float = 0.0
    exposure_ms: float = 10.0
    wavelength_nm: float = 700.0
    grating: int = 1
    channel: str = "BF"
    auto_shutter: bool = True
    shutters: dict[str, bool] = field(default_factory=dict)
    time_index: int = 0
    cell_positions_um: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(20260723)
        self.cell_positions_um = rng.uniform(
            low=(-240.0, -175.0),
            high=(240.0, 175.0),
            size=(28, 2),
        )

    @property
    def focus_strength(self) -> float:
        """Return a zero-to-one signal strength for the current Z position."""
        distance = self.stage_z - self.focus_z
        return float(np.exp(-0.5 * (distance / 3.0) ** 2))

    def set_galvo(self, voltage_xy: np.ndarray) -> None:
        voltage_xy = np.asarray(voltage_xy, dtype=float).reshape(2)
        self.galvo_x = float(voltage_xy[0])
        self.galvo_y = float(voltage_xy[1])

    def galvo_pixel_yx(self) -> np.ndarray:
        """Map the current galvo voltage to a camera pixel coordinate."""
        x = self.image_width / 2 + (
            self.galvo_x / self.maximum_galvo_voltage
        ) * self.image_width * 0.35
        y = self.image_height / 2 + (
            self.galvo_y / self.maximum_galvo_voltage
        ) * self.image_height * 0.35
        return np.array([y, x], dtype=float)

    def _cell_pixels_yx(self) -> np.ndarray:
        positions = self.cell_positions_um.copy()
        phase = self.time_index * 0.18
        positions[:, 0] += 2.5 * np.sin(phase + np.arange(len(positions)))
        positions[:, 1] += 1.8 * np.cos(phase + np.arange(len(positions)) * 0.7)
        x = self.image_width / 2 + (
            positions[:, 0] - self.stage_x
        ) / self.microns_per_pixel
        y = self.image_height / 2 + (
            positions[:, 1] - self.stage_y
        ) / self.microns_per_pixel
        return np.column_stack((y, x))

    @staticmethod
    def _add_gaussian(
        image: np.ndarray,
        center_y: float,
        center_x: float,
        amplitude: float,
        sigma: float,
    ) -> None:
        radius = max(2, int(np.ceil(4 * sigma)))
        y0 = max(0, int(center_y) - radius)
        y1 = min(image.shape[0], int(center_y) + radius + 1)
        x0 = max(0, int(center_x) - radius)
        x1 = min(image.shape[1], int(center_x) + radius + 1)
        if y0 >= y1 or x0 >= x1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        image[y0:y1, x0:x1] += amplitude * np.exp(
            -((yy - center_y) ** 2 + (xx - center_x) ** 2) / (2 * sigma**2)
        )

    def render_image(self) -> np.ndarray:
        """Render a repeatable brightfield or Raman-alignment image."""
        image = np.full(
            (self.image_height, self.image_width),
            700.0,
            dtype=np.float32,
        )
        contrast = 250.0 + 650.0 * self.focus_strength
        for center_y, center_x in self._cell_pixels_yx():
            self._add_gaussian(image, center_y, center_x, contrast, sigma=8.0)
            self._add_gaussian(image, center_y, center_x, -0.45 * contrast, sigma=3.0)

        if self.channel.upper() == "RM" and any(self.shutters.values()):
            laser_y, laser_x = self.galvo_pixel_yx()
            self._add_gaussian(image, laser_y, laser_x, 12_000.0, sigma=3.0)

        seed = (
            10_000
            + int(round(self.stage_x * 10)) * 3
            + int(round(self.stage_y * 10)) * 5
            + int(round(self.stage_z * 10)) * 7
            + self.time_index * 11
        ) % (2**32)
        noise = np.random.default_rng(seed).normal(0.0, 8.0, image.shape)
        return np.clip(image + noise, 0, 65_535).astype(np.uint16)

    def render_spectrum(
        self,
        voltage_xy: np.ndarray,
        exposure_ms: float,
        sample_index: int = 0,
    ) -> np.ndarray:
        """Render a Raman-like spectrum for one galvo target."""
        self.set_galvo(voltage_xy)
        pixels = np.arange(self.spectrum_pixels, dtype=float)
        exposure_scale = max(float(exposure_ms), 0.1) / 1000.0
        baseline = 120.0 + 0.015 * pixels

        target_yx = self.galvo_pixel_yx()
        cell_pixels = self._cell_pixels_yx()
        nearest_cell = float(
            np.min(np.linalg.norm(cell_pixels - target_yx, axis=1))
        )
        cell_strength = float(np.exp(-0.5 * (nearest_cell / 24.0) ** 2))
        signal_scale = exposure_scale * self.focus_strength * (0.25 + 1.75 * cell_strength)

        spectrum = baseline.copy()
        for center, width, amplitude in (
            (410.0, 13.0, 900.0),
            (855.0, 24.0, 1_600.0),
            (1320.0, 18.0, 1_050.0),
        ):
            spectrum += signal_scale * amplitude * np.exp(
                -0.5 * ((pixels - center) / width) ** 2
            )

        seed = (
            20_000
            + int(round(self.galvo_x * 1000)) * 3
            + int(round(self.galvo_y * 1000)) * 5
            + int(round(self.stage_z * 100)) * 7
            + sample_index * 13
        ) % (2**32)
        noise = np.random.default_rng(seed).normal(
            0.0,
            3.0 + np.sqrt(np.maximum(spectrum, 0.0)) * 0.08,
            spectrum.shape,
        )
        return np.clip(spectrum + noise, 0, None).astype(np.float32)
