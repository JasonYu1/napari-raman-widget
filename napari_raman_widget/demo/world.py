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
    microns_per_pixel: float = 1.0
    maximum_galvo_voltage: float = 1.8
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
    time_index: int = 0
    sample_width_um: float = 8000.0
    sample_height_um: float = 6000.0
    number_of_cells: int = 5200
    cell_positions_um: np.ndarray = field(init=False, repr=False)
    cell_radii_px: np.ndarray = field(init=False, repr=False)
    cell_colors_rgb: np.ndarray = field(init=False, repr=False)
    laser_pixel_yx: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(20260723)
        self.cell_positions_um = rng.uniform(
            low=(-self.sample_width_um / 2, -self.sample_height_um / 2),
            high=(self.sample_width_um / 2, self.sample_height_um / 2),
            size=(self.number_of_cells, 2),
        )
        self.cell_radii_px = rng.uniform(
            3.0, 9.5, size=self.number_of_cells
        )
        palette = np.array(
            [
                (76, 201, 240),
                (114, 222, 138),
                (255, 190, 92),
                (232, 112, 190),
                (145, 128, 255),
                (255, 112, 105),
            ],
            dtype=np.float32,
        )
        self.cell_colors_rgb = palette[
            rng.integers(0, len(palette), size=self.number_of_cells)
        ]

    @property
    def img_shape(self) -> np.ndarray:
        """Camera shape expected by ``mda_simulator.FakeDemoCamera``."""
        return np.array([self.image_height, self.image_width], dtype=int)

    def snap_img(
        self,
        xy: tuple[float, float],
        c: int = 0,
        z: float = 0,
        exposure: float = 1,
    ) -> np.ndarray:
        """Render from the stage state owned by the singleton MMCore."""
        self.stage_x, self.stage_y = map(float, xy)
        self.stage_z = float(z)
        self.exposure_ms = float(exposure)
        channels = ("BF", "RM")
        channel_index = int(c)
        self.channel = (
            channels[channel_index]
            if 0 <= channel_index < len(channels)
            else "BF"
        )
        return self.render_image()

    def increment_time(self, delta_t: float = 1) -> None:
        """Advance simulated cell motion for ``FakeDemoCamera``."""
        self.time_index += float(delta_t)

    @property
    def focus_strength(self) -> float:
        """Return a zero-to-one signal strength for the current Z position."""
        distance = self.stage_z - self.focus_z
        return float(np.exp(-0.5 * (distance / 3.0) ** 2))

    def set_galvo(self, voltage_xy: np.ndarray) -> None:
        voltage_xy = np.asarray(voltage_xy, dtype=float).reshape(2)
        self.galvo_x = float(voltage_xy[0])
        self.galvo_y = float(voltage_xy[1])
        self.laser_pixel_yx = self.galvo_pixel_yx()

    def set_laser_pixel(self, point_yx: np.ndarray) -> None:
        """Aim the visible fake Raman laser directly at an image pixel."""
        self.laser_pixel_yx = np.asarray(point_yx, dtype=float).reshape(2)

    def galvo_pixel_yx(self) -> np.ndarray:
        """Invert the MDA transform from normalized YX to simulated volts."""
        # RamanEngine passes normalized napari points in YX order through
        # BF_to_volts(..., max_volts=1.8).  Preserve that order here so the
        # MDA and direct-pixel collection paths address the identical sample
        # coordinate over the full field of view.
        y = self.image_height / 2 + (
            self.galvo_x / self.maximum_galvo_voltage
        ) * self.image_height / 2
        x = self.image_width / 2 + (
            self.galvo_y / self.maximum_galvo_voltage
        ) * self.image_width / 2
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

    def snap_points_to_cell_centers(self, points_yx: np.ndarray) -> np.ndarray:
        """Refine detected demo-cell points to the simulated sphere centers."""
        points = np.asarray(points_yx, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, 2)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points_yx must have shape (N, 2).")
        if len(points) == 0:
            return points.copy()

        cell_pixels = self._cell_pixels_yx()
        visible = (
            (cell_pixels[:, 0] >= 0)
            & (cell_pixels[:, 0] < self.image_height)
            & (cell_pixels[:, 1] >= 0)
            & (cell_pixels[:, 1] < self.image_width)
        )
        visible_centers = cell_pixels[visible]
        if len(visible_centers) == 0:
            return points.copy()

        squared_distances = np.sum(
            (points[:, None, :] - visible_centers[None, :, :]) ** 2,
            axis=2,
        )
        nearest_indices = np.argmin(squared_distances, axis=1)
        return visible_centers[nearest_indices].copy()

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

    @staticmethod
    def _sphere_patch(
        center_y: float,
        center_x: float,
        radius: float,
        shape: tuple[int, int],
    ) -> tuple[slice, slice, np.ndarray, np.ndarray] | None:
        """Return a clipped solid-sphere mask and directional light field."""
        edge = max(2, int(np.ceil(radius + 1)))
        y0 = max(0, int(np.floor(center_y)) - edge)
        y1 = min(shape[0], int(np.floor(center_y)) + edge + 1)
        x0 = max(0, int(np.floor(center_x)) - edge)
        x1 = min(shape[1], int(np.floor(center_x)) + edge + 1)
        if y0 >= y1 or x0 >= x1:
            return None
        yy, xx = np.mgrid[y0:y1, x0:x1]
        ny = (yy - center_y) / max(radius, 1.0)
        nx = (xx - center_x) / max(radius, 1.0)
        radius_squared = nx**2 + ny**2
        mask = radius_squared <= 1.0
        surface_z = np.sqrt(np.clip(1.0 - radius_squared, 0.0, 1.0))
        # Diffuse spherical shading plus a small upper-left specular highlight.
        light = 0.25 + 0.72 * surface_z
        light += 0.30 * np.exp(-((nx + 0.32) ** 2 + (ny + 0.32) ** 2) / 0.075)
        return slice(y0, y1), slice(x0, x1), mask, light

    def _visible_cells(self):
        pixels = self._cell_pixels_yx()
        margin = float(np.max(self.cell_radii_px)) + 2.0
        visible = (
            (pixels[:, 0] >= -margin)
            & (pixels[:, 0] < self.image_height + margin)
            & (pixels[:, 1] >= -margin)
            & (pixels[:, 1] < self.image_width + margin)
        )
        return (
            pixels[visible],
            self.cell_radii_px[visible],
            self.cell_colors_rgb[visible],
        )

    def render_image(self) -> np.ndarray:
        """Render a repeatable brightfield or Raman-alignment image."""
        image = np.full(
            (self.image_height, self.image_width),
            700.0,
            dtype=np.float32,
        )
        contrast = 350.0 + 900.0 * self.focus_strength
        if self.channel.upper() == "RM":
            image.fill(100.0)
            contrast *= 1.8
        visible_pixels, visible_radii, visible_colors = self._visible_cells()
        for (center_y, center_x), radius, color in zip(
            visible_pixels, visible_radii, visible_colors
        ):
            patch = self._sphere_patch(
                center_y, center_x, float(radius), image.shape
            )
            if patch is None:
                continue
            ys, xs, mask, light = patch
            luminance = float(np.dot(color, (0.2126, 0.7152, 0.0722)) / 170.0)
            local = image[ys, xs]
            local[mask] += contrast * luminance * light[mask]

        if self.channel.upper() == "RM" and self.laser_pixel_yx is not None:
            laser_y, laser_x = self.laser_pixel_yx
            self._add_gaussian(image, laser_y, laser_x, 12_000.0, sigma=3.0)

        seed = (
            10_000
            + int(round(self.stage_x * 10)) * 3
            + int(round(self.stage_y * 10)) * 5
            + int(round(self.stage_z * 10)) * 7
            + int(round(self.time_index * 1000)) * 11
        ) % (2**32)
        noise = np.random.default_rng(seed).normal(0.0, 8.0, image.shape)
        return np.clip(image + noise, 0, 65_535).astype(np.uint16)

    def render_color_image(self) -> np.ndarray:
        """Render a colored 3-D preview while MM retains a 2-D camera frame."""
        is_raman = self.channel.upper() == "RM"
        background = np.array((7, 7, 14) if is_raman else (28, 32, 38))
        image = np.empty((self.image_height, self.image_width, 3), dtype=np.float32)
        image[...] = background
        visible_pixels, visible_radii, visible_colors = self._visible_cells()
        for (center_y, center_x), radius, color in zip(
            visible_pixels, visible_radii, visible_colors
        ):
            patch = self._sphere_patch(
                center_y, center_x, float(radius), image.shape[:2]
            )
            if patch is None:
                continue
            ys, xs, mask, light = patch
            local = image[ys, xs]
            shaded = color[None, None, :] * light[..., None]
            if is_raman:
                shaded *= 1.12
            local[mask] = shaded[mask]

        if is_raman and self.laser_pixel_yx is not None:
            laser = np.zeros(image.shape[:2], dtype=np.float32)
            laser_y, laser_x = self.laser_pixel_yx
            self._add_gaussian(laser, laser_y, laser_x, 255.0, sigma=3.0)
            image[..., 0] += laser
            image[..., 1] += laser * 0.18
        return np.clip(image, 0, 255).astype(np.uint8)

    def render_spectrum(
        self,
        voltage_xy: np.ndarray,
        exposure_ms: float,
        sample_index: int = 0,
    ) -> np.ndarray:
        """Render a Raman-like spectrum for one galvo target."""
        self.set_galvo(voltage_xy)
        return self.render_spectrum_at_pixel(
            self.galvo_pixel_yx(), exposure_ms, sample_index=sample_index
        )

    def render_spectrum_at_pixel(
        self,
        target_yx: np.ndarray,
        exposure_ms: float,
        sample_index: int = 0,
    ) -> np.ndarray:
        """Render a spectrum at a camera pixel, bypassing galvo calibration."""
        target_yx = np.asarray(target_yx, dtype=float).reshape(2)
        pixels = np.arange(self.spectrum_pixels, dtype=float)
        exposure_scale = max(float(exposure_ms), 0.1) / 1000.0
        baseline = np.full(self.spectrum_pixels, 35.0 * exposure_scale + 8.0)

        cell_pixels = self._cell_pixels_yx()
        distances = np.linalg.norm(cell_pixels - target_yx, axis=1)
        nearest_index = int(np.argmin(distances))
        nearest_cell = float(distances[nearest_index])
        # Only a laser position lying on a visible cell produces Raman peaks.
        # The surrounding field contributes detector background and noise only.
        hit_radius = max(5.0, float(self.cell_radii_px[nearest_index]) * 1.5)
        cell_strength = 1.0 if nearest_cell <= hit_radius else 0.0
        signal_scale = exposure_scale * self.focus_strength * cell_strength

        grooves_per_mm = {1: 300.0, 2: 600.0, 3: 1200.0}[self.grating]
        dispersion_scale = grooves_per_mm / 300.0
        detector_center = (self.spectrum_pixels - 1) / 2
        wavelength_offset = (self.wavelength_nm - 700.0) * 3.0 * dispersion_scale
        grating_efficiency = {1: 1.0, 2: 0.82, 3: 0.62}[self.grating]

        spectrum = baseline.copy()
        for center, width, amplitude in (
            (410.0, 13.0, 900.0),
            (855.0, 24.0, 1_600.0),
            (1320.0, 18.0, 1_050.0),
        ):
            displayed_center = (
                detector_center
                + (center - detector_center) * dispersion_scale
                - wavelength_offset
            )
            displayed_width = max(2.0, width / np.sqrt(dispersion_scale))
            spectrum += signal_scale * grating_efficiency * amplitude * np.exp(
                -0.5 * ((pixels - displayed_center) / displayed_width) ** 2
            )

        seed = (
            20_000
            + int(round(target_yx[1] * 10)) * 3
            + int(round(target_yx[0] * 10)) * 5
            + int(round(self.stage_z * 100)) * 7
            + sample_index * 13
        ) % (2**32)
        noise = np.random.default_rng(seed).normal(
            0.0,
            3.0 + np.sqrt(np.maximum(spectrum, 0.0)) * 0.08,
            spectrum.shape,
        )
        return np.clip(spectrum + noise, 0, None).astype(np.float32)
