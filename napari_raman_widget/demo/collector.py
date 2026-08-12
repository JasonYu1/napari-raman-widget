"""Raman spectra collector for demonstration mode."""

from __future__ import annotations

import numpy as np

from .daq import DemoDAQ
from .world import DemoWorld


class DemoSpectraCollector:
    """Generate spectra with the same point-by-pixel shape as real data."""

    detector_rows = 256

    def __init__(self, world: DemoWorld, daq: DemoDAQ, core=None) -> None:
        self.world = world
        self.daq = daq
        self.core = core

    def _sync_stage(self) -> None:
        """Read the authoritative stage state from the shared core."""
        if self.core is None:
            return
        try:
            x, y = self.core.getXYPosition()
            self.world.stage_x = float(x)
            self.world.stage_y = float(y)
            self.world.stage_z = float(self.core.getPosition())
        except Exception:
            return

    def collect_spectra_pts(
        self,
        volts: np.ndarray,
        exposure: float,
        read_mode: str = "fvb",
        track_center: int | None = None,
        track_height: int | None = None,
    ) -> np.ndarray:
        self._sync_stage()
        volts = np.asarray(volts, dtype=float)
        if volts.ndim == 1:
            volts = volts.reshape(1, 2)
        if volts.ndim != 2 or volts.shape[1] != 2:
            raise ValueError("volts must have shape (N, 2).")
        spectra = []
        for index, voltage_xy in enumerate(volts):
            self.daq.galvo.write(voltage_xy)
            spectra.append(
                self.world.render_spectrum(voltage_xy, exposure, sample_index=index)
            )
        return self._apply_read_mode(
            np.asarray(spectra), read_mode, track_center, track_height
        )

    def collect_spectra_image_points(
        self,
        points_yx: np.ndarray,
        exposure: float,
        read_mode: str = "fvb",
        track_center: int | None = None,
        track_height: int | None = None,
    ) -> np.ndarray:
        """Collect directly at napari image points without galvo calibration."""
        self._sync_stage()
        points_yx = np.asarray(points_yx, dtype=float)
        if points_yx.ndim == 1:
            points_yx = points_yx.reshape(1, 2)
        if points_yx.ndim != 2 or points_yx.shape[1] != 2:
            raise ValueError("points_yx must have shape (N, 2).")
        if len(points_yx):
            self.world.set_laser_pixel(points_yx[-1])
        spectra = np.asarray(
            [
                self.world.render_spectrum_at_pixel(
                    point_yx, exposure, sample_index=index
                )
                for index, point_yx in enumerate(points_yx)
            ]
        )
        return self._apply_read_mode(
            spectra, read_mode, track_center, track_height
        )

    def _apply_read_mode(
        self,
        spectra: np.ndarray,
        read_mode: str,
        track_center: int | None,
        track_height: int | None,
    ) -> np.ndarray:
        """Mirror the Andor read-mode shapes for demonstration collection."""
        if not isinstance(read_mode, str):
            raise TypeError("read_mode must be 'fvb', 'single_track', or 'image'")
        read_mode = read_mode.lower()
        if read_mode not in {"fvb", "single_track", "image"}:
            raise ValueError("read_mode must be 'fvb', 'single_track', or 'image'")
        if read_mode == "fvb":
            return spectra
        if read_mode == "single_track":
            if track_center is None or track_height is None:
                raise ValueError(
                    "track_center and track_height are required for "
                    "single_track mode"
                )
            if not 0 <= int(track_center) < self.detector_rows:
                raise ValueError(
                    f"track_center must be between 0 and {self.detector_rows - 1}"
                )
            if not 2 <= int(track_height) <= self.detector_rows:
                raise ValueError(
                    f"track_height must be between 2 and {self.detector_rows}"
                )
            return spectra

        rows = np.arange(self.detector_rows, dtype=float)
        vertical_profile = 0.04 + np.exp(
            -0.5 * ((rows - (self.detector_rows - 1) / 2) / 18.0) ** 2
        )
        return spectra[:, None, :] * vertical_profile[None, :, None]

    def get_wavelength(self) -> float:
        return self.world.wavelength_nm

    def set_wavelength(self, wavelength: float) -> None:
        self.world.wavelength_nm = float(wavelength)

    def get_number_gratings(self) -> int:
        return 3

    def get_grating(self) -> int:
        return self.world.grating

    def set_grating(self, grating: int) -> None:
        grating = int(grating)
        if grating not in (1, 2, 3):
            raise ValueError("grating must be 1, 2, or 3.")
        self.world.grating = grating

    def get_grating_info(self, grating: int) -> tuple[float, float, float, float]:
        information = {
            1: (300.0, 500.0, 0.0, 0.0),
            2: (600.0, 750.0, 0.0, 0.0),
            3: (1200.0, 1000.0, 0.0, 0.0),
        }
        grating = int(grating)
        return information[grating]
