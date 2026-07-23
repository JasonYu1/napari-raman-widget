"""Raman spectra collector for demonstration mode."""

from __future__ import annotations

import numpy as np

from .daq import DemoDAQ
from .world import DemoWorld


class DemoSpectraCollector:
    """Generate spectra with the same point-by-pixel shape as real data."""

    def __init__(self, world: DemoWorld, daq: DemoDAQ) -> None:
        self.world = world
        self.daq = daq

    def collect_spectra_pts(
        self,
        volts: np.ndarray,
        exposure: float,
    ) -> np.ndarray:
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
        return np.asarray(spectra)

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
