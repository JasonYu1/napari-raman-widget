"""DAQ and galvo simulation for demonstration mode."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from .world import DemoWorld


class DemoGalvoTiming:
    def __init__(self) -> None:
        self.rate: float | None = None
        self.sample_mode: Any | None = None

    def cfg_samp_clk_timing(self, rate: float, sample_mode: Any = None) -> None:
        self.rate = float(rate)
        self.sample_mode = sample_mode


class DemoGalvo:
    def __init__(self, world: DemoWorld) -> None:
        self.world = world
        self.running = False
        self.timing = DemoGalvoTiming()
        self.out_stream = SimpleNamespace(output_buf_size=1000)

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def write(self, voltage_xy: np.ndarray) -> None:
        self.world.set_galvo(voltage_xy)


class DemoDAQ:
    """DAQ object exposing the public and private galvo attributes in use."""

    def __init__(self, world: DemoWorld) -> None:
        self.galvo = DemoGalvo(world)
        self._galvo = self.galvo
