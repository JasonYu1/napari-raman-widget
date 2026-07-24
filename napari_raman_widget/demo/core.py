"""Micro-Manager-like core for demonstration mode."""

from __future__ import annotations

from typing import Any

import numpy as np

from .world import DemoWorld


def configure_demo_channels(core: Any) -> None:
    """Replace an MM demo config's channels with deterministic BF/RM presets."""
    group = "Channel"
    if group in {str(name) for name in core.getAvailableConfigGroups()}:
        core.deleteConfigGroup(group)
    # Empty configurations are indistinguishable to MMCore: selecting RM
    # would immediately report BF again.  Use two harmless DemoCamera gain
    # values so each preset has a genuine, trackable hardware state.  The
    # FakeDemoCamera override produces the actual image data.
    camera = str(core.getCameraDevice())
    gain = float(core.getProperty(camera, "Gain"))
    core.defineConfig(group, "BF", camera, "Gain", f"{gain:g}")
    core.defineConfig(group, "RM", camera, "Gain", f"{gain + 1:g}")
    core.setChannelGroup(group)
    core.setConfig(group, "BF")


class DemoMDARunner:
    """Minimal MDA state exposed through ``core.mda``."""

    def __init__(self) -> None:
        self.engine: Any | None = None
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class DemoCore:
    """Subset of the CMMCorePlus interface used by the widget."""

    def __init__(self, world: DemoWorld) -> None:
        self.world = world
        self.mda = DemoMDARunner()
        self._last_image = world.render_image()
        self._continuous = False
        self._loaded_config: str | None = None
        self._last_sequence: Any | None = None

    def waitForSystem(self) -> None:
        return None

    def getXYPosition(self) -> tuple[float, float]:
        return self.world.stage_x, self.world.stage_y

    def getXPosition(self) -> float:
        return self.world.stage_x

    def getYPosition(self) -> float:
        return self.world.stage_y

    def setXYPosition(self, x: float, y: float) -> None:
        self.world.stage_x = float(x)
        self.world.stage_y = float(y)

    def getPosition(self) -> float:
        return self.world.stage_z

    def setPosition(self, z: float) -> None:
        self.setZPosition(z)

    def setZPosition(self, z: float) -> None:
        self.world.stage_z = float(z)

    def getImageWidth(self) -> int:
        return self.world.image_width

    def getImageHeight(self) -> int:
        return self.world.image_height

    def setExposure(self, exposure: float) -> None:
        self.world.exposure_ms = float(exposure)

    def getExposure(self) -> float:
        return self.world.exposure_ms

    def setConfig(self, group: str, configuration: str) -> None:
        if group.lower() == "channel":
            self.world.channel = str(configuration)

    def getAvailableConfigs(self, group: str) -> tuple[str, ...]:
        if group.lower() == "channel":
            return ("BF", "RM")
        return ()

    def snap(self) -> np.ndarray:
        self._last_image = self.world.render_image()
        return self._last_image.copy()

    def snapImage(self) -> None:
        self._last_image = self.world.render_image()

    def getImage(self) -> np.ndarray:
        return self._last_image.copy()

    def startContinuousSequenceAcquisition(self, interval_ms: float = 0) -> None:
        self._continuous = True

    def stopSequenceAcquisition(self) -> None:
        self._continuous = False

    def loadSystemConfiguration(self, config_path: str) -> None:
        self._loaded_config = str(config_path)

    def unloadAllDevices(self) -> None:
        self._continuous = False
        self.mda.engine = None

    def register_mda_engine(self, engine: Any) -> None:
        self.mda.engine = engine

    def run_mda(self, sequence: Any) -> None:
        """Record an MDA sequence and establish its initial virtual state."""
        self.mda.cancelled = False
        self._last_sequence = sequence
        positions = getattr(sequence, "stage_positions", ()) or ()
        if positions:
            position = positions[0]
            if getattr(position, "x", None) is not None and getattr(position, "y", None) is not None:
                self.setXYPosition(position.x, position.y)
        self.snapImage()
