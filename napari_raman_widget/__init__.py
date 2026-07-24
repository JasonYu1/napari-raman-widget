"""napari widget for controlling the Raman microscopy rig.

The public GUI objects are imported lazily so hardware-independent modules,
including the demonstration backend, remain usable without importing napari
and Qt first.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DemoWidget", "HardwareWidget", "load_experiment"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Load optional, GUI-dependent public objects on first access."""
    if name == "HardwareWidget":
        from .hardware_widget import HardwareWidget

        return HardwareWidget
    if name == "DemoWidget":
        from .demo_widget import DemoWidget

        return DemoWidget
    if name == "load_experiment":
        from .dataset import load_experiment

        return load_experiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
