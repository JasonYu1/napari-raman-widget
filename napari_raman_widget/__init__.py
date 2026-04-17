"""napari widget for controlling the CNS Raman microscopy rig."""
from .widget import HardwareWidget
from .dataset import load_experiment

__all__ = ["HardwareWidget", "load_experiment"]
__version__ = "0.1.0"