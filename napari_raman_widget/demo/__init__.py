"""Hardware-free demonstration backend for napari-raman-widget."""

from .backend import DemoBackend, create_demo_backend
from .collector import DemoSpectraCollector
from .core import DemoCore, configure_demo_channels
from .daq import DemoDAQ, DemoGalvo
from .transformer import DemoCoordinateTransformer
from .world import DemoWorld

__all__ = [
    "DemoBackend",
    "DemoCore",
    "DemoCoordinateTransformer",
    "DemoDAQ",
    "DemoGalvo",
    "DemoSpectraCollector",
    "DemoWorld",
    "configure_demo_channels",
    "create_demo_backend",
]
