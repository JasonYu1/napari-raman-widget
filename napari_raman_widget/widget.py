"""Backward-compatible import for the real-hardware Raman widget.

New code should import :class:`HardwareWidget` from ``hardware_widget``.
The simulator lives independently in ``demo_widget``.
"""

from .hardware_widget import HardwareWidget

__all__ = ["HardwareWidget"]
