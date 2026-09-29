"""Calibration tools for Raman targeting and stage alignment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .models import (
    apply_vandermonde,
    apply_vandermonde_model,
    fit_vandermonde,
    load_vandermonde_model,
    save_vandermonde_model,
)

if TYPE_CHECKING:
    from .calibrator import Calibrator, ManualImageSelector
    from .coordinate_transform import CoordTransformer
    from .stage_points import StagePointPicker


def __getattr__(name: str) -> Any:
    """Load interactive calibration classes only when they are requested."""
    if name in {"Calibrator", "ManualImageSelector"}:
        from .calibrator import Calibrator, ManualImageSelector

        value = {
            "Calibrator": Calibrator,
            "ManualImageSelector": ManualImageSelector,
        }[name]
    elif name == "CoordTransformer":
        from .coordinate_transform import CoordTransformer

        value = CoordTransformer
    elif name == "StagePointPicker":
        from .stage_points import StagePointPicker

        value = StagePointPicker
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public exports in interactive discovery."""
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "Calibrator",
    "CoordTransformer",
    "ManualImageSelector",
    "StagePointPicker",
    "apply_vandermonde",
    "apply_vandermonde_model",
    "fit_vandermonde",
    "load_vandermonde_model",
    "save_vandermonde_model",
]
