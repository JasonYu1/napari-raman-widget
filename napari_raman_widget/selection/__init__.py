"""Point selection tools for Raman acquisition."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .masks import (
    add_mask_with_hole,
    find_clear_center_point,
    get_n_most_centered_coms,
)
from .refine import (
    refine_cell_source_points,
    refine_points_to_label_centers,
)

if TYPE_CHECKING:
    from .automatic import automated_point_selections
    from .grid import grid_point_selections
    from .layers import create_point_sources
    from .manual import (
        center_manual_selections,
        manual_point_selections,
    )

_LAZY_EXPORTS = {
    "automated_point_selections": ("automatic", "automated_point_selections"),
    "grid_point_selections": ("grid", "grid_point_selections"),
    "create_point_sources": ("layers", "create_point_sources"),
    "center_manual_selections": ("manual", "center_manual_selections"),
    "manual_point_selections": ("manual", "manual_point_selections"),
}


def __getattr__(name: str) -> Any:
    """Load engine-dependent selection helpers only when requested."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error

    value = getattr(
        import_module(f"{__name__}.{module_name}"),
        attribute_name,
    )
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public exports in interactive discovery."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

__all__ = [
    "add_mask_with_hole",
    "automated_point_selections",
    "center_manual_selections",
    "create_point_sources",
    "find_clear_center_point",
    "get_n_most_centered_coms",
    "grid_point_selections",
    "manual_point_selections",
    "refine_cell_source_points",
    "refine_points_to_label_centers",
]
