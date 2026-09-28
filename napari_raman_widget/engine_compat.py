"""Compatibility helpers for optional RamanEngine features."""

from __future__ import annotations

from importlib import import_module
from inspect import Parameter, signature


def make_pillar_shape_filter(enabled: bool):
    """Create the optional post-Cellpose pillar filter when requested.

    Importing lazily keeps automated selection compatible with older pinned
    ``raman-mda-engine`` revisions while the checkbox is left off.
    """
    if not bool(enabled):
        return None

    try:
        aiming = import_module("raman_mda_engine.aiming")
        filter_type = aiming.PillarShapeFilter
    except (AttributeError, ImportError) as error:
        raise RuntimeError(
            "Installed raman-mda-engine does not support pillar suppression. "
            "Update it to a revision that provides PillarShapeFilter."
        ) from error

    return filter_type()


def pillar_suppression_kwargs(engine_type, enabled: bool) -> dict[str, bool]:
    """Build constructor kwargs for engines with optional pillar suppression.

    Older pinned engine revisions remain usable while the option is off. If a
    user enables the checkbox with such a revision, fail early with an
    actionable message instead of an unexpected-keyword ``TypeError``.
    """
    enabled = bool(enabled)
    try:
        parameters = signature(engine_type).parameters
    except (TypeError, ValueError):
        parameters = {}

    supports_option = "suppress_pillars" in parameters or any(
        parameter.kind is Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if supports_option:
        return {"suppress_pillars": enabled}
    if enabled:
        raise RuntimeError(
            "Installed raman-mda-engine does not support pillar suppression. "
            "Update it to a revision that provides "
            "RamanEngine(suppress_pillars=...)."
        )
    return {}
