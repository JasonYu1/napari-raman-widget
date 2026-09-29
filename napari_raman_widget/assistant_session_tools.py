"""Assistant adapters for optional, session-scoped spectral calibration."""

from __future__ import annotations

import json

from .assistant_plot_tools import resolve_plot_panel
from .spectral_calibration_ui import (
    clear_spectral_calibration,
    load_spectral_calibration_file,
)


def get_spectral_calibration_state(owner):
    calibration = getattr(owner, "spectral_calibration", None)
    return {
        "loaded": calibration is not None,
        "degree": calibration.degree if calibration is not None else None,
        "reference_points": (
            len(calibration.pixel_positions) if calibration is not None else 0
        ),
        "new_plot_axis_default": "pixel",
        "existing_plots_keep_their_calibration": True,
    }


def _load_wavenumber_calibration(owner, inputs):
    unknown = set(inputs) - {"file", "panel_id"}
    if unknown:
        raise ValueError(f"Unknown calibration options: {sorted(unknown)}")
    # Check the target before changing any session or plot state.
    panel = None
    if "panel_id" in inputs:
        if not isinstance(inputs["panel_id"], str) or not inputs["panel_id"]:
            raise ValueError("panel_id must be a nonempty ID from list_plots.")
        panel = resolve_plot_panel(owner, inputs["panel_id"])
        if not hasattr(panel, "show_wavenumber_check"):
            raise ValueError("This panel has no spectral-axis control.")
        if getattr(panel, "_calibrating", False):
            raise ValueError("Finish or cancel the plot's axis calibration first.")
    path = inputs.get("file", owner.spectral_calibration_path.text())
    calibration = load_spectral_calibration_file(owner, path)
    if panel is not None:
        panel.spectral_calibration = calibration
        checkbox = panel.show_wavenumber_check
        checkbox.setEnabled(True)
        checkbox.setToolTip("Use calibrated Raman shift instead of pixels")
        # Refresh through the same signal as the GUI, without opting into
        # wavenumbers or changing an existing user's units preference.
        checkbox.toggled.emit(checkbox.isChecked())
    result = get_spectral_calibration_state(owner)
    result["file"] = owner.spectral_calibration_path.text()
    result["applied_to_panel_id"] = inputs.get("panel_id")
    result["note"] = (
        "New plots still start in pixels. Use configure_plot with "
        "show_wavenumber=true to opt in. Other existing plots are unchanged."
    )
    return json.dumps(result)


def _clear_wavenumber_calibration(owner, inputs):
    if inputs:
        raise ValueError("clear_wavenumber_calibration takes no parameters.")
    clear_spectral_calibration(owner)
    return json.dumps(get_spectral_calibration_state(owner))


SESSION_ACTIONS = [
    {
        "name": "load_wavenumber_calibration",
        "label": "Load optional wavenumber calibration",
        "readonly": True,  # GUI/session settings only; no hardware or writes
        "handler": _load_wavenumber_calibration,
        "params": [
            {
                "name": "file", "attr": None, "kind": "text",
                "description": (
                    "User-selected calibration JSON path. Omit to load the "
                    "current Setup path. Never guess a file path."
                ),
            },
            {
                "name": "panel_id", "attr": None, "kind": "text",
                "description": (
                    "Optional stable ID from list_plots: also attach the "
                    "model to that existing spectrum plot. Without an ID, "
                    "only future plots receive it. Does not switch units."
                ),
            },
        ],
        "description": (
            "Load and validate an optional pixel-to-wavenumber JSON, shared "
            "with the Setup loader. New plots always start in pixels. Can "
            "explicitly attach to one existing plot; does not modify others."
        ),
    },
    {
        "name": "clear_wavenumber_calibration",
        "label": "Clear optional wavenumber calibration",
        "readonly": True,
        "handler": _clear_wavenumber_calibration,
        "params": [],
        "description": (
            "Clear the session calibration for future plots; delete no files. "
            "Existing plots retain their model; configure_plot can switch "
            "those plots back to pixels with show_wavenumber=false."
        ),
    },
]
