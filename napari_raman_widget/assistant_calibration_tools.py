"""Assistant tools for inspecting calibration results and progress.

These handlers only read or select data already present in GUI panels.  They
never acquire data, move hardware, save files, or fit calibration models.
"""

from __future__ import annotations

import json
from numbers import Real
from typing import Any

import numpy as np

from .assistant_plot_tools import resolve_plot_panel
from .log_window import LogWindow
from .plot_windows import CalibrationPlotWindow, SpectrumWindow


DEFAULT_TRACE_LIMIT = 2048
MAX_TRACE_LIMIT = 4096
DEFAULT_POINT_LIMIT = 50
MAX_POINT_LIMIT = 200
DEFAULT_LOG_TAIL_CHARS = 2000
MAX_LOG_TAIL_CHARS = 12000

CALIBRATION_CONTROL_WIDGET_ATTRIBUTES = {
    "degree": "calibration_degree_input",
}


def _parameter(name, kind, description, *, schema=None):
    parameter = {
        "name": name,
        "attr": None,
        "kind": kind,
        "description": description,
    }
    if schema is not None:
        parameter["schema"] = schema
    return parameter


def _validate_input(tool_input, allowed, tool_name):
    if tool_input is None:
        return {}
    if not isinstance(tool_input, dict):
        raise ValueError(f"{tool_name} input must be an object")
    unknown = sorted(set(tool_input) - set(allowed))
    if unknown:
        raise ValueError(f"unknown {tool_name} option(s): {', '.join(unknown)}")
    return tool_input


def _requested_panel_id(tool_input):
    if "panel_id" not in tool_input:
        return None
    panel_id = tool_input["panel_id"]
    if not isinstance(panel_id, str) or not panel_id.strip():
        raise ValueError("panel_id must be a non-empty string")
    return panel_id


def _as_bool(value):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"expected a boolean, got {value!r}")
    return bool(value)


def _bounded_int(value, *, minimum, maximum, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be an integer")
    numeric = float(value)
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be a finite integer")
    result = int(numeric)
    if result < minimum or result > maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return result


def _json_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _array_summary(values):
    if values is None:
        return {"available": False}
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return {"available": False}
    flat = array.reshape(-1)
    finite = flat[np.isfinite(flat)]
    summary = {
        "available": bool(array.size),
        "shape": list(array.shape),
        "value_count": int(array.size),
        "finite_count": int(finite.size),
        "nonfinite_count": int(array.size - finite.size),
    }
    if finite.size:
        summary.update(
            {
                "minimum": float(np.min(finite)),
                "maximum": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "median": float(np.median(finite)),
            }
        )
    return summary


def _panel_id(panel):
    return getattr(panel, "_plot_panel_id", None)


def _point_coordinates(panel, index):
    point = panel._point_positions[index]
    result = {
        "image_x": _json_number(point[0]),
        "image_y": _json_number(point[1]),
    }
    laser_positions = panel._dataset_array("laser_pos")
    if (
        laser_positions is not None
        and laser_positions.ndim >= 2
        and index < laser_positions.shape[0]
        and laser_positions.shape[1] >= 2
    ):
        result.update(
            {
                "galvo_x_v": _json_number(laser_positions[index, 0]),
                "galvo_y_v": _json_number(laser_positions[index, 1]),
            }
        )
    return result


def _point_inventory(panel, limit):
    selectable = {int(index) for index in panel._plotted_indices}
    point_count = len(panel._point_positions)
    items = []
    for index in range(min(point_count, limit)):
        items.append(
            {
                "point_index": index + 1,
                "selectable": index in selectable,
                "has_spectrum": panel._spectrum_for_point(index) is not None,
                **_point_coordinates(panel, index),
            }
        )
    return {
        "total_count": point_count,
        "returned_count": len(items),
        "truncated": point_count > limit,
        "points": items,
    }


def inspect_calibration_result(owner, tool_input):
    """Inspect or select one point in an existing calibration result panel."""
    tool_input = _validate_input(
        tool_input,
        {
            "panel_id",
            "point_index",
            "include_point_inventory",
            "point_limit",
            "include_trace",
            "trace_limit",
        },
        "calibration-result",
    )
    panel_id = _requested_panel_id(tool_input)
    panel = resolve_plot_panel(owner, panel_id)
    if not isinstance(panel, CalibrationPlotWindow):
        raise ValueError(
            "selected plot panel is not a calibration result; choose a "
            "Calibration panel_id or make that tab current"
        )

    include_trace = (
        _as_bool(tool_input["include_trace"])
        if "include_trace" in tool_input
        else False
    )
    include_inventory = (
        _as_bool(tool_input["include_point_inventory"])
        if "include_point_inventory" in tool_input
        else False
    )
    trace_limit = (
        _bounded_int(
            tool_input["trace_limit"],
            minimum=1,
            maximum=MAX_TRACE_LIMIT,
            name="trace_limit",
        )
        if "trace_limit" in tool_input
        else DEFAULT_TRACE_LIMIT
    )
    point_limit = (
        _bounded_int(
            tool_input["point_limit"],
            minimum=1,
            maximum=MAX_POINT_LIMIT,
            name="point_limit",
        )
        if "point_limit" in tool_input
        else DEFAULT_POINT_LIMIT
    )
    if "trace_limit" in tool_input and not include_trace:
        raise ValueError("trace_limit requires include_trace=true")
    if "point_limit" in tool_input and not include_inventory:
        raise ValueError(
            "point_limit requires include_point_inventory=true"
        )

    point_index = None
    if "point_index" in tool_input:
        if not len(panel._point_positions):
            raise ValueError("the calibration result contains no points")
        point_index = _bounded_int(
            tool_input["point_index"],
            minimum=1,
            maximum=len(panel._point_positions),
            name="point_index",
        )

    previous_index = panel.selected_index
    if point_index is not None:
        if not panel.select_point(point_index - 1):
            raise ValueError(
                f"point_index {point_index} is not a finite selectable "
                "calibration acquisition"
            )

    selected_index = panel.selected_index
    result: dict[str, Any] = {
        "panel_id": _panel_id(panel),
        "panel_title": panel.windowTitle(),
        "point_count": int(len(panel._point_positions)),
        "selectable_point_count": int(len(panel._plotted_indices)),
        "selected_point_index": (
            None if selected_index is None else int(selected_index) + 1
        ),
        "selection_changed": selected_index != previous_index,
        "selection_metadata": panel.selection_label.text(),
    }

    if selected_index is None:
        result["stored_spectra"] = {"available": False}
        result["displayed_spectrum"] = {"available": False}
    else:
        result.update(_point_coordinates(panel, selected_index))
        stored = None
        if (
            panel.raw_spectra is not None
            and panel.raw_spectra.ndim >= 1
            and selected_index < panel.raw_spectra.shape[0]
        ):
            stored = panel.raw_spectra[selected_index]
        result["stored_spectra"] = _array_summary(stored)
        result["displayed_spectrum"] = _array_summary(
            panel.selected_spectrum
        )

        if include_trace:
            spectrum = panel.selected_spectrum
            if spectrum is None:
                result["trace"] = {
                    "available": False,
                    "reason": "the selected point has no displayable spectrum",
                }
            else:
                spectrum = np.asarray(spectrum, dtype=float).reshape(-1)
                returned = min(len(spectrum), trace_limit)
                if panel.spectrum_line is None:
                    axis = np.arange(len(spectrum), dtype=float)
                else:
                    axis = np.asarray(
                        panel.spectrum_line.get_xdata(), dtype=float
                    ).reshape(-1)
                result["trace"] = {
                    "available": True,
                    "source": (
                        "displayed spectrum; repeated stored traces are "
                        "combined by CalibrationPlotWindow"
                    ),
                    "axis_label": panel.ax_spectrum.get_xlabel(),
                    "total_length": int(len(spectrum)),
                    "returned_length": int(returned),
                    "truncated": len(spectrum) > returned,
                    "x": [_json_number(value) for value in axis[:returned]],
                    "intensity": [
                        _json_number(value) for value in spectrum[:returned]
                    ],
                }

    if include_inventory:
        result["point_inventory"] = _point_inventory(panel, point_limit)

    return json.dumps(result, sort_keys=True, separators=(",", ":"))


def _newest_calibration_log(owner):
    for panel in reversed(list(getattr(owner, "_plot_windows", ()) or ())):
        if (
            isinstance(panel, LogWindow)
            and "calibration" in panel.windowTitle().casefold()
        ):
            return panel
    return None


def _resolve_calibration_log(owner, panel_id):
    panel = resolve_plot_panel(owner, panel_id)
    if panel_id is None and not (
        isinstance(panel, LogWindow)
        and "calibration" in panel.windowTitle().casefold()
    ):
        panel = _newest_calibration_log(owner)
        if panel is None:
            raise ValueError(
                "there is no open calibration log"
            )
    if not isinstance(panel, LogWindow) or (
        "calibration" not in panel.windowTitle().casefold()
    ):
        raise ValueError(
            "selected plot panel is not a calibration log; choose a "
            "Calibration log panel_id or make that tab current"
        )
    return panel


def _progress_payload(log, *, tail_chars):
    progress_available = not log.progress_widget.isHidden()
    result: dict[str, Any] = {
        "panel_id": _panel_id(log),
        "panel_title": log.windowTitle(),
        "progress_available": progress_available,
        # Keep the response key for clients that used the former synchronous
        # implementation, but describe the worker-backed behavior accurately.
        "synchronous_constraint": (
            "Calibration runs in a background worker. This is the latest "
            "worker-reported progress delivered to the UI, not a live "
            "detector query. Completed counts advance at real acquisition "
            "or processing boundaries; no intermediate progress is estimated."
        ),
    }
    if progress_available:
        minimum = int(log.progress_bar.minimum())
        maximum = int(log.progress_bar.maximum())
        value = int(log.progress_bar.value())
        label = log.progress_label.text()
        indeterminate = minimum == 0 and maximum == 0
        lifecycle_state = getattr(log, "_progress_state", None)
        if lifecycle_state in {"idle", "running", "stopping", "stopped", "complete", "failed"}:
            # A full counter can still mean "saving" or "restoring hardware".
            # Only the worker's terminal acknowledgement marks completion.
            state = "busy" if lifecycle_state == "running" else lifecycle_state
        else:
            # Older log panels do not expose an explicit lifecycle state.
            failed = log.progress_bar.format() == "Failed"
            waiting = label == "Waiting to start"
            complete = not failed and not indeterminate and value >= maximum
            busy = not failed and not waiting and (
                indeterminate or (value < maximum)
            )
            state = (
                "failed"
                if failed
                else "complete"
                if complete
                else "busy"
                if busy
                else "idle"
            )
        result["progress"] = {
            "state": state,
            "label": label,
            "indeterminate": indeterminate,
            "completed": None if indeterminate else value,
            "total": None if indeterminate else maximum,
            "display_format": log.progress_bar.format(),
        }

    text = log.text.toPlainText()
    result["log"] = {
        "total_characters": len(text),
        "returned_characters": min(len(text), tail_chars),
        "truncated": len(text) > tail_chars,
        "recent_text": text[-tail_chars:] if tail_chars else "",
    }
    return result


def query_calibration_progress(owner, tool_input):
    """Read visible calibration-log progress and a bounded recent log tail."""
    tool_input = _validate_input(
        tool_input, {"panel_id", "tail_chars"}, "calibration-progress"
    )
    panel_id = _requested_panel_id(tool_input)
    tail_chars = (
        _bounded_int(
            tool_input["tail_chars"],
            minimum=0,
            maximum=MAX_LOG_TAIL_CHARS,
            name="tail_chars",
        )
        if "tail_chars" in tool_input
        else DEFAULT_LOG_TAIL_CHARS
    )
    log = _resolve_calibration_log(owner, panel_id)
    return json.dumps(
        _progress_payload(log, tail_chars=tail_chars),
        sort_keys=True,
        separators=(",", ":"),
    )


def control_spectral_axis_calibration(owner, tool_input):
    """Start, cancel, or inspect the existing manual spectral-axis workflow."""
    tool_input = _validate_input(
        tool_input,
        {"panel_id", "operation", "degree"},
        "spectral-axis-calibration",
    )
    panel_id = _requested_panel_id(tool_input)
    panel = resolve_plot_panel(owner, panel_id)
    if not isinstance(panel, SpectrumWindow):
        raise ValueError(
            "selected plot panel is not a Spectrum panel; choose a Spectrum "
            "panel_id or make that tab current"
        )

    operation = tool_input.get("operation", "status")
    if not isinstance(operation, str):
        raise ValueError("operation must be status, start, or cancel")
    operation = operation.strip().lower()
    if operation not in {"status", "start", "cancel"}:
        raise ValueError("operation must be status, start, or cancel")

    if operation == "start":
        if panel._calibrating:
            raise ValueError(
                "spectral-axis calibration is already active; continuing it "
                "preserves the accepted points, or cancel it before restarting"
            )
        if "degree" in tool_input:
            degree = _bounded_int(
                tool_input["degree"],
                minimum=panel.calibration_degree_input.minimum(),
                maximum=panel.calibration_degree_input.maximum(),
                name="degree",
            )
            panel.calibration_degree_input.setValue(degree)
        panel._start_calibration()
    elif operation == "cancel":
        if "degree" in tool_input:
            raise ValueError("degree is only valid when operation is start")
        if panel._calibrating:
            panel._cancel_calibration()
    elif "degree" in tool_input:
        raise ValueError("degree is only valid when operation is start")

    degree = int(panel.calibration_degree_input.value())
    accepted = [
        {"pixel": int(pixel), "known_shift_cm-1": float(shift)}
        for pixel, shift in zip(
            panel._calibration_pixels, panel._known_shifts
        )
    ]
    result = {
        "panel_id": _panel_id(panel),
        "panel_title": panel.windowTitle(),
        "operation": operation,
        "state": "active" if panel._calibrating else "inactive",
        "degree": degree,
        "required_peak_count": degree + 1,
        "accepted_peak_count": len(accepted),
        "accepted_peaks": accepted,
        "pending_pixel": (
            None if panel._pending_pixel is None else int(panel._pending_pixel)
        ),
        "finish_enabled": bool(panel.finish_calibration_btn.isEnabled()),
        "instructions": panel.calibration_help.text(),
        "manual_finish_required": (
            "Peak clicks, known Raman-shift entry, and Finish and save remain "
            "manual GUI steps. This tool never invents reference shifts and "
            "never saves or overwrites a calibration file."
        ),
    }
    return json.dumps(result, sort_keys=True, separators=(",", ":"))


def get_calibration_state(owner):
    """Return compact open calibration state for a broader state query."""
    result_panel = None
    log_panel = None
    for panel in reversed(list(getattr(owner, "_plot_windows", ()) or ())):
        if result_panel is None and isinstance(panel, CalibrationPlotWindow):
            result_panel = panel
        if (
            log_panel is None
            and isinstance(panel, LogWindow)
            and "calibration" in panel.windowTitle().casefold()
        ):
            log_panel = panel
        if result_panel is not None and log_panel is not None:
            break

    state = {
        "latest_calibration_result": None,
        "latest_calibration_log": None,
    }
    if result_panel is not None:
        state["latest_calibration_result"] = {
            "panel_id": _panel_id(result_panel),
            "selected_point_index": (
                None
                if result_panel.selected_index is None
                else int(result_panel.selected_index) + 1
            ),
            "point_count": int(len(result_panel._point_positions)),
        }
    if log_panel is not None:
        state["latest_calibration_log"] = _progress_payload(
            log_panel, tail_chars=0
        )
    return state


CALIBRATION_ACTIONS = [
    {
        "name": "inspect_calibration_result",
        "label": "Inspect calibration result",
        "readonly": True,
        "handler": inspect_calibration_result,
        "params": [
            _parameter(
                "panel_id",
                "text",
                "Stable calibration-result panel ID. Omit to use the current plot tab.",
            ),
            _parameter(
                "point_index",
                "int",
                "Optional 1-based acquisition point to select and inspect. Omit to preserve the current selection.",
                schema={"type": "integer", "minimum": 1},
            ),
            _parameter(
                "include_point_inventory",
                "check",
                "Include a bounded list of 1-based point indices and coordinates.",
                schema={"type": "boolean"},
            ),
            _parameter(
                "point_limit",
                "int",
                f"Maximum inventory entries (1-{MAX_POINT_LIMIT}).",
                schema={
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_POINT_LIMIT,
                },
            ),
            _parameter(
                "include_trace",
                "check",
                "Explicitly include numerical X and intensity values for the displayed selected spectrum.",
                schema={"type": "boolean"},
            ),
            _parameter(
                "trace_limit",
                "int",
                f"Maximum trace values returned (1-{MAX_TRACE_LIMIT}).",
                schema={
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_TRACE_LIMIT,
                },
            ),
        ],
        "description": (
            "Inspect an existing calibration result and optionally select one "
            "real acquisition point by its 1-based index. Returns coordinates "
            "and compact stored/displayed spectrum statistics by default; a "
            "bounded numerical trace is returned only when explicitly asked. "
            "This changes only the plot selection and never accesses hardware."
        ),
    },
    {
        "name": "query_calibration_progress",
        "label": "Read calibration progress",
        "readonly": True,
        "handler": query_calibration_progress,
        "params": [
            _parameter(
                "panel_id",
                "text",
                "Stable Calibration log panel ID. Omit for the current calibration log or the newest open calibration log when another plot is current.",
            ),
            _parameter(
                "tail_chars",
                "int",
                f"Recent log characters to return (0-{MAX_LOG_TAIL_CHARS}).",
                schema={
                    "type": "integer",
                    "minimum": 0,
                    "maximum": MAX_LOG_TAIL_CHARS,
                },
            ),
        ],
        "description": (
            "Read the real progress-bar values, stage label, state, and a "
            "bounded tail of an existing Calibration log. Calibration runs "
            "in a background worker; this read-only query uses its latest "
            "UI-delivered counts without accessing hardware. Stopping means "
            "cleanup is pending, not stopped or complete, even if the count "
            "is full. This tool never estimates missing progress or monitors "
            "the job automatically."
        ),
    },
    {
        "name": "control_spectral_axis_calibration",
        "label": "Control spectral-axis calibration",
        "readonly": True,
        "handler": control_spectral_axis_calibration,
        "params": [
            _parameter(
                "panel_id",
                "text",
                "Stable Spectrum panel ID. Omit to use the current plot tab.",
            ),
            _parameter(
                "operation",
                "combo",
                "Use status to inspect, start to open the manual peak workflow, or cancel to leave it without saving.",
                schema={
                    "type": "string",
                    "enum": ["status", "start", "cancel"],
                },
            ),
            _parameter(
                "degree",
                "int",
                "Optional polynomial degree for start; validated against the Spectrum panel's allowed range.",
                schema={"type": "integer", "minimum": 1, "maximum": 9},
            ),
        ],
        "description": (
            "Read, start, or cancel the selected Spectrum panel's existing "
            "manual pixel-to-wavenumber calibration workflow. Starting may "
            "set a validated polynomial degree. Peak selection, known-shift "
            "entry, and Finish and save remain manual GUI steps; this tool "
            "never fabricates reference shifts or saves/overwrites a file."
        ),
    },
]


__all__ = [
    "CALIBRATION_ACTIONS",
    "CALIBRATION_CONTROL_WIDGET_ATTRIBUTES",
    "control_spectral_axis_calibration",
    "get_calibration_state",
    "inspect_calibration_result",
    "query_calibration_progress",
]
