"""Assistant adapters for inspecting and configuring open plot panels.

These actions deliberately operate through the plots' existing Qt controls.
They do not create results, close tabs, or emulate manual canvas gestures.
Interactive peak picking and calibration-point selection remain dedicated
manual/calibration workflows rather than generic plot configuration.
"""

from __future__ import annotations

import json
import math
from numbers import Integral, Real


__all__ = [
    "PLOT_ACTIONS",
    "PLOT_CONTROL_WIDGET_ATTRIBUTES",
    "get_plot_state",
    "get_plot_workspace_state",
    "resolve_plot_panel",
]


# Public inventory used by coverage tests and by callers that need to explain
# which real controls back the assistant-facing names.  Mode buttons and
# navigation fields are included even though they only exist on some panels.
PLOT_CONTROL_WIDGET_ATTRIBUTES = {
    "white_background": "white_background_check",
    "fix_y_scale": "fix_y_scale_check",
    "show_wavenumber": "show_wavenumber_check",
    "smoothing": "smoothing_check",
    "smoothing_window": "smoothing_window_input",
    "baseline_subtraction": "baseline_subtraction_check",
    "baseline_lambda": "baseline_lambda_input",
    "remove_spectral_bias": "remove_spectral_bias_check",
    "spectrum_view": "toggle_btn",
    "detector_view": "toggle_btn",
    "detector_start_row": "start_row_input",
    "detector_end_row": "end_row_input",
    "grid_view": "_mode_btn",
    "grid_z_index": "_z_input",
    "dataset_t_index": "t_input",
    "dataset_p_index": "p_input",
    "dataset_z_index": "z_input",
}

_CONFIGURE_ARGUMENTS = frozenset(PLOT_CONTROL_WIDGET_ATTRIBUTES) | {
    "panel_id"
}


def _p(name, kind, description, *, enum=None):
    """Build a chat action parameter without importing ``chat_panel``."""
    param = {
        "name": name,
        "attr": None,
        "kind": kind,
        "description": description,
    }
    if enum is not None:
        param["enum"] = list(enum)
    return param


def _workspace(owner):
    """Return an existing plot workspace without constructing one."""
    return getattr(owner, "_plot_workspace", None)


def resolve_plot_panel(owner, panel_id=None):
    """Resolve an open panel by stable ID, or return the current panel.

    This is intentionally read-only: asking about plots never creates the
    workspace that the user has not opened yet.
    """
    workspace = _workspace(owner)
    if workspace is None:
        raise ValueError("No plot workspace is open.")
    if workspace.tabs.count() == 0:
        raise ValueError("No plots are open.")
    if panel_id is None:
        panel = workspace.tabs.currentWidget()
        if panel is None:
            raise ValueError("No plot is currently selected.")
        return panel
    if not isinstance(panel_id, str) or not panel_id.strip():
        raise ValueError("panel_id must be a non-empty string.")
    panel = workspace.panel_for_id(panel_id)
    if panel is None:
        available = [
            workspace.panel_id_for(workspace.tabs.widget(index))
            for index in range(workspace.tabs.count())
        ]
        suffix = f" Available panel IDs: {', '.join(available)}." if available else ""
        raise ValueError(f"Unknown plot panel_id {panel_id!r}.{suffix}")
    return panel


def _checkbox_state(panel, name, values, limits, enabled):
    attr = PLOT_CONTROL_WIDGET_ATTRIBUTES[name]
    widget = getattr(panel, attr, None)
    if widget is None:
        return
    values[name] = bool(widget.isChecked())
    enabled[name] = bool(widget.isEnabled())


def _numeric_state(panel, name, values, limits, enabled):
    attr = PLOT_CONTROL_WIDGET_ATTRIBUTES[name]
    widget = getattr(panel, attr, None)
    if widget is None:
        return
    value = widget.value()
    if isinstance(value, Integral):
        value = int(value)
    else:
        value = float(value)
    values[name] = value
    enabled[name] = bool(widget.isEnabled())
    limits[name] = {
        "minimum": int(widget.minimum())
        if isinstance(widget.minimum(), Integral)
        else float(widget.minimum()),
        "maximum": int(widget.maximum())
        if isinstance(widget.maximum(), Integral)
        else float(widget.maximum()),
    }


def _panel_state(workspace, panel):
    panel_type = type(panel).__name__
    values = {}
    limits = {}
    enabled = {}
    for name in (
        "white_background",
        "fix_y_scale",
        "show_wavenumber",
        "smoothing",
        "baseline_subtraction",
    ):
        _checkbox_state(panel, name, values, limits, enabled)
    for name in ("smoothing_window", "baseline_lambda"):
        _numeric_state(panel, name, values, limits, enabled)

    if panel_type == "SpectrumWindow":
        values["spectrum_view"] = "mean" if panel._show_mean else "all"
        enabled["spectrum_view"] = bool(panel.toggle_btn.isEnabled())
        limits["spectrum_view"] = {"choices": ["mean", "all"]}
        if getattr(panel, "spectral_bias", None) is not None:
            _checkbox_state(
                panel, "remove_spectral_bias", values, limits, enabled
            )
    elif panel_type == "DetectorImageWindow":
        values["detector_view"] = (
            "spectrum" if panel._show_spectrum else "image"
        )
        enabled["detector_view"] = bool(panel.toggle_btn.isEnabled())
        limits["detector_view"] = {"choices": ["image", "spectrum"]}
        _numeric_state(
            panel, "detector_start_row", values, limits, enabled
        )
        _numeric_state(panel, "detector_end_row", values, limits, enabled)
        if not panel._show_spectrum:
            for name in (
                "fix_y_scale",
                "show_wavenumber",
                "smoothing",
                "smoothing_window",
                "baseline_subtraction",
                "baseline_lambda",
            ):
                if name in enabled:
                    enabled[name] = False
    elif panel_type == "GridScanPlotWindow":
        values["grid_view"] = "mean" if panel._show_average else "point"
        enabled["grid_view"] = bool(panel._mode_btn.isEnabled())
        limits["grid_view"] = {"choices": ["mean", "point"]}
        if hasattr(panel, "_z_input"):
            _numeric_state(panel, "grid_z_index", values, limits, enabled)
    elif panel_type == "DatasetViewerWindow":
        for name in (
            "dataset_t_index",
            "dataset_p_index",
            "dataset_z_index",
        ):
            _numeric_state(panel, name, values, limits, enabled)

    state = {
        "panel_id": workspace.panel_id_for(panel),
        "title": panel.windowTitle(),
        "type": panel_type,
        "current": panel is workspace.tabs.currentWidget(),
        "supported_controls": list(values),
        "values": values,
        "enabled": enabled,
        "limits": limits,
    }
    if hasattr(panel, "show_wavenumber_check"):
        state["spectral_calibration_available"] = (
            getattr(panel, "spectral_calibration", None) is not None
        )
    if panel_type == "CalibrationPlotWindow":
        selected = getattr(panel, "selected_index", None)
        state["selected_calibration_point_index"] = (
            None if selected is None else int(selected) + 1
        )
    elif panel_type == "GridScanPlotWindow":
        state["selected_grid_point_index"] = int(panel._sel)
    elif panel_type == "DatasetViewerWindow":
        state["selected_dataset_point_index"] = int(panel._pt_selected)
    if panel_type == "LogWindow":
        progress = panel.progress_bar
        state["progress"] = {
            "visible": not panel.progress_widget.isHidden(),
            "stage": panel.progress_label.text(),
            "minimum": int(progress.minimum()),
            "maximum": int(progress.maximum()),
            "value": int(progress.value()),
            "indeterminate": progress.minimum() == progress.maximum() == 0,
        }
    return state


def get_plot_state(owner):
    """Return JSON-serializable state for open panels in current tab order."""
    workspace = _workspace(owner)
    if workspace is None:
        return []
    return [
        _panel_state(workspace, workspace.tabs.widget(index))
        for index in range(workspace.tabs.count())
    ]


def get_plot_workspace_state(owner):
    """Return visibility/docking state without constructing a workspace."""
    workspace = _workspace(owner)
    if workspace is None:
        return {
            "exists": False,
            "visible": False,
            "floating": None,
            "panel_count": 0,
            "current_panel_id": None,
        }
    dock = workspace._dock
    try:
        visible = dock is not None and not dock.isHidden()
        floating = dock.isFloating() if dock is not None else workspace._floating
    except RuntimeError:
        visible = False
        floating = workspace._floating
    current = workspace.tabs.currentWidget()
    return {
        "exists": True,
        "visible": bool(visible),
        "floating": bool(floating),
        "panel_count": int(workspace.tabs.count()),
        "current_panel_id": (
            workspace.panel_id_for(current) if current is not None else None
        ),
    }


def _validate_arguments(tool_input, allowed):
    if not isinstance(tool_input, dict):
        raise ValueError("Plot tool input must be an object.")
    unknown = sorted(set(tool_input) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown plot argument(s): {', '.join(unknown)}.")


def _as_bool(name, value):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false.")
    return value


def _as_int(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def _as_float(name, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return value


def _choice(name, value, choices):
    if value not in choices:
        rendered = ", ".join(repr(choice) for choice in choices)
        raise ValueError(f"{name} must be one of {rendered}.")
    return value


def _widget(panel, name):
    attr = PLOT_CONTROL_WIDGET_ATTRIBUTES[name]
    widget = getattr(panel, attr, None)
    if widget is None:
        raise ValueError(
            f"{name} is not supported by {type(panel).__name__}."
        )
    return widget


def _in_range(name, value, widget):
    minimum = widget.minimum()
    maximum = widget.maximum()
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum:g} and {maximum:g}; "
            f"got {value:g}."
        )


def _h_list_plots(owner, tool_input):
    _validate_arguments(tool_input, ())
    return json.dumps(
        {
            "workspace": get_plot_workspace_state(owner),
            "plots": get_plot_state(owner),
        },
        sort_keys=True,
    )


def _h_configure_plot(owner, tool_input):
    """Validate a complete request, then apply it through real controls."""
    _validate_arguments(tool_input, _CONFIGURE_ARGUMENTS)
    panel = resolve_plot_panel(owner, tool_input.get("panel_id"))
    panel_type = type(panel).__name__
    operations = []

    detector_target = None
    if panel_type == "DetectorImageWindow":
        detector_target = "spectrum" if panel._show_spectrum else "image"
        if "detector_view" in tool_input:
            detector_target = _choice(
                "detector_view",
                tool_input["detector_view"],
                ("image", "spectrum"),
            )

    if "spectrum_view" in tool_input:
        if panel_type != "SpectrumWindow":
            raise ValueError(
                f"spectrum_view is not supported by {panel_type}."
            )
        target = _choice(
            "spectrum_view", tool_input["spectrum_view"], ("mean", "all")
        )
        if not panel.toggle_btn.isEnabled() and (
            (target == "mean") != panel._show_mean
        ):
            raise ValueError(
                "spectrum_view is unavailable during axis calibration."
            )
        operations.append(
            lambda target=target: panel.toggle_btn.click()
            if ((target == "mean") != panel._show_mean)
            else None
        )

    if "detector_view" in tool_input:
        if panel_type != "DetectorImageWindow":
            raise ValueError(
                f"detector_view is not supported by {panel_type}."
            )
        operations.append(
            lambda target=detector_target: panel.toggle_btn.click()
            if ((target == "spectrum") != panel._show_spectrum)
            else None
        )

    if "grid_view" in tool_input:
        if panel_type != "GridScanPlotWindow":
            raise ValueError(f"grid_view is not supported by {panel_type}.")
        target = _choice(
            "grid_view", tool_input["grid_view"], ("mean", "point")
        )
        operations.append(
            lambda target=target: panel._mode_btn.click()
            if ((target == "mean") != panel._show_average)
            else None
        )

    numeric_names = (
        "detector_start_row",
        "detector_end_row",
        "grid_z_index",
        "dataset_t_index",
        "dataset_p_index",
        "dataset_z_index",
    )
    numeric_values = {}
    for name in numeric_names:
        if name not in tool_input:
            continue
        widget = _widget(panel, name)
        value = _as_int(name, tool_input[name])
        _in_range(name, value, widget)
        numeric_values[name] = value

    if panel_type == "DetectorImageWindow":
        start = numeric_values.get(
            "detector_start_row", panel.start_row_input.value()
        )
        end = numeric_values.get("detector_end_row", panel.end_row_input.value())
        if start > end:
            raise ValueError(
                "detector_start_row must not exceed detector_end_row."
            )
    for name in numeric_names:
        if name in numeric_values:
            widget = _widget(panel, name)
            value = numeric_values[name]
            operations.append(
                lambda widget=widget, value=value: widget.setValue(value)
            )

    processing_requested = any(
        name in tool_input
        for name in (
            "smoothing",
            "smoothing_window",
            "baseline_subtraction",
            "baseline_lambda",
        )
    )
    if (
        processing_requested
        and panel_type == "DetectorImageWindow"
        and detector_target != "spectrum"
    ):
        raise ValueError(
            "Spectrum processing is only available in detector_view='spectrum'."
        )

    if "smoothing_window" in tool_input:
        widget = _widget(panel, "smoothing_window")
        smoothing_check = _widget(panel, "smoothing")
        if not smoothing_check.isEnabled():
            raise ValueError("smoothing_window is unavailable for this plot's data.")
        value = _as_int("smoothing_window", tool_input["smoothing_window"])
        _in_range("smoothing_window", value, widget)
        if value % 2 == 0:
            raise ValueError("smoothing_window must be an odd integer.")
        operations.append(
            lambda widget=widget, value=value: widget.setValue(value)
        )

    if "baseline_lambda" in tool_input:
        widget = _widget(panel, "baseline_lambda")
        baseline_check = _widget(panel, "baseline_subtraction")
        if not baseline_check.isEnabled():
            raise ValueError("baseline_lambda is unavailable for this plot's data.")
        value = _as_float("baseline_lambda", tool_input["baseline_lambda"])
        _in_range("baseline_lambda", value, widget)
        operations.append(
            lambda widget=widget, value=value: widget.setValue(value)
        )

    for name in (
        "baseline_subtraction",
        "smoothing",
        "remove_spectral_bias",
    ):
        if name not in tool_input:
            continue
        if name == "remove_spectral_bias" and (
            panel_type != "SpectrumWindow"
            or getattr(panel, "spectral_bias", None) is None
        ):
            raise ValueError(
                "remove_spectral_bias is unavailable because this spectrum "
                "has no dark-noise bias."
            )
        widget = _widget(panel, name)
        value = _as_bool(name, tool_input[name])
        if value and not widget.isEnabled():
            raise ValueError(f"{name} is unavailable for this plot's data.")
        operations.append(lambda widget=widget, value=value: widget.setChecked(value))

    if "show_wavenumber" in tool_input:
        widget = _widget(panel, "show_wavenumber")
        value = _as_bool("show_wavenumber", tool_input["show_wavenumber"])
        if (
            panel_type == "DetectorImageWindow"
            and detector_target != "spectrum"
        ):
            raise ValueError(
                "show_wavenumber is only available in "
                "detector_view='spectrum'."
            )
        if value and getattr(panel, "spectral_calibration", None) is None:
            raise ValueError(
                "show_wavenumber requires a loaded spectral calibration."
            )
        if value and not widget.isEnabled():
            raise ValueError(
                "show_wavenumber is currently unavailable, for example while "
                "axis calibration is in progress."
            )
        operations.append(
            lambda widget=widget, value=value: widget.setChecked(value)
        )

    for name in ("fix_y_scale", "white_background"):
        if name not in tool_input:
            continue
        widget = _widget(panel, name)
        value = _as_bool(name, tool_input[name])
        if (
            name == "fix_y_scale"
            and panel_type == "DetectorImageWindow"
            and detector_target != "spectrum"
        ):
            raise ValueError(
                "fix_y_scale is only available in detector_view='spectrum'."
            )
        operations.append(lambda widget=widget, value=value: widget.setChecked(value))

    for operation in operations:
        operation()
    state = _panel_state(_workspace(owner), panel)
    return json.dumps(state, sort_keys=True)


def _h_show_plot_workspace(owner, tool_input):
    _validate_arguments(tool_input, ("panel_id", "floating"))
    panel_id = tool_input.get("panel_id")
    panel = resolve_plot_panel(owner, panel_id) if panel_id is not None else None
    floating = tool_input.get("floating")
    if floating is not None:
        floating = _as_bool("floating", floating)

    workspace = _workspace(owner)
    if workspace is None:
        show = getattr(owner, "_show_plot_workspace", None)
        if show is None:
            raise ValueError("This widget cannot show a plot workspace.")
        workspace = show()
    else:
        workspace.show_in_viewer()
    if panel is not None:
        workspace.tabs.setCurrentWidget(panel)
    dock = workspace._dock
    if floating is not None:
        if dock is None:
            raise ValueError("The plot workspace has no dock to reposition.")
        dock.setFloating(floating)
        dock.show()
        dock.raise_()
    current = workspace.tabs.currentWidget()
    return json.dumps(
        {
            "visible": dock is not None and not dock.isHidden(),
            "floating": dock.isFloating() if dock is not None else None,
            "current_panel_id": (
                workspace.panel_id_for(current) if current is not None else None
            ),
        },
        sort_keys=True,
    )


def _h_hide_plot_workspace(owner, tool_input):
    _validate_arguments(tool_input, ())
    workspace = _workspace(owner)
    if workspace is None:
        return "Plot workspace is not open."
    workspace._hide_workspace()
    return "Plot workspace hidden; its tabs remain open."


PLOT_ACTIONS = [
    {
        "name": "list_plots",
        "label": "List open plots",
        "readonly": True,
        "handler": _h_list_plots,
        "params": [],
        "description": (
            "List open plot and log tabs with stable panel IDs, current values, "
            "supported controls, ranges, selection, and progress state."
        ),
    },
    {
        "name": "configure_plot",
        "label": "Configure an open plot",
        "readonly": True,
        "handler": _h_configure_plot,
        "params": [
            _p(
                "panel_id",
                "text",
                "Stable panel ID from list_plots; omit for current.",
            ),
            _p(
                "white_background",
                "check",
                "Use a white rather than transparent plot background.",
            ),
            _p(
                "fix_y_scale",
                "check",
                "Keep the current spectrum Y limits fixed.",
            ),
            _p(
                "show_wavenumber",
                "check",
                "Show calibrated Raman shift; requires calibration.",
            ),
            _p(
                "smoothing",
                "check",
                "Enable display-only Savitzky-Golay smoothing.",
            ),
            _p(
                "smoothing_window",
                "int",
                "Odd smoothing window within the panel's reported limits.",
            ),
            _p(
                "baseline_subtraction",
                "check",
                "Enable display-only AsLS baseline subtraction.",
            ),
            _p(
                "baseline_lambda",
                "float",
                "AsLS lambda within the panel's reported limits.",
            ),
            _p(
                "remove_spectral_bias",
                "check",
                "Subtract the loaded dark-noise bias for this spectrum.",
            ),
            _p(
                "spectrum_view",
                "combo",
                "Mean spectrum or all acquired traces.",
                enum=("mean", "all"),
            ),
            _p(
                "detector_view",
                "combo",
                "Detector image or row-sum spectrum.",
                enum=("image", "spectrum"),
            ),
            _p("detector_start_row", "int", "First detector row in the row sum."),
            _p("detector_end_row", "int", "Last detector row in the row sum."),
            _p(
                "grid_view",
                "combo",
                "Mean or selected-point grid spectrum.",
                enum=("mean", "point"),
            ),
            _p("grid_z_index", "int", "Z-plane index for a grid z-scan."),
            _p("dataset_t_index", "int", "Dataset time index."),
            _p("dataset_p_index", "int", "Dataset stage-position index."),
            _p("dataset_z_index", "int", "Dataset z-plane index."),
        ],
        "description": (
            "Configure an existing plot through its real controls. All fields "
            "are validated before any change; use list_plots for IDs and limits."
        ),
    },
    {
        "name": "show_plot_workspace",
        "label": "Show plot workspace",
        "readonly": True,
        "handler": _h_show_plot_workspace,
        "params": [
            _p(
                "panel_id",
                "text",
                "Stable panel ID to focus; omit to keep current.",
            ),
            _p("floating", "check", "True to float the workspace, false to dock it."),
        ],
        "description": (
            "Show the plot workspace, optionally focus an existing panel and "
            "choose floating or docked presentation."
        ),
    },
    {
        "name": "hide_plot_workspace",
        "label": "Hide plot workspace",
        "readonly": True,
        "handler": _h_hide_plot_workspace,
        "params": [],
        "description": "Hide the workspace without closing or deleting any tabs.",
    },
]
