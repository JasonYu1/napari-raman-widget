"""Session calibrations and persistent RACER fit preferences."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from qtpy.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QWidget,
)

from .cosmic_ray_removal import RacerCalibration, load_racer_calibration
from .plot_windows import set_cosmic_ray_calibration
from .racer_preferences import (
    load_calibration_options,
    load_calibration_paths,
    save_calibration_options,
    save_calibration_paths,
)
from .racer_settings_dialog import RacerCalibrationSettingsDialog
from .spectra import racer_detector_signature

__all__ = [
    "add_racer_calibration_controls",
    "clear_racer_calibration",
    "load_racer_calibration_from_dialogs",
    "matching_racer_calibration",
    "show_racer_calibration_settings",
]


_UNLOADED_TOOLTIP = (
    "Load a closed-shutter Zarr acquisition for display-only RACER cosmic-ray "
    "removal. An illuminated open reference is optional for RACER-1D (FVB): "
    "leave the reference path empty to use closed-shutter variance only. That "
    "fallback does not model signal-dependent shot noise. RACER-2D (Image) "
    "requires an illuminated reference to calibrate its optical row profile. "
    "The read mode is detected from the stores, and the calibration is "
    "session-only. Saved paths are restored without loading a calibration; "
    "click Load RACER calibration to load the selected files."
)


def _status(owner: Any, message: str) -> None:
    status = getattr(owner, "status", None)
    if status is not None and hasattr(status, "setText"):
        status.setText(message)


def _loaded_tooltip(calibration: RacerCalibration) -> str:
    dark_path, reference_path = calibration.source_paths
    details = (
        f"Loaded {calibration.read_mode} RACER calibration at "
        f"{calibration.exposure_ms:g} ms for detector shape "
        f"{calibration.detector_shape}; gain "
        f"{calibration.gain_e_per_adu:g} e-/ADU; saturation "
        f"{calibration.saturation_adu:g} ADU.\n"
        f"Closed shutter: {dark_path}\n"
    )
    if reference_path is None:
        details += (
            "Variance source: closed-shutter variance only (no illuminated "
            "reference; signal-dependent shot noise is not modeled).\n"
        )
    else:
        reference_purpose = (
            "Illuminated open-reference spectra (1D reference-level variance)"
            if calibration.read_mode == "fvb"
            else "Illuminated open-reference images "
            "(2D spatial-profile calibration)"
        )
        details += f"{reference_purpose}: {reference_path}\n"
    details += "Used by compatible open and newly created plots in this session."
    options = getattr(calibration, "calibration_options", {})
    if options:
        details += "\nCalibration settings used: " + "; ".join(
            f"{group}.{name}={value}"
            for group, values in options.items()
            for name, value in values.items()
        )
    if calibration.metadata_warnings:
        details += "\nWarnings: " + "; ".join(calibration.metadata_warnings)
    return details


def _set_load_tooltip(owner: Any, text: str) -> None:
    button = getattr(owner, "load_racer_calibration_btn", None)
    if button is not None:
        button.setToolTip(text)


def add_racer_calibration_controls(owner: Any, layout: Any) -> None:
    """Add session-only RACER load/clear controls to the loading layout."""

    owner.racer_calibration = None
    owner.racer_calibration_options = {}
    owner._racer_preferences_error = None
    try:
        owner.racer_calibration_options = load_calibration_options()
    except (OSError, ValueError) as error:
        owner._racer_preferences_error = (
            "Saved RACER settings could not be read; using defaults. "
            f"{error}"
        )
    owner._racer_paths_error = None
    try:
        paths = load_calibration_paths()
    except (OSError, ValueError) as error:
        paths = {}
        owner._racer_paths_error = f"Saved RACER file paths could not be read: {error}"
    layout.addWidget(QLabel("RACER calibration (FVB spectra / Image frames):"))
    owner.racer_calibration_path_controls = QWidget()
    paths_layout = QVBoxLayout(owner.racer_calibration_path_controls)
    paths_layout.setContentsMargins(0, 0, 0, 0)
    for key, title, field_name, button_name in (
        ("closed_shutter", "Closed shutter:", "racer_closed_shutter_path",
         "browse_closed_shutter_racer_btn"),
        ("open_reference", "Open reference:", "racer_open_reference_path",
         "browse_open_reference_racer_btn"),
    ):
        path_row = QHBoxLayout()
        label = QLabel(title)
        field = QLineEdit(paths.get(key) or "")
        field.setPlaceholderText(
            "Select a closed-shutter Zarr stack"
            if key == "closed_shutter"
            else "None (optional for FVB / 1D; required for Image / 2D)"
        )
        field.setClearButtonEnabled(True)
        label.setBuddy(field)
        field.editingFinished.connect(lambda: _remember_racer_paths(owner))
        setattr(owner, field_name, field)
        browse = QPushButton("...")
        browse.setFixedWidth(30)
        browse.setToolTip(f"Choose {title.rstrip(':').lower()} Zarr; load manually")
        browse.clicked.connect(
            lambda _checked=False, kind=key: _browse_racer_path(owner, kind)
        )
        setattr(owner, button_name, browse)
        path_row.addWidget(label)
        path_row.addWidget(field, 1)
        path_row.addWidget(browse)
        paths_layout.addLayout(path_row)
    layout.addWidget(owner.racer_calibration_path_controls)
    row = QHBoxLayout()

    owner.load_racer_calibration_btn = QPushButton(
        "Load RACER calibration\N{HORIZONTAL ELLIPSIS}"
    )
    owner.load_racer_calibration_btn.setToolTip(_UNLOADED_TOOLTIP)
    owner.load_racer_calibration_btn.clicked.connect(
        lambda: load_racer_calibration_from_dialogs(owner)
    )
    row.addWidget(owner.load_racer_calibration_btn)

    owner.clear_racer_calibration_btn = QPushButton("Clear")
    owner.clear_racer_calibration_btn.setToolTip(
        "Clear the session RACER calibration from open and future plots. "
        "Keep the remembered file paths for the next manual load."
    )
    owner.clear_racer_calibration_btn.setEnabled(False)
    owner.clear_racer_calibration_btn.clicked.connect(
        lambda: clear_racer_calibration(owner)
    )
    row.addWidget(owner.clear_racer_calibration_btn)

    owner.racer_calibration_settings_btn = QPushButton(
        "Advanced calibration settings\N{HORIZONTAL ELLIPSIS}"
    )
    owner.racer_calibration_settings_btn.clicked.connect(
        lambda: show_racer_calibration_settings(owner)
    )
    row.addWidget(owner.racer_calibration_settings_btn)
    layout.addLayout(row)
    _update_settings_tooltip(owner)
    _update_path_tooltips(owner)


def _selected_racer_paths(owner: Any) -> dict[str, str | None]:
    return {
        "closed_shutter": owner.racer_closed_shutter_path.text().strip() or None,
        "open_reference": owner.racer_open_reference_path.text().strip() or None,
    }


def _unloaded_tooltip(owner: Any) -> str:
    paths = _selected_racer_paths(owner)
    details = (
        f"{_UNLOADED_TOOLTIP}\nClosed shutter: {paths['closed_shutter'] or 'None'}"
        f"\nOpen reference: {paths['open_reference'] or 'None'}"
    )
    if owner._racer_paths_error:
        details += f"\n{owner._racer_paths_error}"
    return details


def _update_path_tooltips(owner: Any) -> None:
    for field in (owner.racer_closed_shutter_path, owner.racer_open_reference_path):
        details = (
            f"{field.text().strip() or 'None'}\n"
            "Paths are remembered for next time. Click Load RACER calibration "
            "to apply the selected files."
        )
        if owner._racer_paths_error:
            details += f"\n{owner._racer_paths_error}"
        field.setToolTip(details)
    if owner.racer_calibration is None:
        _set_load_tooltip(owner, _unloaded_tooltip(owner))


def _remember_racer_paths(owner: Any, *, report: bool = True) -> str | None:
    """Persist path text only; never inspect or load the selected Zarr stores."""
    paths = _selected_racer_paths(owner)
    try:
        save_calibration_paths(paths["closed_shutter"], paths["open_reference"])
    except (OSError, ValueError) as error:
        owner._racer_paths_error = (
            "RACER file paths are selected for this session but could not be "
            f"saved for next time: {error}"
        )
    else:
        owner._racer_paths_error = None
    _update_path_tooltips(owner)
    if report:
        _status(owner, "Status: " + (
            owner._racer_paths_error
            or "RACER file paths remembered; click Load RACER calibration to apply them"
        ))
    return owner._racer_paths_error


def _browse_racer_path(owner: Any, kind: str) -> None:
    """Choose a remembered input without fitting or replacing a calibration."""
    paths = _selected_racer_paths(owner)
    title = (
        "Select closed-shutter RACER calibration Zarr"
        if kind == "closed_shutter"
        else "Select illuminated open-reference RACER Zarr"
    )
    start = paths[kind] or paths["closed_shutter"] or str(Path.cwd())
    selected = QFileDialog.getExistingDirectory(owner, title, start)
    if not selected:
        return
    field = (
        owner.racer_closed_shutter_path if kind == "closed_shutter"
        else owner.racer_open_reference_path
    )
    field.setText(selected)
    _remember_racer_paths(owner)


def _update_settings_tooltip(owner: Any) -> None:
    options = getattr(owner, "racer_calibration_options", {})
    details = (
        "Save RACER's calibration settings for future widget sessions. "
        "Changes apply on the next calibration load. Reload to update an active calibration."
    )
    if options:
        details += "\nPending settings: " + "; ".join(
            f"{group}.{name}={value}"
            for group, values in options.items()
            for name, value in values.items()
        )
    else:
        details += "\nUsing RACER defaults."
    error = getattr(owner, "_racer_preferences_error", None)
    if error:
        details += f"\n{error}"
    owner.racer_calibration_settings_btn.setToolTip(details)


def show_racer_calibration_settings(owner: Any) -> None:
    """Save optional fit settings for the next load; keep the active fit intact."""
    read_mode = getattr(getattr(owner, "racer_calibration", None), "read_mode", "fvb")
    mode_combo = getattr(owner, "collect_read_mode_combo", None)
    if mode_combo is not None:
        read_mode = mode_combo.currentData()
    try:
        dialog = RacerCalibrationSettingsDialog(
            getattr(owner, "racer_calibration_options", {}),
            owner,
            read_mode=read_mode,
            settings_warning=getattr(owner, "_racer_preferences_error", None),
        )
    except Exception as error:
        _status(owner, f"Status: RACER calibration settings unavailable -- {error}")
        return
    if dialog.exec_() == QDialog.Accepted:
        owner.racer_calibration_options = dialog.calibration_options
        try:
            save_calibration_options(owner.racer_calibration_options)
        except (OSError, ValueError) as error:
            message = (
                "RACER settings changed for this session, but could not be "
                f"saved for next time: {error}. Open settings and save again to retry."
            )
            owner._racer_preferences_error = message
        else:
            owner._racer_preferences_error = None
            message = (
                "RACER calibration settings saved for future sessions; "
                "load a calibration to apply them"
            )
        _update_settings_tooltip(owner)
        _status(owner, f"Status: {message}")
    dialog.deleteLater()


def load_racer_calibration_from_dialogs(owner: Any) -> RacerCalibration | None:
    """Manually load the displayed paths, choosing files when none are selected."""

    paths = _selected_racer_paths(owner)
    closed_shutter = paths["closed_shutter"]
    open_reference = paths["open_reference"]
    if not closed_shutter:
        closed_shutter = QFileDialog.getExistingDirectory(
            owner,
            "Select closed-shutter RACER calibration Zarr",
            str(Path.cwd()),
        )
        if not closed_shutter:
            _status(owner, "Status: RACER calibration loading cancelled")
            return None
        if not open_reference:
            open_reference = QFileDialog.getExistingDirectory(
                owner,
                "Select illuminated open-reference RACER Zarr "
                "(optional for 1D; Cancel uses dark variance, required for 2D)",
                str(Path(closed_shutter).parent),
            )
        owner.racer_closed_shutter_path.setText(closed_shutter)
        owner.racer_open_reference_path.setText(open_reference or "")
    path_save_warning = _remember_racer_paths(owner, report=False)
    reference_path = Path(open_reference) if open_reference else None

    previous = getattr(owner, "racer_calibration", None)
    try:
        options = getattr(owner, "racer_calibration_options", {})
        kwargs = {"calibration_options": options} if options else {}
        calibration = load_racer_calibration(
            Path(closed_shutter), reference_path, **kwargs
        )
        read_mode = calibration.read_mode
        if read_mode not in {"fvb", "image"}:
            raise ValueError(
                f"selected files use unsupported {read_mode!r} readout; "
                "RACER requires FVB spectra or Image frames"
            )
        if isinstance(calibration, RacerCalibration):
            if calibration.detector_signature is None:
                if read_mode == "fvb":
                    if calibration.open_reference_path is None:
                        raise ValueError(
                            "the selected RACER-1D closed-shutter store is "
                            "legacy data without a detector acquisition "
                            "signature. It cannot be verified against current "
                            "detector data; recollect the closed-shutter stack "
                            "with the current widget"
                        )
                    raise ValueError(
                        "the selected RACER-1D stores are a legacy pair "
                        "without a detector acquisition signature. RACER-1D "
                        "does not use a 2D spatial profile; the illuminated "
                        "reference spectrum stack measures variance at its "
                        "signal level. Neither store can be verified against "
                        "current detector data; recollect both stacks with "
                        "the current widget"
                    )
                raise ValueError(
                    "the selected RACER-2D stores are a legacy pair without "
                    "a detector acquisition signature. Their spatial profile "
                    "and detector settings cannot be verified against the "
                    "current detector; recollect both stacks with the current "
                    "widget"
                )
            if getattr(owner, "collector", None) is None:
                raise ValueError(
                    "connect the detector before loading a RACER calibration "
                    "so its acquisition settings can be verified"
                )
            signature = racer_detector_signature(
                owner.collector,
                np.empty(calibration.detector_shape),
                read_mode,
                calibration.exposure_ms,
                include_acquisition_timing=False,
            )
            compatible, reason = calibration.compatibility(
                np.empty(calibration.detector_shape),
                exposure_ms=calibration.exposure_ms,
                read_mode=read_mode,
                detector_signature=signature,
            )
            if not compatible:
                raise ValueError(reason)
    except Exception as error:
        message = f"RACER calibration load failed -- {error}"
        if path_save_warning:
            message += f". {path_save_warning}"
        if previous is None:
            _status(owner, f"Status: {message}")
            _set_load_tooltip(owner, message)
        else:
            retained = (
                f"{message}. The previous calibration remains active."
            )
            _status(owner, f"Status: {retained}")
            _set_load_tooltip(
                owner, f"{retained}\n\n{_loaded_tooltip(previous)}"
            )
        return None

    owner.racer_calibration = calibration
    owner.clear_racer_calibration_btn.setEnabled(True)
    _set_load_tooltip(owner, _loaded_tooltip(calibration))
    compatible_plots, total_plots = _sync_open_plots(owner, calibration)
    plot_summary = (
        f"; enabled {compatible_plots}/{total_plots} open spectral plots"
        if total_plots
        else ""
    )
    warning_summary = (
        "; warning: " + "; ".join(calibration.metadata_warnings)
        if calibration.metadata_warnings
        else ""
    )
    if path_save_warning:
        warning_summary += f"; {path_save_warning}"
    variance_summary = (
        ", closed-shutter variance only"
        if calibration.open_reference_path is None
        else ""
    )
    _status(
        owner,
        "Status: RACER calibration loaded for this session "
        f"({calibration.read_mode}, {calibration.exposure_ms:g} ms"
        f"{variance_summary})"
        f"{plot_summary}{warning_summary}",
    )
    return calibration


def clear_racer_calibration(owner: Any) -> None:
    """Clear the session calibration from new and already-open plots."""

    owner.racer_calibration = None
    owner.clear_racer_calibration_btn.setEnabled(False)
    _set_load_tooltip(owner, _unloaded_tooltip(owner))
    _sync_open_plots(owner, None)
    _status(
        owner,
        "Status: RACER calibration cleared from open and future plots",
    )


def matching_racer_calibration(
    owner: Any,
    values: Any,
    exposure_ms: float,
    read_mode: str,
    *,
    update_tooltip: bool = True,
) -> RacerCalibration | None:
    """Return the session calibration only when it matches acquired values."""

    calibration = getattr(owner, "racer_calibration", None)
    if calibration is None:
        return None
    try:
        compatibility_values = np.asarray(values)
        if read_mode == "fvb" and compatibility_values.ndim > 2:
            compatibility_values = compatibility_values.reshape(
                -1, compatibility_values.shape[-1]
            )
        elif read_mode == "image" and compatibility_values.ndim > 3:
            compatibility_values = compatibility_values.reshape(
                -1, *compatibility_values.shape[-2:]
            )
        compatibility_kwargs = {
            "exposure_ms": exposure_ms,
            "read_mode": read_mode,
        }
        if isinstance(calibration, RacerCalibration):
            compatibility_kwargs["detector_signature"] = (
                racer_detector_signature(
                    owner.collector,
                    compatibility_values,
                    read_mode,
                    exposure_ms,
                )
            )
        compatible, reason = calibration.compatibility(
            compatibility_values, **compatibility_kwargs
        )
    except Exception as error:
        if update_tooltip:
            _set_load_tooltip(
                owner,
                f"Loaded RACER calibration could not be checked: {error}",
            )
        return None
    if compatible:
        if update_tooltip:
            _set_load_tooltip(owner, _loaded_tooltip(calibration))
        return calibration
    if update_tooltip:
        _set_load_tooltip(
            owner,
            "Loaded RACER calibration is not compatible with this result: "
            f"{reason}",
        )
    return None


def _sync_open_plots(
    owner: Any, calibration: RacerCalibration | None
) -> tuple[int, int]:
    """Apply one session calibration to every compatible open spectral plot."""

    compatible_count = 0
    total_count = 0
    for panel in tuple(getattr(owner, "_plot_windows", ())):
        if not hasattr(panel, "cosmic_ray_removal_check"):
            continue
        total_count += 1
        candidate = None
        if calibration is not None:
            candidate = matching_racer_calibration(
                owner,
                panel._cosmic_ray_values,
                panel.cosmic_ray_exposure_ms,
                panel.cosmic_ray_read_mode,
                update_tooltip=False,
            )
        if set_cosmic_ray_calibration(panel, candidate):
            compatible_count += 1
    return compatible_count, total_count
