"""Read-only Raman MDA checks and an explicit pre-run review."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import os
from pathlib import Path

import numpy as np
from qtpy.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QTextBrowser, QVBoxLayout,
)

from .position_specs import resolve_position_specs


@dataclass(frozen=True)
class MdaReview:
    rows: tuple[tuple[str, str], ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    minimum_seconds: float | None = None
    settings_signature: str = ""

    @property
    def text(self):
        lines = [f"{label}: {value}" for label, value in self.rows]
        if self.errors:
            lines.extend(["", "FIX BEFORE STARTING"])
            lines.extend(f"• {message}" for message in self.errors)
        if self.warnings:
            lines.extend(["", "PLEASE REVIEW"])
            lines.extend(f"• {message}" for message in self.warnings)
        return "\n".join(lines)


def _duration(seconds):
    seconds = math.ceil(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours} h {minutes} min {seconds} s"


def _numbers(text, label, *, integer=False):
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{label} cannot be empty.")
    try:
        values = [int(part) if integer else float(part) for part in parts]
    except ValueError as error:
        raise ValueError(f"{label} must contain comma-separated numbers.") from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must contain finite numbers.")
    return values


def _check_busy(owner, errors):
    if getattr(owner, "_hardware_shutdown_started", False):
        errors.append("Hardware is shutting down; reconnect before starting.")
    jobs = getattr(owner, "_acquisition_jobs", None)
    if jobs is not None and jobs.is_running:
        errors.append("Another acquisition is running. Stop it or wait for completion.")
    if getattr(owner, "_live_raman_worker", None) is not None:
        errors.append("Stop live Raman spectra before starting MDA.")
    demo_timer = getattr(owner, "demo_live_timer", None)
    if demo_timer is not None and demo_timer.isActive():
        errors.append("Turn off demo live imaging before starting MDA.")
    thread = getattr(owner, "_raman_mda_thread", None)
    core = owner.core
    if core is None:
        return
    try:
        running = getattr(getattr(core, "mda", None), "is_running", False)
        if (running() if callable(running) else running) or (
            thread is not None and thread.is_alive()
        ):
            errors.append("An MDA is already running. Stop it or wait for completion.")
        live = getattr(core, "isSequenceRunning", None)
        if callable(live) and live():
            errors.append("Turn off live camera imaging before starting MDA.")
    except Exception as error:
        errors.append(f"Cannot check acquisition status: {error}")


def _check_output(text, errors, warnings):
    # Match run_raman_mda's relative-path semantics; do not create a directory.
    try:
        path = Path(text or "data/run").resolve()
        if path.exists():
            if not path.is_dir():
                errors.append("Output path is a file, not a directory.")
            elif next(path.iterdir(), None) is not None:
                warnings.append("Output folder is not empty. Choose a new folder to avoid mixing runs or overwriting data.")
        ancestor = path
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        if not ancestor.is_dir():
            errors.append("An output-path parent is a file, not a directory.")
        elif not os.access(ancestor, os.W_OK):
            errors.append("Output directory or its nearest existing parent is not writable.")
        return str(path)
    except (OSError, ValueError) as error:
        errors.append(f"Cannot use output folder: {error}")
        return text


def _cell_counts(sources, position_count, batch, errors):
    """Count selected anchors and pattern exposures without aiming hardware."""
    cells = 0
    collections = 0
    for source in sources:
        if "cell" not in str(getattr(source, "name", "")).lower():
            continue
        try:
            data = np.asarray(source._points.data, dtype=float)
            if data.size == 0:
                continue
            pos_axis = int(source._pos_idx)
            if data.ndim != 2 or not 0 <= pos_axis < data.shape[1] - 2:
                raise ValueError("invalid point dimensions")
            if not np.isfinite(data).all():
                raise ValueError("coordinates must be finite")
            positions = data[:, pos_axis]
            if np.any((positions < 0) | (positions >= position_count)
                      | (positions != np.floor(positions))):
                raise ValueError("point position indices are outside the prepared sequence")
            multiplier = float(source.transformer.multiplier)
            if not math.isfinite(multiplier) or multiplier < 1 or not multiplier.is_integer():
                raise ValueError("invalid aiming-pattern point count")
            if batch and multiplier < 2:
                raise ValueError("integrated batch mode needs at least two pattern points")
            cells += len(data)
            collections += len(data) * (1 if batch else int(multiplier))
        except (AttributeError, TypeError, ValueError) as error:
            errors.append(f"Cannot review cell source {getattr(source, 'name', '')!r}: {error}. Run selection again.")
    if cells == 0:
        errors.append("No selected cell/target points. Prepare a selection or stage grid first.")
    return cells, collections


def _settings_signature(owner, seq, sources):
    """Detect edits during the modal review, including equal-count point moves."""
    digest = sha256()
    digest.update(repr((id(owner.core), id(owner.collector),
                        id(owner.transformer), seq)).encode())
    for name in sorted(vars(owner)):
        # Include acquisition/selection controls, but not UI-only containers.
        if not (name.startswith(("mda_", "sel_")) or name == "cfg_path"):
            continue
        if not name.endswith(("_input", "_check", "_combo", "_path")) and name != "cfg_path":
            continue
        control = getattr(owner, name)
        for getter in ("currentText", "value", "isChecked", "text"):
            method = getattr(control, getter, None)
            if callable(method):
                digest.update(repr((name, method())).encode())
                break
    for source in sources:
        layer = getattr(source, "_points", None)
        if layer is not None:
            data = np.asarray(layer.data)
            digest.update(repr((id(source), data.shape, str(data.dtype))).encode())
            digest.update(data.tobytes())
    return digest.hexdigest()


def build_mda_review(owner, *, demo=False):
    """Inspect controls and prepared sources, without writing files or starting hardware."""
    errors, warnings, rows = [], [], []
    if owner.core is None:
        errors.append("Not connected. Connect the microscope first.")
    if owner.collector is None or owner.transformer is None:
        errors.append("Collector or coordinate transformer is missing. Reconnect with both configured.")
    _check_busy(owner, errors)
    output = _check_output(owner.mda_dir_input.text().strip(), errors, warnings)
    rows.append(("Output folder", output))

    selection = owner.selection_results or {}
    seq = selection.get("new_seq")
    sources = selection.get("sources", [])
    positions = len(seq.stage_positions) if seq is not None else 0
    if not positions or not sources or "autofocus_p" not in selection:
        errors.append("No complete prepared selection/sequence. Run automated selection, manual selection, or Generate stage grid first.")
    if seq is not None:
        if seq.time_plan is None:
            errors.append("Prepared sequence has no time plan. Prepare the selection again.")
        if not demo and not any(channel.config == "BF" for channel in seq.channels):
            errors.append("Prepared sequence needs a BF channel for Raman acquisition. Prepare the selection again.")
    rows.append(("Stage positions", f"{positions:,}"))

    batch = selection.get("batch", owner.sel_batch_combo.currentText() == "True")
    cells, collections = _cell_counts(sources, positions, batch, errors)
    rows.extend([
        ("Selected cells / target anchors", f"{cells:,} (current selection)"),
        ("Acquisition mode", "Integrated batch" if batch else "Individual pattern points"),
    ])
    af_choice = selection.get("autofocus_object", owner.sel_af_combo.currentText())
    autofocus = str(af_choice or "").lower() not in ("none", "")
    tracking = owner.mda_seg_track_check.isChecked()
    if tracking and owner.mda_auto_add_cells_check.isChecked() and batch:
        errors.append("Automatically adding new cells is not supported in integrated batch mode.")
    if tracking:
        warnings.append("Tracking can change target counts; timing uses only the current selection.")
    try:
        afp, imgp = resolve_position_specs(
            owner.mda_afp_input.text(), owner.mda_imgp_input.text(),
            positions, selection.get("autofocus_p", []),
        )
        if any(p < 0 or p >= positions for p in afp + imgp):
            raise ValueError("Prepared autofocus/imaging indices are outside the sequence. Run selection again.")
        rows.append(("Autofocus", f"{af_choice}, {len(set(afp))} positions" if autofocus else "Off"))
        rows.append(("Imaging positions", f"{len(set(imgp))} of {positions}"))
        if autofocus and not afp:
            warnings.append("Autofocus is enabled but no autofocus positions are selected.")
    except (TypeError, ValueError) as error:
        errors.append(str(error))
    rows.append(("Segment and track", "On" if tracking else "Off"))
    rows.append(("Refocus / segment cadence", f"Every {owner.mda_refocus_input.value()} time point(s)"))

    exposure = float(owner.mda_exp_input.value())
    loops = int(owner.mda_loops_input.value())
    interval = float(owner.mda_interval_input.value())
    rows.extend([
        ("Time points", f"{loops:,}"),
        ("Interval", f"{interval:g} s (start-to-start)"),
        ("Raman exposure", f"{exposure:g} ms per {'integrated collection' if batch else 'pattern point'}"),
        ("Raman glass offset", f"{owner.mda_raman_off_input.value():g} µm"),
    ])
    if not math.isfinite(exposure) or exposure < 73.8:
        errors.append("Raman exposure must be at least 73.8 ms per collection.")
    if loops < 1 or not math.isfinite(interval) or interval < 0:
        errors.append("Use at least one time point and a finite, non-negative interval.")

    minimum = None
    try:
        z = _numbers(owner.mda_zrel_input.text(), "Z relative")
        rz = _numbers(owner.mda_rz_input.text(), "Raman Z indices", integer=True)
        if any(index < 0 or index >= len(z) for index in rz):
            raise ValueError(f"Raman Z indices must be between 0 and {len(z) - 1}.")
        rows.append(("Z planes", f"{len(z)}; offsets: {', '.join(f'{v:g}' for v in z)} µm"))
        rows.append(("Raman Z planes", f"{len(set(rz))}; indices: {', '.join(map(str, sorted(set(rz))))}"))
        if len(set(rz)) != len(rz):
            warnings.append("Repeated Raman Z indices select the same plane, not repeat exposures.")
        count = collections * len(set(rz)) * loops
        rows.append(("Planned cell Raman collections", f"{count:,} (current targets only)"))
        per_loop = collections * len(set(rz)) * exposure / 1000
        if cells and loops > 0 and math.isfinite(per_loop) and per_loop > 0 and math.isfinite(interval) and interval >= 0:
            minimum = (loops - 1) * max(interval, per_loop) + per_loop
            rows.append(("Estimated minimum duration", f"At least {_duration(minimum)}"))
            if loops > 1 and interval > 0 and per_loop > interval:
                warnings.append("Raman exposures alone exceed the requested interval; the acquisition will run behind schedule.")
    except ValueError as error:
        errors.append(str(error))
    if minimum is None:
        rows.append(("Estimated minimum duration", "Unavailable until selection/settings are valid"))
    warnings.append("Timing includes current cell Raman exposures and scheduled intervals only. Imaging, autofocus, other sources, motion, settling, readout, processing, and saving add time; this is not a completion-time prediction.")

    channels, seen = [], set()
    for entry in owner.mda_channel_rows:
        name = entry["combo"].currentText()
        if entry["combo"].isEnabled() and name and name not in seen:
            seen.add(name)
            channels.append(f"{name} ({entry['exp'].value():g} ms)")
    rows.append(("Extra fluorescence channels", ", ".join(channels) if channels else "None"))
    if demo:
        rows.insert(0, ("Mode", "Demo — simulated acquisition"))
        if channels:
            warnings.append("Demo MDA uses BF/RM channels only; extra fluorescence channels are not acquired.")
    return MdaReview(tuple(rows), tuple(errors), tuple(warnings), minimum,
                     _settings_signature(owner, seq, sources))


class MdaReviewDialog(QDialog):
    def __init__(self, review, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Review Raman MDA")
        self.resize(660, 650)
        layout = QVBoxLayout(self)
        heading = QLabel("Fix the issues below before starting." if review.errors
                         else "Review this experiment before starting acquisition.")
        heading.setWordWrap(True)
        layout.addWidget(heading)
        self.summary = QTextBrowser()
        self.summary.setPlainText(review.text)
        layout.addWidget(self.summary, 1)
        buttons = QDialogButtonBox()
        self.back_button = buttons.addButton("Back to settings", QDialogButtonBox.RejectRole)
        self.start_button = buttons.addButton("Start acquisition", QDialogButtonBox.AcceptRole)
        self.start_button.setEnabled(not review.errors)
        self.start_button.setAutoDefault(False)
        self.back_button.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def confirm_raman_mda(owner, *, demo=False):
    """Review first; cancellation and validation failures never start hardware."""
    if getattr(owner, "_mda_review_open", False):
        return False
    owner._mda_review_open = True
    dialog = None
    try:
        review = build_mda_review(owner, demo=demo)
        dialog = MdaReviewDialog(review, owner)
        accepted = dialog.exec_() == QDialog.Accepted
        if review.errors:
            owner.status.setText(f"Status: MDA not started — {review.errors[0]}")
            return False
        if not accepted:
            owner.status.setText("Status: MDA cancelled before starting; settings kept")
            return False
        # Modal dialogs still process Qt events. Recheck in case a background
        # task changed connection, selection, or settings during the review.
        if build_mda_review(owner, demo=demo) != review:
            owner.status.setText("Status: MDA setup changed during review; click Run Raman MDA to review again")
            return False
        owner._confirmed_mda_summary = review.text
        return True
    except Exception as error:
        owner.status.setText(f"Status: cannot review MDA — {error}")
        return False
    finally:
        owner._mda_review_open = False
        if dialog is not None:
            dialog.deleteLater()
