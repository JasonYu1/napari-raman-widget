"""Apply the shared output directory without reconnecting microscope hardware."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from weakref import WeakSet, ref

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QPushButton


_OUTPUT_OWNERS = WeakSet()
_INPUT_PATH_FIELDS = (
    "cfg_path", "tf_path", "sel_vdm_path", "dark_noise_path",
    "spectral_calibration_path", "mda_track_cfg_input", "px2stage_ds_path",
)


def add_output_folder_controls(owner, out_row, loading_layout):
    """Keep folder application explicit: editing or browsing stages a path."""
    owner.apply_output_folder_btn = QPushButton("Apply folder")
    owner.apply_output_folder_btn.setToolTip(
        "Use this folder for new relative saves without reconnecting. "
        "Stop acquisitions and live imaging first."
    )
    owner.apply_output_folder_btn.clicked.connect(owner.apply_output_folder)
    out_row.addWidget(owner.apply_output_folder_btn)
    owner.output_folder_status = QLabel(f"Active folder: {Path.cwd()}")
    owner.output_folder_status.setTextFormat(Qt.PlainText)
    owner.output_folder_status.setWordWrap(True)
    owner.output_folder_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
    loading_layout.addWidget(owner.output_folder_status)
    _OUTPUT_OWNERS.add(owner)
    owner_ref = ref(owner)

    def forget_owner(*_args):
        widget = owner_ref()
        if widget is not None:
            _OUTPUT_OWNERS.discard(widget)

    owner.destroyed.connect(forget_owner)


def _busy_reason(owner):
    if getattr(owner, "_hardware_shutdown_started", False):
        return "hardware is shutting down"
    jobs = getattr(owner, "_acquisition_jobs", None)
    if jobs is not None and jobs.is_running:
        return "an acquisition is still running or finishing"
    if getattr(owner, "_live_raman_worker", None) is not None:
        return "live Raman spectra are running or stopping"
    thread = getattr(owner, "_raman_mda_thread", None)
    if thread is not None and thread.is_alive():
        return "an MDA is still running"
    timer = getattr(owner, "demo_live_timer", None)
    if timer is not None and timer.isActive():
        return "demo live imaging is running"
    core = getattr(owner, "core", None)
    running = getattr(getattr(core, "mda", None), "is_running", False)
    mda_running = running() if callable(running) else running
    if mda_running:
        return "an MDA is still running"
    live = getattr(core, "isSequenceRunning", None)
    if callable(live) and live():
        return "live camera imaging is running"
    return None


def _input_references(owners):
    """Resolve existing inputs before cwd changes; leave output names relative."""
    fields, guards = [], []
    for owner in owners:
        for name in _INPUT_PATH_FIELDS:
            field = getattr(owner, name, None)
            text = field.text().strip() if field is not None else ""
            if text:
                path = Path(text).expanduser()
                if not path.is_absolute() and path.exists():
                    fields.append((field, str(path.resolve())))
        guard = getattr(owner, "core_guard", None)
        config = getattr(guard, "config_file", None)
        if config and not Path(config).is_absolute():
            # Recovery must keep targeting the loaded CFG even if that file
            # has since been moved/deleted; never retarget it to a new cwd.
            guards.append((guard, str(Path(config).expanduser().resolve())))
    return fields, guards


def apply_output_folder(owner, *, show_success=True, preserve_inputs=True):
    """Change cwd only while all registered Raman panels are idle.

    Hardware Connect passes preserve_inputs=False because its relative input
    paths have historically been resolved *inside* the chosen output folder.
    Explicit Apply keeps existing inputs pointing to their original files.
    """
    owners = [owner, *(item for item in _OUTPUT_OWNERS if item is not owner)]
    try:
        for item in owners:
            reason = _busy_reason(item)
            if reason:
                owner.status.setText(
                    f"Status: output folder unchanged — {reason}. "
                    "Stop it or wait before applying a folder."
                )
                return False
    except Exception as error:
        owner.status.setText(
            f"Status: output folder unchanged — cannot check acquisition status: {error}"
        )
        return False

    try:
        text = owner.out_path.text().strip()
        target = Path(text).expanduser().resolve() if text else Path.cwd()
        fields, guards = [], []
        if text:
            reference_owners = owners if preserve_inputs else [
                item for item in owners if item is not owner
            ]
            fields, guards = _input_references(reference_owners)
            target.mkdir(parents=True, exist_ok=True)
            # Changing cwd succeeds for some read-only folders. Check that
            # future saves can create a file before reporting success.
            with tempfile.TemporaryFile(prefix=".raman-output-check-", dir=target):
                pass
            os.chdir(target)
    except (OSError, ValueError, RuntimeError) as error:
        owner.status.setText(f"Status: output folder unchanged — could not apply folder: {error}")
        return False

    for field, path in fields:
        field.setText(path)
    for guard, path in guards:
        guard.set_config_file(path)
    if text:
        owner.out_path.setText(str(target))
    for item in owners:
        label = getattr(item, "output_folder_status", None)
        if label is not None:
            label.setText(f"Active folder: {target}")
    if show_success:
        owner.status.setText(f"Status: output folder applied for new saves → {target}")
    return True
