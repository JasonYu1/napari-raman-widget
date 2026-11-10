"""Machine-local RACER preferences, independent of the optional library."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path


__all__ = [
    "calibration_paths_path",
    "calibration_settings_path",
    "load_calibration_paths",
    "load_calibration_options",
    "save_calibration_paths",
    "save_calibration_options",
]

_FILENAME = "racer-calibration-settings.json"
_GROUPS = {"image", "spectrum_dark", "spectrum_reference"}


def calibration_settings_path() -> Path:
    """Return the same preference location for hardware and demo widgets."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME")
        root = Path(base) if base else Path.home() / ".local" / "share"
    return root / "napari-raman-widget" / _FILENAME


def calibration_paths_path() -> Path:
    """Return the remembered-files document beside the fit preferences."""
    return calibration_settings_path().with_name("racer-calibration-files.json")


def _copy_options(options: Mapping) -> dict[str, dict[str, int | float]]:
    """Validate the stored shape without importing or depending on RACER."""
    if not isinstance(options, Mapping):
        raise ValueError("RACER calibration settings must be a JSON object")
    copied = {}
    for group, values in options.items():
        if group not in _GROUPS:
            raise ValueError(f"unknown RACER calibration settings group: {group!r}")
        if not isinstance(values, Mapping):
            raise ValueError(f"RACER {group} settings must be a parameter object")
        copied[group] = {}
        for name, value in values.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("RACER calibration parameter names must be nonempty strings")
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{group}.{name} must be a finite number")
            if isinstance(value, Integral):
                copied[group][name] = int(value)
            else:
                if not math.isfinite(value):
                    raise ValueError(f"{group}.{name} must be finite")
                copied[group][name] = float(value)
    return copied


def _destination(path: str | Path | None) -> Path:
    return (
        calibration_settings_path() if path is None else Path(path).expanduser()
    ).resolve()


def load_calibration_options(
    path: str | Path | None = None,
) -> dict[str, dict[str, int | float]]:
    """Read saved options; a missing file means use RACER's defaults.

    Malformed data raises ``ValueError`` and file access errors propagate.
    Parameter support and fit constraints are checked when RACER is used.
    """
    destination = _destination(path)
    try:
        with destination.open("r", encoding="utf-8") as stream:
            options = json.load(stream)
    except FileNotFoundError:
        return {}
    return _copy_options(options)


def save_calibration_options(
    options: Mapping,
    path: str | Path | None = None,
) -> Path:
    """Atomically save fit options and return the destination file.

    This replaces only the RACER preference document. Saving an empty mapping
    remembers Restore Defaults. Validation occurs before touching the file.
    """
    copied = _copy_options(options)
    destination = _destination(path)
    return _atomic_write(destination, copied)


def _stored_paths(values: Mapping) -> dict[str, str | None]:
    """Validate path strings without checking or opening either Zarr store."""
    if not isinstance(values, Mapping) or set(values) != {
        "closed_shutter", "open_reference"
    }:
        raise ValueError(
            "RACER calibration files must contain closed_shutter and open_reference"
        )
    copied = {}
    for name, value in values.items():
        if value is None:
            copied[name] = None
            continue
        if (
            not isinstance(value, str) or not value.strip() or "\0" in value
            or not Path(value).is_absolute()
        ):
            raise ValueError(f"RACER {name} must be an absolute path")
        copied[name] = value
    return copied


def load_calibration_paths(
    path: str | Path | None = None,
) -> dict[str, str | None]:
    """Read remembered paths only; never load or fit a calibration.

    A missing preference file returns an empty mapping. The selected stores may
    be offline or moved; their existence is checked only on a manual load.
    """
    destination = (
        calibration_paths_path() if path is None else Path(path).expanduser()
    ).resolve()
    try:
        with destination.open("r", encoding="utf-8") as stream:
            values = json.load(stream)
    except FileNotFoundError:
        return {}
    return _stored_paths(values)


def save_calibration_paths(
    closed_shutter: str | Path | None,
    open_reference: str | Path | None,
    path: str | Path | None = None,
) -> Path:
    """Remember the selected files atomically without reading their contents."""
    values = {}
    for name, value in (
        ("closed_shutter", closed_shutter), ("open_reference", open_reference)
    ):
        if value is None:
            values[name] = None
            continue
        if (
            not isinstance(value, (str, Path)) or not str(value).strip()
            or "\0" in str(value)
        ):
            raise ValueError(f"RACER {name} must be a nonempty path")
        values[name] = str(Path(value).expanduser().resolve())
    copied = _stored_paths(values)
    destination = (
        calibration_paths_path() if path is None else Path(path).expanduser()
    ).resolve()
    return _atomic_write(destination, copied)


def _atomic_write(destination: Path, values: Mapping) -> Path:
    """Replace one preference document, keeping the previous file on failure."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(values, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
