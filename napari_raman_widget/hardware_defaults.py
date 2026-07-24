"""Discovery and parsing for machine-local HardwareWidget defaults."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULTS_ENV_VAR",
    "DEFAULTS_FILENAME",
    "load_hardware_defaults",
    "resolve_defaults_path",
]


DEFAULTS_ENV_VAR = "NAPARI_RAMAN_DEFAULTS"
DEFAULTS_FILENAME = "napari_raman_defaults.json"


def _candidate_paths() -> list[Path]:
    """Return defaults-file candidates in priority order."""
    candidates: list[Path] = []
    configured = os.getenv(DEFAULTS_ENV_VAR, "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())

    candidates.extend(
        (
            Path.cwd() / DEFAULTS_FILENAME,
            Path(__file__).resolve().parents[1] / DEFAULTS_FILENAME,
        )
    )

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def load_hardware_defaults(
    path: str | Path | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    """Load the first available defaults file.

    An explicit ``path`` takes priority. Otherwise, the path named by
    ``NAPARI_RAMAN_DEFAULTS`` is checked before the current/project directory.
    Missing automatically discovered files are normal and return no defaults.
    """
    if path is not None:
        candidates = [Path(path).expanduser().resolve()]
        required = True
    else:
        candidates = _candidate_paths()
        required = bool(os.getenv(DEFAULTS_ENV_VAR, "").strip())

    for index, candidate in enumerate(candidates):
        if not candidate.is_file():
            if required and index == 0:
                raise FileNotFoundError(
                    f"Hardware defaults file not found: {candidate}"
                )
            continue

        with candidate.open("r", encoding="utf-8") as stream:
            values = json.load(stream)
        if not isinstance(values, dict):
            raise ValueError(
                f"Hardware defaults must be a JSON object: {candidate}"
            )
        return candidate, values

    return None, {}


def resolve_defaults_path(value: Any, defaults_file: Path) -> str:
    """Resolve one configured path relative to the defaults file."""
    text = str(value).strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = defaults_file.parent / path
    return str(path.resolve())
