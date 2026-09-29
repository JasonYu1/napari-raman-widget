"""Cooperative cancellation and progress contracts for hardware work."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ProgressCallback = Callable[[int, int, str], None]
CancelCheck = Callable[[], bool]


class AcquisitionCancelled(Exception):
    """A user requested a stop at a safe boundary between acquisitions."""

    partial_result: dict[str, Any] | None = None


def check_cancelled(cancel_check: CancelCheck | None) -> None:
    """Stop before the next hardware action, never midway through exposure."""
    if cancel_check is not None and cancel_check():
        raise AcquisitionCancelled("Acquisition stopped by the user.")
