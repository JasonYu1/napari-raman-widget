"""Orderly MMCore shutdown helpers that do not require Qt."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

__all__ = ["HardwareShutdownResult", "shutdown_core_hardware"]


@dataclass(frozen=True)
class HardwareShutdownResult:
    """Outcome of one best-effort hardware shutdown."""

    mda_thread_stopped: bool
    devices_unloaded: bool
    errors: tuple[str, ...]


def _core_method(core: Any, guard: Any, name: str) -> Callable[..., Any] | None:
    """Return an unwrapped Core method when a retry guard is installed."""
    originals = getattr(guard, "originals", None)
    if isinstance(originals, dict):
        original = originals.get(name)
        if callable(original):
            return original
    method = getattr(core, name, None)
    return method if callable(method) else None


def shutdown_core_hardware(
    core: Any,
    *,
    guard: Any = None,
    mda_thread: Any = None,
    join_timeout: float = 5.0,
    release_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> HardwareShutdownResult:
    """Cancel acquisition and release device handles before Python exits.

    Retry wrappers are deliberately bypassed. A shutdown failure must never
    trigger a configuration reload while the application is closing.
    """
    errors: list[str] = []

    mda = getattr(core, "mda", None)
    cancel = getattr(mda, "cancel", None)
    if callable(cancel):
        try:
            cancel()
        except Exception as error:
            errors.append(f"cancel MDA: {error}")

    stop = _core_method(core, guard, "stopSequenceAcquisition")
    if stop is not None:
        try:
            stop()
        except Exception as error:
            errors.append(f"stop sequence acquisition: {error}")

    thread_stopped = True
    if (
        mda_thread is not None
        and mda_thread is not threading.current_thread()
        and callable(getattr(mda_thread, "is_alive", None))
        and mda_thread.is_alive()
    ):
        try:
            mda_thread.join(timeout=max(0.0, float(join_timeout)))
        except Exception as error:
            errors.append(f"join MDA thread: {error}")
        thread_stopped = not mda_thread.is_alive()
        if not thread_stopped:
            errors.append("MDA thread did not stop before shutdown timeout")

    # Stop once more after the MDA worker has had an opportunity to finish its
    # current event. This also handles a late camera-sequence start.
    if stop is not None:
        try:
            stop()
        except Exception as error:
            errors.append(f"final stop sequence acquisition: {error}")

    unload = _core_method(core, guard, "unloadAllDevices")
    devices_unloaded = False
    if unload is None:
        errors.append("unloadAllDevices is unavailable")
    else:
        try:
            unload()
            devices_unloaded = True
        except Exception as error:
            errors.append(f"unload devices: {error}")

    if devices_unloaded and release_delay > 0:
        sleep(float(release_delay))

    return HardwareShutdownResult(
        mda_thread_stopped=thread_stopped,
        devices_unloaded=devices_unloaded,
        errors=tuple(errors),
    )
