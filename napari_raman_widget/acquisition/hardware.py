"""Hardware connection and cleanup helpers."""

from __future__ import annotations

import time
from typing import Any

__all__ = ["unload"]


def _disconnect_core_events(core: Any) -> None:
    """Disconnect callbacks attached to relevant core events."""
    event_names = (
        "channelGroupChanged",
        "configGroupChanged",
        "propertyChanged",
        "systemConfigurationLoaded",
        "configSet",
    )

    events = getattr(core, "events", None)

    if events is None:
        return

    for event_name in event_names:
        signal = getattr(
            events,
            event_name,
            None,
        )

        if signal is None:
            continue

        try:
            signal.disconnect()
        except Exception:
            # The signal may already be disconnected.
            continue


def unload(
    core: Any,
    attempts: int = 20,
    initial_delay: float = 0.1,
    delay_increment: float = 1.0,
) -> None:
    """Disconnect callbacks and unload all microscope devices.

    Parameters
    ----------
    core
        Microscope core whose devices should be unloaded.
    attempts
        Maximum number of unload attempts.
    initial_delay
        Delay before the first attempt, in seconds.
    delay_increment
        Additional delay added after each failed attempt.

    Raises
    ------
    RuntimeError
        If the devices cannot be unloaded after all attempts.
    """
    if attempts < 1:
        raise ValueError(
            "attempts must be at least one."
        )

    delay = float(initial_delay)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            time.sleep(delay)
            _disconnect_core_events(core)
            core.unloadAllDevices()
            core.waitForSystem()
            return

        except Exception as error:
            last_error = error

            if attempt < attempts:
                delay += delay_increment

    raise RuntimeError(
        "Could not unload the microscope devices "
        f"after {attempts} attempts."
    ) from last_error