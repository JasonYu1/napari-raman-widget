"""Runtime retry protection for the shared Micro-Manager core.

The guard patches only the in-process ``CMMCorePlus`` singleton instance. It
does not modify pymmcore-plus, napari-micromanager, or any other installed
package. Widgets already holding the singleton see the guarded methods as soon
as :func:`install_core_guard` is called.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from functools import wraps
from typing import Any

from .acquisition.autofocus import _retry_core_operation

__all__ = ["CoreRetryGuard", "install_core_guard"]


logger = logging.getLogger(__name__)

_READ_OPERATION = re.compile(r"^(?:get|has|is)[A-Z]")

# Read-only calls and absolute/idempotent commands may be retried safely. Do
# not add relative moves or run_mda: retrying either may duplicate an action.
_SAFE_COMMANDS = {
    "enableContinuousFocus",
    "setAutoFocusOffset",
    "setAutoShutter",
    "setChannelGroup",
    "setConfig",
    "setExposure",
    "setPixelSizeUm",
    "setPosition",
    "setProperty",
    "setROI",
    "setShutterOpen",
    "setState",
    "setXYPosition",
    "setZPosition",
    "snapImage",
    "startContinuousSequenceAcquisition",
    "stopSequenceAcquisition",
    "waitForDevice",
    "waitForSystem",
}

# Match the synchronization behavior of the explicit try_* helpers in
# acquisition/autofocus.py. Image-buffer reads are intentionally absent: a
# waitForSystem call during continuous acquisition could stall live mode.
_WAIT_AFTER_COMMAND = {
    "setAutoShutter",
    "setConfig",
    "setExposure",
    "setPosition",
    "setShutterOpen",
    "setXYPosition",
    "setZPosition",
    "snapImage",
    "stopSequenceAcquisition",
}


class CoreRetryGuard:
    """Retry transient ``RuntimeError`` failures on one MMCore instance."""

    _MARKER = "_napari_raman_core_retry_guard"

    def __init__(
        self,
        core: Any,
        *,
        attempts: int = 3,
        initial_delay: float = 0.0,
        delay_increment: float = 0.5,
    ) -> None:
        if attempts < 1:
            raise ValueError("attempts must be at least one")
        if initial_delay < 0:
            raise ValueError("initial_delay cannot be negative")
        if delay_increment < 0:
            raise ValueError("delay_increment cannot be negative")

        self.core = core
        self.attempts = int(attempts)
        self.initial_delay = float(initial_delay)
        self.delay_increment = float(delay_increment)
        self.originals: dict[str, Callable[..., Any]] = {}

    def install(self) -> CoreRetryGuard:
        """Patch safe operations and return the active guard."""
        existing = getattr(self.core, self._MARKER, None)
        if isinstance(existing, CoreRetryGuard):
            return existing

        names = {
            name
            for name in dir(self.core)
            if _READ_OPERATION.match(name) or name in _SAFE_COMMANDS
        }

        # Capture every original first. Some convenience methods call other
        # core methods, which should resolve to the installed wrappers.
        for name in sorted(names):
            method = getattr(self.core, name, None)
            if callable(method):
                self.originals[name] = method

        installed: list[str] = []
        try:
            for name, original in self.originals.items():
                setattr(self.core, name, self._make_wrapper(name, original))
                installed.append(name)
            setattr(self.core, self._MARKER, self)
        except Exception:
            for name in installed:
                setattr(self.core, name, self.originals[name])
            raise

        logger.info(
            "Installed Raman core retry guard on %d MMCore operations",
            len(installed),
        )
        return self

    def uninstall(self) -> None:
        """Restore original methods, primarily for tests and shutdown."""
        if getattr(self.core, self._MARKER, None) is not self:
            return
        for name, original in self.originals.items():
            setattr(self.core, name, original)
        delattr(self.core, self._MARKER)

    def _make_wrapper(
        self,
        name: str,
        original: Callable[..., Any],
    ) -> Callable[..., Any]:
        @wraps(original)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            failure_count = 0

            def operation() -> Any:
                nonlocal failure_count
                try:
                    result = original(*args, **kwargs)
                    if name in _WAIT_AFTER_COMMAND:
                        wait_for_system = self.originals.get("waitForSystem")
                        if wait_for_system is not None:
                            wait_for_system()
                    return result
                except RuntimeError as error:
                    failure_count += 1
                    logger.warning(
                        "core.%s failed (attempt %d/%d): %s",
                        name,
                        failure_count,
                        self.attempts,
                        error,
                    )
                    raise

            return _retry_core_operation(
                f"core.{name}",
                operation,
                attempts=self.attempts,
                initial_delay=self.initial_delay,
                delay_increment=self.delay_increment,
            )

        return guarded


def install_core_guard(
    core: Any,
    *,
    attempts: int = 3,
    initial_delay: float = 0.0,
    delay_increment: float = 0.5,
) -> CoreRetryGuard:
    """Install retry protection on the shared core, once per process."""
    existing = getattr(core, CoreRetryGuard._MARKER, None)
    if isinstance(existing, CoreRetryGuard):
        return existing
    return CoreRetryGuard(
        core,
        attempts=attempts,
        initial_delay=initial_delay,
        delay_increment=delay_increment,
    ).install()
