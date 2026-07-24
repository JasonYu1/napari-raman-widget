"""Runtime retry protection for the shared Micro-Manager core.

The guard patches only the in-process ``CMMCorePlus`` singleton instance. It
does not modify pymmcore-plus, napari-micromanager, or any other installed
package. Widgets already holding the singleton see the guarded methods as soon
as :func:`install_core_guard` is called.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from .acquisition.autofocus import _retry_core_operation

__all__ = ["CoreRetryGuard", "install_core_guard"]


logger = logging.getLogger(__name__)

_READ_OPERATION = re.compile(r"^(?:get|has|is)[A-Z]")

# These methods read transient camera buffers.  Retrying the read by itself is
# unsafe: a configuration reload clears the buffer, so recovery must repeat the
# complete snap-and-read transaction instead.  RamanEngine owns that sequence.
_CAMERA_BUFFER_READS = {
    "getImage",
    "getLastImage",
    "getLastTaggedImage",
    "getNBeforeLastImage",
    "getTaggedImage",
    "popNextImage",
    "popNextTaggedImage",
}

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
        attempts: int = 10,
        initial_delay: float = 0.0,
        delay_increment: float = 0.5,
        config_file: str | os.PathLike[str] | None = None,
        reload_after_failures: int | None = None,
        reload_attempts: int = 3,
        reload_delay: float = 0.1,
    ) -> None:
        if attempts < 1:
            raise ValueError("attempts must be at least one")
        if initial_delay < 0:
            raise ValueError("initial_delay cannot be negative")
        if delay_increment < 0:
            raise ValueError("delay_increment cannot be negative")
        if reload_attempts < 1:
            raise ValueError("reload_attempts must be at least one")
        if reload_delay < 0:
            raise ValueError("reload_delay cannot be negative")

        if reload_after_failures is None:
            reload_after_failures = attempts // 2 if attempts > 1 else None
        if reload_after_failures is not None and not (
            1 <= reload_after_failures < attempts
        ):
            raise ValueError(
                "reload_after_failures must be between one and attempts - 1"
            )

        self.core = core
        self.attempts = int(attempts)
        self.initial_delay = float(initial_delay)
        self.delay_increment = float(delay_increment)
        self.config_file = os.fspath(config_file) if config_file else None
        self.reload_after_failures = reload_after_failures
        self.reload_attempts = int(reload_attempts)
        self.reload_delay = float(reload_delay)
        self.originals: dict[str, Callable[..., Any]] = {}
        self._reload_lock = threading.RLock()
        self._reloading = False
        self._pause_lock = threading.RLock()
        self._pause_count = 0
        self._reload_methods = {
            name: getattr(core, name, None)
            for name in (
                "loadSystemConfiguration",
                "stopSequenceAcquisition",
                "unloadAllDevices",
            )
        }

    def install(self) -> CoreRetryGuard:
        """Patch safe operations and return the active guard."""
        existing = getattr(self.core, self._MARKER, None)
        if isinstance(existing, CoreRetryGuard):
            return existing

        names = {
            name
            for name in dir(self.core)
            if (
                _READ_OPERATION.match(name) or name in _SAFE_COMMANDS
            )
            and name not in _CAMERA_BUFFER_READS
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

    def set_config_file(
        self,
        config_file: str | os.PathLike[str] | None,
    ) -> None:
        """Set or clear the configuration used for automatic recovery."""
        self.config_file = os.fspath(config_file) if config_file else None

    @property
    def paused(self) -> bool:
        """Whether callers currently bypass retry and reload wrappers."""
        with self._pause_lock:
            return self._pause_count > 0

    def pause(self) -> None:
        """Temporarily let the active acquisition engine own recovery."""
        with self._pause_lock:
            self._pause_count += 1

    def resume(self) -> None:
        """Release one matching :meth:`pause` call."""
        with self._pause_lock:
            if self._pause_count == 0:
                return
            self._pause_count -= 1

    def reload_configuration(self) -> bool:
        """Reload the configured MMCore system using unwrapped methods."""
        if not self.config_file:
            logger.warning(
                "Core recovery requested, but no Micro-Manager config is set"
            )
            return False

        unload = self._reload_methods.get("unloadAllDevices")
        load = self._reload_methods.get("loadSystemConfiguration")
        if not callable(unload) or not callable(load):
            logger.warning("Core does not expose configuration reload methods")
            return False

        with self._reload_lock:
            if self._reloading:
                return False
            self._reloading = True
            try:
                last_error: Exception | None = None
                for reload_attempt in range(1, self.reload_attempts + 1):
                    try:
                        if self.reload_delay > 0:
                            time.sleep(self.reload_delay * reload_attempt)
                        logger.warning(
                            "Reloading Micro-Manager config %s (attempt %d/%d)",
                            self.config_file,
                            reload_attempt,
                            self.reload_attempts,
                        )
                        stop = self._reload_methods.get(
                            "stopSequenceAcquisition"
                        )
                        if callable(stop):
                            try:
                                stop()
                            except Exception:
                                pass
                        unload()
                        load(self.config_file)
                        wait = self.originals.get("waitForSystem")
                        if wait is not None:
                            wait()

                        # Match the rig warm-up used by HardwareWidget and the
                        # Raman engine, without making recovery fail on a rig
                        # that does not define one of these channel presets.
                        set_config = self.originals.get("setConfig")
                        if set_config is not None:
                            for channel in ("GFP", "BF"):
                                try:
                                    set_config("Channel", channel)
                                    if wait is not None:
                                        wait()
                                except Exception as error:
                                    logger.warning(
                                        "Channel warm-up %s failed: %s",
                                        channel,
                                        error,
                                    )
                        logger.warning("Micro-Manager config reload succeeded")
                        return True
                    except Exception as error:
                        last_error = error
                        logger.warning(
                            "Micro-Manager config reload attempt %d failed: %s",
                            reload_attempt,
                            error,
                        )
                logger.error(
                    "Micro-Manager config reload failed after %d attempts: %s",
                    self.reload_attempts,
                    last_error,
                )
                return False
            finally:
                self._reloading = False

    def _make_wrapper(
        self,
        name: str,
        original: Callable[..., Any],
    ) -> Callable[..., Any]:
        @wraps(original)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            if self.paused:
                return original(*args, **kwargs)

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
                    if (
                        not self._reloading
                        and self.config_file is not None
                        and self.reload_after_failures is not None
                        and failure_count == self.reload_after_failures
                    ):
                        self.reload_configuration()
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
    attempts: int = 10,
    initial_delay: float = 0.0,
    delay_increment: float = 0.5,
    config_file: str | os.PathLike[str] | None = None,
    reload_after_failures: int | None = None,
    reload_attempts: int = 3,
    reload_delay: float = 0.1,
) -> CoreRetryGuard:
    """Install retry protection on the shared core, once per process."""
    existing = getattr(core, CoreRetryGuard._MARKER, None)
    if isinstance(existing, CoreRetryGuard):
        existing.set_config_file(config_file)
        return existing
    return CoreRetryGuard(
        core,
        attempts=attempts,
        initial_delay=initial_delay,
        delay_increment=delay_increment,
        config_file=config_file,
        reload_after_failures=reload_after_failures,
        reload_attempts=reload_attempts,
        reload_delay=reload_delay,
    ).install()
