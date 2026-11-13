"""Exclusive full-configuration recovery for the local Raman engine wrapper.

Only ordinary MMCore calls belong inside the hardware gate. In particular,
recovery must not wait for Qt layer callbacks or join hardware-calling threads.
"""

from __future__ import annotations

import math
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from .core_access import install_core_access
from .serial_port_diagnostics import diagnose_serial_port


class ReloadCancelled(RuntimeError):
    """Recovery was interrupted by acquisition cancellation or shutdown."""


class ReloadDiagnostics:
    """Bound and rate-limit passive COM17 checks, including slow native calls."""

    def __init__(self, report: Callable[[str], None]) -> None:
        self.report = report
        self._last_check = float("-inf")
        self._lock = threading.Lock()

    def check_com17(self, reason: str) -> None:
        # A watchdog and a failed load can arrive together. Neither may wait
        # for the other diagnostic, and repeated retries must not flood logs.
        if not self._lock.acquire(blocking=False):
            return
        try:
            now = time.monotonic()
            if now - self._last_check < 30.0:
                return
            self._last_check = now
            self.report(f"[COM17 diagnostics] {reason}; checking open handles")
            try:
                details = diagnose_serial_port("COM17", timeout=5.0)
            except Exception as error:
                details = f"Handle inspection unavailable: {error}"
            self.report(f"[COM17 diagnostics]\n{details}")
        finally:
            self._lock.release()

    def failed(self, error: Exception) -> None:
        if "COM17" in str(error).upper():
            self.check_com17(f"configuration reload failed: {error}")

    @contextmanager
    def watch(self, stage: str, *, timeout: float = 30.0) -> Iterator[None]:
        """Report a slow native call without calling the core from the timer."""
        finished = threading.Event()

        def stalled() -> None:
            if not finished.is_set():
                self.check_com17(f"{stage} still running after {timeout:g} s")

        timer = threading.Timer(timeout, stalled)
        timer.daemon = True
        timer.start()
        try:
            yield
        finally:
            finished.set()
            timer.cancel()


class ConfigurationReloader:
    """Keep full recovery exclusive through unloading, settling and warm-up."""

    def __init__(self, core: Any, report: Callable[[str], None]) -> None:
        self.core = core
        self.access = install_core_access(core)
        self.report = report
        self.diagnostics = ReloadDiagnostics(report)
        self._cancel = threading.Event()

    def request_shutdown(self) -> None:
        # No core call or lock acquisition: this must work while recovery owns
        # the core, including when the GUI is closing the application.
        self._cancel.set()

    def _check_cancelled(self) -> None:
        mda = getattr(self.core, "mda", None)
        is_running = getattr(mda, "is_running", None)
        # pymmcore-plus has cancel() but no public cancellation-state accessor.
        # Read its flag only while a sequence is active; do not consume/reset it.
        mda_cancelled = (
            callable(is_running)
            and is_running()
            and getattr(mda, "_canceled", False) is True
        )
        if self._cancel.is_set() or mda_cancelled:
            raise ReloadCancelled("configuration reload cancelled")

    def _wait(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while True:
            self._check_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._cancel.wait(min(0.1, remaining))

    def reload(
        self,
        config_file: str | None,
        *,
        attempts: int = 100,
        post_unload_delay: float = 2.0,
    ) -> None:
        if not isinstance(attempts, int) or attempts < 1:
            raise ValueError("N must be an integer of at least one")
        if not math.isfinite(post_unload_delay) or post_unload_delay < 0:
            raise ValueError("post_unload_delay must be finite and non-negative")

        guard = getattr(self.core, "_napari_raman_core_retry_guard", None)
        self.report("[hardware recovery] Full reload requested; other hardware calls will wait")
        with self.access.exclusive():
            # The engine owns recovery. Nested pausing preserves the widget's
            # existing acquisition-wide pause and prevents recursive reloads.
            if guard is not None:
                guard.pause()
            try:
                last_error: Exception | None = None
                for attempt in range(1, attempts + 1):
                    self._wait(0.1 + (attempt - 1))
                    self.report(f"reloading config (attempt {attempt}/{attempts})...")
                    try:
                        stop = getattr(self.core, "stopSequenceAcquisition", None)
                        if callable(stop):
                            with self.diagnostics.watch("stopping camera sequence"):
                                try:
                                    stop()
                                except Exception as error:
                                    self.report(f"[hardware recovery] Camera stop failed; continuing full unload: {error}")
                        self._check_cancelled()
                        with self.diagnostics.watch("unloading devices"):
                            self.core.unloadAllDevices()
                        self.report(f"[hardware recovery] Devices unloaded; waiting {post_unload_delay:g} s for serial handles to release")
                        self._wait(post_unload_delay)
                        with self.diagnostics.watch("loading configuration"):
                            self.core.loadSystemConfiguration(config_file or "test3.cfg")
                        self._check_cancelled()
                        with self.diagnostics.watch("waiting for devices and warming up channels"):
                            self.core.waitForSystem()
                            for channel in ("GFP", "BF"):
                                self._check_cancelled()
                                self.core.setConfig("Channel", channel)
                                self.core.waitForSystem()
                        self.report("configuration reload succeeded; hardware access resumed after recovery")
                        return
                    except ReloadCancelled:
                        raise
                    except Exception as error:
                        last_error = error
                        self.report(f"configuration reload failed (attempt {attempt}/{attempts}): {error}")
                        self.diagnostics.failed(error)
                raise RuntimeError(
                    f"configuration reload failed after {attempts} attempts: {last_error}"
                ) from last_error
            except ReloadCancelled:
                self.report("[hardware recovery] Reload cancelled; releasing hardware access")
                raise
            finally:
                if guard is not None:
                    guard.resume()
