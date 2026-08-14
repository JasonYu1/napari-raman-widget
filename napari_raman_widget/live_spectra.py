"""Background worker for interruptible, one-frame-at-a-time live spectra."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from qtpy.QtCore import QObject, Signal, Slot


class LiveSpectrumWorker(QObject):
    """Repeatedly call an acquisition function until a stop is requested."""

    spectrum_ready = Signal(object, int, float)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        acquire_one: Callable[[], object],
        *,
        minimum_cycle_seconds: float = 0.0,
    ) -> None:
        super().__init__()
        self._acquire_one = acquire_one
        self._minimum_cycle_seconds = max(0.0, float(minimum_cycle_seconds))
        self._stop_requested = threading.Event()

    def request_stop(self) -> None:
        """Stop after the detector's current acquisition call returns."""
        self._stop_requested.set()

    @Slot()
    def run(self) -> None:
        count = 0
        try:
            while not self._stop_requested.is_set():
                cycle_started = time.perf_counter()
                spectrum = self._acquire_one()
                elapsed = time.perf_counter() - cycle_started
                count += 1
                self.spectrum_ready.emit(spectrum, count, elapsed)

                remaining = self._minimum_cycle_seconds - elapsed
                if remaining > 0 and self._stop_requested.wait(remaining):
                    break
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")
        finally:
            self.finished.emit()


__all__ = ["LiveSpectrumWorker"]
