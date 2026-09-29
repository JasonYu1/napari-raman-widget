"""One serialized, cooperatively stoppable acquisition job per Raman widget."""

from __future__ import annotations

import threading
import time

from qtpy.QtCore import QEvent, QObject, QThread, QTimer, Qt, Signal, Slot
from qtpy.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QTabWidget, QVBoxLayout, QWidget

from .acquisition.control import AcquisitionCancelled, check_cancelled
from .log_window import LogWindow, _StdoutRedirector


# A running QThread must outlive a closing/deleted panel. These references are
# released only after the worker has actually returned, never on a stop request.
_RUNNING_WORKERS = set()


class AcquisitionWorker(QThread):
    progress = Signal(int, int, str)

    def __init__(self, operation, log):
        super().__init__()
        self.operation = operation
        self.log = log
        self.stop_event = threading.Event()
        self.result = None
        self.outcome = None
        self.error = None

    def request_stop(self):
        self.stop_event.set()

    def run(self):
        try:
            with _StdoutRedirector(self.log):
                check_cancelled(self.stop_event.is_set)
                self.result = self.operation(self.progress.emit, self.stop_event.is_set)
            self.outcome = "stopped" if self.result.get("stopped", False) else "complete"
        except AcquisitionCancelled as error:
            self.outcome = "stopped"
            self.error = str(error)
        except Exception as error:
            self.outcome = "failed"
            self.error = f"{type(error).__name__}: {error}"


class AcquisitionJobs(QObject):
    """GUI-thread coordinator; operations receive snapshots, never Qt fields."""

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.worker = None
        self.log = None
        self._restore_controls = []
        self._close_target = None
        self._close_requested = False
        self.panel = QWidget(owner)
        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(6, 4, 6, 4)
        self.label = QLabel("No acquisition running")
        self.label.setTextFormat(Qt.PlainText)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        row = QHBoxLayout()
        self.bar = QProgressBar()
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        row.addWidget(self.bar, 1)
        self.elapsed = QLabel("0:00")
        row.addWidget(self.elapsed)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setToolTip("Request a stop after the current acquisition batch")
        self.stop_button.clicked.connect(self.request_stop)
        row.addWidget(self.stop_button)
        self.log_button = QPushButton("Log")
        self.log_button.clicked.connect(self.show_log)
        row.addWidget(self.log_button)
        layout.addLayout(row)
        self.panel.hide()
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self._update_elapsed)

    @property
    def is_running(self):
        # Keep actions locked until the GUI completion callback has run too.
        return self.worker is not None

    def available(self):
        owner = self.owner
        if self.is_running:
            owner.status.setText("Status: wait for the current acquisition or request Stop")
            return False
        if getattr(owner, "_live_raman_worker", None) is not None:
            owner.status.setText("Status: stop live spectra before starting another acquisition")
            return False
        core = getattr(owner, "core", None)
        mda = getattr(core, "mda", None)
        running = getattr(mda, "is_running", None)
        mda_running = bool(running() if callable(running) else running)
        mda_thread = getattr(owner, "_raman_mda_thread", None)
        if mda_thread is not None:
            mda_running = mda_running or mda_thread.is_alive()
        if mda_running:
            owner.status.setText("Status: stop the running MDA before starting another acquisition")
            return False
        if getattr(owner, "_hardware_shutdown_started", False):
            owner.status.setText("Status: hardware is shutting down")
            return False
        return True

    def start(self, title, operation, on_success):
        if not self.available():
            return False
        self.title = title
        self._on_success = on_success
        self.log = LogWindow(f"{title} log", show_progress=True)
        self.log.stop_requested.connect(self.request_stop)
        self.log.start_progress(f"Preparing {title.lower()}", cancellable=True)
        self.owner._show_plot(self.log)
        self.label.setText(f"Preparing {title.lower()}")
        self.owner.status.setText(f"Status: {title.lower()} running")
        self.bar.setRange(0, 0)
        self.stop_button.setEnabled(True)
        self.panel.show()
        self._started = time.monotonic()
        self._update_elapsed()
        self.timer.start()
        self._restore_controls = []
        tabs = self.owner.workflow_tabs
        if isinstance(tabs, QTabWidget):
            controls = [tabs.widget(i) for i in range(tabs.count())
                        if tabs.tabText(i) != "Assistant"]
        else:
            controls = [tabs]
        controls.append(getattr(self.owner, "main_window", None))
        for control in controls:
            if isinstance(control, QWidget) and control is not self.owner.window():
                self._restore_controls.append((control, control.isEnabled()))
                control.setEnabled(False)
        self._close_target = self.owner.window()
        self._close_target.installEventFilter(self)
        self._close_requested = False
        worker = AcquisitionWorker(operation, self.log)
        self.worker = worker
        _RUNNING_WORKERS.add(worker)
        worker.progress.connect(self._progress)
        worker.finished.connect(self._finished)
        self.owner.destroyed.connect(worker.request_stop)
        # Also release workers when a dock is deleted without a Close event.
        # This callback touches no GUI objects and runs even after our QObject
        # coordinator has been destroyed.
        worker.finished.connect(lambda: _RUNNING_WORKERS.discard(worker), Qt.DirectConnection)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        return True

    @Slot(int, int, str)
    def _progress(self, completed, total, stage):
        if not self.is_running:
            return
        self.log.update_progress(completed, total, stage)
        total = max(1, int(total))
        self.bar.setRange(0, total)
        self.bar.setValue(max(0, min(total, int(completed))))
        self.bar.setFormat("%v / %m")
        if not self.worker.stop_event.is_set():
            self.label.setText(stage)

    def _update_elapsed(self):
        seconds = max(0, int(time.monotonic() - self._started))
        minutes, seconds = divmod(seconds, 60)
        self.elapsed.setText(f"{minutes}:{seconds:02d}")

    def show_log(self):
        if self.log is not None:
            self.owner._show_plot(self.log)

    @Slot()
    def request_stop(self):
        if self.worker is None:
            return
        self.worker.request_stop()
        self.stop_button.setEnabled(False)
        self.label.setText("Stopping after the current batch; restoring hardware…")
        self.owner.status.setText("Status: stop requested — waiting for the current batch")
        # The log's button disables itself and emits at most once.
        self.log.request_stop()

    @Slot()
    def _finished(self):
        worker = self.worker
        if worker is None:
            return
        self.timer.stop()
        self._update_elapsed()
        self.stop_button.setEnabled(False)
        try:
            if worker.outcome == "failed":
                raise RuntimeError(worker.error)
            if worker.result is not None:
                self._on_success(worker.result)
            if worker.outcome == "stopped":
                message = f"{self.title} stopped"
                self.log.cancel_progress(message)
                self.label.setText(message)
                if self.bar.maximum() <= self.bar.minimum():
                    self.bar.setRange(0, 1)
                    self.bar.setValue(0)
                self.bar.setFormat("Stopped")
                if worker.result is None:
                    self.owner.status.setText(f"Status: {message}; no dataset saved")
            else:
                message = f"{self.title} complete"
                self.log.finish_progress(message)
                self.label.setText(message)
                self.bar.setRange(0, max(1, self.bar.maximum()))
                self.bar.setValue(self.bar.maximum())
                self.bar.setFormat("%v / %m")
        except Exception as error:
            message = f"{self.title} failed: {error}"
            self.log.append(f"\n{message}\n")
            self.log.fail_progress(message)
            self.label.setText(message)
            self.owner.status.setText(f"Status: {message}")
            self.bar.setRange(0, max(1, self.bar.maximum()))
            if self.bar.value() >= self.bar.maximum():
                self.bar.setMaximum(self.bar.maximum() + 1)
            self.bar.setFormat("Failed")
        finally:
            for control, enabled in self._restore_controls:
                try:
                    control.setEnabled(enabled)
                except RuntimeError:
                    pass
            self._restore_controls.clear()
            target = self._close_target
            if target is not None:
                try:
                    target.removeEventFilter(self)
                except RuntimeError:
                    target = None
            self._close_target = None
            self.worker = None
            _RUNNING_WORKERS.discard(worker)
            if self._close_requested and target is not None:
                QTimer.singleShot(0, target.close)

    def eventFilter(self, watched, event):
        if (event.type() == QEvent.Close
                and watched is getattr(self, "_close_target", None)
                and getattr(self, "worker", None) is not None):
            event.ignore()
            self._close_requested = True
            self.request_stop()
            return True
        return False

    def stop_and_wait(self, milliseconds=5000):
        """Shutdown must not unload devices under a still-running worker."""
        if self.worker is None:
            return True
        self.worker.request_stop()
        return self.worker.wait(milliseconds)
