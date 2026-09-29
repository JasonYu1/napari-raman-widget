"""A streaming stdout log window and a context manager that tees into it."""
import re
import sys
import threading
import time
from collections import deque

from qtpy.QtCore import Qt, QTimer, Signal, Slot
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


_ANSI_RE = re.compile(
    r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b[PX^_].*?\x1b\\'
)
# Andor GetAcquisitionProgress returns (DRV_SUCCESS, accumulation, series).
# Do not hide other return codes, warning text, or arbitrary numeric tuples.
_CAMERA_PROGRESS_RE = re.compile(r"\s*\(20002,\s*\d+,\s*\d+\)\s*")
_MAX_LOG_CHARS = 2_000_000
_STDOUT_REDIRECT_LOCK = threading.RLock()


class LogWindow(QWidget):
    """Embeddable panel showing streaming stdout text."""

    stop_requested = Signal()
    _update_requested = Signal(str, object)

    def __init__(self, title="Log", *, show_progress=False):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(700, 400)
        self._update_requested.connect(self._handle_update)
        self._progress_state = "idle"
        self._cancellable = False
        self._started_at = None
        self._log_entries = deque()
        self._log_chars = 0
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(250)
        self._elapsed_timer.timeout.connect(self._update_elapsed)

        self.progress_widget = QWidget(self)
        progress_layout = QVBoxLayout(self.progress_widget)
        progress_layout.setContentsMargins(8, 8, 8, 4)
        progress_layout.setSpacing(4)
        self.progress_label = QLabel("Waiting to start", self.progress_widget)
        self.progress_label.setTextFormat(Qt.PlainText)
        self.progress_label.setWordWrap(True)
        self.progress_bar = QProgressBar(self.progress_widget)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        status_layout = QHBoxLayout()
        self.elapsed_label = QLabel("Elapsed: 00:00", self.progress_widget)
        self.stop_button = QPushButton("Stop", self.progress_widget)
        self.stop_button.setToolTip(
            "Request a safe stop after the current hardware step finishes."
        )
        self.stop_button.clicked.connect(self.request_stop)
        self.stop_button.hide()
        status_layout.addWidget(self.elapsed_label)
        status_layout.addStretch(1)
        status_layout.addWidget(self.stop_button)
        progress_layout.addLayout(status_layout)
        self.progress_widget.setVisible(show_progress)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet(
            "font-family: Consolas, 'DejaVu Sans Mono', monospace;"
        )
        self.text.setMaximumBlockCount(20_000)
        self.show_camera_diagnostics = QCheckBox("Show camera diagnostics")
        self.show_camera_diagnostics.setToolTip(
            "Show routine successful CCD acquisition status tuples. "
            "Errors and warnings are always shown. Recent log text is retained "
            "up to 2 MB; the original terminal output is unchanged."
        )
        self.show_camera_diagnostics.toggled.connect(self._render_log)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.progress_widget)
        layout.addWidget(self.show_camera_diagnostics)
        layout.addWidget(self.text)

    def append(self, s: str):
        """Append text safely from either the GUI thread or a worker."""
        self._post("append", (str(s), False))

    def append_camera_diagnostic(self, s: str):
        """Retain routine CCD output without cluttering the default view."""
        self._post("append", (str(s), True))

    def _post(self, action, payload=None):
        try:
            self._update_requested.emit(action, payload)
        except RuntimeError:
            # A closing/deleted log must never interrupt hardware cleanup.
            pass

    @Slot(str, object)
    def _handle_update(self, action, payload):
        if action == "append":
            self._append_text(*payload)
        elif action == "start":
            self._start_progress(*payload)
        elif action == "progress":
            self._update_progress(*payload)
        elif action == "terminal":
            self._terminal_progress(*payload)
        elif action == "stop":
            self._request_stop()
        if action != "append":
            # Legacy synchronous workflows still need their status painted.
            # repaint() dispatches paint only, not timers, clicks, or queued
            # commands that could reenter a hardware operation.
            self.progress_widget.repaint()

    def _append_text(self, s, camera_diagnostic):
        s = _ANSI_RE.sub('', s).replace("\r\n", "\n").replace("\r", "\n")
        if not s:
            return
        self._log_entries.append((s, camera_diagnostic))
        self._log_chars += len(s)
        trimmed = False
        while self._log_chars > _MAX_LOG_CHARS:
            oldest, diagnostic = self._log_entries.popleft()
            excess = self._log_chars - _MAX_LOG_CHARS
            self._log_chars -= min(excess, len(oldest))
            if len(oldest) > excess:
                self._log_entries.appendleft((oldest[excess:], diagnostic))
            trimmed = True
        if trimmed:
            self._render_log()
        elif not camera_diagnostic or self.show_camera_diagnostics.isChecked():
            self._insert_text(s)

    def _insert_text(self, s):
        self.text.moveCursor(self.text.textCursor().End)
        self.text.insertPlainText(s)
        self.text.moveCursor(self.text.textCursor().End)
        self.text.viewport().repaint()

    def _render_log(self):
        show_diagnostics = self.show_camera_diagnostics.isChecked()
        self.text.setPlainText("".join(
            text for text, diagnostic in self._log_entries
            if show_diagnostics or not diagnostic
        ))
        self.text.moveCursor(self.text.textCursor().End)
        self.text.viewport().repaint()

    def start_progress(self, stage="Starting", *, cancellable=False):
        """Show an indeterminate progress bar until a total is available."""
        self._post("start", (str(stage), bool(cancellable)))

    def _start_progress(self, stage, cancellable):
        self._progress_state = "running"
        self._cancellable = cancellable
        self._started_at = time.monotonic()
        self._update_elapsed()
        self._elapsed_timer.start()
        self.stop_button.setText("Stop")
        self.stop_button.setEnabled(True)
        self.stop_button.setVisible(cancellable)
        self.progress_widget.show()
        self.progress_label.setText(stage)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("")

    def update_progress(self, completed, total, stage):
        """Display a completed/total update from a running operation."""
        self._post("progress", (completed, total, str(stage)))

    def _update_progress(self, completed, total, stage):
        if self._progress_state in {"complete", "failed", "stopped"}:
            return
        if self._progress_state == "idle":
            self._start_progress(stage, False)
        total = max(1, int(total))
        completed = min(total, max(0, int(completed)))
        self.progress_widget.show()
        if self._progress_state == "stopping":
            stage = f"Stop requested — waiting for current step. {stage}"
        self.progress_label.setText(stage)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(completed)
        self.progress_bar.setFormat("%v / %m")

    def finish_progress(self, stage="Complete"):
        """Mark the operation complete after all work has returned."""
        self._post("terminal", ("complete", str(stage)))

    def fail_progress(self, stage="Failed"):
        """Show failure without leaving a misleading full progress bar."""
        self._post("terminal", ("failed", str(stage)))

    def cancel_progress(self, stage="Stopped"):
        """Mark a safe stop only after the operation has actually returned."""
        self._post("terminal", ("stopped", str(stage)))

    def _terminal_progress(self, state, stage):
        self._progress_state = state
        self._cancellable = False
        self._elapsed_timer.stop()
        self._update_elapsed()
        self.stop_button.hide()
        self.progress_widget.show()
        if state == "complete":
            maximum = self.progress_bar.maximum()
            if maximum <= self.progress_bar.minimum():
                self.progress_bar.setRange(0, 1)
                maximum = 1
            self.progress_bar.setValue(maximum)
            self.progress_bar.setFormat("%v / %m")
        else:
            minimum = self.progress_bar.minimum()
            maximum = self.progress_bar.maximum()
            value = self.progress_bar.value()
            if maximum <= minimum:
                self.progress_bar.setRange(0, 1)
                self.progress_bar.setValue(0)
            elif value >= maximum:
                self.progress_bar.setMaximum(maximum + 1)
            self.progress_bar.setFormat("Failed" if state == "failed" else "Stopped")
        self.progress_label.setText(stage)

    @Slot()
    def request_stop(self):
        """Request cancellation without pretending the hardware stopped yet."""
        self._post("stop")

    def _request_stop(self):
        if not self._cancellable or self._progress_state != "running":
            return
        self._progress_state = "stopping"
        self.stop_button.setEnabled(False)
        self.stop_button.setText("Stop requested")
        self.progress_label.setText(
            "Stop requested — waiting for the current hardware step to finish"
        )
        self.stop_requested.emit()

    def _update_elapsed(self):
        seconds = 0 if self._started_at is None else max(
            0, int(time.monotonic() - self._started_at)
        )
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed = f"{minutes:02d}:{seconds:02d}"
        if hours:
            elapsed = f"{hours:02d}:{elapsed}"
        self.elapsed_label.setText(f"Elapsed: {elapsed}")


class _StdoutRedirector:
    """Context manager that tees sys.stdout into a LogWindow."""

    def __init__(self, log_window: LogWindow):
        self.log_window = log_window
        self._orig_stdout = None
        self._active = False
        self._thread_id = None
        self._pending = ""
        self._after_cr = False
        self._lock = threading.RLock()

    def write(self, s):
        with self._lock:
            original = self._orig_stdout
            if original is not None and original is not self:
                original.write(s)
            # sys.stdout is process-wide. Only capture the initiating task's
            # thread; unrelated GUI activity still reaches the real terminal.
            if not self._active or self._thread_id != threading.get_ident():
                return len(s)
            for char in s:
                if char == "\n" and self._after_cr:
                    self._after_cr = False
                    continue
                self._after_cr = False
                if char in "\r\n":
                    self._emit_line(self._pending + "\n")
                    self._pending = ""
                    self._after_cr = char == "\r"
                else:
                    self._pending += char
            return len(s)

    def _emit_line(self, line):
        cleaned = _ANSI_RE.sub("", line)
        if _CAMERA_PROGRESS_RE.fullmatch(cleaned):
            self.log_window.append_camera_diagnostic(line)
        else:
            self.log_window.append(line)

    def flush(self):
        with self._lock:
            original = self._orig_stdout
            if original is not None and original is not self:
                original.flush()
            # Keep incomplete lines until a delimiter/exit so a fragmented
            # camera status cannot leak into the visible log on flush().

    def __enter__(self):
        with _STDOUT_REDIRECT_LOCK:
            self._orig_stdout = sys.stdout
            self._thread_id = threading.get_ident()
            self._active = True
            sys.stdout = self
        return self

    def __exit__(self, *exc):
        with self._lock:
            try:
                if self._pending:
                    self._emit_line(self._pending)
                    self._pending = ""
            finally:
                with _STDOUT_REDIRECT_LOCK:
                    self._active = False
                    original = self._orig_stdout
                    # Independent worker contexts need not finish in stack
                    # order. Never restore a redirector whose task has ended.
                    while isinstance(original, _StdoutRedirector) and not original._active:
                        original = original._orig_stdout
                    if sys.stdout is self and original is not None:
                        sys.stdout = original
