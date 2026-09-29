"""A streaming stdout log window and a context manager that tees into it."""
import re
import sys

from qtpy.QtCore import QEventLoop, Qt
from qtpy.QtWidgets import (
    QApplication,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


_ANSI_RE = re.compile(
    r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b[PX^_].*?\x1b\\'
)


class LogWindow(QWidget):
    """Embeddable panel showing streaming stdout text."""

    def __init__(self, title="Log", *, show_progress=False):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(700, 400)

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
        self.progress_widget.setVisible(show_progress)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet("font-family: monospace;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.progress_widget)
        layout.addWidget(self.text)

    def append(self, s: str):
        # Strip ANSI escape codes (rich/colored terminal output).
        s = _ANSI_RE.sub('', s)
        self.text.moveCursor(self.text.textCursor().End)
        self.text.insertPlainText(s)
        self.text.moveCursor(self.text.textCursor().End)

    def start_progress(self, stage="Starting"):
        """Show an indeterminate progress bar until a total is available."""
        self.progress_widget.show()
        self.progress_label.setText(stage)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("")
        self._refresh_progress_display()

    def update_progress(self, completed, total, stage):
        """Display a completed/total update from a running operation."""
        total = max(1, int(total))
        completed = min(total, max(0, int(completed)))
        self.progress_widget.show()
        self.progress_label.setText(str(stage))
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(completed)
        self.progress_bar.setFormat("%v / %m")
        self._refresh_progress_display()

    def finish_progress(self, stage="Complete"):
        """Mark the operation complete after all work has returned."""
        self.progress_widget.show()
        maximum = self.progress_bar.maximum()
        if maximum <= self.progress_bar.minimum():
            self.progress_bar.setRange(0, 1)
            maximum = 1
        self.progress_bar.setValue(maximum)
        self.progress_bar.setFormat("%v / %m")
        self.progress_label.setText(stage)
        self._refresh_progress_display()

    def fail_progress(self, stage="Failed"):
        """Show failure without leaving a misleading full progress bar."""
        self.progress_widget.show()
        minimum = self.progress_bar.minimum()
        maximum = self.progress_bar.maximum()
        value = self.progress_bar.value()
        if maximum <= minimum:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
        elif value >= maximum:
            self.progress_bar.setMaximum(maximum + 1)
        self.progress_bar.setFormat("Failed")
        self.progress_label.setText(stage)
        self._refresh_progress_display()

    @staticmethod
    def _refresh_progress_display():
        app = QApplication.instance()
        if app is not None:
            app.processEvents(QEventLoop.ExcludeUserInputEvents)


class _StdoutRedirector:
    """Context manager that tees sys.stdout into a LogWindow."""

    def __init__(self, log_window: LogWindow):
        self.log_window = log_window
        self._orig_stdout = None

    def write(self, s):
        original = getattr(self, "_orig_stdout", None)
        if original is not None and original is not self:
            original.write(s)
        try:
            self.log_window.append(s)
            from qtpy.QtWidgets import QApplication
            QApplication.processEvents()
        except Exception:
            pass

    def flush(self):
        original = getattr(self, "_orig_stdout", None)
        if original is not None and original is not self:
            original.flush()

    def __enter__(self):
        self._orig_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *exc):
        original = getattr(self, "_orig_stdout", None)
        if sys.stdout is self and original is not None:
            sys.stdout = original
