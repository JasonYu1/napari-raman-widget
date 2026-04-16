"""A streaming stdout log window and a context manager that tees into it."""
import re
import sys

from qtpy.QtWidgets import QMainWindow, QPlainTextEdit


_ANSI_RE = re.compile(
    r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b[PX^_].*?\x1b\\'
)


class LogWindow(QMainWindow):
    """Pop-up window showing streaming stdout text."""

    def __init__(self, title="Log"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(700, 400)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet("font-family: monospace;")
        self.setCentralWidget(self.text)

    def append(self, s: str):
        # Strip ANSI escape codes (rich/colored terminal output).
        s = _ANSI_RE.sub('', s)
        self.text.moveCursor(self.text.textCursor().End)
        self.text.insertPlainText(s)
        self.text.moveCursor(self.text.textCursor().End)


class _StdoutRedirector:
    """Context manager that tees sys.stdout into a LogWindow."""

    def __init__(self, log_window: LogWindow):
        self.log_window = log_window
        self._orig_stdout = None

    def write(self, s):
        if self._orig_stdout is not None:
            self._orig_stdout.write(s)
        try:
            self.log_window.append(s)
            from qtpy.QtWidgets import QApplication
            QApplication.processEvents()
        except Exception:
            pass

    def flush(self):
        if self._orig_stdout is not None:
            self._orig_stdout.flush()

    def __enter__(self):
        self._orig_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *exc):
        sys.stdout = self._orig_stdout