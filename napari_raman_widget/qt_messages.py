"""Narrow filtering for known harmless Qt platform messages."""

from __future__ import annotations

import sys


_installed = False
_previous_handler = None


def _is_invisible_combo_mouse_grab_warning(message: str) -> bool:
    """Return whether *message* is the noisy Windows QComboBox warning."""
    return (
        message.startswith(
            "QWindowsWindow::setMouseGrabEnabled: "
            "Not setting mouse grab for invisible window"
        )
        and "QComboBoxPrivateContainerClassWindow" in message
    )


def _message_handler(message_type, context, message) -> None:
    if _is_invisible_combo_mouse_grab_warning(message):
        return

    if _previous_handler is not None:
        _previous_handler(message_type, context, message)
        return

    # Installing a Qt handler replaces Qt's default stderr printer. Preserve
    # every message that is not the one specifically filtered above.
    stream = getattr(sys, "__stderr__", None)
    if stream is not None:
        stream.write(f"{message}\n")
        stream.flush()


def install_qt_message_filter() -> None:
    """Install the filter once, retaining any pre-existing Qt handler."""
    global _installed, _previous_handler
    if _installed:
        return
    try:
        # Import lazily so headless test/package environments that intentionally
        # install qtpy without PyQt/PySide can still import this module.
        from qtpy.QtCore import qInstallMessageHandler
    except ImportError:
        return
    _previous_handler = qInstallMessageHandler(_message_handler)
    _installed = True
