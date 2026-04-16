"""Small Qt UI helpers used across the widget."""
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QGroupBox, QWidget


def make_collapsible(title: str, expanded: bool = True) -> QGroupBox:
    """A checkable QGroupBox that hides its contents when unchecked."""
    box = QGroupBox(title)
    box.setCheckable(True)
    box.setChecked(expanded)

    def _toggle(checked):
        for child in box.findChildren(QWidget):
            child.setVisible(checked)
    box.toggled.connect(_toggle)

    if not expanded:
        QTimer.singleShot(0, lambda: _toggle(False))

    return box