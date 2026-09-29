"""Small Qt UI helpers used across the widget."""

from collections.abc import Callable, Iterable, Sequence

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class CollapsibleGroupBox(QGroupBox):
    """A checkable group box that collapses one dedicated body widget.

    Child widgets may have their own conditional visibility (for example,
    z-scan or tracking options).  Hiding only the body preserves those states
    when the section is collapsed and expanded again.
    """

    def __init__(self, title: str, *, expanded: bool = True):
        super().__init__(title)
        self.setCheckable(True)

        self._body = QWidget(self)
        self._shell_layout = QVBoxLayout()
        # Napari's QGroupBox stylesheet paints the title inside the top
        # margin.  Reserve enough space that the opaque body never overlaps
        # the title baseline when a section is expanded.
        self._shell_layout.setContentsMargins(6, 18, 6, 6)
        self._shell_layout.setSpacing(0)
        super().setLayout(self._shell_layout)
        self._shell_layout.addWidget(self._body)

        self.toggled.connect(self._body.setVisible)
        self.setChecked(expanded)
        # Defer the initial hide until callers have populated the body.
        self._initial_visibility_timer = QTimer(self)
        self._initial_visibility_timer.setSingleShot(True)
        self._initial_visibility_timer.timeout.connect(self._sync_body_visibility)
        self._initial_visibility_timer.start(0)

    def _sync_body_visibility(self) -> None:
        self._body.setVisible(self.isChecked())

    @property
    def content_widget(self) -> QWidget:
        """Return the widget controlled by the collapsible header."""
        return self._body

    def setLayout(self, layout) -> None:  # noqa: N802 - Qt API compatibility
        """Install ``layout`` on the collapsible body, not the group box."""
        if layout is self._shell_layout:
            return
        current = self._body.layout()
        if current is layout:
            return
        if current is not None:
            raise RuntimeError("A collapsible section already has a layout")
        # Most callers construct a bare layout, whose platform-default margins
        # are too wide once nested inside a group box and a napari dock.
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(5)
        self._body.setLayout(layout)


def make_collapsible(title: str, expanded: bool = True) -> QGroupBox:
    """Create a section whose body collapses without changing child state."""
    return CollapsibleGroupBox(title, expanded=expanded)


def make_panel_header(
    help_widget: QWidget,
    show_plots: Callable[[], None],
) -> tuple[QWidget, QPushButton]:
    """Build the compact, fixed header shared by the main dock panels."""
    header = QWidget()
    layout = QHBoxLayout(header)
    layout.setContentsMargins(6, 2, 6, 2)
    layout.setSpacing(6)
    layout.addStretch(1)
    layout.addWidget(help_widget)

    plots_button = QPushButton("Plots")
    plots_button.setFlat(True)
    plots_button.setToolTip("Show plots and analysis windows")
    plots_button.clicked.connect(show_plots)
    layout.addWidget(plots_button)
    return header, plots_button


def make_workflow_tabs(
    groups: Sequence[tuple[str, Iterable[QWidget]]],
    *,
    parent: QWidget | None = None,
    expanding_tabs: Iterable[str] = (),
) -> QTabWidget:
    """Make scrollable form tabs or full-height, independently scrolling panes."""
    tabs = QTabWidget(parent)
    tabs.setDocumentMode(True)
    expanding_tabs = set(expanding_tabs)

    for title, sections in groups:
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(6)
        for section in sections:
            if section is not None:
                content_layout.addWidget(section, 1 if title in expanding_tabs else 0)
        if title in expanding_tabs:
            tabs.addTab(content, title)
            continue
        content_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        tabs.addTab(scroll, title)

    return tabs


def _walk_layouts(layout):
    """Yield a layout and all layouts nested beneath it."""
    yield layout
    for index in range(layout.count()):
        item = layout.itemAt(index)
        child_layout = item.layout()
        if child_layout is not None:
            yield from _walk_layouts(child_layout)
            continue
        widget = item.widget()
        if widget is not None and widget.layout() is not None:
            yield from _walk_layouts(widget.layout())


def align_form_rows(
    roots: Iterable[QWidget],
    *,
    maximum_label_width: int = 135,
) -> None:
    """Align repeated ``label + field`` rows in narrow napari docks."""
    rows: list[tuple[QHBoxLayout, QLabel, QWidget]] = []
    preferred_width = 0
    shrinkable_fields = (QAbstractSpinBox, QComboBox, QLineEdit)

    for root in roots:
        # Standalone instructional labels otherwise advertise their full text
        # as the section's minimum width.  Word wrapping lets narrow docks
        # determine the width instead.
        for label in root.findChildren(QLabel):
            label.setMinimumWidth(0)
            if len(label.text()) > 28:
                label.setWordWrap(True)
        for checkbox in root.findChildren(QCheckBox):
            checkbox.setMinimumWidth(0)
            checkbox_policy = checkbox.sizePolicy()
            checkbox_policy.setHorizontalPolicy(QSizePolicy.Ignored)
            checkbox.setSizePolicy(checkbox_policy)

        root_layout = root.layout()
        if root_layout is None:
            continue
        for layout in _walk_layouts(root_layout):
            if not isinstance(layout, QHBoxLayout) or layout.count() == 0:
                continue
            layout.setSpacing(6)
            for index in range(layout.count()):
                widget = layout.itemAt(index).widget()
                if not isinstance(widget, shrinkable_fields):
                    continue
                widget.setMinimumWidth(0)
                field_policy = widget.sizePolicy()
                field_policy.setHorizontalPolicy(QSizePolicy.Ignored)
                widget.setSizePolicy(field_policy)

            if layout.count() < 2:
                continue
            label = layout.itemAt(0).widget()
            field = layout.itemAt(1).widget()
            if not isinstance(label, QLabel) or field is None:
                first = layout.itemAt(0).widget()
                if isinstance(first, shrinkable_fields):
                    layout.setStretch(0, 1)
                continue
            # Rows such as demo X/Y contain two label/field pairs and should
            # use their natural compact label widths rather than one form-wide
            # label column.
            label_count = sum(
                isinstance(layout.itemAt(index).widget(), QLabel)
                for index in range(layout.count())
            )
            if label_count > 1:
                continue
            rows.append((layout, label, field))
            preferred_width = max(preferred_width, label.sizeHint().width())

    label_width = min(preferred_width, maximum_label_width)
    for layout, label, field in rows:
        label.setMinimumWidth(label_width)
        label.setMaximumWidth(label_width)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        label.setWordWrap(True)
        field_policy = field.sizePolicy()
        field_policy.setHorizontalPolicy(QSizePolicy.Ignored)
        field.setSizePolicy(field_policy)
        layout.setStretch(0, 0)
        layout.setStretch(1, 1)
