"""A reusable, dockable workspace for Raman results and acquisition logs."""

from __future__ import annotations

import re
from collections import Counter
from weakref import ref

from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedLayout,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .log_window import LogWindow


_PANEL_NAMES = {
    "CalibrationPlotWindow": "Calibration",
    "DetectorImageWindow": "Detector",
    "SpectrumWindow": "Spectrum",
    "ReferenceSpectraWindow": "Reference",
    "GridScanPlotWindow": "Grid scan",
    "DatasetViewerWindow": "Dataset",
}

_PANEL_ID_PREFIXES = {
    "CalibrationPlotWindow": "calibration",
    "DetectorImageWindow": "detector",
    "SpectrumWindow": "spectrum",
    "ReferenceSpectraWindow": "reference",
    "GridScanPlotWindow": "grid-scan",
    "DatasetViewerWindow": "dataset",
    "LogWindow": "log",
}


def _panel_id_prefix(panel):
    prefix = _PANEL_ID_PREFIXES.get(type(panel).__name__)
    if prefix is not None:
        return prefix
    name = re.sub(r"(?<!^)(?=[A-Z])", "-", type(panel).__name__).lower()
    name = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return name or "panel"


class PlotWorkspace(QWidget):
    """Keep plot content alive while its Napari dock is moved or hidden.

    The workspace starts floating. Closing a result tab releases it.
    Closing the workspace only hides or
    detaches the dock; the owner's Plots button reopens the same results.
    """

    def __init__(self, owner):
        super().__init__(owner)
        self._owner = ref(owner)
        owner.destroyed.connect(self.deleteLater)
        self._dock = None
        self._floating = True
        self._counts = Counter()
        self._panel_id_counts = Counter()
        self._panel_ids = {}
        self._title_callbacks = {}
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        header = QHBoxLayout()
        self.title_label = QLabel("Plots and acquisition logs")
        self.title_label.setTextFormat(Qt.PlainText)
        self.title_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.title_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        header.addWidget(self.title_label, 1)
        self.float_button = QPushButton("Float")
        self.float_button.setToolTip(
            "Float the plot workspace in a separate window. You can also "
            "drag the dock's title bar out of Napari and back into an edge."
        )
        self.float_button.clicked.connect(self._toggle_floating)
        header.addWidget(self.float_button)
        self.hide_button = QPushButton("Hide")
        self.hide_button.setToolTip("Hide the workspace; use Plots to reopen it")
        self.hide_button.clicked.connect(self._hide_workspace)
        header.addWidget(self.hide_button)
        layout.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(True)
        self.tabs.setTabsClosable(True)
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setElideMode(Qt.ElideRight)
        self.tabs.tabCloseRequested.connect(self.close_panel)
        self.tabs.currentChanged.connect(self._update_title)
        self.empty_label = QLabel(
            "Your spectra, scans, calibration views, and logs will appear here.\n"
            "Drag tabs to reorder them. Use the button above to dock "
            "or float this workspace."
        )
        self.empty_label.setWordWrap(True)
        self.empty_label.setAlignment(Qt.AlignCenter)
        self._pages = QStackedLayout()
        self._pages.addWidget(self.empty_label)
        self._pages.addWidget(self.tabs)
        layout.addLayout(self._pages, 1)

    def sizeHint(self):
        return QSize(850, 380)

    def minimumSizeHint(self):
        return QSize(360, 220)

    def show_in_viewer(self):
        """Show or recreate our dock using only Napari's public window API."""
        owner = self._owner()
        if owner is None:
            return
        dock = self._dock
        # Napari's close-panel action detaches content before deleting the
        # wrapper. Reuse this workspace even if Plots is clicked immediately.
        try:
            attached = dock is not None and dock.widget() is self
        except RuntimeError:
            attached = False
        if not attached:
            floating = self._floating
            dock = owner.viewer.window.add_dock_widget(
                self,
                name=(
                    "Raman Demo Plots"
                    if type(owner).__name__ == "DemoWidget"
                    else "Raman Plots"
                ),
                area="bottom",
                allowed_areas=("left", "right", "top", "bottom"),
                add_vertical_stretch=False,
            )
            self._dock = dock
            dock.topLevelChanged.connect(self._floating_changed)
            dock.destroyed.connect(lambda: self._forget_dock(dock))
            dock.setFloating(floating)
            if floating:
                dock.resize(1000, 650)
        self.show()
        dock.show()
        dock.raise_()
        self._floating_changed(dock.isFloating())
        return dock

    def _forget_dock(self, dock):
        if self._dock is dock:
            self._dock = None

    def _toggle_floating(self):
        dock = self.show_in_viewer()
        if dock is not None:
            floating = not dock.isFloating()
            dock.setFloating(floating)
            if floating:
                dock.resize(1000, 650)
            dock.show()
            dock.raise_()

    def _floating_changed(self, floating):
        self._floating = bool(floating)
        self.float_button.setText("Dock back" if floating else "Float")

    def _hide_workspace(self):
        if self._dock is not None:
            try:
                self._dock.hide()
            except RuntimeError:
                self._dock = None

    def add_panel(self, panel):
        """Add a plot once, or focus its existing tab."""
        index = self.tabs.indexOf(panel)
        if index < 0:
            prefix = _panel_id_prefix(panel)
            panel_id = getattr(panel, "_plot_panel_id", None)
            match = (
                re.fullmatch(rf"{re.escape(prefix)}-(\d+)", panel_id)
                if isinstance(panel_id, str)
                else None
            )
            if match is None or panel_id in self._panel_ids.values():
                self._panel_id_counts[prefix] += 1
                panel_id = f"{prefix}-{self._panel_id_counts[prefix]}"
            else:
                self._panel_id_counts[prefix] = max(
                    self._panel_id_counts[prefix], int(match.group(1))
                )
            panel._plot_panel_id = panel_id
            self._panel_ids[panel] = panel_id
            category = _PANEL_NAMES.get(type(panel).__name__, panel.windowTitle())
            category = category or "Plot"
            self._counts[category] += 1
            count = self._counts[category]
            caption = category if count == 1 else f"{category} {count}"
            index = self.tabs.addTab(panel, caption)
            def callback(_title):
                self._update_panel_title(panel)

            self._title_callbacks[panel] = callback
            panel.windowTitleChanged.connect(callback)
            owner = self._owner()
            if owner is not None and panel not in owner._plot_windows:
                owner._plot_windows.append(panel)
        self._pages.setCurrentWidget(self.tabs)
        self.tabs.setCurrentIndex(index)
        panel.show()
        self._update_panel_title(panel)
        self.show_in_viewer()
        return panel

    def panel_id_for(self, panel):
        """Return the stable workspace-local identifier for an open panel."""
        if self.tabs.indexOf(panel) < 0:
            raise ValueError("The requested plot panel is not open.")
        try:
            return self._panel_ids[panel]
        except KeyError as error:
            raise ValueError("The plot panel has no workspace identifier.") from error

    def panel_for_id(self, panel_id):
        """Return an open panel by stable identifier, or ``None``."""
        for panel, candidate in self._panel_ids.items():
            if candidate == panel_id and self.tabs.indexOf(panel) >= 0:
                return panel
        return None

    def _update_panel_title(self, panel):
        index = self.tabs.indexOf(panel)
        if index >= 0:
            self.tabs.setTabToolTip(index, panel.windowTitle())
        if panel is self.tabs.currentWidget():
            self._update_title()

    def _update_title(self, _index=None):
        panel = self.tabs.currentWidget()
        title = (
            panel.windowTitle()
            if panel is not None
            else "Plots and acquisition logs"
        )
        self.title_label.setText(title)
        self.title_label.setToolTip(title)

    def close_panel(self, index):
        """Close one result; request a clean stop if it is the live spectrum."""
        panel = self.tabs.widget(index)
        if panel is None:
            return
        owner = self._owner()
        if owner is not None:
            if panel is getattr(owner, "_live_raman_window", None):
                context = getattr(owner, "_live_raman_context", None)
                if context is not None:
                    context["plot_closed"] = True
                owner._stop_live_raman()
                owner._live_raman_window = None
            if panel in owner._plot_windows:
                owner._plot_windows.remove(panel)
        callback = self._title_callbacks.pop(panel, None)
        if callback is not None:
            panel.windowTitleChanged.disconnect(callback)
        self.tabs.removeTab(index)
        self._panel_ids.pop(panel, None)
        panel.close()
        panel.setParent(None)
        if not isinstance(panel, LogWindow):
            panel.deleteLater()
        # A running operation can still append to its closed log. Its local
        # Python reference owns the unparented log until the operation ends.
        if self.tabs.count() == 0:
            self._pages.setCurrentWidget(self.empty_label)
        self._update_title()


def show_plot_workspace(owner):
    """Return the shared workspace for one hardware or demonstration panel."""
    workspace = getattr(owner, "_plot_workspace", None)
    if workspace is None:
        workspace = PlotWorkspace(owner)
        owner._plot_workspace = workspace
    workspace.show_in_viewer()
    return workspace


def show_plot(owner, panel):
    """Present a result or streaming log in the owner's tabbed workspace."""
    return show_plot_workspace(owner).add_panel(panel)
