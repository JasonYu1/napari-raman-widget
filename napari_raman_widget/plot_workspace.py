"""A reusable, dockable workspace for Raman results and acquisition logs."""

from __future__ import annotations

import re
from collections import Counter
from weakref import ref

from qtpy.QtCore import QEvent, QSize, Qt
from qtpy.QtWidgets import (
    QDockWidget,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QStackedLayout,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .log_window import LogWindow


_DOCK_AREA_NAMES = {
    Qt.LeftDockWidgetArea: "left",
    Qt.RightDockWidgetArea: "right",
    Qt.TopDockWidgetArea: "top",
    Qt.BottomDockWidgetArea: "bottom",
}


def _ancestor(widget, widget_type):
    """Find a Qt container without relying on Napari's private attributes."""
    while widget is not None:
        widget = widget.parentWidget()
        if isinstance(widget, widget_type):
            return widget
    return None


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

    The workspace starts tabbed beside its Raman controls. Closing a result
    tab releases it.
    Closing the workspace only hides or
    detaches the dock; the owner's Plots button reopens the same results.
    """

    def __init__(self, owner):
        super().__init__(owner)
        self._owner = ref(owner)
        owner.destroyed.connect(self.deleteLater)
        self._dock = None
        self._floating = False
        self._dock_area = None
        self._tab_peers = None
        self._counts = Counter()
        self._panel_id_counts = Counter()
        self._panel_ids = {}
        self._title_callbacks = {}
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        self.title_label = QLabel("Plots and acquisition logs")
        self.title_label.setTextFormat(Qt.PlainText)
        self.title_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.title_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # Napari owns the dock title bar and its close, hide, and float icons.
        # Keep only the acquisition title here, without duplicate controls.
        layout.addWidget(self.title_label)

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
            "Drag tabs to reorder them. Use the title bar to dock or float "
            "this workspace; use Plots to reopen it after hiding or closing."
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

    def event(self, event):
        # Napari's native close action reparents the content before removing
        # its dock. Capture the user's actual layout while it still exists.
        if event.type() == QEvent.ParentAboutToChange:
            self._remember_dock_layout()
        elif event.type() == QEvent.ParentChange and self.parentWidget() is None:
            # Never-shown widgets skip ParentAboutToChange. Napari has not
            # removed the old (now empty) dock from its main window yet.
            self._remember_dock_layout(allow_detached=True)
        return super().event(event)

    def _remember_dock_layout(self, *, allow_detached=False):
        dock = getattr(self, "_dock", None)
        try:
            if dock is None:
                return
            content = dock.widget()
            if content is not self and not (allow_detached and content is None):
                return
            main = _ancestor(dock, QMainWindow)
            if main is None:
                return
            area = _DOCK_AREA_NAMES.get(main.dockWidgetArea(dock))
            if area is not None:
                self._dock_area = area
            self._floating = dock.isFloating()
            self._tab_peers = [ref(peer) for peer in main.tabifiedDockWidgets(dock)]
        except RuntimeError:
            # The Qt wrapper may already have been deleted during shutdown.
            return

    def _initial_dock_layout(self, owner):
        owner_dock = _ancestor(owner, QDockWidget)
        main = _ancestor(owner_dock, QMainWindow)
        area = (
            _DOCK_AREA_NAMES.get(main.dockWidgetArea(owner_dock))
            if main is not None else None
        )
        peers = (
            [ref(owner_dock)]
            if main is not None and area is not None and not owner_dock.isFloating()
            else []
        )
        # Standalone or floating controls cannot form a native tab group;
        # leave them alone and use their last dock area (or the right side).
        return self._dock_area or area or "right", peers if self._tab_peers is None else self._tab_peers

    @staticmethod
    def _restore_tab_group(dock, peers):
        main = _ancestor(dock, QMainWindow)
        if main is None:
            return
        for peer_ref in peers:
            peer = peer_ref()
            try:
                if (peer is not None and _ancestor(peer, QMainWindow) is main
                        and not peer.isFloating()
                        and main.dockWidgetArea(peer) == main.dockWidgetArea(dock)):
                    main.tabifyDockWidget(peer, dock)
                    break
            except RuntimeError:
                # A saved sibling may have been closed since this workspace.
                continue

    def show_in_viewer(self):
        """Show our native Napari dock without moving an existing workspace."""
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
            area, peers = self._initial_dock_layout(owner)
            dock = owner.viewer.window.add_dock_widget(
                self,
                name=(
                    "Raman Demo Plots"
                    if type(owner).__name__ == "DemoWidget"
                    else "Raman Plots"
                ),
                area=area,
                allowed_areas=("left", "right", "top", "bottom"),
                add_vertical_stretch=False,
            )
            self._dock = dock
            self._dock_area = area
            dock.topLevelChanged.connect(self._floating_changed)
            dock.destroyed.connect(lambda: self._forget_dock(dock))
            dock.setFloating(floating)
            if floating:
                dock.resize(1000, 650)
            else:
                self._restore_tab_group(dock, peers)
        self.show()
        if dock.isMinimized():
            dock.setWindowState(dock.windowState() & ~Qt.WindowMinimized)
        dock.show()
        dock.raise_()
        self._floating_changed(dock.isFloating())
        return dock

    def _forget_dock(self, dock):
        if self._dock is dock:
            self._dock = None

    def _floating_changed(self, floating):
        self._floating = bool(floating)

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
