"""Dock ownership and live-result lifecycle checks without a microscope."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QCoreApplication, QEvent, Qt
from qtpy.QtWidgets import (
    QApplication, QDockWidget, QMainWindow, QPushButton, QTabBar, QVBoxLayout,
    QWidget,
)

from napari_raman_widget.log_window import LogWindow
from napari_raman_widget.plot_workspace import show_plot, show_plot_workspace


class _Window:
    def __init__(self):
        self.main = QMainWindow()
        self.docks = []

    def add_dock_widget(self, panel, **kwargs):
        dock = QDockWidget(kwargs["name"], self.main)
        dock.setWidget(panel)
        area = {
            "left": Qt.LeftDockWidgetArea,
            "right": Qt.RightDockWidgetArea,
            "top": Qt.TopDockWidgetArea,
            "bottom": Qt.BottomDockWidgetArea,
        }[kwargs["area"]]
        self.main.addDockWidget(area, dock)
        self.docks.append(dock)
        return dock

    def remove_dock_widget(self, dock):
        dock.widget().setParent(None)
        self.main.removeDockWidget(dock)
        self.docks.remove(dock)
        dock.deleteLater()


class PlotWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = _Window()
        self.owner = QWidget()
        self.owner.viewer = SimpleNamespace(window=self.window)
        self.owner._plot_windows = []
        self.owner._live_raman_window = None
        self.owner._live_raman_context = None
        self.owner._stop_live_raman = Mock()
        # Napari may wrap plugin content rather than placing it directly in
        # the dock; exercise discovery through that extra parent as well.
        container = QWidget()
        QVBoxLayout(container).addWidget(self.owner)
        self.owner_dock = QDockWidget("napari-raman", self.window.main)
        self.owner_dock.setWidget(container)
        self.window.main.addDockWidget(Qt.RightDockWidgetArea, self.owner_dock)

    def tearDown(self):
        self.window.main.close()
        self.window.main.deleteLater()
        self.owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def panel(self, title="Spectrum"):
        panel = QWidget()
        panel.setWindowTitle(title)
        return panel

    def test_results_share_dock_and_keep_stable_tabs_during_live_titles(self):
        first = show_plot(self.owner, self.panel())
        second = show_plot(self.owner, self.panel())
        workspace = self.owner._plot_workspace

        self.assertEqual(len(self.window.docks), 1)
        self.assertEqual(workspace.tabs.count(), 2)
        self.assertIs(workspace.tabs.currentWidget(), second)
        caption = workspace.tabs.tabText(1)
        second.setWindowTitle("Spectrum | live #42")
        self.assertEqual(workspace.tabs.tabText(1), caption)
        self.assertEqual(workspace.title_label.text(), "Spectrum | live #42")
        workspace.tabs.tabBar().moveTab(1, 0)
        workspace.close_panel(1)
        self.assertEqual(self.owner._plot_windows, [second])
        self.assertIs(workspace.tabs.widget(0), second)
        self.assertIsNot(first, second)

    def test_floating_and_hidden_workspace_can_be_redocked_and_reopened(self):
        show_plot(self.owner, self.panel())
        workspace = self.owner._plot_workspace
        dock = workspace._dock
        self.assertFalse(dock.isFloating())
        dock.setFloating(True)
        self.assertTrue(dock.isFloating())
        dock.setFloating(False)
        self.assertFalse(dock.isFloating())
        dock.close()
        self.assertTrue(dock.isHidden())
        self.assertIs(show_plot_workspace(self.owner), workspace)
        self.assertFalse(dock.isHidden())
        self.assertFalse(dock.isFloating())
        self.assertEqual(workspace.tabs.count(), 1)

    def test_new_results_do_not_override_user_docking_choice(self):
        show_plot(self.owner, self.panel())
        workspace = self.owner._plot_workspace
        dock = workspace._dock
        self.assertFalse(dock.isFloating())

        dock.setFloating(True)
        show_plot(self.owner, self.panel("Next spectrum"))

        self.assertIs(workspace._dock, dock)
        self.assertTrue(dock.isFloating())
        self.assertEqual(workspace.tabs.count(), 2)

    def test_removed_dock_preserves_user_docking_choice(self):
        show_plot(self.owner, self.panel())
        workspace = self.owner._plot_workspace
        workspace._dock.setFloating(False)
        self.assertFalse(workspace._dock.isFloating())
        self.window.remove_dock_widget(workspace._dock)

        dock = workspace.show_in_viewer()

        self.assertFalse(dock.isFloating())

    def test_removed_napari_dock_reopens_with_its_results(self):
        panel = show_plot(self.owner, self.panel())
        workspace = self.owner._plot_workspace
        old_dock = workspace._dock
        self.window.remove_dock_widget(old_dock)
        new_dock = workspace.show_in_viewer()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

        self.assertIsNot(new_dock, old_dock)
        self.assertIs(workspace._dock, new_dock)
        self.assertFalse(new_dock.isFloating())
        self.assertIn(new_dock, self.window.main.tabifiedDockWidgets(self.owner_dock))
        self.assertEqual(len(self.window.docks), 1)
        self.assertIs(workspace.tabs.widget(0), panel)

    def test_closing_live_tab_requests_stop_and_marks_late_frames_ignored(self):
        panel = show_plot(self.owner, self.panel())
        self.owner._live_raman_window = panel
        self.owner._live_raman_context = {"count": 3}

        self.owner._plot_workspace.close_panel(0)

        self.owner._stop_live_raman.assert_called_once_with()
        self.assertTrue(self.owner._live_raman_context["plot_closed"])
        self.assertIsNone(self.owner._live_raman_window)
        self.assertEqual(self.owner._plot_windows, [])

    def test_hidden_workspace_does_not_stop_live_acquisition(self):
        panel = show_plot(self.owner, self.panel())
        self.owner._live_raman_window = panel
        self.owner._plot_workspace._dock.close()

        self.owner._stop_live_raman.assert_not_called()
        self.assertIs(self.owner._live_raman_window, panel)

    def test_workspace_leaves_window_controls_to_native_dock(self):
        workspace = show_plot_workspace(self.owner)

        self.assertEqual(workspace.findChildren(QPushButton), [])
        features = workspace._dock.features()
        self.assertTrue(features & QDockWidget.DockWidgetClosable)
        self.assertTrue(features & QDockWidget.DockWidgetMovable)
        self.assertTrue(features & QDockWidget.DockWidgetFloatable)

    def test_plots_restores_minimized_floating_workspace(self):
        panel = show_plot(self.owner, self.panel())
        workspace = self.owner._plot_workspace
        workspace._dock.setFloating(True)
        workspace._dock.showMinimized()
        self.assertTrue(workspace._dock.isMinimized())

        show_plot_workspace(self.owner)

        self.assertFalse(workspace._dock.isMinimized())
        self.assertTrue(workspace._dock.isFloating())
        self.assertIs(workspace.tabs.widget(0), panel)

    def test_results_tabify_with_own_controls_not_last_plugin_in_area(self):
        unrelated = QDockWidget("Other plugin", self.window.main)
        unrelated.setWidget(QWidget())
        self.window.main.addDockWidget(Qt.RightDockWidgetArea, unrelated)
        panel = show_plot(self.owner, self.panel())
        workspace = self.owner._plot_workspace
        dock = workspace._dock

        self.assertFalse(dock.isFloating())
        self.assertEqual(self.window.main.dockWidgetArea(dock), Qt.RightDockWidgetArea)
        self.assertIn(dock, self.window.main.tabifiedDockWidgets(self.owner_dock))
        self.assertNotIn(dock, self.window.main.tabifiedDockWidgets(unrelated))
        log = show_plot(self.owner, LogWindow("Acquisition log"))
        self.assertIs(workspace.tabs.currentWidget(), log)
        self.assertIs(workspace.tabs.widget(0), panel)
        self.assertEqual(len(self.window.docks), 1)

    def test_default_uses_actual_controls_area(self):
        self.window.main.addDockWidget(Qt.LeftDockWidgetArea, self.owner_dock)
        workspace = show_plot_workspace(self.owner)
        self.assertEqual(self.window.main.dockWidgetArea(workspace._dock), Qt.LeftDockWidgetArea)
        self.assertIn(workspace._dock, self.window.main.tabifiedDockWidgets(self.owner_dock))

    def test_new_result_raises_native_plot_tab(self):
        self.window.main.resize(1000, 800)
        self.window.main.show()
        workspace = show_plot_workspace(self.owner)
        self.owner_dock.raise_()
        self.app.processEvents()
        show_plot(self.owner, self.panel())
        self.app.processEvents()
        native_bar = next(
            bar for bar in self.window.main.findChildren(QTabBar)
            if "napari-raman" in [bar.tabText(i) for i in range(bar.count())]
        )
        self.assertEqual(native_bar.tabText(native_bar.currentIndex()), workspace._dock.windowTitle())

    def test_moved_separate_dock_stays_put_for_results_hide_and_native_close(self):
        show_plot(self.owner, self.panel())
        workspace = self.owner._plot_workspace
        dock = workspace._dock
        self.window.main.addDockWidget(Qt.BottomDockWidgetArea, dock)
        dock.hide()
        show_plot(self.owner, self.panel("Another result"))
        self.assertIs(workspace._dock, dock)
        self.assertEqual(self.window.main.dockWidgetArea(dock), Qt.BottomDockWidgetArea)
        self.assertEqual(self.window.main.tabifiedDockWidgets(dock), [])

        self.window.remove_dock_widget(dock)
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        new_dock = workspace.show_in_viewer()
        self.assertEqual(self.window.main.dockWidgetArea(new_dock), Qt.BottomDockWidgetArea)
        self.assertEqual(self.window.main.tabifiedDockWidgets(new_dock), [])
        self.assertEqual(workspace.tabs.count(), 2)

    def test_native_close_restores_users_new_tab_group(self):
        show_plot(self.owner, self.panel())
        workspace = self.owner._plot_workspace
        other = QDockWidget("User chosen neighbor", self.window.main)
        other.setWidget(QWidget())
        self.window.main.addDockWidget(Qt.TopDockWidgetArea, other)
        self.window.main.addDockWidget(Qt.TopDockWidgetArea, workspace._dock)
        self.window.main.tabifyDockWidget(other, workspace._dock)
        self.window.remove_dock_widget(workspace._dock)
        new_dock = workspace.show_in_viewer()
        self.assertEqual(self.window.main.dockWidgetArea(new_dock), Qt.TopDockWidgetArea)
        self.assertIn(new_dock, self.window.main.tabifiedDockWidgets(other))
        self.assertNotIn(new_dock, self.window.main.tabifiedDockWidgets(self.owner_dock))

    def test_native_close_ignores_deleted_tab_peer(self):
        show_plot(self.owner, self.panel())
        workspace = self.owner._plot_workspace
        other = QDockWidget("Temporary neighbor", self.window.main)
        other.setWidget(QWidget())
        self.window.main.addDockWidget(Qt.LeftDockWidgetArea, other)
        self.window.main.addDockWidget(Qt.LeftDockWidgetArea, workspace._dock)
        self.window.main.tabifyDockWidget(other, workspace._dock)
        self.window.remove_dock_widget(workspace._dock)
        other.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        new_dock = workspace.show_in_viewer()
        self.assertEqual(self.window.main.dockWidgetArea(new_dock), Qt.LeftDockWidgetArea)
        self.assertEqual(self.window.main.tabifiedDockWidgets(new_dock), [])

    def test_native_close_preserves_floating_choice(self):
        panel = show_plot(self.owner, self.panel())
        workspace = self.owner._plot_workspace
        workspace._dock.setFloating(True)
        self.window.remove_dock_widget(workspace._dock)
        new_dock = workspace.show_in_viewer()
        self.assertTrue(new_dock.isFloating())
        self.assertIs(workspace.tabs.currentWidget(), panel)

    def test_floating_controls_are_not_moved_by_new_results(self):
        self.owner_dock.setFloating(True)
        workspace = show_plot_workspace(self.owner)
        self.assertTrue(self.owner_dock.isFloating())
        self.assertFalse(workspace._dock.isFloating())
        self.assertEqual(self.window.main.dockWidgetArea(workspace._dock), Qt.RightDockWidgetArea)
        self.assertEqual(self.window.main.tabifiedDockWidgets(workspace._dock), [])

    def test_standalone_controls_fall_back_to_right_dock(self):
        self.owner.setParent(None)
        workspace = show_plot_workspace(self.owner)
        self.assertFalse(workspace._dock.isFloating())
        self.assertEqual(self.window.main.dockWidgetArea(workspace._dock), Qt.RightDockWidgetArea)
        self.assertEqual(self.window.main.tabifiedDockWidgets(workspace._dock), [])

    def test_closed_log_can_finish_receiving_output_from_its_producer(self):
        log = show_plot(self.owner, LogWindow("Acquisition log"))
        workspace = self.owner._plot_workspace
        workspace.close_panel(0)
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

        log.append("Acquisition completed\n")
        self.assertIn("Acquisition completed", log.text.toPlainText())
        self.assertEqual(self.owner._plot_windows, [])
        log.deleteLater()


if __name__ == "__main__":
    unittest.main()
