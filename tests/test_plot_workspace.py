"""Dock ownership and live-result lifecycle checks without a microscope."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QCoreApplication, QEvent, Qt
from qtpy.QtWidgets import QApplication, QDockWidget, QMainWindow, QWidget

from napari_raman_widget.log_window import LogWindow
from napari_raman_widget.plot_workspace import show_plot, show_plot_workspace


class _Window:
    def __init__(self):
        self.main = QMainWindow()
        self.docks = []

    def add_dock_widget(self, panel, **kwargs):
        dock = QDockWidget(kwargs["name"], self.main)
        dock.setWidget(panel)
        self.main.addDockWidget(Qt.BottomDockWidgetArea, dock)
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
        self.assertTrue(dock.isFloating())
        self.assertEqual(workspace.float_button.text(), "Dock back")
        workspace.float_button.click()
        self.assertFalse(dock.isFloating())
        self.assertEqual(workspace.float_button.text(), "Float")
        workspace.hide_button.click()
        self.assertTrue(dock.isHidden())
        self.assertIs(show_plot_workspace(self.owner), workspace)
        self.assertFalse(dock.isHidden())
        self.assertFalse(dock.isFloating())
        self.assertEqual(workspace.tabs.count(), 1)

    def test_new_results_do_not_override_user_docking_choice(self):
        show_plot(self.owner, self.panel())
        workspace = self.owner._plot_workspace
        dock = workspace._dock
        self.assertTrue(dock.isFloating())

        workspace.float_button.click()
        show_plot(self.owner, self.panel("Next spectrum"))

        self.assertIs(workspace._dock, dock)
        self.assertFalse(dock.isFloating())
        self.assertEqual(workspace.tabs.count(), 2)

    def test_removed_dock_preserves_user_docking_choice(self):
        show_plot(self.owner, self.panel())
        workspace = self.owner._plot_workspace
        workspace.float_button.click()
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
        self.assertTrue(new_dock.isFloating())
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
        self.owner._plot_workspace.hide_button.click()

        self.owner._stop_live_raman.assert_not_called()
        self.assertIs(self.owner._live_raman_window, panel)

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
