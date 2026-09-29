"""Headless coverage of worker/coordinator lifecycle without hardware widgets."""

import os
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QCoreApplication, QEvent, QTimer, Slot
from qtpy.QtWidgets import QApplication, QLabel, QTabWidget, QVBoxLayout, QWidget

from napari_raman_widget.acquisition.control import check_cancelled
from napari_raman_widget.acquisition_jobs import AcquisitionJobs, _RUNNING_WORKERS


class _Owner(QWidget):
    def __init__(self):
        super().__init__()
        self.workflow_tabs = QWidget(self)
        self.main_window = QWidget(self)
        self.status = QLabel("Status: idle", self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.workflow_tabs)
        layout.addWidget(self.main_window)
        layout.addWidget(self.status)
        self.core = SimpleNamespace(mda=SimpleNamespace(is_running=Mock(return_value=False)))
        self._live_raman_worker = None
        self._hardware_shutdown_started = False
        self.shown_plots = []

    def _show_plot(self, panel):
        self.shown_plots.append(panel)
        panel.show()


class _RecordingJobs(AcquisitionJobs):
    def __init__(self, owner):
        super().__init__(owner)
        self.progress_threads = []

    @Slot(int, int, str)
    def _progress(self, completed, total, stage):
        self.progress_threads.append(threading.get_ident())
        super()._progress(completed, total, stage)


class AcquisitionJobsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.owner = _Owner()
        self.jobs = _RecordingJobs(self.owner)
        self.owner.layout().addWidget(self.jobs.panel)
        self.owner.show()
        self.app.processEvents()
        self.releases = []

    def tearDown(self):
        for release in self.releases:
            release.set()
        if self.jobs.is_running:
            self.jobs.request_stop()
            worker = self.jobs.worker
            self.assertTrue(worker.wait(3000), "Test worker failed to exit")
            self.wait_until(lambda: not self.jobs.is_running)
        for panel in set(self.owner.shown_plots):
            panel.close()
            panel.deleteLater()
        self.owner.close()
        self.owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def wait_until(self, condition, timeout=3):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            threading.Event().wait(0.001)
        self.app.processEvents()
        self.assertTrue(condition(), "Timed out waiting for Qt/worker state")

    def held_operation(self, *, progress=True, result=None):
        entered = threading.Event()
        release = threading.Event()
        self.releases.append(release)
        operation_threads = []

        def operation(report, cancelled):
            operation_threads.append(threading.get_ident())
            if progress:
                report(1, 4, "Background axial scan")
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("Test did not release worker")
            check_cancelled(cancelled)
            return {"saved": True} if result is None else result

        return operation, entered, release, operation_threads

    def test_worker_operation_is_off_thread_but_callbacks_are_on_gui_thread(self):
        gui_thread = threading.get_ident()
        operation, entered, release, operation_threads = self.held_operation()
        success_threads = []
        successes = []

        def on_success(result):
            success_threads.append(threading.get_ident())
            successes.append(result)

        self.assertTrue(self.jobs.start("Reference collection", operation, on_success))
        worker = self.jobs.worker
        self.assertIn(worker, _RUNNING_WORKERS)
        self.assertTrue(entered.wait(timeout=3))
        self.wait_until(lambda: self.jobs.bar.value() == 1)
        self.assertNotEqual(operation_threads, [gui_thread])
        self.assertEqual(len(operation_threads), 1)
        self.assertEqual(self.jobs.progress_threads, [gui_thread])
        self.assertEqual(self.jobs.log.progress_label.text(), "Background axial scan")

        release.set()
        self.wait_until(lambda: not self.jobs.is_running)

        self.assertEqual(success_threads, [gui_thread])
        self.assertEqual(successes, [{"saved": True}])
        self.assertNotIn(worker, _RUNNING_WORKERS)
        self.assertEqual(self.jobs.label.text(), "Reference collection complete")
        self.assertEqual(self.jobs.bar.value(), self.jobs.bar.maximum())
        self.assertFalse(self.jobs.timer.isActive())

    def test_gui_timer_keeps_running_while_acquisition_is_waiting(self):
        operation, entered, release, _ = self.held_operation()
        timer_fired = []
        self.jobs.start("Grid scan", operation, lambda result: None)
        self.assertTrue(entered.wait(timeout=3))
        QTimer.singleShot(0, lambda: timer_fired.append(True))

        self.wait_until(lambda: bool(timer_fired))

        self.assertFalse(release.is_set())
        self.assertTrue(self.jobs.worker.isRunning())
        self.assertTrue(self.jobs.is_running)
        release.set()
        self.wait_until(lambda: not self.jobs.is_running)

    def test_only_one_acquisition_can_start_and_controls_are_restored(self):
        operation, entered, release, _ = self.held_operation()
        self.owner.main_window.setEnabled(False)
        second_operation = Mock(return_value={})
        self.assertTrue(self.jobs.available())
        self.assertTrue(self.jobs.start("Grid scan", operation, lambda result: None))
        self.assertTrue(entered.wait(timeout=3))
        first_worker = self.jobs.worker
        first_log = self.jobs.log

        self.assertFalse(self.owner.workflow_tabs.isEnabled())
        self.assertFalse(self.owner.main_window.isEnabled())
        self.assertFalse(self.jobs.available())
        self.assertFalse(self.jobs.start("Second scan", second_operation, lambda result: None))
        self.assertIs(self.jobs.worker, first_worker)
        self.assertIs(self.jobs.log, first_log)
        self.assertIn("wait for the current acquisition", self.owner.status.text())
        second_operation.assert_not_called()

        release.set()
        self.wait_until(lambda: not self.jobs.is_running)

        self.assertTrue(self.owner.workflow_tabs.isEnabled())
        self.assertFalse(self.owner.main_window.isEnabled())
        self.assertTrue(self.jobs.available())
        self.assertTrue(self.jobs.start("Second scan", second_operation, lambda result: None))
        self.wait_until(lambda: not self.jobs.is_running)
        second_operation.assert_called_once()

    def test_availability_rejects_live_spectra_mda_and_hardware_shutdown(self):
        self.owner._live_raman_worker = object()
        self.assertFalse(self.jobs.available())
        self.assertIn("stop live spectra", self.owner.status.text())
        self.owner._live_raman_worker = None
        self.owner.core.mda.is_running.return_value = True
        self.assertFalse(self.jobs.available())
        self.assertIn("stop the running MDA", self.owner.status.text())
        self.owner.core.mda.is_running.return_value = False
        self.owner._hardware_shutdown_started = True
        self.assertFalse(self.jobs.available())
        self.assertIn("shutting down", self.owner.status.text())
        self.owner._hardware_shutdown_started = False
        self.assertTrue(self.jobs.available())

    def test_assistant_tab_stays_available_while_hardware_tabs_are_locked(self):
        tabs = QTabWidget(self.owner)
        calibration = QWidget()
        assistant = QWidget()
        advanced = QWidget()
        tabs.addTab(calibration, "Calibration")
        tabs.addTab(assistant, "Assistant")
        tabs.addTab(advanced, "Advanced")
        advanced.setEnabled(False)
        self.owner.workflow_tabs = tabs
        self.owner.layout().addWidget(tabs)
        operation, entered, release, _ = self.held_operation()
        self.jobs.start("Calibration", operation, lambda result: None)
        self.assertTrue(entered.wait(timeout=3))

        self.assertTrue(tabs.isEnabled())
        self.assertTrue(assistant.isEnabled())
        self.assertFalse(calibration.isEnabled())
        self.assertFalse(advanced.isEnabled())
        tabs.setCurrentWidget(assistant)
        self.assertIs(tabs.currentWidget(), assistant)

        release.set()
        self.wait_until(lambda: not self.jobs.is_running)

        self.assertTrue(calibration.isEnabled())
        self.assertTrue(assistant.isEnabled())
        self.assertFalse(advanced.isEnabled())

    def test_stop_request_keeps_controls_locked_until_worker_acknowledges(self):
        operation, entered, release, _ = self.held_operation()
        succeeded = Mock()
        self.assertFalse(self.jobs.is_running)
        self.jobs.start("Calibration", operation, succeeded)
        self.assertTrue(entered.wait(timeout=3))
        self.wait_until(lambda: self.jobs.bar.value() == 1)

        self.jobs.log.stop_button.click()

        self.assertTrue(self.jobs.worker.stop_event.is_set())
        self.assertTrue(self.jobs.is_running)
        self.assertFalse(self.jobs.stop_button.isEnabled())
        self.assertFalse(self.jobs.log.stop_button.isEnabled())
        self.assertIn("Stop requested", self.jobs.log.progress_label.text())
        self.assertFalse(self.owner.workflow_tabs.isEnabled())
        self.assertTrue(self.jobs.timer.isActive())

        release.set()
        self.wait_until(lambda: not self.jobs.is_running)

        succeeded.assert_not_called()
        self.assertEqual(self.jobs.label.text(), "Calibration stopped")
        self.assertEqual(self.jobs.log.progress_label.text(), "Calibration stopped")
        self.assertEqual(self.jobs.bar.format(), "Stopped")
        self.assertLess(self.jobs.bar.value(), self.jobs.bar.maximum())
        self.assertFalse(self.jobs.timer.isActive())
        self.assertTrue(self.owner.workflow_tabs.isEnabled())
        self.assertIn("no dataset saved", self.owner.status.text())

    def test_stop_before_first_progress_leaves_a_determinate_stopped_bar(self):
        operation, entered, release, _ = self.held_operation(progress=False)
        self.jobs.start("Reference collection", operation, lambda result: None)
        self.assertTrue(entered.wait(timeout=3))
        self.assertEqual(self.jobs.bar.minimum(), self.jobs.bar.maximum())
        self.jobs.request_stop()
        release.set()
        self.wait_until(lambda: not self.jobs.is_running)

        self.assertGreater(self.jobs.bar.maximum(), self.jobs.bar.minimum())
        self.assertGreaterEqual(self.jobs.bar.value(), self.jobs.bar.minimum())
        self.assertLess(self.jobs.bar.value(), self.jobs.bar.maximum())
        self.assertEqual(self.jobs.bar.format(), "Stopped")

    def test_worker_failure_is_visible_without_a_full_or_busy_progress_bar(self):
        succeeded = Mock()

        def operation(report, cancelled):
            report(4, 4, "Saving")
            raise ValueError("Camera disconnected")

        self.jobs.start("Grid scan", operation, succeeded)
        self.wait_until(lambda: not self.jobs.is_running)

        succeeded.assert_not_called()
        self.assertIn("ValueError: Camera disconnected", self.jobs.label.text())
        self.assertIn("Camera disconnected", self.jobs.log.text.toPlainText())
        self.assertEqual(self.jobs.bar.format(), "Failed")
        self.assertGreater(self.jobs.bar.maximum(), self.jobs.bar.minimum())
        self.assertLess(self.jobs.bar.value(), self.jobs.bar.maximum())
        self.assertEqual(self.jobs.log.progress_bar.format(), "Failed")
        self.assertTrue(self.owner.workflow_tabs.isEnabled())
        self.assertFalse(self.jobs.stop_button.isEnabled())

    def test_result_callback_failure_restores_controls_and_reports_error(self):
        def on_success(result):
            raise RuntimeError("Cannot display result")

        self.jobs.start("Grid scan", lambda report, cancelled: {}, on_success)
        self.wait_until(lambda: not self.jobs.is_running)

        self.assertIn("Cannot display result", self.jobs.label.text())
        self.assertEqual(self.jobs.bar.format(), "Failed")
        self.assertTrue(self.owner.workflow_tabs.isEnabled())
        self.assertFalse(self.jobs.timer.isActive())
        self.assertTrue(self.jobs.available())

    def test_failure_before_first_progress_still_shows_determinate_failure(self):
        def operation(report, cancelled):
            raise RuntimeError("Device unavailable")

        self.jobs.start("Grid scan", operation, lambda result: None)
        self.wait_until(lambda: not self.jobs.is_running)

        self.assertGreater(self.jobs.bar.maximum(), self.jobs.bar.minimum())
        self.assertGreaterEqual(self.jobs.bar.value(), self.jobs.bar.minimum())
        self.assertLess(self.jobs.bar.value(), self.jobs.bar.maximum())
        self.assertEqual(self.jobs.bar.format(), "Failed")
        self.assertIn("Device unavailable", self.jobs.label.text())

    def test_completed_log_can_be_closed_and_reopened_without_restarting(self):
        operation = Mock(return_value={})
        self.jobs.start("Grid scan", operation, lambda result: None)
        self.wait_until(lambda: not self.jobs.is_running)
        log = self.jobs.log
        log.close()
        self.assertFalse(log.isVisible())

        self.jobs.log_button.click()

        self.assertIs(self.owner.shown_plots[-1], log)
        self.assertTrue(log.isVisible())
        self.assertEqual(log.progress_label.text(), "Grid scan complete")
        operation.assert_called_once()
        self.assertFalse(self.jobs.is_running)

    def test_window_close_requests_stop_and_waits_for_worker_before_closing(self):
        operation, entered, release, _ = self.held_operation()
        self.jobs.start("Grid scan", operation, lambda result: None)
        self.assertTrue(entered.wait(timeout=3))
        worker = self.jobs.worker

        self.owner.close()
        self.app.processEvents()

        self.assertTrue(self.owner.isVisible())
        self.assertTrue(self.jobs.is_running)
        self.assertTrue(worker.stop_event.is_set())
        self.assertIn(worker, _RUNNING_WORKERS)
        self.assertFalse(self.owner.workflow_tabs.isEnabled())

        release.set()
        self.wait_until(lambda: not self.jobs.is_running)
        self.wait_until(lambda: not self.owner.isVisible())

        self.assertNotIn(worker, _RUNNING_WORKERS)
        self.assertEqual(self.jobs.bar.format(), "Stopped")

    def test_deleted_owner_requests_stop_and_worker_outlives_qt_coordinator(self):
        owner = _Owner()
        jobs = _RecordingJobs(owner)
        operation, entered, release, _ = self.held_operation()
        completed = Mock()
        jobs.start("Grid scan", operation, completed)
        self.assertTrue(entered.wait(timeout=3))
        worker = jobs.worker
        log = jobs.log
        uncaught = []
        try:
            with patch.object(sys, "excepthook", side_effect=lambda *args: uncaught.append(args)):
                owner.deleteLater()
                QCoreApplication.sendPostedEvents(owner, QEvent.DeferredDelete)

                self.assertTrue(worker.stop_event.is_set())
                self.assertIn(worker, _RUNNING_WORKERS)
                release.set()
                self.assertTrue(worker.wait(3000))
                self.assertEqual(worker.outcome, "stopped")
                self.assertNotIn(worker, _RUNNING_WORKERS)
                self.app.processEvents()

            completed.assert_not_called()
            self.assertEqual(uncaught, [])
        finally:
            release.set()
            log.close()
            log.deleteLater()


if __name__ == "__main__":
    unittest.main()
