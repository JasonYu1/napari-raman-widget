"""Progress-state checks for the embeddable calibration log."""

import os
import io
import sys
import threading
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QCoreApplication, QEvent, QObject, QTimer
from qtpy.QtWidgets import QApplication

from napari_raman_widget.log_window import LogWindow, _StdoutRedirector


class _PaintRecorder(QObject):
    def __init__(self, watched):
        super().__init__(watched)
        self.paint_count = 0
        watched.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Paint:
            self.paint_count += 1
        return False


class LogWindowProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.log = LogWindow("Calibration log", show_progress=True)
        self.log.show()
        self.app.processEvents()

    def tearDown(self):
        self.log.close()

    def test_progress_moves_from_preparing_through_real_counts(self):
        self.log.start_progress("Preparing calibration")
        self.assertEqual(self.log.progress_bar.minimum(), 0)
        self.assertEqual(self.log.progress_bar.maximum(), 0)
        self.assertEqual(
            self.log.progress_label.text(), "Preparing calibration"
        )

        self.log.update_progress(2, 7, "Collecting calibration points")
        self.assertEqual(self.log.progress_bar.maximum(), 7)
        self.assertEqual(self.log.progress_bar.value(), 2)
        self.assertEqual(self.log.progress_bar.format(), "%v / %m")
        self.assertEqual(
            self.log.progress_label.text(),
            "Collecting calibration points",
        )

        self.log.finish_progress("Calibration complete")
        self.assertEqual(self.log.progress_bar.value(), 7)
        self.assertEqual(
            self.log.progress_label.text(), "Calibration complete"
        )

    def test_failure_does_not_leave_a_full_bar(self):
        self.log.update_progress(4, 4, "Calibration dataset saved")

        self.log.fail_progress("Calibration failed")

        self.assertLess(
            self.log.progress_bar.value(), self.log.progress_bar.maximum()
        )
        self.assertEqual(self.log.progress_bar.format(), "Failed")
        self.assertEqual(
            self.log.progress_label.text(), "Calibration failed"
        )

    def test_worker_updates_are_queued_until_gui_processes_them(self):
        def worker():
            self.log.start_progress("Collecting", cancellable=True)
            self.log.update_progress(2, 8, "Axial scan")
            self.log.append("Worker log\n")

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.log.progress_label.text(), "Waiting to start")
        self.assertEqual(self.log.text.toPlainText(), "")

        self.app.processEvents()

        self.assertEqual(self.log.progress_label.text(), "Axial scan")
        self.assertEqual(self.log.progress_bar.value(), 2)
        self.assertEqual(self.log.progress_bar.maximum(), 8)
        self.assertEqual(self.log.text.toPlainText(), "Worker log\n")
        self.assertTrue(self.log.stop_button.isVisible())

    def test_safe_stop_is_requested_once_then_acknowledged(self):
        requests = []
        self.log.stop_requested.connect(lambda: requests.append(True))
        self.log.start_progress("Collecting", cancellable=True)
        self.log.update_progress(3, 10, "Axial scan")

        self.log.stop_button.click()
        self.log.request_stop()
        self.log.update_progress(4, 10, "Finishing current Z plane")

        self.assertEqual(requests, [True])
        self.assertFalse(self.log.stop_button.isEnabled())
        self.assertIn("Stop requested", self.log.progress_label.text())
        self.assertEqual(self.log.progress_bar.value(), 4)
        self.assertTrue(self.log._elapsed_timer.isActive())

        self.log.cancel_progress("Stopped safely")

        self.assertEqual(self.log.progress_label.text(), "Stopped safely")
        self.assertEqual(self.log.progress_bar.format(), "Stopped")
        self.assertFalse(self.log.stop_button.isVisible())
        self.assertFalse(self.log._elapsed_timer.isActive())

    def test_stop_is_hidden_for_non_cancellable_work(self):
        requests = []
        self.log.stop_requested.connect(lambda: requests.append(True))
        self.log.start_progress("Calculating")
        self.log.request_stop()

        self.assertFalse(self.log.stop_button.isVisible())
        self.assertEqual(requests, [])
        self.assertEqual(self.log.progress_label.text(), "Calculating")

    def test_elapsed_freezes_when_finished_and_restarts_on_next_operation(self):
        with patch("napari_raman_widget.log_window.time.monotonic", return_value=100):
            self.log.start_progress("Collecting")
        with patch("napari_raman_widget.log_window.time.monotonic", return_value=167):
            self.log.finish_progress()

        self.assertEqual(self.log.elapsed_label.text(), "Elapsed: 01:07")
        self.assertFalse(self.log._elapsed_timer.isActive())
        self.log.start_progress("Next operation")
        self.assertEqual(self.log.elapsed_label.text(), "Elapsed: 00:00")

    def test_late_worker_progress_cannot_replace_terminal_state(self):
        self.log.start_progress()
        self.log.cancel_progress()
        self.log.update_progress(2, 3, "Late callback")

        self.assertEqual(self.log.progress_label.text(), "Stopped")
        self.assertEqual(self.log.progress_bar.format(), "Stopped")

    def test_log_updates_do_not_pump_nested_gui_events(self):
        with patch.object(
            QApplication, "processEvents", side_effect=AssertionError("reentrant")
        ):
            self.log.start_progress()
            self.log.update_progress(1, 3, "Collecting")
            self.log.append("Status\n")
            self.log.finish_progress()

        self.assertEqual(self.log.progress_label.text(), "Complete")

    def test_synchronous_log_and_progress_paint_without_processing_other_events(self):
        text_paints = _PaintRecorder(self.log.text.viewport())
        progress_paints = _PaintRecorder(self.log.progress_label)
        pending_callback = []
        QTimer.singleShot(0, lambda: pending_callback.append(True))

        self.log.append("Selection step completed\n")
        self.log.start_progress("Refining selected points")

        self.assertGreater(text_paints.paint_count, 0)
        self.assertGreater(progress_paints.paint_count, 0)
        self.assertEqual(pending_callback, [])
        self.app.processEvents()
        self.assertEqual(pending_callback, [True])

    def test_redirector_filters_fragmented_camera_success_only(self):
        original = io.StringIO()
        fragments = [
            "Preparing\n(20", "002, 0, ", "1)", "\r",
            "(20002, 1, 2)\r\n", "(20024, 0, 1)\n",
            "Warning: (20002, 0, 1)\n", "(20002, broken, 1)\n",
            "(42, 0, 1)\n", "(20002, 0)\n", "Done without newline",
        ]
        with patch.object(sys, "stdout", original):
            with _StdoutRedirector(self.log) as redirector:
                for fragment in fragments:
                    self.assertEqual(redirector.write(fragment), len(fragment))
                    redirector.flush()
            self.assertIs(sys.stdout, original)

        visible = self.log.text.toPlainText()
        self.assertTrue(visible.startswith("Preparing\n(20024, 0, 1)\n"))
        self.assertNotIn("\n(20002, 0, 1)\n", visible)
        self.assertNotIn("(20002, 1, 2)", visible)
        self.assertIn("Warning: (20002, 0, 1)", visible)
        self.assertIn("(20002, broken, 1)", visible)
        self.assertIn("(42, 0, 1)", visible)
        self.assertIn("(20002, 0)", visible)
        self.assertTrue(visible.endswith("Done without newline"))
        self.assertEqual(original.getvalue(), "".join(fragments))

        self.log.show_camera_diagnostics.setChecked(True)
        diagnostics = self.log.text.toPlainText()
        self.assertIn("Preparing\n(20002, 0, 1)\n(20002, 1, 2)\n", diagnostics)
        self.assertIn("(20024, 0, 1)", diagnostics)
        self.log.show_camera_diagnostics.setChecked(False)
        self.assertEqual(self.log.text.toPlainText(), visible)

    def test_redirector_worker_captures_only_its_own_thread(self):
        original = io.StringIO()
        entered = threading.Event()
        release = threading.Event()

        def worker():
            with _StdoutRedirector(self.log):
                print("Worker acquisition")
                entered.set()
                release.wait(timeout=5)
                print("(20002, 0, 1)", end="\r")
                print("Worker complete")

        with patch.object(sys, "stdout", original):
            thread = threading.Thread(target=worker)
            thread.start()
            try:
                self.assertTrue(entered.wait(timeout=5))
                print("Unrelated GUI activity")
                self.assertEqual(self.log.text.toPlainText(), "")
            finally:
                release.set()
                thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertIs(sys.stdout, original)

        self.app.processEvents()

        self.assertEqual(
            self.log.text.toPlainText(), "Worker acquisition\nWorker complete\n"
        )
        self.assertIn("Unrelated GUI activity", original.getvalue())
        self.assertIn("(20002, 0, 1)\r", original.getvalue())

    def test_deleted_log_does_not_interrupt_stdout_or_cleanup(self):
        discarded_log = LogWindow()
        discarded_log.deleteLater()
        QCoreApplication.sendPostedEvents(discarded_log, QEvent.DeferredDelete)
        original = io.StringIO()
        with patch.object(sys, "stdout", original):
            with _StdoutRedirector(discarded_log):
                print("Hardware cleanup complete")
            self.assertIs(sys.stdout, original)

        self.assertEqual(original.getvalue(), "Hardware cleanup complete\n")

    def test_overlapping_redirectors_restore_original_after_out_of_order_exit(self):
        second_log = LogWindow("Second acquisition")
        original = io.StringIO()
        first = _StdoutRedirector(self.log)
        second = _StdoutRedirector(second_log)
        try:
            with patch.object(sys, "stdout", original):
                first.__enter__()
                print("First task")
                second.__enter__()
                print("Both active")
                first.__exit__(None, None, None)
                self.assertIs(sys.stdout, second)
                print("Only second task")
                first.write("Late inactive write\n")
                second.__exit__(None, None, None)
                self.assertIs(sys.stdout, original)
                print("Back to original")

            self.assertEqual(self.log.text.toPlainText(), "First task\nBoth active\n")
            self.assertEqual(second_log.text.toPlainText(), "Both active\nOnly second task\n")
            self.assertEqual(
                original.getvalue(),
                "First task\nBoth active\nOnly second task\n"
                "Late inactive write\nBack to original\n",
            )
        finally:
            second_log.close()

    def test_overlapping_worker_redirectors_do_not_recapture_finished_thread(self):
        second_log = LogWindow("Second acquisition")
        original = io.StringIO()
        first_entered = threading.Event()
        second_entered = threading.Event()
        first_exited = threading.Event()
        second_exited = threading.Event()

        def first_worker():
            with _StdoutRedirector(self.log):
                print("First worker active")
                first_entered.set()
                second_entered.wait(timeout=3)
            first_exited.set()
            # This still reaches stdout via the second redirector, whose raw
            # tee chain includes the now-inactive first redirector.
            print("First worker outside context")
            second_exited.wait(timeout=3)

        def second_worker():
            first_entered.wait(timeout=3)
            with _StdoutRedirector(second_log):
                print("Second worker active")
                second_entered.set()
                first_exited.wait(timeout=3)
            second_exited.set()

        try:
            with patch.object(sys, "stdout", original):
                first_thread = threading.Thread(target=first_worker)
                second_thread = threading.Thread(target=second_worker)
                first_thread.start()
                second_thread.start()
                first_thread.join(timeout=5)
                second_thread.join(timeout=5)
                self.assertFalse(first_thread.is_alive())
                self.assertFalse(second_thread.is_alive())
                self.assertIs(sys.stdout, original)

            self.app.processEvents()
            self.assertEqual(self.log.text.toPlainText(), "First worker active\n")
            self.assertEqual(second_log.text.toPlainText(), "Second worker active\n")
            self.assertIn("First worker outside context\n", original.getvalue())
        finally:
            second_log.close()


if __name__ == "__main__":
    unittest.main()
