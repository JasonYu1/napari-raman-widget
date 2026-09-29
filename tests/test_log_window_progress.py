"""Progress-state checks for the embeddable calibration log."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication

from napari_raman_widget.log_window import LogWindow


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


if __name__ == "__main__":
    unittest.main()
