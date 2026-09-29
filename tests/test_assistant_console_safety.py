"""Adversarial edit-boundary checks for the terminal-style Assistant UI."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import Qt
from qtpy.QtGui import QInputMethodEvent, QTextCursor
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication

from napari_raman_widget.assistant_console import AssistantConsole


class AssistantConsoleSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.console = AssistantConsole()
        self.console.show()
        self.console.setFocus()
        self.app.processEvents()

    def tearDown(self):
        self.console.close()

    def transcript_prefix(self):
        """Return everything before the public editable command string."""
        text = self.console.toPlainText()
        command = self.console.command_text()
        if command:
            self.assertTrue(text.endswith(command))
            return text[: -len(command)]
        return text

    def test_cross_boundary_cut_select_all_typing_and_multiline_paste(self):
        self.console.append_message(
            "assistant", "Immutable transcript 😀 <b>plain text</b>"
        )
        self.console.set_command_text("draft 🧪")
        immutable = self.transcript_prefix()

        cursor = self.console.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(
            self.console.document().characterCount() - 1,
            QTextCursor.KeepAnchor,
        )
        self.console.setTextCursor(cursor)
        self.console.cut()
        self.assertEqual(self.transcript_prefix(), immutable)

        self.console.set_command_text("replace me")
        QTest.keyClick(self.console, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(self.console, "x")
        self.assertEqual(self.transcript_prefix(), immutable)
        self.assertEqual(self.console.command_text(), "x")

        QTest.keyClick(self.console, Qt.Key_A, Qt.ControlModifier)
        QApplication.clipboard().setText("first line\r\nsecond line")
        submitted = []
        self.console.commandSubmitted.connect(submitted.append)
        self.console.paste()
        self.assertEqual(self.transcript_prefix(), immutable)
        self.assertEqual(
            self.console.command_text(), "first line\nsecond line"
        )
        self.assertNotIn("\r", self.console.command_text())
        self.assertEqual(submitted, [])

    def test_undo_and_backspace_cannot_rewrite_transcript(self):
        self.console.append_message("system", "System sentinel 😀")
        self.console.set_command_text("abc")
        immutable = self.transcript_prefix()

        QTest.keyClicks(self.console, "d")
        for _ in range(4):
            QTest.keyClick(self.console, Qt.Key_Z, Qt.ControlModifier)
            self.assertEqual(self.transcript_prefix(), immutable)

        self.console.set_command_text("abc")
        QTest.keyClick(self.console, Qt.Key_Home)
        QTest.keyClick(self.console, Qt.Key_Backspace)
        self.assertEqual(self.transcript_prefix(), immutable)
        self.assertEqual(self.console.command_text(), "abc")

    def test_async_append_preserves_astral_draft_and_cursor_offset(self):
        self.console.append_message("assistant", "Before")
        self.console.set_command_text("ab😀cd")
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.PreviousCharacter, QTextCursor.MoveAnchor, 2)
        self.console.setTextCursor(cursor)

        self.console.append_message("tool", "Async result 🧪")
        QTest.keyClicks(self.console, "X")

        self.assertEqual(self.console.command_text(), "ab😀Xcd")
        self.assertIn("Async result 🧪", self.console.toPlainText())

    def test_ime_replacement_cannot_cross_unicode_prompt_boundary(self):
        self.console.append_message("assistant", "Transcript 😀 sentinel")
        self.console.set_command_text("abcdef")
        immutable = self.transcript_prefix()

        QTest.keyClick(self.console, Qt.Key_Home)
        event = QInputMethodEvent()
        event.setCommitString("漢", -8, 10)
        QApplication.sendEvent(self.console, event)

        self.assertEqual(self.transcript_prefix(), immutable)
        self.assertIn("Transcript 😀 sentinel", self.console.toPlainText())

    def test_busy_blocks_keyboard_clipboard_and_ime_but_allows_output(self):
        self.console.append_message("assistant", "Ready")
        self.console.set_command_text("draft")
        self.console.set_busy(True)
        snapshot = self.console.toPlainText()

        QTest.keyClicks(self.console, "blocked")
        QApplication.clipboard().setText("blocked paste")
        self.console.paste()
        event = QInputMethodEvent()
        event.setCommitString("遮")
        QApplication.sendEvent(self.console, event)
        self.assertEqual(self.console.toPlainText(), snapshot)

        self.console.append_message("assistant", "Worker finished")
        self.assertIn("Worker finished", self.console.toPlainText())
        self.console.set_busy(False)
        QTest.keyClicks(self.console, "Z")
        self.assertTrue(self.console.command_text().endswith("Z"))

    def test_submit_is_locally_debounced_and_commits_user_line_once(self):
        submitted = []
        self.console.commandSubmitted.connect(submitted.append)
        self.console.set_command_text("  hello 😀  ")

        self.console.submit_command()
        self.console.submit_command()
        self.app.processEvents()

        self.assertEqual(len(submitted), 1)
        self.assertEqual(submitted[0].strip(), "hello 😀")
        self.assertEqual(self.console.command_text(), "")
        self.assertIn("hello 😀", self.console.toPlainText())


if __name__ == "__main__":
    unittest.main()
