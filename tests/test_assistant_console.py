"""Focused interaction tests for the protected inline assistant prompt."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import Qt
from qtpy.QtGui import QFontDatabase, QInputMethodEvent, QTextCursor
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QPlainTextEdit

from napari_raman_widget.assistant_console import AssistantConsole


class AssistantConsoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.console = AssistantConsole()
        self.console.resize(420, 260)
        self.console.show()
        self.console.setFocus()
        self.app.processEvents()

    def tearDown(self):
        self.console.close()
        self.console.deleteLater()

    def test_is_one_accessible_theme_inheriting_monospace_surface(self):
        self.assertIsInstance(self.console, QPlainTextEdit)
        self.assertEqual(self.console.accessibleName(), "AI assistant console")
        self.assertIn("Enter", self.console.accessibleDescription())
        self.assertGreaterEqual(self.console.font().pointSizeF(), 10)
        self.assertTrue(self.console.font().fixedPitch())
        preferred = [
            family for family in (
                "Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono",
                "Liberation Mono",
            ) if family in QFontDatabase().families()
        ]
        if preferred:
            self.assertEqual(self.console.font().family(), preferred[0])
        self.assertNotIn("background", self.console.styleSheet())
        self.assertEqual(
            self.console.lineWrapMode(), QPlainTextEdit.WidgetWidth
        )
        self.assertIn("does not execute shell", self.console.toPlainText())
        self.assertTrue(self.console.toPlainText().endswith("> "))

    def test_enter_submits_whole_multiline_draft_and_locally_debounces(self):
        submitted = []
        self.console.commandSubmitted.connect(submitted.append)
        QTest.keyClicks(self.console, "first line")
        QTest.keyClick(
            self.console, Qt.Key_Return, Qt.ShiftModifier
        )
        QTest.keyClicks(self.console, "second line")

        self.assertEqual(
            self.console.command_text(), "first line\nsecond line"
        )
        QTest.keyClick(self.console, Qt.Key_Return)
        QTest.keyClick(self.console, Qt.Key_Return)

        self.assertEqual(submitted, ["first line\nsecond line"])
        self.assertTrue(self.console.isReadOnly())
        self.assertIn("[working…]", self.console.toPlainText())
        self.console.set_busy(False)
        self.assertFalse(self.console.isReadOnly())
        self.assertNotIn("[working…]", self.console.toPlainText())
        self.assertTrue(self.console.toPlainText().endswith("> "))

    def test_multiline_paste_is_one_draft_and_never_submits(self):
        submitted = []
        self.console.commandSubmitted.connect(submitted.append)
        QApplication.clipboard().setText("alpha\r\nbeta\rgamma")

        self.console.paste()

        self.assertEqual(self.console.command_text(), "alpha\nbeta\ngamma")
        self.assertEqual(submitted, [])

    def test_history_up_down_restores_the_unsubmitted_scratch_draft(self):
        for command in ("first", "second"):
            self.console.set_command_text(command)
            self.assertTrue(self.console.submit_command())
            self.console.set_busy(False)
        self.console.set_command_text("scratch")

        QTest.keyClick(self.console, Qt.Key_Up)
        self.assertEqual(self.console.command_text(), "second")
        QTest.keyClick(self.console, Qt.Key_Up)
        self.assertEqual(self.console.command_text(), "first")
        QTest.keyClick(self.console, Qt.Key_Down)
        self.assertEqual(self.console.command_text(), "second")
        QTest.keyClick(self.console, Qt.Key_Down)
        self.assertEqual(self.console.command_text(), "scratch")

    def test_async_output_preserves_reverse_selection_and_draft_undo(self):
        self.console.set_command_text("ab😀cd")
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.PreviousCharacter)
        cursor.movePosition(
            QTextCursor.PreviousCharacter, QTextCursor.KeepAnchor, 2
        )
        selected = cursor.selectedText()
        was_reversed = cursor.anchor() > cursor.position()
        self.console.setTextCursor(cursor)

        self.console.append_message("tool", "Output arrived 🧪")

        restored = self.console.textCursor()
        self.assertEqual(restored.selectedText(), selected)
        self.assertEqual(restored.anchor() > restored.position(), was_reversed)
        QTest.keyClicks(self.console, "X")
        edited = self.console.command_text()
        self.assertIn("X", edited)
        QTest.keyClick(self.console, Qt.Key_Z, Qt.ControlModifier)
        self.assertEqual(self.console.command_text(), "ab😀cd")
        self.assertIn("Output arrived 🧪", self.console.toPlainText())

    def test_busy_preserves_draft_and_restores_exactly_one_active_prompt(self):
        self.console.set_command_text("keep this")

        self.console.set_busy(True)
        self.assertTrue(self.console.isReadOnly())
        self.assertNotIn("keep this", self.console.toPlainText())
        self.console.append_message("assistant", "Finished")
        self.console.set_busy(False)

        self.assertEqual(self.console.command_text(), "keep this")
        self.assertFalse(self.console.isReadOnly())
        self.assertTrue(self.console.toPlainText().endswith("> keep this"))

    def test_programmatic_prefix_mutation_is_rolled_back(self):
        self.console.append_message("system", "Immutable sentinel 😀")
        self.console.set_command_text("draft")
        original = self.console.toPlainText()
        cursor = QTextCursor(self.console.document())
        cursor.setPosition(0)

        cursor.insertText("BAD")

        self.assertEqual(self.console.toPlainText(), original)
        self.assertEqual(self.console.command_text(), "draft")

    def test_ime_output_is_deferred_until_composition_commits(self):
        preedit = QInputMethodEvent("かな", [])
        QApplication.sendEvent(self.console, preedit)
        self.console.append_message("assistant", "Queued output")
        self.assertNotIn("Queued output", self.console.toPlainText())

        commit = QInputMethodEvent()
        commit.setCommitString("仮名")
        QApplication.sendEvent(self.console, commit)

        self.assertEqual(self.console.command_text(), "仮名")
        self.assertIn("Queued output", self.console.toPlainText())


if __name__ == "__main__":
    unittest.main()
