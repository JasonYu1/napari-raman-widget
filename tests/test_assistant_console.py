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

    def test_replace_conversation_bulk_restores_messages_and_history(self):
        submitted = []
        self.console.commandSubmitted.connect(submitted.append)
        entries = [
            {"who": "user", "text": "inspect sample"},
            {"who": "assistant", "text": "Found a peak 🔬"},
            {"who": "tool", "text": "Plot configured"},
            {"who": "system", "text": "Profile restored"},
            {"who": "user", "text": "compare\r\nsecond scan"},
        ]

        self.console.replace_conversation(
            entries, commands=["saved first", "saved\r\nsecond"]
        )

        text = self.console.toPlainText()
        self.assertIn("> inspect sample\n", text)
        self.assertIn("Assistant: Found a peak 🔬\n", text)
        self.assertIn("Tool: Plot configured\n", text)
        self.assertIn("System: Profile restored\n", text)
        self.assertIn("> compare\nsecond scan\n", text)
        self.assertTrue(text.endswith("> "))
        self.assertEqual(self.console.command_text(), "")
        self.assertEqual(submitted, [])

        QTest.keyClick(self.console, Qt.Key_Up)
        self.assertEqual(self.console.command_text(), "saved\nsecond")
        QTest.keyClick(self.console, Qt.Key_Up)
        self.assertEqual(self.console.command_text(), "saved first")
        QTest.keyClick(self.console, Qt.Key_Down)
        self.assertEqual(self.console.command_text(), "saved\nsecond")
        QTest.keyClick(self.console, Qt.Key_Down)
        self.assertEqual(self.console.command_text(), "")

    def test_replace_conversation_infers_history_and_empty_reset_is_fresh(self):
        self.console.replace_conversation([
            {"who": "user", "text": "one"},
            {"who": "assistant", "text": "reply"},
            {"who": "user", "text": "two\rthree"},
        ])

        QTest.keyClick(self.console, Qt.Key_Up)
        self.assertEqual(self.console.command_text(), "two\nthree")
        QTest.keyClick(self.console, Qt.Key_Up)
        self.assertEqual(self.console.command_text(), "one")

        self.console.replace_conversation([])

        self.assertEqual(
            self.console.toPlainText(),
            self.console._BANNER + self.console.PROMPT,
        )
        self.assertEqual(self.console.command_text(), "")
        QTest.keyClick(self.console, Qt.Key_Up)
        self.assertEqual(self.console.command_text(), "")
        QTest.keyClick(self.console, Qt.Key_Z, Qt.ControlModifier)
        self.assertEqual(self.console.command_text(), "")

    def test_replace_conversation_clears_draft_ime_and_deferred_output(self):
        self.console.set_command_text("old draft")
        preedit = QInputMethodEvent("かな", [])
        QApplication.sendEvent(self.console, preedit)
        self.console.append_message("assistant", "stale deferred output")
        self.assertNotIn("stale deferred output", self.console.toPlainText())

        self.console.replace_conversation([
            {"who": "assistant", "text": "restored output"},
        ])

        self.assertEqual(self.console.command_text(), "")
        self.assertIn("Assistant: restored output", self.console.toPlainText())
        self.assertNotIn("old draft", self.console.toPlainText())
        self.assertNotIn("stale deferred output", self.console.toPlainText())
        self.assertFalse(self.console._ime_active)
        self.assertEqual(self.console._pending_messages, [])

    def test_replace_conversation_rejects_busy_or_invalid_state_atomically(self):
        self.console.replace_conversation([
            {"who": "assistant", "text": "keep me"},
        ], commands=["kept command"])
        self.console.set_command_text("keep draft")
        self.console.set_busy(True)
        busy_text = self.console.toPlainText()

        with self.assertRaises(RuntimeError):
            self.console.replace_conversation([])
        self.assertEqual(self.console.toPlainText(), busy_text)
        self.console.set_busy(False)
        self.assertEqual(self.console.command_text(), "keep draft")

        original = self.console.toPlainText()
        invalid_calls = (
            lambda: self.console.replace_conversation("not a list"),
            lambda: self.console.replace_conversation([{"who": "alien", "text": "x"}]),
            lambda: self.console.replace_conversation([{"who": "user"}]),
            lambda: self.console.replace_conversation(
                [{"who": "user", "text": "x"}], commands=[""]
            ),
        )
        for invalid_call in invalid_calls:
            with self.subTest(invalid_call=invalid_call):
                with self.assertRaises((TypeError, ValueError)):
                    invalid_call()
                self.assertEqual(self.console.toPlainText(), original)
                self.assertEqual(self.console.command_text(), "keep draft")

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
