"""Opt-in profile lifecycle through the real Qt assistant, without an API."""

import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QCoreApplication, QEvent, Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QMessageBox, QWidget

from napari_raman_widget.assistant_history import HistoryError, HistoryStore
from napari_raman_widget.chat_panel import ChatPanel


UI = "napari_raman_widget.assistant_memory_ui"


class AssistantMemoryUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "history"
        self.owner = QWidget()
        self.owner._plot_windows = []
        self.panels = []
        self.chat = self.make_panel()

    def tearDown(self):
        for panel in self.panels:
            panel.deleteLater()
        self.owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.temp.cleanup()

    def make_panel(self):
        panel = ChatPanel(self.owner, history_store=HistoryStore(self.root))
        self.panels.append(panel)
        return panel

    def enable(self, panel=None):
        panel = panel or self.chat
        with patch(f"{UI}.QMessageBox.question", return_value=QMessageBox.Yes):
            panel.history_controls.save_check.setChecked(True)
        self.assertTrue(panel.history_controls._enabled)
        return panel.history_controls._active

    def conversation(self, question="Remember sample A", answer="Noted.", panel=None, tool=False, fail=False):
        panel = panel or self.chat
        responses = []
        if tool:
            responses.append(SimpleNamespace(content=[SimpleNamespace(
                type="tool_use", id="saved-tool-1", name="list_plots", input={},
            )], stop_reason="tool_use"))
        responses.append(RuntimeError("Mock API failure") if fail else SimpleNamespace(
            content=[SimpleNamespace(type="text", text=answer)], stop_reason="end_turn",
        ))
        create = Mock(side_effect=responses)
        api = SimpleNamespace(Anthropic=Mock(return_value=SimpleNamespace(
            messages=SimpleNamespace(create=create),
        )))
        with patch.dict(sys.modules, {"anthropic": api}):
            panel.console.set_command_text(question)
            QTest.keyClick(panel.console, Qt.Key_Return)
            deadline = time.monotonic() + 10
            while panel._busy and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)
            self.app.processEvents()
        self.assertFalse(panel._busy)
        return create

    def test_default_private_and_declined_opt_in_do_not_create_files(self):
        self.conversation()
        self.assertFalse(self.root.exists())
        with patch(f"{UI}.QMessageBox.question", return_value=QMessageBox.No):
            self.chat.history_controls.save_check.setChecked(True)
        self.assertFalse(self.chat.history_controls.save_check.isChecked())
        self.assertFalse(self.root.exists())

    def test_opt_in_saves_current_chat_and_restores_without_replaying(self):
        self.conversation(tool=True)
        profile = self.enable()
        with patch.object(ChatPanel, "_execute_tool") as execute:
            restored = self.make_panel()
        execute.assert_not_called()
        self.assertEqual(restored.history_controls._active, profile)
        self.assertIn("Remember sample A", restored.console.toPlainText())
        self.assertIn("Noted.", restored.console.toPlainText())
        self.assertEqual(len(restored._messages), 4)
        self.assertIsInstance(restored._messages[1]["content"][0], dict)
        QTest.keyClick(restored.console, Qt.Key_Up)
        self.assertEqual(restored.console.command_text(), "Remember sample A")
        create = self.conversation("What was my sample?", "Sample A.", panel=restored)
        self.assertEqual(create.call_args.kwargs["messages"][0]["content"], "Remember sample A")

    def test_completed_turn_is_saved_without_widget_close(self):
        profile = self.enable()
        self.conversation()
        saved = HistoryStore(self.root).load_profile(profile)
        self.assertEqual(saved["messages"][0]["content"], "Remember sample A")
        self.assertIn({"who": "assistant", "text": "Noted."}, saved["transcript"])

    def test_new_profile_and_switch_keep_customer_context_separate(self):
        first = self.enable()
        self.conversation()
        controls = self.chat.history_controls
        with patch(f"{UI}.QInputDialog.getText", return_value=("Sample B", True)):
            controls._new_profile()
        second = controls._active
        self.assertNotEqual(first, second)
        self.assertEqual(self.chat._messages, [])
        self.assertNotIn("Remember sample A", self.chat.console.toPlainText())
        QTest.keyClick(self.chat.console, Qt.Key_Up)
        self.assertEqual(self.chat.console.command_text(), "")
        self.conversation("Remember sample B", "B noted.")
        controls._select_profile(controls.profile_combo.findData(first))
        self.assertIn("Remember sample A", self.chat.console.toPlainText())
        self.assertNotIn("Remember sample B", self.chat.console.toPlainText())
        self.assertEqual(self.chat._messages[0]["content"], "Remember sample A")

    def test_opt_out_starts_private_session_and_does_not_delete_saved_profile(self):
        first = self.enable()
        self.conversation()
        self.chat.history_controls.save_check.setChecked(False)
        self.assertEqual(self.chat._messages, [])
        self.assertEqual(self.chat.console._history, [])
        self.assertIsNone(self.chat.history_controls._active)
        self.conversation("Private sample C", "Private reply")
        saved = HistoryStore(self.root).load_profile(first)
        self.assertNotIn("Private sample C", str(saved))
        restored = self.make_panel()
        self.assertEqual(restored._messages, [])
        self.assertFalse(restored.history_controls._enabled)

    def test_enabling_private_chat_creates_new_profile_not_customer_merge(self):
        first = self.enable()
        self.conversation()
        self.chat.history_controls.save_check.setChecked(False)
        self.conversation("Separate private customer", "Okay")
        second = self.enable()
        self.assertNotEqual(first, second)
        self.assertNotIn("Separate private customer", str(HistoryStore(self.root).load_profile(first)))
        self.assertNotIn("Remember sample A", str(HistoryStore(self.root).load_profile(second)))

    def test_clear_removes_saved_and_model_and_recall_history_only_for_current_profile(self):
        first = self.enable()
        self.conversation()
        controls = self.chat.history_controls
        controls._create_profile("Other customer")
        second = controls._active
        self.conversation("Other customer details", "Okay")
        controls._select_profile(controls.profile_combo.findData(first))
        self.chat.console.set_command_text("Unsent secret")
        with patch(f"{UI}.QMessageBox.question", return_value=QMessageBox.Yes):
            controls._clear_history()
        self.assertEqual(self.chat._messages, [])
        self.assertEqual(self.chat.console.command_text(), "")
        self.assertEqual(self.chat.console._history, [])
        self.assertEqual(HistoryStore(self.root).load_profile(first)["messages"], [])
        self.assertIn("Other customer details", str(HistoryStore(self.root).load_profile(second)))
        self.chat.console.undo()
        self.assertNotIn("Unsent secret", self.chat.console.toPlainText())
        restored = self.make_panel()
        self.assertEqual(restored._messages, [])

    def test_declined_clear_and_switch_preserve_private_draft_and_context(self):
        self.conversation()
        self.chat.console.set_command_text("Draft")
        with patch(f"{UI}.QMessageBox.question", return_value=QMessageBox.No):
            self.chat.history_controls._clear_history()
            self.chat.history_controls._new_profile()
        self.assertEqual(self.chat.console.command_text(), "Draft")
        self.assertEqual(self.chat._messages[0]["content"], "Remember sample A")
        self.assertFalse(self.root.exists())

    def test_controls_and_direct_slots_cannot_switch_while_request_running(self):
        self.enable()
        controls = self.chat.history_controls
        current = controls._active
        self.chat._on_set_busy(True)
        self.assertFalse(controls.save_check.isEnabled())
        self.assertFalse(controls.profile_combo.isEnabled())
        self.assertFalse(controls.menu_button.isEnabled())
        with patch(f"{UI}.QMessageBox.question") as question:
            controls._toggle_saving(False)
            controls._new_profile()
            controls._select_profile(0)
            controls._clear_history()
            controls._delete_profile()
        self.assertFalse(controls.delete_action.isEnabled())
        question.assert_not_called()
        self.assertEqual(controls._active, current)
        self.chat._on_set_busy(False)

    def test_save_failure_reports_problem_without_losing_current_chat(self):
        self.enable()
        controls = self.chat.history_controls
        with patch.object(controls.store, "save_profile", side_effect=HistoryError("Disk full")):
            self.conversation()
        self.assertFalse(controls._enabled)
        self.assertTrue(controls._unavailable)
        self.assertIn("Disk full", self.chat.console.toPlainText())
        self.assertEqual(self.chat._messages[0]["content"], "Remember sample A")
        self.assertFalse(self.chat.console.isReadOnly())

    def test_unreadable_settings_fail_closed_without_deleting_data(self):
        store = HistoryStore(self.root)
        with patch.object(store, "read_settings", side_effect=HistoryError("Invalid saved JSON")):
            panel = ChatPanel(self.owner, history_store=store)
        self.panels.append(panel)
        self.assertEqual(panel._messages, [])
        self.assertFalse(panel.history_controls.save_check.isEnabled())
        self.assertIn("Invalid saved JSON", panel.console.toPlainText())
        self.assertFalse(self.root.exists())

    def test_final_api_failure_preserves_attempted_tools_when_resumed(self):
        self.enable()
        self.conversation(tool=True, fail=True)
        self.assertIn("Mock API failure", self.chat.console.toPlainText())
        restored = self.make_panel()
        self.assertEqual(len(restored._messages), 4)
        self.assertEqual(restored._messages[1]["content"][0]["name"], "list_plots")
        self.assertEqual(restored._messages[2]["content"][0]["tool_use_id"], "saved-tool-1")
        self.assertIn("Application notice", restored._messages[-1]["content"])
        with patch.object(restored, "_execute_tool") as execute:
            self.conversation("What happened?", "The tool was attempted.", panel=restored)
        execute.assert_not_called()

    def test_interrupted_tool_batch_retains_known_results_and_marks_remainder_unknown(self):
        profile = self.enable()
        self.chat._messages = [
            {"role": "user", "content": "Run the requested steps"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "one", "name": "first", "input": {}},
                {"type": "tool_use", "id": "two", "name": "second", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "one", "content": "First action completed."},
            ]},
        ]
        self.chat._record_conversation_failure()
        self.chat.history_controls._save_current()
        saved = HistoryStore(self.root).load_profile(profile)
        results = saved["messages"][2]["content"]
        self.assertEqual(results[0]["content"], "First action completed.")
        self.assertTrue(results[1]["is_error"])
        self.assertIn("status unknown", results[1]["content"])
        self.assertIn("Do not replay", results[1]["content"])

    def test_failed_profile_registration_removes_only_new_orphan_file(self):
        self.conversation()
        controls = self.chat.history_controls
        with patch.object(controls.store, "write_settings", side_effect=HistoryError("Settings unwritable")):
            with patch(f"{UI}.QMessageBox.question", return_value=QMessageBox.Yes):
                controls.save_check.setChecked(True)
        self.assertFalse(controls._enabled)
        self.assertEqual(list((self.root / "profiles").glob("*.json")), [])
        self.assertEqual(self.chat._messages[0]["content"], "Remember sample A")

    def test_saved_context_is_bounded_in_memory_as_well_as_on_disk(self):
        profile = self.enable()
        self.chat._messages = [
            message for index in range(35) for message in [
                {"role": "user", "content": f"Question {index}"},
                {"role": "assistant", "content": f"Reply {index}"},
            ]
        ]
        self.chat.history_controls._save_current()
        self.assertEqual(len(self.chat._messages), 60)
        self.assertEqual(self.chat._messages[0]["content"], "Question 5")
        self.assertEqual(self.chat._messages, HistoryStore(self.root).load_profile(profile)["messages"])

    def test_delete_selected_profile_keeps_other_profiles_and_starts_private(self):
        first = self.enable()
        self.conversation("Customer A details", "A noted.")
        controls = self.chat.history_controls
        controls._create_profile("Customer B")
        second = controls._active
        self.conversation("Customer B details", "B noted.")
        controls._select_profile(controls.profile_combo.findData(first))
        self.chat.console.set_command_text("Unsent draft")
        other_path = self.root / "profiles" / f"{second}.json"
        other_before = other_path.read_bytes()
        self.assertTrue(controls.delete_action.isEnabled())
        with patch(f"{UI}.QMessageBox.question", return_value=QMessageBox.Yes) as confirm:
            controls.delete_action.trigger()
        self.assertIn('"Profile 1"', confirm.call_args.args[2])
        self.assertEqual(confirm.call_args.args[-1], QMessageBox.No)
        self.assertFalse((self.root / "profiles" / f"{first}.json").exists())
        self.assertEqual(other_path.read_bytes(), other_before)
        self.assertEqual(controls.profile_combo.findData(first), -1)
        self.assertNotEqual(controls.profile_combo.findData(second), -1)
        self.assertIsNone(controls._active)
        self.assertFalse(controls.save_check.isChecked())
        self.assertEqual(controls.profile_combo.currentText(), "Private session")
        self.assertEqual(self.chat._messages, [])
        self.assertEqual(self.chat.console._history, [])
        self.assertEqual(self.chat.console.command_text(), "")
        self.chat.console.undo()
        self.assertNotIn("Unsent draft", self.chat.console.toPlainText())
        self.assertNotIn("Customer A details", self.chat.console.toPlainText())
        restored = self.make_panel()
        self.assertEqual(restored.history_controls._profiles, [{"id": second, "name": "Customer B"}])
        self.assertEqual(restored._messages, [])
        self.assertFalse(restored.history_controls._enabled)
        self.conversation("Private after deletion", "Okay")
        self.assertFalse((self.root / "profiles" / f"{first}.json").exists())

    def test_delete_last_profile_leaves_empty_profile_list(self):
        self.enable()
        controls = self.chat.history_controls
        with patch(f"{UI}.QMessageBox.question", return_value=QMessageBox.Yes):
            controls._delete_profile()
        self.assertEqual(HistoryStore(self.root).read_settings(), {
            "enabled": False, "active_profile": None, "profiles": [],
        })
        self.assertEqual(controls.profile_combo.count(), 1)
        self.assertFalse(controls.delete_action.isEnabled())
        self.assertEqual(self.make_panel().history_controls.profile_combo.count(), 1)
        # Removing a profile frees the name and slot for a new one.
        self.enable()
        self.assertEqual(controls.profile_combo.currentText(), "Profile 1")

    def test_declining_delete_preserves_file_context_draft_and_profile(self):
        first = self.enable()
        self.conversation()
        controls = self.chat.history_controls
        before = {path.name: path.read_bytes() for path in self.root.rglob("*.json")}
        self.chat.console.set_command_text("Draft to keep")
        with patch(f"{UI}.QMessageBox.question", return_value=QMessageBox.No):
            controls._delete_profile()
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.root.rglob("*.json")})
        self.assertEqual(controls._active, first)
        self.assertTrue(controls._enabled)
        self.assertEqual(self.chat.console.command_text(), "Draft to keep")
        self.assertEqual(self.chat._messages[0]["content"], "Remember sample A")

    def test_private_session_cannot_delete_a_profile(self):
        controls = self.chat.history_controls
        self.assertFalse(controls.delete_action.isEnabled())
        with patch(f"{UI}.QMessageBox.question") as confirm:
            with patch.object(controls.store, "delete_profile") as delete:
                controls._delete_profile()
        confirm.assert_not_called()
        delete.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_delete_file_failure_keeps_chat_and_stops_saving(self):
        profile = self.enable()
        self.conversation()
        controls = self.chat.history_controls
        before = HistoryStore(self.root).load_profile(profile)
        with patch.object(controls.store, "clear_profile", side_effect=HistoryError("File locked")):
            with patch(f"{UI}.QMessageBox.question", return_value=QMessageBox.Yes):
                controls._delete_profile()
        self.assertEqual(HistoryStore(self.root).load_profile(profile), before)
        self.assertEqual(controls._active, profile)
        self.assertEqual(self.chat._messages[0]["content"], "Remember sample A")
        self.assertIn("File locked", self.chat.console.toPlainText())
        self.assertFalse(controls.delete_action.isEnabled())

    def test_delete_metadata_failure_forgets_cleared_chat_without_resurrection(self):
        profile = self.enable()
        self.conversation()
        controls = self.chat.history_controls
        with patch.object(controls.store, "write_settings", side_effect=HistoryError("Settings locked")):
            with patch(f"{UI}.QMessageBox.question", return_value=QMessageBox.Yes):
                controls._delete_profile()
        self.assertFalse((self.root / "profiles" / f"{profile}.json").exists())
        self.assertEqual(self.chat._messages, [])
        self.assertEqual(self.chat.console._history, [])
        self.assertTrue(controls._unavailable)
        self.assertFalse(controls._enabled)
        self.assertIn("deletion incomplete", self.chat.console.toPlainText())
        self.assertNotIn("existing files have not been cleared", self.chat.console.toPlainText())
        self.conversation("Continue privately", "Okay")
        self.assertFalse((self.root / "profiles" / f"{profile}.json").exists())
        # Restart can retry removing the remaining empty profile entry.
        restored = self.make_panel()
        self.assertEqual(restored._messages, [])
        self.assertFalse(restored.history_controls._enabled)
        self.assertTrue(restored.history_controls._deletion_pending)
        self.assertTrue(restored.history_controls.delete_action.isEnabled())
        self.conversation("Private after restart", "Okay", panel=restored)
        self.assertFalse((self.root / "profiles" / f"{profile}.json").exists())
        with patch(f"{UI}.QMessageBox.question", return_value=QMessageBox.Yes):
            restored.history_controls._delete_profile()
        self.assertEqual(HistoryStore(self.root).read_settings()["profiles"], [])


if __name__ == "__main__":
    unittest.main()
