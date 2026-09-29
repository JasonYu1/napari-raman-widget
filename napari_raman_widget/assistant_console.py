"""A protected terminal-style text console for the embedded assistant."""

from __future__ import annotations

from qtpy.QtCore import QEvent, QThread, Qt, Signal
from qtpy.QtGui import (
    QFont,
    QFontDatabase,
    QInputMethodEvent,
    QKeySequence,
    QTextCursor,
    QTextOption,
)
from qtpy.QtWidgets import QApplication, QMenu, QPlainTextEdit

try:  # QAction moved from QtWidgets to QtGui in Qt 6.
    from qtpy.QtWidgets import QAction
except ImportError:  # pragma: no cover - depends on the selected Qt binding
    from qtpy.QtGui import QAction


class AssistantConsole(QPlainTextEdit):
    """One text surface containing immutable output and an inline prompt.

    The widget is not a shell or Python console.  It only emits the text after
    its final ``> `` prompt; :class:`ChatPanel` decides how to handle it.
    """

    commandSubmitted = Signal(str)
    _appendRequested = Signal(str, str)
    _busyRequested = Signal(bool)

    PROMPT = "> "
    _BUSY_LINE = "[working…]"
    _BANNER = (
        "Raman AI Assistant\n"
        "Type a request and press Enter. Shift+Enter adds a line; "
        "Up/Down recalls commands.\n"
        "This prompt sends requests to the assistant; it does not execute "
        "shell or Python code.\n\n"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        installed = set(QFontDatabase().families())
        for family in (
            "Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono",
            "Liberation Mono",
        ):
            if family in installed:
                font.setFamily(family)
                break
        font.setStyleHint(QFont.Monospace)
        font.setFixedPitch(True)
        font.setPointSize(11)
        self.setFont(font)
        # Pin only typography, not colors: Napari still supplies its theme.
        # Some offscreen/Windows Qt builds return a proportional system
        # "fixed" font, and application styles can otherwise override it.
        family = font.family().replace("\\", "\\\\").replace("'", "\\'")
        self.setStyleSheet(
            f"QPlainTextEdit {{ font-family: '{family}'; font-size: 11pt; }}"
        )
        self.setAccessibleName("AI assistant console")
        self.setAccessibleDescription(
            "Assistant transcript followed by an editable greater-than prompt. "
            "Press Enter to send and Shift+Enter for a new line."
        )
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.setTabChangesFocus(False)
        self.setAcceptDrops(True)
        self.setUndoRedoEnabled(False)
        self.setMinimumHeight(180)

        self._transcript = self._BANNER
        self._busy = False
        self._busy_draft = ""
        self._history = []
        self._history_index = None
        self._history_scratch = ""
        self._undo_stack = []
        self._redo_stack = []
        self._ime_active = False
        self._ime_before_state = None
        self._pending_messages = []
        self._rendering = False
        self._repairing = False
        self._last_good_draft = ""
        self._transcript_end = 0
        self._prompt_start = None
        self._input_start = None

        self._appendRequested.connect(
            self._append_message_gui, Qt.QueuedConnection
        )
        self._busyRequested.connect(self._set_busy_gui, Qt.QueuedConnection)
        self.textChanged.connect(self._protect_prefix)
        self._render("", caret_to_end=True)

    # ------------------------------------------------------------------
    # Public API used by ChatPanel and tests
    # ------------------------------------------------------------------
    def append_message(self, who, text):
        """Append one labelled message without disturbing the active draft."""
        who = str(who)
        text = str(text)
        if QThread.currentThread() != self.thread():
            self._appendRequested.emit(who, text)
            return
        self._append_message_gui(who, text)

    def set_busy(self, busy):
        """Lock or restore the prompt while an assistant turn is running."""
        busy = bool(busy)
        if QThread.currentThread() != self.thread():
            self._busyRequested.emit(busy)
            return
        self._set_busy_gui(busy)

    def command_text(self):
        """Return only the editable draft, preserving embedded newlines."""
        if self._busy:
            return self._busy_draft
        if self._input_start is None:
            return ""
        return self._text_range(self._input_start, self._document_end())

    def set_command_text(self, text):
        """Replace the draft without changing transcript or command history."""
        text = str(text)
        self._history_index = None
        self._history_scratch = text
        self._undo_stack.clear()
        self._redo_stack.clear()
        if self._busy:
            self._busy_draft = text
            return
        self._render(text, caret_to_end=True)

    def submit_command(self):
        """Commit and emit the current non-blank draft exactly once."""
        if self._busy:
            return False
        command = self.command_text().strip()
        if not command:
            return False
        self._history.append(command)
        self._history_index = None
        self._history_scratch = ""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._transcript = self._with_trailing_newline(
            self._transcript + self.PROMPT + command
        )
        self._busy_draft = ""
        self._busy = True
        self._render("", caret_to_end=True)
        self.commandSubmitted.emit(command)
        return True

    # ------------------------------------------------------------------
    # Transcript/prompt rendering
    # ------------------------------------------------------------------
    @staticmethod
    def _with_trailing_newline(text):
        return text if text.endswith("\n") else text + "\n"

    @staticmethod
    def _normalise_newlines(text):
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def _insert_cursor_text(cursor, text):
        """Insert LF-delimited plain text as real QTextDocument blocks."""
        parts = str(text).split("\n")
        for index, part in enumerate(parts):
            if index:
                cursor.insertBlock()
            if part:
                cursor.insertText(part)

    def _format_message(self, who, text):
        text = self._normalise_newlines(text).rstrip("\n")
        role = who.strip().lower()
        if role in {"user", "you"}:
            rendered = f"{self.PROMPT}{text}"
        else:
            label = {
                "assistant": "Assistant",
                "system": "System",
                "tool": "Tool",
            }.get(role, who.strip().title() or "Message")
            rendered = f"{label}: {text}"
        return self._with_trailing_newline(rendered)

    def _append_message_gui(self, who, text):
        if self._ime_active:
            self._pending_messages.append((who, text))
            return
        snapshot = self._capture_view()
        draft = self.command_text()
        self._transcript = self._with_trailing_newline(
            self._transcript + self._format_message(who, text)
        )
        self._render(draft, snapshot=snapshot)

    def _set_busy_gui(self, busy):
        if busy == self._busy:
            return
        if busy:
            self._busy_draft = self.command_text()
            self._busy = True
            self._render("", caret_to_end=True)
        else:
            self._busy = False
            draft, self._busy_draft = self._busy_draft, ""
            self._render(draft, caret_to_end=True)

    def _render(self, draft, *, snapshot=None, caret_to_end=False):
        draft = str(draft)
        vertical = self.verticalScrollBar()
        horizontal = self.horizontalScrollBar()
        old_vertical = vertical.value()
        old_horizontal = horizontal.value()
        was_at_bottom = vertical.value() >= vertical.maximum()

        self._rendering = True
        try:
            self.document().clear()
            cursor = QTextCursor(self.document())
            cursor.movePosition(QTextCursor.End)
            self._insert_cursor_text(cursor, self._transcript)
            self._transcript_end = cursor.position()
            if self._busy:
                cursor.insertText(self._BUSY_LINE)
                self._prompt_start = None
                self._input_start = None
            else:
                self._prompt_start = cursor.position()
                cursor.insertText(self.PROMPT)
                self._input_start = cursor.position()
                self._insert_cursor_text(cursor, draft)

            if snapshot is not None:
                anchor = self._map_endpoint(snapshot["anchor"])
                position = self._map_endpoint(snapshot["position"])
                restored = QTextCursor(self.document())
                restored.setPosition(anchor)
                restored.setPosition(position, QTextCursor.KeepAnchor)
                self.setTextCursor(restored)
            elif caret_to_end or not self._busy:
                cursor.clearSelection()
                cursor.movePosition(QTextCursor.End)
                self.setTextCursor(cursor)
            else:
                self.setTextCursor(cursor)
        finally:
            self._rendering = False

        self.setReadOnly(self._busy)

        self._last_good_draft = draft if not self._busy else self._busy_draft
        if snapshot is not None:
            if snapshot["at_bottom"]:
                vertical.setValue(vertical.maximum())
            else:
                vertical.setValue(min(snapshot["vertical"], vertical.maximum()))
            horizontal.setValue(
                min(snapshot["horizontal"], horizontal.maximum())
            )
        elif was_at_bottom or caret_to_end:
            vertical.setValue(vertical.maximum())
        else:
            vertical.setValue(min(old_vertical, vertical.maximum()))
            horizontal.setValue(min(old_horizontal, horizontal.maximum()))

    def _capture_view(self):
        cursor = self.textCursor()
        vertical = self.verticalScrollBar()
        return {
            "anchor": self._describe_endpoint(cursor.anchor()),
            "position": self._describe_endpoint(cursor.position()),
            "vertical": vertical.value(),
            "horizontal": self.horizontalScrollBar().value(),
            "at_bottom": vertical.value() >= vertical.maximum(),
        }

    def _describe_endpoint(self, position):
        if self._busy:
            if position >= self._transcript_end:
                return ("tail", position - self._transcript_end)
            return ("absolute", position)
        if position >= self._input_start:
            return ("draft", position - self._input_start)
        if position >= self._prompt_start:
            return ("prompt", position - self._prompt_start)
        return ("absolute", position)

    def _map_endpoint(self, endpoint):
        kind, offset = endpoint
        end = self._document_end()
        if kind == "absolute":
            return max(0, min(int(offset), self._transcript_end))
        if kind == "draft" and self._input_start is not None:
            return max(
                self._input_start,
                min(self._input_start + int(offset), end),
            )
        if kind == "prompt" and self._prompt_start is not None:
            prompt_end = self._input_start
            return max(
                self._prompt_start,
                min(self._prompt_start + int(offset), prompt_end),
            )
        if kind == "tail" and self._busy:
            return max(
                self._transcript_end,
                min(self._transcript_end + int(offset), end),
            )
        return end

    def _document_end(self):
        cursor = QTextCursor(self.document())
        cursor.movePosition(QTextCursor.End)
        return cursor.position()

    def _text_range(self, start, end):
        cursor = QTextCursor(self.document())
        document_end = self._document_end()
        cursor.setPosition(max(0, min(int(start), document_end)))
        cursor.setPosition(
            max(0, min(int(end), document_end)), QTextCursor.KeepAnchor
        )
        return cursor.selection().toPlainText()

    # ------------------------------------------------------------------
    # Draft-only editing and undo
    # ------------------------------------------------------------------
    def _draft_state(self):
        text = self.command_text()
        if self._busy or self._input_start is None:
            return (text, 0, 0)
        cursor = self.textCursor()
        end = self._document_end() - self._input_start
        anchor = max(0, min(cursor.anchor() - self._input_start, end))
        position = max(0, min(cursor.position() - self._input_start, end))
        return (text, anchor, position)

    def _apply_draft_state(self, state):
        text, anchor, position = state
        self._render(text, caret_to_end=True)
        cursor = QTextCursor(self.document())
        end = self._document_end()
        cursor.setPosition(min(self._input_start + anchor, end))
        cursor.setPosition(
            min(self._input_start + position, end), QTextCursor.KeepAnchor
        )
        self.setTextCursor(cursor)

    def _perform_edit(self, edit):
        if self._busy:
            return False
        before = self._draft_state()
        edit()
        after = self._draft_state()
        if before[0] == after[0]:
            return False
        self._undo_stack.append(before)
        self._redo_stack.clear()
        if self._history_index is not None:
            self._history_index = None
            self._history_scratch = after[0]
        self._last_good_draft = after[0]
        return True

    def _cursor_for_insertion(self):
        if self._busy or self._input_start is None:
            return None
        cursor = self.textCursor()
        if cursor.hasSelection():
            start, end = cursor.selectionStart(), cursor.selectionEnd()
            if end <= self._input_start:
                cursor.clearSelection()
                cursor.movePosition(QTextCursor.End)
            elif start < self._input_start:
                cursor.setPosition(self._input_start)
                cursor.setPosition(end, QTextCursor.KeepAnchor)
        elif cursor.position() < self._input_start:
            cursor.movePosition(QTextCursor.End)
        return cursor

    def _cursor_for_deletion(self):
        if self._busy or self._input_start is None:
            return None
        cursor = self.textCursor()
        if cursor.hasSelection():
            start, end = cursor.selectionStart(), cursor.selectionEnd()
            if end <= self._input_start:
                return None
            if start < self._input_start:
                cursor.setPosition(self._input_start)
                cursor.setPosition(end, QTextCursor.KeepAnchor)
            return cursor
        if cursor.position() < self._input_start:
            return None
        return cursor

    def _insert_text(self, text):
        if self._busy or not text:
            return False
        text = self._normalise_newlines(str(text))

        def edit():
            cursor = self._cursor_for_insertion()
            if cursor is None:
                return
            if cursor.hasSelection():
                cursor.removeSelectedText()
            self._insert_cursor_text(cursor, text)
            self.setTextCursor(cursor)

        return self._perform_edit(edit)

    def _delete_selection(self):
        cursor = self._cursor_for_deletion()
        if cursor is None or not cursor.hasSelection():
            return False
        return self._perform_edit(
            lambda: (cursor.removeSelectedText(), self.setTextCursor(cursor))
        )

    def _delete_key(self, *, backwards, by_word=False):
        def edit():
            cursor = self._cursor_for_deletion()
            if cursor is None:
                return
            if cursor.hasSelection():
                cursor.removeSelectedText()
            elif backwards:
                if cursor.position() <= self._input_start:
                    return
                if by_word:
                    original = cursor.position()
                    cursor.movePosition(
                        QTextCursor.PreviousWord, QTextCursor.KeepAnchor
                    )
                    if cursor.selectionStart() < self._input_start:
                        cursor.setPosition(original)
                        cursor.setPosition(
                            self._input_start, QTextCursor.KeepAnchor
                        )
                    cursor.removeSelectedText()
                else:
                    cursor.deletePreviousChar()
            elif by_word:
                cursor.movePosition(
                    QTextCursor.NextWord, QTextCursor.KeepAnchor
                )
                cursor.removeSelectedText()
            else:
                cursor.deleteChar()
            self.setTextCursor(cursor)

        return self._perform_edit(edit)

    def undo(self):
        if self._busy or not self._undo_stack:
            return
        current = self._draft_state()
        state = self._undo_stack.pop()
        self._redo_stack.append(current)
        self._apply_draft_state(state)

    def redo(self):
        if self._busy or not self._redo_stack:
            return
        current = self._draft_state()
        state = self._redo_stack.pop()
        self._undo_stack.append(current)
        self._apply_draft_state(state)

    def clear(self):
        """Clear only the editable draft; never the transcript."""
        self.set_command_text("")

    def insertPlainText(self, text):
        """Route public/plain-text insertion through the protected draft."""
        self._insert_text(text)

    # ------------------------------------------------------------------
    # Keyboard, clipboard, drag/drop, context menu, and IME gateways
    # ------------------------------------------------------------------
    def event(self, event):
        if event.type() == QEvent.ShortcutOverride:
            sequences = (
                QKeySequence.Copy,
                QKeySequence.Cut,
                QKeySequence.Paste,
                QKeySequence.Undo,
                QKeySequence.Redo,
                QKeySequence.SelectAll,
            )
            if any(event.matches(sequence) for sequence in sequences):
                event.accept()
                return True
        return super().event(event)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            self.copy()
            return
        if event.matches(QKeySequence.SelectAll):
            self.selectAll()
            return
        if event.matches(QKeySequence.Cut):
            self.cut()
            return
        if event.matches(QKeySequence.Paste):
            self.paste()
            return
        if event.matches(QKeySequence.Undo):
            self.undo()
            return
        if event.matches(QKeySequence.Redo):
            self.redo()
            return

        key = event.key()
        modifiers = event.modifiers()
        if self._busy:
            if key in (
                Qt.Key_Backspace,
                Qt.Key_Delete,
                Qt.Key_Return,
                Qt.Key_Enter,
            ) or event.text():
                event.accept()
                return
            super().keyPressEvent(event)
            return

        if key in (Qt.Key_Return, Qt.Key_Enter):
            if self._ime_active:
                super().keyPressEvent(event)
            elif modifiers & Qt.ShiftModifier:
                self._insert_text("\n")
            else:
                self.submit_command()
            return
        if key == Qt.Key_Up:
            self._history_previous()
            return
        if key == Qt.Key_Down:
            self._history_next()
            return
        if key == Qt.Key_Home and not (
            modifiers & (Qt.ControlModifier | Qt.MetaModifier)
        ):
            self._move_home(bool(modifiers & Qt.ShiftModifier))
            return
        if key in (Qt.Key_Backspace, Qt.Key_Delete):
            by_word = bool(modifiers & (Qt.ControlModifier | Qt.MetaModifier))
            self._delete_key(
                backwards=key == Qt.Key_Backspace,
                by_word=by_word,
            )
            return

        text = event.text()
        if text and (text == "\t" or all(ord(char) >= 32 for char in text)):
            self._insert_text(text)
            return
        super().keyPressEvent(event)

    def _move_home(self, keep_anchor):
        cursor = self.textCursor()
        if cursor.position() < self._input_start:
            return
        target = max(self._input_start, cursor.block().position())
        cursor.setPosition(
            target,
            QTextCursor.KeepAnchor if keep_anchor else QTextCursor.MoveAnchor,
        )
        self.setTextCursor(cursor)

    def _history_previous(self):
        if not self._history:
            return
        if self._history_index is None:
            self._history_scratch = self.command_text()
            self._history_index = len(self._history) - 1
        elif self._history_index > 0:
            self._history_index -= 1
        self._render(self._history[self._history_index], caret_to_end=True)

    def _history_next(self):
        if self._history_index is None:
            return
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            text = self._history[self._history_index]
        else:
            self._history_index = None
            text = self._history_scratch
        self._render(text, caret_to_end=True)

    def paste(self):
        mime = QApplication.clipboard().mimeData()
        self.insertFromMimeData(mime)

    def cut(self):
        if self._busy or not self._selection_fully_editable():
            return
        self.copy()
        self._delete_selection()

    def _selection_fully_editable(self):
        if self._busy or self._input_start is None:
            return False
        cursor = self.textCursor()
        return cursor.hasSelection() and cursor.selectionStart() >= self._input_start

    def canInsertFromMimeData(self, source):
        return not self._busy and source is not None and source.hasText()

    def insertFromMimeData(self, source):
        if self.canInsertFromMimeData(source):
            # Pasted lines remain one draft and never act as implicit Enter
            # key presses.  A later explicit Enter submits the whole request.
            text = self._normalise_newlines(source.text())
            self._insert_text(text)

    def dragEnterEvent(self, event):
        if not self._busy and event.mimeData().hasText():
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if not self._busy and event.mimeData().hasText():
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if self._busy or not event.mimeData().hasText():
            event.ignore()
            return
        cursor = self.cursorForPosition(event.pos())
        cursor.clearSelection()
        self.setTextCursor(cursor)
        text = self._normalise_newlines(event.mimeData().text())
        self._insert_text(text)
        event.setDropAction(Qt.CopyAction)
        event.accept()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        copy_action = QAction("Copy", menu)
        copy_action.setEnabled(self.textCursor().hasSelection())
        copy_action.triggered.connect(self.copy)
        menu.addAction(copy_action)

        cut_action = QAction("Cut", menu)
        cut_action.setEnabled(self._selection_fully_editable())
        cut_action.triggered.connect(self.cut)
        menu.addAction(cut_action)

        paste_action = QAction("Paste", menu)
        paste_action.setEnabled(
            self.canInsertFromMimeData(QApplication.clipboard().mimeData())
        )
        paste_action.triggered.connect(self.paste)
        menu.addAction(paste_action)

        delete_action = QAction("Delete", menu)
        delete_action.setEnabled(self._selection_fully_editable())
        delete_action.triggered.connect(self._delete_selection)
        menu.addAction(delete_action)
        menu.addSeparator()

        undo_action = QAction("Undo", menu)
        undo_action.setEnabled(not self._busy and bool(self._undo_stack))
        undo_action.triggered.connect(self.undo)
        menu.addAction(undo_action)
        redo_action = QAction("Redo", menu)
        redo_action.setEnabled(not self._busy and bool(self._redo_stack))
        redo_action.triggered.connect(self.redo)
        menu.addAction(redo_action)
        menu.addSeparator()

        select_action = QAction("Select All", menu)
        select_action.triggered.connect(self.selectAll)
        menu.addAction(select_action)
        try:
            menu.exec_(event.globalPos())
        finally:
            menu.deleteLater()

    def inputMethodEvent(self, event):
        if self._busy:
            event.accept()
            return
        cursor = self._cursor_for_insertion()
        if cursor is None:
            event.accept()
            return
        self.setTextCursor(cursor)
        before = self._draft_state()
        if not self._ime_active:
            self._ime_before_state = before

        base = cursor.position()
        document_end = self._document_end()
        replace_start = base + int(event.replacementStart())
        replace_end = replace_start + int(event.replacementLength())
        replace_start = max(self._input_start, min(replace_start, document_end))
        replace_end = max(replace_start, min(replace_end, document_end))
        safe = QInputMethodEvent(event.preeditString(), event.attributes())
        safe.setCommitString(
            event.commitString(), replace_start - base, replace_end - replace_start
        )
        super().inputMethodEvent(safe)
        event.accept()

        was_active = self._ime_active
        self._ime_active = bool(event.preeditString())
        if not self._ime_active and (was_active or event.commitString()):
            after = self._draft_state()
            before = self._ime_before_state or before
            if before[0] != after[0]:
                self._undo_stack.append(before)
                self._redo_stack.clear()
            self._ime_before_state = None
            self._last_good_draft = after[0]
            pending, self._pending_messages = self._pending_messages, []
            for who, text in pending:
                self._append_message_gui(who, text)

    # ------------------------------------------------------------------
    # Last-resort invariant guard for accessibility/programmatic edits
    # ------------------------------------------------------------------
    def _protect_prefix(self):
        if self._rendering or self._repairing:
            return
        if self._busy:
            expected = self._transcript + self._BUSY_LINE
            valid = self.toPlainText() == expected
        else:
            expected = self._transcript + self.PROMPT
            valid = (
                self._input_start is not None
                and self._text_range(0, self._input_start) == expected
            )
        if valid:
            if not self._busy:
                self._last_good_draft = self.command_text()
            return
        self._repairing = True
        try:
            draft = self._busy_draft if self._busy else self._last_good_draft
            self._render(draft, caret_to_end=True)
        finally:
            self._repairing = False

