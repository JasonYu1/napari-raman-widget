"""Opt-in local chat profiles, separate from the assistant's executable tools."""

from __future__ import annotations

import uuid

from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QInputDialog, QMenu, QMessageBox,
    QSizePolicy, QToolButton, QWidget,
)

from .assistant_history import (
    MAX_PROFILES, HistoryError, HistoryStore, ProfileDeletionError,
)


class AssistantHistoryControls(QWidget):
    """Main-thread history controls; never dispatch saved tool calls."""

    def __init__(self, panel, store=None):
        super().__init__(panel)
        self.panel = panel
        self.store = store if store is not None else HistoryStore()
        self._entries = []
        self._profiles = []
        self._active = None
        self._enabled = False
        self._busy = False
        self._unavailable = False
        self._deletion_pending = False

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.save_check = QCheckBox("Save history", self)
        self.save_check.setToolTip(
            "Opt in to local, unencrypted chat history. Saved context is sent "
            "to Anthropic when you send another request. Uncheck for a new "
            "private session; existing saved profiles are kept."
        )
        self.profile_combo = QComboBox(self)
        self.profile_combo.setAccessibleName("Assistant history profile")
        self.profile_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.profile_combo.setMinimumContentsLength(8)
        self.profile_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.profile_combo.setToolTip(
            "Each profile has separate chat context. Profiles are not "
            "password-protected accounts."
        )
        self.menu_button = QToolButton(self)
        self.menu_button.setText("History")
        self.menu_button.setToolTip(f"Manage local chat history in {self.store.root}")
        self.menu_button.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.menu_button)
        self.new_action = menu.addAction("New profile…", self._new_profile)
        self.clear_action = menu.addAction("Clear current history…", self._clear_history)
        menu.addSeparator()
        self.delete_action = menu.addAction("Delete profile…", self._delete_profile)
        self.menu_button.setMenu(menu)
        row.addWidget(self.save_check)
        row.addWidget(self.profile_combo, 1)
        row.addWidget(self.menu_button)
        self.save_check.toggled.connect(self._toggle_saving)
        self.profile_combo.activated.connect(self._select_profile)

        try:
            settings = self.store.read_settings()
            self._profiles = settings["profiles"]
            if settings["enabled"]:
                payload = self.store.load_profile(settings["active_profile"])
                self._replace(payload)
                self._active = settings["active_profile"]
                self._enabled = True
                self._notice("Restored saved chat. Hardware and plot state were not restored.")
        except ProfileDeletionError as exc:
            self._active = exc.profile_id
            self._pending_deletion(exc)
        except (HistoryError, OSError) as exc:
            self._fail(exc)
        self._refresh()

    def record(self, who, text):
        self._entries.append({"who": who, "text": text})

    def set_busy(self, busy):
        self._busy = bool(busy)
        self._refresh()
        if not busy:
            self._save_current()

    def _refresh(self):
        self.save_check.blockSignals(True)
        self.save_check.setChecked(self._enabled)
        self.save_check.blockSignals(False)
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItem("Private session", None)
        for profile in self._profiles:
            self.profile_combo.addItem(profile["name"], profile["id"])
        self.profile_combo.setCurrentIndex(max(0, self.profile_combo.findData(self._active)))
        self.profile_combo.blockSignals(False)
        available = not self._busy and not self._unavailable
        self.save_check.setEnabled(available)
        self.profile_combo.setEnabled(available)
        self.new_action.setEnabled(available)
        self.delete_action.setEnabled(
            not self._busy and self._active is not None
            and (available or self._deletion_pending)
        )
        self.clear_action.setEnabled(not self._busy)
        self.menu_button.setEnabled(not self._busy)

    def _notice(self, text):
        self.panel.console.append_message("system", text)

    def _fail(self, exc):
        # Do not overwrite unreadable history or silently continue saving.
        self._enabled = False
        self._unavailable = True
        self._notice(f"Saved history unavailable: {exc}. Chat can continue in memory; "
                     "existing files have not been cleared. Restart after resolving the problem.")
        self._refresh()

    def _consent(self):
        return QMessageBox.question(
            self, "Save assistant history?",
            "Chat text and tool inputs/results will be stored locally, unencrypted. "
            "Anyone with access to this computer account may read them. Profiles "
            "separate conversations, but are not secure customer accounts.\n\n"
            "Saved context is sent to the configured Anthropic API when you send "
            "another request. Only recent history is retained. Do not include secrets "
            "or sensitive customer data.\n\nEnable saved history?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) == QMessageBox.Yes

    def _write_settings(self, enabled, active, profiles=None):
        self.store.write_settings(enabled, active, self._profiles if profiles is None else profiles)

    def _save_current(self):
        if not self._enabled or self._busy or self._unavailable:
            return
        try:
            payload = self.store.save_profile(self._active, self.panel._messages, self._entries)
            self.panel._messages = payload["messages"]
            self._entries = payload["transcript"]
        except ProfileDeletionError as exc:
            self._pending_deletion(exc)
        except (HistoryError, OSError) as exc:
            self._fail(exc)

    def _replace(self, payload):
        self.panel.console.replace_conversation(payload["transcript"])
        self.panel._messages = payload["messages"]
        self._entries = payload["transcript"]

    def _discard_private_ok(self):
        if not self._entries and not self.panel.console.command_text():
            return True
        if self._enabled and not self.panel.console.command_text():
            return True
        return QMessageBox.question(
            self, "Switch conversation?",
            "Switching discards the current draft and any unsaved private conversation. Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) == QMessageBox.Yes

    def _toggle_saving(self, enabled):
        if self._busy or self._unavailable:
            self._refresh()
            return
        if enabled:
            if self._consent():
                # Never merge a private session into an existing customer's profile.
                self._create_profile(self._unique_name(), keep_current=True)
        else:
            self._switch_private()
        self._refresh()

    def _unique_name(self):
        names = {profile["name"].casefold() for profile in self._profiles}
        index = 1
        while (name := f"Profile {index}").casefold() in names:
            index += 1
        return name

    def _create_profile(self, name, *, keep_current=False):
        if len(self._profiles) >= MAX_PROFILES:
            QMessageBox.warning(self, "Profile limit reached",
                                f"At most {MAX_PROFILES} profiles are supported. "
                                "Delete an unused profile, or clear an existing profile's history to reuse it.")
            return
        profile = {"id": str(uuid.uuid4()), "name": name}
        profiles = self._profiles + [profile]
        written = False
        try:
            payload = self.store.save_profile(
                profile["id"], self.panel._messages if keep_current else [],
                self._entries if keep_current else [],
            )
            written = True
            self._write_settings(True, profile["id"], profiles)
        except (HistoryError, OSError) as exc:
            if written:
                try:
                    # The manifest never registered this newly generated UUID.
                    # Remove only its just-created history, not any old profile.
                    self.store.clear_profile(profile["id"])
                except (HistoryError, OSError):
                    self._notice(f"A new unregistered history file may remain under {self.store.root} "
                                 f"(profile {profile['id']}).")
            self._fail(exc)
            return
        self._profiles = profiles
        self._active = profile["id"]
        self._enabled = True
        if not keep_current:
            self._replace({"messages": [], "transcript": []})
        else:
            self.panel._messages = payload["messages"]
            self._entries = payload["transcript"]
        self._notice(f"Saving recent chat in local profile: {name}.")
        self._refresh()

    def _new_profile(self):
        if self._busy or self._unavailable or not self._discard_private_ok():
            return
        if not self._enabled and not self._consent():
            return
        name, ok = QInputDialog.getText(self, "New local profile", "Profile name:")
        if not ok:
            return
        name = name.strip()
        if not name or len(name) > 80 or any(ord(char) < 32 or ord(char) == 127 for char in name):
            QMessageBox.warning(self, "Invalid profile name", "Use a name of 1–80 characters without control characters.")
            return
        if name.casefold() in {profile["name"].casefold() for profile in self._profiles}:
            QMessageBox.warning(self, "Profile already exists", "Choose another name, or select the existing profile.")
            return
        self._create_profile(name)

    def _switch_private(self):
        if not self._discard_private_ok():
            return
        try:
            self._write_settings(False, None)
        except (HistoryError, OSError) as exc:
            self._fail(exc)
            return
        self._enabled = False
        self._active = None
        self._replace({"messages": [], "transcript": []})
        self._notice("Private session: new chat is not saved. Existing saved profiles are kept.")

    def _select_profile(self, index):
        if self._busy or self._unavailable:
            self._refresh()
            return
        target = self.profile_combo.itemData(index)
        if target == self._active:
            return
        if target is None:
            self._switch_private()
        elif self._discard_private_ok() and (self._enabled or self._consent()):
            try:
                payload = self.store.load_profile(target)
                self._write_settings(True, target)
                self._replace(payload)
                self._active = target
                self._enabled = True
                self._notice("Restored saved chat. Hardware and plot state were not restored.")
            except ProfileDeletionError as exc:
                self._active = target
                self._pending_deletion(exc)
            except (HistoryError, OSError) as exc:
                self._fail(exc)
        self._refresh()

    def _clear_history(self):
        if self._busy:
            return
        saved = self._active is not None
        message = "Clear the current conversation, draft, and command recall history?"
        if saved:
            message += "\n\nThis also deletes this profile's saved chat from this computer. Other profiles are kept."
        message += "\n\nThis cannot delete data already sent to the API provider or copies in backups."
        if QMessageBox.question(self, "Clear current history?", message,
                               QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            if saved:
                self.store.clear_profile(self._active)
        except (HistoryError, OSError) as exc:
            self._fail(exc)
            return
        self._replace({"messages": [], "transcript": []})
        self._notice("History cleared. Future messages will be saved." if self._enabled
                     else "History cleared. This session is not saved.")

    def _delete_profile(self):
        if self._busy or self._active is None or (self._unavailable and not self._deletion_pending):
            return
        profile = next((item for item in self._profiles if item["id"] == self._active), None)
        if profile is None:
            return
        name = profile["name"]
        message = (
            f'Delete local profile "{name}"?\n\n'
            "This permanently removes its saved chat and its name from the list, "
            "and clears this conversation, draft, and command recall. Other profiles "
            "are unchanged.\n\nYou will return to Private session. This cannot be "
            "undone here and does not remove API-provider records or backups."
        )
        if QMessageBox.question(self, "Delete profile?", message,
                               QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            settings = self.store.delete_profile(profile["id"])
        except ProfileDeletionError as exc:
            self._pending_deletion(exc)
            return
        except (HistoryError, OSError) as exc:
            self._fail(exc)
            return
        self._profiles = settings["profiles"]
        self._active = None
        self._enabled = False
        self._unavailable = False
        self._deletion_pending = False
        self._replace({"messages": [], "transcript": []})
        self._notice(f'Profile "{name}" deleted. Private session: new chat is not saved.')
        self._refresh()

    def _pending_deletion(self, exc):
        # A durable marker also blocks restoration/saving after a restart.
        # Forget the current copy and allow retry without reloading its chat.
        self._replace({"messages": [], "transcript": []})
        self._enabled = False
        self._unavailable = True
        self._deletion_pending = True
        self._notice(f"Profile deletion incomplete: {exc}. "
                     "No saved chat is loaded, and saving is blocked. "
                     "Use History > Delete profile again to finish removal.")
        self._refresh()

