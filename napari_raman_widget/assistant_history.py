"""Local, opt-in persistence for assistant conversations.

This module deliberately has no Qt or assistant-client dependencies.  It stores a
small manifest separately from per-profile histories and accepts either the plain
dictionaries used by the API or SDK objects exposing equivalent attributes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


FORMAT_VERSION = 1
MAX_PROFILES = 50
MAX_PROFILE_NAME_LENGTH = 80
MAX_COMPLETE_TURNS = 30
MAX_TRANSCRIPT_EVENTS = 500
MAX_PROFILE_BYTES = 8 * 1024 * 1024
MAX_TEXT_LENGTH = 2 * 1024 * 1024
MAX_JSON_DEPTH = 32

_MISSING = object()
_MANIFEST_KEYS = {"version", "enabled", "active_profile", "profiles"}
_PROFILE_KEYS = {"version", "profile_id", "messages", "transcript"}
_TRANSCRIPT_KEYS = {"who", "text"}


class HistoryError(RuntimeError):
    """Raised when persisted history is invalid, unsafe, or unreadable."""


class ProfileDeletionError(HistoryError):
    """Profile metadata could not be removed after its saved chat was cleared."""

    history_cleared = True

    def __init__(
        self,
        profile_id: str,
        cause: HistoryError,
        *,
        history_cleared: bool = True,
    ) -> None:
        self.profile_id = profile_id
        self.cause = cause
        self.history_cleared = history_cleared
        if history_cleared:
            message = (
                "saved chat was cleared, but the profile entry remains because "
                f"settings could not be updated: {cause}"
            )
        else:
            message = f"profile deletion is pending but saved chat remains: {cause}"
        super().__init__(message)


class HistoryStore:
    """Versioned, profile-isolated local storage for assistant history.

    Reading an unused store is side-effect free.  Directories and files are only
    created when persistence is explicitly enabled or a profile is saved.
    """

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root) if root is not None else _default_root()
        self._manifest_path = self.root / "settings.json"
        self._profiles_path = self.root / "profiles"
        # A ``None`` fingerprint records that this instance observed no file.
        # That is distinct from a path it has never inspected.
        self._fingerprints: dict[Path, str | None] = {}

    def read_settings(self) -> dict[str, Any]:
        """Return validated settings, or disabled defaults for a new store."""
        self._ensure_safe_parent(self._manifest_path)
        if not self._manifest_path.exists():
            self._check_known_state(self._manifest_path, None)
            self._fingerprints[self._manifest_path] = None
            return _empty_settings()
        raw, fingerprint = self._read_json(self._manifest_path, MAX_PROFILE_BYTES)
        self._check_known_state(self._manifest_path, fingerprint)
        settings = _validate_settings_document(raw)
        self._fingerprints[self._manifest_path] = fingerprint
        return settings

    def write_settings(
        self,
        enabled: bool,
        active_profile: str | None,
        profiles: list[dict[str, str]],
    ) -> None:
        """Atomically write the manifest after strict validation."""
        document = _settings_document(enabled, active_profile, profiles)

        self._ensure_safe_parent(self._manifest_path)
        expected_fingerprint = self._current_fingerprint(self._manifest_path)
        if self._manifest_path.exists():
            # Refuse to replace evidence of corruption with apparently valid data.
            raw, _ = self._read_json(self._manifest_path, MAX_PROFILE_BYTES)
            _validate_settings_document(raw)
        elif document == _settings_document(False, None, []):
            return

        self._atomic_write(
            self._manifest_path, document, expected_fingerprint=expected_fingerprint
        )

    def load_profile(self, profile_id: str) -> dict[str, list[Any]]:
        """Load one profile, or return an empty history when it does not exist."""
        canonical_id = _canonical_profile_id(profile_id)
        path = self._profile_path(canonical_id)
        self._ensure_safe_parent(path)
        self._raise_if_deletion_pending(canonical_id, path)
        if not path.exists():
            self._check_known_state(path, None)
            self._fingerprints[path] = None
            return {"messages": [], "transcript": []}
        return self._read_profile(path, canonical_id)

    def save_profile(
        self,
        profile_id: str,
        messages: Sequence[Any],
        transcript: Sequence[Any],
    ) -> dict[str, list[Any]]:
        """Normalize, bound, and atomically persist a profile."""
        canonical_id = _canonical_profile_id(profile_id)
        path = self._profile_path(canonical_id)
        self._ensure_safe_parent(path)
        self._raise_if_deletion_pending(canonical_id, path)
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
            raise HistoryError("messages must be a sequence")
        if isinstance(transcript, (str, bytes)) or not isinstance(transcript, Sequence):
            raise HistoryError("transcript must be a sequence")

        normalized_messages = [_normalize_message(message) for message in messages]
        turns = _complete_turns(normalized_messages)[-MAX_COMPLETE_TURNS:]
        normalized_transcript = [
            _normalize_transcript_event(event) for event in transcript
        ]

        expected_fingerprint = self._current_fingerprint(path)
        if path.exists():
            # As with the manifest, a corrupt file is never silently overwritten.
            self._read_profile(path, canonical_id, remember=False)

        document = _bounded_profile_document(canonical_id, turns, normalized_transcript)
        self._atomic_write(path, document, expected_fingerprint=expected_fingerprint)
        return {
            "messages": document["messages"],
            "transcript": document["transcript"],
        }

    def clear_profile(self, profile_id: str) -> None:
        """Delete one profile history; settings and other profiles are retained."""
        canonical_id = _canonical_profile_id(profile_id)
        path = self._profile_path(canonical_id)
        self._ensure_safe_parent(path)
        if not path.exists():
            self._check_known_state(path, None)
            self._fingerprints[path] = None
            return
        expected_fingerprint = self._current_fingerprint(path)
        assert expected_fingerprint is not None
        self._read_profile(path, canonical_id, remember=False)
        try:
            if self._file_fingerprint(path, MAX_PROFILE_BYTES) != expected_fingerprint:
                raise HistoryError(
                    f"history file changed outside this session: {path.name}"
                )
            path.unlink()
            self._fingerprints[path] = None
            _fsync_directory(path.parent)
        except OSError as exc:
            raise HistoryError(f"could not clear profile {canonical_id}") from exc

    def delete_profile(self, profile_id: str) -> dict[str, Any]:
        """Clear one saved chat and remove its profile from the manifest.

        The saved chat is cleared first so a later manifest failure never restores
        sensitive conversation data.  Such a partial deletion is reported with
        :class:`ProfileDeletionError`.
        """
        canonical_id = _canonical_profile_id(profile_id)
        settings = self.read_settings()
        if canonical_id not in {profile["id"] for profile in settings["profiles"]}:
            raise HistoryError("profile is not listed in saved history settings")

        remaining_profiles = [
            profile for profile in settings["profiles"] if profile["id"] != canonical_id
        ]
        deleting_active = settings["active_profile"] == canonical_id
        updated_enabled = False if deleting_active else settings["enabled"]
        updated_active = None if deleting_active else settings["active_profile"]
        updated_document = _settings_document(
            updated_enabled, updated_active, remaining_profiles
        )
        updated_settings = {
            "enabled": updated_document["enabled"],
            "active_profile": updated_document["active_profile"],
            "profiles": updated_document["profiles"],
        }

        # Validate the profile and its optimistic fingerprint before leaving a
        # durable marker.  A retry intentionally bypasses load/save marker guards.
        path = self._profile_path(canonical_id)
        self._validate_profile_before_deletion(canonical_id, path)
        marker_created = self._create_deletion_marker(canonical_id)
        try:
            self.clear_profile(canonical_id)
        except HistoryError as exc:
            if marker_created:
                try:
                    self._remove_deletion_marker(canonical_id)
                except HistoryError as rollback_exc:
                    raise ProfileDeletionError(
                        canonical_id,
                        rollback_exc,
                        history_cleared=not path.exists(),
                    ) from exc
                raise
            raise ProfileDeletionError(
                canonical_id, exc, history_cleared=not path.exists()
            ) from exc

        try:
            self.write_settings(
                updated_settings["enabled"],
                updated_settings["active_profile"],
                updated_settings["profiles"],
            )
        except HistoryError as exc:
            raise ProfileDeletionError(canonical_id, exc) from exc
        try:
            self._remove_deletion_marker(canonical_id)
        except HistoryError:
            # Metadata and sensitive chat are gone.  A harmless UUID-only marker
            # may remain, but deleted content must never be restored to fix it.
            pass
        return updated_settings

    def _profile_path(self, profile_id: str) -> Path:
        return self._profiles_path / f"{profile_id}.json"

    def _deletion_marker_path(self, profile_id: str) -> Path:
        return self._profiles_path / f"{profile_id}.deleted"

    def _validate_profile_before_deletion(self, profile_id: str, path: Path) -> None:
        self._ensure_safe_parent(path)
        expected = self._current_fingerprint(path)
        if expected is None:
            return
        self._read_profile(path, profile_id, remember=False)
        if self._file_fingerprint(path, MAX_PROFILE_BYTES) != expected:
            raise HistoryError(
                f"history file changed outside this session: {path.name}"
            )
        self._fingerprints[path] = expected

    def _raise_if_deletion_pending(self, profile_id: str, path: Path) -> None:
        if not self._deletion_marker_exists(profile_id):
            return
        raise ProfileDeletionError(
            profile_id,
            HistoryError("profile deletion is pending; retry Delete profile"),
            history_cleared=not path.exists(),
        )

    def _deletion_marker_exists(self, profile_id: str) -> bool:
        marker = self._deletion_marker_path(profile_id)
        self._ensure_safe_parent(marker)
        if marker.is_symlink():
            raise HistoryError(f"deletion marker is unsafe: {marker.name}")
        if not marker.exists():
            return False
        try:
            if not marker.is_file() or marker.stat().st_size != 0:
                raise HistoryError(f"deletion marker is invalid: {marker.name}")
        except OSError as exc:
            raise HistoryError(
                f"could not inspect deletion marker: {marker.name}"
            ) from exc
        return True

    def _create_deletion_marker(self, profile_id: str) -> bool:
        if self._deletion_marker_exists(profile_id):
            return False
        marker = self._deletion_marker_path(profile_id)
        descriptor = -1
        created = False
        try:
            _mkdir_private(self.root)
            _mkdir_private(self._profiles_path)
            try:
                descriptor = os.open(
                    marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
            except FileExistsError:
                if self._deletion_marker_exists(profile_id):
                    return False
                raise
            created = True
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            _chmod_private(marker, 0o600)
            _fsync_directory(marker.parent)
            return True
        except HistoryError:
            raise
        except OSError as exc:
            if descriptor >= 0:
                os.close(descriptor)
            if created:
                try:
                    marker.unlink(missing_ok=True)
                except OSError:
                    pass
            raise HistoryError("could not create profile deletion marker") from exc

    def _remove_deletion_marker(self, profile_id: str) -> None:
        marker = self._deletion_marker_path(profile_id)
        if not self._deletion_marker_exists(profile_id):
            return
        try:
            marker.unlink()
            _fsync_directory(marker.parent)
        except OSError as exc:
            raise HistoryError("could not remove profile deletion marker") from exc

    def _read_profile(
        self, path: Path, expected_profile_id: str, *, remember: bool = True
    ) -> dict[str, list[Any]]:
        raw, fingerprint = self._read_json(path, MAX_PROFILE_BYTES)
        if not isinstance(raw, dict) or set(raw) != _PROFILE_KEYS:
            raise HistoryError("profile has an invalid schema")
        if type(raw.get("version")) is not int or raw["version"] != FORMAT_VERSION:
            raise HistoryError("profile has an unsupported version")
        if raw.get("profile_id") != expected_profile_id:
            raise HistoryError("profile identity does not match its file")

        messages = raw.get("messages")
        transcript = raw.get("transcript")
        if not isinstance(messages, list) or not isinstance(transcript, list):
            raise HistoryError("profile history must contain lists")
        try:
            normalized_messages = [_normalize_message(item) for item in messages]
            turns = _complete_turns(normalized_messages)
            canonical_messages = [item for turn in turns for item in turn]
            canonical_transcript = [
                _normalize_transcript_event(item) for item in transcript
            ]
        except HistoryError:
            raise
        except Exception as exc:  # defensive boundary around SDK-like input helpers
            raise HistoryError("profile history is invalid") from exc

        if len(turns) > MAX_COMPLETE_TURNS:
            raise HistoryError("profile contains too many turns")
        if len(canonical_transcript) > MAX_TRANSCRIPT_EVENTS:
            raise HistoryError("profile transcript is too long")
        _, missing_prompts = _visible_prompt_indices(turns, canonical_transcript)
        if missing_prompts:
            raise HistoryError(
                "profile has model context without a visible user prompt"
            )
        if canonical_messages != messages or canonical_transcript != transcript:
            raise HistoryError("profile history is not canonical or is incomplete")
        if remember:
            self._fingerprints[path] = fingerprint
        return {
            "messages": canonical_messages,
            "transcript": canonical_transcript,
        }

    @staticmethod
    def _read_json(path: Path, size_limit: int) -> tuple[Any, str]:
        try:
            if path.is_symlink() or not path.is_file():
                raise HistoryError(f"history path is not a regular file: {path.name}")
            with path.open("rb") as stream:
                data = stream.read(size_limit + 1)
            if len(data) > size_limit:
                raise HistoryError(f"history file is too large: {path.name}")
            raw = json.loads(data.decode("utf-8"))
            return raw, hashlib.sha256(data).hexdigest()
        except HistoryError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise HistoryError(f"could not read history file: {path.name}") from exc

    def _atomic_write(
        self,
        path: Path,
        document: dict[str, Any],
        *,
        expected_fingerprint: str | None,
    ) -> None:
        try:
            data = _encode_json(document)
        except (TypeError, ValueError) as exc:
            raise HistoryError("history contains data that JSON cannot store") from exc

        parent = path.parent
        temporary: Path | None = None
        try:
            _mkdir_private(self.root)
            if parent != self.root:
                _mkdir_private(parent)
            if path.is_symlink():
                raise HistoryError(f"refusing to replace symbolic link: {path.name}")

            descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=parent
            )
            temporary = Path(temp_name)
            try:
                _chmod_private(temporary, 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    descriptor = -1
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)

            if expected_fingerprint is None:
                if path.exists():
                    raise HistoryError(
                        f"history file was created outside this session: {path.name}"
                    )
            elif (
                self._file_fingerprint(path, MAX_PROFILE_BYTES) != expected_fingerprint
            ):
                raise HistoryError(
                    f"history file changed outside this session: {path.name}"
                )
            os.replace(temporary, path)
            temporary = None
            _chmod_private(path, 0o600)
            _fsync_directory(parent)
            self._fingerprints[path] = hashlib.sha256(data).hexdigest()
        except HistoryError:
            raise
        except OSError as exc:
            raise HistoryError(f"could not write history file: {path.name}") from exc
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def _current_fingerprint(self, path: Path) -> str | None:
        current = (
            self._file_fingerprint(path, MAX_PROFILE_BYTES) if path.exists() else None
        )
        self._check_known_state(path, current)
        return current

    def _check_known_state(self, path: Path, current: str | None) -> None:
        if path in self._fingerprints and self._fingerprints[path] != current:
            raise HistoryError(
                f"history file changed outside this session: {path.name}"
            )

    def _ensure_safe_parent(self, path: Path) -> None:
        if self.root.exists() and (self.root.is_symlink() or not self.root.is_dir()):
            raise HistoryError(f"history directory is unsafe: {self.root}")
        if path.parent != self.root and path.parent.exists():
            if path.parent.is_symlink() or not path.parent.is_dir():
                raise HistoryError(f"history directory is unsafe: {path.parent}")
        if path.is_symlink():
            raise HistoryError(f"history path is unsafe: {path.name}")

    @staticmethod
    def _file_fingerprint(path: Path, size_limit: int) -> str:
        try:
            if path.is_symlink() or not path.is_file():
                raise HistoryError(f"history path is not a regular file: {path.name}")
            with path.open("rb") as stream:
                data = stream.read(size_limit + 1)
            if len(data) > size_limit:
                raise HistoryError(f"history file is too large: {path.name}")
            return hashlib.sha256(data).hexdigest()
        except HistoryError:
            raise
        except OSError as exc:
            raise HistoryError(f"could not read history file: {path.name}") from exc


def _default_root() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "napari-raman-widget" / "assistant"
        return Path.home() / "AppData" / "Local" / "napari-raman-widget" / "assistant"
    if sys.platform == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "napari-raman-widget"
            / "assistant"
        )
    base = os.environ.get("XDG_DATA_HOME")
    if base:
        return Path(base) / "napari-raman-widget" / "assistant"
    return Path.home() / ".local" / "share" / "napari-raman-widget" / "assistant"


def _empty_settings() -> dict[str, Any]:
    return {"enabled": False, "active_profile": None, "profiles": []}


def _settings_document(
    enabled: bool,
    active_profile: str | None,
    profiles: list[dict[str, str]],
) -> dict[str, Any]:
    if type(enabled) is not bool:
        raise HistoryError("enabled must be a boolean")
    if not isinstance(profiles, list) or len(profiles) > MAX_PROFILES:
        raise HistoryError(f"profiles must be a list of at most {MAX_PROFILES} entries")

    canonical_profiles: list[dict[str, str]] = []
    ids: set[str] = set()
    names: set[str] = set()
    for profile in profiles:
        if not isinstance(profile, dict) or set(profile) != {"id", "name"}:
            raise HistoryError("each profile must contain exactly id and name")
        profile_id = _canonical_profile_id(profile["id"])
        name = _profile_name(profile["name"])
        if profile_id in ids:
            raise HistoryError("profile IDs must be unique")
        if name.casefold() in names:
            raise HistoryError("profile names must be unique")
        ids.add(profile_id)
        names.add(name.casefold())
        canonical_profiles.append({"id": profile_id, "name": name})

    canonical_active = None
    if active_profile is not None:
        canonical_active = _canonical_profile_id(active_profile)
        if canonical_active not in ids:
            raise HistoryError("active_profile must identify a listed profile")
    if not enabled and canonical_active is not None:
        raise HistoryError("disabled history cannot have an active profile")
    if enabled and canonical_active is None:
        raise HistoryError("enabled history must have an active profile")

    return {
        "version": FORMAT_VERSION,
        "enabled": enabled,
        "active_profile": canonical_active,
        "profiles": canonical_profiles,
    }


def _validate_settings_document(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != _MANIFEST_KEYS:
        raise HistoryError("settings have an invalid schema")
    if type(raw.get("version")) is not int or raw["version"] != FORMAT_VERSION:
        raise HistoryError("settings have an unsupported version")
    validated = _settings_document(
        raw.get("enabled"), raw.get("active_profile"), raw.get("profiles")
    )
    return {
        "enabled": validated["enabled"],
        "active_profile": validated["active_profile"],
        "profiles": validated["profiles"],
    }


def _canonical_profile_id(value: Any) -> str:
    if not isinstance(value, str):
        raise HistoryError("profile ID must be a canonical UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise HistoryError("profile ID must be a canonical UUID string") from exc
    canonical = str(parsed)
    if value != canonical:
        raise HistoryError("profile ID must use canonical lowercase UUID form")
    return canonical


def _profile_name(value: Any) -> str:
    if not isinstance(value, str):
        raise HistoryError("profile name must be text")
    if not value or value != value.strip() or len(value) > MAX_PROFILE_NAME_LENGTH:
        raise HistoryError(
            f"profile name must be 1-{MAX_PROFILE_NAME_LENGTH} trimmed characters"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise HistoryError("profile name cannot contain control characters")
    return value


def _field(value: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
    else:
        try:
            return getattr(value, name)
        except AttributeError:
            pass
    if default is _MISSING:
        raise HistoryError(f"history item is missing {name}")
    return default


def _normalize_message(message: Any) -> dict[str, Any]:
    role = _field(message, "role")
    if role not in {"user", "assistant"}:
        raise HistoryError("message role must be user or assistant")
    content = _field(message, "content")
    if isinstance(content, str):
        normalized_content: str | list[dict[str, Any]] = _text(content, "message text")
    elif isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        normalized_content = [_normalize_block(block) for block in content]
        if not normalized_content:
            raise HistoryError("message block content cannot be empty")
    else:
        raise HistoryError("message content must be text or a sequence of blocks")
    if isinstance(normalized_content, list):
        block_types = {block["type"] for block in normalized_content}
        if role == "assistant" and not block_types <= {"text", "tool_use"}:
            raise HistoryError("assistant messages support only text and tool_use")
        if role == "user" and not (
            block_types == {"text"} or block_types == {"tool_result"}
        ):
            raise HistoryError(
                "user block messages must contain only text or only tool_result"
            )
    return {"role": role, "content": normalized_content}


def _normalize_block(block: Any) -> dict[str, Any]:
    block_type = _field(block, "type")
    if block_type == "text":
        return {"type": "text", "text": _text(_field(block, "text"), "block text")}
    if block_type == "tool_use":
        tool_id = _bounded_identifier(_field(block, "id"), "tool_use id")
        name = _bounded_identifier(_field(block, "name"), "tool name")
        raw_payload = _field(block, "input")
        if not isinstance(raw_payload, Mapping):
            raise HistoryError("tool input must be an object")
        payload = _json_value(raw_payload, "tool input")
        return {"type": "tool_use", "id": tool_id, "name": name, "input": payload}
    if block_type == "tool_result":
        tool_id = _bounded_identifier(
            _field(block, "tool_use_id"), "tool_result tool_use_id"
        )
        content = _field(block, "content")
        if isinstance(content, str):
            normalized_result: str | list[dict[str, str]] = _text(
                content, "tool result text"
            )
        elif isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            normalized_result = []
            for item in content:
                if _field(item, "type") != "text":
                    raise HistoryError("only text blocks are supported in tool results")
                normalized_result.append(
                    {
                        "type": "text",
                        "text": _text(_field(item, "text"), "tool result text"),
                    }
                )
            if not normalized_result:
                raise HistoryError("tool result content cannot be empty")
        else:
            raise HistoryError("tool result content must be text or text blocks")
        normalized: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": normalized_result,
        }
        is_error = _field(block, "is_error", False)
        if type(is_error) is not bool:
            raise HistoryError("tool result is_error must be a boolean")
        if is_error:
            normalized["is_error"] = True
        return normalized
    raise HistoryError(f"unsupported assistant content block: {block_type!r}")


def _json_value(value: Any, label: str, depth: int = 0) -> Any:
    if depth > MAX_JSON_DEPTH:
        raise HistoryError(f"{label} is nested too deeply")
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is str:
        return _text(value, label)
    if type(value) is float:
        if not math.isfinite(value):
            raise HistoryError(f"{label} cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise HistoryError(f"{label} object keys must be text")
            result[key] = _json_value(item, label, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, label, depth + 1) for item in value]
    raise HistoryError(f"{label} contains an unsupported value")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise HistoryError(f"{label} must be text")
    if len(value) > MAX_TEXT_LENGTH:
        raise HistoryError(f"{label} is too long")
    return value


def _bounded_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise HistoryError(f"{label} must be 1-256 characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise HistoryError(f"{label} cannot contain control characters")
    return value


def _normalize_transcript_event(event: Any) -> dict[str, str]:
    if not isinstance(event, Mapping) or set(event) != _TRANSCRIPT_KEYS:
        raise HistoryError("transcript event must contain exactly who and text")
    who = _bounded_identifier(event["who"], "transcript speaker")
    if who not in {"user", "assistant", "tool", "system"}:
        raise HistoryError("transcript speaker is unsupported")
    text = _text(event["text"], "transcript text")
    return {"who": who, "text": text}


def _complete_turns(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] | None = None
    for message in messages:
        if _starts_user_turn(message):
            if current is not None:
                status = _turn_status(current)
                if status == "invalid":
                    raise HistoryError("conversation has invalid tool message ordering")
                if status == "complete":
                    turns.append(current)
            current = [message]
        elif current is not None:
            current.append(message)
        else:
            raise HistoryError("conversation must begin with an ordinary user message")
    if current is not None:
        status = _turn_status(current)
        if status == "invalid":
            raise HistoryError("conversation has invalid tool message ordering")
        if status == "complete":
            turns.append(current)
    return turns


def _starts_user_turn(message: dict[str, Any]) -> bool:
    if message["role"] != "user":
        return False
    content = message["content"]
    return not (
        isinstance(content, list)
        and content
        and all(block["type"] == "tool_result" for block in content)
    )


def _turn_status(turn: list[dict[str, Any]]) -> str:
    """Return complete/incomplete/invalid for one ordinary user turn."""
    if not turn or not _starts_user_turn(turn[0]):
        return "invalid"

    expected = "assistant"
    seen_tool_ids: set[str] = set()
    pending: set[str] = set()
    for index, message in enumerate(turn[1:], start=1):
        content = message["content"]
        blocks = content if isinstance(content, list) else []
        if expected == "assistant":
            if message["role"] != "assistant":
                return "invalid"
            tool_ids = [block["id"] for block in blocks if block["type"] == "tool_use"]
            if len(tool_ids) != len(set(tool_ids)) or any(
                tool_id in seen_tool_ids for tool_id in tool_ids
            ):
                return "invalid"
            seen_tool_ids.update(tool_ids)
            if tool_ids:
                pending = set(tool_ids)
                expected = "tool_result"
            else:
                return "complete" if index == len(turn) - 1 else "invalid"
        else:
            if message["role"] != "user" or not blocks:
                return "invalid"
            result_ids = [block["tool_use_id"] for block in blocks]
            if len(result_ids) != len(set(result_ids)) or set(result_ids) != pending:
                return "invalid"
            pending.clear()
            expected = "assistant"
    return "incomplete"


def _bounded_profile_document(
    profile_id: str,
    turns: list[list[dict[str, Any]]],
    transcript: list[dict[str, str]],
) -> dict[str, Any]:
    kept_turns = list(turns)
    kept_transcript = _ensure_visible_user_prompts(kept_turns, transcript)
    while True:
        document = {
            "version": FORMAT_VERSION,
            "profile_id": profile_id,
            "messages": [message for turn in kept_turns for message in turn],
            "transcript": kept_transcript,
        }
        encoded = _encode_json(document)
        if len(encoded) <= MAX_PROFILE_BYTES:
            return document
        if len(kept_turns) > 1:
            # Never retain invisible model context merely to preserve its size.
            kept_turns.pop(0)
            continue

        protected, missing = _visible_prompt_indices(kept_turns, kept_transcript)
        if missing:
            raise HistoryError("saved model context has no visible user prompt")
        unprotected = [
            index for index in range(len(kept_transcript)) if index not in protected
        ]
        if unprotected:
            bytes_to_remove = len(encoded) - MAX_PROFILE_BYTES
            removed: set[int] = set()
            removed_bytes = 0
            for index in unprotected:
                removed.add(index)
                removed_bytes += len(_encode_json(kept_transcript[index])) + 1
                if removed_bytes >= bytes_to_remove:
                    break
            kept_transcript = [
                event
                for index, event in enumerate(kept_transcript)
                if index not in removed
            ]
        else:
            raise HistoryError("latest visible conversation turn exceeds storage limit")


def _ensure_visible_user_prompts(
    turns: list[list[dict[str, Any]]], transcript: list[dict[str, str]]
) -> list[dict[str, str]]:
    protected, missing = _visible_prompt_indices(turns, transcript)
    restored_prefix: list[dict[str, str]] = []
    if missing:
        restored_prefix = [
            {"who": "user", "text": _turn_prompt_text(turns[turn_index])}
            for turn_index in missing
        ]
    remaining_slots = MAX_TRANSCRIPT_EVENTS - len(restored_prefix) - len(protected)
    if remaining_slots < 0:
        raise HistoryError("too many required user prompts for transcript limit")
    unprotected = [index for index in range(len(transcript)) if index not in protected]
    newest_unprotected = unprotected[-remaining_slots:] if remaining_slots else []
    selected = protected | set(newest_unprotected)
    visible = restored_prefix + [
        event for index, event in enumerate(transcript) if index in selected
    ]

    _, still_missing = _visible_prompt_indices(turns, visible)
    if still_missing:
        raise HistoryError("could not preserve visible user prompts")
    return visible


def _visible_prompt_indices(
    turns: list[list[dict[str, Any]]], transcript: list[dict[str, str]]
) -> tuple[set[int], list[int]]:
    protected: set[int] = set()
    missing: list[int] = []
    prompts = [_turn_prompt_text(turn) for turn in turns]
    for turn_index in range(len(prompts) - 1, -1, -1):
        prompt = prompts[turn_index]
        match = next(
            (
                index
                for index in range(len(transcript) - 1, -1, -1)
                if index not in protected
                and transcript[index]["who"] == "user"
                and transcript[index]["text"] == prompt
            ),
            None,
        )
        if match is None:
            missing.append(turn_index)
        else:
            protected.add(match)
    missing.reverse()
    return protected, missing


def _turn_prompt_text(turn: list[dict[str, Any]]) -> str:
    content = turn[0]["content"]
    if isinstance(content, str):
        return content
    return "\n".join(block["text"] for block in content)


def _encode_json(document: Any) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _mkdir_private(path: Path) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise HistoryError(f"history directory is unsafe: {path}")
    else:
        path.mkdir(parents=True, mode=0o700, exist_ok=False)
    _chmod_private(path, 0o700)


def _chmod_private(path: Path, mode: int) -> None:
    try:
        path.chmod(mode)
    except OSError:
        # Windows and unusual filesystems may not support POSIX permission modes.
        pass


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


__all__ = [
    "FORMAT_VERSION",
    "HistoryError",
    "HistoryStore",
    "MAX_COMPLETE_TURNS",
    "MAX_PROFILE_BYTES",
    "MAX_TRANSCRIPT_EVENTS",
    "ProfileDeletionError",
]
