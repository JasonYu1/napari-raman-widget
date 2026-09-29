from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from napari_raman_widget.assistant_history import (
    FORMAT_VERSION,
    HistoryError,
    HistoryStore,
    MAX_PROFILE_BYTES,
    ProfileDeletionError,
)


def _id() -> str:
    return str(uuid.uuid4())


def _turn(index: int, text: str = "answer") -> list[dict[str, str]]:
    return [
        {"role": "user", "content": f"question {index}"},
        {"role": "assistant", "content": text},
    ]


@contextmanager
def _raises(error_type, match: str | None = None):
    try:
        yield
    except error_type as exc:
        if match is not None and re.search(match, str(exc)) is None:
            raise AssertionError(f"{exc!r} does not match {match!r}") from exc
    else:
        raise AssertionError(f"{error_type.__name__} was not raised")


def test_unused_store_performs_no_file_io(tmp_path: Path) -> None:
    root = tmp_path / "not-created"
    store = HistoryStore(root)
    profile_id = _id()

    assert store.read_settings() == {
        "enabled": False,
        "active_profile": None,
        "profiles": [],
    }
    assert store.load_profile(profile_id) == {"messages": [], "transcript": []}
    store.clear_profile(profile_id)
    store.write_settings(False, None, [])

    assert not root.exists()


def test_settings_round_trip_and_strict_validation(tmp_path: Path) -> None:
    root = tmp_path / "history"
    store = HistoryStore(root)
    profile_id = _id()
    profiles = [{"id": profile_id, "name": "Sample A"}]

    store.write_settings(True, profile_id, profiles)

    assert HistoryStore(root).read_settings() == {
        "enabled": True,
        "active_profile": profile_id,
        "profiles": profiles,
    }
    raw = json.loads((root / "settings.json").read_text(encoding="utf-8"))
    assert raw["version"] == FORMAT_VERSION

    bad_calls = [
        (1, profile_id, profiles),
        (True, None, profiles),
        (True, _id(), profiles),
        (True, "../escape", profiles),
        (True, profile_id.upper(), profiles),
        (True, profile_id, [{"id": profile_id, "name": " padded "}]),
        (
            True,
            profile_id,
            [
                {"id": profile_id, "name": "Duplicate"},
                {"id": _id(), "name": "duplicate"},
            ],
        ),
    ]
    for enabled, active, entries in bad_calls:
        with _raises(HistoryError):
            store.write_settings(enabled, active, entries)

    bool_version_root = tmp_path / "bool-version"
    bool_version_root.mkdir()
    (bool_version_root / "settings.json").write_text(
        json.dumps(
            {
                "version": True,
                "enabled": False,
                "active_profile": None,
                "profiles": [],
            }
        ),
        encoding="utf-8",
    )
    with _raises(HistoryError, match="unsupported version"):
        HistoryStore(bool_version_root).read_settings()


def test_profile_normalizes_sdk_blocks_without_replay_data(tmp_path: Path) -> None:
    profile_id = _id()
    store = HistoryStore(tmp_path / "history")
    messages = [
        SimpleNamespace(role="user", content="inspect point"),
        SimpleNamespace(
            role="assistant",
            content=[
                SimpleNamespace(type="text", text="Checking."),
                SimpleNamespace(
                    type="tool_use",
                    id="tool-1",
                    name="inspect_calibration_result",
                    input={"point_index": 2, "flags": (True, False)},
                ),
            ],
        ),
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool-1",
                    "content": [SimpleNamespace(type="text", text="stored result")],
                }
            ],
        },
        SimpleNamespace(
            role="assistant",
            content=[SimpleNamespace(type="text", text="Point 2 is selected.")],
        ),
    ]
    transcript = [
        {"who": "user", "text": "inspect point"},
        {"who": "assistant", "text": "Point 2 is selected."},
    ]

    saved = store.save_profile(profile_id, messages, transcript)
    loaded = HistoryStore(store.root).load_profile(profile_id)

    assert saved == loaded
    assert loaded["transcript"] == transcript
    assert loaded["messages"][1]["content"][1] == {
        "type": "tool_use",
        "id": "tool-1",
        "name": "inspect_calibration_result",
        "input": {"point_index": 2, "flags": [True, False]},
    }
    assert loaded["messages"][2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "tool-1",
        "content": [{"type": "text", "text": "stored result"}],
    }


def test_incomplete_tool_chain_is_dropped_and_limits_keep_newest(
    tmp_path: Path,
) -> None:
    profile_id = _id()
    store = HistoryStore(tmp_path / "history")
    messages = [message for index in range(35) for message in _turn(index)]
    messages.extend(
        [
            {"role": "user", "content": "unfinished"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "pending",
                        "name": "query",
                        "input": {},
                    }
                ],
            },
        ]
    )
    transcript = [
        {"who": "assistant", "text": f"event {index}"} for index in range(510)
    ]

    store.save_profile(profile_id, messages, transcript)
    loaded = store.load_profile(profile_id)

    assert len(loaded["messages"]) == 60
    assert loaded["messages"][0]["content"] == "question 5"
    assert loaded["messages"][-1]["content"] == "answer"
    assert all(message["content"] != "unfinished" for message in loaded["messages"])
    assert len(loaded["transcript"]) == 500
    assert loaded["transcript"][0]["text"] == "question 5"
    assert loaded["transcript"][30]["text"] == "event 40"
    assert {
        event["text"] for event in loaded["transcript"] if event["who"] == "user"
    } == {f"question {index}" for index in range(5, 35)}


def test_profiles_are_isolated_and_clear_only_target(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history")
    first, second = _id(), _id()
    store.save_profile(first, _turn(1), [{"who": "user", "text": "question 1"}])
    store.save_profile(second, _turn(2), [{"who": "user", "text": "question 2"}])

    store.clear_profile(first)

    assert store.load_profile(first) == {"messages": [], "transcript": []}
    assert store.load_profile(second)["transcript"][0]["text"] == "question 2"
    assert not (store.root / "profiles" / f"{first}.json").exists()
    assert (store.root / "profiles" / f"{second}.json").exists()


def test_corruption_is_reported_and_never_overwritten(tmp_path: Path) -> None:
    root = tmp_path / "history"
    profile_id = _id()
    store = HistoryStore(root)
    store.write_settings(True, profile_id, [{"id": profile_id, "name": "Profile"}])
    store.save_profile(profile_id, _turn(1), [])

    manifest = root / "settings.json"
    manifest.write_bytes(b"{broken")
    fresh = HistoryStore(root)
    with _raises(HistoryError):
        fresh.read_settings()
    with _raises(HistoryError):
        fresh.write_settings(False, None, [])
    assert manifest.read_bytes() == b"{broken"

    profile = root / "profiles" / f"{profile_id}.json"
    profile.write_bytes(b"[]")
    fresh = HistoryStore(root)
    with _raises(HistoryError):
        fresh.load_profile(profile_id)
    with _raises(HistoryError):
        fresh.save_profile(profile_id, _turn(2), [])
    with _raises(HistoryError):
        fresh.clear_profile(profile_id)
    assert profile.read_bytes() == b"[]"

    deeply_nested = tmp_path / "deeply-nested"
    deeply_nested.mkdir()
    (deeply_nested / "settings.json").write_text("{}", encoding="utf-8")
    with patch(
        "napari_raman_widget.assistant_history.json.loads",
        side_effect=RecursionError("too deeply nested"),
    ):
        with _raises(HistoryError, match="could not read"):
            HistoryStore(deeply_nested).read_settings()

    invisible_root = tmp_path / "invisible-context"
    invisible_store = HistoryStore(invisible_root)
    invisible_id = _id()
    invisible_store.save_profile(invisible_id, _turn(1), [])
    invisible_path = invisible_root / "profiles" / f"{invisible_id}.json"
    invisible_document = json.loads(invisible_path.read_text(encoding="utf-8"))
    invisible_document["transcript"] = []
    invisible_path.write_text(json.dumps(invisible_document), encoding="utf-8")
    with _raises(HistoryError, match="without a visible user prompt"):
        HistoryStore(invisible_root).load_profile(invisible_id)


def test_loaded_files_use_optimistic_conflict_detection(tmp_path: Path) -> None:
    root = tmp_path / "history"
    profile_id = _id()
    first = HistoryStore(root)
    first.write_settings(True, profile_id, [{"id": profile_id, "name": "One"}])
    first.save_profile(profile_id, _turn(1), [])

    second = HistoryStore(root)
    second.read_settings()
    second.load_profile(profile_id)
    first.write_settings(False, None, [{"id": profile_id, "name": "One"}])
    first.save_profile(profile_id, _turn(2), [])

    with _raises(HistoryError, match="changed outside this session"):
        second.write_settings(True, profile_id, [{"id": profile_id, "name": "One"}])
    with _raises(HistoryError, match="changed outside this session"):
        second.save_profile(profile_id, _turn(3), [])
    with _raises(HistoryError, match="changed outside this session"):
        second.clear_profile(profile_id)

    assert first.load_profile(profile_id)["messages"] == _turn(2)


def test_observed_absence_and_external_deletion_are_conflicts(tmp_path: Path) -> None:
    root = tmp_path / "history"
    profile_id = _id()
    observer = HistoryStore(root)
    assert observer.load_profile(profile_id) == {"messages": [], "transcript": []}

    writer = HistoryStore(root)
    writer.save_profile(profile_id, _turn(1), [])
    with _raises(HistoryError, match="changed outside this session"):
        observer.save_profile(profile_id, _turn(2), [])

    observer = HistoryStore(root)
    observer.load_profile(profile_id)
    writer = HistoryStore(root)
    writer.clear_profile(profile_id)
    with _raises(HistoryError, match="changed outside this session"):
        observer.save_profile(profile_id, _turn(3), [])


def test_invalid_or_unsafe_content_is_rejected_before_io(tmp_path: Path) -> None:
    root = tmp_path / "history"
    store = HistoryStore(root)
    unsafe = [
        "../profile.json",
        "not-a-uuid",
        _id().upper(),
    ]
    for profile_id in unsafe:
        with _raises(HistoryError):
            store.load_profile(profile_id)

    profile_id = _id()
    invalid_messages = [
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool",
                        "name": "query",
                        "input": {"bad": float("nan")},
                    }
                ],
            },
        ],
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [{"type": "image", "source": "not supported"}],
            },
        ],
        [
            {"role": "assistant", "content": "no user message"},
        ],
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool",
                        "name": "query",
                        "input": "not an object",
                    }
                ],
            },
        ],
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool",
                        "name": "query",
                        "input": {},
                    }
                ],
            },
        ],
    ]
    for messages in invalid_messages:
        with _raises(HistoryError):
            store.save_profile(profile_id, messages, [])
    with _raises(HistoryError):
        store.save_profile(
            profile_id,
            _turn(1),
            [{"who": "operator", "text": "unsupported speaker"}],
        )
    assert not root.exists()


def test_profile_file_is_bounded_and_permissions_are_private(tmp_path: Path) -> None:
    profile_id = _id()
    store = HistoryStore(tmp_path / "history")
    large_answer = "x" * 1_200_000
    messages = [message for index in range(8) for message in _turn(index, large_answer)]
    transcript = [
        {"who": "user", "text": f"question {index}"} for index in range(8)
    ] + [
        {"who": "system", "text": f"log {index}:" + "y" * 900_000} for index in range(8)
    ]

    saved = store.save_profile(profile_id, messages, transcript)

    profile = store.root / "profiles" / f"{profile_id}.json"
    assert profile.stat().st_size <= MAX_PROFILE_BYTES
    loaded = store.load_profile(profile_id)
    assert saved == loaded
    assert loaded["messages"][-1]["content"] == large_answer
    assert len(loaded["messages"]) < len(messages)
    visible_prompts = {
        event["text"] for event in loaded["transcript"] if event["who"] == "user"
    }
    retained_prompts = {
        message["content"]
        for message in loaded["messages"]
        if message["role"] == "user"
    }
    assert retained_prompts <= visible_prompts
    assert (
        len([event for event in loaded["transcript"] if event["who"] == "system"]) < 8
    )
    if os.name != "nt":
        assert profile.stat().st_mode & 0o077 == 0
        assert store.root.stat().st_mode & 0o077 == 0


def test_transcript_cap_reserves_visible_prompt_for_every_model_turn(
    tmp_path: Path,
) -> None:
    profile_id = _id()
    store = HistoryStore(tmp_path / "history")
    messages = [message for index in range(30) for message in _turn(index)]
    transcript = [{"who": "system", "text": f"event {index}"} for index in range(500)]

    saved = store.save_profile(profile_id, messages, transcript)

    assert len(saved["transcript"]) == 500
    assert saved["transcript"][0] == {"who": "user", "text": "question 0"}
    visible_prompts = {
        event["text"] for event in saved["transcript"] if event["who"] == "user"
    }
    assert visible_prompts == {f"question {index}" for index in range(30)}


def test_transcript_cap_preserves_real_prompt_order(tmp_path: Path) -> None:
    profile_id = _id()
    store = HistoryStore(tmp_path / "history")
    messages = [message for index in range(30) for message in _turn(index)]
    transcript = []
    for turn_index in range(30):
        transcript.append({"who": "user", "text": f"question {turn_index}"})
        transcript.extend(
            {"who": "assistant", "text": f"turn {turn_index} event {event_index}"}
            for event_index in range(19)
        )

    saved = store.save_profile(profile_id, messages, transcript)

    visible_prompts = [
        event["text"] for event in saved["transcript"] if event["who"] == "user"
    ]
    assert visible_prompts == [f"question {index}" for index in range(30)]
    assert visible_prompts[-1] == saved["messages"][-2]["content"]


def test_delete_nonactive_profile_preserves_active_and_other_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "history"
    first, second = _id(), _id()
    profiles = [
        {"id": first, "name": "First"},
        {"id": second, "name": "Second"},
    ]
    store = HistoryStore(root)
    store.write_settings(True, first, profiles)
    store.save_profile(first, _turn(1), [])
    store.save_profile(second, _turn(2), [])

    updated = store.delete_profile(second)

    assert updated == {
        "enabled": True,
        "active_profile": first,
        "profiles": [profiles[0]],
    }
    assert store.read_settings() == updated
    assert store.load_profile(first)["messages"] == _turn(1)
    assert store.load_profile(second) == {"messages": [], "transcript": []}


def test_delete_active_profile_switches_private_including_last_profile(
    tmp_path: Path,
) -> None:
    root = tmp_path / "history"
    first, second = _id(), _id()
    profiles = [
        {"id": first, "name": "First"},
        {"id": second, "name": "Second"},
    ]
    store = HistoryStore(root)
    store.write_settings(True, first, profiles)
    store.save_profile(first, _turn(1), [])
    store.save_profile(second, _turn(2), [])

    updated = store.delete_profile(first)

    assert updated == {
        "enabled": False,
        "active_profile": None,
        "profiles": [profiles[1]],
    }
    assert store.load_profile(second)["messages"] == _turn(2)

    store.write_settings(True, second, [profiles[1]])
    last_updated = store.delete_profile(second)
    assert last_updated == {
        "enabled": False,
        "active_profile": None,
        "profiles": [],
    }
    assert store.read_settings() == last_updated


def test_delete_profile_with_already_cleared_file_removes_metadata(
    tmp_path: Path,
) -> None:
    profile_id = _id()
    profile = {"id": profile_id, "name": "Empty"}
    store = HistoryStore(tmp_path / "history")
    store.write_settings(True, profile_id, [profile])

    updated = store.delete_profile(profile_id)

    assert updated == {
        "enabled": False,
        "active_profile": None,
        "profiles": [],
    }
    assert store.read_settings() == updated


def test_delete_profile_refuses_stale_manifest_or_profile(tmp_path: Path) -> None:
    manifest_root = tmp_path / "stale-manifest"
    profile_id = _id()
    profiles = [{"id": profile_id, "name": "Profile"}]
    observer = HistoryStore(manifest_root)
    observer.write_settings(True, profile_id, profiles)
    observer.save_profile(profile_id, _turn(1), [])
    observer.read_settings()
    writer = HistoryStore(manifest_root)
    writer.write_settings(False, None, profiles)

    with _raises(HistoryError, match="changed outside this session"):
        observer.delete_profile(profile_id)
    assert (manifest_root / "profiles" / f"{profile_id}.json").exists()
    assert writer.read_settings()["profiles"] == profiles

    profile_root = tmp_path / "stale-profile"
    observer = HistoryStore(profile_root)
    observer.write_settings(True, profile_id, profiles)
    observer.save_profile(profile_id, _turn(1), [])
    observer.load_profile(profile_id)
    writer = HistoryStore(profile_root)
    writer.save_profile(profile_id, _turn(2), [])
    manifest_before = (profile_root / "settings.json").read_bytes()

    with _raises(HistoryError, match="changed outside this session"):
        observer.delete_profile(profile_id)
    assert (profile_root / "settings.json").read_bytes() == manifest_before
    assert writer.load_profile(profile_id)["messages"] == _turn(2)


def test_delete_profile_reports_unlink_and_partial_manifest_failures(
    tmp_path: Path,
) -> None:
    root = tmp_path / "history"
    first, second = _id(), _id()
    profiles = [
        {"id": first, "name": "First"},
        {"id": second, "name": "Second"},
    ]
    store = HistoryStore(root)
    store.write_settings(True, first, profiles)
    store.save_profile(first, _turn(1), [])
    store.save_profile(second, _turn(2), [])
    manifest_before = (root / "settings.json").read_bytes()
    first_path = root / "profiles" / f"{first}.json"
    marker_path = root / "profiles" / f"{first}.deleted"
    original_unlink = Path.unlink

    def fail_profile_unlink(path, *args, **kwargs):
        if path == first_path:
            raise PermissionError("file is locked")
        return original_unlink(path, *args, **kwargs)

    try:
        with patch(
            "napari_raman_widget.assistant_history.Path.unlink",
            autospec=True,
            side_effect=fail_profile_unlink,
        ):
            store.delete_profile(first)
    except ProfileDeletionError as exc:
        raise AssertionError("unlink failure must not be reported as partial") from exc
    except HistoryError:
        pass
    else:
        raise AssertionError("HistoryError was not raised")
    assert first_path.exists()
    assert not marker_path.exists()
    assert (root / "settings.json").read_bytes() == manifest_before

    with patch(
        "napari_raman_widget.assistant_history.Path.unlink",
        side_effect=PermissionError("profile and marker are locked"),
    ):
        try:
            store.delete_profile(first)
        except ProfileDeletionError as exc:
            assert exc.history_cleared is False
            assert exc.profile_id == first
        else:
            raise AssertionError("ProfileDeletionError was not raised")
    assert first_path.exists()
    assert marker_path.exists()
    assert marker_path.stat().st_size == 0
    assert (root / "settings.json").read_bytes() == manifest_before

    with patch.object(
        store, "write_settings", side_effect=HistoryError("settings disk is full")
    ):
        try:
            store.delete_profile(first)
        except ProfileDeletionError as exc:
            assert exc.history_cleared is True
            assert exc.profile_id == first
            assert "profile entry remains" in str(exc)
        else:
            raise AssertionError("ProfileDeletionError was not raised")

    assert not first_path.exists()
    assert marker_path.exists()
    assert (root / "settings.json").read_bytes() == manifest_before
    assert (root / "profiles" / f"{second}.json").exists()

    restarted = HistoryStore(root)
    for operation in (
        lambda: restarted.load_profile(first),
        lambda: restarted.save_profile(first, _turn(3), []),
    ):
        try:
            operation()
        except ProfileDeletionError as exc:
            assert exc.history_cleared is True
            assert exc.profile_id == first
        else:
            raise AssertionError("pending deletion did not block profile access")
    assert not first_path.exists()

    finalized = restarted.delete_profile(first)
    assert finalized == {
        "enabled": False,
        "active_profile": None,
        "profiles": [profiles[1]],
    }
    assert restarted.read_settings() == finalized
    assert not marker_path.exists()
    assert (root / "profiles" / f"{second}.json").exists()


class TestAssistantHistory(unittest.TestCase):
    def _run_case(self, case):
        with tempfile.TemporaryDirectory() as directory:
            case(Path(directory))

    def test_unused_store_performs_no_file_io(self):
        self._run_case(test_unused_store_performs_no_file_io)

    def test_settings_round_trip_and_strict_validation(self):
        self._run_case(test_settings_round_trip_and_strict_validation)

    def test_profile_normalizes_sdk_blocks_without_replay_data(self):
        self._run_case(test_profile_normalizes_sdk_blocks_without_replay_data)

    def test_incomplete_tool_chain_is_dropped_and_limits_keep_newest(self):
        self._run_case(test_incomplete_tool_chain_is_dropped_and_limits_keep_newest)

    def test_profiles_are_isolated_and_clear_only_target(self):
        self._run_case(test_profiles_are_isolated_and_clear_only_target)

    def test_corruption_is_reported_and_never_overwritten(self):
        self._run_case(test_corruption_is_reported_and_never_overwritten)

    def test_loaded_files_use_optimistic_conflict_detection(self):
        self._run_case(test_loaded_files_use_optimistic_conflict_detection)

    def test_observed_absence_and_external_deletion_are_conflicts(self):
        self._run_case(test_observed_absence_and_external_deletion_are_conflicts)

    def test_invalid_or_unsafe_content_is_rejected_before_io(self):
        self._run_case(test_invalid_or_unsafe_content_is_rejected_before_io)

    def test_profile_file_is_bounded_and_permissions_are_private(self):
        self._run_case(test_profile_file_is_bounded_and_permissions_are_private)

    def test_transcript_cap_reserves_visible_prompt_for_every_model_turn(self):
        self._run_case(test_transcript_cap_reserves_visible_prompt_for_every_model_turn)

    def test_transcript_cap_preserves_real_prompt_order(self):
        self._run_case(test_transcript_cap_preserves_real_prompt_order)

    def test_delete_nonactive_profile_preserves_active_and_other_file(self):
        self._run_case(test_delete_nonactive_profile_preserves_active_and_other_file)

    def test_delete_active_profile_switches_private_including_last_profile(self):
        self._run_case(
            test_delete_active_profile_switches_private_including_last_profile
        )

    def test_delete_profile_with_already_cleared_file_removes_metadata(self):
        self._run_case(test_delete_profile_with_already_cleared_file_removes_metadata)

    def test_delete_profile_refuses_stale_manifest_or_profile(self):
        self._run_case(test_delete_profile_refuses_stale_manifest_or_profile)

    def test_delete_profile_reports_unlink_and_partial_manifest_failures(self):
        self._run_case(test_delete_profile_reports_unlink_and_partial_manifest_failures)
