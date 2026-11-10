"""Persistent fit preferences survive restarts without importing RACER."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from napari_raman_widget import racer_preferences


class RacerPreferencesTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "preferences" / "racer.json"

    def test_missing_file_uses_defaults_without_creating_anything(self):
        self.assertEqual(racer_preferences.load_calibration_options(self.path), {})
        self.assertFalse(self.path.parent.exists())

    def test_roundtrip_preserves_types_and_small_values_without_mutating_input(self):
        options = {
            "image": {"trace_half_width_rows": 64, "minimum_row_probability": 1e-11},
            "spectrum_dark": {"temporal_iterations": 3},
            "spectrum_reference": {"minimum_sigma_adu": 0.75},
        }
        original = copy.deepcopy(options)
        saved_path = racer_preferences.save_calibration_options(options, self.path)
        restored = racer_preferences.load_calibration_options(self.path)
        self.assertEqual(saved_path, self.path.resolve())
        self.assertEqual(restored, original)
        self.assertEqual(options, original)
        self.assertIsInstance(restored["image"]["trace_half_width_rows"], int)
        self.assertIsInstance(restored["image"]["minimum_row_probability"], float)
        restored["image"]["trace_half_width_rows"] = 3
        self.assertEqual(racer_preferences.load_calibration_options(self.path), original)

    def test_saving_empty_options_remembers_restore_defaults(self):
        racer_preferences.save_calibration_options(
            {"image": {"trace_half_width_rows": 64}}, self.path
        )
        racer_preferences.save_calibration_options({}, self.path)
        self.assertEqual(racer_preferences.load_calibration_options(self.path), {})
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), {})

    def test_load_rejects_corruption_without_rewriting_it(self):
        self.path.parent.mkdir()
        for contents in (
            "{broken", "[]", "null", '{"unknown": {}}',
            '{"image": []}', '{"image": {"x": true}}',
            '{"image": {"x": "64"}}', '{"image": {"x": NaN}}',
            '{"image": {"x": Infinity}}', '{"image": {"": 1}}',
        ):
            with self.subTest(contents=contents):
                self.path.write_text(contents, encoding="utf-8")
                with self.assertRaises(ValueError):
                    racer_preferences.load_calibration_options(self.path)
                self.assertEqual(self.path.read_text(encoding="utf-8"), contents)

    def test_invalid_save_preserves_previous_settings(self):
        racer_preferences.save_calibration_options(
            {"image": {"trace_half_width_rows": 64}}, self.path
        )
        previous = self.path.read_bytes()
        for options in (
            None, [], {"unknown": {}}, {"image": []},
            {"image": {"x": True}}, {"image": {"x": float("nan")}},
            {"image": {"x": float("inf")}}, {"image": {"x": "64"}},
            {"image": {3: 1}}, {"image": {"  ": 1}},
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    racer_preferences.save_calibration_options(options, self.path)
                self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_replace_failure_preserves_original_and_removes_temporary_file(self):
        racer_preferences.save_calibration_options(
            {"image": {"trace_half_width_rows": 64}}, self.path
        )
        original = self.path.read_bytes()
        with patch.object(racer_preferences.os, "replace", side_effect=OSError("locked")):
            with self.assertRaisesRegex(OSError, "locked"):
                racer_preferences.save_calibration_options({}, self.path)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_structural_validation_defers_library_specific_parameter_checks(self):
        options = {"image": {"removed_in_a_future_version": 2}}
        with patch(
            "napari_raman_widget.cosmic_ray_removal._import_racer",
            side_effect=AssertionError("preferences must not import RACER"),
        ):
            racer_preferences.save_calibration_options(options, self.path)
            self.assertEqual(racer_preferences.load_calibration_options(self.path), options)

    def test_omitted_path_uses_shared_app_preference_location(self):
        with patch.object(racer_preferences, "calibration_settings_path", return_value=self.path):
            racer_preferences.save_calibration_options({"image": {"trace_half_width_rows": 64}})
            self.assertEqual(
                racer_preferences.load_calibration_options(),
                {"image": {"trace_half_width_rows": 64}},
            )

    def test_platform_paths_use_user_data_directories_independently_of_cwd(self):
        filename = "racer-calibration-settings.json"
        home = self.root / "home"
        local = self.root / "local"
        xdg = self.root / "xdg"
        cases = (
            ("win32", {"LOCALAPPDATA": str(local)}, local),
            ("win32", {}, home / "AppData" / "Local"),
            ("darwin", {}, home / "Library" / "Application Support"),
            ("linux", {"XDG_DATA_HOME": str(xdg)}, xdg),
            ("linux", {}, home / ".local" / "share"),
        )
        for platform, environment, base in cases:
            with self.subTest(platform=platform, environment=environment):
                with patch.object(racer_preferences.sys, "platform", platform), patch.dict(
                    os.environ, environment, clear=True
                ), patch.object(Path, "home", return_value=home), patch.object(
                    Path, "cwd", side_effect=AssertionError("must not depend on cwd")
                ):
                    self.assertEqual(
                        racer_preferences.calibration_settings_path(),
                        base / "napari-raman-widget" / filename,
                    )

    def test_remembered_files_are_separate_from_fit_preferences(self):
        options = {"image": {"trace_half_width_rows": 64}}
        with patch.object(
            racer_preferences, "calibration_settings_path", return_value=self.path
        ):
            expected_paths_file = self.path.with_name("racer-calibration-files.json")
            self.assertEqual(racer_preferences.calibration_paths_path(), expected_paths_file)
            racer_preferences.save_calibration_options(options)
            racer_preferences.save_calibration_paths(self.root / "dark.zarr", None)
            self.assertEqual(racer_preferences.load_calibration_options(), options)
            self.assertEqual(racer_preferences.load_calibration_paths(), {
                "closed_shutter": str(self.root / "dark.zarr"), "open_reference": None,
            })
            racer_preferences.save_calibration_options({})
            self.assertEqual(racer_preferences.load_calibration_paths()["closed_shutter"],
                             str(self.root / "dark.zarr"))
            self.assertEqual(set(self.path.parent.iterdir()), {self.path, expected_paths_file})

    def test_missing_remembered_files_document_creates_nothing(self):
        self.assertEqual(racer_preferences.load_calibration_paths(self.path), {})
        self.assertFalse(self.path.parent.exists())

    def test_file_paths_roundtrip_including_dark_only_and_partial_selections(self):
        dark = self.root / "closed shutter.zarr"
        reference = self.root / "open reference.zarr"
        for closed, opened in (
            (dark, reference), (dark, None), (None, reference), (None, None)
        ):
            with self.subTest(closed=closed, opened=opened):
                destination = racer_preferences.save_calibration_paths(closed, opened, self.path)
                self.assertEqual(destination, self.path.resolve())
                self.assertEqual(racer_preferences.load_calibration_paths(self.path), {
                    "closed_shutter": str(closed) if closed else None,
                    "open_reference": str(opened) if opened else None,
                })
        self.assertFalse(dark.exists())
        self.assertFalse(reference.exists())

    def test_relative_selections_are_saved_as_absolute_paths(self):
        selected = Path("relative_dark.zarr")
        expected = str(selected.resolve())
        racer_preferences.save_calibration_paths(selected, None, self.path)
        self.assertEqual(racer_preferences.load_calibration_paths(self.path)["closed_shutter"],
                         expected)

    def test_remembered_files_do_not_import_racer_or_open_zarr_stores(self):
        dark = self.root / "offline.zarr"
        reference = self.root / "moved.zarr"
        original_open = Path.open
        opened = []

        def tracked_open(path, *args, **kwargs):
            opened.append(path)
            if path in (dark, reference):
                raise AssertionError("must not open a selected store")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", tracked_open), patch.object(
            Path, "is_dir", side_effect=AssertionError("must not check a selected store")
        ), patch(
            "napari_raman_widget.cosmic_ray_removal._import_racer",
            side_effect=AssertionError("must not import RACER"),
        ):
            racer_preferences.save_calibration_paths(dark, reference, self.path)
            restored = racer_preferences.load_calibration_paths(self.path)
        self.assertEqual(opened, [self.path])
        self.assertEqual(restored["closed_shutter"], str(dark))

    def test_invalid_remembered_files_fail_without_rewriting_document(self):
        self.path.parent.mkdir()
        for contents in (
            "{broken", "[]", "null", "{}", '{"closed_shutter": null}',
            '{"closed_shutter": null, "open_reference": null, "other": null}',
            '{"closed_shutter": [], "open_reference": null}',
            '{"closed_shutter": true, "open_reference": null}',
            '{"closed_shutter": "", "open_reference": null}',
            '{"closed_shutter": "relative.zarr", "open_reference": null}',
            '{"closed_shutter": null, "open_reference": 4}',
        ):
            with self.subTest(contents=contents):
                self.path.write_text(contents, encoding="utf-8")
                with self.assertRaises(ValueError):
                    racer_preferences.load_calibration_paths(self.path)
                self.assertEqual(self.path.read_text(encoding="utf-8"), contents)

    def test_invalid_path_save_preserves_previous_selection(self):
        racer_preferences.save_calibration_paths(self.root / "dark.zarr", None, self.path)
        original = self.path.read_bytes()
        for value in ([], {}, 3, True, "", "   ", "bad\0path"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    racer_preferences.save_calibration_paths(value, None, self.path)
                with self.assertRaises(ValueError):
                    racer_preferences.save_calibration_paths(None, value, self.path)
                self.assertEqual(self.path.read_bytes(), original)

    def test_failed_path_replace_preserves_previous_selection(self):
        racer_preferences.save_calibration_paths(self.root / "dark.zarr", None, self.path)
        original = self.path.read_bytes()
        with patch.object(racer_preferences.os, "replace", side_effect=OSError("locked")):
            with self.assertRaisesRegex(OSError, "locked"):
                racer_preferences.save_calibration_paths(None, None, self.path)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])


if __name__ == "__main__":
    unittest.main()
