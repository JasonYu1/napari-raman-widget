"""Tests for machine-local HardwareWidget defaults."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from napari_raman_widget.hardware_defaults import (
    DEFAULTS_ENV_VAR,
    load_hardware_defaults,
    resolve_defaults_path,
)


class HardwareDefaultsTests(unittest.TestCase):
    def test_loads_explicit_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "defaults.json"
            path.write_text(
                json.dumps(
                    {
                        "micro_manager_config": "scope.cfg",
                        "selection_center_x": 540,
                    }
                ),
                encoding="utf-8",
            )
            loaded_path, values = load_hardware_defaults(path)

        self.assertEqual(loaded_path, path.resolve())
        self.assertEqual(values["micro_manager_config"], "scope.cfg")
        self.assertEqual(values["selection_center_x"], 540)

    def test_environment_file_has_discovery_priority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "machine.json"
            path.write_text(
                json.dumps({"transformer_model": "aiming.json"}),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {DEFAULTS_ENV_VAR: str(path)}):
                loaded_path, values = load_hardware_defaults()

        self.assertEqual(loaded_path, path.resolve())
        self.assertEqual(values["transformer_model"], "aiming.json")

    def test_relative_paths_use_defaults_file_directory(self) -> None:
        defaults_file = Path("C:/settings/napari_raman_defaults.json")
        resolved = resolve_defaults_path("models/aiming.json", defaults_file)
        self.assertEqual(
            Path(resolved),
            Path("C:/settings/models/aiming.json").resolve(),
        )

    def test_rejects_non_object_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "defaults.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_hardware_defaults(path)


if __name__ == "__main__":
    unittest.main()
