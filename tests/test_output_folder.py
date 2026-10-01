"""Applying an output folder is explicit, safe, and independent of hardware."""

import ast
import gc
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtWidgets import QApplication, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from napari_raman_widget import output_folder


class _Guard:
    def __init__(self, config):
        self.config_file = config
        self.updates = []

    def set_config_file(self, config):
        self.config_file = config
        self.updates.append(config)


class OutputFolderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.original_cwd = Path.cwd()
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        os.chdir(self.base)
        self.owners = []

    def tearDown(self):
        os.chdir(self.original_cwd)
        for owner in self.owners:
            owner.close()
            owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.owners.clear()
        gc.collect()
        self.temporary.cleanup()

    def owner(self, path=""):
        owner = QWidget()
        owner.out_path = QLineEdit(path, owner)
        owner.status = QLabel("Status: unchanged", owner)
        owner._hardware_shutdown_started = False
        owner._acquisition_jobs = SimpleNamespace(is_running=False)
        owner._live_raman_worker = None
        owner._raman_mda_thread = None
        owner.demo_live_timer = None
        owner.core = None
        owner.core_guard = None
        owner.apply_output_folder = lambda: output_folder.apply_output_folder(owner)
        layout = QVBoxLayout(owner)
        row = QHBoxLayout()
        row.addWidget(owner.out_path)
        layout.addLayout(row)
        output_folder.add_output_folder_controls(owner, row, layout)
        self.owners.append(owner)
        return owner

    def apply(self, owner, **kwargs):
        return output_folder.apply_output_folder(owner, show_success=False, **kwargs)

    def assert_active(self, owner, path):
        self.assertEqual(owner.output_folder_status.text(), f"Active folder: {path}")

    def test_button_applies_staged_folder_and_creates_parents(self):
        owner = self.owner("nested/results")
        target = self.base / "nested" / "results"
        self.assertFalse(target.exists())
        self.assertEqual(Path.cwd(), self.base)
        self.assert_active(owner, self.base)

        owner.apply_output_folder_btn.click()

        self.assertEqual(Path.cwd(), target)
        self.assertEqual(owner.out_path.text(), str(target))
        self.assert_active(owner, target)
        self.assertIn("applied", owner.status.text())
        self.assertEqual(list(target.iterdir()), [])  # the writable probe is removed

    def test_editing_path_alone_never_changes_active_folder(self):
        owner = self.owner()
        owner.out_path.setText("later")
        self.assertEqual(Path.cwd(), self.base)
        self.assert_active(owner, self.base)
        self.assertFalse((self.base / "later").exists())

    def test_relative_path_is_normalized_and_repeated_apply_does_not_nest(self):
        owner = self.owner("  results  ")
        self.assertTrue(self.apply(owner))
        target = self.base / "results"
        self.assertEqual(owner.out_path.text(), str(target))
        self.assertTrue(self.apply(owner))
        self.assertEqual(Path.cwd(), target)
        self.assertFalse((target / "results").exists())

    def test_blank_path_uses_current_directory(self):
        owner = self.owner("   ")
        with patch.object(output_folder.Path, "mkdir") as make_directory, \
                patch.object(output_folder.tempfile, "TemporaryFile") as probe, \
                patch.object(output_folder.os, "chdir") as change_directory:
            self.assertTrue(self.apply(owner))
            make_directory.assert_not_called()
            probe.assert_not_called()
            change_directory.assert_not_called()
        self.assertEqual(Path.cwd(), self.base)
        self.assertEqual(owner.out_path.text(), "   ")
        self.assert_active(owner, self.base)

    def test_next_relative_save_uses_applied_folder(self):
        owner = self.owner("new-results")
        self.assertTrue(self.apply(owner))
        Path("next-spectrum.txt").write_text("1,2,3", encoding="utf-8")
        self.assertEqual((self.base / "new-results" / "next-spectrum.txt").read_text(
            encoding="utf-8"), "1,2,3")
        self.assertFalse((self.base / "next-spectrum.txt").exists())

    def test_all_registered_panels_show_active_folder_without_changing_pending_paths(self):
        first, second = self.owner("first"), self.owner("second")
        self.assertTrue(self.apply(first))
        target = self.base / "first"
        self.assert_active(first, target)
        self.assert_active(second, target)
        self.assertEqual(second.out_path.text(), "second")
        self.assertEqual(second.status.text(), "Status: unchanged")

    def test_file_target_fails_without_mutating_inputs_or_active_state(self):
        target = self.base / "not-a-folder"
        target.write_text("keep me", encoding="utf-8")
        owner = self.owner(str(target))
        source = self.base / "camera.cfg"
        source.touch()
        owner.cfg_path = QLineEdit("camera.cfg", owner)
        owner.core_guard = _Guard("camera.cfg")
        self.assertFalse(self.apply(owner))
        self.assertEqual(Path.cwd(), self.base)
        self.assertEqual(owner.out_path.text(), str(target))
        self.assertEqual(owner.cfg_path.text(), "camera.cfg")
        self.assertEqual(owner.core_guard.config_file, "camera.cfg")
        self.assertEqual(owner.core_guard.updates, [])
        self.assert_active(owner, self.base)
        self.assertIn("could not apply", owner.status.text())
        self.assertEqual(target.read_text(encoding="utf-8"), "keep me")

    def test_unwritable_probe_blocks_chdir_and_input_changes(self):
        owner = self.owner("readonly")
        (self.base / "camera.cfg").touch()
        owner.cfg_path = QLineEdit("camera.cfg", owner)
        owner.core_guard = _Guard("camera.cfg")
        with patch.object(output_folder.tempfile, "TemporaryFile", side_effect=PermissionError("denied")), \
                patch.object(output_folder.os, "chdir") as change_directory:
            self.assertFalse(self.apply(owner))
            change_directory.assert_not_called()
        self.assertEqual(Path.cwd(), self.base)
        self.assertEqual(owner.out_path.text(), "readonly")
        self.assertEqual(owner.cfg_path.text(), "camera.cfg")
        self.assertEqual(owner.core_guard.updates, [])
        self.assert_active(owner, self.base)
        self.assertIn("denied", owner.status.text())

    def test_failed_chdir_keeps_input_fields_guard_and_labels_unchanged(self):
        owner, second = self.owner("unavailable"), self.owner()
        (self.base / "camera.cfg").touch()
        owner.cfg_path = QLineEdit("camera.cfg", owner)
        owner.core_guard = _Guard("camera.cfg")
        with patch.object(output_folder.os, "chdir", side_effect=OSError("cannot enter")):
            self.assertFalse(self.apply(owner))
        self.assertEqual(Path.cwd(), self.base)
        self.assertEqual(owner.cfg_path.text(), "camera.cfg")
        self.assertEqual(owner.core_guard.updates, [])
        self.assertEqual(owner.out_path.text(), "unavailable")
        self.assert_active(owner, self.base)
        self.assert_active(second, self.base)

    def test_invalid_null_path_is_reported_without_changes(self):
        owner = self.owner("bad\0folder")
        self.assertFalse(self.apply(owner))
        self.assertEqual(Path.cwd(), self.base)
        self.assert_active(owner, self.base)
        self.assertIn("unchanged", owner.status.text())

    def test_busy_conditions_in_another_panel_block_before_filesystem_changes(self):
        owner, blocker = self.owner("blocked"), self.owner()
        cases = (
            ("_hardware_shutdown_started", True),
            ("_acquisition_jobs", SimpleNamespace(is_running=True)),
            ("_live_raman_worker", object()),
            ("_raman_mda_thread", SimpleNamespace(is_alive=lambda: True)),
            ("demo_live_timer", SimpleNamespace(isActive=lambda: True)),
            ("core", SimpleNamespace(mda=SimpleNamespace(is_running=True))),
            ("core", SimpleNamespace(mda=SimpleNamespace(is_running=lambda: True))),
            ("core", SimpleNamespace(isSequenceRunning=lambda: True)),
        )
        for name, value in cases:
            previous = getattr(blocker, name)
            with self.subTest(state=name):
                setattr(blocker, name, value)
                try:
                    with patch.object(output_folder.Path, "mkdir") as make_directory, \
                            patch.object(output_folder.os, "chdir") as change_directory:
                        self.assertFalse(self.apply(owner))
                        make_directory.assert_not_called()
                        change_directory.assert_not_called()
                    self.assertIn("unchanged", owner.status.text())
                    self.assert_active(owner, self.base)
                    self.assert_active(blocker, self.base)
                finally:
                    setattr(blocker, name, previous)
        self.assertEqual(Path.cwd(), self.base)
        self.assertFalse((self.base / "blocked").exists())

    def test_busy_initiator_is_blocked_too(self):
        owner = self.owner("blocked")
        owner._live_raman_worker = object()
        self.assertFalse(self.apply(owner))
        self.assertEqual(Path.cwd(), self.base)
        self.assertFalse((self.base / "blocked").exists())

    def test_status_read_errors_fail_closed_for_other_panels(self):
        owner, blocker = self.owner("blocked"), self.owner()
        failing_call = Mock(side_effect=RuntimeError("device unavailable"))
        cases = (
            ("core", SimpleNamespace(isSequenceRunning=failing_call)),
            ("core", SimpleNamespace(mda=SimpleNamespace(is_running=failing_call))),
            ("_raman_mda_thread", SimpleNamespace(is_alive=failing_call)),
            ("demo_live_timer", SimpleNamespace(isActive=failing_call)),
        )
        for name, value in cases:
            previous = getattr(blocker, name)
            with self.subTest(state=name):
                setattr(blocker, name, value)
                try:
                    with patch.object(output_folder.Path, "mkdir") as make_directory, \
                            patch.object(output_folder.os, "chdir") as change_directory:
                        self.assertFalse(self.apply(owner))
                        make_directory.assert_not_called()
                        change_directory.assert_not_called()
                    self.assertIn("cannot check acquisition status", owner.status.text())
                finally:
                    setattr(blocker, name, previous)
        self.assertEqual(Path.cwd(), self.base)

    def test_idle_hardware_is_queried_without_connection_or_device_changes(self):
        owner = self.owner("results")
        owner.connect_hardware = Mock()
        owner.core = SimpleNamespace(
            mda=SimpleNamespace(is_running=lambda: False),
            isSequenceRunning=Mock(return_value=False),
            unloadAllDevices=Mock(), loadSystemConfiguration=Mock(),
        )
        owner._raman_mda_thread = SimpleNamespace(is_alive=lambda: False)
        owner.demo_live_timer = SimpleNamespace(isActive=lambda: False)
        self.assertTrue(self.apply(owner))
        owner.core.isSequenceRunning.assert_called_once_with()
        owner.connect_hardware.assert_not_called()
        owner.core.unloadAllDevices.assert_not_called()
        owner.core.loadSystemConfiguration.assert_not_called()

    def test_existing_relative_input_files_are_pinned_for_all_registered_owners(self):
        owner, second = self.owner("results"), self.owner()
        names = ("cfg_path", "tf_path", "sel_vdm_path", "dark_noise_path",
                 "spectral_calibration_path", "mda_track_cfg_input", "px2stage_ds_path")
        for index, panel in enumerate((owner, second)):
            for name in names:
                filename = f"{index}-{name}.json"
                (self.base / filename).touch()
                setattr(panel, name, QLineEdit(filename, panel))
            panel.output_name = QLineEdit("new-experiment", panel)
        self.assertTrue(self.apply(owner))
        for index, panel in enumerate((owner, second)):
            for name in names:
                self.assertEqual(getattr(panel, name).text(), str(self.base / f"{index}-{name}.json"))
            self.assertEqual(panel.output_name.text(), "new-experiment")

    def test_missing_empty_and_absolute_input_fields_remain_unchanged(self):
        owner = self.owner("results")
        existing = self.base / "absolute.cfg"
        existing.touch()
        owner.cfg_path = QLineEdit(str(existing), owner)
        owner.tf_path = QLineEdit("missing.json", owner)
        owner.dark_noise_path = QLineEdit("", owner)
        self.assertTrue(self.apply(owner))
        self.assertEqual(owner.cfg_path.text(), str(existing))
        self.assertEqual(owner.tf_path.text(), "missing.json")
        self.assertEqual(owner.dark_noise_path.text(), "")

    def test_relative_zarr_dataset_directory_is_pinned(self):
        owner = self.owner("results")
        dataset = self.base / "pixel-to-stage.zarr"
        dataset.mkdir()
        owner.px2stage_ds_path = QLineEdit("pixel-to-stage.zarr", owner)
        self.assertTrue(self.apply(owner))
        self.assertEqual(owner.px2stage_ds_path.text(), str(dataset))

    def test_guard_relative_configs_are_pinned_even_when_missing(self):
        owner, second = self.owner("results"), self.owner()
        owner.core_guard = _Guard("missing-camera.cfg")
        second.core_guard = _Guard("other-camera.cfg")
        self.assertTrue(self.apply(owner))
        self.assertEqual(owner.core_guard.updates, [str(self.base / "missing-camera.cfg")])
        self.assertEqual(second.core_guard.updates, [str(self.base / "other-camera.cfg")])
        self.assertTrue(self.apply(owner))
        self.assertEqual(len(owner.core_guard.updates), 1)
        self.assertEqual(len(second.core_guard.updates), 1)

    def test_preserve_false_keeps_legacy_relative_input_lookup(self):
        owner, second = self.owner("results"), self.owner()
        (self.base / "camera.cfg").touch()
        for panel in (owner, second):
            panel.cfg_path = QLineEdit("camera.cfg", panel)
            panel.core_guard = _Guard("camera.cfg")
        self.assertTrue(self.apply(owner, preserve_inputs=False))
        self.assertEqual(owner.cfg_path.text(), "camera.cfg")
        self.assertEqual(owner.core_guard.config_file, "camera.cfg")
        self.assertEqual(owner.core_guard.updates, [])
        # Connect preserves its own legacy lookup, not another panel's inputs.
        self.assertEqual(second.cfg_path.text(), str(self.base / "camera.cfg"))
        self.assertEqual(second.core_guard.config_file, str(self.base / "camera.cfg"))
        self.assertEqual(second.core_guard.updates, [str(self.base / "camera.cfg")])
        self.assert_active(owner, self.base / "results")
        self.assert_active(second, self.base / "results")

    def test_silent_success_does_not_replace_existing_status_message(self):
        owner = self.owner("results")
        self.assertTrue(self.apply(owner))
        self.assertEqual(owner.status.text(), "Status: unchanged")

    def test_destroyed_panels_are_unregistered_before_another_apply(self):
        owner, old = self.owner("results"), self.owner()
        old._live_raman_worker = object()
        old.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.owners.remove(old)
        self.assertNotIn(old, output_folder._OUTPUT_OWNERS)
        del old
        gc.collect()
        self.assertTrue(self.apply(owner))


class OutputFolderWidgetWiringTests(unittest.TestCase):
    @staticmethod
    def widget(filename, classname):
        path = Path(__file__).resolve().parents[1] / "napari_raman_widget" / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        widget = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == classname)
        methods = {node.name: node for node in widget.body if isinstance(node, ast.FunctionDef)}
        return tree, methods

    def test_both_widgets_wire_explicit_apply_but_browse_only_stages_path(self):
        for filename, classname in (("hardware_widget.py", "HardwareWidget"),
                                    ("demo_widget.py", "DemoWidget")):
            with self.subTest(widget=classname):
                tree, methods = self.widget(filename, classname)
                imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
                           and node.module == "output_folder"]
                names = {alias.name for node in imports for alias in node.names}
                self.assertTrue({"add_output_folder_controls", "apply_output_folder"} <= names)
                constructor_calls = [node for node in ast.walk(methods["__init__"])
                                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
                self.assertTrue(any(node.func.id == "add_output_folder_controls"
                                    for node in constructor_calls))
                apply_calls = [node for node in ast.walk(methods["apply_output_folder"])
                               if isinstance(node, ast.Call)]
                self.assertTrue(any(isinstance(node.func, ast.Name) and node.func.id == "apply_output_folder"
                                    for node in apply_calls))
                browse_calls = [node for node in ast.walk(methods["browse_out"])
                                if isinstance(node, ast.Call)]
                self.assertTrue(any(isinstance(node.func, ast.Attribute) and node.func.attr == "setText"
                                    and isinstance(node.func.value, ast.Attribute)
                                    and node.func.value.attr == "out_path" for node in browse_calls))
                for node in browse_calls:
                    name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                    self.assertNotIn(name, {"apply_output_folder", "chdir", "connect"})

    def test_hardware_connect_checks_shared_folder_helper_before_devices(self):
        _, methods = self.widget("hardware_widget.py", "HardwareWidget")
        connect = methods["connect"]
        calls = [node for node in ast.walk(connect) if isinstance(node, ast.Call)]
        folder = next(node for node in calls if isinstance(node.func, ast.Name)
                      and node.func.id == "apply_output_folder")
        keywords = {item.arg: item.value for item in folder.keywords}
        self.assertIs(keywords["show_success"].value, False)
        self.assertIs(keywords["preserve_inputs"].value, False)
        device_calls = [node for node in calls if
                        (isinstance(node.func, ast.Name) and node.func.id == "AndorSpectraCollector")
                        or (isinstance(node.func, ast.Attribute) and node.func.attr == "instance"
                            and isinstance(node.func.value, ast.Name) and node.func.value.id == "CMMCorePlus")]
        self.assertEqual(len(device_calls), 2)
        self.assertTrue(all(folder.lineno < node.lineno for node in device_calls))
        gate = next(node for node in connect.body if isinstance(node, ast.If)
                    and isinstance(node.test, ast.UnaryOp) and node.test.operand is folder)
        self.assertTrue(any(isinstance(node, ast.Return) for node in gate.body))


if __name__ == "__main__":
    unittest.main()
