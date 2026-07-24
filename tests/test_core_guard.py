"""Tests for the runtime MMCore retry guard."""

from __future__ import annotations

import unittest

from napari_raman_widget.core_guard import install_core_guard


class FakeCore:
    def __init__(self) -> None:
        self.get_calls = 0
        self.set_calls = 0
        self.relative_calls = 0
        self.mda_calls = 0
        self.wait_calls = 0
        self.reload_count = 0
        self.loaded_config = None
        self.config_history = []
        self.image_calls = 0

    def getPosition(self) -> float:
        self.get_calls += 1
        if self.get_calls < 3:
            raise RuntimeError("temporary read failure")
        return 12.5

    def setXYPosition(self, x: float, y: float) -> None:
        self.set_calls += 1
        if self.set_calls < 2:
            raise RuntimeError("temporary stage failure")

    def setRelativeXYPosition(self, dx: float, dy: float) -> None:
        self.relative_calls += 1
        raise RuntimeError("relative move failure")

    def run_mda(self, sequence: object) -> None:
        self.mda_calls += 1
        raise RuntimeError("MDA failure")

    def waitForSystem(self) -> None:
        self.wait_calls += 1

    def stopSequenceAcquisition(self) -> None:
        pass

    def unloadAllDevices(self) -> None:
        pass

    def loadSystemConfiguration(self, config_file: str) -> None:
        self.reload_count += 1
        self.loaded_config = config_file

    def setConfig(self, group: str, config: str) -> None:
        self.config_history.append((group, config))

    def getImage(self):
        self.image_calls += 1
        raise RuntimeError("empty camera buffer")


class ReloadingFakeCore(FakeCore):
    def getPosition(self) -> float:
        self.get_calls += 1
        if self.reload_count == 0:
            raise RuntimeError("core requires a configuration reload")
        return 42.0


class CoreRetryGuardTests(unittest.TestCase):
    def test_retries_safe_reads_and_absolute_setters(self) -> None:
        core = FakeCore()
        guard = install_core_guard(
            core, attempts=3, initial_delay=0, delay_increment=0
        )
        self.assertEqual(core.getPosition(), 12.5)
        core.setXYPosition(1, 2)
        self.assertEqual(core.get_calls, 3)
        self.assertEqual(core.set_calls, 2)
        self.assertEqual(core.wait_calls, 1)
        guard.uninstall()

    def test_does_not_retry_relative_moves_or_mda(self) -> None:
        core = FakeCore()
        install_core_guard(core, attempts=3, initial_delay=0, delay_increment=0)
        with self.assertRaises(RuntimeError):
            core.setRelativeXYPosition(1, 2)
        with self.assertRaises(RuntimeError):
            core.run_mda(object())
        self.assertEqual(core.relative_calls, 1)
        self.assertEqual(core.mda_calls, 1)

    def test_installation_is_idempotent(self) -> None:
        core = FakeCore()
        first = install_core_guard(core, initial_delay=0, delay_increment=0)
        second = install_core_guard(core, initial_delay=0, delay_increment=0)
        self.assertIs(first, second)
        self.assertEqual(first.attempts, 10)

    def test_reloads_config_halfway_then_retries_original_operation(self) -> None:
        core = ReloadingFakeCore()
        guard = install_core_guard(
            core,
            attempts=10,
            initial_delay=0,
            delay_increment=0,
            config_file="microscope.cfg",
            reload_after_failures=5,
            reload_attempts=1,
            reload_delay=0,
        )

        self.assertEqual(core.getPosition(), 42.0)
        self.assertEqual(core.get_calls, 6)
        self.assertEqual(core.reload_count, 1)
        self.assertEqual(core.loaded_config, "microscope.cfg")
        self.assertEqual(
            core.config_history,
            [("Channel", "GFP"), ("Channel", "BF")],
        )
        guard.uninstall()

    def test_existing_guard_receives_new_config_on_reconnect(self) -> None:
        core = FakeCore()
        first = install_core_guard(core, config_file="first.cfg")
        second = install_core_guard(core, config_file="second.cfg")

        self.assertIs(first, second)
        self.assertEqual(first.config_file, "second.cfg")

    def test_camera_buffer_reads_are_not_wrapped(self) -> None:
        core = FakeCore()
        guard = install_core_guard(
            core,
            attempts=10,
            initial_delay=0,
            delay_increment=0,
            config_file="microscope.cfg",
        )

        with self.assertRaisesRegex(RuntimeError, "empty camera buffer"):
            core.getImage()
        self.assertEqual(core.image_calls, 1)
        self.assertEqual(core.reload_count, 0)
        guard.uninstall()

    def test_pause_bypasses_wrappers_until_resumed(self) -> None:
        core = FakeCore()
        guard = install_core_guard(
            core, attempts=3, initial_delay=0, delay_increment=0
        )

        guard.pause()
        with self.assertRaisesRegex(RuntimeError, "temporary read failure"):
            core.getPosition()
        self.assertEqual(core.get_calls, 1)

        guard.resume()
        self.assertEqual(core.getPosition(), 12.5)
        self.assertEqual(core.get_calls, 3)
        guard.uninstall()


if __name__ == "__main__":
    unittest.main()
