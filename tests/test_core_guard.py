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


if __name__ == "__main__":
    unittest.main()
