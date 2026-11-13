"""Hardware exclusion tests using fake cores and explicit thread handoffs."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from napari_raman_widget.core_access import install_core_access
from napari_raman_widget.core_guard import install_core_guard


class ObservedLock:
    """Expose when another thread reaches the gate, without timing sleeps."""

    def __init__(self, lock):
        self.lock = lock
        self.owner = threading.get_ident()
        self.contended = threading.Event()

    def __enter__(self):
        if threading.get_ident() != self.owner:
            self.contended.set()
        self.lock.acquire()

    def __exit__(self, *args):
        self.lock.release()


class FakeCore:
    def __init__(self):
        self.calls = []

    def run_mda(self, sequence):
        return sequence


HARDWARE_METHODS = (
    "getPosition", "getImage", "getLastImageAndMD", "popNextImage",
    "setRelativePosition", "setRelativeXYPosition", "setRelativeXYZPosition",
    "setProperty", "getSerialPortAnswer", "setSerialPortCommand",
    "readFromSerialPort", "writeToSerialPort", "purgeSerialPort",
    "loadSystemConfiguration", "unloadAllDevices", "loadDevice", "unloadDevice",
    "initializeAllDevices", "setConfig", "waitForSystem", "snapImage",
    "startSequenceAcquisition", "startContinuousSequenceAcquisition",
    "stopSequenceAcquisition", "home", "reset", "snap", "stop",
)


def hardware_method(name):
    def method(self, *args):
        self.calls.append(name)
        return name
    return method


for _name in HARDWARE_METHODS:
    setattr(FakeCore, _name, hardware_method(_name))


class CoreAccessTests(unittest.TestCase):
    def assert_waits_for_exclusive(self, core, operation):
        access = install_core_access(core)
        observed = ObservedLock(access._lock)
        access._lock = observed
        completed = threading.Event()
        errors = []

        def call():
            try:
                operation()
            except BaseException as error:
                errors.append(error)
            finally:
                completed.set()

        worker = threading.Thread(target=call, daemon=True)
        try:
            with access.exclusive():
                worker.start()
                self.assertTrue(observed.contended.wait(2), "call never reached gate")
                self.assertFalse(completed.is_set())
                self.assertEqual(core.calls, [])
            self.assertTrue(completed.wait(2), "call did not resume after recovery")
            self.assertEqual(errors, [])
        finally:
            worker.join(2)
            access._lock = observed.lock

    def test_all_hardware_calls_wait_during_recovery(self):
        for name in HARDWARE_METHODS:
            with self.subTest(method=name):
                core = FakeCore()
                install_core_access(core)
                self.assert_waits_for_exclusive(core, getattr(core, name))
                self.assertEqual(core.calls, [name])

    def test_installation_is_shared_by_instance_only(self):
        core = FakeCore()
        first = install_core_access(core)
        original_wrapper = core.getPosition
        self.assertIs(install_core_access(core), first)
        self.assertIs(core.getPosition, original_wrapper)
        self.assertIsNot(install_core_access(FakeCore()), first)

    def test_owner_can_nest_gate_and_call_hardware(self):
        core = FakeCore()
        access = install_core_access(core)
        finished = threading.Event()

        def recover():
            with access.exclusive(), access.exclusive():
                core.unloadAllDevices()
                core.loadSystemConfiguration("test.cfg")
                core.getImage()
            finished.set()

        worker = threading.Thread(target=recover, daemon=True)
        worker.start()
        self.assertTrue(finished.wait(2), "owner deadlocked on its own hardware call")
        worker.join(2)
        self.assertEqual(core.calls, ["unloadAllDevices", "loadSystemConfiguration", "getImage"])

    def test_exclusive_waits_for_existing_hardware_call(self):
        core = FakeCore()
        entered = threading.Event()
        release = threading.Event()
        exclusive_entered = threading.Event()

        def get_position():
            entered.set()
            if not release.wait(2):
                raise AssertionError("test did not release in-flight call")

        core.getPosition = get_position
        access = install_core_access(core)
        in_flight = threading.Thread(target=core.getPosition, daemon=True)
        in_flight.start()
        self.assertTrue(entered.wait(2))

        observed = ObservedLock(access._lock)
        access._lock = observed

        def recover():
            with access.exclusive():
                exclusive_entered.set()

        recovery = threading.Thread(target=recover, daemon=True)
        try:
            recovery.start()
            self.assertTrue(observed.contended.wait(2))
            self.assertFalse(exclusive_entered.is_set())
        finally:
            release.set()
            in_flight.join(2)
            recovery.join(2)
        self.assertTrue(exclusive_entered.is_set())

    def test_failed_exclusive_context_releases_other_callers(self):
        core = FakeCore()
        access = install_core_access(core)
        with self.assertRaisesRegex(RuntimeError, "reload failed"):
            with access.exclusive():
                raise RuntimeError("reload failed")
        self.assert_waits_for_exclusive(core, core.getImage)

    def test_high_level_mda_orchestration_is_not_gated(self):
        core = FakeCore()
        original = core.run_mda
        install_core_access(core)
        self.assertEqual(core.run_mda, original)

    def test_pausing_retry_guard_keeps_hardware_gate(self):
        core = FakeCore()
        guard = install_core_guard(core)
        guard.pause()
        try:
            self.assert_waits_for_exclusive(core, core.getPosition)
            self.assertEqual(core.calls, ["getPosition"])
        finally:
            guard.resume()

    def test_shutdown_saved_originals_keep_hardware_gate(self):
        core = FakeCore()
        guard = install_core_guard(core)
        self.assert_waits_for_exclusive(core, guard.originals["stopSequenceAcquisition"])
        self.assertEqual(core.calls, ["stopSequenceAcquisition"])

    def test_uninstalling_retry_guard_keeps_hardware_gate(self):
        core = FakeCore()
        guard = install_core_guard(core)
        guard.uninstall()
        self.assert_waits_for_exclusive(core, core.getPosition)
        self.assertEqual(core.calls, ["getPosition"])

    def test_guard_reload_keeps_gate_during_post_unload_delay(self):
        core = FakeCore()
        guard = install_core_guard(
            core, config_file="test.cfg", reload_attempts=1, reload_delay=0,
        )
        access = install_core_access(core)
        observed = ObservedLock(access._lock)
        access._lock = observed
        settling = threading.Event()
        release_settle = threading.Event()
        call_finished = threading.Event()
        reload_results = []

        def settle(duration):
            self.assertEqual(duration, 2.0)
            settling.set()
            if not release_settle.wait(2):
                raise AssertionError("test did not release post-unload delay")

        def unrelated_call():
            core.setRelativeXYPosition(1, 2)
            call_finished.set()

        recovery = threading.Thread(
            target=lambda: reload_results.append(guard.reload_configuration()),
            daemon=True,
        )
        caller = threading.Thread(target=unrelated_call, daemon=True)
        with patch("napari_raman_widget.core_guard.time.sleep", side_effect=settle):
            try:
                recovery.start()
                self.assertTrue(settling.wait(2))
                self.assertEqual(core.calls, ["stopSequenceAcquisition", "unloadAllDevices"])
                observed.contended.clear()
                caller.start()
                self.assertTrue(observed.contended.wait(2))
                self.assertFalse(call_finished.is_set())
                self.assertNotIn("loadSystemConfiguration", core.calls)
            finally:
                release_settle.set()
                recovery.join(2)
                if caller.ident is not None:
                    caller.join(2)
        self.assertEqual(reload_results, [True])
        self.assertTrue(call_finished.is_set())
        self.assertEqual(core.calls[-1], "setRelativeXYPosition")


if __name__ == "__main__":
    unittest.main()
