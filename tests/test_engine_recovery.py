import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from napari_raman_widget.core_guard import install_core_guard
from napari_raman_widget.engine_recovery import (
    ConfigurationReloader,
    ReloadCancelled,
    ReloadDiagnostics,
)


class FakeCore:
    def __init__(self):
        self.calls = []
        self.load_errors = []
        self.load_entered = None
        self.release_load = None
        self.mda = SimpleNamespace(is_running=lambda: False, _canceled=False)

    def stopSequenceAcquisition(self):
        self.calls.append("stop")

    def unloadAllDevices(self):
        self.calls.append("unload")

    def loadSystemConfiguration(self, path):
        self.calls.append(("load", path))
        if self.load_entered is not None:
            self.load_entered.set()
            if not self.release_load.wait(2):
                raise RuntimeError("test load timed out")
        if self.load_errors:
            raise self.load_errors.pop(0)

    def waitForSystem(self):
        self.calls.append("ready")

    def setConfig(self, group, channel):
        self.calls.append((group, channel))

    def getPosition(self):
        self.calls.append("read position")
        return 42


class RecoveryTests(unittest.TestCase):
    def make_reloader(self, core=None):
        core = core or FakeCore()
        messages = []
        reloader = ConfigurationReloader(core, messages.append)
        reloader._wait = lambda delay: core.calls.append(("delay", delay))
        return core, reloader, messages

    def test_full_reload_delays_after_unload_and_preserves_warmup(self):
        core, reloader, messages = self.make_reloader()
        reloader.reload("rig.cfg")
        self.assertEqual(core.calls, [
            ("delay", 0.1), "stop", "unload", ("delay", 2.0),
            ("load", "rig.cfg"), "ready", ("Channel", "GFP"), "ready",
            ("Channel", "BF"), "ready",
        ])
        self.assertTrue(any("succeeded" in msg for msg in messages))

    def test_com17_failure_retries_full_unload_and_reports_owner_once(self):
        core, reloader, messages = self.make_reloader()
        core.load_errors = [RuntimeError('Error in device "COM17"')] * 2
        with patch("napari_raman_widget.engine_recovery.diagnose_serial_port",
                   return_value="PID 123 python.exe: current acquisition process") as scan:
            reloader.reload("rig.cfg", attempts=3)
        self.assertEqual(core.calls.count("unload"), 3)
        self.assertEqual(core.calls.count(("delay", 2.0)), 3)
        scan.assert_called_once_with("COM17", timeout=5.0)
        self.assertTrue(any("PID 123" in msg for msg in messages))

    def test_failure_preserves_original_error_and_restores_guard_pause(self):
        core = FakeCore()
        guard = install_core_guard(core, config_file="rig.cfg", post_unload_delay=0)
        guard.pause()  # acquisition already owns retries
        _, reloader, _ = self.make_reloader(core)
        error = RuntimeError("controller unavailable")
        core.load_errors = [error]
        with self.assertRaises(RuntimeError) as caught:
            reloader.reload("rig.cfg", attempts=1)
        self.assertIs(caught.exception.__cause__, error)
        self.assertTrue(guard.paused)
        guard.resume()
        self.assertFalse(guard.paused)
        self.assertEqual(core.getPosition(), 42)

    def test_warmup_failure_does_not_enter_nested_guard_retries(self):
        core = FakeCore()
        original_config = core.setConfig
        failures = [RuntimeError("channel controller unavailable")]
        def set_config(group, channel):
            original_config(group, channel)
            if failures:
                raise failures.pop()
        core.setConfig = set_config
        guard = install_core_guard(core, config_file="rig.cfg", post_unload_delay=0)
        _, reloader, _ = self.make_reloader(core)
        with patch.object(guard, "reload_configuration") as nested_reload:
            reloader.reload("rig.cfg", attempts=2)
        nested_reload.assert_not_called()
        self.assertFalse(guard.paused)
        self.assertEqual(core.calls.count("unload"), 2)
        self.assertEqual(core.calls.count(("Channel", "GFP")), 2)

    def test_other_hardware_calls_wait_until_entire_reload_finishes(self):
        core, reloader, _ = self.make_reloader()
        core.load_entered = threading.Event()
        core.release_load = threading.Event()
        errors = []
        def recover():
            try:
                reloader.reload("rig.cfg", attempts=1)
            except Exception as error:
                errors.append(error)
        recovery = threading.Thread(target=recover)
        read_started = threading.Event()
        read_finished = threading.Event()
        def read():
            read_started.set()
            core.getPosition()
            read_finished.set()
        reader = threading.Thread(target=read)
        recovery.start()
        try:
            self.assertTrue(core.load_entered.wait(1))
            reader.start()
            self.assertTrue(read_started.wait(1))
            self.assertFalse(read_finished.wait(0.05))
        finally:
            core.release_load.set()
            recovery.join(2)
            if reader.ident is not None:
                reader.join(2)
        self.assertFalse(errors)
        self.assertFalse(recovery.is_alive())
        self.assertTrue(read_finished.is_set())
        self.assertEqual(core.calls[-2:], ["ready", "read position"])

    def test_shutdown_interrupts_post_unload_delay_and_releases_gate(self):
        core = FakeCore()
        reloader = ConfigurationReloader(core, lambda msg: None)
        # Trigger cancellation immediately after unloading, before reopening.
        original_unload = core.unloadAllDevices
        def unload():
            original_unload()
            reloader.request_shutdown()
        core.unloadAllDevices = unload
        with self.assertRaises(ReloadCancelled):
            reloader.reload("rig.cfg", post_unload_delay=60)
        self.assertNotIn(("load", "rig.cfg"), core.calls)
        self.assertEqual(core.getPosition(), 42)

    def test_mda_cancel_interrupts_retry_wait(self):
        core = FakeCore()
        core.mda = SimpleNamespace(is_running=lambda: True, _canceled=True)
        reloader = ConfigurationReloader(core, lambda msg: None)
        with self.assertRaises(ReloadCancelled):
            reloader.reload("rig.cfg")
        self.assertEqual(core.calls, [])

    def test_validation_and_fallback_configuration(self):
        core, reloader, _ = self.make_reloader()
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                reloader.reload(None, post_unload_delay=value)
        with self.assertRaises(ValueError):
            reloader.reload(None, attempts=0)
        reloader.reload(None, attempts=1, post_unload_delay=0)
        self.assertIn(("load", "test3.cfg"), core.calls)

    def test_local_engine_overrides_upstream_reload(self):
        from napari_raman_widget.live_cell_tracking import make_live_cell_engine_type
        class Base:
            def __init__(self):
                self._mmc = FakeCore()
                self._config_file = "rig.cfg"
            def reload(self, N=100):
                raise AssertionError("upstream reload must not be called")
        logs = []
        engine = make_live_cell_engine_type(Base)(tracking_log=logs.append)
        with patch.object(engine._configuration_reloader, "reload") as reload:
            engine.reload(N=2)
        reload.assert_called_once_with("rig.cfg", attempts=2, post_unload_delay=2.0)
        engine.request_shutdown()
        self.assertTrue(engine._configuration_reloader._cancel.is_set())


class DiagnosticsTests(unittest.TestCase):
    def test_diagnostic_failure_does_not_hide_reload_failure(self):
        messages = []
        diagnostics = ReloadDiagnostics(messages.append)
        with patch("napari_raman_widget.engine_recovery.diagnose_serial_port",
                   side_effect=PermissionError("denied")):
            diagnostics.failed(RuntimeError("COM17 could not open"))
        self.assertTrue(any("unavailable: denied" in msg for msg in messages))

    def test_unrelated_error_does_not_inspect_com17(self):
        diagnostics = ReloadDiagnostics(lambda msg: None)
        with patch("napari_raman_widget.engine_recovery.diagnose_serial_port") as scan:
            diagnostics.failed(RuntimeError("COM14 timeout"))
        scan.assert_not_called()

    def test_slow_native_call_gets_watchdog_diagnostic(self):
        logged = threading.Event()
        def report(msg):
            if "example owner" in msg:
                logged.set()
        diagnostics = ReloadDiagnostics(report)
        with patch("napari_raman_widget.engine_recovery.diagnose_serial_port",
                   return_value="example owner"):
            with diagnostics.watch("loading configuration", timeout=0.01):
                self.assertTrue(logged.wait(1))


if __name__ == "__main__":
    unittest.main()
