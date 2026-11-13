"""Ownership diagnostics must be bounded, honest, and hardware independent."""

from __future__ import annotations

import subprocess
import threading
import unittest
from unittest.mock import MagicMock, patch

from napari_raman_widget import serial_port_diagnostics as diagnostics


class SerialPortDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        scan = diagnostics._active_scan
        if scan is not None:
            scan.thread.join(1)
            self.assertFalse(scan.thread.is_alive())
        diagnostics._active_scan = None

    def run_diagnostic(self, records):
        def inspect(port, emit, budget):
            for record in records:
                record = dict(record)
                emit(record.pop("kind"), **record)
        with (
            patch.object(diagnostics.sys, "platform", "win32"),
            patch.object(diagnostics.os, "getpid", return_value=123),
            patch.object(diagnostics, "_inspect_current_process", side_effect=inspect),
            patch.object(diagnostics, "_inspect_external_owners"),
        ):
            return diagnostics.diagnose_serial_port()

    def test_identifies_current_process_and_other_process(self):
        message = self.run_diagnostic([
            {"kind": "mapping", "device": r"\Device\ProlificSerial0"},
            {"kind": "owner", "pid": 123, "process": "python.exe"},
            {"kind": "owner", "pid": 456, "process": "other.exe"},
            {"kind": "summary", "complete": True, "failed_handles": 2},
        ])
        self.assertIn(r"\Device\ProlificSerial0", message)
        self.assertIn("PID 123, python.exe (current acquisition process)", message)
        self.assertIn("PID 456, other.exe (other process)", message)
        self.assertIn("failed handle queries=2", message)
        self.assertNotIn("No matching", message)

    def test_no_matches_never_claims_port_free_even_when_scan_finished(self):
        message = self.run_diagnostic([{"kind": "summary", "complete": True}])
        self.assertIn("does not prove the port is free", message)
        self.assertIn("driver/USB state is not established", message)
        self.assertIn("Current-process character-handle scan finished", message)

    def test_permission_failure_does_not_interrupt_recovery(self):
        with (
            patch.object(diagnostics.sys, "platform", "win32"),
            patch.object(diagnostics, "_inspect_current_process", side_effect=PermissionError("Access denied")),
            patch.object(diagnostics, "_inspect_external_owners"),
        ):
            message = diagnostics.diagnose_serial_port()
        self.assertIn("Current-process inspection unavailable: Access denied", message)
        self.assertIn("does not prove the port is free", message)

    def test_stalled_query_retains_partial_owner_and_prevents_worker_pileup(self):
        release = threading.Event()
        started = threading.Event()
        def blocked(port, emit, budget):
            emit("owner", pid=123, process="python.exe")
            started.set()
            release.wait(3)
        with (
            patch.object(diagnostics.sys, "platform", "win32"),
            patch.object(diagnostics.os, "getpid", return_value=123),
            patch.object(diagnostics, "_inspect_current_process", side_effect=blocked) as inspect,
            patch.object(diagnostics, "_inspect_external_owners"),
        ):
            try:
                message = diagnostics.diagnose_serial_port(timeout=0.02)
                self.assertTrue(started.is_set())
                self.assertIn("current acquisition process", message)
                self.assertIn("still running", message)
                self.assertIn("No additional diagnostic worker", message)
                worker = diagnostics._active_scan.thread
                self.assertTrue(worker.daemon)
                again = diagnostics.diagnose_serial_port(timeout=0.02)
                self.assertIs(diagnostics._active_scan.thread, worker)
                self.assertIn("still running", again)
                inspect.assert_called_once()
                different = diagnostics.diagnose_serial_port("COM18", timeout=0.02)
                self.assertNotIn("Matching open handle", different)
                self.assertIn("inspection for COM17", different)
            finally:
                release.set()
                diagnostics._active_scan.thread.join(1)
            diagnostics.diagnose_serial_port(timeout=0.02)
            self.assertEqual(inspect.call_count, 2)

    def test_non_windows_never_starts_worker(self):
        with (
            patch.object(diagnostics.sys, "platform", "linux"),
            patch.object(diagnostics, "_inspect_current_process") as inspect,
        ):
            self.assertIn("Windows only", diagnostics.diagnose_serial_port())
        inspect.assert_not_called()
        self.assertIsNone(diagnostics._active_scan)

    def test_invalid_port_or_timeout_never_starts_worker(self):
        with patch.object(diagnostics, "_inspect_current_process") as inspect:
            self.assertIn("invalid COM port", diagnostics.diagnose_serial_port("COM17; echo test"))
            for timeout in (0, -1, float("nan"), float("inf")):
                self.assertIn("invalid timeout", diagnostics.diagnose_serial_port(timeout=timeout))
        inspect.assert_not_called()
        self.assertIsNone(diagnostics._active_scan)

    def test_incomplete_scan_and_duplicate_owner_are_reported_once(self):
        owner = {"kind": "owner", "pid": 456, "process": "other.exe"}
        message = self.run_diagnostic([
            owner, owner,
            {"kind": "summary", "complete": False, "failed_handles": 7},
        ])
        self.assertEqual(message.count("PID 456"), 1)
        self.assertIn("scan incomplete", message)
        self.assertIn("failed handle queries=7", message)

    def test_external_tool_failure_does_not_interrupt_recovery(self):
        with (
            patch.object(diagnostics.sys, "platform", "win32"),
            patch.object(diagnostics, "_inspect_current_process"),
            patch.object(diagnostics, "_inspect_external_owners", side_effect=subprocess.TimeoutExpired([], 1)),
        ):
            message = diagnostics.diagnose_serial_port()
        self.assertIn("Other-process inspection unavailable", message)

    def test_handle_output_supports_inline_and_multiline_formats(self):
        device = r"\Device\ProlificSerial0"
        output = (
            "python.exe pid: 123 type: File 1F0: " + device + "\n"
            "other.exe pid: 456 USER\\user\n"
            "  50: File (RW-) " + device + "\n"
            "other.exe pid: 456 type: File 1E0: " + device + "\n"
            "unrelated.exe pid: 789 type: File 222: C:\\ordinary.txt\n"
        )
        self.assertEqual(diagnostics._parse_handle_output(output, device), [
            {"pid": 123, "process": "python.exe"},
            {"pid": 456, "process": "other.exe"},
        ])

    def test_handle_output_requires_exact_device_token(self):
        device = r"\Device\ProlificSerial0"
        output = (
            "wrong.exe pid: 456 type: File 50: " + device + "1\n"
            "also-wrong.exe pid: 789 type: File 50: prefix" + device + "\n"
            "python.exe pid: 123 type: File 50: " + device.lower() + "  \n"
        )
        self.assertEqual(diagnostics._parse_handle_output(output, device), [
            {"pid": 123, "process": "python.exe"},
        ])

    @unittest.skipUnless(diagnostics.sys.platform == "win32", "Windows-only imports")
    def test_missing_handle_reports_own_process_limit(self):
        records = []
        with patch("shutil.which", return_value=None):
            diagnostics._inspect_external_owners(
                r"\Device\ProlificSerial0",
                lambda kind, **values: records.append(values),
                1,
            )
        self.assertIn("limited to the current acquisition process", records[0]["message"])

    @unittest.skipUnless(diagnostics.sys.platform == "win32", "Windows-only imports")
    def test_handle_is_not_launched_without_prior_eula_acceptance(self):
        with (
            patch("shutil.which", return_value=r"C:\Tools\handle64.exe"),
            patch("winreg.OpenKey", side_effect=FileNotFoundError),
            patch.object(diagnostics.subprocess, "run") as run,
        ):
            diagnostics._inspect_external_owners(r"\Device\ProlificSerial0", lambda *a, **k: None, 1)
        run.assert_not_called()

    @unittest.skipUnless(diagnostics.sys.platform == "win32", "Windows-only imports")
    def test_existing_accepted_handle_runs_hidden_with_timeout_and_no_eula_switch(self):
        records = []
        with (
            patch("shutil.which", return_value=r"C:\Tools\handle64.exe"),
            patch("winreg.OpenKey"),
            patch("winreg.QueryValueEx", return_value=(1, 4)),
            patch.object(diagnostics.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "python.exe pid: 123 type: File 50: \\Device\\ProlificSerial0", "")) as run,
        ):
            diagnostics._inspect_external_owners(
                r"\Device\ProlificSerial0",
                lambda kind, **values: records.append({"kind": kind, **values}),
                1,
            )
        args, kwargs = run.call_args
        self.assertEqual(args[0], [r"C:\Tools\handle64.exe", "-nobanner", r"\Device\ProlificSerial0"])
        self.assertEqual(kwargs["timeout"], 1)
        self.assertEqual(kwargs["creationflags"], subprocess.CREATE_NO_WINDOW)
        self.assertEqual(records, [{"kind": "owner", "pid": 123, "process": "python.exe"}])

    @unittest.skipUnless(diagnostics.sys.platform == "win32", "native ctypes definitions need Windows")
    def test_native_mapping_failure_does_not_open_any_file(self):
        kernel = MagicMock()
        kernel.QueryDosDeviceW.return_value = 0
        with (
            patch("ctypes.WinDLL", side_effect=[kernel, MagicMock()]),
            patch("ctypes.get_last_error", return_value=5),
            patch.object(diagnostics.os, "open") as open_file,
        ):
            with self.assertRaises(PermissionError):
                diagnostics._inspect_current_process("COM17", lambda *a, **k: None, 1)
        open_file.assert_not_called()
        kernel.OpenProcess.assert_not_called()
        kernel.DuplicateHandle.assert_not_called()
        kernel.CloseHandle.assert_not_called()


if __name__ == "__main__":
    unittest.main()
