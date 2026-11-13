"""Read-only serial ownership diagnostics, bounded even if a driver stalls.

Native inspection queries only handles already owned by this Python process.
It never opens a COM port, duplicates handles, or closes borrowed handles.
One daemon worker is retained until it exits, preventing stuck-query pileups.
Other processes can be inspected only with an existing, licensed Handle tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import re
import subprocess
import sys
import threading
import time
from typing import Any, Callable


_MAX_OUTPUT = 32768
_MAX_OWNERS = 32
Emit = Callable[..., None]


@dataclass
class _Scan:
    port: str
    records: list[dict[str, Any]] = field(default_factory=list)
    lock: Any = field(default_factory=threading.Lock)
    thread: threading.Thread | None = None


_scan_lock = threading.Lock()
_active_scan: _Scan | None = None


def _one_line(value: object, limit: int = 500) -> str:
    return " ".join(str(value).split())[:limit]


def _format_result(
    port: str,
    records: list[dict[str, Any]],
    parent_pid: int,
    problem: str = "",
) -> str:
    lines = [f"Serial ownership diagnostic for {port}:"]
    owners: set[int] = set()
    summary = None
    for record in records:
        kind = record.get("kind")
        if kind == "mapping":
            lines.append(f"  Windows device: {_one_line(record.get('device'))}")
        elif kind == "owner":
            try:
                pid = int(record["pid"])
            except (KeyError, TypeError, ValueError):
                continue
            if pid in owners or len(owners) >= _MAX_OWNERS:
                continue
            owners.add(pid)
            identity = "current acquisition process" if pid == parent_pid else "other process"
            name = _one_line(record.get("process") or "process name unavailable")
            lines.append(f"  Matching open handle: PID {pid}, {name} ({identity}).")
        elif kind in ("error", "limitation"):
            lines.append(f"  {_one_line(record.get('message'))}")
        elif kind == "summary":
            summary = record
    if not owners:
        lines.append("  No matching process handle was visible; this does not prove the port is free.")
    if summary:
        state = "finished" if summary.get("complete") else "incomplete"
        lines.append(
            f"  Current-process character-handle scan {state}; "
            f"failed handle queries={_one_line(summary.get('failed_handles', 0))}."
        )
    elif not problem:
        problem = "Current-process ownership scan did not complete."
    if problem:
        lines.append(f"  {_one_line(problem)}")
    lines.append("  Snapshot only: handles may change, and driver/USB state is not established by this scan.")
    return "\n".join(lines)


def _append(scan: _Scan, kind: str, **values: Any) -> None:
    with scan.lock:
        if len(scan.records) < 64:
            scan.records.append({"kind": kind, **values})


def _scan_worker(scan: _Scan, budget: float) -> None:
    emit = lambda kind, **values: _append(scan, kind, **values)
    deadline = time.monotonic() + budget
    try:
        _inspect_current_process(scan.port, emit, budget)
    except Exception as exc:
        emit("error", message=f"Current-process inspection unavailable: {_one_line(exc)}")
    try:
        with scan.lock:
            device = next(
                (record["device"] for record in scan.records if record["kind"] == "mapping"),
                "",
            )
        _inspect_external_owners(device, emit, max(0.0, deadline - time.monotonic()))
    except Exception as exc:
        emit("limitation", message=f"Other-process inspection unavailable: {_one_line(exc)}")


def diagnose_serial_port(port: str = "COM17", timeout: float = 5.0) -> str:
    """Return ownership evidence without opening COM or interrupting recovery.

    A stuck native metadata query cannot safely be cancelled in this process.
    Caller waiting is bounded; its single daemon worker remains registered until
    it really exits. Later calls report the existing query instead of starting
    another. No hardware/API queries happen at module import or application startup.
    """
    global _active_scan
    try:
        port = str(port).strip().upper()
        if not re.fullmatch(r"COM[1-9][0-9]{0,4}", port):
            return "Serial ownership diagnostic unavailable: invalid COM port name."
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout <= 0:
            return f"Serial ownership diagnostic for {port} unavailable: invalid timeout."
        if sys.platform != "win32":
            return f"Serial ownership diagnostic for {port} unavailable: Windows only."
        with _scan_lock:
            if _active_scan is not None and _active_scan.thread.is_alive():
                scan = _active_scan
                already_running = True
            else:
                scan = _Scan(port)
                scan.thread = threading.Thread(
                    target=_scan_worker,
                    args=(scan, timeout * 0.8),
                    name="serial-ownership-diagnostic",
                    daemon=True,
                )
                _active_scan = scan
                scan.thread.start()
                already_running = False
        if not already_running:
            scan.thread.join(timeout)
        with scan.lock:
            records = list(scan.records) if scan.port == port else []
        problem = ""
        if scan.thread.is_alive():
            problem = (
                f"Metadata inspection for {scan.port} is still running"
                + (f" after {timeout:g}s" if not already_running else "")
                + "; results are partial. No additional diagnostic worker will start "
                "until it exits. Recovery may continue."
            )
        return _format_result(port, records, os.getpid(), problem)
    except Exception as exc:
        return f"Serial ownership diagnostic unavailable: {_one_line(type(exc).__name__)}: {_one_line(exc)}"


def _inspect_current_process(port: str, emit: Emit, budget: float) -> None:
    import ctypes
    from ctypes import wintypes
    import msvcrt

    deadline = time.monotonic() + budget
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")
    handle = wintypes.HANDLE
    ulong_ptr = ctypes.c_size_t

    class HandleEntry(ctypes.Structure):
        _fields_ = [
            ("object", ctypes.c_void_p), ("pid", ulong_ptr), ("value", ulong_ptr),
            ("access", wintypes.ULONG), ("backtrace", wintypes.USHORT),
            ("type_index", wintypes.USHORT), ("attributes", wintypes.ULONG),
            ("reserved", wintypes.ULONG),
        ]

    class UnicodeString(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.USHORT), ("maximum_length", wintypes.USHORT),
            ("buffer", ctypes.c_void_p),
        ]

    kernel.QueryDosDeviceW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    kernel.QueryDosDeviceW.restype = wintypes.DWORD
    kernel.GetFileType.argtypes = [handle]
    kernel.GetFileType.restype = wintypes.DWORD
    ntdll.NtQuerySystemInformation.argtypes = [
        wintypes.ULONG, ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG)
    ]
    ntdll.NtQuerySystemInformation.restype = wintypes.LONG
    ntdll.NtQueryObject.argtypes = [
        handle, wintypes.ULONG, ctypes.c_void_p, wintypes.ULONG,
        ctypes.POINTER(wintypes.ULONG),
    ]
    ntdll.NtQueryObject.restype = wintypes.LONG

    mapping = ctypes.create_unicode_buffer(32768)
    if not kernel.QueryDosDeviceW(port, mapping, len(mapping)):
        raise ctypes.WinError(ctypes.get_last_error())
    device = mapping.value
    emit("mapping", device=device[:500])

    # This ordinary source-file descriptor is the only file we open and close.
    # It supplies the File object-type index for this Windows handle snapshot.
    descriptor = os.open(__file__, os.O_RDONLY)
    try:
        marker = msvcrt.get_osfhandle(descriptor)
        size = 1024 * 1024
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Time budget exhausted reading the handle table")
            storage = ctypes.create_string_buffer(size)
            needed = wintypes.ULONG()
            status = ntdll.NtQuerySystemInformation(64, storage, size, ctypes.byref(needed))
            if status >= 0:
                break
            if status & 0xFFFFFFFF != 0xC0000004:
                raise OSError(
                    "Handle enumeration denied/unavailable "
                    f"(NTSTATUS 0x{status & 0xFFFFFFFF:08X})"
                )
            size = max(size * 2, needed.value + 65536)
            if size > 64 * 1024 * 1024:
                raise OSError("Handle table exceeds the 64 MiB diagnostic limit")
        count = ulong_ptr.from_buffer(storage).value
        offset = 2 * ctypes.sizeof(ulong_ptr)
        if count > (size - offset) // ctypes.sizeof(HandleEntry):
            raise OSError("Unexpected Windows handle-table layout")
        entries = (HandleEntry * count).from_buffer(storage, offset)
        pid = os.getpid()
        own_entries = [entry for entry in entries if entry.pid == pid]
        file_type = next(
            (entry.type_index for entry in own_entries if entry.value == marker), None
        )
        if file_type is None:
            raise OSError("Current-process File handle type was not visible")
        failed = 0
        complete = True
        for entry in own_entries:
            if time.monotonic() >= deadline:
                complete = False
                break
            if entry.type_index != file_type or entry.value == marker:
                continue
            # These are borrowed handles of THIS process, never duplicates.
            # Do not close them. Only character devices qualify for name queries.
            file_kind = kernel.GetFileType(handle(entry.value))
            if file_kind != 2:  # FILE_TYPE_CHAR
                if file_kind == 0:
                    failed += 1
                continue
            name_buffer = ctypes.create_string_buffer(65536)
            needed = wintypes.ULONG()
            status = ntdll.NtQueryObject(
                handle(entry.value), 1, name_buffer, len(name_buffer), ctypes.byref(needed)
            )
            if status < 0:
                failed += 1
                continue
            name = UnicodeString.from_buffer(name_buffer)
            if not name.buffer or not name.length:
                continue
            start = ctypes.addressof(name_buffer)
            if name.length % 2 or not start <= name.buffer <= start + len(name_buffer) - name.length:
                failed += 1
                continue
            object_name = ctypes.wstring_at(name.buffer, name.length // 2).rstrip("\0")
            if object_name.casefold() == device.casefold():
                emit("owner", pid=pid, process=os.path.basename(sys.executable))
        emit("summary", complete=complete, failed_handles=failed)
    finally:
        os.close(descriptor)


def _parse_handle_output(output: str, device: str) -> list[dict[str, Any]]:
    owners: dict[int, dict[str, Any]] = {}
    current = None
    device_token = re.compile(r"(?:^|\s)" + re.escape(device) + r"(?=\s|$)", re.IGNORECASE)
    for line in output[:_MAX_OUTPUT].splitlines():
        match = re.match(r"^\s*(.+?)\s+pid:\s*(\d+)\b", line, re.IGNORECASE)
        if match:
            current = {"pid": int(match[2]), "process": match[1].strip()[:260]}
        if current and device_token.search(line):
            owners[current["pid"]] = current
            if len(owners) >= _MAX_OWNERS:
                break
    return list(owners.values())


def _inspect_external_owners(device: str, emit: Emit, timeout: float) -> None:
    import shutil
    import winreg

    executable = shutil.which("handle64.exe") or shutil.which("handle.exe")
    if not executable:
        emit("limitation", message=(
            "Other-process owners cannot be checked: Sysinternals Handle is not installed "
            "on PATH. Native inspection is limited to the current acquisition process."
        ))
        return
    # Never trigger a first-run license dialog, or autoaccept an EULA.
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Sysinternals\Handle") as key:
            accepted = winreg.QueryValueEx(key, "EulaAccepted")[0] == 1
    except OSError:
        accepted = False
    if not accepted:
        emit("limitation", message=(
            "Other-process inspection unavailable: Handle's EULA has not already "
            "been accepted by this Windows user; the tool was not launched."
        ))
        return
    if not device or timeout <= 0:
        emit("limitation", message="Other-process inspection unavailable: no device mapping or time budget.")
        return
    result = subprocess.run(
        [executable, "-nobanner", device],
        capture_output=True, text=True, errors="replace", timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False,
    )
    owners = _parse_handle_output(result.stdout, device)
    for owner in owners:
        emit("owner", **owner)
    if result.returncode:
        emit("limitation", message=(
            f"Other-process inspection incomplete: Handle exited {result.returncode}; "
            f"{_one_line(result.stderr or result.stdout)}"
        ))
    elif not owners:
        emit("limitation", message=(
            "Handle found no visible matching process; permissions or driver state "
            "may still hide an owner."
        ))
