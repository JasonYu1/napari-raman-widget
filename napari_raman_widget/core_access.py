"""Shared, in-process exclusion for hardware access during core recovery.

This gate is separate from retry policy: even camera-buffer reads, relative
moves, and calls made while the retry guard is paused must wait for recovery.
Install it on the shared core before handing the core to widgets or workers.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any

__all__ = ["CoreAccess", "install_core_access"]

# MMCore's low-level API uses camelCase. Include the few hardware operations
# with entirely lowercase names as well. Do not hold the gate around run_mda,
# which may start/join a thread that itself needs the core, or around context
# managers/generators whose bodies execute after the method returns.
_CORE_METHOD = re.compile(r"^[a-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*$")
_LOWERCASE_OPERATIONS = {"home", "reset", "snap", "stop"}
_UNGATED_HELPERS = {"setContext", "iterDevices", "iterProperties", "iterConfigGroups"}
_INSTALL_LOCK = threading.RLock()


class CoreAccess:
    """A reentrant hardware gate shared by all callers of one core instance.

    ``exclusive()`` waits for an in-flight core call and then reserves the core
    for the calling thread until the context exits. The owner can call patched
    methods normally, including nested calls from synchronous core callbacks.

    Keep GUI orchestration and thread joins outside an exclusive context: a
    recovery owner must not wait for a GUI callback/worker that may itself be
    waiting to use the core. Normal queued Qt notifications do not require the
    receiver to complete. This gate cannot interrupt a hung native device call
    or protect a second process, another core instance, or raw methods saved
    before installation.
    """

    _MARKER = "_napari_raman_core_access"

    def __init__(self, core: Any) -> None:
        self.core = core
        self._lock = threading.RLock()

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """Reserve hardware access until the entire recovery is complete."""
        with self._lock:
            yield

    def _wrap(self, original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(original)
        def gated(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return original(*args, **kwargs)

        return gated

    def _install(self) -> None:
        originals: dict[str, Callable[..., Any]] = {}
        for name in dir(self.core):
            if name in _UNGATED_HELPERS:
                continue
            if _CORE_METHOD.match(name) or name in _LOWERCASE_OPERATIONS:
                method = getattr(self.core, name, None)
                if callable(method):
                    originals[name] = method

        installed: list[str] = []
        try:
            for name, original in originals.items():
                setattr(self.core, name, self._wrap(original))
                installed.append(name)
            setattr(self.core, self._MARKER, self)
        except Exception:
            for name in installed:
                setattr(self.core, name, originals[name])
            raise


def install_core_access(core: Any) -> CoreAccess:
    """Install hardware exclusion once on this exact shared core instance.

    This adds no retry behavior. Install beneath ``CoreRetryGuard`` so its
    saved originals and paused calls retain the same exclusion guarantee.
    """
    with _INSTALL_LOCK:
        existing = getattr(core, CoreAccess._MARKER, None)
        if isinstance(existing, CoreAccess):
            return existing
        access = CoreAccess(core)
        access._install()
        return access
