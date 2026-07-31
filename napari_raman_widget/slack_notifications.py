"""Optional Slack notifications for background Raman MDA failures."""

from __future__ import annotations

import logging
import os
import traceback
from collections.abc import Callable, Iterable
from functools import wraps
from threading import Thread
from typing import Any, ParamSpec, TypeVar

import requests

__all__ = [
    "SLACK_WEBHOOK_ENV_VAR",
    "notify_exception",
    "run_mda_with_notifications",
    "send_slack_message",
    "set_webhook_url",
    "slack_notify",
]


SLACK_WEBHOOK_ENV_VAR = "NAPARI_RAMAN_SLACK_WEBHOOK_URL"
_REQUEST_TIMEOUT_SECONDS = 5.0
_MAX_TRACEBACK_CHARACTERS = 35_000
_webhook_url_override: str | None = None
_logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def set_webhook_url(url: str | None) -> None:
    """Override the Slack webhook URL for this process.

    Passing ``None`` restores lookup through
    ``NAPARI_RAMAN_SLACK_WEBHOOK_URL``. Passing an empty string explicitly
    disables Slack notifications for the process.
    """
    global _webhook_url_override
    _webhook_url_override = None if url is None else str(url).strip()


def _webhook_url() -> str:
    if _webhook_url_override is not None:
        return _webhook_url_override
    return os.getenv(SLACK_WEBHOOK_ENV_VAR, "").strip()


def send_slack_message(message: str) -> bool:
    """Send one webhook message without allowing Slack to break acquisition.

    Returns ``True`` after a successful request and ``False`` when Slack is
    disabled or the request fails.
    """
    url = _webhook_url()
    if not url:
        return False

    try:
        response = requests.post(
            url,
            json={"text": str(message)},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        # Notification is secondary. Never replace the microscope exception
        # with a webhook/network exception. Do not log ``error`` itself: a
        # requests exception may contain the secret webhook URL.
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        detail = f"HTTP {status}" if status is not None else type(error).__name__
        _logger.error("Could not send Slack notification (%s)", detail)
        return False
    return True


def notify_exception(
    error: BaseException,
    *,
    context: str = "Raman MDA",
    mention_channel: bool = True,
) -> bool:
    """Send a formatted traceback for a real exception, never a warning."""
    if isinstance(error, Warning):
        return False

    formatted = "".join(
        traceback.TracebackException.from_exception(error).format()
    )
    if len(formatted) > _MAX_TRACEBACK_CHARACTERS:
        formatted = "... traceback truncated ...\n" + formatted[
            -_MAX_TRACEBACK_CHARACTERS:
        ]

    mention = " <!channel>" if mention_channel else ""
    return send_slack_message(
        f"{context} failed!{mention}\n```\n{formatted}\n```"
    )


def slack_notify(function: Callable[P, R]) -> Callable[P, R]:
    """Notify Slack if a synchronous function raises a real exception.

    This is useful for ordinary functions. Background MDA must use
    :func:`run_mda_with_notifications` so exceptions raised on its worker
    thread are observed.
    """

    @wraps(function)
    def _wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return function(*args, **kwargs)
        except Warning:
            # A warning only reaches here if warning filters promoted it to an
            # exception. Do not send an alert, but preserve that configured
            # behavior.
            raise
        except Exception as error:
            notify_exception(error, context=function.__qualname__)
            raise

    return _wrapped


def _run_mda_target(
    runner: Any,
    events: Iterable[Any],
    output: Any,
    context: str,
    on_error: Callable[[Exception], None] | None,
) -> None:
    try:
        runner.run(events, output=output)
    except Warning:
        # Normal warnings.warn(...) calls do not enter this branch. It only
        # handles warnings explicitly promoted to exceptions, which should not
        # page Slack as hardware failures.
        raise
    except Exception as error:
        if on_error is not None:
            try:
                on_error(error)
            except Exception as callback_error:
                _logger.error(
                    "MDA error callback failed: %s", callback_error
                )
        notify_exception(error, context=context)
        # A bare raise retains the acquisition traceback. ``raise error``
        # would add this helper as a misleading traceback origin.
        raise


def run_mda_with_notifications(
    core: Any,
    events: Iterable[Any],
    *,
    output: Any = None,
    context: str = "Raman MDA",
    on_error: Callable[[Exception], None] | None = None,
) -> Thread:
    """Run an MDA on a thread and alert Slack only for real exceptions.

    This mirrors ``CMMCorePlus.run_mda(..., block=False)`` while placing the
    exception handler inside the worker thread, where acquisition failures
    actually occur.
    """
    runner = core.mda
    if runner.is_running():
        raise ValueError(
            "Cannot start an MDA while the previous MDA is still running."
        )

    thread = Thread(
        target=_run_mda_target,
        args=(runner, events, output, context, on_error),
        name="RamanMDA",
    )
    thread.start()
    return thread
