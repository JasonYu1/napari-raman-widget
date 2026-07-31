import os
import unittest
import warnings
from unittest.mock import Mock, patch

from napari_raman_widget import slack_notifications


class _Runner:
    def __init__(self, action=None, running=False):
        self.action = action
        self.running = running
        self.calls = []

    def is_running(self):
        return self.running

    def run(self, events, *, output=None):
        self.calls.append((events, output))
        if self.action is not None:
            return self.action()
        return None


class _Core:
    def __init__(self, runner):
        self.mda = runner


class SlackNotificationTests(unittest.TestCase):
    def setUp(self):
        slack_notifications.set_webhook_url(None)

    def tearDown(self):
        slack_notifications.set_webhook_url(None)

    @patch.object(slack_notifications.requests, "post")
    def test_message_uses_environment_webhook(self, post):
        response = Mock()
        post.return_value = response
        with patch.dict(
            os.environ,
            {slack_notifications.SLACK_WEBHOOK_ENV_VAR: "https://example.test/hook"},
        ):
            sent = slack_notifications.send_slack_message("MDA failed")

        self.assertTrue(sent)
        post.assert_called_once_with(
            "https://example.test/hook",
            json={"text": "MDA failed"},
            timeout=5.0,
        )
        response.raise_for_status.assert_called_once_with()

    @patch.object(slack_notifications.requests, "post")
    def test_no_webhook_is_a_noop(self, post):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(
                slack_notifications.send_slack_message("MDA failed")
            )
        post.assert_not_called()

    @patch.object(slack_notifications.requests, "post")
    def test_webhook_failure_does_not_raise(self, post):
        post.side_effect = slack_notifications.requests.ConnectionError(
            "offline"
        )
        slack_notifications.set_webhook_url("https://example.test/hook")

        with self.assertLogs(slack_notifications.__name__, "ERROR"):
            sent = slack_notifications.send_slack_message("MDA failed")

        self.assertFalse(sent)

    @patch.object(slack_notifications, "notify_exception")
    def test_normal_warning_does_not_notify_or_stop_mda(self, notify):
        def warn():
            warnings.warn("camera temperature is settling", UserWarning)

        runner = _Runner(warn)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            slack_notifications._run_mda_target(
                runner, ["event"], None, "Raman MDA", None
            )

        self.assertEqual(len(seen), 1)
        notify.assert_not_called()

    @patch.object(slack_notifications, "notify_exception")
    def test_promoted_warning_does_not_notify(self, notify):
        warning = UserWarning("promoted warning")
        runner = _Runner(lambda: (_ for _ in ()).throw(warning))

        with self.assertRaises(UserWarning) as caught:
            slack_notifications._run_mda_target(
                runner, ["event"], None, "Raman MDA", None
            )

        self.assertIs(caught.exception, warning)
        notify.assert_not_called()

    @patch.object(slack_notifications, "notify_exception")
    def test_real_mda_error_notifies_callback_and_is_reraised(self, notify):
        error = RuntimeError("camera disconnected")
        callback = Mock()
        runner = _Runner(lambda: (_ for _ in ()).throw(error))

        with self.assertRaises(RuntimeError) as caught:
            slack_notifications._run_mda_target(
                runner, ["event"], None, "Raman MDA", callback
            )

        self.assertIs(caught.exception, error)
        callback.assert_called_once_with(error)
        notify.assert_called_once_with(error, context="Raman MDA")

    def test_background_runner_matches_nonblocking_mda(self):
        runner = _Runner()
        thread = slack_notifications.run_mda_with_notifications(
            _Core(runner), ["event"], output="writer"
        )
        thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(runner.calls, [(["event"], "writer")])

    def test_does_not_start_while_an_mda_is_running(self):
        with self.assertRaisesRegex(ValueError, "previous MDA"):
            slack_notifications.run_mda_with_notifications(
                _Core(_Runner(running=True)), ["event"]
            )


if __name__ == "__main__":
    unittest.main()
