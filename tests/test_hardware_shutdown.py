import unittest

from napari_raman_widget.hardware_shutdown import shutdown_core_hardware


class _FakeMDA:
    def __init__(self):
        self.cancel_calls = 0

    def cancel(self):
        self.cancel_calls += 1


class _FakeThread:
    def __init__(self):
        self.alive = True
        self.join_calls = []

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.join_calls.append(timeout)
        self.alive = False


class _FakeCore:
    def __init__(self):
        self.mda = _FakeMDA()
        self.stop_calls = 0
        self.unload_calls = 0

    def stopSequenceAcquisition(self):
        self.stop_calls += 1

    def unloadAllDevices(self):
        self.unload_calls += 1


class _FakeGuard:
    def __init__(self, core):
        self.originals = {
            "stopSequenceAcquisition": core.stopSequenceAcquisition,
            "unloadAllDevices": core.unloadAllDevices,
        }


class HardwareShutdownTests(unittest.TestCase):
    def test_shutdown_cancels_joins_and_unloads_without_retry_wrappers(self):
        core = _FakeCore()
        guard = _FakeGuard(core)
        thread = _FakeThread()
        delays = []

        # If this wrapped method is used instead of guard.originals, the test
        # fails and would represent a dangerous reload during application exit.
        core.stopSequenceAcquisition = lambda: (_ for _ in ()).throw(
            AssertionError("wrapped stop method used")
        )

        result = shutdown_core_hardware(
            core,
            guard=guard,
            mda_thread=thread,
            join_timeout=3.0,
            release_delay=1.5,
            sleep=delays.append,
        )

        self.assertEqual(core.mda.cancel_calls, 1)
        self.assertEqual(core.stop_calls, 2)
        self.assertEqual(core.unload_calls, 1)
        self.assertEqual(thread.join_calls, [3.0])
        self.assertEqual(delays, [1.5])
        self.assertTrue(result.mda_thread_stopped)
        self.assertTrue(result.devices_unloaded)
        self.assertEqual(result.errors, ())


if __name__ == "__main__":
    unittest.main()
