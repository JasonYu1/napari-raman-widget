import unittest

from napari_raman_widget.live_spectra import LiveSpectrumWorker


class LiveSpectrumWorkerTests(unittest.TestCase):
    def test_repeats_until_stop_is_requested(self) -> None:
        acquisitions = []
        received = []
        worker = None

        def acquire_one():
            acquisitions.append(len(acquisitions) + 1)
            if len(acquisitions) == 3:
                worker.request_stop()
            return acquisitions[-1]

        worker = LiveSpectrumWorker(acquire_one)
        worker.spectrum_ready.connect(
            lambda value, count, _elapsed: received.append((value, count))
        )

        worker.run()

        self.assertEqual(acquisitions, [1, 2, 3])
        self.assertEqual(received, [(1, 1), (2, 2), (3, 3)])


if __name__ == "__main__":
    unittest.main()
