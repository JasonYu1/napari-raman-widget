"""Truthful progress reporting for finite selected-point collections."""

import unittest

import numpy as np

from napari_raman_widget.acquisition.control import AcquisitionCancelled
from napari_raman_widget.point_collection import collect_repeated_spectra


class _RecordingAcquire:
    def __init__(self):
        self.calls = []
        self.results = []

    def __call__(self, points, exposure_ms, **read_mode_options):
        points = np.asarray(points, dtype=float)
        self.calls.append((points.copy(), float(exposure_ms), dict(read_mode_options)))
        call_index = len(self.calls)
        result = np.column_stack(
            (points, np.full(len(points), call_index, dtype=float))
        )
        self.results.append(result)
        return result


class PointCollectionProgressTests(unittest.TestCase):
    def run_collection(
        self,
        *,
        repeats=6,
        exposure_ms=2500,
        acquire=None,
        progress=None,
        cancelled=lambda: False,
        stage_prefix=None,
    ):
        acquire = acquire or _RecordingAcquire()
        events = []
        progress = progress or (lambda *event: events.append(event))
        spectra = collect_repeated_spectra(
            acquire,
            np.asarray([1.5, -2.0]),
            repeats,
            exposure_ms,
            read_mode_options={
                "read_mode": "single_track",
                "track_center": 185,
                "track_height": 140,
            },
            progress=progress,
            cancelled=cancelled,
            point_index=7,
            stage_prefix=stage_prefix,
        )
        return spectra, acquire, events

    def test_progress_advances_only_after_real_batches_and_concatenates(self):
        acquire = _RecordingAcquire()
        events = []

        def progress(*event):
            events.append((event, len(acquire.calls)))

        spectra, _, _ = self.run_collection(
            acquire=acquire,
            progress=progress,
        )

        self.assertEqual([len(call[0]) for call in acquire.calls], [2, 2, 2])
        self.assertEqual(
            [(event[0], event[1]) for event, _ in events],
            [(0, 6), (2, 6), (4, 6), (6, 6)],
        )
        self.assertEqual([calls_seen for _, calls_seen in events], [0, 1, 2, 3])
        self.assertTrue(
            all(isinstance(event[2], str) and event[2] for event, _ in events)
        )

        for points, exposure, options in acquire.calls:
            np.testing.assert_array_equal(
                points, np.tile([1.5, -2.0], (len(points), 1))
            )
            self.assertEqual(exposure, 2500)
            self.assertEqual(
                options,
                {
                    "read_mode": "single_track",
                    "track_center": 185,
                    "track_height": 140,
                },
            )
        np.testing.assert_array_equal(
            spectra,
            np.concatenate(acquire.results, axis=0),
        )

    def test_custom_stage_prefix_describes_dark_detector_work(self):
        _, _, events = self.run_collection(stage_prefix="Collecting dark spectra")

        self.assertEqual(
            [event[:2] for event in events],
            [(0, 6), (2, 6), (4, 6), (6, 6)],
        )
        self.assertTrue(
            all(event[2].startswith("Collecting dark spectra (") for event in events)
        )

    def test_batches_are_bounded_and_never_leave_a_final_singleton(self):
        cases = (
            # Four exposures target roughly five seconds; the trailing one is
            # merged into a real batch instead of being duplicated or dropped.
            (9, 1250, 5),
            # Short exposures remain bounded by the collector batch cap, with
            # one extra sample allowed when absorbing a trailing singleton.
            (65, 100, 33),
        )
        for repeats, exposure_ms, maximum_batch in cases:
            with self.subTest(repeats=repeats, exposure_ms=exposure_ms):
                spectra, acquire, _ = self.run_collection(
                    repeats=repeats,
                    exposure_ms=exposure_ms,
                )
                sizes = [len(call[0]) for call in acquire.calls]
                self.assertEqual(sum(sizes), repeats)
                self.assertTrue(all(2 <= size <= maximum_batch for size in sizes))
                self.assertEqual(spectra.shape[0], repeats)

    def test_cancellation_is_observed_between_completed_batches(self):
        acquire = _RecordingAcquire()
        events = []
        stop_requested = False

        def progress(*event):
            nonlocal stop_requested
            events.append(event)
            if event[0] == 2:
                stop_requested = True

        with self.assertRaises(AcquisitionCancelled):
            self.run_collection(
                acquire=acquire,
                progress=progress,
                cancelled=lambda: stop_requested,
            )

        self.assertEqual([len(call[0]) for call in acquire.calls], [2])
        self.assertEqual([event[:2] for event in events], [(0, 6), (2, 6)])

    def test_rejects_a_batch_with_the_wrong_first_dimension(self):
        calls = []
        events = []

        def acquire(points, exposure_ms, **read_mode_options):
            calls.append(np.asarray(points).copy())
            return np.zeros((len(points) - 1, 4), dtype=float)

        with self.assertRaises(ValueError):
            self.run_collection(
                acquire=acquire, progress=lambda *event: events.append(event)
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual([event[:2] for event in events], [(0, 6)])


if __name__ == "__main__":
    unittest.main()
