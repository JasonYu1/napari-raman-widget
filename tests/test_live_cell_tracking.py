import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from napari_raman_widget.live_cell_tracking import make_live_cell_engine_type


class _Layer:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)

    def add(self, rows):
        self.data = np.vstack([self.data, np.atleast_2d(rows)])


class _Identity:
    def transform(self, points):
        return np.asarray(points, dtype=float)


class _TwoPointPattern:
    def transform(self, anchors):
        anchors = np.asarray(anchors, dtype=float)
        offsets = np.asarray([[0.0, 0.0], [0.01, 0.0]])
        return (anchors[:, None, :] + offsets).reshape(-1, 2)


class _PointSource:
    def __init__(self, layer, transformer=None):
        self.name = "cells"
        self._points = layer
        self._pos_idx = 1
        self._img_shape = (100, 100)
        self.transformer = transformer or _Identity()

    def get_mda_points(self, event, transform=True):
        rows = self._points.data[
            self._points.data[:, self._pos_idx] == event.index.get("p")
        ]
        points = rows[:, -2:] / np.asarray(self._img_shape)
        if transform:
            return self.transformer.transform(points)
        return points


class _StaleUpdateBase:
    def __init__(self, sources, during_update=None, fail=False):
        self._sources = list(sources)
        self.during_update = during_update or (lambda: None)
        self.fail = fail
        self._last_segments = {0: np.ones((4, 4), dtype=np.uint16)}

    @property
    def aiming_sources(self):
        return self._sources

    def update_aim(self, pos, event, img, use_same_img=False):
        del event, img, use_same_img
        counts = [
            np.count_nonzero(source._points.data[:, 1] == pos)
            for source in self.aiming_sources
        ]
        self.during_update()
        if self.fail:
            raise RuntimeError("tracking failed")
        for source, count in zip(self.aiming_sources, counts, strict=True):
            updated = np.array(source._points.data, copy=True)
            selected = updated[:, 1] == pos
            tracked = np.column_stack(
                [
                    np.arange(count, dtype=float) + 100,
                    np.arange(count, dtype=float) + 200,
                ]
            )
            # This is the stale blanket assignment used by the base engine.
            updated[selected, -2:] = tracked
            source._points.data = updated
        self._last_segments[pos] = np.full((4, 4), 2, dtype=np.uint16)
        return "updated"

    def record_raman(self, event):
        del event
        return None


class _GroupingRecordBase:
    def __init__(self, sources, fail=False):
        self._sources = list(sources)
        self.collected = None
        self.fail = fail

    @property
    def aiming_sources(self):
        return self._sources

    def update_aim(self, pos, event, img, use_same_img=False):
        del pos, event, img, use_same_img

    def record_raman(self, event):
        if self.fail:
            raise RuntimeError("collection failed")
        kept = []
        for source in self.aiming_sources:
            points = np.asarray(source.get_mda_points(event), dtype=float)
            labels = np.where(points[:, 0] < 0.5, 1, 2)
            ids, counts = np.unique(labels, return_counts=True)
            selected_id = ids[np.argmax(counts)]
            kept.append(points[labels == selected_id])
        self.collected = np.vstack(kept)
        return self.collected


class _AutoAddBase:
    def __init__(self, sources, during_update=None, batch=False):
        self._sources = list(sources)
        self.during_update = during_update or (lambda: None)
        self._batch = batch
        self._scale = 1
        self._circle_center = (15, 15)

        previous = np.zeros((30, 30), dtype=np.uint16)
        previous[3:10, 3:10] = 10
        current = np.zeros_like(previous)
        current[4:11, 4:11] = 99
        current[18:25, 19:26] = 42
        tracked = np.zeros((2, 30, 30), dtype=np.uint16)
        tracked[0, 3:10, 3:10] = 1
        tracked[1, 4:11, 4:11] = 1
        tracked[1, 18:25, 19:26] = 2

        self._last_segments = {0: previous}
        self.current = current
        self.tracked = tracked
        self._tracks = {}

    @property
    def aiming_sources(self):
        return self._sources

    def update_aim(self, pos, event, img, use_same_img=False):
        del event, img, use_same_img
        self.during_update()
        self._last_segments[pos] = self.current
        self._tracks[pos] = self.tracked

    def record_raman(self, event):
        del event


class _AutofocusBase:
    def __init__(self, initial_z, best_z, result):
        self.initial_z = initial_z
        self.best_z = best_z
        self.result = result
        self.z_moves = []

    def autofocus_w_raman(self, last_z, *args, **kwargs):
        self.autofocus_call = (last_z, args, kwargs)
        return self.initial_z, self.best_z, self.result

    def try_set_ZPosition(self, position):
        self.z_moves.append(position)


class _FakeTrenchTracker:
    def __init__(self, points_yx, labels):
        self.points_yx = np.asarray(points_yx, dtype=float)
        self.labels = np.asarray(labels, dtype=np.uint16)
        self.call = None
        self.reset_count = 0

    def update(self, **kwargs):
        self.call = kwargs
        count = len(kwargs["targets"])
        return SimpleNamespace(
            points_yx=self.points_yx,
            labels=self.labels,
            accepted=np.ones(count, dtype=bool),
            reasons=("accepted",) * count,
            geometry_valid=True,
            geometry_score=1.0,
        )

    def reset(self):
        self.reset_count += 1


class AutofocusSafeguardTests(unittest.TestCase):
    def make_engine(self, *, initial_z=100.0, best_z=100.0):
        result = object()
        engine_type = make_live_cell_engine_type(_AutofocusBase)
        engine = engine_type(initial_z, best_z, result)
        return engine, result

    def test_shift_over_four_um_restores_previous_z(self):
        for best_z in (104.01, 95.99):
            with self.subTest(best_z=best_z):
                engine, result = self.make_engine(best_z=best_z)

                initial_z, guarded_z, returned_result = engine.autofocus_w_raman(
                    100.0,
                    "point",
                    time=1,
                )

                self.assertEqual(initial_z, 100.0)
                self.assertEqual(guarded_z, 100.0)
                self.assertIs(returned_result, result)
                self.assertEqual(engine.z_moves, [100.0])
                self.assertEqual(
                    engine.autofocus_call,
                    (100.0, ("point",), {"time": 1}),
                )

    def test_shift_of_exactly_four_um_is_accepted(self):
        for best_z in (104.0, 96.0):
            with self.subTest(best_z=best_z):
                engine, result = self.make_engine(best_z=best_z)

                initial_z, guarded_z, returned_result = (
                    engine.autofocus_w_raman(100.0, "point")
                )

                self.assertEqual(initial_z, 100.0)
                self.assertEqual(guarded_z, best_z)
                self.assertIs(returned_result, result)
                self.assertEqual(engine.z_moves, [])

    def test_guard_compares_with_previous_z_not_scan_start(self):
        engine, _ = self.make_engine(initial_z=105.0, best_z=100.0)

        _, guarded_z, _ = engine.autofocus_w_raman(100.0, "point")

        self.assertEqual(guarded_z, 100.0)
        self.assertEqual(engine.z_moves, [])

    def test_nonfinite_focus_result_restores_previous_z(self):
        for best_z in (np.nan, np.inf, -np.inf):
            with self.subTest(best_z=best_z):
                engine, _ = self.make_engine(best_z=best_z)

                _, guarded_z, _ = engine.autofocus_w_raman(100.0, "point")

                self.assertEqual(guarded_z, 100.0)
                self.assertEqual(engine.z_moves, [100.0])


class TrenchTrackingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.original = np.asarray(
            [
                [0, 0, 0, 0, 10, 20],
                [0, 0, 0, 0, 30, 40],
            ],
            dtype=float,
        )
        self.event = SimpleNamespace(index={"p": 0, "t": 1})

    def test_safe_points_and_labels_replace_base_tracking_proposals(self):
        layer = _Layer(self.original)
        source = _PointSource(layer)
        safe_labels = np.zeros((4, 4), dtype=np.uint16)
        safe_labels[1:3, 1:3] = 7
        tracker = _FakeTrenchTracker(
            [[11, 21], [31, 41]],
            safe_labels,
        )
        engine_type = make_live_cell_engine_type(_StaleUpdateBase)

        with patch(
            "napari_raman_widget.live_cell_tracking.TrenchConstraintTracker",
            return_value=tracker,
        ), patch(
            "napari_raman_widget.new_cell_targets.find_new_cell_targets",
            return_value=(
                np.empty(0, dtype=np.int64),
                np.empty((0, 2), dtype=float),
            ),
        ) as find_new:
            engine = engine_type(
                [source],
                auto_add_new_cells=True,
                constrain_trench_tracking=True,
            )
            engine.update_aim(0, self.event, np.zeros((4, 4)))

        np.testing.assert_allclose(
            layer.data[:, -2:],
            [[11, 21], [31, 41]],
        )
        np.testing.assert_array_equal(engine._last_segments[0], safe_labels)
        np.testing.assert_array_equal(find_new.call_args.args[1], safe_labels)
        self.assertIsNotNone(tracker.call)
        targets = tracker.call["targets"]
        np.testing.assert_allclose(
            [target.previous_yx for target in targets],
            self.original[:, -2:],
        )
        np.testing.assert_allclose(
            [target.proposed_yx for target in targets],
            [[100, 200], [101, 201]],
        )

    def test_incompatible_geometry_holds_previous_points_and_safe_mask(self):
        layer = _Layer(self.original)
        source = _PointSource(layer)
        engine_type = make_live_cell_engine_type(_StaleUpdateBase)
        engine = engine_type(
            [source],
            constrain_trench_tracking=True,
        )

        engine.update_aim(0, self.event, np.zeros((4, 4)))

        np.testing.assert_allclose(layer.data, self.original)
        self.assertFalse(np.any(engine._last_segments[0]))
        self.assertEqual(engine._tracks[0].shape, (2, 4, 4))


class LiveCellTrackingTests(unittest.TestCase):
    def setUp(self):
        self.original = np.asarray(
            [
                [0, 0, 0, 0, 10, 20],
                [0, 0, 0, 0, 30, 40],
            ],
            dtype=float,
        )
        self.event = SimpleNamespace(index={"p": 0, "t": 1})

    def test_manual_add_survives_stale_tracking_writeback(self):
        layer = _Layer(self.original)
        source = _PointSource(layer)

        def add_point():
            layer.data = np.vstack(
                [layer.data, [0, 0, 0, 0, 50, 60]]
            )

        engine_type = make_live_cell_engine_type(_StaleUpdateBase)
        engine = engine_type([source], during_update=add_point)

        result = engine.update_aim(0, self.event, np.zeros((4, 4)))

        self.assertEqual(result, "updated")
        np.testing.assert_allclose(
            layer.data,
            [
                [0, 0, 0, 0, 100, 200],
                [0, 0, 0, 0, 101, 201],
                [0, 0, 0, 0, 50, 60],
            ],
        )

    def test_manual_delete_is_not_resurrected(self):
        layer = _Layer(self.original)
        source = _PointSource(layer)

        def delete_point():
            layer.data = layer.data[1:].copy()

        engine_type = make_live_cell_engine_type(_StaleUpdateBase)
        engine = engine_type([source], during_update=delete_point)

        engine.update_aim(0, self.event, np.zeros((4, 4)))

        np.testing.assert_allclose(
            layer.data,
            [[0, 0, 0, 0, 101, 201]],
        )

    def test_manual_move_wins_over_in_flight_tracking(self):
        layer = _Layer(self.original)
        source = _PointSource(layer)

        def move_point():
            layer.data[0, -2:] = [77, 88]

        engine_type = make_live_cell_engine_type(_StaleUpdateBase)
        engine = engine_type([source], during_update=move_point)

        engine.update_aim(0, self.event, np.zeros((4, 4)))

        np.testing.assert_allclose(layer.data[0, -2:], [77, 88])
        np.testing.assert_allclose(layer.data[1, -2:], [101, 201])

    def test_real_layer_is_restored_when_base_update_fails(self):
        layer = _Layer(self.original)
        source = _PointSource(layer)
        engine_type = make_live_cell_engine_type(_StaleUpdateBase)
        engine = engine_type([source], fail=True)

        with self.assertRaisesRegex(RuntimeError, "tracking failed"):
            engine.update_aim(0, self.event, np.zeros((4, 4)))

        self.assertIs(source._points, layer)
        np.testing.assert_allclose(layer.data, self.original)

    def test_record_raman_groups_each_cell_anchor_separately(self):
        layer = _Layer(
            [
                [0, 0, 0, 0, 20, 20],
                [0, 0, 0, 0, 80, 80],
            ]
        )
        source = _PointSource(layer, transformer=_TwoPointPattern())
        engine_type = make_live_cell_engine_type(_GroupingRecordBase)
        engine = engine_type([source], auto_add_new_cells=True)

        collected = engine.record_raman(self.event)

        self.assertTrue(engine.auto_add_new_cells)
        self.assertEqual(collected.shape, (4, 2))
        self.assertEqual(np.count_nonzero(collected[:, 0] < 0.5), 2)
        self.assertEqual(np.count_nonzero(collected[:, 0] >= 0.5), 2)
        self.assertIs(engine.aiming_sources[0], source)

    def test_record_raman_restores_sources_after_collection_failure(self):
        layer = _Layer([[0, 0, 0, 0, 20, 20]])
        source = _PointSource(layer, transformer=_TwoPointPattern())
        engine_type = make_live_cell_engine_type(_GroupingRecordBase)
        engine = engine_type([source], fail=True)

        with self.assertRaisesRegex(RuntimeError, "collection failed"):
            engine.record_raman(self.event)

        self.assertIs(engine.aiming_sources[0], source)

    def test_auto_add_appends_one_target_for_a_new_tracked_cell(self):
        layer = _Layer([[0, 0, 0, 0, 7, 7]])
        source = _PointSource(layer)
        engine_type = make_live_cell_engine_type(_AutoAddBase)
        engine = engine_type([source], auto_add_new_cells=True)

        engine.update_aim(0, self.event, np.zeros((30, 30)))

        self.assertEqual(len(layer.data), 2)
        new_pixel = np.floor(layer.data[1, -2:]).astype(int)
        self.assertEqual(engine.tracked[1, new_pixel[0], new_pixel[1]], 2)

    def test_manual_add_on_new_cell_prevents_automatic_duplicate(self):
        layer = _Layer([[0, 0, 0, 0, 7, 7]])
        source = _PointSource(layer)

        def manually_add_new_cell():
            layer.add([[0, 0, 0, 0, 21, 22]])

        engine_type = make_live_cell_engine_type(_AutoAddBase)
        engine = engine_type(
            [source],
            during_update=manually_add_new_cell,
            auto_add_new_cells=True,
        )

        engine.update_aim(0, self.event, np.zeros((30, 30)))

        self.assertEqual(len(layer.data), 2)
        np.testing.assert_allclose(layer.data[1, -2:], [21, 22])

    def test_raw_mask_track_fallback_does_not_treat_renumbering_as_new(self):
        layer = _Layer([[0, 0, 0, 0, 7, 7]])
        source = _PointSource(layer)
        engine_type = make_live_cell_engine_type(_AutoAddBase)
        engine = engine_type([source], auto_add_new_cells=True)

        previous = np.zeros((120, 120), dtype=np.uint16)
        previous[3:10, 3:10] = 10
        current = np.zeros_like(previous)
        current[4:11, 4:11] = 99
        current[90:97, 90:97] = 10
        engine._last_segments[0] = previous
        engine.current = current
        # This exact raw pair is the engine's btrack-failure fallback and must
        # not be interpreted by comparing its unrelated raw label numbers.
        engine.tracked = np.stack([previous, current])

        engine.update_aim(0, self.event, np.zeros((120, 120)))

        self.assertEqual(len(layer.data), 2)
        new_pixel = np.floor(layer.data[1, -2:]).astype(int)
        self.assertEqual(current[new_pixel[0], new_pixel[1]], 10)

    def test_auto_add_skips_t0_reused_masks_and_batch_mode(self):
        cases = (
            ({"p": 0, "t": 0}, False, False),
            ({"p": 0, "t": 1}, True, False),
            ({"p": 0, "t": 1}, False, True),
        )
        engine_type = make_live_cell_engine_type(_AutoAddBase)
        for index, use_same_img, batch in cases:
            with self.subTest(
                index=index,
                use_same_img=use_same_img,
                batch=batch,
            ):
                layer = _Layer([[0, 0, 0, 0, 7, 7]])
                source = _PointSource(layer)
                engine = engine_type(
                    [source],
                    batch=batch,
                    auto_add_new_cells=True,
                )
                event = SimpleNamespace(index=index)

                engine.update_aim(
                    0,
                    event,
                    np.zeros((30, 30)),
                    use_same_img=use_same_img,
                )

                self.assertEqual(len(layer.data), 1)


if __name__ == "__main__":
    unittest.main()
