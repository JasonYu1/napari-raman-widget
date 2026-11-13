import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from napari_raman_widget.live_cell_tracking import make_live_cell_engine_type
from napari_raman_widget.trench_segmentation import TrenchCellposeSegmenter
from napari_raman_widget.trench_tracking import DEFAULT_TRENCH_CONFIG


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


class _DummyCellposeSegmenter:
    def segment(self, image, **kwargs):
        del kwargs
        return np.zeros(np.asarray(image).shape, dtype=np.uint16)


class _MovementBase(_StaleUpdateBase):
    def __init__(self, sources, proposed_yx, **kwargs):
        super().__init__(sources, **kwargs)
        self.proposed_yx = np.asarray(proposed_yx, dtype=float)

    def update_aim(self, pos, event, img, use_same_img=False):
        del event, img, use_same_img
        self.during_update()
        for source in self.aiming_sources:
            updated = np.array(source._points.data, copy=True)
            updated[updated[:, source._pos_idx] == pos, -2:] = self.proposed_yx
            source._points.data = updated


class _SegmentingBase(_StaleUpdateBase):
    def __init__(self, sources, **kwargs):
        self._cellpose_segmenter = _DummyCellposeSegmenter()
        self._segment_crop = True
        super().__init__(sources, **kwargs)


class _SignalRecorder:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


class _EmittingBase(_MovementBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_events = SimpleNamespace(
            aimUpdated=_SignalRecorder(), otherSignal=_SignalRecorder(),
        )
        self.raman_events = self.original_events

    def update_aim(self, pos, event, img, use_same_img=False):
        previous = np.vstack([
            s._points.data[s._points.data[:, s._pos_idx] == pos, -2:]
            for s in self.aiming_sources
        ])
        super().update_aim(pos, event, img, use_same_img=use_same_img)
        self.raman_events.otherSignal.emit("unchanged")
        self.raman_events.aimUpdated.emit(
            event, img, self._last_segments[pos],
            np.atleast_2d(self.proposed_yx), previous,
        )
        if self.fail:
            raise RuntimeError("tracking failed after emission")


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


class _MaskValidatingRecordBase(_MovementBase):
    def __init__(self, *args, fail_record=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._segment_and_track = True
        self._image_x = self._image_y = 100
        self._scale = 1
        self._last_segments = {0: np.zeros((100, 100), dtype=np.uint16)}
        self.fail_record = fail_record
        self.collected = None

    def record_raman(self, event):
        if self.fail_record:
            raise RuntimeError("collection failed")
        collected = []
        for source in self.aiming_sources:
            pts = np.asarray(source.get_mda_points(event), dtype=float).reshape(-1, 2)
            if self._segment_and_track and len(pts):
                index = np.floor(pts * 100).astype(int)
                labels = self._last_segments[event.index['p']][index[:, 0], index[:, 1]]
                pts = pts[labels > 0]
            collected.append(pts)
        self.collected = np.vstack(collected) if collected else np.empty((0, 2))
        return self.collected


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
    def __init__(self, points_yx, labels, accepted=None):
        self.points_yx = np.asarray(points_yx, dtype=float)
        self.labels = np.asarray(labels, dtype=np.uint16)
        self.accepted = accepted
        self.call = None
        self.reset_count = 0

    def update(self, **kwargs):
        self.call = kwargs
        count = len(kwargs["targets"])
        return SimpleNamespace(
            points_yx=self.points_yx,
            labels=self.labels,
            accepted=(np.ones(count, dtype=bool) if self.accepted is None
                      else np.asarray(self.accepted, dtype=bool)),
            reasons=tuple("accepted" if value else "cell_area" for value in
                          ([True] * count if self.accepted is None else self.accepted)),
            geometry_valid=True,
            geometry_score=1.0,
        )

    def reset(self):
        self.reset_count += 1


class AutofocusSafeguardTests(unittest.TestCase):
    def make_engine(
        self,
        *,
        initial_z=100.0,
        best_z=100.0,
        maximum_shift=5.5,
    ):
        result = object()
        engine_type = make_live_cell_engine_type(_AutofocusBase)
        engine = engine_type(
            initial_z,
            best_z,
            result,
            maximum_autofocus_z_shift_um=maximum_shift,
        )
        return engine, result

    def test_shift_over_default_five_point_five_um_restores_previous_z(self):
        for best_z in (105.51, 94.49):
            with self.subTest(best_z=best_z):
                engine, result = self.make_engine(best_z=best_z)

                initial_z, guarded_z, returned_result = engine.autofocus_w_raman(
                    100.0,
                    "point",
                    t=1,
                    p=0,
                )

                self.assertEqual(initial_z, 100.0)
                self.assertEqual(guarded_z, 100.0)
                self.assertIs(returned_result, result)
                self.assertEqual(engine.z_moves, [100.0])
                self.assertEqual(
                    engine.autofocus_call,
                    (100.0, ("point",), {"t": 1, "p": 0}),
                )

    def test_first_timepoint_allows_large_shifts_at_every_position(self):
        for best_z in (112.0, 88.0):
            with self.subTest(best_z=best_z):
                engine, result = self.make_engine(best_z=best_z)
                for position in (0, 1, 2):
                    initial_z, guarded_z, returned_result = (
                        engine.autofocus_w_raman(
                            100.0, "point", 0, position
                        )
                    )
                    self.assertEqual(initial_z, 100.0)
                    self.assertEqual(guarded_z, best_z)
                    self.assertIs(returned_result, result)
                    self.assertEqual(engine.z_moves, [])
                    self.assertEqual(
                        engine.autofocus_call,
                        (100.0, ("point", 0, position), {}),
                    )

    def test_first_timepoint_keyword_index_also_exempts_configured_limit(self):
        engine, result = self.make_engine(best_z=112.0, maximum_shift=2.0)

        _, guarded_z, returned_result = engine.autofocus_w_raman(
            100.0, pt="point", t=0, p=3
        )

        self.assertEqual(guarded_z, 112.0)
        self.assertIs(returned_result, result)
        self.assertEqual(engine.z_moves, [])
        self.assertEqual(
            engine.autofocus_call,
            (100.0, (), {"pt": "point", "t": 0, "p": 3}),
        )

    def test_later_timepoint_restores_focus_established_at_first_timepoint(self):
        engine, _ = self.make_engine(best_z=112.0)
        _, first_z, _ = engine.autofocus_w_raman(100.0, "point", 0, 0)
        self.assertEqual(first_z, 112.0)

        engine.best_z = 120.0
        _, guarded_z, _ = engine.autofocus_w_raman(first_z, "point", 3, 0)

        self.assertEqual(guarded_z, first_z)
        self.assertEqual(engine.z_moves, [first_z])

    def test_missing_time_index_retains_jump_limit(self):
        engine, _ = self.make_engine(best_z=112.0)

        _, guarded_z, _ = engine.autofocus_w_raman(100.0, "point")

        self.assertEqual(guarded_z, 100.0)
        self.assertEqual(engine.z_moves, [100.0])

    def test_shift_of_exactly_default_five_point_five_um_is_accepted(self):
        for best_z in (105.5, 94.5):
            with self.subTest(best_z=best_z):
                engine, result = self.make_engine(best_z=best_z)

                initial_z, guarded_z, returned_result = (
                    engine.autofocus_w_raman(100.0, "point", 1, 0)
                )

                self.assertEqual(initial_z, 100.0)
                self.assertEqual(guarded_z, best_z)
                self.assertIs(returned_result, result)
                self.assertEqual(engine.z_moves, [])

    def test_configured_shift_limit_is_used(self):
        rejected, _ = self.make_engine(best_z=102.01, maximum_shift=2.0)
        _, rejected_z, _ = rejected.autofocus_w_raman(100.0, "point", 1, 0)
        self.assertEqual(rejected_z, 100.0)
        self.assertEqual(rejected.z_moves, [100.0])

        accepted, _ = self.make_engine(best_z=102.0, maximum_shift=2.0)
        _, accepted_z, _ = accepted.autofocus_w_raman(100.0, "point", 1, 0)
        self.assertEqual(accepted_z, 102.0)
        self.assertEqual(accepted.z_moves, [])
        self.assertEqual(accepted.maximum_autofocus_z_shift_um, 2.0)

    def test_invalid_shift_limit_is_rejected(self):
        for maximum_shift in (0, -1, np.nan, np.inf):
            with self.subTest(maximum_shift=maximum_shift):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    self.make_engine(maximum_shift=maximum_shift)

    def test_guard_compares_with_previous_z_not_scan_start(self):
        engine, _ = self.make_engine(initial_z=105.0, best_z=100.0)

        _, guarded_z, _ = engine.autofocus_w_raman(100.0, "point", 1, 0)

        self.assertEqual(guarded_z, 100.0)
        self.assertEqual(engine.z_moves, [])

    def test_nonfinite_focus_result_restores_previous_z(self):
        for t in (0, 1):
            for best_z in (np.nan, np.inf, -np.inf):
                with self.subTest(t=t, best_z=best_z):
                    engine, _ = self.make_engine(best_z=best_z)

                    _, guarded_z, _ = engine.autofocus_w_raman(
                        100.0, "point", t=t, p=0
                    )

                    self.assertEqual(guarded_z, 100.0)
                    self.assertEqual(engine.z_moves, [100.0])


class TrackingMovementSafeguardTests(unittest.TestCase):
    def make_engine(self, proposed, **kwargs):
        layer = _Layer([[0, 0, 0, 0, 100, 100]])
        source = _PointSource(layer)
        engine = make_live_cell_engine_type(_MovementBase)(
            [source], proposed, **kwargs
        )
        return engine, layer

    def test_default_limit_uses_full_resolution_euclidean_distance(self):
        for scale in (1, 4):
            for pillar_mode in (False, True):
                for proposed, expected in (
                    ([127, 136], [127, 136]),  # 27/36/45 right triangle
                    ([127, 136.01], [100, 100]),
                    ([100, 54.99], [100, 100]),
                ):
                    with self.subTest(scale=scale, pillars=pillar_mode, point=proposed):
                        engine, layer = self.make_engine(proposed)
                        engine._scale = scale
                        engine._suppress_pillars = pillar_mode
                        engine.update_aim(
                            0, SimpleNamespace(index={"p": 0, "t": 1}),
                            np.zeros((30, 30)),
                        )
                        self.assertEqual(engine.maximum_tracking_movement_px, 45.0)
                        np.testing.assert_allclose(layer.data[0, -2:], expected)

    def test_custom_limit_and_rejected_jump_do_not_move_next_reference(self):
        engine, layer = self.make_engine(
            [100, 112.6], maximum_tracking_movement_px=12.5
        )
        engine.update_aim(
            0, SimpleNamespace(index={"p": 0, "t": 0}), np.zeros((30, 30))
        )
        np.testing.assert_allclose(layer.data[0, -2:], [100, 100])
        engine.proposed_yx = np.asarray([100, 112.5])
        engine.update_aim(
            0, SimpleNamespace(index={"p": 0, "t": 1}), np.zeros((30, 30))
        )
        np.testing.assert_allclose(layer.data[0, -2:], [100, 112.5])

    def test_nonfinite_proposals_keep_previous_point(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                engine, layer = self.make_engine([100, value])
                engine.update_aim(
                    0, SimpleNamespace(index={"p": 0, "t": 1}), np.zeros((30, 30))
                )
                np.testing.assert_allclose(layer.data[0, -2:], [100, 100])

    def test_limit_is_applied_individually_at_current_position(self):
        layer = _Layer([
            [0, 0, 0, 0, 100, 100],
            [0, 0, 0, 0, 200, 200],
            [0, 1, 0, 0, 300, 300],
        ])
        engine = make_live_cell_engine_type(_MovementBase)(
            [_PointSource(layer)], [[110, 100], [250, 200]]
        )
        engine.update_aim(
            0, SimpleNamespace(index={"p": 0, "t": 1}), np.zeros((30, 30)),
            use_same_img=True,
        )
        np.testing.assert_allclose(
            layer.data[:, -2:], [[110, 100], [200, 200], [300, 300]]
        )

    def test_rejected_tracking_does_not_undo_manual_move(self):
        engine, layer = self.make_engine([200, 200])

        def move_point():
            layer.data[0, -2:] = [80, 90]

        engine.during_update = move_point
        engine.update_aim(
            0, SimpleNamespace(index={"p": 0, "t": 1}), np.zeros((30, 30))
        )
        np.testing.assert_allclose(layer.data[0, -2:], [80, 90])

    def test_invalid_limit_is_rejected(self):
        for value in (0, -1, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "maximum_tracking_movement_px must be finite and positive"
            ):
                self.make_engine([100, 100], maximum_tracking_movement_px=value)


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

    def test_image_tracking_skips_particle_assignment_and_publishes_final_result(self):
        layer = _Layer(self.original[:1])
        engine = make_live_cell_engine_type(_SegmentingBase)(
            [_PointSource(layer)], fail=True, constrain_trench_tracking=True,
        )
        engine._scale = 1
        engine._image_x = engine._image_y = 100
        engine._circle_center = (50, 50)
        engine.raman_events = SimpleNamespace(aimUpdated=_SignalRecorder())
        raw_mask = np.ones((100, 100), dtype=np.uint16)
        final_mask = raw_mask * 7
        tracker = _FakeTrenchTracker([[11, 21]], final_mask)
        tracker.config = DEFAULT_TRENCH_CONFIG
        engine._trench_tracker = tracker
        with patch.object(engine._trench_cellpose_segmenter, 'segment', return_value=raw_mask) as segment:
            engine.update_aim(0, self.event, np.zeros((100, 100)))
        segment.assert_called_once()
        self.assertFalse(segment.call_args.kwargs['crop'])
        np.testing.assert_allclose(layer.data[0, -2:], [11, 21])
        np.testing.assert_allclose(tracker.call['targets'][0].proposed_yx, [10, 20])
        calls = engine.raman_events.aimUpdated.calls
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0][2], final_mask)
        np.testing.assert_allclose(calls[0][3], [[11, 21]])

    def test_saved_notification_contains_final_mask_and_live_aim_once(self):
        layer = _Layer(self.original[:1])
        final_mask = np.full((4, 4), 7, dtype=np.uint16)
        tracker = _FakeTrenchTracker([[11, 21]], final_mask)
        with patch(
            "napari_raman_widget.live_cell_tracking.TrenchConstraintTracker",
            return_value=tracker,
        ):
            engine = make_live_cell_engine_type(_EmittingBase)(
                [_PointSource(layer)], [50, 50], constrain_trench_tracking=True,
            )
            engine.update_aim(0, self.event, np.zeros((4, 4)))
        calls = engine.original_events.aimUpdated.calls
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0][2], final_mask)
        np.testing.assert_allclose(calls[0][3], [[11, 21]])
        np.testing.assert_allclose(calls[0][4], [[10, 20]])
        self.assertIs(engine.raman_events, engine.original_events)
        self.assertEqual(engine.original_events.otherSignal.calls, [("unchanged",)])

    def test_notification_uses_manual_edit_that_wins_over_refinement(self):
        layer = _Layer(self.original[:1])
        tracker = _FakeTrenchTracker([[11, 21]], np.ones((4, 4), dtype=np.uint16))
        def move_point():
            layer.data[0, -2:] = [17, 23]
        with patch(
            "napari_raman_widget.live_cell_tracking.TrenchConstraintTracker",
            return_value=tracker,
        ):
            engine = make_live_cell_engine_type(_EmittingBase)(
                [_PointSource(layer)], [50, 50], during_update=move_point,
                constrain_trench_tracking=True,
            )
            engine.update_aim(0, self.event, np.zeros((4, 4)))
        np.testing.assert_allclose(engine.original_events.aimUpdated.calls[0][3], [[17, 23]])

    def test_failed_update_does_not_publish_proposal_and_restores_signaler(self):
        layer = _Layer(self.original[:1])
        engine = make_live_cell_engine_type(_EmittingBase)(
            [_PointSource(layer)], [50, 50], fail=True,
        )
        with self.assertRaisesRegex(RuntimeError, "after emission"):
            engine.update_aim(0, self.event, np.zeros((4, 4)))
        self.assertEqual(engine.original_events.aimUpdated.calls, [])
        self.assertIs(engine.raman_events, engine.original_events)
        np.testing.assert_allclose(layer.data, self.original[:1])

    def test_saved_notification_uses_ordinary_movement_guard_result(self):
        layer = _Layer(self.original[:1])
        engine = make_live_cell_engine_type(_EmittingBase)(
            [_PointSource(layer)], [100, 100],
        )
        engine.update_aim(0, self.event, np.zeros((4, 4)))
        np.testing.assert_allclose(engine.original_events.aimUpdated.calls[0][3], [[10, 20]])

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
            engine._circle_center = (10.0, 20.0)
            engine.update_aim(0, self.event, np.zeros((4, 4)))

        np.testing.assert_allclose(
            layer.data[:, -2:],
            [[11, 21], [31, 41]],
        )
        np.testing.assert_array_equal(engine._last_segments[0], safe_labels)
        np.testing.assert_array_equal(find_new.call_args.args[1], safe_labels)
        self.assertIsNotNone(tracker.call)
        self.assertEqual(tracker.call["trench_center_yx"], (10.0, 20.0))
        targets = tracker.call["targets"]
        np.testing.assert_allclose(
            [target.previous_yx for target in targets],
            self.original[:, -2:],
        )
        np.testing.assert_allclose(
            [target.proposed_yx for target in targets],
            [[100, 200], [101, 201]],
        )

    def test_trench_mode_replaces_upstream_circle_segmenter(self):
        layer = _Layer(self.original)
        source = _PointSource(layer)
        engine_type = make_live_cell_engine_type(_SegmentingBase)

        engine = engine_type([source], constrain_trench_tracking=True)

        self.assertIsInstance(
            engine._cellpose_segmenter,
            TrenchCellposeSegmenter,
        )
        self.assertIsInstance(
            engine._cellpose_segmenter.delegate,
            _DummyCellposeSegmenter,
        )
        self.assertFalse(engine._segment_crop)

    def test_non_trench_mode_keeps_upstream_circle_segmenter(self):
        layer = _Layer(self.original)
        source = _PointSource(layer)
        engine_type = make_live_cell_engine_type(_SegmentingBase)

        engine = engine_type([source], constrain_trench_tracking=False)

        self.assertIsInstance(
            engine._cellpose_segmenter,
            _DummyCellposeSegmenter,
        )
        self.assertTrue(engine._segment_crop)

    def test_universal_limit_configures_trench_tracking_and_segmentation(self):
        config = replace(DEFAULT_TRENCH_CONFIG, wall_clearance_px=6)
        engine = make_live_cell_engine_type(_SegmentingBase)(
            [_PointSource(_Layer(self.original))],
            constrain_trench_tracking=True,
            maximum_tracking_movement_px=12.5,
            trench_tracking_config=config,
        )

        self.assertEqual(engine._trench_tracker.config.maximum_step_px, 12.5)
        self.assertEqual(engine._trench_tracker.config.wall_clearance_px, 6)
        self.assertIs(engine._cellpose_segmenter.config, engine._trench_tracker.config)
        self.assertEqual(config.maximum_step_px, 45.0)

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


class TrackingSuccessLoggingTests(unittest.TestCase):
    def test_success_and_hold_are_forwarded_to_log_panel(self):
        messages = []
        layer = _Layer([[0, 0, 0, 0, 10, 20]])
        engine = make_live_cell_engine_type(_EmittingBase)(
            [_PointSource(layer)], [13, 24], tracking_log=messages.append,
        )
        event = SimpleNamespace(index={'p': 0, 't': 1})
        with patch('builtins.print'):
            engine.update_aim(0, event, np.zeros((4, 4)))
            engine.proposed_yx = np.asarray([100, 100])
            engine.update_aim(0, event, np.zeros((4, 4)))
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0],
            '[tracking] p=0, t=1: cells point 1 updated; moved 5.00 px\n')
        self.assertIn('[tracking safeguard]', messages[1])
        self.assertIn('held 1', messages[1])
        np.testing.assert_allclose(layer.data[0, -2:], [13, 24])

    def test_log_display_failure_does_not_interrupt_tracking(self):
        def unavailable(message):
            raise RuntimeError('log panel closed')
        layer = _Layer([[0, 0, 0, 0, 10, 20]])
        engine = make_live_cell_engine_type(_EmittingBase)(
            [_PointSource(layer)], [13, 24], tracking_log=unavailable,
        )
        with patch('builtins.print'):
            engine.update_aim(0, SimpleNamespace(index={'p': 0, 't': 1}),
                np.zeros((4, 4)))
        np.testing.assert_allclose(layer.data[0, -2:], [13, 24])
        self.assertEqual(len(engine.original_events.aimUpdated.calls), 1)

    def test_logs_applied_distance_in_full_resolution_including_stationary_cell(self):
        for trench in (False, True):
            for point, distance in (([13, 24], '5.00'), ([10, 20], '0.00')):
                with self.subTest(trench=trench, point=point):
                    layer = _Layer([[0, 2, 0, 0, 10, 20]])
                    tracker = _FakeTrenchTracker([point], np.ones((25, 25)))
                    with patch(
                        'napari_raman_widget.live_cell_tracking.TrenchConstraintTracker',
                        return_value=tracker,
                    ):
                        engine = make_live_cell_engine_type(_EmittingBase)(
                            [_PointSource(layer)], [40, 50] if trench else point,
                            constrain_trench_tracking=trench,
                        )
                    engine._scale = 4
                    engine._last_segments[2] = np.ones((25, 25), dtype=np.uint16)
                    messages = []
                    def record_print(message, **kwargs):
                        if str(message).startswith('[tracking]'):
                            # Notification and live layer must already contain
                            # the final result when the success line is printed.
                            np.testing.assert_allclose(layer.data[0, -2:], point)
                            np.testing.assert_allclose(
                                engine.original_events.aimUpdated.calls[-1][3], [point]
                            )
                            self.assertTrue(kwargs.get('flush'))
                            messages.append(message)
                    with patch('builtins.print', side_effect=record_print):
                        engine.update_aim(2, SimpleNamespace(index={'p': 2, 't': 7}),
                            np.zeros((100, 100)))
                    self.assertEqual(messages, [
                        f'[tracking] p=2, t=7: cells point 1 updated; moved {distance} px'
                    ])

    def test_only_accepted_target_is_reported_in_mixed_trench_update(self):
        layer = _Layer([[0, 0, 0, 0, 10, 20], [0, 0, 0, 0, 30, 40]])
        tracker = _FakeTrenchTracker([[13, 24], [30, 40]],
            np.ones((4, 4)), accepted=[True, False])
        with patch(
            'napari_raman_widget.live_cell_tracking.TrenchConstraintTracker',
            return_value=tracker,
        ):
            engine = make_live_cell_engine_type(_EmittingBase)(
                [_PointSource(layer)], [[40, 50], [60, 70]],
                constrain_trench_tracking=True,
            )
        with patch('builtins.print') as printed:
            engine.update_aim(0, SimpleNamespace(index={'p': 0, 't': 1}), np.zeros((4, 4)))
        messages = [call.args[0] for call in printed.call_args_list]
        self.assertTrue(any('held 1/2' in message for message in messages))
        self.assertEqual([m for m in messages if m.startswith('[tracking]')], [
            '[tracking] p=0, t=1: cells point 1 updated; moved 5.00 px'
        ])

    def test_manual_move_or_deletion_is_not_reported_as_tracking_success(self):
        for deleted in (False, True):
            with self.subTest(deleted=deleted):
                layer = _Layer([[0, 0, 0, 0, 10, 20]])
                def edit():
                    if deleted:
                        layer.data = np.empty((0, 6))
                    else:
                        layer.data[0, -2:] = [17, 23]
                engine = make_live_cell_engine_type(_EmittingBase)(
                    [_PointSource(layer)], [13, 24], during_update=edit,
                )
                with patch('builtins.print') as printed:
                    engine.update_aim(0, SimpleNamespace(index={'p': 0, 't': 1}),
                        np.zeros((4, 4)))
                self.assertFalse(any(str(call.args[0]).startswith('[tracking]')
                    for call in printed.call_args_list))

    def test_rejected_movement_and_tracking_exception_do_not_print_success(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                layer = _Layer([[0, 0, 0, 0, 10, 20]])
                engine = make_live_cell_engine_type(_EmittingBase)(
                    [_PointSource(layer)], [100, 100], fail=fail,
                )
                with patch('builtins.print') as printed:
                    if fail:
                        with self.assertRaises(RuntimeError):
                            engine.update_aim(0, SimpleNamespace(index={'p': 0, 't': 1}),
                                np.zeros((4, 4)))
                    else:
                        engine.update_aim(0, SimpleNamespace(index={'p': 0, 't': 1}),
                            np.zeros((4, 4)))
                self.assertFalse(any(str(call.args[0]).startswith('[tracking]')
                    for call in printed.call_args_list))


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

    def test_empty_mask_records_previous_anchor_and_pattern_instead_of_skipping(self):
        layer = _Layer([[0, 0, 0, 0, 10, 20]])
        source = _PointSource(layer, transformer=_TwoPointPattern())
        engine = make_live_cell_engine_type(_MaskValidatingRecordBase)(
            [source], [12, 23],
        )
        engine.update_aim(0, self.event, np.zeros((100, 100)))
        mask = engine._last_segments[0]
        points = engine.record_raman(self.event)
        np.testing.assert_allclose(points, [[0.1, 0.2], [0.11, 0.2]])
        np.testing.assert_allclose(layer.data[0, -2:], [10, 20])
        self.assertIs(engine._last_segments[0], mask)
        self.assertFalse(np.any(mask))
        self.assertTrue(engine._segment_and_track)
        self.assertIs(engine.aiming_sources[0], source)

    def test_mask_on_unrelated_cell_still_records_previous_target(self):
        layer = _Layer([[0, 0, 0, 0, 10, 20]])
        engine = make_live_cell_engine_type(_MaskValidatingRecordBase)(
            [_PointSource(layer)], [12, 23],
        )
        engine._last_segments[0][60:80, 60:80] = 9
        engine.update_aim(0, self.event, np.zeros((100, 100)))
        np.testing.assert_allclose(engine.record_raman(self.event), [[0.1, 0.2]])

    def test_valid_current_raman_target_keeps_normal_validation(self):
        layer = _Layer([[0, 0, 0, 0, 10, 20]])
        engine = make_live_cell_engine_type(_MaskValidatingRecordBase)(
            [_PointSource(layer)], [12, 23],
        )
        engine._last_segments[0][12, 23] = 7
        engine.update_aim(0, self.event, np.zeros((100, 100)))
        np.testing.assert_allclose(engine.record_raman(self.event), [[0.12, 0.23]])
        np.testing.assert_allclose(layer.data[0, -2:], [12, 23])

    def test_raman_fallback_respects_manual_move(self):
        layer = _Layer([[0, 0, 0, 0, 10, 20]])
        def move():
            layer.data[0, -2:] = [22, 33]
        engine = make_live_cell_engine_type(_MaskValidatingRecordBase)(
            [_PointSource(layer)], [12, 23], during_update=move,
        )
        engine.update_aim(0, self.event, np.zeros((100, 100)))
        np.testing.assert_allclose(engine.record_raman(self.event), [[0.22, 0.33]])
        np.testing.assert_allclose(layer.data[0, -2:], [22, 33])

    def test_raman_fallback_restores_validation_after_collection_exception(self):
        layer = _Layer([[0, 0, 0, 0, 10, 20]])
        source = _PointSource(layer)
        engine = make_live_cell_engine_type(_MaskValidatingRecordBase)(
            [source], [12, 23], fail_record=True,
        )
        engine.update_aim(0, self.event, np.zeros((100, 100)))
        with self.assertRaisesRegex(RuntimeError, "collection failed"):
            engine.record_raman(self.event)
        self.assertTrue(engine._segment_and_track)
        self.assertIs(engine.aiming_sources[0], source)

    def test_raman_fallback_republishes_the_actual_previous_aim(self):
        class EmittingRecordBase(_MaskValidatingRecordBase):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.raman_events = SimpleNamespace(aimUpdated=_SignalRecorder())

            def update_aim(self, pos, event, img, use_same_img=False):
                super().update_aim(pos, event, img, use_same_img)
                self.raman_events.aimUpdated.emit(
                    event, img, self._last_segments[pos],
                    np.array([[12, 23]]), np.array([[10, 20]]),
                )

        layer = _Layer([[0, 0, 0, 0, 10, 20]])
        engine = make_live_cell_engine_type(EmittingRecordBase)(
            [_PointSource(layer)], [12, 23],
        )
        engine.update_aim(0, self.event, np.zeros((100, 100)))
        engine.record_raman(self.event)
        calls = engine.raman_events.aimUpdated.calls
        self.assertEqual(len(calls), 2)
        np.testing.assert_allclose(calls[-1][3], [[10, 20]])
        np.testing.assert_allclose(engine.collected, [[0.10, 0.20]])
        self.assertFalse(np.any(calls[-1][2]))

    def test_manual_add_survives_stale_tracking_writeback(self):
        layer = _Layer(self.original)
        source = _PointSource(layer)

        def add_point():
            layer.data = np.vstack(
                [layer.data, [0, 0, 0, 0, 50, 60]]
            )

        engine_type = make_live_cell_engine_type(_StaleUpdateBase)
        engine = engine_type(
            [source], during_update=add_point, maximum_tracking_movement_px=500
        )

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
        engine = engine_type(
            [source], during_update=delete_point, maximum_tracking_movement_px=500
        )

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
        engine = engine_type(
            [source], during_update=move_point, maximum_tracking_movement_px=500
        )

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
