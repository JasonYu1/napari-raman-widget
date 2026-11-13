"""Tests for one-shot cell-point refinement."""

from __future__ import annotations

import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

_REFINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "napari_raman_widget"
    / "selection"
    / "refine.py"
)
_SPEC = spec_from_file_location("_selection_refine", _REFINE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_REFINE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REFINE)
refine_cell_source_points = _REFINE.refine_cell_source_points
refine_points_to_label_centers = _REFINE.refine_points_to_label_centers


class _FakePoints:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data


class _FakeSource:
    def __init__(self, data: np.ndarray) -> None:
        self._points = _FakePoints(data)


class _FakeCore:
    def __init__(self, image_shape=(12, 12)) -> None:
        self.xy = (1.0, 2.0)
        self.channel = "RM"
        self.image_shape = image_shape
        self.xy_history: list[tuple[float, float]] = []
        self.channel_history: list[str] = []
        self.snap_count = 0
        self.stop_count = 0
        self.image = np.zeros(image_shape, dtype=np.uint16)

    def getXYPosition(self):
        return self.xy

    def setXYPosition(self, x, y):
        self.xy = (float(x), float(y))
        self.xy_history.append(self.xy)

    def getCurrentConfig(self, group):
        return self.channel

    def setConfig(self, group, channel):
        self.channel = channel
        self.channel_history.append(channel)

    def waitForSystem(self):
        return None

    def stopSequenceAcquisition(self):
        self.stop_count += 1

    def snapImage(self):
        self.snap_count += 1

    def getImage(self):
        return np.array(self.image, copy=True)


class SelectionRefineTests(unittest.TestCase):
    def test_refinement_sends_only_the_circle_roi_to_segmenter(self) -> None:
        source = _FakeSource(
            np.array([[0, 0, 0, 0, 50, 50]], dtype=float)
        )
        sequence = SimpleNamespace(
            stage_positions=[SimpleNamespace(x=10.0, y=20.0)]
        )
        core = _FakeCore(image_shape=(100, 100))
        segment_shapes = []

        def segmenter(image, **kwargs):
            segment_shapes.append(image.shape)
            mask = np.zeros(image.shape, dtype=np.int32)
            mask[9, 9] = 1
            return mask

        refine_cell_source_points(
            core,
            source,
            sequence,
            center_yx=(50.0, 50.0),
            radius=5.0,
            stage_settle_time=0,
            segmentation_scale=4,
            segmenter=segmenter,
            show_progress=False,
        )

        self.assertEqual(segment_shapes, [(19, 19)])
        np.testing.assert_allclose(source._points.data[0, -2:], [50, 50])

    def test_points_snap_to_containing_or_nearby_labels(self) -> None:
        mask = np.zeros((20, 20), dtype=np.int32)
        mask[2:6, 4:8] = 1
        mask[12:16, 13:17] = 2
        points = np.array(
            [
                [3.0, 5.0],
                [10.0, 12.0],
                [0.0, 19.0],
            ]
        )

        refined, matched = refine_points_to_label_centers(
            points,
            mask,
            max_distance=5.0,
        )

        np.testing.assert_allclose(refined[0], [3.5, 5.5])
        np.testing.assert_allclose(refined[1], [13.5, 14.5])
        np.testing.assert_allclose(refined[2], points[2])
        np.testing.assert_array_equal(matched, [True, True, False])

    def test_refinement_updates_only_yx_and_restores_hardware_state(self) -> None:
        data = np.array(
            [
                [0, 0, 0, 0, 2, 2],
                [0, 1, 0, 0, 3, 3],
            ],
            dtype=float,
        )
        source = _FakeSource(data.copy())
        sequence = SimpleNamespace(
            stage_positions=[
                SimpleNamespace(x=10.0, y=20.0),
                SimpleNamespace(x=10.0, y=20.0),
            ]
        )
        core = _FakeCore()
        segment_scales = []

        def segmenter(image, **kwargs):
            segment_scales.append(kwargs["scale"])
            # Mimic scale=4 output; the helper must restore image coordinates.
            mask = np.zeros((3, 3), dtype=np.int32)
            mask[1, 1] = 1
            return mask

        result = refine_cell_source_points(
            core,
            source,
            sequence,
            center_yx=(6.0, 6.0),
            radius=6.0,
            stage_settle_time=0,
            segmenter=segmenter,
            show_progress=False,
        )

        np.testing.assert_allclose(source._points.data[:, :4], data[:, :4])
        np.testing.assert_allclose(
            source._points.data[:, -2:],
            [[5.5, 5.5], [5.5, 5.5]],
        )
        self.assertEqual(result, {"total": 2, "matched": 2, "moved": 2, "unmatched": 0})
        self.assertEqual(core.snap_count, 1)
        self.assertEqual(core.stop_count, 1)
        self.assertEqual(segment_scales, [4])
        self.assertEqual(core.channel_history, ["BF", "RM"])
        self.assertEqual(core.xy_history[-1], (1.0, 2.0))

    def test_refinement_removes_pillar_shaped_cellpose_label(self) -> None:
        source = _FakeSource(
            np.array([[0, 0, 0, 0, 6, 6]], dtype=float)
        )
        sequence = SimpleNamespace(
            stage_positions=[SimpleNamespace(x=10.0, y=20.0)]
        )
        core = _FakeCore()
        class ShapeFilter:
            def filter(self, labels, *, image_shape):
                self.image_shape = image_shape
                filtered = labels.copy()
                removed = filtered == 1
                filtered[removed] = 0
                return SimpleNamespace(
                    filtered_labels=filtered,
                    removed_mask=removed,
                    removed_labels=(1,),
                    measurements=(),
                )

        def segmenter(image, **kwargs):
            mask = np.zeros(image.shape, dtype=np.int32)
            mask[4:9, 4:9] = 1
            return mask

        result = refine_cell_source_points(
            core,
            source,
            sequence,
            center_yx=(6.0, 6.0),
            radius=6.0,
            stage_settle_time=0,
            segmentation_scale=1,
            segmenter=segmenter,
            show_progress=False,
            suppress_pillars=True,
            pillar_shape_filter=ShapeFilter(),
        )

        np.testing.assert_allclose(source._points.data[0, -2:], [6, 6])
        self.assertEqual(result["matched"], 0)
        self.assertEqual(result["moved"], 0)

    def test_trench_refinement_rejects_extra_move_outside_safe_label(self) -> None:
        source = _FakeSource(
            np.array([[0, 0, 0, 0, 5, 5]], dtype=float)
        )
        sequence = SimpleNamespace(
            stage_positions=[SimpleNamespace(x=10.0, y=20.0)]
        )
        core = _FakeCore()
        raw = np.zeros(core.image_shape, dtype=np.int32)
        raw[4:9, 4:9] = 7
        safe = raw.copy()

        def segmenter(image, **kwargs):
            del image, kwargs
            return raw

        trench_result = SimpleNamespace(
            filtered_labels=safe,
            geometry_valid=True,
            geometry_score=1.0,
        )
        geometry = SimpleNamespace(
            geometry_valid=True,
            geometry_score=1.0,
            safe_union=np.ones(core.image_shape, dtype=bool),
            pose="registered-pose",
        )
        with patch.object(
            _REFINE,
            "registered_trench_geometry",
            return_value=geometry,
        ), patch.object(
            _REFINE,
            "filter_labels_to_trenches",
            return_value=trench_result,
        ) as trench_filter:
            result = refine_cell_source_points(
                core,
                source,
                sequence,
                center_yx=(6.0, 6.0),
                radius=6.0,
                stage_settle_time=0,
                segmentation_scale=1,
                segmenter=segmenter,
                point_refiner=lambda points: np.zeros_like(points),
                constrain_trenches=True,
                show_progress=False,
            )

        np.testing.assert_allclose(source._points.data[0, -2:], [6, 6])
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["moved"], 1)
        np.testing.assert_array_equal(
            trench_filter.call_args.args[0],
            np.zeros(core.image_shape, dtype=np.uint16),
        )
        np.testing.assert_array_equal(trench_filter.call_args.args[1], raw)
        self.assertEqual(
            trench_filter.call_args.kwargs["pose_override"],
            "registered-pose",
        )

    def test_trench_refinement_replaces_circle_with_registered_input(self) -> None:
        source = _FakeSource(
            np.array([[0, 0, 0, 0, 6, 6]], dtype=float)
        )
        sequence = SimpleNamespace(
            stage_positions=[SimpleNamespace(x=10.0, y=20.0)]
        )
        core = _FakeCore()
        core.image = np.arange(144, dtype=np.uint16).reshape(12, 12)
        safe_union = np.zeros(core.image_shape, dtype=bool)
        safe_union[3:10, 4:9] = True
        raw = np.zeros(core.image_shape, dtype=np.int32)
        raw[4:9, 4:9] = 3
        received = {}

        def segmenter(image, **kwargs):
            received["image"] = np.array(image, copy=True)
            received["kwargs"] = dict(kwargs)
            return raw

        geometry = SimpleNamespace(
            geometry_valid=True,
            geometry_score=0.9,
            safe_union=safe_union,
            pose="registered-pose",
        )
        trench_result = SimpleNamespace(
            filtered_labels=raw,
            geometry_valid=True,
            geometry_score=0.9,
        )
        with patch.object(
            _REFINE,
            "registered_trench_geometry",
            return_value=geometry,
        ), patch.object(
            _REFINE,
            "filter_labels_to_trenches",
            return_value=trench_result,
        ) as trench_filter:
            result = refine_cell_source_points(
                core,
                source,
                sequence,
                center_yx=(6.0, 6.0),
                radius=2.0,
                stage_settle_time=0,
                segmentation_scale=1,
                cellpose_upsample=3,
                segmenter=segmenter,
                constrain_trenches=True,
                show_progress=False,
            )

        self.assertEqual(received["image"].shape, (36, 36))
        enlarged_safe = np.repeat(
            np.repeat(safe_union, 3, axis=0),
            3,
            axis=1,
        )
        enlarged_image = np.repeat(
            np.repeat(core.image, 3, axis=0),
            3,
            axis=1,
        )
        np.testing.assert_array_equal(
            received["image"][enlarged_safe],
            enlarged_image[enlarged_safe],
        )
        self.assertTrue(
            np.all(
                received["image"][~enlarged_safe]
                == np.median(core.image[safe_union])
            )
        )
        self.assertFalse(received["kwargs"]["crop"])
        self.assertIsNone(received["kwargs"]["circle_center"])
        self.assertTrue(np.isinf(received["kwargs"]["circle_radius"]))
        self.assertEqual(result["matched"], 1)
        self.assertEqual(
            trench_filter.call_args.kwargs["pose_override"],
            "registered-pose",
        )

    def test_trench_refinement_fails_closed_on_bad_geometry(self) -> None:
        original = np.array([[0, 0, 0, 0, 6, 6]], dtype=float)
        source = _FakeSource(original.copy())
        sequence = SimpleNamespace(
            stage_positions=[SimpleNamespace(x=10.0, y=20.0)]
        )
        core = _FakeCore()

        segmenter = Mock(
            return_value=np.zeros(core.image_shape, dtype=np.uint32)
        )
        with patch.object(
            _REFINE,
            "registered_trench_geometry",
            return_value=SimpleNamespace(
                geometry_valid=False,
                geometry_score=0.2,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "not recognized"):
                refine_cell_source_points(
                    core,
                    source,
                    sequence,
                    center_yx=(6.0, 6.0),
                    radius=6.0,
                    stage_settle_time=0,
                    segmentation_scale=1,
                    segmenter=segmenter,
                    constrain_trenches=True,
                    show_progress=False,
                )

        np.testing.assert_array_equal(source._points.data, original)
        segmenter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
