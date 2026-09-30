import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from napari_raman_widget.roi_sampling import sample_roi
from napari_raman_widget.spatial_mapping import (
    scan_image_reference,
    snapshot_scan_roi,
    snapshot_scan_shape,
)


class _ShapesLayer:
    def __init__(self, data, selected=(), mode="select", shape_type=None,
                 matrix=None, translate=(0, 0)):
        self.data = data
        self.selected_data = set(selected)
        self.mode = mode
        self.shape_type = ["rectangle"] * len(data) if shape_type is None else shape_type
        self.matrix = np.eye(2) if matrix is None else np.asarray(matrix)
        self.translate = np.asarray(translate)

    def data_to_world(self, point):
        point = np.asarray(point)
        if point.shape != (2,):
            raise AssertionError("Transforms must be called one 2D point at a time")
        return self.matrix @ point + self.translate


class _ImageLayer:
    def __init__(self, shape=(8, 10), *, name="Camera", rgb=False, visible=True,
                 matrix=None, translate=(0, 0), multiscale=False):
        self.name = name
        self.data = np.zeros(shape, dtype=np.uint8)
        self.rgb = rgb
        self.ndim = len(shape) - int(rgb)
        self.visible = visible
        self.multiscale = multiscale
        self.matrix = np.eye(2) if matrix is None else np.asarray(matrix)
        self.translate = np.asarray(translate)

    def data_to_world(self, point):
        point = np.asarray(point)
        if point.shape != (2,):
            raise AssertionError("Transforms must be called one 2D point at a time")
        return self.matrix @ point + self.translate

    def world_to_data(self, point):
        point = np.asarray(point)
        if point.shape != (2,):
            raise AssertionError("Transforms must be called one 2D point at a time")
        return np.linalg.solve(self.matrix, point - self.translate)


class SpatialMappingTests(unittest.TestCase):
    def test_both_widgets_stabilize_shape_before_scanning(self):
        package = Path(__file__).resolve().parents[1] / "napari_raman_widget"
        for filename in ("hardware_widget.py", "demo_widget.py"):
            with self.subTest(filename=filename):
                source = (package / filename).read_text(encoding="utf-8")
                self.assertIn("return start_grid_scan(self)", source)
        workflow = (package / "scan_workflows.py").read_text(encoding="utf-8")
        self.assertRegex(workflow, r"snapshot_scan_roi\(shapes,\s*image_layer=")
        self.assertIn("snapshot_scan_shape(shapes)\n", workflow)

    def test_preview_copy_does_not_change_layer_interaction(self):
        shape = np.array([[0, 0], [0, 3], [2, 3], [2, 0]])
        layer = _ShapesLayer([shape], selected={0}, mode="select")
        snapshot = snapshot_scan_shape(layer, finish_interaction=False)
        self.assertEqual(layer.mode, "select")
        self.assertEqual(layer.selected_data, {0})
        shape[0, 0] = 99
        self.assertEqual(snapshot[0, 0], 0)

    def test_snapshots_selected_shape_then_makes_layer_noninteractive(self):
        first = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        second = np.array([[2, 3], [2, 8], [7, 8], [7, 3]])
        layer = _ShapesLayer([first, second], selected={1})

        snapshot = snapshot_scan_shape(layer)

        np.testing.assert_array_equal(snapshot, second)
        self.assertEqual(layer.mode, "pan_zoom")
        self.assertEqual(layer.selected_data, set())

        second[0, 0] = 99
        self.assertEqual(snapshot[0, 0], 2)

    def test_uses_latest_shape_when_none_is_selected(self):
        first = np.array([[0, 0], [0, 1]])
        second = np.array([[2, 3], [7, 8]])

        snapshot = snapshot_scan_shape(_ShapesLayer([first, second]))

        np.testing.assert_array_equal(snapshot, second)

    def test_rejects_empty_shapes_layer(self):
        with self.assertRaisesRegex(RuntimeError, "Shapes layer is empty"):
            snapshot_scan_shape(_ShapesLayer([]))

    def test_rejects_shape_without_two_coordinates(self):
        layer = _ShapesLayer([np.array([[1.0]])])

        with self.assertRaisesRegex(RuntimeError, "usable coordinates"):
            snapshot_scan_shape(layer)


class ScanROISnapshotTests(unittest.TestCase):
    def setUp(self):
        self.rectangle = np.array([[1, 2], [1, 6], [4, 6], [4, 2]], dtype=float)

    def test_transformed_shape_maps_through_world_into_reference_pixels(self):
        shape_matrix = np.array([[0.0, -2.0], [3.0, 1.0]])
        image_matrix = np.array([[2.0, 0.5], [0.25, 1.5]])
        shape_offset = np.array([15.0, -7.0])
        image_offset = np.array([-5.0, 4.0])
        layer = _ShapesLayer(
            [self.rectangle], selected={0}, matrix=shape_matrix,
            translate=shape_offset,
        )
        reference = _ImageLayer(
            name="Live camera", matrix=image_matrix, translate=image_offset,
        )

        snapshot = snapshot_scan_roi(layer, image_layer=reference)

        expected = np.linalg.solve(
            image_matrix,
            shape_matrix @ self.rectangle.T + (shape_offset - image_offset)[:, None],
        ).T
        np.testing.assert_allclose(snapshot.vertices_yx, expected)
        self.assertEqual(snapshot.image_layer_name, "Live camera")
        self.assertEqual(snapshot.shape_type, "rectangle")
        self.assertEqual(snapshot.shape_index, 0)
        self.assertEqual(layer.mode, "select")
        self.assertEqual(layer.selected_data, {0})

    def test_roi_type_is_retained_without_mutating_layer_or_sharing_storage(self):
        for kind in ("rectangle", "ellipse", "polygon"):
            with self.subTest(kind=kind):
                original = self.rectangle.copy()
                layer = _ShapesLayer([original], selected={0}, shape_type=[kind])
                snapshot = snapshot_scan_roi(layer)
                self.assertEqual(snapshot.shape_type, kind)
                self.assertEqual(layer.mode, "select")
                self.assertEqual(layer.selected_data, {0})
                self.assertEqual(layer.shape_type, [kind])
                original[0] = [99, 99]
                np.testing.assert_array_equal(snapshot.vertices_yx, self.rectangle)
                with self.assertRaises(TypeError):
                    snapshot.vertices_yx[0][0] = 5

    def test_latest_shape_or_exact_single_selection_is_used(self):
        second = self.rectangle + 10
        layer = _ShapesLayer(
            [self.rectangle, second], shape_type=["rectangle", "ellipse"]
        )
        latest = snapshot_scan_roi(layer)
        self.assertEqual(latest.shape_index, 1)
        self.assertEqual(latest.shape_type, "ellipse")
        np.testing.assert_array_equal(latest.vertices_yx, second)
        layer.selected_data = {0}
        selected = snapshot_scan_roi(layer)
        self.assertEqual(selected.shape_index, 0)
        self.assertEqual(selected.shape_type, "rectangle")
        np.testing.assert_array_equal(selected.vertices_yx, self.rectangle)

    def test_scalar_and_enum_like_shape_types_are_supported(self):
        for types in ("ellipse", [SimpleNamespace(value="ELLIPSE")]):
            with self.subTest(types=types):
                layer = _ShapesLayer([self.rectangle], shape_type=types)
                self.assertEqual(snapshot_scan_roi(layer).shape_type, "ellipse")

    def test_multiple_selected_shapes_are_rejected_without_clearing_selection(self):
        layer = _ShapesLayer([self.rectangle, self.rectangle + 1], selected={0, 1})
        with self.assertRaisesRegex(ValueError, "only one shape"):
            snapshot_scan_roi(layer)
        self.assertEqual(layer.selected_data, {0, 1})
        self.assertEqual(layer.mode, "select")

    def test_empty_and_three_dimensional_shapes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            snapshot_scan_roi(_ShapesLayer([]))
        vertices_3d = np.column_stack((np.ones(4), self.rectangle))
        with self.assertRaisesRegex(ValueError, "2D Shapes"):
            snapshot_scan_roi(_ShapesLayer([vertices_3d]))

    def test_open_line_and_path_are_rejected_before_sampling(self):
        for kind in ("line", "path"):
            with self.subTest(kind=kind):
                layer = _ShapesLayer([self.rectangle], shape_type=[kind])
                with self.assertRaisesRegex(ValueError, "closed rectangle, ellipse, or polygon"):
                    snapshot_scan_roi(layer)

    def test_degenerate_closed_snapshot_is_rejected_by_geometry_sampling(self):
        for kind in ("rectangle", "ellipse", "polygon"):
            with self.subTest(kind=kind):
                vertices = np.array([[1, 1], [1, 2], [1, 3], [1, 4]], dtype=float)
                roi = snapshot_scan_roi(_ShapesLayer([vertices], shape_type=[kind]))
                with self.assertRaisesRegex(ValueError, "degenerate"):
                    sample_roi(roi.vertices_yx, roi.shape_type, total_points=10)

    def test_nonidentity_shape_transform_requires_a_reference_image(self):
        for matrix, translate in (
            (np.diag([2.0, 3.0]), (0, 0)),
            (np.eye(2), (10, 20)),
            (np.array([[1.0, 0.3], [0.0, 1.0]]), (0, 0)),
            (np.array([[0.0, -1.0], [1.0, 0.0]]), (0, 0)),
        ):
            with self.subTest(matrix=matrix, translate=translate):
                layer = _ShapesLayer([self.rectangle], matrix=matrix, translate=translate)
                with self.assertRaisesRegex(ValueError, "matching 2D camera image"):
                    snapshot_scan_roi(layer)

    def test_non_2d_reference_and_nonfinite_mapped_coordinates_are_rejected(self):
        layer = _ShapesLayer([self.rectangle])
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            snapshot_scan_roi(layer, image_layer=_ImageLayer(shape=(2, 8, 10)))
        reference = _ImageLayer()
        reference.world_to_data = lambda point: (np.nan, 0.0)
        with self.assertRaisesRegex(ValueError, "finite 2D image-pixel"):
            snapshot_scan_roi(layer, image_layer=reference)


class ScanImageReferenceTests(unittest.TestCase):
    def setUp(self):
        # Avoid napari's import-time resource/cache writes in these pure tests.
        fake_layers = ModuleType("napari.layers")
        fake_layers.Image = _ImageLayer
        patcher = patch.dict(sys.modules, {"napari.layers": fake_layers})
        patcher.start()
        self.addCleanup(patcher.stop)

    def reference(self, layers):
        return scan_image_reference(SimpleNamespace(layers=layers), width=10, height=8)

    def test_no_visible_images_uses_identity_pixel_convention(self):
        self.assertIsNone(self.reference([]))
        self.assertIsNone(self.reference([_ImageLayer(visible=False), object()]))

    def test_matching_camera_dimensions_accepts_grayscale_rgb_and_rgba(self):
        for shape, rgb in (((8, 10), False), ((8, 10, 3), True), ((8, 10, 4), True)):
            with self.subTest(shape=shape):
                camera = _ImageLayer(shape=shape, rgb=rgb)
                self.assertIs(self.reference([camera]), camera)

    def test_incompatible_visible_image_dimensions_are_rejected(self):
        for image in (
            _ImageLayer(shape=(10, 8)),
            _ImageLayer(shape=(8, 11)),
            _ImageLayer(shape=(2, 8, 10)),
            _ImageLayer(multiscale=True),
        ):
            with self.subTest(shape=image.data.shape, multiscale=image.multiscale):
                with self.assertRaisesRegex(ValueError, "matching the camera dimensions"):
                    self.reference([image])

    def test_multiple_images_with_identical_affine_geometry_are_unambiguous(self):
        matrix = [[0.0, -2.0], [3.0, 0.5]]
        first = _ImageLayer(matrix=matrix, translate=(100, 200), name="BF")
        second = _ImageLayer(matrix=matrix, translate=(100, 200), name="Fluorescence")
        self.assertIs(self.reference([first, second]), first)

    def test_mismatched_full_affine_geometry_is_rejected_not_silently_chosen(self):
        reference = _ImageLayer()
        for other in (
            _ImageLayer(translate=(0, 5)),
            _ImageLayer(matrix=[[2, 0], [0, 1]]),
            _ImageLayer(matrix=[[1, 0.5], [0, 1]]),
            _ImageLayer(matrix=[[0, -1], [1, 0]]),
        ):
            with self.subTest(matrix=other.matrix, translate=other.translate):
                with self.assertRaisesRegex(ValueError, "different transforms"):
                    self.reference([reference, other])

    def test_hidden_or_differently_sized_images_do_not_make_reference_ambiguous(self):
        camera = _ImageLayer()
        hidden = _ImageLayer(visible=False, translate=(9, 9))
        thumbnail = _ImageLayer(shape=(4, 5), translate=(20, 20))
        self.assertIs(self.reference([hidden, thumbnail, object(), camera]), camera)


if __name__ == "__main__":
    unittest.main()
