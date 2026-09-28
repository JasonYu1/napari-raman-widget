import unittest
from pathlib import Path

import numpy as np

from napari_raman_widget.spatial_mapping import snapshot_scan_shape


class _ShapesLayer:
    def __init__(self, data, selected=(), mode="select"):
        self.data = data
        self.selected_data = set(selected)
        self.mode = mode


class SpatialMappingTests(unittest.TestCase):
    def test_both_widgets_stabilize_shape_before_scanning(self):
        package = Path(__file__).resolve().parents[1] / "napari_raman_widget"
        for filename in ("hardware_widget.py", "demo_widget.py"):
            with self.subTest(filename=filename):
                source = (package / filename).read_text(encoding="utf-8")
                self.assertIn("shape0 = snapshot_scan_shape(shapes)", source)

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


if __name__ == "__main__":
    unittest.main()
