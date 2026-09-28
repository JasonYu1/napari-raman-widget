"""Regression tests for widget aiming-pattern construction."""

from __future__ import annotations

import unittest

import numpy as np

from napari_raman_widget.aiming_patterns import make_point_transformer


class AimingPatternTests(unittest.TestCase):
    def test_single_hardware_pattern_preserves_exact_selected_points(self):
        selected = np.array([[0.317, 0.684], [0.811, 0.129]])

        for shape in ("Square", "Circle"):
            with self.subTest(shape=shape):
                transformer = make_point_transformer(shape, 30.0, 1, 1344)
                transformed = transformer.transform(selected)

                np.testing.assert_array_equal(transformed, selected)
                self.assertEqual(transformer.multiplier, 1)
                self.assertIsNot(transformed, selected)

    def test_nonpositive_count_is_clamped_to_one_exact_point(self):
        selected = np.array([[0.25, 0.75]])

        transformer = make_point_transformer("Square", 30.0, 0, 1344)

        np.testing.assert_array_equal(transformer.transform(selected), selected)
        self.assertEqual(transformer.multiplier, 1)

    def test_single_pattern_rejects_malformed_coordinates(self):
        transformer = make_point_transformer("Circle", 30.0, 1, 1344)

        with self.assertRaisesRegex(ValueError, "shape"):
            transformer.transform(np.array([0.25, 0.75]))


if __name__ == "__main__":
    unittest.main()
