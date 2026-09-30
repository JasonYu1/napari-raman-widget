"""Pure geometry tests for complete, equally spaced ROI sampling grids."""

import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.path import Path

from napari_raman_widget import roi_sampling
from napari_raman_widget.roi_sampling import (
    MAX_ROI_POINTS,
    roi_outline,
    sample_roi,
    sample_roi_plan,
)


class RoiSamplingTests(unittest.TestCase):
    def assert_inside_polygon(self, points, vertices):
        vertices = np.asarray(vertices)
        inside = Path(vertices).contains_points(points)
        for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
            edge = end - start
            fraction = np.clip((points - start) @ edge / (edge @ edge), 0, 1)
            inside |= (
                np.linalg.norm(points - (start + fraction[:, None] * edge), axis=1)
                < 1e-10
            )
        self.assertTrue(inside.all())

    def assert_complete_lattice(self, points, spacing, vertices, kind):
        self.assertGreaterEqual(len(points), 2)
        self.assertLessEqual(len(points), MAX_ROI_POINTS)
        self.assertEqual(len(np.unique(points, axis=0)), len(points))
        explicit = sample_roi(vertices, kind, mode="spacing", spacing_px=spacing)
        np.testing.assert_array_equal(points, explicit)
        origin = points.min(axis=0)
        indices = (points - origin) / spacing
        np.testing.assert_allclose(indices, np.rint(indices), atol=1e-10)

    def ellipse_corners(self):
        center = np.array([30.0, 50.0])
        a = np.array([8.0, 6.0])
        b = np.array([-2.0, 5.0])  # Deliberate shear, not perpendicular to a.
        corners = np.array(
            [center - a - b, center + a - b, center + a + b, center - a + b]
        )
        return corners, center, np.column_stack((a, b))

    def test_count_is_complete_lattice_deterministic_and_inside_concave_polygon(self):
        polygon = np.array(
            [[0, 0], [0, 8], [2, 8], [2, 2], [8, 2], [8, 0]], dtype=float
        )
        for count in (2, 3, 7, 401):
            with self.subTest(count=count):
                points, spacing = sample_roi_plan(
                    polygon, "polygon", total_points=count
                )
                self.assert_complete_lattice(points, spacing, polygon, "polygon")
                self.assert_inside_polygon(points, polygon)
                np.testing.assert_array_equal(
                    points, sample_roi(polygon, "polygon", total_points=count)
                )

    def test_count_grid_covers_roi_not_polygon_bounding_box(self):
        polygon = [[0, 0], [0, 8], [2, 8], [2, 2], [8, 2], [8, 0]]
        points, spacing = sample_roi_plan(polygon, "polygon", total_points=2800)
        self.assert_complete_lattice(points, spacing, polygon, "polygon")
        self.assertLess(abs(len(points) - 2800), 150)
        self.assertFalse(np.any((points[:, 0] > 2) & (points[:, 1] > 2)))

    def test_few_points_cover_high_vertex_polygon_with_full_lattice(self):
        angles = np.arange(1000) * 2 * np.pi / 1000
        polygon = np.column_stack((np.cos(angles), np.sin(angles)))
        for count in (2, 3, 20):
            points, spacing = sample_roi_plan(polygon, "polygon", total_points=count)
            self.assert_complete_lattice(points, spacing, polygon, "polygon")
            self.assert_inside_polygon(points, polygon)
            if count == 20:
                self.assertTrue(np.all(np.ptp(points, axis=0) > 1.3))
                quadrants = (points[:, 0] >= 0).astype(int) * 2 + (points[:, 1] >= 0)
                self.assertTrue(np.all(np.bincount(quadrants, minlength=4) >= 2))

    def test_reversed_winding_and_repeated_closure_are_valid(self):
        polygon = np.array(
            [[0, 0], [0, 7], [2, 7], [2, 3], [7, 3], [7, 0]], dtype=float
        )
        closed_reverse = np.vstack((polygon[::-1], polygon[-1]))
        points, spacing = sample_roi_plan(closed_reverse, "polygon", total_points=99)
        self.assert_complete_lattice(points, spacing, closed_reverse, "polygon")
        self.assert_inside_polygon(points, polygon)

    def test_count_points_stay_inside_rotated_rectangle(self):
        rectangle = np.array([[0, 2], [2, 4], [4, 2], [2, 0]], dtype=float)
        points, spacing = sample_roi_plan(rectangle, "rectangle", total_points=103)
        self.assert_complete_lattice(points, spacing, rectangle, "rectangle")
        self.assert_inside_polygon(points, rectangle)
        self.assertLessEqual(np.max(np.abs(points - 2).sum(axis=1)), 2 + 1e-12)

    def test_count_inside_rotated_sheared_ellipse(self):
        corners, center, axes = self.ellipse_corners()
        points, spacing = sample_roi_plan(corners, "ellipse", total_points=503)
        self.assert_complete_lattice(points, spacing, corners, "ellipse")
        coefficients = np.linalg.solve(axes, (points - center).T).T
        radius_squared = np.sum(coefficients**2, axis=1)
        self.assertTrue(np.all(radius_squared <= 1 + 1e-12))
        np.testing.assert_array_equal(
            points, sample_roi(corners, "ellipse", total_points=503)
        )

    def test_ellipse_outline_is_actual_boundary_not_corner_box(self):
        corners, center, axes = self.ellipse_corners()
        outline = roi_outline(corners, "ellipse", samples=96)
        self.assertEqual(outline.shape, (97, 2))
        np.testing.assert_array_equal(outline[0], outline[-1])
        coefficients = np.linalg.solve(axes, (outline - center).T).T
        np.testing.assert_allclose(np.sum(coefficients**2, axis=1), 1, atol=1e-13)

    def test_spacing_rectangle_keeps_pixel_distance_and_includes_boundaries(self):
        points = sample_roi(
            [[10, 20], [16, 26]], "rectangle", mode="spacing", spacing_px=2
        )
        expected = np.array(
            [[y, x] for x in (20, 22, 24, 26) for y in (10, 12, 14, 16)]
        )
        np.testing.assert_allclose(points, expected, atol=1e-12)
        self.assertEqual(len(points), 16)

    def test_spacing_concave_polygon_omits_unselected_corner(self):
        polygon = [[0, 0], [0, 4], [1, 4], [1, 1], [4, 1], [4, 0]]
        points = sample_roi(polygon, "polygon", mode="spacing", spacing_px=1)
        expected = np.array(
            [[y, x] for x in range(5) for y in range(5) if y <= 1 or x <= 1]
        )
        np.testing.assert_allclose(points, expected, atol=1e-12)
        self.assertEqual(len(points), 16)

    def test_spacing_rotated_rectangle_contains_boundary_vertices(self):
        diamond = [[0, 2], [2, 4], [4, 2], [2, 0]]
        points = sample_roi(diamond, "rectangle", mode="spacing", spacing_px=1)
        expected = np.array(
            [[y, x] for x in range(5) for y in range(5) if abs(y - 2) + abs(x - 2) <= 2]
        )
        np.testing.assert_allclose(points, expected, atol=1e-12)
        self.assertEqual(len(points), 13)

    def test_spacing_ellipse_is_lattice_clipped_inside_including_tangencies(self):
        points = sample_roi([[-2, -2], [2, 2]], "ellipse", mode="spacing", spacing_px=1)
        expected = np.array(
            [[y, x] for x in range(-2, 3) for y in range(-2, 3) if x * x + y * y <= 4]
        )
        np.testing.assert_allclose(points, expected, atol=1e-12)

    def test_spacing_sheared_ellipse_anchors_at_true_ellipse_minimum(self):
        corners, center, axes = self.ellipse_corners()
        points = sample_roi(corners, "ellipse", mode="spacing", spacing_px=1.3)
        coefficients = np.linalg.solve(axes, (points - center).T).T
        self.assertTrue(np.all(np.sum(coefficients**2, axis=1) <= 1 + 1e-12))
        origin = center - np.sqrt(np.sum(axes**2, axis=1))
        lattice_indices = (points - origin) / 1.3
        np.testing.assert_allclose(
            lattice_indices, np.rint(lattice_indices), atol=1e-12
        )

    def test_fractional_scanlines_match_independent_clipped_lattice(self):
        rng = np.random.default_rng(49)
        for _ in range(20):
            count = int(rng.integers(5, 18))
            angles = (
                (np.arange(count) + rng.uniform(-0.2, 0.2, count)) * 2 * np.pi / count
            )
            radii = rng.uniform(3, 12, count)
            vertices = (
                np.column_stack((np.cos(angles), np.sin(angles))) * radii[:, None]
            )
            low, high = vertices.min(axis=0), vertices.max(axis=0)
            ys = low[0] + np.arange(int((high[0] - low[0]) / 1.17) + 1) * 1.17
            xs = low[1] + np.arange(int((high[1] - low[1]) / 1.17) + 1) * 1.17
            lattice = np.array([[y, x] for x in xs for y in ys])
            inside = Path(vertices).contains_points(lattice)
            for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
                edge = end - start
                fraction = np.clip((lattice - start) @ edge / (edge @ edge), 0, 1)
                closest = start + fraction[:, None] * edge
                inside |= np.linalg.norm(lattice - closest, axis=1) < 1e-12
            actual = sample_roi(vertices, "polygon", mode="spacing", spacing_px=1.17)
            np.testing.assert_allclose(actual, lattice[inside], atol=1e-12)

    def test_count_very_thin_rotated_roi_does_not_allocate_bounding_grid(self):
        # A nearly diagonal strip has an enormous bounding-box/ROI area ratio.
        a, b = np.array([1e6, 1e6]), np.array([-1e-5, 1e-5])
        vertices = np.array([np.zeros(2), a, a + b, b])
        # This aligned thin strip has a feasible diagonal subset of the full
        # camera-axis lattice. Count-only search must avoid a dense box mesh.
        points, spacing = sample_roi_plan(vertices, "rectangle", total_points=5000)
        self.assert_complete_lattice(points, spacing, vertices, "rectangle")
        coordinates = np.linalg.solve(np.column_stack((a, b)), points.T).T
        self.assertTrue(np.all(coordinates >= -1e-4))
        self.assertTrue(np.all(coordinates <= 1 + 1e-4))

    def test_square_targets_use_equal_pitch_in_both_axes_without_padding(self):
        for target, expected_count in ((2, 4), (4, 4), (7, 9), (9, 9), (400, 400)):
            with self.subTest(target=target):
                points, spacing = sample_roi_plan(
                    [[0, 0], [100, 100]], "rectangle", total_points=target
                )
                self.assertEqual(len(points), expected_count)
                self.assert_complete_lattice(
                    points, spacing, [[0, 0], [100, 100]], "rectangle"
                )
                ys, xs = np.unique(points[:, 0]), np.unique(points[:, 1])
                np.testing.assert_allclose(np.diff(ys), spacing)
                np.testing.assert_allclose(np.diff(xs), spacing)
                if target == 400:
                    self.assertAlmostEqual(spacing, 100 / 19)

    def test_nonsquare_target_reports_actual_complete_grid_count(self):
        vertices = [[0, 0], [60, 100]]
        points, spacing = sample_roi_plan(vertices, "rectangle", total_points=400)
        self.assert_complete_lattice(points, spacing, vertices, "rectangle")
        self.assertLess(abs(len(points) - 400), 25)
        self.assertNotEqual(len(points), 400)

    def test_target_search_counts_without_repeated_point_allocations(self):
        angles = np.arange(2048) * 2 * np.pi / 2048
        polygon = np.column_stack((np.cos(angles), np.sin(angles)))
        with patch.object(
            roi_sampling, "_lattice", wraps=roi_sampling._lattice
        ) as lattice:
            points, spacing = sample_roi_plan(polygon, "polygon", total_points=401)
        evaluations = [
            call
            for call in lattice.call_args_list
            if call.kwargs.get("collect") is False
        ]
        self.assertLessEqual(len(evaluations), roi_sampling._MAX_TARGET_EVALUATIONS)
        self.assertEqual(len(lattice.call_args_list) - len(evaluations), 1)
        budget = evaluations[0].kwargs["budget"]
        self.assertGreaterEqual(budget[0], 0)
        self.assertGreaterEqual(budget[1], 0)
        self.assert_complete_lattice(points, spacing, polygon, "polygon")

    def test_maximum_count_is_exact_and_larger_count_rejected(self):
        points = sample_roi(
            [[0, 0], [100, 100]], "rectangle", total_points=MAX_ROI_POINTS
        )
        self.assertEqual(len(points), MAX_ROI_POINTS)
        with self.assertRaisesRegex(ValueError, "250,000"):
            sample_roi(
                [[0, 0], [100, 100]], "rectangle", total_points=MAX_ROI_POINTS + 1
            )

    def test_spacing_too_dense_is_rejected_not_truncated(self):
        with self.assertRaisesRegex(ValueError, "more than 250,000"):
            sample_roi([[0, 0], [500, 500]], "rectangle", mode="spacing", spacing_px=1)
        with self.assertRaisesRegex(ValueError, "too small|too many scan lines"):
            sample_roi(
                [[0, 0], [500, 500]], "ellipse", mode="spacing", spacing_px=1e-20
            )

    def test_fewer_than_two_actual_spacing_points_rejected(self):
        for kind in ("rectangle", "ellipse"):
            with self.subTest(kind=kind), self.assertRaisesRegex(
                ValueError, "fewer than two"
            ):
                sample_roi([[0, 0], [1, 1]], kind, mode="spacing", spacing_px=10)

    def test_invalid_count_and_spacing_rejected(self):
        for count in (0, 1, -2, 2.5, float("nan"), float("inf"), True, "no"):
            with self.subTest(count=count), self.assertRaises(ValueError):
                sample_roi([[0, 0], [10, 10]], "rectangle", total_points=count)
        for spacing in (0, -1, float("nan"), float("inf"), "no"):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                sample_roi(
                    [[0, 0], [10, 10]], "rectangle", mode="spacing", spacing_px=spacing
                )

    def test_invalid_geometry_and_unsupported_shapes_rejected(self):
        cases = [
            ([[0, 0], [1, 1]], "line"),
            ([[0, 0], [1, 1], [2, 0]], "path"),
            ([[0, 0], [0, 1], [0, 2]], "polygon"),
            ([[0, 0], [2, 2], [0, 2], [2, 0]], "polygon"),
            ([[0, 0], [4, 0], [4, 4], [2, 0], [0, 4]], "polygon"),
            ([[0, 0], [0, 0], [2, 2], [0, 2]], "polygon"),
            ([[0, 0], [0, 1], [2, 2], [1, 0]], "ellipse"),
            ([[0, 0], [float("nan"), 2]], "rectangle"),
            ([[0, 0], [float("inf"), 2]], "rectangle"),
            ([], "polygon"),
        ]
        for vertices, kind in cases:
            with self.subTest(vertices=vertices, kind=kind), self.assertRaises(
                ValueError
            ):
                sample_roi(vertices, kind)

    def test_input_arrays_remain_unchanged(self):
        vertices = np.array([[0, 0], [0, 4], [2, 4], [2, 0]], dtype=float)
        original = vertices.copy()
        sample_roi(vertices, "rectangle", total_points=31)
        roi_outline(vertices, "rectangle")
        np.testing.assert_array_equal(vertices, original)


if __name__ == "__main__":
    unittest.main()
