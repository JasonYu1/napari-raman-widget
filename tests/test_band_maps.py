import unittest
from unittest.mock import patch

import numpy as np

from napari_raman_widget.band_maps import band_map_values


class BandMapTests(unittest.TestCase):
    def test_constant_and_negative_spectra_keep_signed_area(self):
        spectra = np.array([[4, 4, 4, 4], [-3, -3, -3, -3]], dtype=np.int16)
        actual = band_map_values(spectra, [0, 1, 2, 3], (0, 3))
        np.testing.assert_allclose(actual, [12, -9])

    def test_linear_spectra_integrate_exactly_at_fractional_edges(self):
        axis = np.array([0., 1., 3., 6.])
        spectra = np.array([2 * axis + 3, -axis + 10])
        low, high = 0.25, 4.5
        expected = [(high**2 + 3 * high) - (low**2 + 3 * low),
                    (-high**2 / 2 + 10 * high) - (-low**2 / 2 + 10 * low)]
        np.testing.assert_allclose(band_map_values(spectra, axis, (low, high)), expected)

    def test_narrow_band_inside_one_bin_interval(self):
        actual = band_map_values([[0, 10, 20]], [0, 1, 2], (0.2, 0.3))
        np.testing.assert_allclose(actual, [0.25])

    def test_nonuniform_axis_uses_coordinate_widths(self):
        actual = band_map_values([[0, 2, 4, 1]], [0, 1, 3, 4], (0, 4))
        np.testing.assert_allclose(actual, [9.5])

    def test_two_bins_and_exact_edges(self):
        np.testing.assert_allclose(band_map_values([[2, 6]], [10, 20], (10, 20)), [40])

    def test_descending_axis_matches_ascending_for_band_and_ratio(self):
        axis = np.array([0., 1., 3., 6.])
        spectra = np.array([axis + 1, axis * 2 + 5])
        for second_band in (None, (3.5, 5.75)):
            with self.subTest(second_band=second_band):
                expected = band_map_values(spectra, axis, (0.2, 2.5), second_band)
                actual = band_map_values(spectra[:, ::-1], axis[::-1], (0.2, 2.5), second_band)
                np.testing.assert_allclose(actual, expected)

    def test_nonfinite_samples_outside_exact_edges_are_ignored(self):
        spectra = [[np.nan, 1, 2, 3, np.inf], [-np.inf, 4, 4, 4, np.nan]]
        np.testing.assert_allclose(band_map_values(spectra, np.arange(5), (1, 3)), [4, 8])

    def test_nonfinite_internal_and_interpolation_samples_mask_only_their_point(self):
        spectra = [[np.nan, 1, 2, 3], [0, 1, np.inf, 3], [0, 1, 2, np.nan], [0, 1, 2, 3]]
        np.testing.assert_allclose(
            band_map_values(spectra, np.arange(4), (0.5, 2.5)),
            [np.nan, np.nan, np.nan, 3], equal_nan=True,
        )

    def test_ratio_uses_areas_not_sums_and_retains_negative_numerator(self):
        spectra = [[2, 2, 1, 1, 1], [-2, -2, 1, 1, 1]]
        np.testing.assert_allclose(
            band_map_values(spectra, np.arange(5), (0, 1), (2, 4)), [1, -1]
        )

    def test_ratio_masks_zero_negative_and_tiny_denominators(self):
        spectra = [[1, 1, 0, 0], [1, 1, -1, -1], [1, 1, 1e-13, 1e-13], [1, 1, 2, 2]]
        np.testing.assert_allclose(
            band_map_values(spectra, np.arange(4), (0, 1), (2, 3)),
            [np.nan, np.nan, np.nan, 0.5], equal_nan=True,
        )

    def test_custom_denominator_floor_is_inclusive(self):
        spectra = [[1, 1, 2, 2], [1, 1, 3, 3]]
        np.testing.assert_allclose(
            band_map_values(spectra, np.arange(4), (0, 1), (2, 3), denominator_floor=2),
            [np.nan, 1 / 3], equal_nan=True,
        )

    def test_zero_floor_allows_tiny_positive_denominator(self):
        actual = band_map_values([[1, 1, 1e-15, 1e-15]], np.arange(4), (0, 1), (2, 3),
                                 denominator_floor=0)
        np.testing.assert_allclose(actual, [1e15])

    def test_nonfinite_in_either_ratio_band_masks_result(self):
        spectra = [[np.nan, 1, 2, 2], [1, 1, 2, np.inf], [1, 1, 2, 2]]
        np.testing.assert_allclose(
            band_map_values(spectra, np.arange(4), (0, 1), (2, 3)),
            [np.nan, np.nan, 0.5], equal_nan=True,
        )

    def test_gap_between_ratio_bands_does_not_affect_result(self):
        actual = band_map_values([[1, 1, np.nan, 2, 2]], np.arange(5), (0, 1), (3, 4))
        np.testing.assert_allclose(actual, [0.5])

    def test_empty_point_set_returns_empty_one_dimensional_array(self):
        for band_b in (None, (0, 1)):
            actual = band_map_values(np.empty((0, 2)), [0, 1], (0, 1), band_b)
            self.assertEqual(actual.shape, (0,))
            self.assertEqual(actual.dtype, np.dtype(float))

    def test_inputs_are_unchanged_and_read_only_inputs_work(self):
        axis = np.arange(5, dtype=float)[::-1]
        spectra = np.array([[5, 4, 3, 2, 1], [1, 2, np.nan, 4, 5]], dtype=float)
        original_axis, original_spectra = axis.copy(), spectra.copy()
        axis.flags.writeable = spectra.flags.writeable = False
        band_map_values(spectra, axis, (0.25, 1.75), (3, 4))
        np.testing.assert_array_equal(axis, original_axis)
        np.testing.assert_array_equal(spectra, original_spectra)

    def test_integer_samples_do_not_overflow_during_integration(self):
        actual = band_map_values(np.array([[255, 255]], dtype=np.uint8), [0, 2], (0, 2))
        np.testing.assert_allclose(actual, [510])

    def test_chunk_boundaries_match_unrestricted_integration(self):
        axis = np.arange(9, dtype=float)
        spectra = np.arange(11 * 9, dtype=np.float32).reshape(11, 9) + 1
        spectra[3, 4] = np.nan
        spectra[5, 8] = np.inf
        spectra[9, 1] = -np.inf
        for second_band in (None, (6.25, 7.8)):
            with self.subTest(second_band=second_band):
                expected = band_map_values(spectra, axis, (1.25, 6.5), second_band)
                # Seven selected numerator bins -> three rows per chunk,
                # including a partial last chunk and invalid boundary rows.
                with patch("napari_raman_widget.band_maps._MAX_INTEGRATION_SAMPLES", 21):
                    actual = band_map_values(spectra, axis, (1.25, 6.5), second_band)
                    descending = band_map_values(spectra[:, ::-1], axis[::-1],
                                                 (1.25, 6.5), second_band)
                np.testing.assert_allclose(actual, expected, equal_nan=True)
                np.testing.assert_allclose(descending, expected, equal_nan=True)

    def test_chunk_budget_smaller_than_one_spectrum_still_integrates(self):
        with patch("napari_raman_widget.band_maps._MAX_INTEGRATION_SAMPLES", 1):
            actual = band_map_values([[1, 2, 3], [2, 4, 6]], [0, 1, 2], (0, 2))
        np.testing.assert_allclose(actual, [4, 8])

    def test_overflowing_area_or_ratio_is_masked(self):
        actual = band_map_values([[1e308, 1e308]], [0, 10], (0, 10))
        self.assertTrue(np.isnan(actual[0]))
        ratio = band_map_values([[1e308, 1e308, 1e-308, 1e-308]], np.arange(4),
                                (0, 1), (2, 3), denominator_floor=0)
        self.assertTrue(np.isnan(ratio[0]))

    def test_invalid_axes_raise_clear_errors(self):
        for axis in ([], [1], [[0, 1]], [0, 0], [0, 2, 1], [0, np.nan], [np.inf, 1],
                     [0, 1j], ["bad", "axis"]):
            with self.subTest(axis=axis), self.assertRaisesRegex(ValueError, "axis"):
                band_map_values([[1, 2]], axis, (0, 1))

    def test_invalid_shapes_and_nonreal_samples_raise(self):
        for spectra in ([1, 2], [[[1, 2]]], [[1, 2, 3]], [[1j, 2]], [["bad", "value"]],
                        [[1, 2], [1]]):
            with self.subTest(spectra=spectra), self.assertRaisesRegex(ValueError, "Spectra"):
                band_map_values(spectra, [0, 1], (0, 1))

    def test_invalid_bounds_raise_for_both_bands(self):
        for band in ((1, 1), (1, 0), (-0.1, 1), (0, 2.1), (np.nan, 1), (0, np.inf),
                     (0,), (0, 1, 2), [[0, 1]], (0, 1j), ("bad", 1), None):
            with self.subTest(band=band), self.assertRaises(ValueError):
                band_map_values([[1, 2, 3]], [0, 1, 2], band)
            if band is not None:
                with self.subTest(second_band=band), self.assertRaises(ValueError):
                    band_map_values([[1, 2, 3]], [0, 1, 2], (0, 1), band)

    def test_invalid_floor_is_rejected_even_for_plain_intensity(self):
        for floor in (-1, np.nan, np.inf, -np.inf, "bad", None, 1j):
            with self.subTest(floor=floor), self.assertRaisesRegex(ValueError, "floor"):
                band_map_values([[1, 2]], [0, 1], (0, 1), denominator_floor=floor)


if __name__ == "__main__":
    unittest.main()
