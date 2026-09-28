import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from napari_raman_widget.spectra import (
    asls_baseline,
    save_collection_record,
    smooth_spectra,
    spectral_bias_from_dark_noise,
    subtract_spectral_bias,
    sum_detector_rows,
)


class CollectionRecordTests(unittest.TestCase):
    def test_saves_data_and_metadata_in_one_zarr_dataset(self) -> None:
        data = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
        metadata = {
            "read_mode": "image",
            "collection_elapsed_seconds": 1.25,
        }
        with tempfile.TemporaryDirectory() as directory:
            store_path = save_collection_record(
                Path(directory) / "point-collection.npy",
                data,
                metadata,
            )

            self.assertEqual(store_path.suffix, ".zarr")
            self.assertEqual(
                [path.name for path in Path(directory).iterdir()],
                ["point-collection.zarr"],
            )
            saved = xr.open_zarr(store_path, consolidated=True)
            try:
                np.testing.assert_array_equal(saved["signal"].values, data)
                self.assertEqual(
                    saved["signal"].dims,
                    ("repeat", "detector_y", "detector_x"),
                )
                self.assertEqual(saved.attrs["data_shape"], [2, 3, 4])
                self.assertEqual(saved.attrs["data_dtype"], "uint16")
                self.assertEqual(
                    saved.attrs["collection_elapsed_seconds"], 1.25
                )
            finally:
                saved.close()


class DetectorRowSumTests(unittest.TestCase):
    def test_sums_an_inclusive_row_range(self) -> None:
        image = np.arange(20).reshape(4, 5)

        result = sum_detector_rows(image, 1, 2)

        np.testing.assert_array_equal(result, image[1] + image[2])
        self.assertEqual(result.dtype, np.dtype(float))

    def test_rejects_reversed_or_out_of_bounds_ranges(self) -> None:
        image = np.zeros((4, 5))

        with self.assertRaisesRegex(ValueError, "end_row"):
            sum_detector_rows(image, 3, 2)
        with self.assertRaisesRegex(ValueError, "start_row"):
            sum_detector_rows(image, -1, 2)

    def test_requires_a_detector_image(self) -> None:
        with self.assertRaisesRegex(ValueError, "detector image"):
            sum_detector_rows(np.zeros((2, 3, 4)), 0, 1)


class SpectralBiasTests(unittest.TestCase):
    def test_subtracts_filtered_dark_noise_and_keeps_negative_values(self) -> None:
        dark_noise = np.array(
            [
                [10.0, 20.0, 30.0],
                [12.0, 22.0, 32.0],
                [11.0, 21.0, 31.0],
            ]
        )
        spectra = np.array([[9.0, 25.0, 40.0]])

        bias = spectral_bias_from_dark_noise(dark_noise)
        corrected = subtract_spectral_bias(spectra, bias)

        np.testing.assert_allclose(bias, [11.0, 21.0, 31.0])
        np.testing.assert_allclose(corrected, [[-2.0, 4.0, 9.0]])

    def test_rejects_mismatched_spectral_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            subtract_spectral_bias(
                np.zeros((2, 4)),
                np.zeros(3),
            )

    def test_dark_noise_must_contain_repeated_spectra(self) -> None:
        with self.assertRaisesRegex(ValueError, "dark noise"):
            spectral_bias_from_dark_noise(np.zeros(4))


class SpectralSmoothingTests(unittest.TestCase):
    def test_order_three_smoothing_preserves_a_cubic(self) -> None:
        x = np.arange(21, dtype=float)
        cubic = 2 * x**3 - 4 * x**2 + 3 * x - 7

        smoothed = smooth_spectra(cubic, 7)

        np.testing.assert_allclose(smoothed, cubic, atol=1e-9)

    def test_smoothing_supports_multiple_spectra(self) -> None:
        spectra = np.vstack([np.arange(9), np.arange(9) ** 2])
        smoothed = smooth_spectra(spectra, 5)
        self.assertEqual(smoothed.shape, spectra.shape)

    def test_smoothing_requires_a_valid_odd_window(self) -> None:
        for invalid_window in (3, 4, 10):
            with self.subTest(window=invalid_window):
                with self.assertRaisesRegex(ValueError, "smoothing window"):
                    smooth_spectra(np.zeros(9), invalid_window)


class AsymmetricLeastSquaresBaselineTests(unittest.TestCase):
    def test_estimates_a_smooth_baseline_without_removing_a_peak(self) -> None:
        x = np.linspace(0, 1, 200)
        expected_baseline = 10 + 3 * x + 2 * x**2
        peak = 30 * np.exp(-((x - 0.5) / 0.03) ** 2)
        spectrum = expected_baseline + peak

        baseline = asls_baseline(spectrum, lam=1e5)
        corrected = spectrum - baseline

        self.assertLess(
            np.mean(np.abs(baseline - expected_baseline)),
            0.25,
        )
        self.assertGreater(corrected.max(), 25)

    def test_fits_rows_independently_without_mutating_input(self) -> None:
        x = np.arange(21, dtype=float)
        spectra = np.vstack([2 * x + 5, -0.5 * x + 20])
        original = spectra.copy()

        baselines = asls_baseline(spectra)

        self.assertEqual(baselines.shape, spectra.shape)
        np.testing.assert_allclose(baselines, spectra, atol=1e-7)
        np.testing.assert_array_equal(spectra, original)

    def test_preserves_sparse_finite_samples_when_a_fit_is_impossible(self) -> None:
        spectrum = np.array([np.nan, 4.0, np.inf, 7.0, np.nan])

        baseline = asls_baseline(spectrum)

        np.testing.assert_array_equal(
            np.isnan(baseline),
            [1, 0, 1, 0, 1],
        )
        np.testing.assert_allclose(
            spectrum[[1, 3]] - baseline[[1, 3]],
            [4.0, 7.0],
        )

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "one- or two-dimensional"):
            asls_baseline(np.zeros((2, 3, 4)))
        with self.assertRaisesRegex(ValueError, "at least 3 pixels"):
            asls_baseline(np.zeros(2))
        for invalid_lambda in (0, -1, np.inf, True):
            with self.subTest(lam=invalid_lambda):
                with self.assertRaisesRegex(ValueError, "lambda"):
                    asls_baseline(np.zeros(5), lam=invalid_lambda)


if __name__ == "__main__":
    unittest.main()
