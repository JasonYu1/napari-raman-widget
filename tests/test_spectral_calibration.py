import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from napari_raman_widget.spectral_calibration import (
    PixelToWavenumberCalibration,
    load_pixel_to_wavenumber_calibration,
    pixel_to_shift,
    save_pixel_to_wavenumber_calibration,
)


class PixelToWavenumberCalibrationTests(unittest.TestCase):
    def test_pixel_to_shift_fits_quadratic(self) -> None:
        known_pixels = [0, 10, 20]
        known_shifts = [100, 250, 500]
        result = pixel_to_shift([0, 5, 10, 20], known_pixels, known_shifts)
        np.testing.assert_allclose(result, [100, 162.5, 250, 500])

    def test_pixel_to_shift_accepts_a_higher_degree(self) -> None:
        result = pixel_to_shift(
            [-2, -1, 0, 1, 2],
            [-2, -1, 0, 1],
            [-8, -1, 0, 1],
            degree=3,
        )
        np.testing.assert_allclose(result, [-8, -1, 0, 1, 8], atol=1e-12)

    def test_json_round_trip(self) -> None:
        calibration = PixelToWavenumberCalibration(
            [20, 0, 10], [500, 100, 250]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_pixel_to_wavenumber_calibration(
                Path(directory) / "raman-calibration", calibration
            )
            loaded = load_pixel_to_wavenumber_calibration(path)
            self.assertEqual(path.suffix, ".json")
            np.testing.assert_allclose(
                loaded.pixel_positions, [0, 10, 20]
            )
            np.testing.assert_allclose(loaded.known_shifts, [100, 250, 500])
            with path.open(encoding="utf-8") as stream:
                saved = json.load(stream)
            self.assertEqual(saved["calibration_type"], "pixel_to_wavenumber")
            self.assertEqual(saved["units"], "cm^-1")
            self.assertEqual(saved["polynomial_degree"], 2)
            self.assertEqual(len(saved["coefficients"]), 3)

    def test_requires_degree_plus_one_unique_points(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 3"):
            PixelToWavenumberCalibration([0, 1], [100, 200])
        with self.assertRaisesRegex(ValueError, "at least 4"):
            PixelToWavenumberCalibration(
                [0, 1, 2], [100, 200, 300], degree=3
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            PixelToWavenumberCalibration([0, 0, 1], [100, 200, 300])

    def test_json_round_trip_preserves_selected_degree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            calibration = PixelToWavenumberCalibration(
                [0, 1, 2, 3], [0, 1, 8, 27], degree=3
            )
            path = save_pixel_to_wavenumber_calibration(
                Path(directory) / "cubic.json", calibration
            )
            loaded = load_pixel_to_wavenumber_calibration(path)
            self.assertEqual(loaded.degree, 3)
            self.assertEqual(len(loaded.coefficients), 4)
            np.testing.assert_allclose(loaded.transform([4]), [64])

    def test_rejects_invalid_degree(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            PixelToWavenumberCalibration([0], [100], degree=0)
        with self.assertRaisesRegex(ValueError, "integer"):
            PixelToWavenumberCalibration(
                [0, 1, 2], [100, 200, 300], degree=1.5
            )


if __name__ == "__main__":
    unittest.main()
