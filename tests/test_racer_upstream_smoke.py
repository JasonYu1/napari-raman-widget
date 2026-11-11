"""Smoke test against RACER's real public reader and calibrator API."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from napari_raman_widget.cosmic_ray_removal import load_racer_calibration
from napari_raman_widget.spectra import (
    racer_detector_signature,
    save_collection_record,
    save_dark_noise_artifacts,
)


RACER_AVAILABLE = importlib.util.find_spec("racer_raman") is not None


class _Collector:
    detector_rows = 256


@unittest.skipUnless(
    RACER_AVAILABLE,
    "racer-raman is not installed in this test environment",
)
class RacerUpstreamSmokeTests(unittest.TestCase):
    def test_generated_fvb_pair_fits_and_corrects_with_public_api(self) -> None:
        rng = np.random.default_rng(20261109)
        frames = 60
        columns = 48
        dark = rng.normal(100.0, 2.0, (frames, columns))
        for frame, column in zip(
            range(0, frames, 4), range(2, 2 + frames // 4), strict=True
        ):
            dark[frame, column : column + 2] += 120.0
        x = np.linspace(-1.0, 1.0, columns)
        reference = (
            100.0
            + 500.0 * np.exp(-0.5 * (x / 0.25) ** 2)[None, :]
            + rng.normal(0.0, 2.0, (frames, columns))
        )
        collector = _Collector()
        signature = racer_detector_signature(
            collector, dark, "fvb", 1000.0
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _bias_path, dark_store = save_dark_noise_artifacts(
                root / "dark.npy",
                dark,
                collector=collector,
                read_mode="fvb",
                exposure_ms=1000.0,
            )
            reference_store = save_collection_record(
                root / "reference.zarr",
                reference,
                {
                    "read_mode": "fvb",
                    "exposure_ms": 1000.0,
                    "racer_role": "open_reference",
                    "units": "ADU",
                    "gain_e_per_adu": 0.7,
                    "saturation_adu": 65535.0,
                    "gain_source": "racer_default_assumption",
                    "racer_detector_signature": signature,
                },
            )

            calibration = load_racer_calibration(
                dark_store, reference_store
            )
            raw = np.array(reference[0], copy=True)
            raw[20:22] += 300.0
            original = raw.copy()
            corrected = calibration.correct(raw)

        np.testing.assert_array_equal(raw, original)
        self.assertEqual(corrected.shape, original.shape)
        self.assertTrue(np.all(np.isfinite(corrected)))

    def test_generated_fvb_dark_only_fits_and_corrects_with_public_api(
        self,
    ) -> None:
        rng = np.random.default_rng(20261110)
        frames = 60
        columns = 48
        dark = rng.normal(100.0, 2.0, (frames, columns))
        for frame, column in zip(
            range(0, frames, 4), range(2, 2 + frames // 4), strict=True
        ):
            dark[frame, column : column + 2] += 120.0

        collector = _Collector()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _bias_path, dark_store = save_dark_noise_artifacts(
                root / "dark.npy",
                dark,
                collector=collector,
                read_mode="fvb",
                exposure_ms=1000.0,
            )
            calibration = load_racer_calibration(dark_store, None)
            x = np.linspace(-1.0, 1.0, columns)
            raw = 100.0 + 500.0 * np.exp(-0.5 * (x / 0.25) ** 2)
            raw[20:22] += 300.0
            original = raw.copy()
            corrected = calibration.correct(raw)

        np.testing.assert_array_equal(raw, original)
        self.assertIsNone(calibration.source_paths[1])
        self.assertEqual(corrected.shape, original.shape)
        self.assertTrue(np.all(np.isfinite(corrected)))


if __name__ == "__main__":
    unittest.main()
