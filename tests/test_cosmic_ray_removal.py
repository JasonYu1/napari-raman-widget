import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import zarr

from napari_raman_widget.cosmic_ray_removal import (
    RacerCalibration,
    load_racer_calibration,
)


def _fake_racer(stores):
    module = types.ModuleType("racer_raman")

    def _table(experiment, specification):
        exposure = specification["exposure_ms"]
        columns = specification["columns"]
        frames = range(specification.get("frames", 3))
        if specification["read_mode"] == "image":
            rows = specification["rows"]
            index = pd.MultiIndex.from_product(
                [[experiment], [exposure], frames, rows],
                names=("experiment", "exposure_ms", "frame", "row"),
            )
        else:
            index = pd.MultiIndex.from_product(
                [[experiment], [exposure], frames],
                names=("experiment", "exposure_ms", "frame"),
            )
        values = np.arange(len(index) * len(columns), dtype=np.int16).reshape(
            len(index), len(columns)
        )
        return pd.DataFrame(
            values,
            index=index,
            columns=pd.Index(columns, name="column"),
        )

    def _read(stores_by_experiment, expected_mode):
        experiment, path = next(iter(stores_by_experiment.items()))
        specification = stores[Path(path).resolve()]
        if specification["read_mode"] != expected_mode:
            raise ValueError(
                f"store uses {specification['read_mode']}, not {expected_mode}"
            )
        return _table(experiment, specification)

    module.read_spectrum_zarr = lambda values: _read(values, "fvb")
    module.read_image_zarr = lambda values: _read(values, "image")

    class SpectrumCalibrator:
        fit_calls = []

        def __init__(self):
            self.correct_calls = 0
            self.correct_many_calls = 0

        @classmethod
        def fit(cls, dark, reference, **kwargs):
            cls.fit_calls.append((dark, reference, kwargs))
            return cls()

        def correct(self, values):
            self.correct_calls += 1
            values[...] += 1
            return values.astype(np.float32) + 0.5

        def correct_many(self, values):
            self.correct_many_calls += 1
            values[...] += 1
            return values.astype(np.float32) + 0.5

    class ImageCalibrator:
        fit_calls = []

        def __init__(self):
            self.correct_calls = 0

        @classmethod
        def fit(cls, dark, reference, **kwargs):
            cls.fit_calls.append((dark, reference, kwargs))
            return cls()

        def correct(self, values):
            self.correct_calls += 1
            values[...] += 2
            return values.astype(np.float64) + 0.25

    module.SpectrumCalibrator = SpectrumCalibrator
    module.ImageCalibrator = ImageCalibrator
    return module


class RacerCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.stores = {}

    def tearDown(self):
        self.temporary_directory.cleanup()

    def store(self, name, **specification):
        path = self.root / f"{name}.zarr"
        root = zarr.open_group(str(path), mode="w")
        metadata = specification.pop(
            "metadata",
            {
                "racer_role": (
                    "closed_shutter"
                    if name.startswith("dark")
                    else "open_reference"
                ),
                "units": "ADU",
                "gain_e_per_adu": 0.7,
                "saturation_adu": 65535.0,
                "racer_detector_signature": {
                    "schema_version": 2,
                    "read_mode": specification["read_mode"],
                    "columns": list(specification["columns"]),
                    "rows": list(specification.get("rows", ())),
                    "camera_serial": "TEST-1",
                    "actual_exposure_ms": specification["exposure_ms"],
                    "gain_source": "configured_for_test",
                },
            },
        )
        root.attrs.update(metadata)
        self.stores[path.resolve()] = specification
        return path

    def install_fake(self):
        fake = _fake_racer(self.stores)
        return fake, patch.dict(sys.modules, {"racer_raman": fake})

    def test_image_cleanup_repairs_compact_off_trace_event_only(self):
        class IdentityImageCalibrator:
            @staticmethod
            def correct(values):
                return np.array(values, dtype=float, copy=True)

        shape = (6, 15)
        background = np.full(shape, 10.0)
        dark_master = np.full(shape, 10.0)
        dark_variance = np.ones(shape)
        support = np.zeros(shape, dtype=bool)
        support[2, 7] = True
        calibration = RacerCalibration(
            read_mode="image",
            exposure_ms=1000.0,
            detector_shape=shape,
            detector_columns=tuple(range(shape[1])),
            detector_rows=tuple(range(shape[0])),
            gain_e_per_adu=1.0,
            saturation_adu=65535.0,
            detector_signature=None,
            metadata_warnings=(),
            closed_shutter_path=self.root / "dark.zarr",
            open_reference_path=self.root / "reference.zarr",
            _calibrator=IdentityImageCalibrator(),
            _image_background_adu=background,
            _image_dark_master_adu=dark_master,
            _image_dark_variance_adu2=dark_variance,
            _image_support_mask=support,
        )
        raw = background.copy()
        raw[0, 1] = 100.0
        raw[0, 2] = 15.0
        raw[2, 7] = 100.0
        raw[5, :12] = 15.0
        raw[5, 5] = 100.0
        original = raw.copy()

        corrected = calibration.correct(raw)

        np.testing.assert_array_equal(raw, original)
        self.assertEqual(corrected[0, 1], 10.0)
        self.assertEqual(corrected[0, 2], 10.0)
        self.assertEqual(corrected[2, 7], 100.0)
        np.testing.assert_array_equal(corrected[5, :12], raw[5, :12])
        self.assertTrue(calibration.full_frame_image_cleanup)

    def test_loads_fvb_once_and_corrects_singles_and_batches(self):
        dark = self.store(
            "dark", read_mode="fvb", exposure_ms=1000, columns=(0, 1, 2)
        )
        reference = self.store(
            "reference",
            read_mode="fvb",
            exposure_ms=1000,
            columns=(0, 1, 2),
        )
        fake, installed = self.install_fake()

        with installed:
            calibration = load_racer_calibration(dark, reference)
            self.assertEqual(calibration.read_mode, "fvb")
            self.assertEqual(calibration.exposure_ms, 1000.0)
            self.assertEqual(calibration.detector_shape, (3,))
            self.assertEqual(
                calibration.source_paths, (dark.resolve(), reference.resolve())
            )
            self.assertEqual(len(fake.SpectrumCalibrator.fit_calls), 1)
            fit_kwargs = fake.SpectrumCalibrator.fit_calls[0][2]
            self.assertEqual(
                fit_kwargs,
                {"exposure_ms": 1000.0, "saturation_adu": 65535.0},
            )
            self.assertEqual(calibration.detector_columns, (0, 1, 2))
            self.assertEqual(calibration.gain_e_per_adu, 0.7)
            self.assertEqual(calibration.saturation_adu, 65535.0)
            self.assertFalse(calibration.metadata_warnings)

            compatible, reason = calibration.compatibility(
                np.zeros((2, 3)), 1000, "fvb"
            )
            self.assertTrue(compatible)
            self.assertEqual(reason, "")

            raw = np.array([1, 2, 3], dtype=np.int16)
            original = raw.copy()
            corrected = calibration.correct(raw)
            np.testing.assert_array_equal(raw, original)
            self.assertEqual(corrected.shape, raw.shape)
            self.assertEqual(corrected.dtype, np.dtype(np.float32))

            batch = np.arange(6, dtype=np.uint16).reshape(2, 3)
            original_batch = batch.copy()
            corrected_batch = calibration.correct(batch)
            np.testing.assert_array_equal(batch, original_batch)
            self.assertEqual(corrected_batch.shape, batch.shape)
            self.assertEqual(corrected_batch.dtype, np.dtype(np.float32))
            self.assertEqual(len(fake.SpectrumCalibrator.fit_calls), 1)

    def test_loads_dark_only_fvb_with_closed_shutter_variance(self):
        dark = self.store(
            "dark-only", read_mode="fvb", exposure_ms=1000, columns=(0, 1, 2)
        )
        fake, installed = self.install_fake()

        with installed:
            calibration = load_racer_calibration(dark, None)

        self.assertEqual(calibration.read_mode, "fvb")
        self.assertEqual(calibration.source_paths, (dark.resolve(), None))
        self.assertIsNone(calibration.open_reference_path)
        self.assertEqual(len(fake.SpectrumCalibrator.fit_calls), 1)
        fitted_dark, fitted_reference, fit_kwargs = (
            fake.SpectrumCalibrator.fit_calls[0]
        )
        self.assertIs(fitted_dark, fitted_reference)
        self.assertEqual(
            fit_kwargs,
            {"exposure_ms": 1000.0, "saturation_adu": 65535.0},
        )
        warning = " ".join(calibration.metadata_warnings)
        self.assertIn("closed-shutter variance only", warning)
        self.assertIn("signal-dependent shot noise is not modeled", warning)

        raw = np.array([1, 2, 3], dtype=np.int16)
        corrected = calibration.correct(raw)
        self.assertEqual(corrected.shape, raw.shape)

    def test_dark_only_image_requires_an_illuminated_reference(self):
        dark = self.store(
            "dark-image-only",
            read_mode="image",
            exposure_ms=1000,
            rows=(0, 1),
            columns=(0, 1, 2),
        )
        fake, installed = self.install_fake()

        with installed, self.assertRaisesRegex(
            ValueError, "open-reference acquisition is required"
        ):
            load_racer_calibration(dark, None)

        self.assertFalse(fake.ImageCalibrator.fit_calls)

    def test_new_image_api_receives_explicit_zero_background_with_warning(self):
        dark = self.store(
            "dark-image",
            read_mode="image",
            exposure_ms=1000,
            rows=(0, 1),
            columns=(0, 1, 2),
        )
        reference = self.store(
            "reference-image",
            read_mode="image",
            exposure_ms=1000,
            rows=(0, 1),
            columns=(0, 1, 2),
        )
        fake, installed = self.install_fake()

        class ImageCalibratorWithBackground:
            fit_calls = []

            @classmethod
            def fit(
                cls,
                dark_values,
                reference_values,
                instrument_background_adu,
                **kwargs,
            ):
                cls.fit_calls.append(
                    (
                        dark_values,
                        reference_values,
                        np.array(instrument_background_adu, copy=True),
                        kwargs,
                    )
                )
                return cls()

            @staticmethod
            def correct(values):
                return np.array(values, copy=True)

        fake.ImageCalibrator = ImageCalibratorWithBackground
        with installed:
            calibration = load_racer_calibration(dark, reference)

        self.assertEqual(len(ImageCalibratorWithBackground.fit_calls), 1)
        background = ImageCalibratorWithBackground.fit_calls[0][2]
        np.testing.assert_array_equal(background, np.zeros((2, 3)))
        self.assertTrue(
            any(
                "instrument-background map" in warning
                for warning in calibration.metadata_warnings
            )
        )

    def test_dark_only_still_validates_closed_store_metadata(self):
        dark = self.store(
            "dark-wrong-units",
            read_mode="fvb",
            exposure_ms=1000,
            columns=(0, 1),
            metadata={
                "racer_role": "closed_shutter",
                "units": "electrons",
            },
        )
        _fake, installed = self.install_fake()

        with installed, self.assertRaisesRegex(
            ValueError, "closed-shutter store units must be 'ADU'"
        ):
            load_racer_calibration(dark, None)

    def test_loads_image_once_and_corrects_frames_without_mutating_raw(self):
        dark = self.store(
            "dark",
            read_mode="image",
            exposure_ms=2000,
            rows=(0, 1),
            columns=(0, 1, 2),
        )
        reference = self.store(
            "reference",
            read_mode="image",
            exposure_ms=2000,
            rows=(0, 1),
            columns=(0, 1, 2),
        )
        fake, installed = self.install_fake()

        with installed:
            calibration = load_racer_calibration(dark, reference)
            self.assertEqual(calibration.read_mode, "image")
            self.assertEqual(calibration.detector_shape, (2, 3))
            self.assertEqual(len(fake.ImageCalibrator.fit_calls), 1)
            self.assertEqual(
                fake.ImageCalibrator.fit_calls[0][2],
                {"gain_e_per_adu": 0.7, "saturation_adu": 65535.0},
            )

            image = np.arange(6, dtype=np.int16).reshape(2, 3)
            image_before = image.copy()
            corrected_image = calibration.correct(image)
            np.testing.assert_array_equal(image, image_before)
            self.assertEqual(corrected_image.dtype, np.dtype(np.float64))

            frames = np.arange(12, dtype=np.int16).reshape(2, 2, 3)
            frames_before = frames.copy()
            corrected_frames = calibration.correct(frames)
            np.testing.assert_array_equal(frames, frames_before)
            self.assertEqual(corrected_frames.shape, frames.shape)
            self.assertEqual(len(fake.ImageCalibrator.fit_calls), 1)

    def test_compatibility_reports_mode_exposure_and_geometry_mismatches(self):
        dark = self.store(
            "dark", read_mode="fvb", exposure_ms=50, columns=(0, 1, 2, 3)
        )
        reference = self.store(
            "reference", read_mode="fvb", exposure_ms=50, columns=(0, 1, 2, 3)
        )
        _fake, installed = self.install_fake()

        with installed:
            calibration = load_racer_calibration(dark, reference)
            cases = (
                ((np.zeros(4), 50, "single_track"), "unsupported read mode"),
                ((np.zeros((2, 4)), 50, "image"), "read mode mismatch"),
                ((np.zeros(4), 51, "fvb"), "exposure mismatch"),
                ((np.zeros(5), 50, "fvb"), "detector geometry mismatch"),
                ((np.zeros((1, 2, 4)), 50, "fvb"), "FVB values must have"),
            )
            for arguments, expected_reason in cases:
                with self.subTest(reason=expected_reason):
                    compatible, reason = calibration.compatibility(*arguments)
                    self.assertFalse(compatible)
                    self.assertIn(expected_reason, reason)

            with self.assertRaisesRegex(ValueError, "detector geometry mismatch"):
                calibration.correct(np.zeros(5))

    def test_same_shape_shifted_roi_and_signature_mismatches_are_rejected(self):
        metadata = {
            "gain_e_per_adu": 1.25,
            "saturation_adu": 32000,
            "racer_detector_signature": {
                "read_mode": "fvb",
                "detector_x_start": 100,
                "detector_x_stop_exclusive": 104,
                "detector_x_bin": 1,
            },
        }
        dark = self.store(
            "dark-roi",
            read_mode="fvb",
            exposure_ms=50,
            columns=(100, 101, 102, 103),
            metadata=metadata,
        )
        reference = self.store(
            "reference-roi",
            read_mode="fvb",
            exposure_ms=50,
            columns=(100, 101, 102, 103),
            metadata=metadata,
        )
        _fake, installed = self.install_fake()

        with installed:
            calibration = load_racer_calibration(dark, reference)
            compatible, reason = calibration.compatibility(
                np.zeros(4), 50, "fvb"
            )
            self.assertFalse(compatible)
            self.assertIn("column ROI/binning mismatch", reason)

            compatible, reason = calibration.compatibility(
                np.zeros(4),
                50,
                "fvb",
                detector_columns=(100, 101, 102, 103),
                detector_signature={"read_mode": "fvb", "detector_x_bin": 2},
            )
            self.assertFalse(compatible)
            self.assertIn("acquisition settings", reason)

            compatible, reason = calibration.compatibility(
                np.zeros(4),
                50,
                "fvb",
                detector_columns=(100, 101, 102, 103),
                detector_signature=metadata["racer_detector_signature"],
            )
            self.assertTrue(compatible, reason)

    def test_metadata_must_match_and_missing_values_are_reported(self):
        signature = {"read_mode": "fvb", "detector_x_bin": 1}
        dark = self.store(
            "dark-metadata",
            read_mode="fvb",
            exposure_ms=10,
            columns=(0, 1),
            metadata={
                "gain_e_per_adu": 0.7,
                "saturation_adu": 40000,
                "racer_detector_signature": signature,
            },
        )
        reference = self.store(
            "reference-metadata",
            read_mode="fvb",
            exposure_ms=10,
            columns=(0, 1),
            metadata={
                "gain_e_per_adu": 0.7,
                "saturation_adu": 41000,
                "racer_detector_signature": signature,
            },
        )
        _fake, installed = self.install_fake()
        with installed, self.assertRaisesRegex(
            ValueError, "saturation_adu.*does not match"
        ):
            load_racer_calibration(dark, reference)

        dark_missing = self.store(
            "dark-missing-metadata",
            read_mode="fvb",
            exposure_ms=10,
            columns=(0, 1),
            metadata={},
        )
        reference_missing = self.store(
            "reference-missing-metadata",
            read_mode="fvb",
            exposure_ms=10,
            columns=(0, 1),
            metadata={},
        )
        _fake, installed = self.install_fake()
        with installed:
            calibration = load_racer_calibration(
                dark_missing, reference_missing
            )
        self.assertGreaterEqual(len(calibration.metadata_warnings), 3)
        self.assertTrue(
            any("gain_e_per_adu" in item for item in calibration.metadata_warnings)
        )

    def test_roles_units_and_minimum_frame_count_are_validated(self):
        metadata = {
            "racer_role": "open_reference",
            "units": "ADU",
            "gain_e_per_adu": 0.7,
            "saturation_adu": 65535,
            "racer_detector_signature": {"read_mode": "fvb"},
        }
        dark = self.store(
            "dark-wrong-role",
            read_mode="fvb",
            exposure_ms=10,
            columns=(0, 1),
            metadata=metadata,
        )
        reference = self.store(
            "reference-role",
            read_mode="fvb",
            exposure_ms=10,
            columns=(0, 1),
            metadata=metadata,
        )
        _fake, installed = self.install_fake()
        with installed, self.assertRaisesRegex(
            ValueError, "closed-shutter store has racer_role"
        ):
            load_racer_calibration(dark, reference)

        dark_short = self.store(
            "dark-short",
            read_mode="fvb",
            exposure_ms=10,
            columns=(0, 1),
            frames=2,
        )
        reference_short = self.store(
            "reference-short",
            read_mode="fvb",
            exposure_ms=10,
            columns=(0, 1),
            frames=2,
        )
        _fake, installed = self.install_fake()
        with installed, self.assertRaisesRegex(
            ValueError, "at least three frames"
        ):
            load_racer_calibration(dark_short, reference_short)

    def test_rejects_mismatched_calibration_mode_exposure_and_geometry(self):
        cases = (
            (
                dict(read_mode="fvb", exposure_ms=10, columns=(0, 1)),
                dict(
                    read_mode="image",
                    exposure_ms=10,
                    rows=(0, 1),
                    columns=(0, 1),
                ),
                "read modes do not match",
            ),
            (
                dict(read_mode="fvb", exposure_ms=10, columns=(0, 1)),
                dict(read_mode="fvb", exposure_ms=11, columns=(0, 1)),
                "exposures do not match",
            ),
            (
                dict(read_mode="fvb", exposure_ms=10, columns=(0, 1)),
                dict(read_mode="fvb", exposure_ms=10, columns=(0, 2)),
                "geometry does not match",
            ),
            (
                dict(
                    read_mode="image",
                    exposure_ms=10,
                    rows=(0, 1),
                    columns=(0, 1),
                ),
                dict(
                    read_mode="image",
                    exposure_ms=10,
                    rows=(0, 2),
                    columns=(0, 1),
                ),
                "geometry does not match",
            ),
        )
        for index, (dark_spec, reference_spec, expected) in enumerate(cases):
            with self.subTest(expected=expected):
                dark = self.store(f"dark-{index}", **dark_spec)
                reference = self.store(f"reference-{index}", **reference_spec)
                _fake, installed = self.install_fake()
                with installed, self.assertRaisesRegex(ValueError, expected):
                    load_racer_calibration(dark, reference)

    def test_rejects_unsupported_store_mode(self):
        dark = self.store(
            "dark", read_mode="single_track", exposure_ms=10, columns=(0, 1)
        )
        reference = self.store(
            "reference", read_mode="single_track", exposure_ms=10, columns=(0, 1)
        )
        _fake, installed = self.install_fake()

        with installed, self.assertRaisesRegex(
            ValueError, "not a supported RACER read mode"
        ):
            load_racer_calibration(dark, reference)

    def test_missing_dependency_error_includes_install_command(self):
        dark = self.store(
            "dark", read_mode="fvb", exposure_ms=10, columns=(0, 1)
        )
        reference = self.store(
            "reference", read_mode="fvb", exposure_ms=10, columns=(0, 1)
        )
        missing = ModuleNotFoundError(
            "No module named 'racer_raman'", name="racer_raman"
        )

        with patch.object(importlib, "import_module", side_effect=missing) as import_module:
            with self.assertRaisesRegex(ImportError, "pip install") as raised:
                load_racer_calibration(dark, reference)
        import_module.assert_called_once_with("racer_raman")
        self.assertIn("racer-raman", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
