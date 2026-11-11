"""Parameter validation and forwarding to RACER's existing calibration API."""

from __future__ import annotations

import copy
import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from napari_raman_widget.cosmic_ray_removal import load_racer_calibration
from napari_raman_widget.racer_settings import (
    calibration_parameter_specs,
    normalize_calibration_options,
)


def _fake_racer():
    calls = []

    def calibrate_image(
        dark,
        reference,
        instrument_background_adu,
        *,
        gain_e_per_adu=0.7,
        saturation_adu=65535.0,
        trace_half_width_rows=12,
        profile_training_frames=25,
        event_seed_sigma=8.0,
        event_grow_sigma=3.0,
        minimum_row_probability=1.0e-10,
        maximum_row_probability=0.05,
        inner_validation_fraction=0.2,
    ):
        calls.append(("image", dict(locals())))
        return types.SimpleNamespace()

    def calibrate_spectrum_dark(
        stack,
        *,
        exposure_ms,
        saturation_adu=65535.0,
        temporal_iterations=2,
        minimum_sigma_adu=0.5,
        seed_sigma=8.0,
        growth_sigma=3.0,
        maximum_samples=4000,
        seed=20260819,
    ):
        calls.append(("spectrum_dark", dict(locals())))
        return types.SimpleNamespace(variance_adu2=np.ones(stack.shape[-1]))

    def calibrate_spectrum_reference(
        stack,
        dark,
        *,
        saturation_adu=65535.0,
        temporal_iterations=2,
        minimum_sigma_adu=0.5,
    ):
        calls.append(("spectrum_reference", dict(locals())))
        return types.SimpleNamespace()

    class ImageCalibrator:
        def __init__(self, calibration=None, config=None, columns=None):
            self._calibration = calibration
            self.config = config
            self.columns = columns

        @classmethod
        def fit(cls, dark, reference, **kwargs):
            calls.append(("image_fit", kwargs))
            return cls()

    class SpectrumCalibrator:
        def __init__(
            self, dark=None, reference=None, saturation=None, config=None,
            columns=None,
        ):
            self._dark = dark
            self.reference = reference
            self.saturation = saturation
            self.config = config
            self.columns = columns

        @classmethod
        def fit(cls, dark, reference, **kwargs):
            calls.append(("spectrum_fit", kwargs))
            return cls()

    class ImageConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def validate(self):
            pass

    return types.SimpleNamespace(
        calibrate_image=calibrate_image,
        calibrate_spectrum_dark=calibrate_spectrum_dark,
        calibrate_spectrum_reference=calibrate_spectrum_reference,
        ImageCalibrator=ImageCalibrator,
        SpectrumCalibrator=SpectrumCalibrator,
        ImageConfig=ImageConfig,
        HybridSpectrumConfig=types.SimpleNamespace,
        calls=calls,
    )


class CalibrationParameterTests(unittest.TestCase):
    def setUp(self):
        self.racer = _fake_racer()

    def normalize(self, options, **kwargs):
        return normalize_calibration_options(options, racer=self.racer, **kwargs)

    def test_specs_use_installed_defaults_and_exclude_acquisition_metadata(self):
        def custom_image(dark, reference, *, trace_half_width_rows=17):
            pass

        self.racer.calibrate_image = custom_image
        groups = calibration_parameter_specs(self.racer)
        self.assertEqual(groups["image"][0].default, 17)
        self.assertTrue(groups["image"][0].is_integer)
        for group in groups.values():
            names = {spec.name for spec in group}
            self.assertTrue(names.isdisjoint(
                {"gain_e_per_adu", "saturation_adu", "exposure_ms"}
            ))

    def test_small_probability_default_has_enough_display_precision(self):
        specs = {
            spec.name: spec
            for spec in calibration_parameter_specs(self.racer)["image"]
        }
        probability = specs["minimum_row_probability"]
        self.assertEqual(round(probability.default, probability.decimals), 1e-10)
        self.assertLessEqual(probability.minimum, probability.default)

    def test_defaults_are_omitted_without_mutating_input(self):
        defaults = {
            group: {spec.name: spec.default for spec in specs}
            for group, specs in calibration_parameter_specs(self.racer).items()
        }
        original = copy.deepcopy(defaults)
        self.assertEqual(self.normalize(defaults), {})
        self.assertEqual(defaults, original)
        self.assertEqual(normalize_calibration_options(None), {})
        self.assertEqual(normalize_calibration_options({}), {})

    def test_normalization_copies_changes_and_preserves_integer_type(self):
        options = {"image": {"trace_half_width_rows": 64.0}}
        result = self.normalize(options, detector_rows=256)
        self.assertEqual(result, {"image": {"trace_half_width_rows": 64}})
        self.assertIsInstance(result["image"]["trace_half_width_rows"], int)
        result["image"]["trace_half_width_rows"] = 4
        self.assertEqual(options["image"]["trace_half_width_rows"], 64.0)

    def test_invalid_numeric_options_are_rejected(self):
        invalid = (
            ("trace_half_width_rows", -1),
            ("trace_half_width_rows", 4.5),
            ("trace_half_width_rows", True),
            ("trace_half_width_rows", "64"),
            ("profile_training_frames", 0),
            ("event_seed_sigma", float("nan")),
            ("event_seed_sigma", float("inf")),
            ("event_seed_sigma", -float("inf")),
            ("minimum_row_probability", 0.0),
            ("inner_validation_fraction", 1.0),
        )
        for name, value in invalid:
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(ValueError, name):
                    self.normalize({"image": {name: value}})

    def test_cross_parameter_constraints_use_unsupplied_defaults(self):
        invalid = (
            {"image": {"event_seed_sigma": 3.0}},
            {"image": {"event_grow_sigma": 8.0}},
            {"spectrum_dark": {"seed_sigma": 3.0}},
            {"spectrum_dark": {"growth_sigma": 8.0}},
            {"image": {"minimum_row_probability": 0.05}},
        )
        for options in invalid:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.normalize(options)

    def test_trace_width_must_leave_off_trace_detector_rows(self):
        with self.assertRaisesRegex(ValueError, "trace_half_width_rows"):
            self.normalize(
                {"image": {"trace_half_width_rows": 128}}, detector_rows=256
            )
        self.assertEqual(
            self.normalize(
                {"image": {"trace_half_width_rows": 127}}, detector_rows=256
            ),
            {"image": {"trace_half_width_rows": 127}},
        )

    def test_unknown_or_metadata_options_are_not_silently_ignored(self):
        invalid = (
            {"unknown": {}},
            {"image": {"not_a_racer_parameter": 4}},
            {"image": {"gain_e_per_adu": 1.2}},
            {"spectrum_dark": {"exposure_ms": 500}},
            {"image": []},
        )
        for options in invalid:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.normalize(options)

    def test_unavailable_public_calibrator_has_actionable_error(self):
        del self.racer.calibrate_image
        with self.assertRaisesRegex(RuntimeError, "calibrate_image"):
            calibration_parameter_specs(self.racer)


class CalibrationOptionsLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        # The loader canonicalizes paths, including Windows runner 8.3 aliases.
        self.root = Path(self.temporary_directory.name).resolve()
        self.dark_path = self.root / "dark.zarr"
        self.reference_path = self.root / "reference.zarr"
        self.dark_path.mkdir()
        self.reference_path.mkdir()
        self.racer = _fake_racer()

    def load(self, *, mode="image", options=None, reference=True, stacks=None):
        if stacks is None:
            shape = (3, 32, 4) if mode == "image" else (3, 4)
            dark = np.arange(np.prod(shape), dtype=float).reshape(shape)
            illuminated = dark + 1000.0
        else:
            dark, illuminated = stacks

        def read_store(racer, path, experiment):
            stack = dark if path == self.dark_path else illuminated
            levels = [[experiment], [1000.0], range(stack.shape[0])]
            names = ["experiment", "exposure_ms", "frame"]
            if mode == "image":
                levels.append(range(stack.shape[1]))
                names.append("row")
            index = pd.MultiIndex.from_product(levels, names=names)
            table = pd.DataFrame(
                stack.reshape(-1, stack.shape[-1]),
                index=index,
                columns=pd.Index(range(stack.shape[-1]), name="column"),
            )
            return mode, table

        def metadata(path):
            return {
                "racer_role": (
                    "closed_shutter" if path == self.dark_path else "open_reference"
                ),
                "units": "ADU",
                "gain_e_per_adu": 1.3,
                "saturation_adu": 32000.0,
            }

        with (
            patch("napari_raman_widget.cosmic_ray_removal._import_racer", return_value=self.racer),
            patch("napari_raman_widget.cosmic_ray_removal._read_store", side_effect=read_store),
            patch("napari_raman_widget.cosmic_ray_removal._zarr_metadata", side_effect=metadata),
        ):
            return load_racer_calibration(
                self.dark_path,
                self.reference_path if reference else None,
                calibration_options=options,
            )

    def test_custom_image_width_reaches_racer_with_arrays_and_zero_background(self):
        options = {
            "image": {"trace_half_width_rows": 4},
            "spectrum_dark": {"seed_sigma": 9.0},
        }
        calibration = self.load(options=options)
        self.assertEqual([name for name, _ in self.racer.calls], ["image"])
        arguments = self.racer.calls[0][1]
        self.assertEqual(arguments["dark"].shape, (3, 32, 4))
        self.assertEqual(arguments["reference"].shape, (3, 32, 4))
        self.assertEqual(arguments["trace_half_width_rows"], 4)
        self.assertEqual(arguments["gain_e_per_adu"], 1.3)
        self.assertEqual(arguments["saturation_adu"], 32000.0)
        np.testing.assert_array_equal(arguments["instrument_background_adu"], np.zeros((32, 4)))
        self.assertEqual(calibration._calibrator.columns, (0, 1, 2, 3))
        self.assertEqual(calibration._calibrator.config.gain_e_per_adu, 1.3)
        self.assertEqual(calibration.calibration_options, {"image": {"trace_half_width_rows": 4}})
        options["image"]["trace_half_width_rows"] = 8
        self.assertEqual(calibration.calibration_options["image"]["trace_half_width_rows"], 4)

    def test_default_values_keep_existing_public_fit_path(self):
        calibration = self.load(options={"image": {"trace_half_width_rows": 12}})
        self.assertEqual([name for name, _ in self.racer.calls], ["image_fit"])
        self.assertEqual(calibration.calibration_options, {})

    def test_fvb_dark_and_reference_options_go_to_the_matching_fit(self):
        calibration = self.load(
            mode="fvb",
            options={
                "spectrum_dark": {"seed_sigma": 9.0},
                "spectrum_reference": {"minimum_sigma_adu": 0.75},
            },
        )
        self.assertEqual(
            [name for name, _ in self.racer.calls],
            ["spectrum_dark", "spectrum_reference"],
        )
        dark = self.racer.calls[0][1]
        reference = self.racer.calls[1][1]
        self.assertEqual(dark["stack"].shape, (3, 4))
        self.assertEqual(dark["seed_sigma"], 9.0)
        self.assertEqual(dark["exposure_ms"], 1000.0)
        self.assertEqual(reference["minimum_sigma_adu"], 0.75)
        self.assertEqual(reference["saturation_adu"], 32000.0)
        self.assertIs(reference["dark"], calibration._calibrator._dark)
        self.assertEqual(calibration._calibrator.columns, (0, 1, 2, 3))

    def test_custom_dark_only_fvb_uses_dark_stack_for_reference(self):
        self.load(
            mode="fvb", reference=False,
            options={"spectrum_dark": {"maximum_samples": 200}},
        )
        dark = self.racer.calls[0][1]
        reference = self.racer.calls[1][1]
        self.assertEqual(dark["maximum_samples"], 200)
        np.testing.assert_array_equal(dark["stack"], reference["stack"])

    def test_unsupported_option_fails_before_any_fit(self):
        with self.assertRaisesRegex(ValueError, "unknown_parameter"):
            self.load(options={"image": {"unknown_parameter": 2}})
        self.assertEqual(self.racer.calls, [])

    @unittest.skipUnless(
        importlib.util.find_spec("racer_raman") is not None,
        "racer-raman is not installed",
    )
    def test_custom_image_width_fits_with_real_racer_public_api(self):
        import racer_raman

        self.racer = racer_raman
        rng = np.random.default_rng(20261109)
        frames, rows, columns = 40, 32, 24
        dark = rng.normal(100.0, 2.0, (frames, rows, columns))
        for frame in range(0, frames, 2):
            row = 2 + (frame * 3) % (rows - 4)
            column = 2 + frame % (columns - 4)
            dark[frame, row, column:column + 2] += 100.0 + frame * 3
        profile = np.exp(-0.5 * ((np.arange(rows) - 16) / 3.0) ** 2)
        reference = (
            100.0 + 500.0 * profile[None, :, None]
            + rng.normal(0.0, 2.0, (6, rows, columns))
        )
        calibration = self.load(
            options={"image": {"trace_half_width_rows": 4}},
            stacks=(dark, reference),
        )
        model = calibration._calibrator._calibration.model
        self.assertGreater(model.support_y.size, 0)
        self.assertLessEqual(np.bincount(model.support_x).max(), 9)
        raw = reference[0].copy()
        raw[16, 12] += 300.0
        original = raw.copy()
        corrected = calibration.correct(raw)
        np.testing.assert_array_equal(raw, original)
        self.assertEqual(corrected.shape, raw.shape)
        self.assertTrue(np.all(np.isfinite(corrected)))


if __name__ == "__main__":
    unittest.main()
