"""Editable fit parameters from the installed RACER public calibration API.

This module supplies UI metadata and validates options; the fits themselves
remain RACER's implementations. Defaults come from the installed signatures
so both the pinned release and newer local RACER versions are supported.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

__all__ = [
    "CalibrationParameterSpec",
    "GROUP_LABELS",
    "calibration_parameter_specs",
    "normalize_calibration_options",
]

GROUP_LABELS = {
    "image": "Image / 2D",
    "spectrum_dark": "FVB / 1D dark",
    "spectrum_reference": "FVB / 1D reference",
}
_FUNCTIONS = {
    "image": "calibrate_image",
    "spectrum_dark": "calibrate_spectrum_dark",
    "spectrum_reference": "calibrate_spectrum_reference",
}
_METADATA_PARAMETERS = {"gain_e_per_adu", "saturation_adu", "exposure_ms"}
_TOOLTIPS = {
    "profile_training_frames": "Number of illuminated frames used to train the optical profile.",
    "mad_to_sigma": "Scale factor converting the median absolute deviation to a noise standard deviation.",
    "pilot_frames": "Number of dark frames used for the initial robust dark estimate.",
    "offset_stride": "Sampling stride used when estimating frame offsets.",
    "variance_clip_sigma": "Clipping threshold, in noise standard deviations, for the dark variance fit.",
    "minimum_sigma_adu": "Minimum fitted noise standard deviation, in detector ADU.",
    "event_seed_sigma": "Dark-event seed threshold in noise standard deviations; must exceed the growth threshold.",
    "seed_sigma": "Dark-event seed threshold in noise standard deviations; must exceed the growth threshold.",
    "event_grow_sigma": "Threshold for including neighboring pixels in a dark event.",
    "growth_sigma": "Threshold for including neighboring pixels in a dark event.",
    "trace_half_width_rows": "Rows on either side of the fitted optical trace. A wider trace includes more detector rows; some rows must remain outside it.",
    "profile_smoothing_width_columns": "Smoothing window across detector columns for the optical profile.",
    "ridge_smoothing_width_columns": "Median-filter window across detector columns for the optical trace ridge.",
    "ridge_smoothing_sigma_columns": "Gaussian smoothing standard deviation across columns for the optical trace ridge.",
    "row_smoothing_width": "Smoothing window in detector rows for the dark-event probability.",
    "row_smoothing_sigma_rows": "Gaussian smoothing standard deviation in rows for the dark-event probability.",
    "minimum_row_probability": "Lower bound on the fitted per-row particle-event probability.",
    "maximum_row_probability": "Upper bound on the fitted per-row particle-event probability.",
    "maximum_amplitude_components": "Maximum number of components considered when fitting the particle-charge distribution.",
    "charge_ceiling_adu": "Charge threshold in ADU used to identify ceiling events during calibration.",
    "inner_validation_fraction": "Fraction of dark frames reserved for charge-prior model validation; strictly between zero and one.",
    "charge_prior_seed": "Random seed used when fitting the particle-charge prior.",
    "temporal_iterations": "Number of robust temporal-statistics fitting iterations.",
    "maximum_samples": "Maximum number of charge samples used to fit the spectrum particle calibration.",
    "seed": "Random seed used when sampling and fitting the spectrum particle calibration.",
}


@dataclass(frozen=True, slots=True)
class CalibrationParameterSpec:
    group: str
    name: str
    label: str
    default: int | float
    minimum: int | float
    maximum: int | float
    decimals: int
    tooltip: str

    @property
    def is_integer(self) -> bool:
        return isinstance(self.default, int)


def calibration_parameter_specs(
    racer: Any = None,
) -> dict[str, tuple[CalibrationParameterSpec, ...]]:
    """Describe numeric fit options supported by this RACER installation."""

    if racer is None:
        from .cosmic_ray_removal import _import_racer

        racer = _import_racer()
    result = {}
    for group, function_name in _FUNCTIONS.items():
        function = getattr(racer, function_name, None)
        if not callable(function):
            raise RuntimeError(
                "installed RACER does not expose "
                f"{function_name}; advanced calibration settings are unavailable"
            )
        specs = []
        for name, parameter in inspect.signature(function).parameters.items():
            default = parameter.default
            if (
                name in _METADATA_PARAMETERS
                or isinstance(default, bool)
                or not isinstance(default, Real)
                or parameter.kind
                not in (parameter.KEYWORD_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            ):
                continue
            if isinstance(default, Integral):
                default = int(default)
                minimum = 0 if name in {
                    "trace_half_width_rows", "charge_prior_seed", "seed"
                } or default == 0 else 1
                maximum = 2_147_483_647
                decimals = 0
            else:
                default = float(default)
                minimum, maximum, decimals = 1.0e-12, 1.0e12, 12
                if default == 0:
                    minimum = 0.0
                if "probability" in name or name.endswith("_fraction"):
                    maximum = 1.0 - 1.0e-12
            label = name.replace("_", " ").capitalize().replace("adu", "ADU")
            if name == "trace_half_width_rows":
                label = "Trace half-width (rows)"
            tooltip = _TOOLTIPS.get(name, f"RACER {name} fit parameter.")
            specs.append(CalibrationParameterSpec(
                group, name, label, default, minimum, maximum, decimals,
                f"{tooltip}\nRACER parameter: {name}. Default: {default:g}.",
            ))
        result[group] = tuple(specs)
    return result


def normalize_calibration_options(
    options: Mapping[str, Mapping[str, int | float]] | None,
    *,
    racer: Any = None,
    detector_rows: int | None = None,
) -> dict[str, dict[str, int | float]]:
    """Validate settings and return a copy containing only changed defaults.

    Unrecognized parameters fail explicitly rather than being silently ignored.
    Image geometry is checked when image overrides and a row count are supplied.
    Empty options do not import the optional RACER dependency.
    """

    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise ValueError("calibration options must be grouped by calibration type")
    for group, values in options.items():
        if group not in _FUNCTIONS:
            raise ValueError(f"unknown RACER calibration settings group: {group!r}")
        if not isinstance(values, Mapping):
            raise ValueError(f"RACER {group} settings must be a parameter mapping")
    if not any(options.values()):
        return {}

    specs_by_group = calibration_parameter_specs(racer)
    normalized = {}
    for group, supplied in options.items():
        if not supplied:
            continue
        specs = {spec.name: spec for spec in specs_by_group[group]}
        values = {name: spec.default for name, spec in specs.items()}
        changes = {}
        for name, value in supplied.items():
            if name not in specs:
                raise ValueError(
                    f"RACER {group} parameter {name!r} is not supported by "
                    "the installed calibration API"
                )
            spec = specs[name]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{group}.{name} must be a finite number")
            if not isinstance(value, Integral) and not math.isfinite(value):
                raise ValueError(f"{group}.{name} must be finite")
            if spec.is_integer and int(value) != value:
                raise ValueError(f"{group}.{name} must be an integer")
            if not spec.minimum <= value <= spec.maximum:
                raise ValueError(
                    f"{group}.{name} must be between {spec.minimum:g} "
                    f"and {spec.maximum:g}"
                )
            value = int(value) if spec.is_integer else float(value)
            values[name] = value
            if value != spec.default:
                changes[name] = value
        for seed, grow in (
            ("event_seed_sigma", "event_grow_sigma"),
            ("seed_sigma", "growth_sigma"),
        ):
            if seed in values and grow in values and values[seed] <= values[grow]:
                raise ValueError(f"{group}.{seed} must be greater than {grow}")
        minimum = values.get("minimum_row_probability")
        maximum = values.get("maximum_row_probability")
        if minimum is not None and maximum is not None and minimum >= maximum:
            raise ValueError(
                "minimum_row_probability must be below maximum_row_probability"
            )
        half_width = values.get("trace_half_width_rows")
        if (
            group == "image" and changes and detector_rows is not None
            and half_width is not None and 2 * half_width + 1 >= detector_rows
        ):
            raise ValueError(
                "trace_half_width_rows must leave detector rows outside the trace "
                f"(2 × half width + 1 must be less than {detector_rows})"
            )
        if changes:
            normalized[group] = changes
    return normalized
