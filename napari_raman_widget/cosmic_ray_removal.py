"""Session-scoped RACER-Raman calibration and correction helpers.

The optional :mod:`cosmic_ray_bayes` dependency is imported only when a
calibration is loaded.  A loaded calibration owns one fitted upstream
calibrator and can therefore be shared by plot windows without refitting on
every redraw.
"""

from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .racer_settings import normalize_calibration_options

__all__ = ["RacerCalibration", "load_racer_calibration"]


_SUPPORTED_READ_MODES = ("fvb", "image")
_DEFAULT_GAIN_E_PER_ADU = 0.7
_DEFAULT_SATURATION_ADU = 65535.0
_IMAGE_OFF_SUPPORT_SEED_SIGMA = 8.0
_IMAGE_OFF_SUPPORT_GROWTH_SIGMA = 4.0
_IMAGE_OFF_SUPPORT_MAX_PIXELS = 100
_IMAGE_OFF_SUPPORT_MAX_WIDTH = 10


def _image_off_support_calibration(
    calibrator: Any,
    detector_shape: tuple[int, int],
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Extract the pinned RACER image model needed outside trace support.

    RACER's public image correction deliberately operates only on its optical
    trace support.  The fitted object still owns robust closed-shutter master
    and variance images, which let us conservatively clean positive particle
    events elsewhere on the detector.  Keep this adapter optional so a future
    upstream object layout fails closed to trace-only correction.
    """

    fitted = getattr(calibrator, "_calibration", None)
    model = getattr(fitted, "model", None)
    try:
        background = np.asarray(fitted.background_adu, dtype=float)
        master = np.asarray(fitted.dark_master_adu, dtype=float)
        variance = np.asarray(fitted.dark_variance_adu2, dtype=float)
        support_y = np.asarray(model.support_y, dtype=int)
        support_x = np.asarray(model.support_x, dtype=int)
    except (AttributeError, TypeError, ValueError):
        return None, None, None, None
    diffuse = getattr(fitted, "diffuse_photo_adu", None)
    if diffuse is not None:
        try:
            diffuse = np.asarray(diffuse, dtype=float)
        except (TypeError, ValueError):
            return None, None, None, None
        if (
            diffuse.shape != (detector_shape[1],)
            or np.any(~np.isfinite(diffuse))
        ):
            return None, None, None, None
        # Newer RACER revisions keep the diffuse photo level separate from
        # D + B^inst.  Off-trace event detection needs the complete expected
        # background, while older revisions already include it here.
        background = background + diffuse[None, :]
    if (
        background.shape != detector_shape
        or master.shape != detector_shape
        or variance.shape != detector_shape
        or np.any(~np.isfinite(background))
        or np.any(~np.isfinite(master))
        or np.any(~np.isfinite(variance))
        or np.any(variance <= 0)
        or support_y.shape != support_x.shape
        or np.any((support_y < 0) | (support_y >= detector_shape[0]))
        or np.any((support_x < 0) | (support_x >= detector_shape[1]))
    ):
        return None, None, None, None
    support = np.zeros(detector_shape, dtype=bool)
    support[support_y, support_x] = True
    return background, master, variance, support


def _fvb_dark_sigma(calibrator: Any, detector_length: int) -> np.ndarray | None:
    """Return RACER's robust closed-shutter sigma for repeat consensus."""

    fitted_dark = getattr(calibrator, "_dark", None)
    try:
        variance = np.asarray(fitted_dark.variance_adu2, dtype=float)
    except (AttributeError, TypeError, ValueError):
        return None
    if (
        variance.shape != (detector_length,)
        or np.any(~np.isfinite(variance))
        or np.any(variance <= 0)
    ):
        return None
    return np.sqrt(variance)


def _clean_image_off_support(
    raw: np.ndarray,
    corrected: np.ndarray,
    *,
    background: np.ndarray | None,
    dark_master: np.ndarray | None,
    dark_variance: np.ndarray | None,
    support_mask: np.ndarray | None,
    gain_e_per_adu: float,
    saturation_adu: float,
) -> np.ndarray:
    """Clean positive dark-calibrated events outside RACER's trace support."""

    if (
        background is None
        or dark_master is None
        or dark_variance is None
        or support_mask is None
    ):
        return corrected
    available = (
        ~support_mask
        & np.isfinite(raw)
        & (raw < float(saturation_adu))
    )
    if not np.any(available):
        return corrected
    frame_offset = float(np.median((raw - background)[available]))
    expected = background + frame_offset
    observation_variance = dark_variance + (
        np.clip(expected - dark_master, 0.0, None) / float(gain_e_per_adu)
    )
    standardized = np.zeros(raw.shape, dtype=float)
    standardized[available] = (
        (raw[available] - expected[available])
        / np.sqrt(observation_variance[available])
    )
    seeds = available & (standardized >= _IMAGE_OFF_SUPPORT_SEED_SIGMA)
    if not np.any(seeds):
        return corrected
    growth = available & (standardized >= _IMAGE_OFF_SUPPORT_GROWTH_SIGMA)
    particle_mask = ndi.binary_propagation(
        seeds,
        structure=np.ones((3, 3), dtype=bool),
        mask=growth,
    )
    if not np.any(particle_mask):
        return corrected
    labels, label_count = ndi.label(
        particle_mask, structure=np.ones((3, 3), dtype=bool)
    )
    guarded = np.zeros(raw.shape, dtype=bool)
    for label in range(1, int(label_count) + 1):
        rows, columns = np.nonzero(labels == label)
        if rows.size == 0 or rows.size > _IMAGE_OFF_SUPPORT_MAX_PIXELS:
            continue
        width = int(columns.max() - columns.min() + 1)
        if width <= _IMAGE_OFF_SUPPORT_MAX_WIDTH:
            guarded[rows, columns] = True
    if not np.any(guarded):
        return corrected
    result = np.array(corrected, dtype=float, copy=True)
    result[guarded] = expected[guarded]
    return result


def _import_racer() -> Any:
    """Import RACER-Raman with an actionable optional-dependency error."""

    try:
        module = importlib.import_module("cosmic_ray_bayes")
    except ModuleNotFoundError as error:
        if error.name == "cosmic_ray_bayes":
            raise ImportError(
                "RACER-Raman is not installed. Install it with "
                "`pip install \"cosmic-ray-bayes[io] @ "
                "git+https://github.com/JasonYu1/racer-raman.git@"
                "364f201736567416989eb5b2a55ae34fa7b250ff\"`."
            ) from error
        raise ImportError(
            "RACER-Raman could not be imported because its dependency "
            f"{error.name!r} is missing. Reinstall cosmic-ray-bayes and its "
            "dependencies."
        ) from error

    required = (
        "ImageCalibrator",
        "SpectrumCalibrator",
        "read_image_zarr",
        "read_spectrum_zarr",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ImportError(
            "The installed cosmic-ray-bayes package is incompatible; missing "
            f"public API: {', '.join(missing)}."
        )
    return module


def _zarr_directory(path: str | Path, label: str) -> Path:
    result = Path(path).expanduser()
    if not result.is_dir():
        raise FileNotFoundError(
            f"{label} must be an existing Zarr directory: {result}"
        )
    return result.resolve()


def _read_store(racer: Any, path: Path, experiment: str) -> tuple[str, Any]:
    """Infer one supported read mode by using RACER's public Zarr readers."""

    failures: list[tuple[str, ValueError]] = []
    for read_mode, reader_name in (
        ("fvb", "read_spectrum_zarr"),
        ("image", "read_image_zarr"),
    ):
        reader = getattr(racer, reader_name)
        try:
            table = reader({experiment: path})
        except ValueError as error:
            failures.append((read_mode, error))
            continue
        return read_mode, table

    details = "; ".join(
        f"{read_mode}: {error}" for read_mode, error in failures
    )
    raise ValueError(
        f"{path} is not a supported RACER read mode ('fvb' or 'image')"
        + (f": {details}" if details else "")
    )


def _zarr_metadata(path: Path) -> dict[str, Any]:
    """Return root attributes from one calibration store."""

    try:
        zarr = importlib.import_module("zarr")
    except ModuleNotFoundError as error:
        raise ImportError(
            "RACER Zarr calibration loading requires zarr; reinstall "
            "napari-raman-widget with its declared dependencies."
        ) from error
    try:
        root = zarr.open_group(str(path), mode="r")
        return dict(root.attrs)
    except Exception as error:
        raise ValueError(
            f"could not read calibration metadata from {path}: {error}"
        ) from error


def _positive_metadata_value(
    closed: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
    name: str,
    default: float,
) -> tuple[float, str | None]:
    """Resolve one detector parameter and report default use."""

    closed_present = name in closed
    reference_present = reference is not None and name in reference
    if reference is not None and closed_present != reference_present:
        raise ValueError(
            f"calibration metadata {name!r} is present in only one store"
        )
    if not closed_present:
        return float(default), (
            f"{name} is absent; using RACER's documented default "
            f"{default:g}"
        )
    try:
        closed_value = float(closed[name])
        reference_value = (
            closed_value if reference is None else float(reference[name])
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"calibration metadata {name!r} must be numeric"
        ) from error
    if (
        not np.isfinite(closed_value)
        or closed_value <= 0
        or not np.isfinite(reference_value)
        or reference_value <= 0
    ):
        raise ValueError(
            f"calibration metadata {name!r} must be finite and positive"
        )
    if closed_value != reference_value:
        raise ValueError(
            f"calibration metadata {name!r} does not match between stores"
        )
    return closed_value, None


def _detector_signature(
    closed: Mapping[str, Any], reference: Mapping[str, Any] | None
) -> tuple[str | None, str | None]:
    """Return one canonical acquisition signature from the selected stores."""

    name = "racer_detector_signature"
    closed_present = name in closed
    reference_present = reference is not None and name in reference
    if reference is not None and closed_present != reference_present:
        raise ValueError(
            f"calibration metadata {name!r} is present in only one store"
        )
    if not closed_present:
        return None, (
            "detector acquisition signature is absent; gain/ROI/readout "
            "matching cannot be fully verified"
        )

    def canonical(value: Any) -> str:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"calibration metadata {name!r} is not valid JSON"
                ) from error
        if not isinstance(value, Mapping):
            raise ValueError(
                f"calibration metadata {name!r} must be an object"
            )
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"calibration metadata {name!r} must contain JSON values"
            ) from error

    closed_signature = canonical(closed[name])
    if reference is not None:
        reference_signature = canonical(reference[name])
        if closed_signature != reference_signature:
            raise ValueError(
                "calibration detector acquisition settings do not match "
                "between closed-shutter and open-reference stores"
            )
    signature_values = json.loads(closed_signature)
    warning_parts = []
    if signature_values.get("gain_source") == "racer_default_assumption":
        warning_parts.append(
            "gain_e_per_adu and saturation_adu use RACER's 0.7 and 65535 "
            "default assumptions; verify both values for the detector before "
            "scientific use"
        )
    if not signature_values.get("camera_serial"):
        warning_parts.append(
            "camera serial is unavailable; detector identity cannot be "
            "fully verified"
        )
    if signature_values.get("actual_exposure_ms") is None:
        warning_parts.append(
            "actual SDK exposure timing is unavailable; requested exposure "
            "is used for matching"
        )
    warning = "; ".join(warning_parts) or None
    return closed_signature, warning


def _store_metadata_warnings(
    closed: Mapping[str, Any], reference: Mapping[str, Any] | None
) -> tuple[str, ...]:
    """Validate store roles/units and report unverifiable legacy metadata."""

    warnings = []
    role_specs = [(closed, "closed-shutter", {"closed_shutter"})]
    if reference is not None:
        role_specs.append(
            (
                reference,
                "open-reference",
                {"open_reference", "open_reference_candidate"},
            )
        )
    for metadata, label, accepted_roles in role_specs:
        role = metadata.get("racer_role")
        if role is None:
            warnings.append(
                f"{label} store has no racer_role; its optical role cannot "
                "be verified"
            )
        elif str(role) not in accepted_roles:
            expected = " or ".join(sorted(accepted_roles))
            raise ValueError(
                f"{label} store has racer_role={role!r}; expected {expected}"
            )

        units = metadata.get("units")
        if units is None:
            warnings.append(
                f"{label} store has no units metadata; ADU is assumed"
            )
        elif str(units).strip().upper() != "ADU":
            raise ValueError(
                f"{label} store units must be 'ADU', not {units!r}"
            )

    if (
        reference is not None
        and reference.get("racer_role") == "open_reference_candidate"
    ):
        warnings.append(
            "the open-reference store is labeled as a candidate; verify it "
            "is a stable, representative open reference"
        )
    return tuple(warnings)


@dataclass(frozen=True, slots=True)
class _StoreCalibrationData:
    table: Any = field(repr=False)
    exposure_ms: float
    columns: tuple[Any, ...]
    rows: tuple[Any, ...] | None

    @property
    def detector_shape(self) -> tuple[int, ...]:
        if self.rows is None:
            return (len(self.columns),)
        return (len(self.rows), len(self.columns))


def _calibration_stack(data: _StoreCalibrationData) -> np.ndarray:
    """Convert a validated table for RACER's low-level numeric fit functions.

    Retain the public wrapper's complete, ordered frame-row-grid check before
    reshaping image data, so an override cannot change detector coordinates.
    """

    values = np.asarray(data.table.to_numpy())
    if values.dtype.kind not in "uif":
        raise ValueError("calibration must contain numeric detector values")
    if data.rows is None:
        return values
    index = data.table.index
    frames = index.get_level_values("frame")
    frame_ids = frames.unique()
    rows = index.get_level_values("row")
    if (
        bool(index.has_duplicates)
        or len(index) != len(frame_ids) * len(data.rows)
        or not np.array_equal(frames, np.repeat(frame_ids, len(data.rows)))
        or not np.array_equal(rows, np.tile(data.rows, len(frame_ids)))
    ):
        raise ValueError(
            "image calibration must contain a complete, ordered frame-row grid"
        )
    return values.reshape(len(frame_ids), len(data.rows), len(data.columns))


def _calibration_data(
    table: Any,
    *,
    experiment: str,
    read_mode: str,
) -> _StoreCalibrationData:
    """Extract one experiment while retaining detector coordinate labels."""

    if not all(hasattr(table, name) for name in ("columns", "index", "to_numpy")):
        raise TypeError("RACER Zarr readers must return a pandas DataFrame")

    index_names = tuple(table.index.names)
    for required in ("experiment", "exposure_ms"):
        if required not in index_names:
            raise ValueError(
                f"{experiment} calibration is missing the {required!r} index"
            )

    experiments = tuple(table.index.get_level_values("experiment").unique())
    if experiments != (experiment,):
        raise ValueError(
            f"{experiment} calibration must contain exactly that experiment"
        )

    raw_exposures = tuple(
        table.index.get_level_values("exposure_ms").unique()
    )
    if len(raw_exposures) != 1:
        raise ValueError(
            f"{experiment} calibration must contain exactly one exposure"
        )
    try:
        exposure_ms = float(raw_exposures[0])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{experiment} calibration exposure_ms must be numeric"
        ) from error
    if not np.isfinite(exposure_ms) or exposure_ms <= 0:
        raise ValueError(
            f"{experiment} calibration exposure_ms must be finite and positive"
        )

    selected = table.xs(
        (experiment, raw_exposures[0]),
        level=("experiment", "exposure_ms"),
    )
    columns = tuple(selected.columns.tolist())
    if not columns or bool(getattr(selected.columns, "has_duplicates", False)):
        raise ValueError(
            f"{experiment} calibration detector columns must be unique and nonempty"
        )

    if read_mode == "fvb":
        if tuple(selected.index.names) != ("frame",):
            raise ValueError(
                f"{experiment} FVB calibration must be indexed by frame"
            )
        rows = None
    else:
        if tuple(selected.index.names) != ("frame", "row"):
            raise ValueError(
                f"{experiment} image calibration must be indexed by frame and row"
            )
        rows = tuple(selected.index.get_level_values("row").unique().tolist())
        if not rows:
            raise ValueError(
                f"{experiment} image calibration must contain detector rows"
            )

    frame_count = len(selected.index.get_level_values("frame").unique())
    if frame_count < 3:
        raise ValueError(
            f"{experiment} calibration must contain at least three frames; "
            f"found {frame_count}"
        )

    return _StoreCalibrationData(
        table=selected,
        exposure_ms=exposure_ms,
        columns=columns,
        rows=rows,
    )


@dataclass(frozen=True, slots=True)
class RacerCalibration:
    """One fitted, session-scoped RACER calibration.

    ``detector_shape`` describes a measurement, excluding its optional repeat
    axis: ``(detector_x,)`` for FVB and ``(detector_y, detector_x)`` for image
    readout.
    """

    read_mode: str
    exposure_ms: float
    detector_shape: tuple[int, ...]
    detector_columns: tuple[Any, ...]
    detector_rows: tuple[Any, ...] | None
    gain_e_per_adu: float
    saturation_adu: float
    detector_signature: str | None
    metadata_warnings: tuple[str, ...]
    closed_shutter_path: Path
    open_reference_path: Path | None
    _calibrator: Any = field(repr=False, compare=False)
    _fvb_dark_sigma_adu: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    _image_background_adu: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    _image_dark_master_adu: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    _image_dark_variance_adu2: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    _image_support_mask: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    calibration_options: Mapping[str, Mapping[str, int | float]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.read_mode not in _SUPPORTED_READ_MODES:
            raise ValueError(
                "read_mode must be one of " + ", ".join(_SUPPORTED_READ_MODES)
            )
        expected_dimensions = 1 if self.read_mode == "fvb" else 2
        shape = tuple(int(length) for length in self.detector_shape)
        if len(shape) != expected_dimensions or any(length < 1 for length in shape):
            raise ValueError("detector_shape does not match the RACER read mode")
        exposure = float(self.exposure_ms)
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError("exposure_ms must be finite and positive")
        columns = tuple(self.detector_columns)
        rows = None if self.detector_rows is None else tuple(self.detector_rows)
        if len(columns) != shape[-1] or len(set(columns)) != len(columns):
            raise ValueError("detector_columns must match detector_shape")
        if self.read_mode == "image" and (
            rows is None
            or len(rows) != shape[-2]
            or len(set(rows)) != len(rows)
        ):
            raise ValueError("detector_rows must match detector_shape")
        if self.read_mode == "fvb" and rows is not None:
            raise ValueError("FVB calibration must not define detector_rows")
        gain = float(self.gain_e_per_adu)
        saturation = float(self.saturation_adu)
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError("gain_e_per_adu must be finite and positive")
        if not np.isfinite(saturation) or saturation <= 0:
            raise ValueError("saturation_adu must be finite and positive")
        warnings = tuple(str(message).strip() for message in self.metadata_warnings)
        if any(not message for message in warnings):
            raise ValueError("metadata_warnings must not contain empty messages")
        object.__setattr__(self, "detector_shape", shape)
        object.__setattr__(self, "exposure_ms", exposure)
        object.__setattr__(self, "detector_columns", columns)
        object.__setattr__(self, "detector_rows", rows)
        object.__setattr__(self, "gain_e_per_adu", gain)
        object.__setattr__(self, "saturation_adu", saturation)
        object.__setattr__(self, "metadata_warnings", warnings)
        object.__setattr__(
            self,
            "calibration_options",
            MappingProxyType({
                group: MappingProxyType(dict(values))
                for group, values in self.calibration_options.items()
            }),
        )
        object.__setattr__(
            self, "closed_shutter_path", Path(self.closed_shutter_path).resolve()
        )
        reference_path = self.open_reference_path
        if self.read_mode == "image" and reference_path is None:
            raise ValueError("Image calibration requires an open-reference path")
        object.__setattr__(
            self,
            "open_reference_path",
            None if reference_path is None else Path(reference_path).resolve(),
        )

        fvb_sigma = self._fvb_dark_sigma_adu
        if fvb_sigma is not None:
            fvb_sigma = np.array(fvb_sigma, dtype=float, copy=True)
            if (
                self.read_mode != "fvb"
                or fvb_sigma.shape != shape
                or np.any(~np.isfinite(fvb_sigma))
                or np.any(fvb_sigma <= 0)
            ):
                raise ValueError("FVB dark sigma must match detector_shape")
            fvb_sigma.setflags(write=False)
            object.__setattr__(self, "_fvb_dark_sigma_adu", fvb_sigma)

        image_arrays = (
            self._image_background_adu,
            self._image_dark_master_adu,
            self._image_dark_variance_adu2,
            self._image_support_mask,
        )
        present = tuple(value is not None for value in image_arrays)
        if any(present) and not all(present):
            raise ValueError(
                "full-frame image cleanup requires background, dark master, "
                "dark variance, and support mask"
            )
        if all(present):
            if self.read_mode != "image":
                raise ValueError("full-frame image cleanup requires image mode")
            background = np.array(image_arrays[0], dtype=float, copy=True)
            master = np.array(image_arrays[1], dtype=float, copy=True)
            variance = np.array(image_arrays[2], dtype=float, copy=True)
            support = np.array(image_arrays[3], dtype=bool, copy=True)
            if (
                background.shape != shape
                or master.shape != shape
                or variance.shape != shape
                or support.shape != shape
                or np.any(~np.isfinite(background))
                or np.any(~np.isfinite(master))
                or np.any(~np.isfinite(variance))
                or np.any(variance <= 0)
            ):
                raise ValueError(
                    "full-frame image cleanup arrays must match detector_shape"
                )
            for value in (background, master, variance, support):
                value.setflags(write=False)
            object.__setattr__(self, "_image_background_adu", background)
            object.__setattr__(self, "_image_dark_master_adu", master)
            object.__setattr__(self, "_image_dark_variance_adu2", variance)
            object.__setattr__(self, "_image_support_mask", support)

    @property
    def source_paths(self) -> tuple[Path, Path | None]:
        """Closed-shutter and optional open-reference paths, in that order."""

        return self.closed_shutter_path, self.open_reference_path

    @property
    def fvb_dark_sigma_adu(self) -> np.ndarray | None:
        """Robust per-column closed-shutter sigma, when RACER exposes it."""

        return self._fvb_dark_sigma_adu

    @property
    def full_frame_image_cleanup(self) -> bool:
        """Whether image correction also covers compact off-trace events."""

        return self._image_background_adu is not None

    def compatibility(
        self,
        values: Any,
        exposure_ms: float,
        read_mode: str,
        *,
        detector_columns: Any = None,
        detector_rows: Any = None,
        detector_signature: Any = None,
    ) -> tuple[bool, str]:
        """Return whether detector values match this fitted calibration."""

        if read_mode not in _SUPPORTED_READ_MODES:
            return (
                False,
                f"unsupported read mode {read_mode!r}; expected 'fvb' or 'image'",
            )
        if read_mode != self.read_mode:
            return (
                False,
                "read mode mismatch: calibration uses "
                f"{self.read_mode!r}, data use {read_mode!r}",
            )
        if isinstance(exposure_ms, (bool, np.bool_)):
            return False, "exposure_ms must be a finite positive number"
        try:
            exposure = float(exposure_ms)
        except (TypeError, ValueError):
            return False, "exposure_ms must be a finite positive number"
        if not np.isfinite(exposure) or exposure <= 0:
            return False, "exposure_ms must be a finite positive number"
        if exposure != self.exposure_ms:
            return (
                False,
                "exposure mismatch: calibration uses "
                f"{self.exposure_ms:g} ms, data use {exposure:g} ms",
            )

        try:
            array = np.asarray(values)
        except Exception as error:
            return False, f"detector values cannot be converted to an array: {error}"
        if array.dtype.kind not in "uif":
            return False, "detector values must be numeric"

        if self.read_mode == "fvb":
            if array.ndim not in (1, 2):
                return (
                    False,
                    "FVB values must have shape (detector_x,) or "
                    "(repeats, detector_x)",
                )
            if array.ndim == 2 and array.shape[0] == 0:
                return False, "FVB values must contain at least one spectrum"
            actual_shape = (array.shape[-1],)
        else:
            if array.ndim not in (2, 3):
                return (
                    False,
                    "image values must have shape (detector_y, detector_x) or "
                    "(repeats, detector_y, detector_x)",
                )
            if array.ndim == 3 and array.shape[0] == 0:
                return False, "image values must contain at least one frame"
            actual_shape = tuple(array.shape[-2:])

        if actual_shape != self.detector_shape:
            return (
                False,
                "detector geometry mismatch: calibration expects "
                f"{self.detector_shape}, data have {actual_shape}",
            )
        actual_columns = (
            tuple(range(actual_shape[-1]))
            if detector_columns is None
            else tuple(detector_columns)
        )
        if actual_columns != self.detector_columns:
            return (
                False,
                "detector column ROI/binning mismatch: calibration uses "
                f"{self.detector_columns}, data use {actual_columns}",
            )
        if self.read_mode == "image":
            actual_rows = (
                tuple(range(actual_shape[-2]))
                if detector_rows is None
                else tuple(detector_rows)
            )
            if actual_rows != self.detector_rows:
                return (
                    False,
                    "detector row ROI/binning mismatch: calibration uses "
                    f"{self.detector_rows}, data use {actual_rows}",
                )
        if detector_signature is not None:
            try:
                if isinstance(detector_signature, str):
                    current_values = json.loads(detector_signature)
                else:
                    current_values = detector_signature
            except (TypeError, ValueError, json.JSONDecodeError):
                return False, "current detector acquisition signature is invalid"
            if not isinstance(current_values, Mapping):
                return False, "current detector acquisition signature is invalid"
            current_values = dict(current_values)
            if self.detector_signature is None:
                return (
                    False,
                    "calibration lacks the detector acquisition signature "
                    "needed to verify gain, binning, and readout settings",
                )
            expected_values = json.loads(self.detector_signature)
            if current_values.get("actual_exposure_ms") is None:
                current_values["actual_exposure_ms"] = None
                expected_values["actual_exposure_ms"] = None
            current_signature = json.dumps(
                current_values, sort_keys=True, separators=(",", ":")
            )
            expected_signature = json.dumps(
                expected_values, sort_keys=True, separators=(",", ":")
            )
            if current_signature != expected_signature:
                return (
                    False,
                    "detector acquisition settings do not match the calibration",
                )
        return True, ""

    def correct(
        self,
        values: Any,
        *,
        detector_columns: Any = None,
        detector_rows: Any = None,
    ) -> np.ndarray:
        """Correct values without mutating them or refitting the calibration."""

        compatible, reason = self.compatibility(
            values,
            exposure_ms=self.exposure_ms,
            read_mode=self.read_mode,
            detector_columns=detector_columns,
            detector_rows=detector_rows,
        )
        if not compatible:
            raise ValueError(reason)

        raw = np.array(values, copy=True)
        def correct_image(frame: np.ndarray) -> np.ndarray:
            trace_corrected = np.asarray(self._calibrator.correct(frame))
            return _clean_image_off_support(
                frame,
                trace_corrected,
                background=self._image_background_adu,
                dark_master=self._image_dark_master_adu,
                dark_variance=self._image_dark_variance_adu2,
                support_mask=self._image_support_mask,
                gain_e_per_adu=self.gain_e_per_adu,
                saturation_adu=self.saturation_adu,
            )

        if self.read_mode == "fvb":
            if raw.ndim == 1:
                corrected = self._calibrator.correct(raw)
            else:
                corrected = self._calibrator.correct_many(raw)
        elif raw.ndim == 2:
            corrected = correct_image(raw)
        else:
            corrected = np.stack(
                [correct_image(frame.copy()) for frame in raw],
                axis=0,
            )

        result = np.array(corrected, copy=True)
        if result.shape != raw.shape:
            raise RuntimeError(
                "RACER returned detector data with an unexpected shape: "
                f"{result.shape} != {raw.shape}"
            )
        return result


def load_racer_calibration(
    closed_shutter_path: str | Path,
    open_reference_path: str | Path | None = None,
    *,
    calibration_options: Mapping[str, Mapping[str, int | float]] | None = None,
) -> RacerCalibration:
    """Load RACER stores and fit one reusable calibration.

    Omitting ``open_reference_path`` enables a dark-variance-only fallback for
    FVB spectra. Image/2D calibration still requires an illuminated reference
    because RACER derives its spatial optical profile from that acquisition.
    ``calibration_options`` contains nondefault fit parameters grouped as
    ``image``, ``spectrum_dark``, and ``spectrum_reference``. They are forwarded
    to RACER's public calibrators without changing its correction algorithms.
    """

    closed_path = _zarr_directory(closed_shutter_path, "closed_shutter_path")
    reference_path = (
        None
        if open_reference_path is None
        else _zarr_directory(open_reference_path, "open_reference_path")
    )
    if reference_path is not None and closed_path == reference_path:
        raise ValueError(
            "closed-shutter and open-reference calibrations must be separate "
            "Zarr directories"
        )

    racer = _import_racer()
    options = normalize_calibration_options(calibration_options, racer=racer)
    closed_metadata = _zarr_metadata(closed_path)
    reference_metadata = (
        None if reference_path is None else _zarr_metadata(reference_path)
    )
    store_warnings = _store_metadata_warnings(
        closed_metadata, reference_metadata
    )
    closed_mode, closed_table = _read_store(
        racer, closed_path, "closed_shutter"
    )
    closed = _calibration_data(
        closed_table,
        experiment="closed_shutter",
        read_mode=closed_mode,
    )
    if reference_path is None:
        if closed_mode != "fvb":
            raise ValueError(
                "an illuminated open-reference acquisition is required for "
                "RACER-2D Image calibration; dark-only variance is supported "
                "only for FVB/RACER-1D"
            )
        reference = None
    else:
        reference_mode, reference_table = _read_store(
            racer, reference_path, "open_reference"
        )
        if closed_mode != reference_mode:
            raise ValueError(
                "calibration read modes do not match: closed-shutter uses "
                f"{closed_mode!r}, open-reference uses {reference_mode!r}"
            )
        reference = _calibration_data(
            reference_table,
            experiment="open_reference",
            read_mode=reference_mode,
        )
        if closed.exposure_ms != reference.exposure_ms:
            raise ValueError(
                "calibration exposures do not match: closed-shutter uses "
                f"{closed.exposure_ms:g} ms, open-reference uses "
                f"{reference.exposure_ms:g} ms"
            )
        if closed.columns != reference.columns or closed.rows != reference.rows:
            raise ValueError(
                "calibration detector geometry does not match between "
                "closed-shutter and open-reference acquisitions"
            )

    active_groups = (
        ("spectrum_dark", "spectrum_reference")
        if closed_mode == "fvb" else ("image",)
    )
    applied_options = {
        group: options[group] for group in active_groups if group in options
    }
    if closed_mode == "image" and applied_options:
        applied_options = normalize_calibration_options(
            applied_options, racer=racer, detector_rows=len(closed.rows)
        )

    gain_e_per_adu, gain_warning = _positive_metadata_value(
        closed_metadata,
        reference_metadata,
        "gain_e_per_adu",
        _DEFAULT_GAIN_E_PER_ADU,
    )
    saturation_adu, saturation_warning = _positive_metadata_value(
        closed_metadata,
        reference_metadata,
        "saturation_adu",
        _DEFAULT_SATURATION_ADU,
    )
    detector_signature, signature_warning = _detector_signature(
        closed_metadata, reference_metadata
    )
    warnings = store_warnings + tuple(
        message
        for message in (gain_warning, saturation_warning, signature_warning)
        if message is not None
    )
    if reference is None:
        warnings += (
            "FVB calibration uses closed-shutter variance only; "
            "signal-dependent shot noise is not modeled, so bright sample "
            "features may be over-flagged",
        )

    fvb_dark_sigma = None
    image_background = None
    image_dark_master = None
    image_dark_variance = None
    image_support = None
    if closed_mode == "fvb":
        if applied_options:
            dark_frames = _calibration_stack(closed)
            reference_frames = (
                dark_frames if reference is None else _calibration_stack(reference)
            )
            dark_calibration = racer.calibrate_spectrum_dark(
                dark_frames,
                exposure_ms=closed.exposure_ms,
                saturation_adu=saturation_adu,
                **applied_options.get("spectrum_dark", {}),
            )
            reference_calibration = racer.calibrate_spectrum_reference(
                reference_frames,
                dark_calibration,
                saturation_adu=saturation_adu,
                **applied_options.get("spectrum_reference", {}),
            )
            calibrator = racer.SpectrumCalibrator(
                dark_calibration,
                reference_calibration,
                saturation_adu,
                racer.HybridSpectrumConfig(),
                closed.columns,
            )
        else:
            calibrator = racer.SpectrumCalibrator.fit(
                closed.table,
                closed.table if reference is None else reference.table,
                exposure_ms=closed.exposure_ms,
                saturation_adu=saturation_adu,
            )
        fvb_dark_sigma = _fvb_dark_sigma(
            calibrator, closed.detector_shape[-1]
        )
    else:
        image_fit = (
            racer.calibrate_image if applied_options else racer.ImageCalibrator.fit
        )
        fit_parameters = inspect.signature(image_fit).parameters
        image_fit_args = (
            [_calibration_stack(closed), _calibration_stack(reference)]
            if applied_options else [closed.table, reference.table]
        )
        if "instrument_background_adu" in fit_parameters:
            image_fit_args.append(np.zeros(closed.detector_shape, dtype=float))
            warnings += (
                "installed RACER requires a separately measured instrument-"
                "background map, but these stores do not contain one; a zero "
                "map is assumed. Use a measured exposure-series background "
                "for quantitative image correction",
            )
        image_result = image_fit(
            *image_fit_args,
            gain_e_per_adu=gain_e_per_adu,
            saturation_adu=saturation_adu,
            **applied_options.get("image", {}),
        )
        if applied_options:
            config = racer.ImageConfig(
                gain_e_per_adu=gain_e_per_adu, saturation_adu=saturation_adu
            )
            config.validate()
            calibrator = racer.ImageCalibrator(image_result, config, closed.columns)
        else:
            calibrator = image_result
        (
            image_background,
            image_dark_master,
            image_dark_variance,
            image_support,
        ) = _image_off_support_calibration(calibrator, closed.detector_shape)

    return RacerCalibration(
        read_mode=closed_mode,
        exposure_ms=closed.exposure_ms,
        detector_shape=closed.detector_shape,
        detector_columns=closed.columns,
        detector_rows=closed.rows,
        gain_e_per_adu=gain_e_per_adu,
        saturation_adu=saturation_adu,
        detector_signature=detector_signature,
        metadata_warnings=warnings,
        closed_shutter_path=closed_path,
        open_reference_path=reference_path,
        _calibrator=calibrator,
        _fvb_dark_sigma_adu=fvb_dark_sigma,
        _image_background_adu=image_background,
        _image_dark_master_adu=image_dark_master,
        _image_dark_variance_adu2=image_dark_variance,
        _image_support_mask=image_support,
        calibration_options=applied_options,
    )
