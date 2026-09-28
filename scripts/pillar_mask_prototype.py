"""Standalone proof-of-concept for suppressing microfluidic trap pillars.

This script is intentionally separate from the napari/Cellpose runtime.  It
uses one representative trap pair as a weighted template, registers that
template to the repeated staggered lattice in each field of view, builds a
geometric mask for the two pillars at every lattice site, and replaces masked
pixels with a feathered local-background estimate.

The defaults are calibrated for the supplied run_14 images (1024 x 1344).
They are exposed as command-line options so the prototype can be tested and
adjusted without changing application code.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
from skimage.feature import peak_local_max


@dataclass(frozen=True)
class Calibration:
    """Geometry and reference information for one device/magnification."""

    reference_position: int = 0
    reference_anchor_y: float = 140.0
    reference_anchor_x: float = 790.0
    lattice_v1_y: float = 11.195
    lattice_v1_x: float = 286.913
    lattice_v2_y: float = 203.945
    lattice_v2_x: float = 135.644
    pillar_center_offset_x: float = 48.0
    pillar_center_offset_y: float = 0.0
    pillar_length: float = 96.0
    pillar_width: float = 66.0
    template_length: float = 78.0
    template_width: float = 44.0
    template_height: int = 121
    template_width_px: int = 181
    background_sigma: float = 32.0
    background_exclusion: int = 10
    feather_sigma: float = 1.5
    site_score_threshold: float = 0.25

    @property
    def reference_anchor(self) -> tuple[float, float]:
        return self.reference_anchor_y, self.reference_anchor_x

    @property
    def lattice_v1(self) -> np.ndarray:
        return np.asarray((self.lattice_v1_y, self.lattice_v1_x), dtype=float)

    @property
    def lattice_v2(self) -> np.ndarray:
        return np.asarray((self.lattice_v2_y, self.lattice_v2_x), dtype=float)


@dataclass(frozen=True)
class Registration:
    position: int
    anchor_y: float
    anchor_x: float
    n_full_anchors: int
    mean_score: float
    median_score: float
    score_p10: float
    peak_score: float


def _parse_int_spec(value: str) -> list[int]:
    """Parse comma-separated integers and inclusive ranges such as 0-4,7."""

    result: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start, end = int(start_text), int(end_text)
            step = 1 if end >= start else -1
            result.extend(range(start, end + step, step))
        else:
            result.append(int(chunk))
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return sorted(set(result))


def _rotated_rectangle(
    shape: tuple[int, int],
    center_y: float,
    center_x: float,
    length: float,
    width: float,
    angle_degrees: float,
) -> np.ndarray:
    """Return a boolean mask for a rotated rectangle."""

    yy, xx = np.indices(shape, dtype=np.float32)
    theta = np.deg2rad(angle_degrees)
    dx = xx - float(center_x)
    dy = yy - float(center_y)
    along = dx * np.cos(theta) + dy * np.sin(theta)
    across = -dx * np.sin(theta) + dy * np.cos(theta)
    return (np.abs(along) <= length / 2.0) & (np.abs(across) <= width / 2.0)


def _pair_mask(
    shape: tuple[int, int],
    anchor_y: float,
    anchor_x: float,
    *,
    length: float,
    width: float,
    offset_y: float,
    offset_x: float,
) -> np.ndarray:
    """Rasterize the two oppositely tilted pillars for one trap."""

    left = _rotated_rectangle(
        shape,
        anchor_y + offset_y,
        anchor_x - offset_x,
        length,
        width,
        -45.0,
    )
    right = _rotated_rectangle(
        shape,
        anchor_y + offset_y,
        anchor_x + offset_x,
        length,
        width,
        45.0,
    )
    return left | right


def _load_stack(
    input_dir: Path,
    position: int,
    z_planes: Sequence[int],
) -> tuple[list[Path], np.ndarray]:
    paths = [
        input_dir / f"t000_p{position:03d}_c000_z{z:03d}.tiff"
        for z in z_planes
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing input TIFF(s): " + ", ".join(missing))
    planes = [np.asarray(tifffile.imread(path)) for path in paths]
    shape = planes[0].shape
    if any(plane.ndim != 2 or plane.shape != shape for plane in planes):
        raise ValueError(f"position {position} does not contain a uniform 2-D stack")
    return paths, np.stack(planes, axis=0)


def _detection_features(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build signed-contrast and edge projections for registration only."""

    signed_planes: list[np.ndarray] = []
    edge_planes: list[np.ndarray] = []
    for plane in stack:
        image = np.asarray(plane, dtype=np.float32)
        smooth = ndi.gaussian_filter(image, 1.0)
        signed = smooth - ndi.gaussian_filter(image, 30.0)
        scale = float(np.percentile(np.abs(signed), 95.0))
        signed = np.clip(signed / max(scale, 1.0), -6.0, 6.0)
        edge = np.hypot(ndi.sobel(signed, axis=0), ndi.sobel(signed, axis=1))
        edge = ndi.gaussian_filter(edge, 1.0)
        edge = np.minimum(edge, np.percentile(edge, 99.7))
        signed_planes.append(signed)
        edge_planes.append(edge)
    return np.median(signed_planes, axis=0), np.max(edge_planes, axis=0)


def _centered_patch(
    image: np.ndarray,
    center_yx: tuple[float, float],
    patch_shape: tuple[int, int],
) -> np.ndarray:
    height, width = patch_shape
    if height % 2 != 1 or width % 2 != 1:
        raise ValueError("template dimensions must be odd")
    cy, cx = (int(round(value)) for value in center_yx)
    half_y, half_x = height // 2, width // 2
    if not (
        half_y <= cy < image.shape[0] - half_y
        and half_x <= cx < image.shape[1] - half_x
    ):
        raise ValueError("reference template extends beyond the image")
    return image[cy - half_y : cy + half_y + 1, cx - half_x : cx + half_x + 1]


def _weighted_ncc(
    image: np.ndarray,
    template: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    """Weighted, zero-mean normalized cross-correlation."""

    image = np.asarray(image, dtype=np.float32)
    template = np.asarray(template, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    weight_sum = float(weight.sum())
    template_mean = float((template * weight).sum() / weight_sum)
    zero_mean_template = (template - template_mean) * weight
    template_energy = float(np.sum(zero_mean_template**2))
    kernel = weight[::-1, ::-1]
    numerator = fftconvolve(
        image,
        zero_mean_template[::-1, ::-1],
        mode="same",
    )
    local_sum = fftconvolve(image, kernel, mode="same")
    local_sum_sq = fftconvolve(image**2, kernel, mode="same")
    local_weight = fftconvolve(np.ones_like(image), kernel, mode="same")
    local_variance = local_sum_sq - local_sum**2 / np.maximum(local_weight, 1.0)
    denominator = np.sqrt(np.maximum(local_variance * template_energy, 1e-8))
    return np.clip(numerator / denominator, -1.0, 1.0)


def _build_template(
    signed: np.ndarray,
    edge: np.ndarray,
    calibration: Calibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    template_shape = calibration.template_height, calibration.template_width_px
    patch_signed = _centered_patch(
        signed,
        calibration.reference_anchor,
        template_shape,
    )
    patch_edge = _centered_patch(
        edge,
        calibration.reference_anchor,
        template_shape,
    )
    cy, cx = np.asarray(template_shape, dtype=float) / 2.0
    weight = _pair_mask(
        template_shape,
        cy,
        cx,
        length=calibration.template_length,
        width=calibration.template_width,
        offset_y=calibration.pillar_center_offset_y,
        offset_x=calibration.pillar_center_offset_x,
    )
    return patch_signed, patch_edge, weight


def _registration_score(
    signed: np.ndarray,
    edge: np.ndarray,
    template_signed: np.ndarray,
    template_edge: np.ndarray,
    template_weight: np.ndarray,
) -> np.ndarray:
    signed_score = _weighted_ncc(signed, template_signed, template_weight)
    edge_score = _weighted_ncc(edge, template_edge, template_weight)
    return 0.55 * signed_score + 0.45 * edge_score


def _grid_points(
    origin_yx: Sequence[float],
    image_shape: tuple[int, int],
    calibration: Calibration,
    *,
    margin_y: float,
    margin_x: float,
) -> np.ndarray:
    """Generate unique lattice anchors inside a requested image margin."""

    origin = np.asarray(origin_yx, dtype=float)
    points: set[tuple[int, int]] = set()
    for i in range(-12, 13):
        for j in range(-12, 13):
            point = origin + i * calibration.lattice_v1 + j * calibration.lattice_v2
            y, x = (int(round(value)) for value in point)
            if (
                -margin_y <= y < image_shape[0] + margin_y
                and -margin_x <= x < image_shape[1] + margin_x
            ):
                points.add((y, x))
    return np.asarray(sorted(points), dtype=int)


def _full_anchor_scores(
    score: np.ndarray,
    origin_yx: Sequence[float],
    calibration: Calibration,
    *,
    sample_radius: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    margin_y = calibration.template_height // 2 + 2
    margin_x = calibration.template_width_px // 2 + 2
    anchors = _grid_points(
        origin_yx,
        score.shape,
        calibration,
        margin_y=-float(margin_y),
        margin_x=-float(margin_x),
    )
    values: list[float] = []
    for y, x in anchors:
        y0, y1 = y - sample_radius, y + sample_radius + 1
        x0, x1 = x - sample_radius, x + sample_radius + 1
        values.append(float(np.max(score[y0:y1, x0:x1])))
    return anchors, np.asarray(values, dtype=float)


def _register_lattice(
    score: np.ndarray,
    position: int,
    calibration: Calibration,
) -> Registration:
    margin_y = calibration.template_height // 2 + 2
    margin_x = calibration.template_width_px // 2 + 2
    candidate_peaks = peak_local_max(
        score,
        min_distance=35,
        num_peaks=100,
        exclude_border=(margin_y, margin_x),
    )
    if candidate_peaks.size == 0:
        raise RuntimeError(f"position {position}: no registration peaks found")

    ranked: list[tuple[float, float, np.ndarray]] = []
    for candidate in candidate_peaks:
        anchors, values = _full_anchor_scores(score, candidate, calibration)
        if len(values) < 8:
            continue
        # Reward a consistently good lattice rather than one exceptional object.
        objective = 0.65 * float(values.mean()) + 0.35 * float(np.median(values))
        ranked.append((objective, float(score[tuple(candidate)]), candidate.astype(float)))
    if not ranked:
        raise RuntimeError(f"position {position}: no valid lattice candidates found")
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_origin = ranked[0][2]

    # Integer refinement around the best candidate.  All pillars retain one
    # global phase so an attached cell cannot pull an individual mask off-grid.
    refined: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            origin = best_origin + (dy, dx)
            anchors, values = _full_anchor_scores(score, origin, calibration)
            objective = 0.65 * float(values.mean()) + 0.35 * float(np.median(values))
            refined.append((objective, origin, anchors, values))
    refined.sort(key=lambda item: item[0], reverse=True)
    _, origin, anchors, values = refined[0]
    return Registration(
        position=position,
        anchor_y=float(origin[0]),
        anchor_x=float(origin[1]),
        n_full_anchors=len(anchors),
        mean_score=float(values.mean()),
        median_score=float(np.median(values)),
        score_p10=float(np.percentile(values, 10.0)),
        peak_score=float(np.max(values)),
    )


def _render_full_mask(
    image_shape: tuple[int, int],
    registration: Registration,
    calibration: Calibration,
    score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, float, bool]]]:
    anchors = _grid_points(
        (registration.anchor_y, registration.anchor_x),
        image_shape,
        calibration,
        margin_y=70.0,
        margin_x=100.0,
    )
    mask = np.zeros(image_shape, dtype=bool)
    accepted: list[tuple[int, int]] = []
    decisions: list[tuple[int, int, float, bool]] = []
    sampled: list[tuple[int, int, float, bool]] = []
    for y, x in anchors:
        clipped_y = int(np.clip(y, 0, image_shape[0] - 1))
        clipped_x = int(np.clip(x, 0, image_shape[1] - 1))
        y0, y1 = max(0, clipped_y - 2), min(image_shape[0], clipped_y + 3)
        x0, x1 = max(0, clipped_x - 2), min(image_shape[1], clipped_x + 3)
        site_score = float(np.max(score[y0:y1, x0:x1]))
        sampled.append((int(y), int(x), site_score, False))

    # The wall-crossing fields contain a mathematically valid lattice on their
    # empty left side.  Infer that boundary only when low/high response sites
    # are cleanly separated in x, then use it to control partial-edge masks.
    margin_y = calibration.template_height // 2 + 2
    margin_x = calibration.template_width_px // 2 + 2
    full_sites = [
        row
        for row in sampled
        if margin_y <= row[0] < image_shape[0] - margin_y
        and margin_x <= row[1] < image_shape[1] - margin_x
    ]
    present_x = [
        x for _, x, value, _ in full_sites
        if value >= calibration.site_score_threshold
    ]
    absent_x = [
        x for _, x, value, _ in full_sites
        if value < calibration.site_score_threshold
    ]
    wall_x: float | None = None
    if len(present_x) >= 6 and len(absent_x) >= 3:
        rightmost_absent = float(max(absent_x))
        leftmost_present = float(min(present_x))
        if leftmost_present - rightmost_absent >= 60.0:
            wall_x = 0.5 * (rightmost_absent + leftmost_present)

    for y, x, site_score, _ in sampled:
        pair = _pair_mask(
            image_shape,
            float(y),
            float(x),
            length=calibration.pillar_length,
            width=calibration.pillar_width,
            offset_y=calibration.pillar_center_offset_y,
            offset_x=calibration.pillar_center_offset_x,
        )
        center_outside = not (
            0 <= y < image_shape[0]
            and 0 <= x < image_shape[1]
        )
        partial_edge_site = (
            center_outside
            and bool(np.any(pair))
            and (wall_x is None or x > wall_x)
        )
        is_present = (
            (
                site_score >= calibration.site_score_threshold
                and (wall_x is None or x > wall_x)
            )
            or partial_edge_site
        )
        decisions.append((y, x, site_score, is_present))
        if not is_present:
            continue
        accepted.append((y, x))
        mask |= pair
    return mask, np.asarray(accepted, dtype=int).reshape((-1, 2)), decisions


def _background_fill(
    image: np.ndarray,
    mask: np.ndarray,
    calibration: Calibration,
    *,
    texture_seed: int,
) -> np.ndarray:
    """Replace pillars with matched illumination and camera texture."""

    work = np.asarray(image, dtype=np.float32)
    fill_region = ndi.binary_dilation(mask, iterations=3)
    exclusion = ndi.binary_dilation(
        fill_region,
        iterations=calibration.background_exclusion,
    )
    valid_boolean = ~exclusion
    valid = valid_boolean.astype(np.float32)
    numerator = ndi.gaussian_filter(work * valid, calibration.background_sigma)
    denominator = ndi.gaussian_filter(valid, calibration.background_sigma)
    background = numerator / np.maximum(denominator, 1e-4)

    # Restore realistic high-frequency texture instead of leaving unnaturally
    # flat diamonds.  Resampling safe camera residuals preserves the noise
    # distribution without copying cell or pillar-shaped structures.
    residual = work - ndi.gaussian_filter(work, 1.2)
    safe_residual = residual[valid_boolean]
    residual_center = float(np.median(safe_residual))
    residual_sigma = 1.4826 * float(
        np.median(np.abs(safe_residual - residual_center))
    )
    texture_limit = max(3.5 * residual_sigma, 1.0)
    texture_pool = safe_residual - residual_center
    texture_pool = texture_pool[np.abs(texture_pool) <= texture_limit]
    rng = np.random.default_rng(texture_seed)
    sampled_texture = rng.choice(texture_pool, size=work.shape, replace=True)
    replacement = background + sampled_texture

    # Put the transition beyond the geometric mask so the feathered band lies
    # in background rather than through the pillar's dark rim.
    alpha = ndi.gaussian_filter(fill_region.astype(np.float32), calibration.feather_sigma)
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha[mask] = 1.0
    cleaned = work * (1.0 - alpha) + replacement * alpha
    if np.issubdtype(image.dtype, np.integer):
        limits = np.iinfo(image.dtype)
        cleaned = np.clip(np.rint(cleaned), limits.min, limits.max)
    return cleaned.astype(image.dtype, copy=False)


def _normalize_u8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, (0.2, 99.8))
    normalized = np.clip((image - lo) / max(float(hi - lo), 1.0), 0.0, 1.0)
    return np.asarray(np.rint(normalized * 255.0), dtype=np.uint8)


def _overlay_rgb(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = _normalize_u8(image)
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    alpha = 0.38
    rgb[mask] = rgb[mask] * (1.0 - alpha) + np.asarray((255, 45, 20)) * alpha
    outline = mask ^ ndi.binary_erosion(mask)
    rgb[outline] = (255, 230, 0)
    return np.asarray(np.clip(rgb, 0, 255), dtype=np.uint8)


def _save_position_overlay(
    output_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    anchors: np.ndarray,
    registration: Registration,
) -> None:
    overlay = Image.fromarray(_overlay_rgb(image, mask))
    draw = ImageDraw.Draw(overlay)
    for y, x in anchors:
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(0, 255, 255), width=1)
    draw.rectangle((0, 0, 525, 29), fill=(0, 0, 0))
    draw.text(
        (7, 7),
        (
            f"p{registration.position:03d}  mean={registration.mean_score:.3f}  "
            f"median={registration.median_score:.3f}  n={registration.n_full_anchors}"
        ),
        fill=(255, 255, 255),
    )
    overlay.save(output_path)


def _make_overview(
    output_path: Path,
    rows: Sequence[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    *,
    title: str,
) -> None:
    thumb_w, thumb_h = 224, 171
    label_h = 24
    n_columns = len(rows)
    sheet = Image.new("RGB", (n_columns * thumb_w, 3 * (thumb_h + label_h)), "black")
    draw = ImageDraw.Draw(sheet)
    for column, (position, original, mask, cleaned) in enumerate(rows):
        panels = (
            ("original", np.repeat(_normalize_u8(original)[..., None], 3, axis=2)),
            ("pillar mask", _overlay_rgb(original, mask)),
            ("cleaned", np.repeat(_normalize_u8(cleaned)[..., None], 3, axis=2)),
        )
        for row_index, (label, panel) in enumerate(panels):
            x0 = column * thumb_w
            y0 = row_index * (thumb_h + label_h)
            thumbnail = Image.fromarray(panel).resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
            sheet.paste(thumbnail, (x0, y0 + label_h))
            draw.text((x0 + 5, y0 + 5), f"p{position:03d} {label}", fill="white")
    sheet.save(output_path, pnginfo=None)


def _write_registration_csv(path: Path, registrations: Iterable[Registration]) -> None:
    fieldnames = list(Registration.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for registration in registrations:
            writer.writerow(asdict(registration))


def _write_site_csv(
    path: Path,
    site_rows: Iterable[tuple[int, int, int, float, bool]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("position", "anchor_y", "anchor_x", "score", "masked"))
        writer.writerows(site_rows)


def run(
    input_dir: Path,
    output_dir: Path,
    positions: Sequence[int],
    z_planes: Sequence[int],
    calibration: Calibration,
) -> list[Registration]:
    cleaned_dir = output_dir / "cleaned"
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    for directory in (output_dir, cleaned_dir, mask_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _, reference_stack = _load_stack(
        input_dir,
        calibration.reference_position,
        z_planes,
    )
    reference_signed, reference_edge = _detection_features(reference_stack)
    template_signed, template_edge, template_weight = _build_template(
        reference_signed,
        reference_edge,
        calibration,
    )

    registrations: list[Registration] = []
    site_rows: list[tuple[int, int, int, float, bool]] = []
    overview_rows: dict[int, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = {
        z: [] for z in z_planes
    }
    for position in positions:
        paths, stack = _load_stack(input_dir, position, z_planes)
        signed, edge = _detection_features(stack)
        score = _registration_score(
            signed,
            edge,
            template_signed,
            template_edge,
            template_weight,
        )
        registration = _register_lattice(score, position, calibration)
        mask, anchors, site_decisions = _render_full_mask(
            stack.shape[1:],
            registration,
            calibration,
            score,
        )
        site_rows.extend(
            (position, y, x, site_score, accepted)
            for y, x, site_score, accepted in site_decisions
        )
        registrations.append(registration)

        tifffile.imwrite(
            mask_dir / f"t000_p{position:03d}_pillar_mask.tiff",
            mask.astype(np.uint8),
            photometric="minisblack",
            metadata={"axes": "YX", "meaning": "1=pillar exclusion mask"},
        )
        representative_index = min(range(len(z_planes)), key=lambda i: abs(z_planes[i] - 1))
        _save_position_overlay(
            overlay_dir / f"p{position:03d}_registration.png",
            stack[representative_index],
            mask,
            anchors,
            registration,
        )

        for stack_index, (z, source_path) in enumerate(zip(z_planes, paths, strict=True)):
            original = stack[stack_index]
            cleaned = _background_fill(
                original,
                mask,
                calibration,
                texture_seed=10_000 * position + z,
            )
            destination = cleaned_dir / source_path.name
            tifffile.imwrite(
                destination,
                cleaned,
                photometric="minisblack",
                metadata={
                    "axes": "YX",
                    "processing": "registered paired-pillar mask + local background fill",
                },
            )
            Image.fromarray(_overlay_rgb(original, mask)).save(
                overlay_dir / f"{source_path.stem}_overlay.png"
            )
            overview_rows[z].append((position, original, mask, cleaned))

        print(
            f"p{position:03d}: anchor=({registration.anchor_y:.1f}, "
            f"{registration.anchor_x:.1f}) mean={registration.mean_score:.3f} "
            f"median={registration.median_score:.3f} p10={registration.score_p10:.3f} "
            f"n_full={registration.n_full_anchors} n_masked={len(anchors)} "
            f"mask_fraction={mask.mean():.3f}"
        )

    for z, rows in overview_rows.items():
        _make_overview(
            output_dir / f"overview_z{z:03d}.png",
            rows,
            title=f"z{z:03d}",
        )
    _write_registration_csv(output_dir / "registration.csv", registrations)
    _write_site_csv(output_dir / "site_scores.csv", site_rows)
    (output_dir / "calibration.json").write_text(
        json.dumps(asdict(calibration), indent=2) + "\n",
        encoding="utf-8",
    )
    return registrations


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--positions", type=_parse_int_spec, default=_parse_int_spec("0-9"))
    parser.add_argument("--z-planes", type=_parse_int_spec, default=_parse_int_spec("0-2"))
    parser.add_argument("--reference-position", type=int, default=0)
    parser.add_argument("--reference-anchor-y", type=float, default=140.0)
    parser.add_argument("--reference-anchor-x", type=float, default=790.0)
    parser.add_argument("--pillar-length", type=float, default=96.0)
    parser.add_argument("--pillar-width", type=float, default=66.0)
    parser.add_argument("--pillar-offset-x", type=float, default=48.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    calibration = Calibration(
        reference_position=args.reference_position,
        reference_anchor_y=args.reference_anchor_y,
        reference_anchor_x=args.reference_anchor_x,
        pillar_length=args.pillar_length,
        pillar_width=args.pillar_width,
        pillar_center_offset_x=args.pillar_offset_x,
    )
    run(
        args.input_dir,
        args.output_dir,
        args.positions,
        args.z_planes,
        calibration,
    )


if __name__ == "__main__":
    main()
