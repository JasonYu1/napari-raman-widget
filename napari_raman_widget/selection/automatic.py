"""Automatic segmentation and Raman target selection."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm.auto import tqdm

from napari_raman_widget.calibration.models import (
    apply_vandermonde_model,
    load_vandermonde_model,
)
from napari_raman_widget.engine_compat import make_pillar_shape_filter
from napari_raman_widget.trench_tracking import (
    filter_labels_to_trenches,
    registered_trench_geometry,
)

from .masks import get_n_most_centered_coms

__all__ = ["automated_point_selections"]


def get_seq_from_napari(*args: Any, **kwargs: Any) -> Any:
    """Load the engine helper only when an automatic selection is run."""
    from raman_mda_engine.utils import get_seq_from_napari as implementation

    return implementation(*args, **kwargs)


def segment_single_img(*args: Any, **kwargs: Any) -> Any:
    """Load Cellpose and its PyTorch runtime only when segmentation is run."""
    from raman_mda_engine.aiming.autotracking import (
        segment_single_img as implementation,
    )

    return implementation(*args, **kwargs)


_NO_AUTOFOCUS = {
    None,
    "",
    "none",
}


def _is_no_autofocus(
    autofocus_object: str | None,
) -> bool:
    """Return whether autofocus point creation is disabled."""
    if isinstance(autofocus_object, str):
        autofocus_object = (
            autofocus_object.strip().lower()
        )

    return autofocus_object in _NO_AUTOFOCUS


def _add_point(
    source: Any,
    position_index: int,
    point_yx: np.ndarray,
) -> None:
    """Add one point to a six-dimensional point source."""
    source._points.add(
        [
            0,
            position_index,
            0,
            0,
            float(point_yx[0]),
            float(point_yx[1]),
        ]
    )


def _retain_same_label_refinements(
    original_yx: np.ndarray,
    refined_yx: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Keep a refinement only when it remains in the same safe cell label."""

    original = np.atleast_2d(np.asarray(original_yx, dtype=float))
    refined = np.atleast_2d(np.asarray(refined_yx, dtype=float))
    if refined.shape != original.shape:
        raise ValueError("cell_point_refiner must preserve the cell point shape.")
    mask = np.asarray(labels)
    accepted = original.copy()
    height, width = mask.shape
    for index, (before, after) in enumerate(zip(original, refined, strict=True)):
        before_pixel = np.rint(before).astype(int)
        after_pixel = np.rint(after).astype(int)
        before_inside = (
            0 <= before_pixel[0] < height and 0 <= before_pixel[1] < width
        )
        after_inside = (
            0 <= after_pixel[0] < height and 0 <= after_pixel[1] < width
        )
        if not before_inside or not after_inside:
            continue
        before_label = int(mask[tuple(before_pixel)])
        after_label = int(mask[tuple(after_pixel)])
        if before_label != 0 and after_label == before_label:
            accepted[index] = after
    return accepted


def _circle_selection_mask(
    shape: tuple[int, int],
    center: tuple[float, float] | None,
    radius: float,
) -> np.ndarray:
    """Return the circular automated-selection region."""

    if len(shape) != 2 or any(int(value) <= 0 for value in shape):
        raise ValueError("selection image shape must be two-dimensional")
    radius = float(radius)
    if radius < 0:
        raise ValueError("radius cannot be negative")
    center_yx = (
        np.asarray(shape, dtype=float) / 2.0
        if center is None
        else np.asarray(center, dtype=float)
    )
    if center_yx.shape != (2,) or not np.all(np.isfinite(center_yx)):
        raise ValueError("center must contain one finite (y, x) coordinate")
    if np.isinf(radius):
        return np.ones(shape, dtype=bool)
    rows, columns = np.ogrid[: shape[0], : shape[1]]
    return (
        (rows - center_yx[0]) ** 2 + (columns - center_yx[1]) ** 2
        <= radius**2
    )


def _mask_union(
    masks: tuple[np.ndarray, ...],
    shape: tuple[int, int],
) -> np.ndarray:
    """Return a validated union for registered trench masks."""

    if not masks:
        return np.zeros(shape, dtype=bool)
    prepared = tuple(np.asarray(mask, dtype=bool) for mask in masks)
    if any(mask.shape != shape for mask in prepared):
        raise ValueError("registered trench masks must match the image")
    return np.logical_or.reduce(prepared)


def _segment_trench_cellpose_input(
    image: np.ndarray,
    allowed_mask: np.ndarray,
    *,
    upsample: int,
    cellpose_model: str,
) -> np.ndarray:
    """Segment the tight crop of cap-safe trenches inside the selection circle."""

    frame = np.asarray(image)
    allowed = np.asarray(allowed_mask, dtype=bool)
    if frame.ndim != 2 or allowed.shape != frame.shape:
        raise ValueError("trench mask must match the two-dimensional image")
    if not np.any(allowed):
        return np.zeros(frame.shape, dtype=np.uint32)

    pixels = np.argwhere(allowed)
    lower = np.maximum(pixels.min(axis=0) - 4, 0)
    upper = np.minimum(pixels.max(axis=0) + 5, frame.shape)
    y0, x0 = (int(value) for value in lower)
    y1, x1 = (int(value) for value in upper)
    crop = np.array(frame[y0:y1, x0:x1], copy=True)
    allowed_crop = allowed[y0:y1, x0:x1]
    crop[~allowed_crop] = np.median(frame[allowed])

    factor = max(1, int(upsample))
    enlarged = np.repeat(np.repeat(crop, factor, axis=0), factor, axis=1)
    enlarged_mask = np.asarray(
        segment_single_img(
            enlarged,
            scale=1,
            crop=False,
            cellpose_model=cellpose_model,
        )
    )
    if enlarged_mask.ndim != 2:
        raise ValueError("Cellpose returned a non-image trench mask")
    restored_crop = enlarged_mask[::factor, ::factor]
    restored_crop = restored_crop[: crop.shape[0], : crop.shape[1]]
    restored_crop = np.where(allowed_crop, restored_crop, 0)
    restored = np.zeros(frame.shape, dtype=restored_crop.dtype)
    restored[y0:y1, x0:x1] = restored_crop
    return restored


def _background_point_in_region(
    label_mask: np.ndarray,
    allowed_mask: np.ndarray,
    *,
    threshold: float,
    center: tuple[float, float] | None,
) -> np.ndarray:
    """Choose clear background inside the circle but outside every trench."""

    labels = np.asarray(label_mask)
    allowed = np.asarray(allowed_mask, dtype=bool)
    if labels.ndim != 2 or allowed.shape != labels.shape:
        raise ValueError("background region must match the label image")
    if threshold < 0:
        raise ValueError("background threshold cannot be negative")

    background = labels == 0
    if np.any(~background):
        clearance = distance_transform_edt(background)
    else:
        clearance = np.full(labels.shape, np.inf, dtype=float)
    fallback_candidates = np.argwhere(allowed & background)
    if not len(fallback_candidates):
        raise ValueError(
            "No background pixel exists inside the selection circle and "
            "outside the registered trenches."
        )
    clear_candidates = np.argwhere(
        allowed & background & (clearance >= float(threshold))
    )
    candidates = (
        clear_candidates if len(clear_candidates) else fallback_candidates
    )
    reference = (
        np.asarray(labels.shape, dtype=float) / 2.0
        if center is None
        else np.asarray(center, dtype=float)
    )
    index = int(np.argmin(np.linalg.norm(candidates - reference, axis=1)))
    return candidates[index].astype(float)


def _fill_interlabel_trench_gaps(
    label_mask: np.ndarray,
    trench_masks: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Split axial background gaps between neighboring labels in each trench.

    Cellpose sometimes leaves a background strip between two packed cells.
    Within each registered trench, the strip between the facing axial edges
    of neighboring labels is assigned to whichever label is spatially nearer.
    Pixels before the first label and after the last label remain background.
    """

    result = np.asarray(label_mask).copy()
    if result.ndim != 2 or not np.issubdtype(result.dtype, np.integer):
        raise ValueError("label_mask must be a two-dimensional integer image")

    for trench_mask in trench_masks:
        safe = np.asarray(trench_mask, dtype=bool)
        if safe.shape != result.shape:
            raise ValueError("trench masks must match the label image")
        trench_pixels = np.argwhere(safe)
        if len(trench_pixels) < 3:
            continue

        lower = trench_pixels.min(axis=0)
        upper = trench_pixels.max(axis=0) + 1
        y0, x0 = (int(value) for value in lower)
        y1, x1 = (int(value) for value in upper)
        local_safe = safe[y0:y1, x0:x1]
        local_labels = result[y0:y1, x0:x1]
        coordinates = np.argwhere(local_safe)
        centered = coordinates - coordinates.mean(axis=0)
        covariance = centered.T @ centered
        _, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, -1]

        measurements = []
        values = np.unique(local_labels[local_safe])
        for value in values[values != 0]:
            pixels = np.argwhere(local_safe & (local_labels == value))
            if not len(pixels):
                continue
            projections = pixels @ axis
            measurements.append(
                (
                    float(projections.mean()),
                    float(projections.min()),
                    float(projections.max()),
                    int(value),
                )
            )
        measurements.sort(key=lambda item: item[0])

        for left, right in zip(measurements, measurements[1:]):
            gap_start = left[2]
            gap_end = right[1]
            if gap_end <= gap_start:
                continue
            empty_pixels = np.argwhere(local_safe & (local_labels == 0))
            if not len(empty_pixels):
                break
            projections = empty_pixels @ axis
            between = (projections > gap_start) & (projections < gap_end)
            gap_pixels = empty_pixels[between]
            if not len(gap_pixels):
                continue

            left_distance = distance_transform_edt(local_labels != left[3])
            right_distance = distance_transform_edt(local_labels != right[3])
            gap_index = tuple(gap_pixels.T)
            use_left = left_distance[gap_index] <= right_distance[gap_index]
            chosen = np.where(use_left, left[3], right[3])
            local_labels[gap_index] = chosen.astype(result.dtype, copy=False)

    return result


def _selection_exclusion_mask(
    trench_safe_mask: np.ndarray | None,
    pillar_removed_mask: np.ndarray | None,
) -> np.ndarray | None:
    """Combine trench and pillar exclusions for cell targets."""

    exclusion = None
    if trench_safe_mask is not None:
        exclusion = ~np.asarray(trench_safe_mask, dtype=bool)
    if pillar_removed_mask is not None:
        pillars = np.asarray(pillar_removed_mask, dtype=bool)
        exclusion = pillars.copy() if exclusion is None else exclusion | pillars
    return exclusion


def automated_point_selections(
    core: Any,
    viewer: Any,
    main_window: Any,
    point_transformer: Any,
    N: int,
    center: tuple[float, float] | None = None,
    radius: float = 250,
    autofocus_object: str | None = "glass",
    bkd_thres: float = 50,
    batch: bool = True,
    center_cell: bool = False,
    vandermonde_model_path: str | Path | None = None,
    cellpose_model: str = "cyto2",
    stage_settle_time: float = 5,
    image_settle_time: float = 1,
    direct_pixel_to_stage: bool = False,
    block_mda: bool = False,
    show_masks: bool = False,
    cellpose_upsample: int = 1,
    cell_point_refiner: Callable[[np.ndarray], np.ndarray] | None = None,
    suppress_pillars: bool = False,
    constrain_trenches: bool = False,
):
    """Segment images and add automatically selected Raman targets.

    Parameters
    ----------
    core
        Microscope core used to move the stage and acquire images.
    viewer
        Napari viewer receiving point layers.
    main_window
        MDA widget from which the current sequence is obtained.
    point_transformer
        Transformer used by the Raman MDA point sources.
    N
        Maximum number of selected points for each original field of
        view.
    center
        Selection center in ``(y, x)`` order. The image center is used
        when omitted.
    radius
        Maximum cell distance from the selection center.
    autofocus_object
        Autofocus target. Use ``None`` or ``"None"`` to disable the
        autofocus point layer.
    bkd_thres
        Minimum background clearance for an autofocus point.
    batch
        When true, create one repeated stage position per selected cell.
    center_cell
        When true, calculate a corrected stage position that places each
        selected cell at the center of its new field of view.
    vandermonde_model_path
        Pixel-offset-to-stage-offset model. Required when
        ``center_cell`` is true.
    cellpose_model
        Segmentation model passed to the Raman MDA engine.
    stage_settle_time
        Delay after moving to a stage position.
    image_settle_time
        Delay after acquiring an image.
    cell_point_refiner
        Optional demo-only correction applied to Cellpose cell centers. The
        autofocus/background point, when present, is left unchanged.
    suppress_pillars
        When true, remove Cellpose labels whose rotated shape fits at least 75%
        of the measured 78 x 56 pixel pillar rectangle.
    constrain_trenches
        When true, fit one global rotation and translation of the learned
        trapezoid lattice, segment only cap-aware trench pixels inside the
        circular selection region, and place the optional background target
        inside that circle but outside every registered trench.

    Returns
    -------
    sources
        Cell point source followed by an optional autofocus source.
    autofocus_positions
        Position indices used by the autofocus workflow.
    sequence
        Updated MDA sequence.
    """
    if N < 1:
        raise ValueError(
            "N must be at least one."
        )

    no_autofocus = _is_no_autofocus(
        autofocus_object
    )

    coefficients = None
    degree = None

    if center_cell and not direct_pixel_to_stage:
        if vandermonde_model_path is None:
            raise ValueError(
                "vandermonde_model_path is required "
                "when center_cell is enabled."
            )

        coefficients, degree = (
            load_vandermonde_model(
                vandermonde_model_path
            )
        )

    sequence = get_seq_from_napari(
        main_window
    )

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    selected_by_position: list[np.ndarray] = []
    corrected_positions = []
    pillar_shape_filter = make_pillar_shape_filter(suppress_pillars)

    for position_index in tqdm(
        range(len(sequence.stage_positions)),
        desc="Selecting Raman targets",
    ):
        original_position = (
            sequence.stage_positions[
                position_index
            ]
        )

        core.setXYPosition(
            original_position.x,
            original_position.y,
        )
        core.waitForSystem()

        if stage_settle_time > 0:
            time.sleep(stage_settle_time)

        core.snapImage()
        core.waitForSystem()
        image = np.asarray(core.getImage())
        core.waitForSystem()

        if image_settle_time > 0:
            time.sleep(image_settle_time)

        images.append(image)
        trench_geometry = None
        trench_safe_mask = None
        selection_circle_mask = None
        raw_trench_union = None
        if constrain_trenches:
            trench_geometry = registered_trench_geometry(image)
            if not trench_geometry.geometry_valid:
                mask = np.zeros(image.shape, dtype=np.uint32)
                masks.append(mask)
                if show_masks:
                    viewer.add_labels(
                        mask,
                        name=f"Cellpose mask p{position_index}",
                    )
                print(
                    "Learned trench geometry was not recognized at "
                    f"position {position_index} "
                    f"(score={trench_geometry.geometry_score:.3f}); "
                    "skipping Cellpose and this field without cell or "
                    "autofocus targets."
                )
                selected_by_position.append(np.empty((0, 2), dtype=float))
                continue
            selection_circle_mask = _circle_selection_mask(
                image.shape,
                center,
                radius,
            )
            raw_trench_union = _mask_union(
                trench_geometry.trench_masks,
                image.shape,
            )
            trench_safe_mask = (
                np.asarray(trench_geometry.safe_union, dtype=bool)
                & selection_circle_mask
            )
            segmentation_image = image
        else:
            segmentation_image = image

        if constrain_trenches:
            # The circle chooses the ROI; learned geometry removes walls,
            # gaps, and the closed-tip cap within that ROI.
            mask = _segment_trench_cellpose_input(
                segmentation_image,
                trench_safe_mask,
                upsample=cellpose_upsample,
                cellpose_model=cellpose_model,
            )
        elif int(cellpose_upsample) > 1:
            from napari_raman_widget.demo.cellpose import (
                segment_upsampled_demo_region,
            )

            mask = segment_upsampled_demo_region(
                segmentation_image,
                center_yx=(
                    center
                    if center is not None
                    else tuple(
                        np.asarray(segmentation_image.shape, dtype=float) / 2
                    )
                ),
                radius=radius,
                upsample=int(cellpose_upsample),
                cellpose_model=cellpose_model,
                segmenter=segment_single_img,
            )
        else:
            mask = segment_single_img(
                segmentation_image,
                scale=1,
                cellpose_model=cellpose_model,
                circle_center=center,
                circle_radius=radius,
            )
        raw_mask = np.asarray(mask)
        cellpose_label_count = int(np.count_nonzero(np.unique(raw_mask)))
        pillar_removed_mask = None
        if pillar_shape_filter is not None:
            shape_result = pillar_shape_filter.filter(
                raw_mask,
                image_shape=image.shape,
            )
            mask = shape_result.filtered_labels
            pillar_removed_mask = shape_result.removed_mask
            removed = [
                measurement
                for measurement in shape_result.measurements
                if measurement.removed
            ]
            print(
                "Pillar shape filter for automated selection at "
                f"position {position_index}: removed={len(removed)}/"
                f"{len(shape_result.measurements)}, "
                f"rule={pillar_shape_filter.config.pillar_length:g}x"
                f"{pillar_shape_filter.config.pillar_width:g}px@"
                f"{pillar_shape_filter.config.minimum_rectangle_fit:.0%}, "
                "cellpose_input=raw, labels="
                f"{[(item.label, round(item.rectangle_fit_score, 3)) for item in removed]}"
            )
        else:
            mask = raw_mask
        if constrain_trenches:
            trench_result = filter_labels_to_trenches(
                image,
                mask,
                pose_override=trench_geometry.pose,
            )
            mask = trench_result.filtered_labels
            trench_safe_mask = (
                np.asarray(trench_result.safe_interior_mask, dtype=bool)
                & selection_circle_mask
            )
            mask = np.where(trench_safe_mask, mask, 0)
            circle_safe_masks = tuple(
                np.asarray(safe_mask, dtype=bool) & selection_circle_mask
                for safe_mask in trench_geometry.safe_masks
            )
            mask = _fill_interlabel_trench_gaps(
                mask,
                circle_safe_masks,
            )
            if trench_result.geometry_valid:
                print(
                    "Trench geometry filter for automated selection at "
                    f"position {position_index}: "
                    f"score={trench_result.geometry_score:.3f}, "
                    "one global rotation/translation, "
                    "remaining_labels="
                    f"{int(np.count_nonzero(np.unique(mask)))}"
                )
            else:
                print(
                    "Learned trench geometry was not recognized at "
                    f"position {position_index} "
                    f"(score={trench_result.geometry_score:.3f}); "
                    "skipping this field without cell or autofocus targets."
                )
        masks.append(mask)
        if show_masks:
            viewer.add_labels(
                mask,
                name=f"Cellpose mask p{position_index}",
            )

        if constrain_trenches and not trench_result.geometry_valid:
            selected_by_position.append(np.empty((0, 2), dtype=float))
            continue

        selection_exclusion_mask = _selection_exclusion_mask(
            trench_safe_mask,
            pillar_removed_mask,
        )

        requested_points = N if not center_cell or no_autofocus else N + 1
        selection_radius = np.inf if center_cell else radius
        if constrain_trenches:
            cell_limit = requested_points - (0 if no_autofocus else 1)
            cell_points = get_n_most_centered_coms(
                mask,
                N=cell_limit,
                center=center,
                radius=selection_radius,
                autofocus_object=None,
                bkd_threshold=bkd_thres,
                exclusion_mask=selection_exclusion_mask,
                exclusion_margin=0,
            )
            if no_autofocus:
                selected_points = cell_points
            else:
                background_allowed = selection_circle_mask & ~raw_trench_union
                if pillar_removed_mask is not None:
                    background_allowed &= ~np.asarray(
                        pillar_removed_mask,
                        dtype=bool,
                    )
                background_point = _background_point_in_region(
                    mask,
                    background_allowed,
                    threshold=bkd_thres,
                    center=center,
                )
                selected_points = np.vstack((background_point, cell_points))
        else:
            selected_points = get_n_most_centered_coms(
                mask,
                N=requested_points,
                center=center,
                radius=selection_radius,
                autofocus_object=autofocus_object,
                bkd_threshold=bkd_thres,
                exclusion_mask=selection_exclusion_mask,
                exclusion_margin=0,
            )

        if cell_point_refiner is not None and selected_points.size:
            cell_start = 0 if no_autofocus else 1
            refined_points = np.asarray(selected_points, dtype=float).copy()
            if len(refined_points) > cell_start:
                refined_cells = np.asarray(
                    cell_point_refiner(refined_points[cell_start:]),
                    dtype=float,
                )
                if refined_cells.shape != refined_points[cell_start:].shape:
                    raise ValueError(
                        "cell_point_refiner must preserve the cell point shape."
                    )
                if constrain_trenches:
                    refined_cells = _retain_same_label_refinements(
                        refined_points[cell_start:],
                        refined_cells,
                        mask,
                    )
                refined_points[cell_start:] = refined_cells
            selected_points = refined_points

        selected_cell_count = max(
            0,
            len(selected_points) - (0 if no_autofocus else 1),
        )
        print(
            f"Automated selection at position {position_index}: "
            f"cellpose_labels={cellpose_label_count}, "
            f"remaining_labels={int(np.count_nonzero(np.unique(mask)))}, "
            f"selected_cells={selected_cell_count}"
        )

        selected_by_position.append(
            selected_points
        )

        if not center_cell:
            continue

        if selected_points.size == 0:
            continue

        if no_autofocus:
            cell_points = selected_points
        else:
            cell_points = selected_points[1:]

        if center is None:
            image_center_yx = (
                np.asarray(
                    image.shape[:2],
                    dtype=float,
                )
                / 2
            )
        else:
            image_center_yx = np.asarray(
                center,
                dtype=float,
            )

        for cell_yx in cell_points:
            offset_yx = (
                cell_yx - image_center_yx
            )
            offset_xy = np.array(
                [
                    offset_yx[1],
                    offset_yx[0],
                ]
            )

            if direct_pixel_to_stage:
                # The shared correction below subtracts these values.  Negate
                # the direct offsets so the demo stage moves toward the cell.
                stage_dx, stage_dy = -offset_xy
            else:
                stage_dx, stage_dy = apply_vandermonde_model(
                    offset_xy,
                    coefficients,
                    degree,
                )

            corrected_position = (
                original_position.replace(
                    x=float(
                        original_position.x
                        - stage_dx
                    ),
                    y=float(
                        original_position.y
                        - stage_dy
                    ),
                )
            )
            corrected_positions.append(
                corrected_position
            )

    if center_cell:
        return _finish_centered_selection(
            core=core,
            viewer=viewer,
            point_transformer=point_transformer,
            sequence=sequence,
            images=images,
            selected_by_position=selected_by_position,
            corrected_positions=corrected_positions,
            no_autofocus=no_autofocus,
            center=center,
            block_mda=block_mda,
        )

    return _finish_original_position_selection(
        core=core,
        viewer=viewer,
        point_transformer=point_transformer,
        sequence=sequence,
        selected_by_position=selected_by_position,
        no_autofocus=no_autofocus,
        batch=batch,
        block_mda=block_mda,
    )


def _create_sources(
    viewer: Any,
    point_transformer: Any,
    no_autofocus: bool,
):
    """Create the required cell and autofocus sources."""
    from .layers import create_point_sources

    if no_autofocus:
        return create_point_sources(
            viewer,
            point_transformer,
            size=15,
            names=("cells",),
            colors=("#aa0000ff",),
        )

    return create_point_sources(
        viewer,
        point_transformer,
        size=15,
    )


def _run_selection_mda(core: Any, sequence: Any, block_mda: bool) -> Any:
    """Run the short selection MDA, blocking when the core supports it."""
    try:
        return core.run_mda(sequence, block=bool(block_mda))
    except TypeError:
        return core.run_mda(sequence)


def _finish_centered_selection(
    core: Any,
    viewer: Any,
    point_transformer: Any,
    sequence: Any,
    images: list[np.ndarray],
    selected_by_position: list[np.ndarray],
    corrected_positions: list[Any],
    no_autofocus: bool,
    center: tuple[float, float] | None,
    block_mda: bool = False,
):
    """Create point sources for independently centered cells."""
    new_sequence = sequence.replace(
        stage_positions=corrected_positions
    )
    sources = _create_sources(
        viewer,
        point_transformer,
        no_autofocus,
    )

    if corrected_positions:
        _run_selection_mda(core, new_sequence, block_mda)

    if center is None:
        if not images:
            image_center_yx = np.zeros(2)
        else:
            image_center_yx = (
                np.asarray(
                    images[0].shape[:2],
                    dtype=float,
                )
                / 2
            )
    else:
        image_center_yx = np.asarray(
            center,
            dtype=float,
        )

    new_position_index = 0

    for selected_points in selected_by_position:
        if selected_points.size == 0:
            continue

        if no_autofocus:
            cell_points = selected_points
            autofocus_point = None
        else:
            autofocus_point = selected_points[0]
            cell_points = selected_points[1:]

        for _ in cell_points:
            if autofocus_point is not None:
                _add_point(
                    sources[1],
                    new_position_index,
                    autofocus_point,
                )

            _add_point(
                sources[0],
                new_position_index,
                image_center_yx,
            )

            if point_transformer.multiplier <= 1:
                _add_point(
                    sources[0],
                    new_position_index,
                    image_center_yx,
                )

            new_position_index += 1

    autofocus_positions = np.arange(
        len(corrected_positions)
    )

    return (
        sources,
        autofocus_positions,
        new_sequence,
    )


def _finish_original_position_selection(
    core: Any,
    viewer: Any,
    point_transformer: Any,
    sequence: Any,
    selected_by_position: list[np.ndarray],
    no_autofocus: bool,
    batch: bool,
    block_mda: bool = False,
):
    """Create point sources while retaining original positions."""
    if no_autofocus:
        cell_slice = slice(None)
        repeats = [
            len(points)
            for points in selected_by_position
        ]
    else:
        cell_slice = slice(1, None)
        repeats = [
            max(0, len(points) - 1)
            for points in selected_by_position
        ]

    repeated_positions = [
        position
        for position, repeat_count in zip(
            sequence.stage_positions,
            repeats,
            strict=True,
        )
        for _ in range(repeat_count)
    ]

    new_sequence = sequence.replace(
        stage_positions=repeated_positions
    )
    sources = _create_sources(
        viewer,
        point_transformer,
        no_autofocus,
    )

    if batch:
        if repeated_positions:
            _run_selection_mda(core, new_sequence, block_mda)

        expanded_indices = [
            original_index
            for original_index, repeat_count
            in enumerate(repeats)
            for _ in range(repeat_count)
        ]

        cell_arrays = [
            points[cell_slice]
            for points in selected_by_position
            if len(points[cell_slice]) > 0
        ]

        if cell_arrays:
            all_cells = np.vstack(cell_arrays)
        else:
            all_cells = np.empty(
                (0, 2),
                dtype=float,
            )

        for new_index in range(
            len(repeated_positions)
        ):
            original_index = expanded_indices[
                new_index
            ]

            if not no_autofocus:
                autofocus_point = (
                    selected_by_position[
                        original_index
                    ][0]
                )
                _add_point(
                    sources[1],
                    new_index,
                    autofocus_point,
                )

            _add_point(
                sources[0],
                new_index,
                all_cells[new_index],
            )

        autofocus_positions = np.cumsum(
            [0] + repeats[:-1]
        )

        return (
            sources,
            autofocus_positions,
            new_sequence,
        )

    _run_selection_mda(core, sequence, block_mda)

    for position_index in range(
        len(sequence.stage_positions)
    ):
        selected_points = (
            selected_by_position[
                position_index
            ]
        )

        if (
            not no_autofocus
            and len(selected_points) > 0
        ):
            _add_point(
                sources[1],
                position_index,
                selected_points[0],
            )

        for cell_point in selected_points[
            cell_slice
        ]:
            _add_point(
                sources[0],
                position_index,
                cell_point,
            )

    autofocus_positions = np.arange(
        len(sequence.stage_positions)
    )

    return (
        sources,
        autofocus_positions,
        sequence,
    )
