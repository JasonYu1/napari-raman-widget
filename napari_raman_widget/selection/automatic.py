"""Automatic segmentation and Raman target selection."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from raman_mda_engine.aiming.autotracking import segment_single_img
from raman_mda_engine.utils import get_seq_from_napari
from tqdm.auto import tqdm

from napari_raman_widget.calibration.models import (
    apply_vandermonde_model,
    load_vandermonde_model,
)

from .layers import create_point_sources
from .masks import get_n_most_centered_coms

__all__ = ["automated_point_selections"]


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

        if int(cellpose_upsample) > 1:
            from napari_raman_widget.demo.cellpose import (
                segment_upsampled_demo_region,
            )

            mask = segment_upsampled_demo_region(
                image,
                center_yx=(
                    center
                    if center is not None
                    else tuple(np.asarray(image.shape, dtype=float) / 2)
                ),
                radius=radius,
                upsample=int(cellpose_upsample),
                cellpose_model=cellpose_model,
                segmenter=segment_single_img,
            )
        else:
            mask = segment_single_img(
                image,
                scale=1,
                cellpose_model=cellpose_model,
                circle_center=center,
                circle_radius=radius,
            )
        mask = np.asarray(mask)
        masks.append(mask)
        if show_masks:
            viewer.add_labels(
                mask,
                name=f"Cellpose mask p{position_index}",
            )

        if center_cell:
            requested_points = (
                N
                if no_autofocus
                else N + 1
            )

            selected_points = (
                get_n_most_centered_coms(
                    mask,
                    N=requested_points,
                    center=center,
                    radius=np.inf,
                    autofocus_object=autofocus_object,
                    bkd_threshold=bkd_thres,
                )
            )
        else:
            selected_points = (
                get_n_most_centered_coms(
                    mask,
                    N=N,
                    center=center,
                    radius=radius,
                    autofocus_object=autofocus_object,
                    bkd_threshold=bkd_thres,
                )
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
                refined_points[cell_start:] = refined_cells
            selected_points = refined_points

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
