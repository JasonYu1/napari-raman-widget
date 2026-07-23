"""Manual Raman target selection and stage centering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from raman_mda_engine.utils import get_seq_from_napari

from napari_raman_widget.calibration.models import (
    apply_vandermonde_model,
    load_vandermonde_model,
)

from .layers import create_point_sources

__all__ = [
    "center_manual_selections",
    "manual_point_selections",
]


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


def _create_sources(
    viewer: Any,
    point_transformer: Any,
    no_autofocus: bool,
):
    """Create the required manual-selection point layers."""
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


def _add_point(
    source: Any,
    position_index: int,
    point_yx: np.ndarray,
) -> None:
    """Add a point to a six-dimensional Raman point source."""
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


def manual_point_selections(
    core: Any,
    viewer: Any,
    main_window: Any,
    point_transformer: Any,
    N: int,
    autofocus_object: str | None = "glass",
    batch: bool = True,
):
    """Prepare napari layers for manually selected Raman targets.

    The MDA sequence must be configured in the MDA widget before this
    function is called.

    In batch mode, every stage position is repeated ``N`` times. The user
    should select ``N`` cell points for each original field of view.

    Outside batch mode, the stage positions are not repeated and the user
    can select a variable number of cells in each field of view.

    Parameters
    ----------
    core
        Microscope core used to initialize the MDA dimensions.
    viewer
        Napari viewer receiving the point layers.
    main_window
        MDA widget containing the current sequence.
    point_transformer
        Transformer used by the Raman MDA point sources.
    N
        Number of cell selections expected per field of view in batch
        mode.
    autofocus_object
        Autofocus target. Use ``None`` or ``"None"`` to create only the
        cell layer.
    batch
        Whether each stage position should be repeated ``N`` times.

    Returns
    -------
    sources
        Empty cell layer followed by an optional autofocus layer.
    autofocus_positions
        Indices identifying the beginning of each field of view.
    sequence
        Sequence used for the subsequent acquisition.
    """
    if batch and N < 1:
        raise ValueError(
            "N must be at least one in batch mode."
        )

    no_autofocus = _is_no_autofocus(
        autofocus_object
    )
    sequence = get_seq_from_napari(
        main_window
    )
    sources = _create_sources(
        viewer,
        point_transformer,
        no_autofocus,
    )

    if batch:
        repeats = [
            N
            for _ in sequence.stage_positions
        ]
        repeated_positions = [
            position
            for position in sequence.stage_positions
            for _ in range(N)
        ]

        new_sequence = sequence.replace(
            stage_positions=repeated_positions
        )

        if repeated_positions:
            core.run_mda(new_sequence)

        autofocus_positions = np.cumsum(
            [0] + repeats[:-1]
        )

        return (
            sources,
            autofocus_positions,
            new_sequence,
        )

    if sequence.stage_positions:
        core.run_mda(sequence)

    autofocus_positions = np.arange(
        len(sequence.stage_positions)
    )

    return (
        sources,
        autofocus_positions,
        sequence,
    )


def center_manual_selections(
    core: Any,
    viewer: Any,
    main_window: Any,
    point_transformer: Any,
    sources: list[Any],
    vandermonde_model_path: str | Path,
    autofocus_object: str | None = "glass",
    center: tuple[float, float] | None = None,
):
    """Convert manually selected cells into centered stage positions.

    Run ``manual_point_selections`` with ``batch=False`` first. After its
    point layers appear, select cell points in the ``cells`` layer. When
    autofocus is enabled, select one autofocus point for every field of
    view containing cells.

    Each selected cell becomes a separate corrected stage position.

    Parameters
    ----------
    core
        Microscope core providing the camera dimensions.
    viewer
        Napari viewer receiving the new centered point layers.
    main_window
        MDA widget containing the original sequence.
    point_transformer
        Transformer used by the Raman MDA point sources.
    sources
        Sources returned by non-batch ``manual_point_selections``.
    vandermonde_model_path
        Pixel-offset-to-stage-offset calibration model.
    autofocus_object
        Autofocus target. Use ``None`` or ``"None"`` to disable the
        autofocus layer.
    center
        Desired image center in ``(y, x)`` order. The camera center is
        used when omitted.

    Returns
    -------
    sources
        New cell and optional autofocus point sources.
    autofocus_positions
        Index of every newly centered stage position.
    sequence
        Sequence containing one corrected position per selected cell.
    """
    if not sources:
        raise ValueError(
            "No manual point sources were provided."
        )

    no_autofocus = _is_no_autofocus(
        autofocus_object
    )
    coefficients, degree = (
        load_vandermonde_model(
            vandermonde_model_path
        )
    )
    sequence = get_seq_from_napari(
        main_window
    )

    image_height = int(
        core.getImageHeight()
    )
    image_width = int(
        core.getImageWidth()
    )

    if center is None:
        image_center_yx = np.array(
            [
                image_height / 2,
                image_width / 2,
            ],
            dtype=float,
        )
    else:
        image_center_yx = np.asarray(
            center,
            dtype=float,
        )

        if image_center_yx.shape != (2,):
            raise ValueError(
                "center must contain one (y, x) coordinate."
            )

    cell_data = np.asarray(
        sources[0]._points.data,
        dtype=float,
    )

    if cell_data.size == 0:
        raise ValueError(
            "No cell points were selected. Select cells "
            "in the cells layer before centering."
        )

    cell_data = np.atleast_2d(
        cell_data
    )

    if cell_data.shape[1] < 3:
        raise ValueError(
            "The cell point data does not contain "
            "position, y, and x coordinates."
        )

    autofocus_data = None

    if not no_autofocus:
        if len(sources) < 2:
            raise ValueError(
                "Autofocus is enabled, but no autofocus "
                "point source was provided."
            )

        autofocus_data = np.asarray(
            sources[1]._points.data,
            dtype=float,
        )

        if autofocus_data.size == 0:
            raise ValueError(
                "Autofocus is enabled, but no autofocus "
                "points were selected."
            )

        autofocus_data = np.atleast_2d(
            autofocus_data
        )

    position_indices = (
        cell_data[:, 1].astype(int)
    )
    corrected_positions = []
    autofocus_points = []

    for position_index in sorted(
        set(position_indices)
    ):
        if (
            position_index < 0
            or position_index
            >= len(sequence.stage_positions)
        ):
            raise ValueError(
                f"Selected position index {position_index} "
                "is outside the MDA sequence."
            )

        original_position = (
            sequence.stage_positions[
                position_index
            ]
        )
        cell_points_yx = cell_data[
            position_indices
            == position_index
        ][:, -2:]

        autofocus_point_yx = None

        if autofocus_data is not None:
            autofocus_here = autofocus_data[
                autofocus_data[:, 1].astype(int)
                == position_index
            ]

            if len(autofocus_here) == 0:
                raise ValueError(
                    "No autofocus point was selected at "
                    f"stage position {position_index}."
                )

            autofocus_point_yx = (
                autofocus_here[0, -2:]
            )

        for cell_point_yx in cell_points_yx:
            pixel_offset_yx = (
                cell_point_yx
                - image_center_yx
            )
            pixel_offset_xy = np.array(
                [
                    pixel_offset_yx[1],
                    pixel_offset_yx[0],
                ]
            )

            stage_dx, stage_dy = (
                apply_vandermonde_model(
                    pixel_offset_xy,
                    coefficients,
                    degree,
                )
            )

            corrected_positions.append(
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

            if autofocus_point_yx is not None:
                autofocus_points.append(
                    autofocus_point_yx
                )

    if not corrected_positions:
        raise ValueError(
            "The selected points did not produce any "
            "corrected stage positions."
        )

    new_sequence = sequence.replace(
        stage_positions=corrected_positions
    )
    new_sources = _create_sources(
        viewer,
        point_transformer,
        no_autofocus,
    )

    core.run_mda(new_sequence)

    for position_index in range(
        len(corrected_positions)
    ):
        if not no_autofocus:
            _add_point(
                new_sources[1],
                position_index,
                autofocus_points[position_index],
            )

        _add_point(
            new_sources[0],
            position_index,
            image_center_yx,
        )

        if point_transformer.multiplier <= 1:
            _add_point(
                new_sources[0],
                position_index,
                image_center_yx,
            )

    autofocus_positions = np.arange(
        len(corrected_positions)
    )

    return (
        new_sources,
        autofocus_positions,
        new_sequence,
    )