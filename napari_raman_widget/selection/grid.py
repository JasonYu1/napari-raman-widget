"""Grid-based Raman point selection."""

from __future__ import annotations

from typing import Any

import numpy as np
from raman_mda_engine.utils import get_seq_from_napari

from .layers import create_point_sources

__all__ = ["grid_point_selections"]


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
    """Create grid cell and optional autofocus layers."""
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


def _update_mda_widget(
    main_window: Any,
    sequence: Any,
) -> Any:
    """Update the visible MDA widget when its sequence control is available."""
    try:
        mda_dock = main_window._dock_widgets[
            "MDA"
        ]
        children = mda_dock.children()

        if len(children) <= 4:
            return sequence

        sequence_control = children[4]

        if not hasattr(
            sequence_control,
            "setValue",
        ):
            return sequence

        sequence_control.setValue(sequence)

        return get_seq_from_napari(
            main_window
        )

    except Exception:
        return sequence


def grid_point_selections(
    core: Any,
    viewer: Any,
    main_window: Any,
    point_transformer: Any,
    fov_x: float,
    fov_y: float,
    x_range: float,
    y_range: float,
    x_step: float,
    y_step: float,
    repeats: int = 2,
    use_blank_images: bool = True,
    autofocus_object: str | None = "None",
):
    """Generate a stage grid with a fixed Raman point in each field.

    The MDA widget must contain at least one stage position before this
    function is called. Its first position supplies the non-XY position
    settings used for every generated grid position.

    Parameters
    ----------
    core
        Microscope core providing the current XY position and camera size.
    viewer
        Napari viewer receiving the point and placeholder image layers.
    main_window
        MDA widget containing the current sequence.
    point_transformer
        Transformer used by the Raman MDA point sources.
    fov_x, fov_y
        Pixel coordinate measured in every generated field of view.
    x_range, y_range
        Stage distance generated on either side of the current position.
    x_step, y_step
        Spacing between generated stage positions.
    repeats
        Number of repeated Raman points at each stage position. At least
        two points are required by the DAQ output.
    use_blank_images
        Add a broadcast placeholder image instead of acquiring an image
        at every generated position.
    autofocus_object
        Autofocus target. Use ``None`` or ``"None"`` to create only the
        cell layer.

    Returns
    -------
    sources
        Cell point source followed by an optional autofocus point source.
    autofocus_positions
        Index of every generated grid position.
    sequence
        MDA sequence containing the generated stage grid.
    """
    repeats = int(repeats)

    if repeats < 2:
        raise ValueError(
            "repeats must be at least two."
        )

    if x_step <= 0 or y_step <= 0:
        raise ValueError(
            "x_step and y_step must be greater than zero."
        )

    if x_range < 0 or y_range < 0:
        raise ValueError(
            "x_range and y_range cannot be negative."
        )

    no_autofocus = _is_no_autofocus(
        autofocus_object
    )
    sequence = get_seq_from_napari(
        main_window
    )

    if not sequence.stage_positions:
        raise ValueError(
            "Add at least one stage position in the "
            "MDA widget before creating a grid."
        )

    origin_x, origin_y = (
        core.getXYPosition()
    )

    x_positions = np.arange(
        origin_x - x_range,
        origin_x + x_range + x_step / 2,
        x_step,
    )
    y_positions = np.arange(
        origin_y - y_range,
        origin_y + y_range + y_step / 2,
        y_step,
    )

    position_template = (
        sequence.stage_positions[0]
    )
    grid_positions = [
        position_template.replace(
            x=float(stage_x),
            y=float(stage_y),
        )
        for stage_x in x_positions
        for stage_y in y_positions
    ]

    new_sequence = sequence.replace(
        stage_positions=grid_positions
    )
    new_sequence = _update_mda_widget(
        main_window,
        new_sequence,
    )

    sources = _create_sources(
        viewer,
        point_transformer,
        no_autofocus,
    )
    number_of_positions = len(
        new_sequence.stage_positions
    )

    if use_blank_images:
        try:
            image_height = int(
                core.getImageHeight()
            )
            image_width = int(
                core.getImageWidth()
            )
        except Exception:
            image_height = 1024
            image_width = 1344

        blank_frame = np.zeros(
            (image_height, image_width),
            dtype=np.uint16,
        )
        placeholder = np.broadcast_to(
            blank_frame,
            (
                1,
                number_of_positions,
                1,
                1,
                image_height,
                image_width,
            ),
        )

        viewer.add_image(
            placeholder,
            name="Grid placeholder",
        )
    else:
        core.run_mda(new_sequence)

    cell_points = np.asarray(
        [
            [
                0,
                position_index,
                0,
                0,
                fov_y,
                fov_x,
            ]
            for position_index
            in range(number_of_positions)
            for _ in range(repeats)
        ],
        dtype=float,
    )

    if len(cell_points) > 0:
        sources[0]._points.add(
            cell_points
        )

    if not no_autofocus:
        autofocus_points = np.asarray(
            [
                [
                    0,
                    position_index,
                    0,
                    0,
                    fov_y,
                    fov_x,
                ]
                for position_index
                in range(number_of_positions)
            ],
            dtype=float,
        )

        if len(autofocus_points) > 0:
            sources[1]._points.add(
                autofocus_points
            )

    autofocus_positions = np.arange(
        number_of_positions
    )

    return (
        sources,
        autofocus_positions,
        new_sequence,
    )