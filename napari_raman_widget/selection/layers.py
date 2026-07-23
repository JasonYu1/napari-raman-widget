"""Napari point-layer creation tools."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from napari_broadcastable_points import BroadcastablePoints
from raman_mda_engine.aiming import PointsLayerSource

__all__ = ["create_point_sources"]


def create_point_sources(
    viewer: Any,
    point_transformer: Callable,
    broadcast_dims: tuple[int, ...] = (0, 2, 3),
    ndim: int = 6,
    size: float = 35,
    names: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
) -> list[PointsLayerSource]:
    """Create napari point layers used by Raman acquisition.

    Parameters
    ----------
    viewer
        Napari viewer receiving the point layers.
    point_transformer
        Coordinate transformer used by the Raman MDA engine.
    broadcast_dims
        Dimensions across which selected points are broadcast.
    ndim
        Number of dimensions used by each point layer.
    size
        Display size of points in napari.
    names
        Names assigned to the point layers.
    colors
        Face colors assigned to the point layers.

    Returns
    -------
    list of PointsLayerSource
        Raman MDA point sources connected to their napari layers.
    """
    if names is None:
        names = (
            "cells",
            "autofocus",
        )

    if colors is None:
        colors = (
            "#aa0000ff",
            "springgreen",
        )

    names = tuple(names)
    colors = tuple(colors)

    if len(names) != len(colors):
        raise ValueError(
            "names and colors must contain the same number of entries."
        )

    if ndim < 2:
        raise ValueError(
            "ndim must be at least two."
        )

    if size <= 0:
        raise ValueError(
            "size must be greater than zero."
        )

    invalid_dimensions = [
        dimension
        for dimension in broadcast_dims
        if dimension < 0 or dimension >= ndim
    ]

    if invalid_dimensions:
        raise ValueError(
            "Every broadcast dimension must be within "
            f"the range 0 to {ndim - 1}."
        )

    sources: list[PointsLayerSource] = []

    for name, color in zip(
        names,
        colors,
        strict=True,
    ):
        points_layer = BroadcastablePoints(
            None,
            broadcast_dims=broadcast_dims,
            ndim=ndim,
            name=name,
            size=size,
            face_color=color,
            border_color="#5500ffff",
        )

        viewer.add_layer(points_layer)

        source = PointsLayerSource(
            points_layer,
            name=name,
            transformer=point_transformer,
        )
        sources.append(source)

    return sources