"""Coordinate transformer used by the demonstration backend."""

from __future__ import annotations

import numpy as np


class DemoCenterPointTransformer:
    """Identity aiming pattern used for one exact demo target."""

    @property
    def multiplier(self) -> int:
        return 1

    def transform(self, coordinates: np.ndarray) -> np.ndarray:
        points = np.asarray(coordinates, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("coordinates must have shape (N, 2).")
        return points.copy()


def make_demo_point_transformer(
    shape: str,
    size_px: float,
    number_of_points: int,
    image_width: int,
):
    """Build a centered point pattern for the hardware-free demo.

    The upstream Square/Circle implementations place a one-point pattern at
    an edge of the requested pattern.  In demo mode a one-point pattern must
    preserve the clicked pixel exactly so the fake laser and saved dataset
    marker remain on the selected sphere.
    """
    number_of_points = max(1, int(number_of_points))
    if number_of_points == 1:
        return DemoCenterPointTransformer()

    from raman_mda_engine.aiming.transformers import Circle, Square

    length = float(size_px) / float(image_width)
    if str(shape).strip().lower() == "circle":
        return Circle(length, number_of_points)
    return Square(length, number_of_points)


class DemoCoordinateTransformer:
    """Map normalized image coordinates directly to normalized Raman space."""

    def BF_to_RM(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
    ) -> np.ndarray:
        coordinates = np.asarray(X, dtype=float)
        if coordinates.ndim == 1:
            if Y is None:
                if coordinates.shape != (2,):
                    raise ValueError("coordinates must contain x and y values.")
                coordinates = coordinates.reshape(1, 2)
            else:
                coordinates = np.column_stack((coordinates, Y))
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates must have shape (N, 2).")
        return coordinates.copy()

    def RM_to_volts(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        max_volts: float = 0.6,
    ) -> np.ndarray:
        coordinates = self.BF_to_RM(X, Y)
        return 2 * float(max_volts) * coordinates - float(max_volts)

    def BF_to_volts(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        max_volts: float = 0.6,
    ) -> np.ndarray:
        return self.RM_to_volts(X, Y, max_volts=max_volts)
