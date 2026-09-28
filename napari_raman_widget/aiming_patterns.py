"""Point-pattern construction shared by the hardware and demo widgets."""

from __future__ import annotations

import numpy as np


class CenteredPointTransformer:
    """Preserve each selected coordinate as one exact aiming point."""

    @property
    def multiplier(self) -> int:
        return 1

    def transform(self, coordinates: np.ndarray) -> np.ndarray:
        points = np.asarray(coordinates, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("coordinates must have shape (N, 2).")
        return points.copy()


def make_point_transformer(
    shape: str,
    size_px: float,
    number_of_points: int,
    image_width: int,
):
    """Build an aiming pattern centered on every selected coordinate.

    The engine's one-point ``Square`` pattern lies on a corner, and its
    one-point ``Circle`` pattern may be empty.  A singleton request therefore
    uses an identity transformer so the selected cell remains the target.
    """
    number_of_points = max(1, int(number_of_points))
    if number_of_points == 1:
        return CenteredPointTransformer()

    from raman_mda_engine.aiming.transformers import Circle, Square

    length = float(size_px) / float(image_width)
    if str(shape).strip().lower() == "circle":
        return Circle(length, number_of_points)
    return Square(length, number_of_points)


__all__ = ["CenteredPointTransformer", "make_point_transformer"]
