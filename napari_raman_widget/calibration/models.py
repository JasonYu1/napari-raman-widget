"""Vandermonde models for mapping image offsets to stage offsets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "apply_vandermonde",
    "apply_vandermonde_model",
    "fit_vandermonde",
    "load_vandermonde_model",
    "save_vandermonde_model",
]


def _vandermonde_design(
    points: np.ndarray,
    degree: int,
) -> np.ndarray:
    """Build a two-dimensional polynomial design matrix.

    The term ordering is graded by total degree. Within each total
    degree, the exponent of x increases while the exponent of y
    decreases.

    For degree 2, the ordering is:

        1, y, x, y**2, x*y, x**2

    This ordering must remain consistent when fitting and applying a
    saved model.
    """
    if degree < 0:
        raise ValueError("degree must be zero or greater.")

    points = np.asarray(points, dtype=float)

    if points.ndim == 1:
        if points.shape[0] != 2:
            raise ValueError("A single point must contain exactly two values.")

        points = points.reshape(1, 2)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2).")

    x = points[:, 0]
    y = points[:, 1]
    terms = []

    for total_degree in range(degree + 1):
        for x_degree in range(total_degree + 1):
            y_degree = total_degree - x_degree
            terms.append((x**x_degree) * (y**y_degree))

    return np.stack(terms, axis=1)


def fit_vandermonde(
    points: np.ndarray,
    targets: np.ndarray,
    degree: int,
) -> np.ndarray:
    """Fit a polynomial mapping from input points to target coordinates.

    Parameters
    ----------
    points
        Input coordinates with shape ``(N, 2)``. For stage calibration,
        these are normally pixel offsets from the image center.
    targets
        Target coordinates with shape ``(N, 2)``. For stage calibration,
        these are normally stage-position offsets.
    degree
        Maximum total polynomial degree.

    Returns
    -------
    numpy.ndarray
        Coefficient matrix with shape ``(number_of_terms, 2)``.
    """
    points = np.asarray(points, dtype=float)
    targets = np.asarray(targets, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2).")

    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("targets must have shape (N, 2).")

    if len(points) != len(targets):
        raise ValueError("points and targets must contain the same number of rows.")

    design = _vandermonde_design(points, degree)
    coefficients, *_ = np.linalg.lstsq(
        design,
        targets,
        rcond=None,
    )

    return coefficients


def apply_vandermonde(
    points: np.ndarray,
    coefficients: np.ndarray,
    degree: int,
) -> np.ndarray:
    """Apply a fitted Vandermonde model to one or more points."""
    design = _vandermonde_design(points, degree)
    coefficients = np.asarray(coefficients, dtype=float)

    expected_terms = design.shape[1]

    if (
        coefficients.ndim != 2
        or coefficients.shape[0] != expected_terms
        or coefficients.shape[1] != 2
    ):
        raise ValueError(
            "coefficients must have shape "
            f"({expected_terms}, 2) for degree {degree}."
        )

    return design @ coefficients


def apply_vandermonde_model(
    source: np.ndarray,
    coefficients: np.ndarray,
    degree: int,
) -> np.ndarray:
    """Apply a pixel-offset-to-stage-offset model.

    A single input point returns an array with shape ``(2,)``. Multiple
    points return an array with shape ``(N, 2)``.
    """
    source = np.asarray(source, dtype=float)
    single_point = source.ndim == 1

    result = apply_vandermonde(
        source,
        coefficients,
        degree,
    )

    if single_point:
        return result[0]

    return result


def save_vandermonde_model(
    json_path: str | Path,
    coefficients: np.ndarray,
    degree: int,
    img_center: np.ndarray | None = None,
    xy_center: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a fitted Vandermonde model as JSON."""
    coefficients = np.asarray(coefficients, dtype=float)

    model: dict[str, Any] = {
        "degree": int(degree),
        "C": coefficients.tolist(),
    }

    if img_center is not None:
        model["img_center"] = np.asarray(img_center, dtype=float).tolist()

    if xy_center is not None:
        model["xy_center"] = np.asarray(xy_center, dtype=float).tolist()

    if metadata is not None:
        model["metadata"] = metadata

    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(model, file, indent=2)


def load_vandermonde_model(
    json_path: str | Path,
) -> tuple[np.ndarray, int]:
    """Load a Vandermonde model from JSON.

    Returns
    -------
    coefficients
        Model coefficient matrix.
    degree
        Polynomial degree used to fit the model.
    """
    path = Path(json_path)

    with path.open(encoding="utf-8") as file:
        model = json.load(file)

    if "C" not in model:
        raise ValueError(f"The model file does not contain 'C': {path}")

    if "degree" not in model:
        raise ValueError(f"The model file does not contain 'degree': {path}")

    coefficients = np.asarray(model["C"], dtype=float)
    degree = int(model["degree"])

    expected_terms = (degree + 1) * (degree + 2) // 2

    if coefficients.shape != (expected_terms, 2):
        raise ValueError(
            f"Expected coefficient shape ({expected_terms}, 2), "
            f"but found {coefficients.shape}."
        )

    return coefficients, degree