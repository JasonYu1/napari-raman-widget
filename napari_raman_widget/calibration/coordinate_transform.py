"""Coordinate transformations for Raman targeting and galvo control."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial.polynomial import polyvander2d
from sklearn.linear_model import Ridge

__all__ = ["CoordTransformer"]


class CoordTransformer:
    """Transform brightfield coordinates into Raman and galvo coordinates."""

    def __init__(
        self,
        coef: np.ndarray,
        intercept: np.ndarray,
        vander_degs: tuple[int, int],
    ) -> None:
        self._coef = np.asarray(coef)
        self._intercept = np.asarray(intercept)
        self._vander_degs = tuple(vander_degs)

    def BF_to_RM(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
    ) -> np.ndarray:
        """Transform relative brightfield coordinates into Raman coordinates."""
        X = np.asarray(X)

        if X.ndim == 2:
            if X.shape[1] != 2:
                raise ValueError("A two-dimensional input must have shape (N, 2).")

            Y = X[:, 1]
            X = X[:, 0]
        elif Y is None:
            raise ValueError("Y must be provided when X is one-dimensional.")

        vander = polyvander2d(X, Y, self._vander_degs)
        return vander @ self._coef.T + self._intercept

    def RM_to_volts(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        max_volts: float = 0.6,
    ) -> np.ndarray:
        """Convert relative Raman coordinates into galvo voltages."""
        X = np.asarray(X)

        if X.ndim == 1:
            if Y is None:
                raise ValueError("Y must be provided when X is one-dimensional.")

            X = np.column_stack((X, Y))
        elif X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("Coordinates must have shape (N, 2).")

        return 2 * max_volts * X - max_volts

    def BF_to_volts(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        max_volts: float = 0.6,
    ) -> np.ndarray:
        """Transform brightfield coordinates directly into galvo voltages."""
        raman_coordinates = self.BF_to_RM(X, Y)

        return self.RM_to_volts(
            raman_coordinates,
            max_volts=max_volts,
        )

    @staticmethod
    def fit_model(
        rel_BF: np.ndarray,
        rel_RM: np.ndarray,
        vander_degs: tuple[int, int] = (3, 3),
        alpha: float = 0.05,
    ) -> Ridge:
        """Fit a polynomial ridge model from brightfield to Raman coordinates."""
        rel_BF = np.asarray(rel_BF)
        rel_RM = np.asarray(rel_RM)

        if rel_BF.ndim != 2 or rel_BF.shape[1] != 2:
            raise ValueError("rel_BF must have shape (N, 2).")

        if rel_RM.ndim != 2 or rel_RM.shape[1] != 2:
            raise ValueError("rel_RM must have shape (N, 2).")

        if len(rel_BF) != len(rel_RM):
            raise ValueError("rel_BF and rel_RM must contain the same number of points.")

        model = Ridge(alpha=alpha)
        vander = polyvander2d(
            rel_BF[:, 0],
            rel_BF[:, 1],
            vander_degs,
        )
        model.fit(vander, rel_RM)

        return model

    @staticmethod
    def save_model(
        savename: str | Path,
        model: Ridge,
        vander_degs: tuple[int, int],
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Save a fitted coordinate-transformation model as JSON."""
        combined_metadata = {
            **(metadata or {}),
            **kwargs,
        }

        model_information = {
            "metadata": combined_metadata,
            "vander_degs": list(vander_degs),
            "coef": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
        }

        path = Path(savename)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(model_information, file, indent=2)

    @classmethod
    def from_json(
        cls,
        fname: str | Path | None = None,
    ) -> "CoordTransformer":
        """Load a coordinate-transformation model from JSON."""
        if fname is None:
            path = Path(__file__).with_name("model.json")
        else:
            path = Path(fname)

        with path.open(encoding="utf-8") as file:
            model_information = json.load(file)

        return cls(
            coef=np.asarray(model_information["coef"]),
            intercept=np.asarray(model_information["intercept"]),
            vander_degs=tuple(model_information["vander_degs"]),
        )