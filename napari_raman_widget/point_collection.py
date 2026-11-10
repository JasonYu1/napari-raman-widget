"""Measured progress for finite repeated detector acquisitions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .acquisition.control import check_cancelled


def collect_repeated_spectra(
    acquire: Callable[..., np.ndarray],
    point: np.ndarray,
    repeats: int,
    exposure_ms: float,
    *,
    read_mode_options: dict,
    progress: Callable[[int, int, str], None],
    cancelled: Callable[[], bool],
    point_index: int | None = None,
    stage_prefix: str | None = None,
) -> np.ndarray:
    """Collect repeats in real batches and report only completed detector work.

    Collector calls contain at least two repeats. Batches target roughly five
    seconds and normally cap at 32 repeats. A final singleton is folded into
    the preceding batch (which can therefore contain 33) because the hardware
    collector requires at least two points.
    """
    repeats = int(repeats)
    if repeats < 2:
        raise ValueError("repeats must be at least 2")

    point = np.asarray(point, dtype=float).reshape(-1)
    if point.shape != (2,):
        raise ValueError("point must contain exactly two coordinates")

    exposure_ms = float(exposure_ms)
    batch_size = max(
        2,
        min(repeats, 32, int(5000 / max(exposure_ms, 1.0))),
    )
    if stage_prefix is None:
        if point_index is None:
            raise ValueError("point_index is required without a stage_prefix")
        stage_prefix = f"Collecting spectra at point {point_index}"
    else:
        stage_prefix = str(stage_prefix).strip()
        if not stage_prefix:
            raise ValueError("stage_prefix must not be empty")
    progress(0, repeats, f"{stage_prefix} (0/{repeats})")

    chunks = []
    completed = 0
    while completed < repeats:
        check_cancelled(cancelled)
        end = min(repeats, completed + batch_size)
        if repeats - end == 1:
            end += 1

        chunk = np.asarray(
            acquire(
                np.tile(point, (end - completed, 1)),
                exposure_ms,
                **read_mode_options,
            )
        )
        if chunk.ndim < 1 or chunk.shape[0] != end - completed:
            raise ValueError("Detector returned an unexpected number of spectra.")
        chunks.append(chunk.copy())
        completed = end
        progress(
            completed,
            repeats,
            f"{stage_prefix} ({completed}/{repeats})",
        )
        if completed < repeats:
            check_cancelled(cancelled)

    return np.concatenate(chunks, axis=0)
