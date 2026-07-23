"""Raman MDA sequence configuration."""

from __future__ import annotations

from typing import Any

from raman_mda_engine.utils import get_seq_from_napari

__all__ = ["set_up_new_seq"]


_MINIMUM_RAMAN_EXPOSURE_MS = 73.8


def set_up_new_seq(
    main_window: Any,
    point_transformer: Any,
    engine: Any,
    seq: Any | None = None,
    total_exposure: float = 1000,
    batch: bool = False,
    z_plan: str = "all",
):
    """Configure Raman exposure and Z-plane metadata.

    Parameters
    ----------
    main_window
        MDA widget containing the current sequence.
    point_transformer
        Raman point transformer whose multiplier determines how many
        acquisitions contribute to the requested total exposure.
    engine
        Raman MDA engine receiving the calculated exposure.
    seq
        Existing sequence to update. When omitted, the current sequence
        is obtained from the MDA widget.
    total_exposure
        Requested total Raman exposure in milliseconds.
    batch
        When false, divide the requested exposure by the point
        transformer's multiplier. In batch mode, use the requested
        exposure directly.
    z_plan
        ``"all"`` collects Raman spectra at every Z position.
        ``"middle"`` collects only the middle Z position.

    Returns
    -------
    object
        Updated MDA sequence containing Raman Z-plane metadata.
    """
    total_exposure = float(total_exposure)

    if total_exposure <= 0:
        raise ValueError(
            "total_exposure must be greater than zero."
        )

    multiplier = float(
        point_transformer.multiplier
    )

    if multiplier <= 0:
        raise ValueError(
            "point_transformer.multiplier must be greater than zero."
        )

    if batch:
        exposure_per_collection = (
            total_exposure
        )
    else:
        exposure_per_collection = (
            total_exposure / multiplier
        )

    if (
        exposure_per_collection
        < _MINIMUM_RAMAN_EXPOSURE_MS
    ):
        raise ValueError(
            "The exposure for each Raman collection must "
            f"be at least {_MINIMUM_RAMAN_EXPOSURE_MS} ms. "
            f"The calculated exposure was "
            f"{exposure_per_collection:.3f} ms."
        )

    engine.default_rm_exposure = (
        exposure_per_collection
    )

    if seq is None:
        seq = get_seq_from_napari(
            main_window
        )

    if seq is None:
        raise ValueError(
            "No MDA sequence is available."
        )

    if seq.z_plan is None:
        number_of_z_positions = 1
    else:
        number_of_z_positions = int(
            seq.z_plan.num_positions()
        )

    if number_of_z_positions < 1:
        raise ValueError(
            "The MDA sequence must contain at least one Z position."
        )

    normalized_z_plan = (
        z_plan.strip().lower()
    )

    if normalized_z_plan == "all":
        raman_z_positions = list(
            range(number_of_z_positions)
        )

    elif normalized_z_plan == "middle":
        raman_z_positions = [
            number_of_z_positions // 2
        ]

    else:
        raise ValueError(
            "z_plan must be either 'all' or 'middle'."
        )

    metadata = dict(
        seq.metadata or {}
    )
    metadata["raman"] = {
        "z": raman_z_positions,
    }

    return seq.replace(
        metadata=metadata
    )