"""Parsing for Raman MDA autofocus and imaging position fields."""

from __future__ import annotations


def parse_position_spec(
    text: str,
    position_count: int,
    label: str = "Positions",
) -> list[int] | None:
    """Parse an MDA position specification.

    Blank or ``None`` keeps the caller's existing selection. A single positive
    integer selects every Nth position starting at zero. Text containing a
    comma is treated as an explicit list of zero-based position indices.
    """
    value = str(text).strip()
    if not value or value.lower() == "none":
        return None
    if position_count < 0:
        raise ValueError("position_count cannot be negative")

    if "," in value:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError(f"{label} is empty")
        try:
            positions = [int(part) for part in parts]
        except ValueError as error:
            raise ValueError(
                f"{label} contains non-integer entries: {value!r}"
            ) from error
    else:
        try:
            interval = int(value)
        except ValueError as error:
            raise ValueError(
                f"{label} must be an interval or comma-separated indices: "
                f"{value!r}"
            ) from error
        if interval < 1:
            raise ValueError(f"{label} interval must be at least 1")
        positions = list(range(0, position_count, interval))

    invalid = [
        position
        for position in positions
        if position < 0 or position >= position_count
    ]
    if invalid:
        raise ValueError(
            f"{label} out of range {invalid} "
            f"(sequence has {position_count} positions)"
        )
    return positions


def resolve_position_specs(
    autofocus_text: str,
    imaging_text: str,
    position_count: int,
    selection_positions,
) -> tuple[list[int], list[int]]:
    """Resolve independent autofocus and imaging position controls.

    Blank values use the positions produced by the selection workflow. In
    particular, reducing the autofocus cadence must not reduce BF imaging.
    """
    selected = [int(position) for position in selection_positions]
    autofocus_override = parse_position_spec(
        autofocus_text, position_count, "Autofocus positions"
    )
    imaging_override = parse_position_spec(
        imaging_text, position_count, "Imaging positions"
    )
    autofocus_positions = (
        selected if autofocus_override is None else autofocus_override
    )
    imaging_positions = (
        selected if imaging_override is None else imaging_override
    )
    return autofocus_positions, imaging_positions
