"""Factory for constructing a coherent demonstration backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .collector import DemoSpectraCollector
from .core import DemoCore
from .daq import DemoDAQ
from .transformer import DemoCoordinateTransformer
from .world import DemoWorld


@dataclass(frozen=True)
class DemoBackend:
    world: DemoWorld
    core: Any
    daq: DemoDAQ
    collector: DemoSpectraCollector
    transformer: DemoCoordinateTransformer


def create_demo_backend(
    *,
    core: Any | None = None,
    world: DemoWorld | None = None,
    **world_options: object,
) -> DemoBackend:
    """Create demo devices that all operate on one virtual sample."""
    if world is None:
        world = DemoWorld(**world_options)
    daq = DemoDAQ(world)
    return DemoBackend(
        world=world,
        core=core if core is not None else DemoCore(world),
        daq=daq,
        collector=DemoSpectraCollector(
            world,
            daq,
            core=core if core is not None else None,
        ),
        transformer=DemoCoordinateTransformer(),
    )
