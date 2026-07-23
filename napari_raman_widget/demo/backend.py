"""Factory for constructing a coherent demonstration backend."""

from __future__ import annotations

from dataclasses import dataclass

from .collector import DemoSpectraCollector
from .core import DemoCore
from .daq import DemoDAQ
from .world import DemoWorld


@dataclass(frozen=True)
class DemoBackend:
    world: DemoWorld
    core: DemoCore
    daq: DemoDAQ
    collector: DemoSpectraCollector


def create_demo_backend(**world_options: object) -> DemoBackend:
    """Create demo devices that all operate on one virtual sample."""
    world = DemoWorld(**world_options)
    daq = DemoDAQ(world)
    return DemoBackend(
        world=world,
        core=DemoCore(world),
        daq=daq,
        collector=DemoSpectraCollector(world, daq),
    )
