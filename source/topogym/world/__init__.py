"""Shared scene construction: the entity registry and the world state. The physics is MuJoCo (topogym.world.mujoco)."""

from topogym.world.entities import EntityRegistry
from topogym.world.state import WorldState

__all__ = [
    "WorldState",
    "EntityRegistry",
]
