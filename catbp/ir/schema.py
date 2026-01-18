"""
catbp/ir/schema.py

Intermediate representation schema for tensor specifications.

Key types:
- AxisType: GEOMETRIC (variable) or TOPOLOGICAL (mode)
- AxisKey: (AxisType, id) tuple for unique axis identification
- Axis: Full axis specification with id, type, and size
- TensorSpec: Collection of axes defining a tensor's shape and semantics
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, TypeAlias


class AxisType(Enum):
    """Type of axis in a tensor."""
    GEOMETRIC = 1      # Ordinary variable axes (VarID)
    TOPOLOGICAL = 2    # Mode axes (ModeID), never to be confused with VarID


# AxisKey uniquely identifies an axis by (type, id)
# This prevents collisions between VarID=3 and ModeID=3
AxisKey: TypeAlias = Tuple[AxisType, int]


@dataclass(frozen=True)
class Axis:
    """
    Specification of a single tensor axis.
    
    Attributes:
        id: Unique identifier (VarID or ModeID)
        kind: Type of axis (GEOMETRIC or TOPOLOGICAL)
        size: Domain size
    """
    id: int
    kind: AxisType
    size: int

    @property
    def key(self) -> AxisKey:
        """Get the unique key for this axis."""
        return (self.kind, self.id)


@dataclass(frozen=True)
class TensorSpec:
    """
    Specification of a tensor's axes.
    
    Attributes:
        axes: Ordered tuple of axis specifications
    """
    axes: Tuple[Axis, ...]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the tensor."""
        return tuple(ax.size for ax in self.axes)

    @property
    def keys(self) -> Tuple[AxisKey, ...]:
        """Get the axis keys in order."""
        return tuple(ax.key for ax in self.axes)

    @property
    def ndim(self) -> int:
        """Get the number of dimensions."""
        return len(self.axes)

    def axis_pos(self) -> Dict[AxisKey, int]:
        """Get mapping from axis key to position."""
        return {ax.key: i for i, ax in enumerate(self.axes)}

    def subset(self, keep_keys: Tuple[AxisKey, ...]) -> "TensorSpec":
        """Get a subset of axes by keys, preserving order of keep_keys."""
        pos = self.axis_pos()
        keep = tuple(self.axes[pos[k]] for k in keep_keys if k in pos)
        return TensorSpec(axes=keep)

    def has_key(self, key: AxisKey) -> bool:
        """Check if this spec contains an axis with the given key."""
        return key in self.axis_pos()

    def get_axis(self, key: AxisKey) -> Axis:
        """Get axis by key."""
        pos = self.axis_pos()
        if key not in pos:
            raise KeyError(f"Axis key {key} not found in spec")
        return self.axes[pos[key]]


def geometric_spec(scope: Tuple[int, ...], domains: Dict[int, int]) -> TensorSpec:
    """
    Create a TensorSpec for geometric (variable) axes only.
    
    Args:
        scope: Tuple of variable IDs
        domains: Map from variable ID to domain size
        
    Returns:
        TensorSpec with GEOMETRIC axes
    """
    return TensorSpec(
        axes=tuple(Axis(id=v, kind=AxisType.GEOMETRIC, size=domains[v]) for v in scope)
    )


def sort_axes(axes: List[Axis]) -> Tuple[Axis, ...]:
    """
    Sort axes into canonical order: GEOMETRIC first, then TOPOLOGICAL; stable by id.
    """
    return tuple(sorted(axes, key=lambda a: (a.kind.value, a.id)))
