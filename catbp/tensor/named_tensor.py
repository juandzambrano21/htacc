"""
catbp/tensor/named_tensor.py

Named tensor operations with canonical ordering and alignment.

A NamedTensor associates variable IDs with tensor axes, enabling:
- Canonical ordering for deterministic operations
- Automatic alignment for broadcasting
- Semiring-aware reduction operations
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np


class AxisType(Enum):
    """Type of axis in a tensor."""
    GEOMETRIC = 1      # Ordinary variable axes (VarID)
    TOPOLOGICAL = 2    # Mode axes (ModeID)


@dataclass(frozen=True)
class Axis:
    """Axis specification with id, type, and size."""
    id: int
    kind: AxisType
    size: int

    @property
    def key(self) -> Tuple[AxisType, int]:
        return (self.kind, self.id)


@dataclass(frozen=True)
class TensorSpec:
    """Specification of tensor axes."""
    axes: Tuple[Axis, ...]

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(ax.size for ax in self.axes)

    @property
    def keys(self) -> Tuple[Tuple[AxisType, int], ...]:
        return tuple(ax.key for ax in self.axes)

    def axis_pos(self) -> Dict[Tuple[AxisType, int], int]:
        return {ax.key: i for i, ax in enumerate(self.axes)}


@dataclass
class NamedTensor:
    """
    A tensor with named variable axes.
    
    Attributes:
        vars: Tuple of variable IDs (axis labels)
        data: The underlying numpy array
        semiring: The semiring for operations
    """
    vars: Tuple[int, ...]
    data: np.ndarray
    semiring: Any

    def __post_init__(self):
        if len(self.vars) != self.data.ndim:
            raise ValueError(
                f"NamedTensor vars/data mismatch: {len(self.vars)} vars but {self.data.ndim} dims"
            )

    def spec(self, domains: Dict[int, int], mode_sizes: Dict[int, int] = None) -> TensorSpec:
        """Generate TensorSpec from this tensor."""
        if mode_sizes is None:
            mode_sizes = {}
        axes = []
        for vid in self.vars:
            if vid in domains:
                axes.append(Axis(id=vid, kind=AxisType.GEOMETRIC, size=domains[vid]))
            elif vid in mode_sizes:
                axes.append(Axis(id=vid, kind=AxisType.TOPOLOGICAL, size=mode_sizes[vid]))
            else:
                # Infer from data shape
                idx = self.vars.index(vid)
                axes.append(Axis(id=vid, kind=AxisType.GEOMETRIC, size=self.data.shape[idx]))
        return TensorSpec(axes=tuple(axes))


def canonical(vars_: Iterable[int]) -> Tuple[int, ...]:
    """Return canonical (sorted) ordering of variables."""
    return tuple(sorted(vars_))


def align_to(t: NamedTensor, target_vars: Tuple[int, ...], domains: Dict[int, int]) -> NamedTensor:
    """
    Return a broadcastable view of t with axes in target_vars order.
    
    Args:
        t: Input named tensor
        target_vars: Target variable ordering
        domains: Variable domain sizes
        
    Returns:
        NamedTensor aligned to target_vars ordering
    """
    if t.vars == target_vars:
        return t

    # Map existing axis -> position
    pos = {v: i for i, v in enumerate(t.vars)}
    
    # Build reshape with singleton dims for missing axes
    new_shape = []
    transpose_axes = []
    for v in target_vars:
        if v in pos:
            transpose_axes.append(pos[v])
            new_shape.append(t.data.shape[pos[v]])
        else:
            new_shape.append(domains[v])

    # First, transpose existing axes into the order they appear in target_vars
    existing_in_target_order = [pos[v] for v in target_vars if v in pos]
    if existing_in_target_order:
        transposed = np.transpose(t.data, axes=existing_in_target_order)
    else:
        transposed = t.data

    # Now expand by inserting singleton axes where missing, then broadcast
    reshape_shape = []
    j = 0
    for v in target_vars:
        if v in pos:
            reshape_shape.append(transposed.shape[j])
            j += 1
        else:
            reshape_shape.append(1)

    view = transposed.reshape(tuple(reshape_shape))
    return NamedTensor(target_vars, view, t.semiring)


def sum_out(t: NamedTensor, elim_vars: Iterable[int]) -> NamedTensor:
    """
    Sum out (marginalize) specified variables using semiring addition.
    
    Args:
        t: Input named tensor
        elim_vars: Variables to eliminate
        
    Returns:
        NamedTensor with eliminated variables removed
    """
    elim = set(elim_vars)
    if not elim:
        return t
        
    keep_vars = tuple(v for v in t.vars if v not in elim)
    if not keep_vars:
        # Reduce all
        if hasattr(t.semiring, 'add_reduce'):
            reduced = t.semiring.add_reduce(t.data, axis=None)
        else:
            reduced = np.sum(t.data)
        return NamedTensor((), np.atleast_1d(reduced).reshape(()), t.semiring)

    axes = tuple(i for i, v in enumerate(t.vars) if v in elim)
    if hasattr(t.semiring, 'add_reduce'):
        reduced = t.semiring.add_reduce(t.data, axis=axes)
    else:
        reduced = np.sum(t.data, axis=axes)
        
    return NamedTensor(keep_vars, reduced, t.semiring)


def mul_all(ts: List[NamedTensor], target_vars: Tuple[int, ...], domains: Dict[int, int]) -> NamedTensor:
    """
    Multiply all tensors together, aligned to target_vars.
    
    Args:
        ts: List of named tensors to multiply
        target_vars: Target variable ordering for result
        domains: Variable domain sizes
        
    Returns:
        Product of all tensors
    """
    if not ts:
        raise ValueError("mul_all needs at least one tensor")
        
    sr = ts[0].semiring
    acc = align_to(ts[0], target_vars, domains).data
    
    for t in ts[1:]:
        x = align_to(t, target_vars, domains).data
        # Broadcast to same shape
        acc = np.broadcast_to(acc, x.shape)
        x = np.broadcast_to(x, acc.shape)
        
        # Semiring multiplication
        if hasattr(sr, 'mul') and callable(sr.mul):
            if hasattr(sr, 'dtype'):
                # SemiringRuntime
                acc = sr.mul(acc, x)
            else:
                # Scalar semiring - check if it's boolean
                if isinstance(sr.zero, bool):
                    acc = np.logical_and(acc, x)
                else:
                    acc = acc * x
        else:
            acc = acc * x
            
    return NamedTensor(target_vars, acc, sr)


def reorder_tensor(data: np.ndarray, old_vars: Tuple[int, ...], new_vars: Tuple[int, ...]) -> np.ndarray:
    """
    Reorder tensor axes from old_vars ordering to new_vars ordering.
    
    Args:
        data: Input array
        old_vars: Current variable ordering
        new_vars: Target variable ordering (must be permutation of old_vars)
        
    Returns:
        Transposed array
    """
    if old_vars == new_vars:
        return data
        
    pos = {v: i for i, v in enumerate(old_vars)}
    perm = [pos[v] for v in new_vars]
    return np.transpose(data, axes=perm)
