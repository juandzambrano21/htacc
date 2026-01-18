"""
catbp/vm/kernels_selector.py

Selector tensor construction for mode variables.

A selector σ_e(J_e, c_e) maps interface states to mode assignments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from catbp.ir.schema import Axis, AxisType, TensorSpec


@dataclass(frozen=True)
class SelectorPayload:
    """
    Payload for constructing a mode selector tensor.
    
    Attributes:
        mode_axis_id: ID of the mode axis
        interface_axis_ids: Variable IDs in the interface J_e
        quotient_map: Mapping from flat interface state to mode ID
        num_modes: Number of modes
        diag_mask: Optional mask for fixed-point feasibility (Patch E)
    """
    mode_axis_id: int
    interface_axis_ids: Tuple[int, ...]
    quotient_map: np.ndarray
    num_modes: int
    diag_mask: Optional[np.ndarray] = None


def vm_load_selector(
    payload: SelectorPayload,
    domains: Dict[int, int],
    mode_sizes: Dict[int, int],
    dtype: type = np.float32,
    one_val=None,
    zero_val=None,
) -> Tuple[np.ndarray, TensorSpec]:
    """
    Construct a selector tensor from payload.
    
    The selector σ_e(J_e, c_e) is a tensor where:
    - σ_e[j, c] = 1 if state j maps to mode c
    - σ_e[j, c] = 0 otherwise
    
    With Patch E (diag_mask), states with H[u,u]=0 are excluded.
    
    Args:
        payload: Selector specification
        domains: Variable domain sizes
        mode_sizes: Mode axis sizes
        dtype: Output dtype
        one_val: Value for allowed entries (semiring one)
        zero_val: Value for disallowed entries (semiring zero)
        
    Returns:
        (selector_array, tensor_spec)
    """
    J = payload.interface_axis_ids
    geo_shape = tuple(domains[v] for v in J)
    flat_dim = int(np.prod(geo_shape, dtype=np.int64))
    Q = int(payload.num_modes)
    
    if one_val is None:
        one_val = 1
    if zero_val is None:
        zero_val = 0

    data = np.full((flat_dim, Q), zero_val, dtype=dtype)
    labels = payload.quotient_map.astype(np.int64, copy=False)
    
    if payload.diag_mask is None:
        # All states are valid
        idx = np.arange(flat_dim, dtype=np.int64)
        # Handle potential -1 labels (infeasible states)
        valid = labels >= 0
        data[idx[valid], labels[valid]] = one_val
    else:
        # Only fixed-point feasible states
        ok = payload.diag_mask.astype(bool, copy=False)
        idx = np.arange(flat_dim, dtype=np.int64)[ok]
        valid_labels = labels[ok]
        valid = valid_labels >= 0
        data[idx[valid], valid_labels[valid]] = one_val
    
    data = data.reshape(geo_shape + (Q,))
    
    # Build spec: geometric axes for interface, topological axis for mode
    axes = [Axis(id=v, kind=AxisType.GEOMETRIC, size=domains[v]) for v in J]
    axes.append(Axis(id=payload.mode_axis_id, kind=AxisType.TOPOLOGICAL, size=mode_sizes[payload.mode_axis_id]))
    
    return data, TensorSpec(axes=tuple(axes))


def vm_load_selector_dense(
    payload: SelectorPayload,
    domains: Dict[int, int]
) -> Tuple[np.ndarray, TensorSpec]:
    """
    Construct a dense selector tensor (legacy interface).
    
    Args:
        payload: Selector specification
        domains: Variable domain sizes
        
    Returns:
        (selector_array, tensor_spec)
    """
    mode_sizes = {payload.mode_axis_id: payload.num_modes}
    return vm_load_selector(payload, domains, mode_sizes, dtype=np.float32)
