"""
catbp/vm/align.py

Tensor alignment utilities for the VM.

Alignment is keyed by AxisKey = (AxisType, id) to prevent collisions
between variable IDs and mode IDs.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from catbp.ir.schema import AxisKey, TensorSpec


def _axis_pos(spec: TensorSpec) -> Dict[AxisKey, int]:
    """Get mapping from axis key to position."""
    return {ax.key: i for i, ax in enumerate(spec.axes)}


def align_array_to(arr: np.ndarray, in_spec: TensorSpec, out_spec: TensorSpec) -> np.ndarray:
    """
    Reorder + reshape (+ broadcast later by numpy) arr from in_spec to out_spec.
    
    Rules:
      - Match axes by AxisKey=(AxisType, id), NOT id alone.
      - Missing out axes are inserted as singleton dims.
      - Extra in axes not in out_spec must be size-1, else compiler bug.
      - Does NOT explicitly broadcast to out_spec.shape; vm_contract will broadcast via ops.
    
    Args:
        arr: Input array
        in_spec: Specification of input array
        out_spec: Target specification
        
    Returns:
        Array reshaped/transposed to match out_spec axis order
        
    Raises:
        ValueError: If input has extra non-singleton axes or size mismatch
    """
    in_pos = _axis_pos(in_spec)
    out_keys = out_spec.keys
    out_key_set = set(out_keys)
    
    # Reject extra nontrivial axes
    for i, ax in enumerate(in_spec.axes):
        if ax.key not in out_key_set and arr.shape[i] != 1:
            raise ValueError(f"align_array_to: input axis {ax.key} not in output but size != 1")
    
    # Transpose input so kept axes appear in out order
    take_axes: List[int] = []
    take_keys: List[AxisKey] = []
    for k in out_keys:
        if k in in_pos:
            take_axes.append(in_pos[k])
            take_keys.append(k)
    
    if take_axes:
        # Keep axes first in correct order; trailing extra size-1 axes follow
        trailing = [i for i in range(arr.ndim) if i not in take_axes]
        arr_t = np.transpose(arr, axes=take_axes + trailing)
        kept_ndim = len(take_axes)
        # Drop trailing size-1 axes safely
        arr_t = arr_t.reshape(arr_t.shape[:kept_ndim])
    else:
        # Scalar-ish contribution
        arr_t = np.array(arr).reshape(())
    
    # Insert singleton dims for missing axes to match out order
    reshape_shape: List[int] = []
    kept_cursor = 0
    for out_ax in out_spec.axes:
        if out_ax.key in in_pos:
            if arr_t.ndim == 0:
                raise ValueError("align_array_to: cannot match axes from scalar")
            if arr_t.shape[kept_cursor] != out_ax.size:
                raise ValueError(f"align_array_to: size mismatch for axis {out_ax.key}")
            reshape_shape.append(out_ax.size)
            kept_cursor += 1
        else:
            reshape_shape.append(1)
    
    return arr_t.reshape(tuple(reshape_shape))
