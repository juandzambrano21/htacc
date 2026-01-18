"""
catbp/vm/kernels.py

VM kernel operations for tensor manipulation.

Core operations:
- vm_unit: Create unit tensor
- vm_contract: Multiply tensors with alignment
- vm_eliminate_keep: Marginalize to keep specified axes
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from catbp.ir.schema import AxisKey, TensorSpec
from catbp.vm.align import align_array_to
from catbp.vm.semiring import VMSemiringRuntime


def vm_unit(out_spec: TensorSpec, sr: VMSemiringRuntime) -> np.ndarray:
    """
    Create a unit tensor (all ones in the semiring).
    
    Args:
        out_spec: Specification for the output tensor
        sr: Semiring runtime
        
    Returns:
        Unit tensor with shape from out_spec
    """
    return sr.one(out_spec.shape).astype(sr.dtype, copy=False)


def vm_contract(
    inputs: List[Tuple[np.ndarray, TensorSpec]],
    out_spec: TensorSpec,
    sr: VMSemiringRuntime
) -> np.ndarray:
    """
    Contract (multiply) multiple tensors with alignment.
    
    Args:
        inputs: List of (array, spec) pairs
        out_spec: Specification for the output tensor
        sr: Semiring runtime
        
    Returns:
        Product of all input tensors, aligned to out_spec
    """
    if not inputs:
        return vm_unit(out_spec, sr)
    
    acc = None
    for arr, spec in inputs:
        # Align to output spec
        view = align_array_to(arr, spec, out_spec).astype(sr.dtype, copy=False)
        # Broadcast to full shape
        aligned = np.broadcast_to(view, out_spec.shape)
        
        if acc is None:
            acc = aligned.copy()
        else:
            acc = sr.mul(acc, aligned)
    
    return acc


def vm_eliminate_keep(
    arr: np.ndarray,
    in_spec: TensorSpec,
    keep_keys: Tuple[AxisKey, ...],
    sr: VMSemiringRuntime
) -> Tuple[np.ndarray, TensorSpec]:
    """
    ⊕-reduce out every axis not in keep_keys.
    
    Output axis order is exactly keep_keys order (filtered to those present).
    
    Args:
        arr: Input array
        in_spec: Specification of input array
        keep_keys: Axis keys to keep (in desired output order)
        sr: Semiring runtime
        
    Returns:
        (reduced_array, output_spec)
    """
    if not keep_keys:
        # Eliminate all axes -> scalar
        if arr.ndim == 0:
            return arr, TensorSpec(axes=())
        out = sr.add_reduce(arr.astype(sr.dtype, copy=False), axis=tuple(range(arr.ndim)))
        return np.asarray(out), TensorSpec(axes=())
    
    pos = in_spec.axis_pos()  # AxisKey -> index
    present = [k for k in keep_keys if k in pos]
    
    if not present:
        # Nothing kept -> scalar
        out = sr.add_reduce(arr.astype(sr.dtype, copy=False), axis=tuple(range(arr.ndim)))
        return np.asarray(out), TensorSpec(axes=())
    
    # Transpose so kept axes (in desired order) come first
    keep_axes = [pos[k] for k in present]
    drop_axes = [i for i in range(arr.ndim) if i not in set(keep_axes)]
    perm = keep_axes + drop_axes
    x = np.transpose(arr, axes=perm) if perm != list(range(arr.ndim)) else arr
    
    # Reduce over trailing dropped axes
    if drop_axes:
        red = tuple(range(len(keep_axes), x.ndim))
        x = sr.add_reduce(x.astype(sr.dtype, copy=False), axis=red)
    else:
        x = x.astype(sr.dtype, copy=False)
    
    # Build output spec in exact order of keep_keys (only those present)
    out_axes = tuple(in_spec.axes[pos[k]] for k in present)
    return x, TensorSpec(axes=out_axes)


def vm_eliminate(
    arr: np.ndarray,
    in_spec: TensorSpec,
    drop_ids: Tuple[int, ...],
    sr: VMSemiringRuntime
) -> Tuple[np.ndarray, TensorSpec]:
    """
    Eliminate axes by ID (legacy interface).
    
    Args:
        arr: Input array
        in_spec: Specification of input array
        drop_ids: Axis IDs to eliminate
        sr: Semiring runtime
        
    Returns:
        (reduced_array, output_spec)
    """
    if not drop_ids:
        return arr, in_spec
    
    drop_set = set(drop_ids)
    keep_keys = tuple(ax.key for ax in in_spec.axes if ax.id not in drop_set)
    return vm_eliminate_keep(arr, in_spec, keep_keys, sr)
