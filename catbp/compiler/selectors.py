"""
catbp/compiler/selectors.py

Selector definition and provider construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from catbp.ir.schema import TensorSpec
from catbp.vm.kernels_selector import SelectorPayload, vm_load_selector

VarID = int
ModeID = int


@dataclass(frozen=True)
class SelectorDef:
    """
    Definition of a mode selector factor.
    
    Attributes:
        mode_axis_id: ID of the mode axis
        interface_vars: Ordered tuple J_e of variable IDs
        quotient_map: Mapping from flat index to mode label
        H_diag_mask: Optional fixed-point feasibility mask
    """
    mode_axis_id: ModeID
    interface_vars: Tuple[VarID, ...]
    quotient_map: np.ndarray
    H_diag_mask: Optional[np.ndarray] = None


def build_diag_mask_from_H(H: np.ndarray) -> np.ndarray:
    """
    Build diagonal feasibility mask from holonomy matrix.
    
    Patch E: allow only fixed-point-feasible states u with H[u,u] = 1.
    """
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")
    return np.asarray(np.diag(H)).astype(bool, copy=False)


def make_selector_provider(
    selectors: Dict[int, SelectorDef],
    *,
    domains: Dict[VarID, int],
    mode_sizes: Dict[ModeID, int],
    dtype: type = np.float32,
    one_val=None,
    zero_val=None,
):
    """
    Create a selector provider function for the VM.
    
    Args:
        selectors: Map from selector_id to SelectorDef
        domains: Variable domain sizes
        mode_sizes: Mode axis sizes
        dtype: Output dtype
        one_val: Value for allowed entries (semiring one)
        zero_val: Value for disallowed entries (semiring zero)
        
    Returns:
        Function selector_provider(selector_id) -> (array, TensorSpec)
    """
    cache: Dict[int, Tuple[np.ndarray, TensorSpec]] = {}
    
    def provider(selector_id: int) -> Tuple[np.ndarray, TensorSpec]:
        if selector_id in cache:
            return cache[selector_id]
        
        sd = selectors[selector_id]
        Q = mode_sizes[sd.mode_axis_id]
        
        payload = SelectorPayload(
            mode_axis_id=sd.mode_axis_id,
            interface_axis_ids=sd.interface_vars,
            quotient_map=np.asarray(sd.quotient_map, dtype=np.int64),
            num_modes=Q,
            diag_mask=(np.asarray(sd.H_diag_mask, dtype=bool) if sd.H_diag_mask is not None else None),
        )
        
        arr, spec = vm_load_selector(
            payload,
            domains=domains,
            mode_sizes=mode_sizes,
            dtype=dtype,
            one_val=one_val,
            zero_val=zero_val,
        )
        cache[selector_id] = (arr, spec)
        return arr, spec
    
    return provider
