"""
catbp/compiler/factors.py

Factor definition and provider construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from catbp.ir.schema import TensorSpec, geometric_spec

VarID = int


@dataclass(frozen=True)
class FactorDef:
    """
    Definition of a geometric factor.
    
    Attributes:
        scope: Ordered tuple of variable IDs
        table: Dense tensor in the same order as scope
    """
    scope: Tuple[VarID, ...]
    table: np.ndarray


def make_factor_provider(
    factors: Dict[int, FactorDef],
    *,
    domains: Dict[VarID, int],
    dtype: type = np.float32,
):
    """
    Create a factor provider function for the VM.
    
    Args:
        factors: Map from factor_id to FactorDef
        domains: Variable domain sizes
        dtype: Output dtype
        
    Returns:
        Function factor_provider(factor_id) -> (array, TensorSpec)
    """
    cache: Dict[int, Tuple[np.ndarray, TensorSpec]] = {}
    
    def provider(factor_id: int) -> Tuple[np.ndarray, TensorSpec]:
        if factor_id in cache:
            return cache[factor_id]
        
        fd = factors[factor_id]
        spec = geometric_spec(fd.scope, domains)
        arr = np.asarray(fd.table, dtype=dtype)
        
        if arr.shape != spec.shape:
            raise ValueError(f"factor {factor_id}: table shape {arr.shape} != spec {spec.shape}")
        
        cache[factor_id] = (arr, spec)
        return arr, spec
    
    return provider
