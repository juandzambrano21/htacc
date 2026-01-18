"""
catbp/runtime/evidence.py

Evidence handling for factor graphs.

Evidence is injected as constraint factors, not domain slicing.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from catbp.tensor.named_tensor import NamedTensor


def unary_evidence(
    var: int,
    allowed: List[int],
    domains: Dict[int, int],
    semiring
) -> NamedTensor:
    """
    Create a unary evidence factor.
    
    Args:
        var: Variable ID
        allowed: List of allowed values
        domains: Variable domain sizes
        semiring: Semiring for the factor
        
    Returns:
        NamedTensor representing the evidence constraint
    """
    d = domains[var]
    data = np.full((d,), semiring.zero, dtype=np.float32)
    for a in allowed:
        data[a] = semiring.one
    return NamedTensor((var,), data, semiring)


def interface_mask(
    vars_: Tuple[int, ...],
    mask: np.ndarray,
    semiring
) -> NamedTensor:
    """
    Create a general k-ary constraint factor.
    
    Args:
        vars_: Variable IDs in the constraint scope
        mask: Boolean or semiring-valued tensor
        semiring: Semiring for the factor
        
    Returns:
        NamedTensor representing the constraint
    """
    data = mask.astype(np.float32)
    return NamedTensor(vars_, data, semiring)


def observe_variable(
    var: int,
    value: int,
    domains: Dict[int, int],
    semiring
) -> NamedTensor:
    """
    Create an observation constraint (single value).
    
    Args:
        var: Variable ID
        value: Observed value
        domains: Variable domain sizes
        semiring: Semiring for the factor
        
    Returns:
        NamedTensor representing the observation
    """
    return unary_evidence(var, [value], domains, semiring)
