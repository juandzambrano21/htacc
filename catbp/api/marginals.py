"""
catbp/api/marginals.py

Marginal computation from beliefs.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from catbp.ir.schema import AxisKey, AxisType, TensorSpec
from catbp.vm.kernels import vm_eliminate_keep
from catbp.vm.semiring import VMSemiringRuntime


def marginal_from_factor_belief(
    belief_data: np.ndarray,
    belief_spec: TensorSpec,
    keep_geo_vars: Tuple[int, ...],
    sr: VMSemiringRuntime
) -> np.ndarray:
    """
    Compute marginal from a factor belief.
    
    Args:
        belief_data: Belief tensor
        belief_spec: Belief specification
        keep_geo_vars: Geometric variables to keep
        sr: Semiring runtime
        
    Returns:
        Marginal tensor
    """
    keep_keys: Tuple[AxisKey, ...] = tuple((AxisType.GEOMETRIC, v) for v in keep_geo_vars)
    
    # Eliminate everything except requested geometric vars
    out, out_spec = vm_eliminate_keep(belief_data, belief_spec, keep_keys, sr)
    
    # Normalize if probabilistic
    if sr.supports_normalize() and out.ndim > 0:
        out = sr.normalize(out, axis=tuple(range(out.ndim)))
    return out


def marginal_variable(slots, plan, var_id: int, sr: VMSemiringRuntime) -> np.ndarray:
    """
    Compute marginal for a single variable.
    
    Args:
        slots: Slot store with beliefs
        plan: Execution plan
        var_id: Variable ID
        sr: Semiring runtime
        
    Returns:
        Marginal distribution for the variable
    """
    # Find a factor containing this variable
    chosen = None
    for fid, spec in plan.belief_spec.items():
        geo_vars = [ax.id for ax in spec.axes if ax.kind == AxisType.GEOMETRIC]
        if var_id in geo_vars:
            chosen = fid
            break
    
    if chosen is None:
        raise ValueError(f"Variable {var_id} not present in any factor belief")
    
    bel_data, bel_spec = slots.get(("bel", chosen))
    return marginal_from_factor_belief(bel_data, bel_spec, (var_id,), sr)
