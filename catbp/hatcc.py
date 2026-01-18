"""
catbp/hatcc.py

HATCC (Holonomy-Aware Tree-Compiled Computation) solver.

This is the main entry point for the categorical belief propagation system.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from catbp.solver import run_hatcc, SolverResult


def hatcc_solve(
    var_domains: Dict[str, int],
    factors: Dict[str, Tuple[Tuple[str, ...], np.ndarray]],
    *,
    semiring: str = "prob",
    root: Optional[str] = None,
    support_eps: float = 0.0,
) -> SolverResult:
    """
    Solve a factor graph using HATCC.
    
    Args:
        var_domains: Map from variable name to domain size
        factors: Map from factor name to (scope, tensor)
        semiring: "prob", "logprob", or "sat"
        root: Optional root factor name
        support_eps: Epsilon for support computation
        
    Returns:
        SolverResult with partition function and beliefs
        
    Example:
        >>> var_domains = {"A": 2, "B": 2}
        >>> factors = {
        ...     "f1": (("A",), np.array([0.3, 0.7])),
        ...     "f2": (("A", "B"), np.array([[0.9, 0.1], [0.2, 0.8]])),
        ... }
        >>> result = hatcc_solve(var_domains, factors)
        >>> print(f"Z = {result.Z}")
    """
    if support_eps > 0:
        support_pred = lambda x: np.abs(x) > support_eps
    else:
        support_pred = lambda x: x != 0
    
    return run_hatcc(
        var_domains_named=var_domains,
        factors_named=factors,
        semiring_name=semiring,
        support_predicate=support_pred,
        root_factor_name=root,
    )


def compute_partition_function(
    var_domains: Dict[str, int],
    factors: Dict[str, Tuple[Tuple[str, ...], np.ndarray]],
    **kwargs
) -> float:
    """
    Compute the partition function Z of a factor graph.
    
    Args:
        var_domains: Map from variable name to domain size
        factors: Map from factor name to (scope, tensor)
        **kwargs: Additional arguments passed to hatcc_solve
        
    Returns:
        Partition function Z
    """
    result = hatcc_solve(var_domains, factors, **kwargs)
    return result.Z


def compute_marginals(
    var_domains: Dict[str, int],
    factors: Dict[str, Tuple[Tuple[str, ...], np.ndarray]],
    variables: Optional[list] = None,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute marginal distributions for variables.
    
    Args:
        var_domains: Map from variable name to domain size
        factors: Map from factor name to (scope, tensor)
        variables: List of variable names (default: all)
        **kwargs: Additional arguments passed to hatcc_solve
        
    Returns:
        Map from variable name to marginal distribution
    """
    from catbp.api.marginals import marginal_from_factor_belief
    from catbp.ir.schema import AxisType
    from catbp.vm.semiring import vm_prob_semiring
    
    result = hatcc_solve(var_domains, factors, **kwargs)
    
    if variables is None:
        variables = list(var_domains.keys())
    
    # Build name -> id mapping
    var_names = sorted(var_domains.keys())
    var_id = {n: i for i, n in enumerate(var_names)}
    
    sr = vm_prob_semiring(np.float64)
    marginals = {}
    
    for var_name in variables:
        vid = var_id[var_name]
        
        # Find a factor containing this variable
        for fid, spec in result.plan.belief_spec.items():
            geo_vars = [ax.id for ax in spec.axes if ax.kind == AxisType.GEOMETRIC]
            if vid in geo_vars:
                bel_data, bel_spec = result.slots[("bel", fid)]
                marg = marginal_from_factor_belief(bel_data, bel_spec, (vid,), sr)
                marginals[var_name] = marg
                break
    
    return marginals
