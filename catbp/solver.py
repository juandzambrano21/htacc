"""
catbp/solver.py

High-level solver interface for belief propagation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from catbp.compiler.topology_compiler import TopologyCompiler
from catbp.compiler.emit import emit_upward_program, emit_downward_program_prefixsuffix, emit_finalize_program
from catbp.runtime.schedule import run_program, finalize_Z_from_slots
from catbp.vm.semiring import vm_prob_semiring, vm_logprob_semiring, vm_sat_semiring
from catbp.vm.kernels_selector import vm_load_selector


@dataclass
class SolverResult:
    """Result from running the solver."""
    Z: float
    slots: Dict
    plan: object


def run_hatcc(
    var_domains_named: Dict[str, int],
    factors_named: Dict[str, Tuple[Tuple[str, ...], np.ndarray]],
    semiring_name: str = "prob",
    support_predicate=None,
    root_factor_name: Optional[str] = None
) -> SolverResult:
    """
    Run the HATCC solver on a factor graph.
    
    Args:
        var_domains_named: Map from variable name to domain size
        factors_named: Map from factor name to (scope_names, data)
        semiring_name: "prob", "logprob", or "sat"
        support_predicate: Function to compute support (default: != 0)
        root_factor_name: Optional root factor name
        
    Returns:
        SolverResult with Z, slots, and plan
    """
    if support_predicate is None:
        support_predicate = lambda x: (x != 0)
    
    # Select semiring
    sr_map = {
        "prob": vm_prob_semiring(np.float64),
        "logprob": vm_logprob_semiring(np.float64),
        "sat": vm_sat_semiring(),
    }
    sr = sr_map[semiring_name]
    
    # Compile
    compiler = TopologyCompiler()
    plan, factor_store, selector_store, domains, mode_sizes = compiler.compile(
        var_domains_named=var_domains_named,
        factors_named=factors_named,
        support_predicate=support_predicate,
        root_factor_name=root_factor_name
    )
    
    # Build providers
    def factor_provider(fid: int):
        arr, spec = factor_store[fid]
        return arr.astype(sr.dtype, copy=False), spec
    
    def selector_provider(sid: int):
        payload = selector_store[sid]
        one_val = np.asarray(sr.one(())).reshape(()).item()
        zero_val = np.asarray(sr.zero(())).reshape(()).item()
        return vm_load_selector(
            payload,
            domains,
            mode_sizes,
            dtype=sr.dtype,
            one_val=one_val,
            zero_val=zero_val,
        )
    
    # Emit programs
    prog_up = emit_upward_program(plan)
    prog_down = emit_downward_program_prefixsuffix(plan)
    prog_fin = emit_finalize_program(plan.root)
    
    # Run
    slots = {}
    slots = run_program(prog_up, sr, factor_provider, selector_provider, slots=slots)
    slots = run_program(prog_down, sr, factor_provider, selector_provider, slots=slots)
    slots = run_program(prog_fin, sr, factor_provider, selector_provider, slots=slots)
    
    Z = finalize_Z_from_slots(slots)
    
    return SolverResult(Z=Z, slots=slots, plan=plan)
