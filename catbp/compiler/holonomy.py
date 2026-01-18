"""
catbp/compiler/holonomy.py

Holonomy computation for mode detection.

The holonomy H_e is the composed transport kernel around a fundamental cycle.
Modes are SCCs of the holonomy digraph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from catbp.compiler.transport import build_transport_kernel_boolean


@dataclass(frozen=True)
class HolonomyModeSpec:
    """
    Specification of modes from holonomy analysis.
    
    Attributes:
        chord: (u, v) factor endpoints
        chord_interface: J_e variable IDs
        num_modes: Number of modes
        mapping: flat_state_idx -> mode_id
        interface_shape: Shape of J_e domains
    """
    chord: Tuple[str, str]
    chord_interface: Tuple[int, ...]
    num_modes: int
    mapping: np.ndarray
    interface_shape: Tuple[int, ...]


def bool_matmul(A: sp.csr_matrix, B: sp.csr_matrix) -> sp.csr_matrix:
    """
    Boolean matrix multiplication with binarization.
    """
    C = A @ B
    if C.nnz:
        C.data[:] = True
        C.eliminate_zeros()
    return C


def interface_size(vars_: Tuple[int, ...], domains: Dict[int, int]) -> int:
    """Compute product of domain sizes."""
    out = 1
    for v in vars_:
        out *= domains[v]
    return int(out)


def compute_holonomy_boolean(
    steps: List[Tuple[int, Tuple[int, ...], Tuple[int, ...]]],
    factor_supports: Dict[int, Tuple[Tuple[int, ...], np.ndarray]],
    domains: Dict[int, int]
) -> sp.csr_matrix:
    """
    Compute boolean holonomy matrix for a cycle.
    
    Args:
        steps: List of (factor_id, in_interface, out_interface)
        factor_supports: Map from factor_id to (scope_vars, bool_tensor)
        domains: Variable domain sizes
        
    Returns:
        Holonomy adjacency H: D_J × D_J (boolean)
    """
    # Start interface is steps[0].in, which equals chord interface J
    J = steps[0][1]
    dimJ = interface_size(J, domains)
    
    # Accumulated mapping M: start(J) -> current interface
    M = sp.identity(dimJ, dtype=np.bool_, format="csr")
    
    curr_iface = J
    for fac, U, V in steps:
        assert U == curr_iface
        S_vars, sup = factor_supports[fac]
        K = build_transport_kernel_boolean(
            factor_support=sup,
            S_vars=S_vars,
            U_vars=U,
            V_vars=V,
            domains=domains
        )
        # Boolean composition
        M = (M @ K).astype(np.bool_)
        M.data[:] = True
        M.eliminate_zeros()
        curr_iface = V
    
    # By construction last V == J, so M is square dimJ×dimJ = holonomy
    assert curr_iface == J
    return M


def quotient_scc_from_holonomy(H: sp.csr_matrix) -> Tuple[int, np.ndarray]:
    """
    Compute SCC quotient from holonomy matrix.
    
    Args:
        H: Boolean holonomy CSR matrix
        
    Returns:
        (num_components, labels) where labels[state_idx] -> mode_id
    """
    n, labels = connected_components(H, directed=True, connection="strong")
    return int(n), labels.astype(np.int32)


def analyze_cycle_support(
    cycle,  # HolonomyCycle
    factor_value_tensors: Dict[str, "NamedTensor"],
    factor_scopes_vid: Dict[str, Tuple[int, ...]],
    domains: Dict[int, int],
) -> HolonomyModeSpec:
    """
    Analyze a holonomy cycle to extract mode specification.
    
    Args:
        cycle: HolonomyCycle with typed steps
        factor_value_tensors: Factor tensors (for support extraction)
        factor_scopes_vid: Factor scopes as VarIDs
        domains: Variable domain sizes
        
    Returns:
        HolonomyModeSpec with mode mapping
    """
    from catbp.compiler.transport import tensor_support, build_transport_kernel_support
    
    J = cycle.chord_interface
    J_shape = tuple(domains[v] for v in J)
    dimJ = int(np.prod(J_shape, dtype=np.int64))
    
    H = sp.identity(dimJ, dtype=np.bool_, format="csr")
    curr_I = J
    
    for step in cycle.steps:
        f = step.factor
        in_I = step.in_interface
        out_I = step.out_interface
        
        U = in_I
        V = out_I
        
        # Support tensor on factor scope
        supp = tensor_support(factor_value_tensors[f])
        
        K = build_transport_kernel_support(
            support_tensor=supp,
            interface_in=U,
            interface_out=V,
            domains=domains
        )
        H = bool_matmul(H, K)
        curr_I = V
    
    # SCC on directed graph
    n_components, labels = connected_components(H, directed=True, connection="strong")
    
    return HolonomyModeSpec(
        chord=cycle.chord,
        chord_interface=J,
        num_modes=int(n_components),
        mapping=labels.astype(np.int32),
        interface_shape=J_shape
    )
