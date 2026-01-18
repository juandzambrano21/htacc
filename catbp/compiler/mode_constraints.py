"""
catbp/compiler/mode_constraints.py

Mode compatibility constraint generation.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp

from catbp.algebra.section import Section
from catbp.compiler.modes import ModeSpec


def _flat_proj_indices(
    full_vars: Tuple[int, ...],
    full_shape: Tuple[int, ...],
    proj_vars: Tuple[int, ...],
    proj_shape: Tuple[int, ...],
) -> np.ndarray:
    """
    For each flat index on D_full, compute the flat index of its restriction to D_proj.
    
    Returns array p of shape (|D_full|,) with entries in [0, |D_proj|).
    """
    if not proj_vars:
        return np.zeros((int(np.prod(full_shape, dtype=np.int64)),), dtype=np.int64)
    
    # Unravel all full indices to coordinates, then pick projected coordinates
    full_dim = int(np.prod(full_shape, dtype=np.int64))
    coords = np.stack(np.unravel_index(np.arange(full_dim, dtype=np.int64), full_shape), axis=1)
    pos = {v: i for i, v in enumerate(full_vars)}
    cols = [pos[v] for v in proj_vars]
    proj_coords = coords[:, cols]
    return np.ravel_multi_index(proj_coords.T, proj_shape).astype(np.int64, copy=False)


def build_pairwise_constraints_on_shared_interface(
    *,
    chord1: Tuple[int, ModeSpec],  # (mode_axis_id, spec)
    chord2: Tuple[int, ModeSpec],
    domains: Dict[int, int],
    dtype=np.float32,
) -> Section:
    """
    Build baseline mode-compatibility constraint on (c_e, c_f).
    
    The constraint is derived from shared variables in chord interfaces:
        S := J_e ∩ J_f
        C(c_e, c_f) = 1 iff ∃ s ∈ D_S such that
            s is realizable by some J_e state in mode c_e (and feasible),
            and by some J_f state in mode c_f (and feasible).
    
    This is the minimum coupling needed to prevent obviously inconsistent
    mode combinations when chord interfaces overlap.
    
    Args:
        chord1: (mode_axis_id, ModeSpec) for first chord
        chord2: (mode_axis_id, ModeSpec) for second chord
        domains: Variable domain sizes
        dtype: Output dtype
        
    Returns:
        Section with domain (m1, m2) and constraint tensor
    """
    from catbp.algebra.semiring import ProbSemiring
    
    m1, ms1 = chord1
    m2, ms2 = chord2
    J1 = tuple(ms1.chord_interface)
    J2 = tuple(ms2.chord_interface)
    
    S = tuple(sorted(set(J1).intersection(J2)))
    Q1 = int(ms1.num_modes)
    Q2 = int(ms2.num_modes)
    
    if not S:
        # No shared variables - all mode combinations allowed
        data = np.ones((Q1, Q2), dtype=dtype)
        return Section(domain=(m1, m2), data=data, semiring=ProbSemiring())
    
    S_shape = tuple(domains[v] for v in S)
    
    J1_shape = tuple(domains[v] for v in J1)
    J2_shape = tuple(domains[v] for v in J2)
    dim1 = int(np.prod(J1_shape, dtype=np.int64))
    dim2 = int(np.prod(J2_shape, dtype=np.int64))
    dimS = int(np.prod(S_shape, dtype=np.int64))
    
    # Mapping flat J -> mode (can include -1 for infeasible states)
    map1 = ms1.labels.astype(np.int64, copy=False)
    map2 = ms2.labels.astype(np.int64, copy=False)
    
    # Projection flat indices J -> S
    p1 = _flat_proj_indices(J1, J1_shape, S, S_shape)
    p2 = _flat_proj_indices(J2, J2_shape, S, S_shape)
    
    # Build incidence matrices A: (Q1 x dimS), B: (Q2 x dimS)
    # A[c, s] = 1 iff exists i with map1[i] = c and p1[i] = s
    ok1 = (map1 >= 0)
    ok2 = (map2 >= 0)
    rows1 = map1[ok1]
    cols1 = p1[ok1]
    rows2 = map2[ok2]
    cols2 = p2[ok2]
    
    A = sp.coo_matrix((np.ones_like(rows1, dtype=np.int8), (rows1, cols1)), shape=(Q1, dimS)).tocsr()
    B = sp.coo_matrix((np.ones_like(rows2, dtype=np.int8), (rows2, cols2)), shape=(Q2, dimS)).tocsr()
    A.data[:] = 1
    B.data[:] = 1
    
    C = (A @ B.T).astype(np.int8)
    if C.nnz:
        C.data[:] = 1
        C.eliminate_zeros()
    
    dense = (C.toarray() > 0).astype(dtype)
    return Section(domain=(m1, m2), data=dense, semiring=ProbSemiring())
