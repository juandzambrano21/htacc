"""
catbp/algebra/support.py

Compiler-time support extraction and mode mapping utilities.

Key idea (Patch D):
  - Compiler topology must be computed on Boolean/SAT semiring (structure only).
  - Runtime may be LogProb/SumProduct/etc. (values).

This module provides:
  - factor_support(phi): boolean Section 1[phi != 0] in SATSemiring
  - build_factor_supports: batch conversion for all factors
  - filter_fixed_points_for_modes: enforce one-loop closure by diagonal feasibility
  - mode_mapping_from_holonomy: compute mode quotient from holonomy matrix
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from catbp.algebra.section import Section
from catbp.algebra.semiring import SATSemiring


def factor_support(
    phi: Section,
    *,
    eps: float = 0.0,
    zero: Optional[Any] = None,
    log_space: bool = False,
    treat_nan_as_zero: bool = True
) -> Section:
    """
    Convert a factor Section φ: D_S -> S (arbitrary semiring values)
    into a Boolean support Section supp(φ): D_S -> {0,1} in SATSemiring:

        supp(x) = 1[ φ(x) != 0 ]   (or robust float/log variants)

    Args:
        phi: Input section with arbitrary semiring values.
        eps: For floats in normal space: treat |φ| > eps as nonzero.
        zero: Optional explicit zero sentinel to compare against when dtype is not float.
              If None, uses phi.semiring.zero.
        log_space: If True, interpret phi.data as log-values; "zero" mass is -inf.
                   Then support(x) = isfinite(phi(x)) and (phi(x) > -inf).
        treat_nan_as_zero: If True, NaNs are treated as zeros (excluded from support).

    Returns:
        Section(domain=phi.domain, data=bool mask, semiring=SATSemiring()).
    """
    data = phi.data

    if data.dtype == bool:
        mask = data
    elif np.issubdtype(data.dtype, np.floating):
        arr = data.copy()

        if treat_nan_as_zero:
            if log_space:
                arr = np.nan_to_num(arr, nan=-np.inf)
            else:
                arr = np.nan_to_num(arr, nan=0.0)

        if log_space:
            # In log-space, valid support is finite (not -inf).
            if eps != 0.0:
                mask = np.isfinite(arr) & (arr > eps)
            else:
                mask = np.isfinite(arr)
        else:
            # Normal float: support by magnitude threshold
            mask = np.isfinite(arr) & (np.abs(arr) > eps)
            # If +inf present (rare), treat as supported:
            mask = mask | (np.isinf(arr) & (arr > 0))
    else:
        z = phi.semiring.zero if zero is None else zero
        mask = (data != z)

    mask = mask.astype(bool, copy=False)
    return Section(phi.domain, mask, SATSemiring())


def build_factor_supports(
    factors: Mapping[str, Section],
    *,
    eps: float = 0.0,
    log_space: bool = False
) -> Dict[str, Section]:
    """
    Batch conversion: name -> boolean support Section in SATSemiring.

    This is what your TopologyEngine should consume as factor_supports.
    """
    out: Dict[str, Section] = {}
    for name, sec in factors.items():
        out[name] = factor_support(sec, eps=eps, log_space=log_space)
    return out


def binarize_csr(M: sp.csr_matrix) -> sp.csr_matrix:
    """
    Ensure CSR matrix is strictly boolean (0/1), pruned.

    Use this after each sparse composition step:
      H = binarize_csr(H @ K)
    """
    if not sp.isspmatrix_csr(M):
        M = M.tocsr()
    if M.nnz == 0:
        return M
    M.data = np.ones_like(M.data, dtype=np.int8)
    M.eliminate_zeros()
    return M


def filter_fixed_points_for_modes(
    H: sp.csr_matrix,
    scc_labels: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Enforce one-loop closure by diagonal feasibility.

    A state u in D_J is *cycle-feasible* iff H[u,u] == 1 (there exists a closed transport
    returning to itself). This avoids "reachable but not returnable to same state" artifacts.

    Args:
        H: Boolean holonomy matrix on D_J (shape N x N).
        scc_labels: SCC label for each state (length N), e.g. from connected_components(..., connection="strong").

    Returns:
        (mapping, num_modes)
            mapping: int array length N where:
                     mapping[u] = mode_id in 0..num_modes-1 for fixed-point feasible states,
                     mapping[u] = -1 for infeasible states (no diagonal support).
            num_modes: number of mode classes among feasible states.
    """
    if not sp.isspmatrix_csr(H):
        H = H.tocsr()

    diag = H.diagonal().astype(bool)
    fixed = np.where(diag)[0]
    if fixed.size == 0:
        return np.full(len(scc_labels), -1, dtype=np.int32), 0

    used_components = sorted(set(int(scc_labels[u]) for u in fixed))
    remap = {old: new for new, old in enumerate(used_components)}

    mapping = np.full(len(scc_labels), -1, dtype=np.int32)
    for u in fixed:
        mapping[u] = remap[int(scc_labels[u])]

    return mapping, len(used_components)


def mode_mapping_from_holonomy(
    H: sp.csr_matrix,
    *,
    require_fixed_point: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Compute the mode quotient map π: D_J -> Q_e from a boolean holonomy matrix H.

    Pipeline:
      1) SCC decomposition on SuppDigraph(H)  (canonical orbit partition)
      2) Optional fixed-point filtering by diagonal feasibility (recommended)

    Args:
        H: Boolean holonomy CSR matrix (N x N).
        require_fixed_point: If True, drop states u with H[u,u]==0 (one-loop closure).
                            If False, every state gets some SCC-based mode id.

    Returns:
        (mapping, num_modes)
            mapping[u] = mode_id (0..num_modes-1) or -1 if filtered out.
    """
    if not sp.isspmatrix_csr(H):
        H = H.tocsr()

    # SCCs on directed graph
    n_comp, labels = connected_components(H, directed=True, connection="strong")
    labels = labels.astype(np.int32, copy=False)

    if not require_fixed_point:
        # Compress labels to 0..k-1 deterministically
        used = sorted(set(int(x) for x in labels.tolist()))
        remap = {old: new for new, old in enumerate(used)}
        mapping = np.array([remap[int(x)] for x in labels], dtype=np.int32)
        return mapping, len(used)

    return filter_fixed_points_for_modes(H, labels)


def mapping_array_to_dict(mapping: np.ndarray) -> Dict[int, int]:
    """
    Convert mapping array (len N, entries in {0..k-1} or -1) into the dict format
    used by vm_load_selector (flat_state_idx -> mode_id), skipping -1.
    """
    out: Dict[int, int] = {}
    for i, m in enumerate(mapping.tolist()):
        if m >= 0:
            out[i] = int(m)
    return out


def holonomy_fixed_point_quotient_map(
    H: sp.csr_matrix,
    labels: np.ndarray
) -> Dict[int, int]:
    """
    Given:
      - H: boolean holonomy matrix on D_J (shape N x N)
      - labels: SCC label array length N (state -> scc_id)

    Enforce one-loop closure:
      Keep ONLY states u with H[u,u] == True.

    Then compress SCC ids used by fixed-point states into 0..k-1.
    Returns:
      dict {flat_state_index -> mode_id} for fixed-point states only.
    """
    diag = H.diagonal().astype(bool)
    fixed = np.where(diag)[0]
    if fixed.size == 0:
        return {}

    used = sorted(set(labels[fixed].tolist()))
    remap = {old: i for i, old in enumerate(used)}

    qm: Dict[int, int] = {}
    for u in fixed:
        qm[int(u)] = remap[int(labels[u])]
    return qm
