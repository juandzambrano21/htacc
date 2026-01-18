"""
catbp/compiler/transport.py

Transport kernel construction for holonomy computation.

The transport kernel K^{U->V} maps interface states through a factor:
    K(u, v) = 1 iff exists z: support(u, v, z) = True
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp

from catbp.algebra.semiring import SATSemiring
from catbp.tensor.named_tensor import NamedTensor, sum_out, canonical


def tensor_support(value_tensor: NamedTensor) -> NamedTensor:
    """
    Extract boolean support from a value tensor.
    
    support(x) = 1 iff value(x) != 0
    """
    sr = value_tensor.semiring
    if hasattr(sr, 'is_zero'):
        data = np.vectorize(lambda x: not sr.is_zero(x), otypes=[bool])(value_tensor.data)
    else:
        data = (value_tensor.data != 0)
    return NamedTensor(value_tensor.vars, data, SATSemiring())


def ravel_multi(state_cols: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Convert multi-index columns to flat indices.
    
    Args:
        state_cols: Array of shape (N, len(shape)) with multi-indices
        shape: Shape for raveling
        
    Returns:
        Array of flat indices
    """
    return np.ravel_multi_index(state_cols.T, shape)


def build_transport_kernel_boolean(
    factor_support: np.ndarray,
    S_vars: Tuple[int, ...],
    U_vars: Tuple[int, ...],
    V_vars: Tuple[int, ...],
    domains: Dict[int, int]
) -> sp.csr_matrix:
    """
    Build boolean transport kernel K^{U->V} from factor support.
    
    K(u, v) = EXISTS z: support(u, v, z) is True.
    Overlap consistency is automatic because overlap vars appear once in (U∪V) projection.
    
    Args:
        factor_support: Boolean tensor on scope S_vars (canonical order)
        S_vars: Factor scope variables
        U_vars: Input interface variables
        V_vars: Output interface variables
        domains: Variable domain sizes
        
    Returns:
        Sparse CSR matrix K of shape (|D_U|, |D_V|)
    """
    S_pos = {v: i for i, v in enumerate(S_vars)}
    
    # Union vars in canonical order
    UV = tuple(sorted(set(U_vars) | set(V_vars)))
    UV_pos = [S_pos[v] for v in UV]
    Z = tuple(v for v in S_vars if v not in set(UV))
    
    # Project support from S to UV by OR over Z (existential)
    perm = UV_pos + [S_pos[v] for v in Z]
    sup_perm = np.transpose(factor_support, axes=perm) if perm else factor_support
    
    UV_shape = tuple(domains[v] for v in UV)
    Z_shape = tuple(domains[v] for v in Z)
    sup_perm = sup_perm.reshape(UV_shape + Z_shape)
    
    if Z_shape:
        # OR over internal axes
        proj = np.any(sup_perm, axis=tuple(range(len(UV_shape), len(UV_shape) + len(Z_shape))))
    else:
        proj = sup_perm.astype(bool)
    
    # Extract valid UV assignments
    valid_uv = np.argwhere(proj)  # (N, len(UV))
    if valid_uv.size == 0:
        rows = int(np.prod([domains[v] for v in U_vars], dtype=np.int64))
        cols = int(np.prod([domains[v] for v in V_vars], dtype=np.int64))
        return sp.csr_matrix((rows, cols), dtype=np.bool_)
    
    u_map = [UV.index(v) for v in U_vars]
    v_map = [UV.index(v) for v in V_vars]
    
    U_shape = tuple(domains[v] for v in U_vars)
    V_shape = tuple(domains[v] for v in V_vars)
    
    u_states = valid_uv[:, u_map]
    v_states = valid_uv[:, v_map]
    
    row = ravel_multi(u_states, U_shape)
    col = ravel_multi(v_states, V_shape)
    
    rows = int(np.prod(U_shape, dtype=np.int64))
    cols = int(np.prod(V_shape, dtype=np.int64))
    
    data = np.ones(len(row), dtype=np.bool_)
    K = sp.coo_matrix((data, (row, col)), shape=(rows, cols)).tocsr()
    K.data[:] = True
    K.eliminate_zeros()
    return K


def build_transport_kernel_support(
    support_tensor: NamedTensor,
    interface_in: Tuple[int, ...],
    interface_out: Tuple[int, ...],
    domains: Dict[int, int]
) -> sp.csr_matrix:
    """
    Build transport kernel from a NamedTensor support.
    
    K(u, v) = 1 iff exists z: support(u, v, z) = True.
    
    Args:
        support_tensor: Boolean tensor on factor scope
        interface_in: Input interface U (VarIDs)
        interface_out: Output interface V (VarIDs)
        domains: Variable domain sizes
        
    Returns:
        Sparse CSR matrix K
    """
    U = tuple(interface_in)
    V = tuple(interface_out)
    
    target = canonical(set(U) | set(V))
    internal = set(support_tensor.vars) - set(target)
    
    joint = sum_out(support_tensor, internal) if internal else support_tensor
    
    # Enumerate True assignments
    valid = np.argwhere(joint.data)
    if valid.size == 0:
        rows = int(np.prod([domains[x] for x in U], dtype=np.int64))
        cols = int(np.prod([domains[x] for x in V], dtype=np.int64))
        return sp.csr_matrix((rows, cols), dtype=np.bool_)
    
    union_vars = joint.vars
    u_pos = [union_vars.index(x) for x in U]
    v_pos = [union_vars.index(x) for x in V]
    
    u_shape = tuple(domains[x] for x in U)
    v_shape = tuple(domains[x] for x in V)
    
    u_states = valid[:, u_pos]
    v_states = valid[:, v_pos]
    
    r = np.ravel_multi_index(u_states.T, u_shape)
    c = np.ravel_multi_index(v_states.T, v_shape)
    
    rows = int(np.prod(u_shape, dtype=np.int64))
    cols = int(np.prod(v_shape, dtype=np.int64))
    data = np.ones(len(r), dtype=np.bool_)
    
    K = sp.coo_matrix((data, (r, c)), shape=(rows, cols)).tocsr()
    K.sum_duplicates()
    K.data[:] = True
    return K
