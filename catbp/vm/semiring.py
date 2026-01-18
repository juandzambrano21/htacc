"""
catbp/vm/semiring.py

Semiring runtime interface for the VM.

Provides vectorized semiring operations for tensor computations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np


def _logsumexp(x: np.ndarray, axis: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """Numerically stable logsumexp."""
    if x.size == 0:
        return np.array(-np.inf)
    
    if axis is None:
        m = np.max(x)
        if np.isneginf(m):
            return np.array(-np.inf)
        return m + np.log(np.sum(np.exp(x - m)))
    
    m = np.max(x, axis=axis, keepdims=True)
    m_safe = np.where(np.isneginf(m), 0.0, m)
    y = np.log(np.sum(np.exp(x - m_safe), axis=axis, keepdims=True)) + m_safe
    y = np.where(np.isneginf(m), -np.inf, y)
    return np.squeeze(y, axis=axis)


def _axis_tuple(axis: Optional[Union[int, Tuple[int, ...]]]) -> Optional[Tuple[int, ...]]:
    if axis is None:
        return None
    if isinstance(axis, int):
        return (axis,)
    return tuple(axis)


@dataclass(frozen=True)
class VMSemiringRuntime:
    """
    Vectorized semiring backend for VM operations.
    
    Attributes:
        name: Identifier for the semiring type
        dtype: Numpy dtype for tensors
        mul: Elementwise ⊗ operation
        add_reduce: ⊕ reduction over axes
        one: Factory for multiplicative identity tensor
        zero: Factory for additive identity tensor
        normalize: Optional normalization function
    """
    name: str
    dtype: np.dtype
    mul: Callable[[np.ndarray, np.ndarray], np.ndarray]
    add_reduce: Callable[[np.ndarray, Tuple[int, ...]], np.ndarray]
    one: Callable[[Tuple[int, ...]], np.ndarray]
    zero: Callable[[Tuple[int, ...]], np.ndarray]
    normalize: Optional[Callable[[np.ndarray, Optional[Tuple[int, ...]]], np.ndarray]] = None

    def supports_normalize(self) -> bool:
        """Check if this semiring supports normalization."""
        return self.normalize is not None


def vm_sat_semiring() -> VMSemiringRuntime:
    """Create a Boolean SAT semiring runtime for VM."""
    return VMSemiringRuntime(
        name="SAT",
        dtype=np.dtype(np.bool_),
        mul=np.logical_and,
        add_reduce=lambda x, axis: np.any(x, axis=axis) if axis else x,
        one=lambda shape: np.ones(shape, dtype=np.bool_),
        zero=lambda shape: np.zeros(shape, dtype=np.bool_),
        normalize=None,
    )


def vm_prob_semiring(dtype: type = np.float64) -> VMSemiringRuntime:
    """Create a probability semiring runtime for VM."""
    dt = np.dtype(dtype)
    
    def _normalize(x: np.ndarray, axis: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        if axis is None:
            s = np.sum(x)
            if s == 0:
                return x
            return x / s
        else:
            s = np.sum(x, axis=axis, keepdims=True)
            s = np.where(s == 0, 1.0, s)
            return x / s
    
    return VMSemiringRuntime(
        name="PROB",
        dtype=dt,
        mul=np.multiply,
        add_reduce=lambda x, axis: np.sum(x, axis=axis, dtype=dt) if axis else x,
        one=lambda shape: np.ones(shape, dtype=dt),
        zero=lambda shape: np.zeros(shape, dtype=dt),
        normalize=_normalize,
    )


def vm_logprob_semiring(dtype: type = np.float64) -> VMSemiringRuntime:
    """Create a log-probability semiring runtime for VM."""
    dt = np.dtype(dtype)
    
    def _add_reduce(x: np.ndarray, axis: Tuple[int, ...]) -> np.ndarray:
        if not axis:
            return x
        return _logsumexp(x, axis=axis)
    
    def _normalize(x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
        ax = _axis_tuple(axis)
        if ax is None:
            z = _logsumexp(x, axis=tuple(range(x.ndim)) if x.ndim else None)
        else:
            z = _logsumexp(x, axis=ax)
            # Expand dims for broadcasting
            for a in sorted(ax):
                z = np.expand_dims(z, axis=a)
        return x - z
    
    return VMSemiringRuntime(
        name="LOGPROB",
        dtype=dt,
        mul=np.add,  # log(a*b) = log(a) + log(b)
        add_reduce=_add_reduce,
        one=lambda shape: np.zeros(shape, dtype=dt),  # log(1) = 0
        zero=lambda shape: np.full(shape, -np.inf, dtype=dt),  # log(0) = -inf
        normalize=_normalize,
    )


def vm_counting_semiring(dtype: type = np.int64) -> VMSemiringRuntime:
    """Create a counting semiring runtime for VM."""
    dt = np.dtype(dtype)
    
    return VMSemiringRuntime(
        name="COUNT",
        dtype=dt,
        mul=np.multiply,
        add_reduce=lambda x, axis: np.sum(x, axis=axis, dtype=dt) if axis else x,
        one=lambda shape: np.ones(shape, dtype=dt),
        zero=lambda shape: np.zeros(shape, dtype=dt),
        normalize=None,
    )
