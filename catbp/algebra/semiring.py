"""
catbp/algebra/semiring.py

Semiring abstractions for SAT, counting, probability, and log-probability computations.

A commutative semiring (S, ⊕, ⊗, 0, 1) provides:
- add (⊕): semiring addition
- mul (⊗): semiring multiplication  
- zero (0): additive identity
- one (1): multiplicative identity
- is_zero(x): check if x equals zero (for support computation)

We provide both Protocol-based semirings for scalar operations and
SemiringRuntime for vectorized numpy operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Tuple, Union
import numpy as np


def _logsumexp(x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
    """Numerically stable logsumexp."""
    if x.size == 0:
        return np.array(-np.inf)
    
    if axis is None:
        m = np.max(x)
        if np.isneginf(m):
            return np.array(-np.inf)
        return m + np.log(np.sum(np.exp(x - m)))
    
    # Handle tuple of axes
    if isinstance(axis, int):
        axis = (axis,)
    
    m = np.max(x, axis=axis, keepdims=True)
    # Guard against -inf
    m_safe = np.where(np.isneginf(m), 0.0, m)
    y = np.log(np.sum(np.exp(x - m_safe), axis=axis, keepdims=True)) + m_safe
    # Handle all-inf case
    y = np.where(np.isneginf(m), -np.inf, y)
    return np.squeeze(y, axis=axis)


def _axis_tuple(axis: Optional[Union[int, Tuple[int, ...]]]) -> Optional[Tuple[int, ...]]:
    if axis is None:
        return None
    if isinstance(axis, int):
        return (axis,)
    return tuple(axis)


class Semiring(Protocol):
    """Protocol for scalar semiring operations."""
    zero: Any
    one: Any

    def add(self, a: Any, b: Any) -> Any: ...
    def mul(self, a: Any, b: Any) -> Any: ...
    def is_zero(self, a: Any) -> bool: ...
    def add_reduce(self, x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray: ...
    def mul_reduce(self, xs: list[np.ndarray]) -> np.ndarray: ...
    def normalize(self, x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray: ...


@dataclass(frozen=True)
class SATSemiring:
    """Boolean support semiring: add=OR, mul=AND."""
    zero: bool = False
    one: bool = True

    def add(self, a: Any, b: Any) -> bool:
        return bool(a) or bool(b)

    def mul(self, a: Any, b: Any) -> bool:
        return bool(a) and bool(b)

    def is_zero(self, a: Any) -> bool:
        return not bool(a)

    def add_reduce(self, x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
        return np.any(x, axis=axis)

    def mul_reduce(self, xs: list[np.ndarray]) -> np.ndarray:
        if not xs:
            raise ValueError("mul_reduce requires at least one array")
        out = xs[0].copy()
        for t in xs[1:]:
            out = np.logical_and(out, t)
        return out

    def normalize(self, x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
        return x  # identity for boolean


@dataclass(frozen=True)
class ProbSemiring:
    """Nonnegative reals: add=+, mul=*."""
    zero: float = 0.0
    one: float = 1.0

    def add(self, a: Any, b: Any) -> float:
        return float(a) + float(b)

    def mul(self, a: Any, b: Any) -> float:
        return float(a) * float(b)

    def is_zero(self, a: Any) -> bool:
        return float(a) == 0.0

    def add_reduce(self, x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
        return np.sum(x, axis=axis)

    def mul_reduce(self, xs: list[np.ndarray]) -> np.ndarray:
        if not xs:
            raise ValueError("mul_reduce requires at least one array")
        out = xs[0].copy()
        for t in xs[1:]:
            out = out * t
        return out

    def normalize(self, x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
        s = np.sum(x, axis=axis, keepdims=True)
        s = np.where(s == 0.0, 1.0, s)
        return x / s


@dataclass(frozen=True)
class LogProbSemiring:
    """Log-space probabilities: add=logsumexp, mul=+."""
    zero: float = -np.inf
    one: float = 0.0

    def add(self, a: Any, b: Any) -> float:
        return float(np.logaddexp(a, b))

    def mul(self, a: Any, b: Any) -> float:
        return float(a) + float(b)

    def is_zero(self, a: Any) -> bool:
        return np.isneginf(float(a))

    def add_reduce(self, x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
        return _logsumexp(x, axis=axis)

    def mul_reduce(self, xs: list[np.ndarray]) -> np.ndarray:
        if not xs:
            raise ValueError("mul_reduce requires at least one array")
        out = xs[0].copy()
        for t in xs[1:]:
            out = out + t
        return out

    def normalize(self, x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
        ax = _axis_tuple(axis)
        if ax is None:
            z = _logsumexp(x, axis=None)
        else:
            z = _logsumexp(x, axis=ax)
            for a in sorted(ax):
                z = np.expand_dims(z, axis=a)
        return x - z


@dataclass(frozen=True)
class CountingSemiring:
    """Integer counting semiring: add=+, mul=*."""
    zero: int = 0
    one: int = 1

    def add(self, a: Any, b: Any) -> int:
        return int(a) + int(b)

    def mul(self, a: Any, b: Any) -> int:
        return int(a) * int(b)

    def is_zero(self, a: Any) -> bool:
        return int(a) == 0

    def add_reduce(self, x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
        return np.sum(x, axis=axis)

    def mul_reduce(self, xs: list[np.ndarray]) -> np.ndarray:
        if not xs:
            raise ValueError("mul_reduce requires at least one array")
        out = xs[0].copy()
        for t in xs[1:]:
            out = out * t
        return out

    def normalize(self, x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
        return x  # no normalization for counting


@dataclass(frozen=True)
class SemiringRuntime:
    """
    Numpy-backed semiring runtime for vectorized operations.
    
    Attributes:
        name: Identifier for the semiring type
        dtype: Numpy dtype for tensors
        add: Elementwise ⊕ operation
        mul: Elementwise ⊗ operation
        add_reduce: ⊕ reduction over axes
        one: Factory for multiplicative identity tensor
        zero: Factory for additive identity tensor
        normalize: Optional normalization function
    """
    name: str
    dtype: np.dtype
    add: Callable[[np.ndarray, np.ndarray], np.ndarray]
    mul: Callable[[np.ndarray, np.ndarray], np.ndarray]
    add_reduce: Callable[[np.ndarray, Optional[Tuple[int, ...]]], np.ndarray]
    one: Callable[[Tuple[int, ...]], np.ndarray]
    zero: Callable[[Tuple[int, ...]], np.ndarray]
    normalize: Optional[Callable[[np.ndarray, Optional[Tuple[int, ...]]], np.ndarray]] = None

    def supports_normalize(self) -> bool:
        return self.normalize is not None

    def maybe_normalize(self, x: np.ndarray, axes: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        if self.normalize is None:
            return x
        return self.normalize(x, axes)


def sat_semiring() -> SemiringRuntime:
    """Create a Boolean SAT semiring runtime."""
    return SemiringRuntime(
        name="SAT",
        dtype=np.dtype(np.bool_),
        add=np.logical_or,
        mul=np.logical_and,
        add_reduce=lambda x, axis: np.any(x, axis=axis) if axis is not None else x,
        one=lambda shape: np.ones(shape, dtype=np.bool_),
        zero=lambda shape: np.zeros(shape, dtype=np.bool_),
        normalize=None,
    )


def prob_semiring(dtype: type = np.float64) -> SemiringRuntime:
    """Create a probability semiring runtime."""
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
    
    return SemiringRuntime(
        name="PROB",
        dtype=dt,
        add=np.add,
        mul=np.multiply,
        add_reduce=lambda x, axis: np.sum(x, axis=axis, dtype=dt) if axis is not None else x,
        one=lambda shape: np.ones(shape, dtype=dt),
        zero=lambda shape: np.zeros(shape, dtype=dt),
        normalize=_normalize,
    )


def logprob_semiring(dtype: type = np.float64) -> SemiringRuntime:
    """Create a log-probability semiring runtime."""
    dt = np.dtype(dtype)
    
    def _add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.logaddexp(x, y)
    
    def _add_reduce(x: np.ndarray, axis: Optional[Tuple[int, ...]]) -> np.ndarray:
        if axis is None:
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
    
    return SemiringRuntime(
        name="LOGPROB",
        dtype=dt,
        add=_add,
        mul=np.add,  # log(a*b) = log(a) + log(b)
        add_reduce=_add_reduce,
        one=lambda shape: np.zeros(shape, dtype=dt),  # log(1) = 0
        zero=lambda shape: np.full(shape, -np.inf, dtype=dt),  # log(0) = -inf
        normalize=_normalize,
    )


def counting_semiring(dtype: type = np.int64) -> SemiringRuntime:
    """Create a counting semiring runtime."""
    dt = np.dtype(dtype)
    
    return SemiringRuntime(
        name="COUNT",
        dtype=dt,
        add=np.add,
        mul=np.multiply,
        add_reduce=lambda x, axis: np.sum(x, axis=axis, dtype=dt) if axis is not None else x,
        one=lambda shape: np.ones(shape, dtype=dt),
        zero=lambda shape: np.zeros(shape, dtype=dt),
        normalize=None,
    )
