"""
catbp/algebra/section.py

A Section is a semiring-valued tensor with a *named* domain (ordered variables).
It is the concrete implementation of Γ(U) = { f: D_U -> S }.

Key operations:
  - star:   (f ⋆ g) on U∪W  (aligned pointwise multiplication)
  - restrict: ρ_{U->T}      (semiring-sum marginalization onto T)
  - unit: identity section  (all-ones tensor)
  - normalize: semiring-specific normalization (no-op for SAT)

Design constraints:
  - Domain ordering is *semantic*: axes correspond 1-1 to domain entries.
  - Determinism: star() uses canonical union ordering by default (sorted).
  - Compiler exactness: restrict(target) outputs axes in exactly target order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, TypeVar, Union
import numpy as np

Var = TypeVar("Var", bound=Any)


@dataclass(frozen=True)
class Section:
    """
    A semiring tensor over an ordered domain.

    Attributes:
        domain: Ordered variables (axis labels).
        data: ndarray shaped by the variable domain sizes in the *same order*.
        semiring: Semiring implementing add/mul/reduce/normalize on ndarrays.
    """
    domain: Tuple[Var, ...]
    data: np.ndarray
    semiring: Any  # Semiring protocol

    def __post_init__(self):
        if len(self.domain) != self.data.ndim:
            raise ValueError(
                f"Section domain rank mismatch: |domain|={len(self.domain)} "
                f"but data.ndim={self.data.ndim}"
            )
        if len(set(self.domain)) != len(self.domain):
            raise ValueError(f"Section domain has duplicates: {self.domain}")

    @staticmethod
    def unit(domain: Sequence[Var], shape: Sequence[int], semiring: Any, dtype: Optional[type] = None) -> "Section":
        """
        Unit section 1_U: constant one on D_U.
        """
        if dtype is None:
            one = semiring.one
            dtype = np.array(one).dtype
        data = np.full(tuple(shape), semiring.one, dtype=dtype)
        return Section(tuple(domain), data, semiring)

    @staticmethod
    def zero_section(domain: Sequence[Var], shape: Sequence[int], semiring: Any, dtype: Optional[type] = None) -> "Section":
        """
        Zero section 0_U: constant zero on D_U.
        """
        if dtype is None:
            z = semiring.zero
            dtype = np.array(z).dtype
        data = np.full(tuple(shape), semiring.zero, dtype=dtype)
        return Section(tuple(domain), data, semiring)

    def axis_of(self, v: Var) -> int:
        """Returns the axis index of variable v in self.domain."""
        return self.domain.index(v)

    def dim_of(self, v: Var) -> int:
        """Returns the size of variable v's axis, inferred from data shape."""
        return self.data.shape[self.axis_of(v)]

    def _aligned_view(self, target_domain: Tuple[Var, ...], target_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Returns a broadcastable (and broadcasted) ndarray aligned to target_domain.

        - Existing axes are permuted into target order.
        - Missing axes become singleton dimensions.
        - Result is broadcasted to target_shape (for pointwise ops).
        """
        src_pos = {v: i for i, v in enumerate(self.domain)}
        
        # Permute existing axes in the order they appear in target_domain
        perm = [src_pos[v] for v in target_domain if v in src_pos]

        data = self.data
        if perm:
            # If perm is not identity (in that reduced space), transpose
            if perm != list(range(len(perm))) or len(perm) != data.ndim:
                data = np.transpose(data, axes=perm)
        else:
            # scalar-ish: no vars in common, treat as rank-0 then reshape later
            data = np.asarray(data)

        # Now reshape inserting singleton axes for missing vars
        shape = []
        j = 0
        for v in target_domain:
            if v in src_pos:
                shape.append(data.shape[j])
                j += 1
            else:
                shape.append(1)

        data = data.reshape(shape)
        # Broadcast to full target shape
        data = np.broadcast_to(data, target_shape)
        return data

    def star(self, other: "Section", union_domain: Optional[Sequence[Var]] = None) -> "Section":
        """
        Join-product (⋆) on the union domain.

        (f ⋆ g)(x_{U∪W}) = f(x_U) ⊗ g(x_W)

        Determinism:
          - If union_domain is None, we use sorted(U ∪ W) as canonical order.
          - This is Patch-C compatible (deterministic flattening).
        """
        def _sr_tag(sr: Any) -> Tuple[type, Optional[str]]:
            return (type(sr), getattr(sr, "name", None))

        if _sr_tag(self.semiring) != _sr_tag(other.semiring):
            raise ValueError("Cannot star sections from different semirings")

        U = set(self.domain)
        W = set(other.domain)
        if union_domain is None:
            union = tuple(sorted(U | W))
        else:
            union = tuple(union_domain)

        # Determine target shape from whichever section carries each variable
        target_shape = []
        for v in union:
            if v in U:
                target_shape.append(self.dim_of(v))
            elif v in W:
                target_shape.append(other.dim_of(v))
            else:
                raise RuntimeError(f"Variable {v} not in either domain")
        target_shape = tuple(target_shape)

        a = self._aligned_view(union, target_shape)
        b = other._aligned_view(union, target_shape)
        
        # Use semiring multiplication
        if hasattr(self.semiring, 'mul') and callable(self.semiring.mul):
            # Check if it's a scalar semiring or runtime
            if hasattr(self.semiring, 'dtype'):
                # It's a SemiringRuntime
                out = self.semiring.mul(a, b)
            else:
                # It's a scalar semiring, use vectorized version
                out = np.vectorize(self.semiring.mul)(a, b)
        else:
            # Fallback to numpy multiplication
            out = a * b
            
        return Section(union, out, self.semiring)

    def restrict(self, target_domain: Sequence[Var]) -> "Section":
        """
        Marginalize (ρ_{U->T}) to exactly the variables in target_domain *in that order*.

        (ρ f)(x_T) = ⊕_{x_{U\\T}} f(x_T, x_{U\\T})

        This is the critical operation used by:
          - TopologyEngine transport kernels (existential quantification in SAT)
          - Sum-Product updates (sum out all but target var)
          - Graph surgery slicing sanity checks

        Important:
          - target_domain must be a subset of self.domain.
          - Output axes order == target_domain order (do not auto-sort here).
        """
        T = tuple(target_domain)
        U = self.domain

        # Validate subset
        Uset = set(U)
        for v in T:
            if v not in Uset:
                raise ValueError(f"restrict target var {v} not in section domain {U}")

        if T == U:
            return self

        # Handle empty target (scalar result)
        if not T:
            # Reduce all axes
            if hasattr(self.semiring, 'add_reduce'):
                result = self.semiring.add_reduce(self.data, axis=tuple(range(self.data.ndim)))
            else:
                result = np.sum(self.data)
            return Section((), np.atleast_1d(result).reshape(()), self.semiring)

        # Permute so kept axes are first in requested order, then eliminated axes
        kept_axes = [U.index(v) for v in T]
        elim_vars = [v for v in U if v not in set(T)]
        elim_axes = [U.index(v) for v in elim_vars]
        perm = kept_axes + elim_axes

        data = np.transpose(self.data, axes=perm) if perm != list(range(len(U))) else self.data

        # Reduce away eliminated axes (which are now trailing)
        if elim_axes:
            axes_to_reduce = tuple(range(len(kept_axes), len(U)))
            if hasattr(self.semiring, 'add_reduce'):
                data = self.semiring.add_reduce(data, axis=axes_to_reduce)
            else:
                data = np.sum(data, axis=axes_to_reduce)

        return Section(T, data, self.semiring)

    def normalize(self, axis: Optional[int] = None) -> "Section":
        """
        Semiring-specific normalization.
        - For probabilistic semirings: normalizes to sum=1 along axis (or globally)
        - For log semiring: subtract logZ
        - For SAT: no-op
        """
        if hasattr(self.semiring, 'normalize'):
            data = self.semiring.normalize(self.data, axis=axis)
        else:
            data = self.data
        return Section(self.domain, data, self.semiring)

    def as_bool_support(self, zero: Optional[Any] = None) -> "Section":
        """
        Convert any-valued section into a boolean support mask section.

        support(x) = (self(x) != 0)

        For float log-space, you typically want support(x)=isfinite(x), so pass zero=-inf.
        """
        from catbp.algebra.semiring import SATSemiring
        
        if zero is None:
            z = self.semiring.zero
        else:
            z = zero

        # Default: value != zero
        if np.issubdtype(self.data.dtype, np.floating) and np.isneginf(z):
            mask = np.isfinite(self.data)
        else:
            mask = (self.data != z)

        mask = mask.astype(bool, copy=False)
        return Section(self.domain, mask, SATSemiring())

    def __repr__(self) -> str:
        return f"Section(domain={self.domain}, shape={self.data.shape}, semiring={type(self.semiring).__name__})"
