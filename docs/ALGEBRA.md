# Algebraic Foundations

This document specifies the algebraic structures underlying CatBP: semirings, sections, and support computation.

## 1. Semirings

### 1.1 Formal Definition

A **commutative semiring** is a tuple `(R, ⊕, ⊗, 0, 1)` where:

1. `(R, ⊕, 0)` is a commutative monoid:
   - `a ⊕ (b ⊕ c) = (a ⊕ b) ⊕ c` (associativity)
   - `a ⊕ b = b ⊕ a` (commutativity)
   - `a ⊕ 0 = a` (identity)

2. `(R, ⊗, 1)` is a commutative monoid:
   - `a ⊗ (b ⊗ c) = (a ⊗ b) ⊗ c` (associativity)
   - `a ⊗ b = b ⊗ a` (commutativity)
   - `a ⊗ 1 = a` (identity)

3. Distributivity: `a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)`

4. Annihilation: `a ⊗ 0 = 0`

### 1.2 Implemented Semirings

#### Boolean / SAT Semiring

```python
@dataclass(frozen=True)
class SATSemiring:
    zero: bool = False
    one: bool = True
    
    def add(self, a, b) -> bool:  # ⊕ = OR
        return bool(a) or bool(b)
    
    def mul(self, a, b) -> bool:  # ⊗ = AND
        return bool(a) and bool(b)
```

**Use case**: Support computation, satisfiability checking, constraint propagation.

**Interpretation**:
- `0 = False` = unsatisfiable
- `1 = True` = satisfiable
- `⊕ = OR` = disjunction (exists a solution)
- `⊗ = AND` = conjunction (both conditions hold)

#### Probability Semiring

```python
@dataclass(frozen=True)
class ProbSemiring:
    zero: float = 0.0
    one: float = 1.0
    
    def add(self, a, b) -> float:  # ⊕ = +
        return float(a) + float(b)
    
    def mul(self, a, b) -> float:  # ⊗ = ×
        return float(a) * float(b)
```

**Use case**: Probability distributions, partition function computation.

**Interpretation**:
- Values are non-negative reals
- `⊕ = +` = sum over alternatives
- `⊗ = ×` = product of independent factors

#### Log-Probability Semiring

```python
@dataclass(frozen=True)
class LogProbSemiring:
    zero: float = -np.inf  # log(0)
    one: float = 0.0       # log(1)
    
    def add(self, a, b) -> float:  # ⊕ = logsumexp
        return float(np.logaddexp(a, b))
    
    def mul(self, a, b) -> float:  # ⊗ = +
        return float(a) + float(b)
```

**Use case**: Numerically stable computation when probabilities span many orders of magnitude.

**Interpretation**:
- Values are log-probabilities: `v = log(p)`
- `⊕ = logsumexp` = `log(exp(a) + exp(b))`
- `⊗ = +` = `log(p₁ × p₂) = log(p₁) + log(p₂)`

#### Counting Semiring

```python
@dataclass(frozen=True)
class CountingSemiring:
    zero: int = 0
    one: int = 1
    
    def add(self, a, b) -> int:  # ⊕ = +
        return int(a) + int(b)
    
    def mul(self, a, b) -> int:  # ⊗ = ×
        return int(a) * int(b)
```

**Use case**: Counting solutions (#SAT), model counting.

**Interpretation**:
- Values are non-negative integers
- `⊕ = +` = sum of counts
- `⊗ = ×` = product of counts (Cartesian product size)

### 1.3 Semiring Protocol

The `Semiring` protocol defines the interface for all semiring implementations:

```python
class Semiring(Protocol):
    zero: Any
    one: Any
    
    def add(self, a: Any, b: Any) -> Any: ...
    def mul(self, a: Any, b: Any) -> Any: ...
    def is_zero(self, a: Any) -> bool: ...
    def add_reduce(self, x: np.ndarray, axis: Optional[...]) -> np.ndarray: ...
    def mul_reduce(self, xs: list[np.ndarray]) -> np.ndarray: ...
    def normalize(self, x: np.ndarray, axis: Optional[...]) -> np.ndarray: ...
```

### 1.4 Vectorized Runtime

For efficient computation, `SemiringRuntime` provides numpy-backed operations:

```python
@dataclass(frozen=True)
class SemiringRuntime:
    name: str
    dtype: np.dtype
    add: Callable[[np.ndarray, np.ndarray], np.ndarray]
    mul: Callable[[np.ndarray, np.ndarray], np.ndarray]
    add_reduce: Callable[[np.ndarray, Optional[Tuple[int, ...]]], np.ndarray]
    one: Callable[[Tuple[int, ...]], np.ndarray]
    zero: Callable[[Tuple[int, ...]], np.ndarray]
    normalize: Optional[Callable[...]]
```

Factory functions create configured runtimes:
- `sat_semiring()` → Boolean operations with `np.logical_or`, `np.logical_and`
- `prob_semiring(dtype)` → Float operations with `np.add`, `np.multiply`
- `logprob_semiring(dtype)` → Log-space with `logsumexp`, `np.add`
- `counting_semiring(dtype)` → Integer operations

## 2. Sections

### 2.1 Definition

A **section** over variable set `U` is a semiring-valued function on the joint domain:

```
σ: D_U → R
```

where `D_U = D_{u₁} × ... × D_{uₖ}` for variables `U = {u₁, ..., uₖ}` in canonical order.

### 2.2 Implementation

```python
@dataclass(frozen=True)
class Section:
    domain: Tuple[Var, ...]    # Ordered variables (axis labels)
    data: np.ndarray           # Dense tensor
    semiring: Any              # Semiring for operations
```

**Invariants**:
- `len(domain) == data.ndim`
- `domain` has no duplicates
- `data.shape[i]` corresponds to domain size of `domain[i]`

### 2.3 Star Product (Join)

The **star product** combines sections on overlapping domains:

```
(f ⋆ g)(x_{U∪W}) = f(x_U) ⊗ g(x_W)
```

Implementation:

```python
def star(self, other: Section, union_domain: Optional[...] = None) -> Section:
    # 1. Determine union domain (default: sorted)
    union = sorted(set(self.domain) | set(other.domain))
    
    # 2. Align both tensors to union domain
    a = self._aligned_view(union, target_shape)
    b = other._aligned_view(union, target_shape)
    
    # 3. Apply semiring multiplication
    out = semiring.mul(a, b)
    
    return Section(union, out, self.semiring)
```

**Properties**:
- Commutative: `f ⋆ g = g ⋆ f`
- Associative: `(f ⋆ g) ⋆ h = f ⋆ (g ⋆ h)`
- Unit: `f ⋆ 1_∅ = f` where `1_∅` is the scalar unit section

### 2.4 Restriction (Marginalization)

**Restriction** projects a section onto a subset of variables:

```
(ρ_{U→T} f)(x_T) = ⊕_{x_{U\T}} f(x_T, x_{U\T})
```

Implementation:

```python
def restrict(self, target_domain: Sequence[Var]) -> Section:
    # 1. Validate target ⊆ self.domain
    # 2. Permute: kept axes first, eliminated axes last
    # 3. Sum-reduce eliminated axes using semiring.add_reduce
    # 4. Return Section with target_domain
```

**Properties**:
- Projection: `ρ_{V→T} ∘ ρ_{U→V} = ρ_{U→T}` for `T ⊆ V ⊆ U`
- Push-pull: `ρ(f ⋆ g) = f ⋆ ρ(g)` when `f` doesn't depend on eliminated variables

### 2.5 Unit Section

The **unit section** on domain `U` is the constant-one function:

```python
@staticmethod
def unit(domain, shape, semiring, dtype=None) -> Section:
    data = np.full(shape, semiring.one, dtype=dtype)
    return Section(tuple(domain), data, semiring)
```

### 2.6 Normalization

For probabilistic semirings, **normalization** scales values to sum to one:

```python
def normalize(self, axis: Optional[int] = None) -> Section:
    data = self.semiring.normalize(self.data, axis=axis)
    return Section(self.domain, data, self.semiring)
```

For SAT semiring, normalization is the identity operation.

## 3. Support Computation

### 3.1 Definition

The **support** of a section is the set of configurations with non-zero values:

```
supp(σ) = { x ∈ D_U : σ(x) ≠ 0_R }
```

### 3.2 Boolean Projection

Support is represented as a Boolean section in the SAT semiring:

```python
def factor_support(
    phi: Section,
    *,
    eps: float = 0.0,
    log_space: bool = False,
    treat_nan_as_zero: bool = True
) -> Section:
    """Convert any-valued section to boolean support."""
    
    if log_space:
        # In log-space, zero mass is -∞
        mask = np.isfinite(data)
    else:
        # Normal: nonzero by magnitude
        mask = np.abs(data) > eps
    
    return Section(phi.domain, mask.astype(bool), SATSemiring())
```

### 3.3 Batch Support Extraction

```python
def build_factor_supports(
    factors: Mapping[str, Section],
    *,
    eps: float = 0.0,
    log_space: bool = False
) -> Dict[str, Section]:
    """Convert all factors to boolean supports."""
    return {name: factor_support(sec, eps=eps, log_space=log_space)
            for name, sec in factors.items()}
```

### 3.4 Why Support Matters

Support computation is **essential** for HATCC:

1. **Topology uses SAT semiring**: Holonomy computation operates on support, not values
2. **Transport kernels**: `K(u,v) = 1` iff `∃z: supp(u,v,z)`
3. **Mode quotient**: SCCs of support graph define modes
4. **Separation of concerns**: Structure (support) is compiled once; values are runtime

## 4. Relation to Message Passing

### 4.1 Messages as Sections

In belief propagation, a **message** from factor `a` to variable `v` is a section on `{v}`:

```
μ_{a→v}(x_v) = ⊕_{x_{S_a \ {v}}} φ_a(x_{S_a}) ⊗ ∏_{u∈S_a\{v}} μ_{u→a}(x_u)
```

This is exactly:
1. Star product of factor with incoming messages
2. Restrict to target variable

### 4.2 Beliefs as Sections

A **belief** at factor `a` is a section on `S_a`:

```
β_a(x_{S_a}) = φ_a(x_{S_a}) ⊗ ∏_{e} μ_{e→a}(x_{I_e})
```

Product of local factor with all incoming messages.

### 4.3 Tree Inference

On a tree-structured factor graph:
1. Upward pass: Compute messages from leaves to root
2. Downward pass: Compute messages from root to leaves
3. Beliefs: Product of factor with all messages

CatBP generalizes this to loopy graphs via mode augmentation.

## 5. Mode Mapping from Holonomy

### 5.1 Quotient Construction

Given holonomy matrix `H` on interface `D_J`:

```python
def mode_mapping_from_holonomy(
    H: sp.csr_matrix,
    *,
    require_fixed_point: bool = True
) -> Tuple[np.ndarray, int]:
    """Compute mode quotient π: D_J → Q_e."""
    
    # 1. SCC decomposition (orbit partition)
    n_comp, labels = connected_components(H, directed=True, connection="strong")
    
    # 2. Optional: filter by fixed-point feasibility
    if require_fixed_point:
        diag = H.diagonal()
        valid = (diag > 0)
        # Remap valid SCCs to 0..k-1
    
    return mapping, num_modes
```

### 5.2 Fixed-Point Feasibility

A state `u ∈ D_J` is **fixed-point feasible** iff `H[u,u] = 1`:
- There exists a consistent assignment completing the cycle
- The state can be "returned to" after transport around the loop

States with `H[u,u] = 0` are excluded from valid modes.

### 5.3 Mode Selectors

A **mode selector** is a section on `(J, mode)`:

```
σ_e(j, c) = 1 iff π(j) = c and j is feasible
```

This enforces consistency between interface states and mode assignments.
