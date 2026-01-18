# Formal Specification of CatBP

This document provides the formal mathematical specification of the Categorical Belief Propagation system.

## 1. Objects

### 1.1 Variables

A **variable** is a named entity with an associated finite domain.

```
Variable := (name: String, domain: ℕ⁺)
```

The domain size `d` defines the set of possible values `{0, 1, ..., d-1}`.

**Implementation**: Variables are assigned unique integer IDs (`VarID`) during compilation. The mapping `domains: Dict[VarID, int]` stores domain sizes.

### 1.2 Domains

Given a set of variables `V = {v₁, ..., vₖ}` with domains `D_{v_i}`, the **joint domain** is:

```
D_V := D_{v₁} × D_{v₂} × ... × D_{vₖ}
```

The ordering is always **canonical** (sorted by VarID) to ensure deterministic semantics.

### 1.3 Factors

A **factor** is a function from a joint domain to a semiring:

```
Factor φ: D_S → R
```

where:
- `S ⊆ V` is the **scope** (set of variables the factor depends on)
- `R` is a commutative semiring

**Implementation**: Factors are represented as dense tensors with axes corresponding to scope variables in canonical order.

### 1.4 Sections

A **section** over a variable set `U` is a semiring-valued function on `D_U`:

```
Section σ: D_U → R
Γ(U) := { σ: D_U → R }
```

Sections are the fundamental data type for beliefs, messages, and factors.

**Key operations**:

1. **Star product** (join): For sections `f ∈ Γ(U)` and `g ∈ Γ(W)`:
   ```
   (f ⋆ g)(x_{U∪W}) := f(x_U) ⊗ g(x_W)
   ```

2. **Restriction** (marginalization): For `T ⊆ U`:
   ```
   (ρ_{U→T} f)(x_T) := ⊕_{x_{U\T}} f(x_T, x_{U\T})
   ```

### 1.5 Supports

The **support** of a section is its boolean projection:

```
supp(σ) := { x ∈ D_U : σ(x) ≠ 0_R }
```

For computational purposes, support is represented as a Boolean section in the SAT semiring.

**Implementation**: `factor_support()` extracts boolean support from any semiring-valued section.

## 2. Algebraic Structures

### 2.1 Semiring Axioms

A **commutative semiring** `(R, ⊕, ⊗, 0, 1)` satisfies:

| Axiom | Statement |
|-------|-----------|
| A1 | `(R, ⊕, 0)` is a commutative monoid |
| A2 | `(R, ⊗, 1)` is a commutative monoid |
| A3 | `a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)` (distributivity) |
| A4 | `a ⊗ 0 = 0` (annihilation) |

### 2.2 Implemented Semirings

| Semiring | ⊕ | ⊗ | 0 | 1 | Use Case |
|----------|---|---|---|---|----------|
| **SAT** | OR | AND | False | True | Boolean satisfiability, support computation |
| **Prob** | + | × | 0.0 | 1.0 | Probability distributions |
| **LogProb** | logsumexp | + | -∞ | 0.0 | Numerically stable log-probabilities |
| **Counting** | + | × | 0 | 1 | Solution counting (#SAT) |

### 2.3 Required Laws

For correctness of compilation and execution:

1. **Associativity of ⋆**: `(f ⋆ g) ⋆ h = f ⋆ (g ⋆ h)`
2. **Commutativity of ⋆**: `f ⋆ g = g ⋆ f`
3. **Naturality of ρ**: `ρ_{V→T} ∘ ρ_{U→V} = ρ_{U→T}` for `T ⊆ V ⊆ U`
4. **Push-pull**: `ρ(f ⋆ g) = f ⋆ ρ(g)` when `f` does not depend on eliminated variables

### 2.4 Consequences if Violated

- **Non-commutative semiring**: Star product ordering becomes significant; execution order must be specified
- **Non-associative**: Contraction semantics become ambiguous
- **Annihilation failure**: Zero-support states may propagate incorrectly

## 3. Topological Structures

### 3.1 Factor Graph

A **factor graph** `G = (V, F, E)` consists of:
- Variables `V`
- Factors `F`
- Edges `E ⊆ V × F` connecting variables to factors containing them

### 3.2 Nerve Graph

The **nerve** `G_N = (F, E_N)` is the intersection graph of factors:
- Nodes: Factors
- Edges: `(f, g) ∈ E_N` iff `scope(f) ∩ scope(g) ≠ ∅`
- Edge labels: Interface `I_{fg} := scope(f) ∩ scope(g)`

### 3.3 Simplicial Complex (2-Skeleton)

The **Kan-relevant 2-skeleton** consists of:
- **0-simplices** (vertices): Factors
- **1-simplices** (edges): Pairs with non-empty pairwise intersection
- **2-simplices** (triangles): Triples `{a, b, c}` with non-empty triple intersection

### 3.4 Cycles and Fundamental Cycles

Given a spanning tree `T` of `G_N`:
- A **chord** is an edge in `G_N \ T`
- The **fundamental cycle** of chord `e = (u, v)` is `e ∪ P_{u,v}` where `P_{u,v}` is the unique tree path

### 3.5 Holonomy

**Holonomy** measures the obstruction to consistent transport around a cycle.

For a fundamental cycle with chord interface `J_e`:

1. **Transport kernel** `K^{U→V}`: For factor `f` with `U, V ⊆ scope(f)`:
   ```
   K(u, v) = 1 iff ∃z: supp_f(u, v, z) = True
   ```

2. **Holonomy matrix** `H_e`: Composition of transport kernels around the cycle:
   ```
   H_e = K₁ ∘ K₂ ∘ ... ∘ Kₖ : D_{J_e} → D_{J_e}
   ```

3. **Mode quotient**: SCCs of the holonomy digraph define equivalence classes (modes)

### 3.6 Homology Interpretation

- **H₁(nerve) = 0**: All cycles are boundary; tree BP is exact
- **H₁(nerve) ≠ 0**: Non-trivial cycles exist; mode variables needed

The system computes `dim(H₁)` over GF(2) to identify essential chords.

## 4. Inference Objective

### 4.1 Partition Function

Given factors `{φ_a}_{a∈F}`, the **partition function** is:

```
Z := ⊕_{x∈D_V} ⊗_{a∈F} φ_a(x_{S_a})
```

### 4.2 Marginals

The **marginal** of variable set `T` is:

```
p(x_T) := (1/Z) · ⊕_{x_{V\T}} ⊗_{a∈F} φ_a(x_{S_a})
```

### 4.3 Mode-Augmented Objective

With mode variables `{c_e}` for chords `e`:

```
Z = ⊕_{x,c} ⊗_{a∈F} φ_a(x_{S_a}) · ⊗_e σ_e(x_{J_e}, c_e)
```

where `σ_e` is the selector enforcing mode consistency.

## 5. Correctness Criteria

### 5.1 Compilation Correctness

A compilation is **correct** iff:

1. **Structural preservation**: The augmented graph has tree-structure
2. **Mode completeness**: Every joint state consistent with some global assignment maps to exactly one mode per chord
3. **Selector validity**: `σ_e(j, c) = 1` iff state `j` belongs to mode `c` and is holonomy-feasible

### 5.2 Execution Validity

An execution is **valid** iff:

1. **Determinism**: Same inputs produce identical outputs across runs
2. **Semiring consistency**: All operations respect semiring axioms
3. **Axis alignment**: Tensor contractions respect variable correspondence

### 5.3 Result Consistency

Results are **consistent** iff:

1. **Z correctness**: Computed Z equals brute-force enumeration
2. **Marginal normalization**: Marginals sum to 1 (for probability semiring)
3. **Global consistency**: Marginals are marginals of a valid joint distribution



## 6. Complexity

The HATCC algorithm has complexity:

```
O(n² · d_max + c · k_max · δ_max³ + n · δ_max²)
```

where:
- `n`: Number of factors
- `d_max`: Maximum factor scope size
- `c`: Number of fundamental cycles (chords)
- `k_max`: Maximum cycle length
- `δ_max`: Maximum interface size

The number of modes per chord is bounded by the interface domain size, yielding tractable inference when interfaces are small.
