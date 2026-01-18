# Compiler and Runtime Semantics

This document specifies the semantics of compilation, planning, and execution in CatBP.

## 1. Compilation Semantics

### 1.1 Overview

Compilation transforms a factor graph specification into an executable plan:

```
Compile: (VarDomains, Factors) → BPPlan
```

The compilation is **deterministic**: identical inputs always produce identical plans.

### 1.2 Input Specification

```python
var_domains_named: Dict[str, int]     # Variable name → domain size
factors_named: Dict[str, Tuple[Tuple[str, ...], np.ndarray]]  # Factor name → (scope, tensor)
```

### 1.3 ID Assignment

All named entities are assigned canonical integer IDs:

```python
# Variables: sorted by name
var_names = sorted(var_domains_named.keys())
var_id = {name: i for i, name in enumerate(var_names)}

# Factors: sorted by name
fac_names = sorted(factors_named.keys())
fac_id = {name: i for i, name in enumerate(fac_names)}
```

**Invariant**: ID assignment is deterministic based on lexicographic ordering.

### 1.4 Scope Canonicalization

Factor scopes are converted to canonical form:

```python
scope = tuple(sorted(var_id[v] for v in scope_names))
```

**Invariant**: All scopes are sorted tuples of VarIDs.

### 1.5 Support Extraction

Boolean support is extracted from factor tensors:

```python
support_predicate = lambda x: (x != 0)  # Default
factor_supports[fid] = BoolSection(domain=scope, data=support_predicate(data))
```

**Invariant**: Support computation uses SAT semiring exclusively.

### 1.6 Nerve Construction

The nerve graph is built from factor scopes:

```python
G = nx.Graph()
for fid in scopes:
    G.add_node(fid)

for a, b in pairs(scopes):
    inter = scope[a] ∩ scope[b]
    if inter:
        G.add_edge(a, b, interface=inter, weight=log_interface_size(inter))
```

### 1.7 Backbone Selection

Maximum spanning tree by interface weight:

```python
T = nx.maximum_spanning_tree(G, weight="weight")
```

**Invariant**: Backbone is a connected spanning tree.

### 1.8 Chord Analysis

For each chord `(u, v)`:

1. Compute fundamental cycle (tree path + chord)
2. Check H₁ contribution (may skip if boundary)
3. Compute holonomy matrix
4. Extract mode quotient via SCC

```python
labels, ncomp, diag_mask = topo.holonomy_and_quotient(cycle_factors, J, scopes)
```

### 1.9 Mode Variable Creation

For each chord with non-trivial holonomy:

```python
mode_axis_id = next_mode_id
next_mode_id += 1

payload = SelectorPayload(
    mode_axis_id=mode_axis_id,
    interface_axis_ids=J,
    quotient_map=labels,
    num_modes=ncomp,
    diag_mask=diag_mask
)
```

**Invariant**: Mode axis IDs are distinct from VarIDs (uses separate namespace via AxisType).

### 1.10 Compilation Output

```python
@dataclass(frozen=True)
class BPPlan:
    root: NodeID
    nodes: Dict[NodeID, NodePlan]
    postorder: Tuple[NodeID, ...]
    preorder: Tuple[NodeID, ...]
    belief_spec: Dict[NodeID, TensorSpec]
    keep_axis_keys_on_edge: Dict[Tuple[NodeID, NodeID], Tuple[AxisKey, ...]]
```

## 2. Planning Semantics

### 2.1 Node Plan

Each backbone node has a plan:

```python
@dataclass(frozen=True)
class NodePlan:
    id: NodeID
    parent: NodeID | None      # None for root
    children: Tuple[NodeID, ...]
    vars: Tuple[VarID, ...]    # Geometric variables in scope
    anchored_modes: Tuple[ModeID, ...]  # Mode selectors anchored here
```

### 2.2 Traversal Orders

**Postorder** (leaves to root): For upward message passing
```
postorder = DFS_postorder(root, children)
```

**Preorder** (root to leaves): For downward message passing
```
preorder = DFS_preorder(root, children)
```

### 2.3 Belief Specification

Each node's belief has a tensor specification:

```python
belief_spec[node] = TensorSpec(axes=(
    *[Axis(v, GEOMETRIC, domains[v]) for v in node.vars],
    *[Axis(m, TOPOLOGICAL, mode_sizes[m]) for m in node.modes]
))
```

**Invariant**: Geometric axes come before topological axes.

### 2.4 Message Specification

Messages between nodes keep only shared axes:

```python
keep_axis_keys_on_edge[(child, parent)] = tuple(
    (GEOMETRIC, v) for v in (child.vars ∩ parent.vars)
) + tuple(
    (TOPOLOGICAL, m) for m in straddling_modes
)
```

**Straddling modes**: Mode axes whose chord endpoints are in different subtrees.

## 3. Execution Semantics

### 3.1 Program Structure

Execution consists of three programs:

1. **Upward program**: Postorder pass computing messages to parents
2. **Downward program**: Preorder pass computing beliefs and messages to children
3. **Finalize program**: Compute partition function Z

### 3.2 Instruction Execution

Each instruction has defined semantics:

#### UNIT
```
Reg[dst] := (one_tensor(spec.shape), spec)
```
Creates tensor filled with semiring one.

#### LOAD_FACTOR
```
Reg[dst] := factor_provider(factor_id)
```
Loads factor tensor from external provider.

#### LOAD_SELECTOR
```
Reg[dst] := selector_provider(selector_id)
```
Loads mode selector tensor.

#### LOAD_SLOT
```
Reg[dst] := Slots[slot_key]
```
Loads tensor from message/belief slot.

#### STORE_SLOT
```
Slots[slot_key] := Reg[src]
```
Stores tensor to slot.

#### CONTRACT
```
Reg[dst] := ⊗_{i} align(Reg[inputs[i]], out_spec)
```
Multiplies all inputs with alignment to output specification.

#### ELIMINATE
```
Reg[dst] := ⊕_{axes ∉ keep_keys} Reg[src]
```
Marginalizes (sums out) axes not in keep_keys.

#### NORMALIZE
```
Reg[src] := normalize(Reg[src])
```
Applies semiring-specific normalization.

#### FREE
```
for r in srcs: del Reg[r]
```
Releases register memory.

### 3.3 Upward Pass Semantics

For each node in postorder:

```
belief[a] = factor[a] ⊗ (⊗_{m ∈ anchored} selector[m]) ⊗ (⊗_{c ∈ children} msg[c→a])
msg[a→parent] = ρ_{keep_keys(a,parent)}(belief[a])
```

**Invariant**: Child messages are available before parent processing.

### 3.4 Downward Pass Semantics

For each node in preorder:

```
belief[a] = factor[a] ⊗ selectors ⊗ msg[parent→a] ⊗ (⊗_{c} msg[c→a])

For each child c:
    excl_c = factor[a] ⊗ selectors ⊗ msg[parent→a] ⊗ (⊗_{c' ≠ c} msg[c'→a])
    msg[a→c] = ρ_{keep_keys(a,c)}(excl_c)
```

**Optimization**: Prefix-suffix decomposition avoids redundant computation when many children exist.

### 3.5 Finalization Semantics

```
Z = ρ_{∅}(belief[root])
```

Marginalizes all axes to obtain scalar partition function.

## 4. Determinism Guarantees

### 4.1 Compilation Determinism

The following are deterministic:
- ID assignment (lexicographic)
- Scope ordering (sorted)
- Graph iteration (sorted node/edge order)
- MST selection (tie-breaking by edge ordering)

### 4.2 Execution Determinism

The following are deterministic:
- Instruction ordering (fixed sequence)
- Tensor operations (numpy reproducibility)
- Slot keys (tuple hashing)

**Note**: Floating-point operations may have platform-dependent rounding, but identical inputs on the same platform produce identical outputs.

### 4.3 Non-Determinism Sources

The system avoids:
- Random number generators
- Hash-based iteration without sorting
- Parallel execution with race conditions
- System-dependent ordering

## 5. Evidence Propagation

### 5.1 Evidence Model

Evidence is incorporated as **multiplicative constraints**, not domain restriction:

```python
def unary_evidence(var, allowed, domains, semiring) -> NamedTensor:
    data = np.full((d,), semiring.zero)
    for a in allowed:
        data[a] = semiring.one
    return NamedTensor((var,), data, semiring)
```

### 5.2 Formal Meaning

With evidence `E`:

```
Z_E = ⊕_{x: E(x)=1} ⊗_{a} φ_a(x_{S_a})
    = ⊕_{x} ⊗_{a} φ_a(x_{S_a}) ⊗ E(x)
```

Evidence acts as an additional factor constraining the summation.

### 5.3 Evidence Integration

Evidence factors are added to the factor graph before compilation:

```python
factors["evidence_v"] = ((v,), evidence_tensor)
```

The compiler treats evidence factors identically to other factors.

## 6. Error Semantics

### 6.1 Compilation Errors

| Error | Cause | Handling |
|-------|-------|----------|
| `ValueError` | Domain/shape mismatch | Raised at Section construction |
| `KeyError` | Unknown variable/factor | Raised during lookup |
| `AssertionError` | Internal invariant violation | Indicates bug |

### 6.2 Execution Errors

| Error | Cause | Handling |
|-------|-------|----------|
| `KeyError` | Missing slot/register | Raised at instruction execution |
| `ValueError` | Axis alignment failure | Raised in align.py |
| Numerical | Overflow/underflow | May produce inf/nan |

### 6.3 Semantic Errors

| Condition | Symptom | Resolution |
|-----------|---------|------------|
| All-zero support | Z = 0 | UNSAT; no valid configurations |
| Inconsistent modes | Z ≠ expected | Check holonomy computation |
| Normalization of zero | NaN marginals | Handle Z = 0 case explicitly |

## 7. Correctness Criteria

### 7.1 Partition Function

For correct compilation and execution:

```
Z_computed = Z_brute = ⊕_{x∈D_V} ⊗_{a∈F} φ_a(x_{S_a})
```

Verified by brute-force enumeration on small instances.

### 7.2 Marginals

For correct marginals:

```
p(x_v) = (1/Z) · belief_a(x_v) for any factor a containing v
```

Verified by:
1. Marginals sum to 1 (probability semiring)
2. Marginals consistent across factors sharing variables

### 7.3 Mode Consistency

For correct mode handling:

```
∀ chord e, ∀ j ∈ D_{J_e}: σ_e(j, π(j)) = 1 ∧ (∀c ≠ π(j): σ_e(j,c) = 0)
```

Selectors enforce unique mode assignment per interface state.
