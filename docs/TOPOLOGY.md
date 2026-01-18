# Topological Foundations

This document describes the topological structures in CatBP: factor graph structure, nerve construction, cycles, holonomy, and homology.

## 1. Why Topology Is Required

Standard belief propagation assumes tree structure for exact inference. On graphs with cycles (loops), BP produces approximations that may:
- Fail to converge
- Converge to incorrect fixed points
- Miss global constraints

**Topological analysis** detects when tree BP is insufficient:
- **Nerve construction** captures factor dependencies
- **Cycle detection** identifies potential obstructions
- **Homology** filters essential cycles (non-boundary)
- **Holonomy** quantifies the obstruction at each cycle

The key insight: exact inference requires consistent parallel transport around every cycle, which fails when holonomy is non-trivial.

## 2. Factor Graph Structure

### 2.1 Definition

A **factor graph** consists of:
- Variables `V = {v₁, ..., vₙ}` with finite domains
- Factors `F = {f₁, ..., fₘ}` with scopes `S_a ⊆ V`
- Bipartite structure: variables connect to factors containing them

### 2.2 Implementation

```python
class FactorGraphStructure:
    var_card: Dict[str, int]           # Variable cardinalities
    factors: Dict[str, FactorDef]      # Factor definitions
    var_to_factors: Dict[str, List[str]]  # Incidence
```

**Key operations**:
- `add_variable(name, card)`: Register variable with domain size
- `add_factor(name, scope)`: Register factor with scope
- `interface(f1, f2)`: Compute shared variables `S_{f1} ∩ S_{f2}`
- `factors_containing(var)`: Get all factors involving a variable

### 2.3 Canonical Ordering

Scopes are always stored in **canonical (sorted) order**:

```python
@dataclass(frozen=True)
class FactorDef:
    name: str
    scope: Tuple[str, ...]  # Always sorted
    
    def __post_init__(self):
        if self.scope != tuple(sorted(self.scope)):
            object.__setattr__(self, 'scope', tuple(sorted(self.scope)))
```

This ensures deterministic behavior across all downstream operations.

## 3. Nerve Graph

### 3.1 Definition

The **nerve** `G_N = (F, E_N)` is the intersection graph of factors:
- **Nodes**: Factors
- **Edges**: `(a, b) ∈ E_N` iff `S_a ∩ S_b ≠ ∅`
- **Edge labels**: Interface `I_{ab} = S_a ∩ S_b`

### 3.2 Implementation

```python
class Nerve:
    def __init__(self, struct: FactorGraphStructure):
        self.struct = struct
        self.g = nx.Graph()  # NetworkX graph
        self._build()
    
    def _build(self):
        # Add all factors as nodes
        for f in self.struct.factors:
            self.g.add_node(f)
        
        # Add edges for intersecting scopes
        for v, fs in self.struct.var_to_factors.items():
            for i in range(len(fs)):
                for j in range(i + 1, len(fs)):
                    a, b = fs[i], fs[j]
                    I = self.struct.interface(a, b)
                    if I:
                        w = log_interface_size(I)
                        self.g.add_edge(a, b, interface=I, weight=w)
```

### 3.3 Edge Weights

Edges are weighted by **log interface size**:

```
w(a, b) = Σ_{v ∈ I_{ab}} log|D_v|
```

Larger interfaces = larger weight = preferred for spanning tree (minimizes message size during BP).

## 4. Simplicial Complex (2-Skeleton)

### 4.1 Kan-Relevant Structure

The **2-skeleton** extends the nerve to include triangles:
- **0-simplices** (vertices): Factors
- **1-simplices** (edges): Pairs with pairwise non-empty intersection
- **2-simplices** (triangles): Triples `{a, b, c}` with non-empty triple intersection

```python
@dataclass(frozen=True)
class Nerve2Skeleton:
    vertices: Tuple[Node, ...]
    edges: Tuple[Tuple[Node, Node], ...]
    triangles: Tuple[Tuple[Node, Node, Node], ...]
```

### 4.2 Why Triangles Matter

Triangles are **Kan fillers** that kill 1-cycles:
- A cycle `a → b → c → a` is **boundary** if `{a, b, c}` forms a triangle
- Boundary cycles don't contribute to H₁
- Only non-boundary cycles require mode variables

### 4.3 Construction

```python
def build_2_skeleton_from_scopes(factor_scopes: Dict[Node, Tuple[Var, ...]]) -> Nerve2Skeleton:
    nodes = sorted(factor_scopes.keys())
    
    # Edges: pairwise intersection
    edges = [(u, v) for u, v in combinations(nodes, 2)
             if set(factor_scopes[u]) & set(factor_scopes[v])]
    
    # Triangles: triple intersection
    triangles = [(a, b, c) for a, b, c in combinations(nodes, 3)
                 if set(factor_scopes[a]) & set(factor_scopes[b]) & set(factor_scopes[c])]
    
    return Nerve2Skeleton(nodes, edges, triangles)
```

## 5. Backbone Tree and Chords

### 5.1 Backbone Selection

The **backbone** is a spanning tree of the nerve, chosen to minimize message complexity:

```python
def choose_backbone_tree(nerve: nx.Graph) -> nx.Graph:
    return nx.maximum_spanning_tree(nerve, weight="weight")
```

Maximum spanning tree by interface weight = largest interfaces in the tree = smallest messages along non-tree edges.

### 5.2 Chords

A **chord** is an edge in the nerve but not in the backbone:

```
Chords = E_N \ E_T
```

Each chord defines a **fundamental cycle**: the chord plus the unique tree path between its endpoints.

### 5.3 Rooted Tree Structure

For BP message passing, the tree is rooted:

```python
def root_tree(tree: nx.Graph, root: int) -> Tuple[Dict, Dict]:
    parent = {root: None}
    children = {fid: [] for fid in tree.nodes()}
    
    stack = [root]
    while stack:
        u = stack.pop()
        for v in tree.neighbors(u):
            if v not in parent:
                parent[v] = u
                children[u].append(v)
                stack.append(v)
    
    return parent, children
```

## 6. Cycles and Holonomy

### 6.1 Fundamental Cycle Definition

For chord `e = (u, v)` with backbone path `P = u → p₁ → ... → v`:

```
Cycle_e = (u, p₁, ..., v, u)  [via chord]
```

The cycle is **typed** with transport steps specifying interfaces:

```python
@dataclass(frozen=True)
class TransportStep:
    factor: str
    in_interface: Tuple[str, ...]
    out_interface: Tuple[str, ...]

@dataclass(frozen=True)
class HolonomyCycle:
    chord: Tuple[str, str]
    chord_interface: Tuple[str, ...]  # J_e = S_u ∩ S_v
    steps: Tuple[TransportStep, ...]
```

### 6.2 Transport Kernels

A **transport kernel** `K^{U→V}` maps interface states through a factor:

```
K(u, v) = 1 iff ∃z ∈ D_{S\(U∪V)}: supp_φ(u, v, z)
```

Implementation using sparse matrices:

```python
def build_transport_kernel_boolean(
    factor_support: np.ndarray,  # Boolean tensor on scope
    S_vars: Tuple[int, ...],     # Scope variables
    U_vars: Tuple[int, ...],     # Input interface
    V_vars: Tuple[int, ...],     # Output interface
    domains: Dict[int, int]
) -> sp.csr_matrix:
    # 1. Project support to U ∪ V (existential quantification over internals)
    # 2. Extract valid (u, v) pairs
    # 3. Build sparse CSR matrix
```

### 6.3 Holonomy Matrix

The **holonomy** `H_e` is the composition of transport kernels around the cycle:

```
H_e = K₁ ∘ K₂ ∘ ... ∘ Kₖ : D_{J_e} → D_{J_e}
```

Implementation:

```python
def compute_holonomy_boolean(
    steps: List[Tuple[int, Tuple[int, ...], Tuple[int, ...]]],
    factor_supports: Dict[int, Tuple[Tuple[int, ...], np.ndarray]],
    domains: Dict[int, int]
) -> sp.csr_matrix:
    J = steps[0][1]  # Chord interface
    dimJ = prod(domains[v] for v in J)
    
    H = sp.identity(dimJ, dtype=np.bool_)
    
    for fac, U, V in steps:
        K = build_transport_kernel_boolean(...)
        H = boolean_matmul(H, K)
    
    return H
```

### 6.4 Meaning of Holonomy in Inference

**Trivial holonomy** (`H = I`):
- Any interface state returns to itself after transport
- Tree BP is exact on this cycle
- No mode variable needed

**Non-trivial holonomy** (`H ≠ I`):
- Some states map to different states after transport
- Parallel transport is inconsistent
- Mode variables needed to track equivalence classes

## 7. Homology Computation

### 7.1 Purpose

Not all chords contribute essential cycles. **Homology computation** identifies which chords are truly needed:

- Chords whose fundamental cycles are **boundaries** can be ignored
- Only **H₁-essential** chords require mode variables

### 7.2 GF(2) Linear Algebra

Computation uses arithmetic over GF(2) (binary field):

```python
def _gf2_reduce(v: int, pivots: Dict[int, int]) -> int:
    """Reduce bitvector v by pivot basis using XOR."""
    while v:
        p = v.bit_length() - 1
        b = pivots.get(p)
        if b is None:
            break
        v ^= b
    return v

def _gf2_insert(v: int, pivots: Dict[int, int]) -> bool:
    """Insert v into pivot basis if independent."""
    v = _gf2_reduce(v, pivots)
    if v == 0:
        return False
    p = v.bit_length() - 1
    pivots[p] = v
    return True
```

### 7.3 Boundary Subspace

The **boundary subspace** B₁ is generated by triangle boundaries:

```python
def _triangle_boundary_vec(tri, e2i: Dict) -> int:
    """Boundary of triangle as GF(2) edge vector."""
    a, b, c = tri
    v = 0
    for edge in [(a,b), (b,c), (a,c)]:
        v ^= (1 << e2i[canon(edge)])
    return v
```

### 7.4 Non-Trivial H₁ Chords

```python
def nontrivial_h1_chords(
    *,
    tree: nx.Graph,
    nerve: nx.Graph,
    triangles: Iterable[Tuple[Node, Node, Node]]
) -> List[Edge]:
    """Find chords with non-trivial H₁ classes."""
    
    # 1. Build boundary subspace from triangles
    B_pivots = {}
    for tri in triangles:
        b = triangle_boundary(tri)
        gf2_insert(b, B_pivots)
    
    # 2. Test each chord's fundamental cycle
    H_pivots = {}
    kept = []
    
    for chord in chords:
        cyc = fundamental_cycle_vec(chord)
        cls = gf2_reduce(cyc, B_pivots)  # mod boundaries
        cls = gf2_reduce(cls, H_pivots)  # mod already-kept
        if cls != 0:
            gf2_insert(cls, H_pivots)
            kept.append(chord)
    
    return kept
```

## 8. When Homology Is Zero vs Non-Zero

### 8.1 H₁ = 0 (Exact Tree BP)

Conditions implying H₁ = 0:
- Factor graph is a tree
- All chords have triangular fillers
- Junction tree exists with tractable treewidth

**Consequence**: Standard tree BP yields exact marginals.

### 8.2 H₁ ≠ 0 (Mode Variables Needed)

Conditions implying H₁ ≠ 0:
- Graph has irreducible cycles
- Some chords lack triangle fillers
- Grid graphs, random graphs with many cycles

**Consequence**: Mode variables must be introduced to track holonomy equivalence classes.

### 8.3 Dimension of H₁

`dim(H₁)` = number of independent non-boundary cycles = number of mode variables needed

For a connected graph:
```
dim(H₁) = |E_N| - |V| + 1 - (triangles killing cycles)
```

## 9. Mode Quotient

### 9.1 SCC Decomposition

The holonomy matrix `H_e` induces a directed graph on `D_{J_e}`. Its **strongly connected components** define mode equivalence:

```python
def quotient_scc_from_holonomy(H: sp.csr_matrix) -> Tuple[int, np.ndarray]:
    n, labels = connected_components(H, directed=True, connection="strong")
    return n, labels.astype(np.int32)
```

### 9.2 Fixed-Point Filtering

States with `H[u,u] = 0` cannot complete a consistent cycle and are excluded:

```python
def filter_fixed_points_for_modes(H, scc_labels):
    diag = H.diagonal().astype(bool)
    fixed = np.where(diag)[0]
    # Remap valid SCCs to 0..k-1
```

### 9.3 Mode Variable

For each H₁-essential chord `e`:
- Mode axis `c_e` with domain `{0, 1, ..., k_e - 1}`
- Selector `σ_e(j, c)` enforcing consistency
- Anchored at chord endpoints in the execution plan

## 10. Summary: Topological Pipeline

```
Factor Graph
     │
     ▼
┌────────────────┐
│ Nerve Graph    │  Intersection graph of factors
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ 2-Skeleton     │  Add triangles for Kan structure
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ Backbone Tree  │  Maximum spanning tree
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ Chords         │  Non-tree edges
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ H₁ Computation │  Filter boundary cycles (GF(2))
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ Holonomy       │  Transport kernel composition
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ Mode Quotient  │  SCC + fixed-point filter
└───────┬────────┘
        │
        ▼
Mode Variables + Selectors
```
