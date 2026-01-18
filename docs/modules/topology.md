# topology/

## Intent

The `topology/` module represents factor graph structure and computes topological invariants needed for HATCC. It constructs the nerve graph, identifies simplicial structure for homology, and provides algorithms for detecting non-trivial cycles.

## Public Surface

### From `structure.py`

| Export | Type | Description |
|--------|------|-------------|
| `FactorDef` | dataclass | Factor definition with canonical scope |
| `FactorGraphStructure` | class | Structure-only factor graph representation |
| `add_variable()` | method | Register a variable |
| `add_factor()` | method | Register a factor with scope |
| `interface()` | method | Compute shared variables between factors |
| `get_scope()` | method | Get factor scope |
| `factors_containing()` | method | Get factors containing a variable |

### From `nerve.py`

| Export | Type | Description |
|--------|------|-------------|
| `Nerve` | class | Nerve graph with interface labels |
| `build_nerve_graph()` | function | Build nerve from integer-indexed factors |

### From `simplicial.py`

| Export | Type | Description |
|--------|------|-------------|
| `Nerve2Skeleton` | dataclass | 2-skeleton (vertices, edges, triangles) |
| `build_2_skeleton_from_scopes()` | function | Build 2-skeleton from factor scopes |
| `build_2_skeleton_from_nerve_graph()` | function | Build 2-skeleton using existing nerve |

### From `homology.py`

| Export | Type | Description |
|--------|------|-------------|
| `nontrivial_h1_chords()` | function | Find H₁-essential chords |

## Internal Details

### Nerve Construction

The `Nerve` class wraps a NetworkX graph:
- Nodes: factor names
- Edges: pairs with non-empty scope intersection
- Edge attributes: `interface` (shared variables), `weight` (log interface size)

### 2-Skeleton Construction

Triangles are identified by testing triple intersection:
```python
for a, b, c in combinations(nodes, 3):
    inter = scope[a] ∩ scope[b] ∩ scope[c]
    if inter:
        triangles.append((a, b, c))
```

### GF(2) Homology

The homology computation uses binary arithmetic:
- Edges are encoded as bit vectors (position = edge index)
- Triangle boundaries: XOR of three edge bits
- Independence tested via pivot-based Gaussian elimination

## Invariants

1. **Canonical scopes**: Factor scopes are always sorted tuples
2. **Interface symmetry**: `interface(a, b) == interface(b, a)`
3. **Triangle validity**: Triangle edges must exist in nerve
4. **Chord definition**: Chord = nerve edge \ tree edge

## Extension Points

### Custom Nerve Weights

Override `Nerve._build()` to change edge weights:
```python
class CustomNerve(Nerve):
    def _build(self):
        # Your weight function
        ...
```

### Alternative Homology

Replace `nontrivial_h1_chords()` for different cycle selection:
```python
def custom_chord_selection(tree, nerve, triangles):
    # Your algorithm
    ...
```

## Usage Examples

```python
from catbp.topology.structure import FactorGraphStructure
from catbp.topology.nerve import Nerve
from catbp.topology.simplicial import build_2_skeleton_from_nerve_graph
from catbp.topology.homology import nontrivial_h1_chords
import networkx as nx

# Build structure
struct = FactorGraphStructure()
struct.add_variable("A", 2)
struct.add_variable("B", 2)
struct.add_variable("C", 2)
struct.add_factor("f1", ["A", "B"])
struct.add_factor("f2", ["B", "C"])
struct.add_factor("f3", ["A", "C"])

# Build nerve
nerve = Nerve(struct)

# Build 2-skeleton
scopes = {f: struct.get_scope(f) for f in struct.all_factors()}
sk2 = build_2_skeleton_from_nerve_graph(nerve.g, scopes)

# Find essential chords
tree = nx.maximum_spanning_tree(nerve.g, weight="weight")
chords = nontrivial_h1_chords(tree=tree, nerve=nerve.g, triangles=sk2.triangles)
```
