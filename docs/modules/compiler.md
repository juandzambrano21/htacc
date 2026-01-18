# compiler/

## Intent

The `compiler/` module transforms factor graph topology into an executable belief propagation plan. It orchestrates backbone selection, cycle analysis, holonomy computation, mode variable creation, and program emission.

## Public Surface

### From `topology_compiler.py`

| Export | Type | Description |
|--------|------|-------------|
| `TopologyCompiler` | class | Main compilation orchestrator |
| `compile()` | method | Factor graph → execution plan |

### From `backbone.py`

| Export | Type | Description |
|--------|------|-------------|
| `choose_backbone_tree()` | function | MST on nerve graph |
| `choose_backbone_tree_from_nerve()` | function | MST using Nerve object |
| `root_tree()` | function | Root tree at specified node |
| `compute_euler_tour()` | function | Compute DFS timestamps |
| `in_subtree()` | function | O(1) subtree membership query |
| `path_in_tree()` | function | Get path between nodes |

### From `cycles.py`

| Export | Type | Description |
|--------|------|-------------|
| `TransportStep` | dataclass | Single step in holonomy cycle |
| `HolonomyCycle` | dataclass | Complete typed cycle |
| `build_fundamental_cycle()` | function | Build cycle for chord |
| `select_chords_via_h1()` | function | Filter H₁-essential chords |
| `build_h1_cycles()` | function | Build cycles for essential chords |
| `chord_cycle_steps()` | function | Build cycle steps with integer IDs |

### From `transport.py`

| Export | Type | Description |
|--------|------|-------------|
| `tensor_support()` | function | Extract boolean support |
| `build_transport_kernel_boolean()` | function | Sparse transport matrix |
| `build_transport_kernel_support()` | function | Transport from NamedTensor |

### From `holonomy.py`

| Export | Type | Description |
|--------|------|-------------|
| `HolonomyModeSpec` | dataclass | Mode specification from holonomy |
| `compute_holonomy_boolean()` | function | Holonomy matrix computation |
| `quotient_scc_from_holonomy()` | function | SCC-based mode quotient |
| `analyze_cycle_support()` | function | Extract mode spec from cycle |

### From `modes.py`

| Export | Type | Description |
|--------|------|-------------|
| `ModeSpec` | dataclass | Mode variable specification |
| `to_selector_payload()` | method | Convert to VM payload |

### From `plan.py`

| Export | Type | Description |
|--------|------|-------------|
| `NodePlan` | dataclass | Per-node execution plan |
| `BPPlan` | dataclass | Complete BP execution plan |
| `build_bp_plan()` | function | Construct plan from components |

### From `emit.py`

| Export | Type | Description |
|--------|------|-------------|
| `emit_upward_program()` | function | Generate upward pass IR |
| `emit_downward_program_prefixsuffix()` | function | Generate downward pass IR |
| `emit_finalize_program()` | function | Generate Z computation IR |
| `run_upward_sweep()` | function | Direct Section-based execution |

### From `factors.py` and `selectors.py`

| Export | Type | Description |
|--------|------|-------------|
| `FactorDef` | dataclass | Factor definition |
| `make_factor_provider()` | function | Create factor loader |
| `SelectorDef` | dataclass | Selector definition |
| `make_selector_provider()` | function | Create selector loader |

## Internal Passes

### Pass 1: ID Assignment

Assign canonical integer IDs to variables and factors:
```python
var_id = {name: i for i, name in enumerate(sorted(var_names))}
fac_id = {name: i for i, name in enumerate(sorted(fac_names))}
```

### Pass 2: Support Extraction

Convert factors to boolean support for topology analysis:
```python
factor_supports[fid] = BoolSection(domain=scope, data=support_predicate(data))
```

### Pass 3: Nerve Construction

Build intersection graph:
```python
G.add_edge(a, b, interface=inter, weight=log_interface_size(inter))
```

### Pass 4: Backbone Selection

Maximum spanning tree:
```python
T = nx.maximum_spanning_tree(G, weight="weight")
```

### Pass 5: Rooting

Root tree at selected node, compute parent/children:
```python
parent, children = root_tree(T, root)
```

### Pass 6: Chord Analysis

For each non-tree edge:
1. Compute fundamental cycle
2. Check H₁ contribution
3. Compute holonomy matrix
4. Extract mode quotient

### Pass 7: Plan Assembly

Combine all information into BPPlan:
- Node plans with vars and anchored modes
- Belief specs with all axes
- Message keep-keys for each edge

## Invariants

1. **Backbone is spanning tree**: Connected, acyclic, covers all factors
2. **Mode ID isolation**: Mode IDs are in separate namespace from VarIDs
3. **Selector anchoring**: Each selector anchored at both chord endpoints
4. **Message completeness**: Every tree edge has defined keep-keys

## Extension Points

### Custom Backbone

Replace `choose_backbone_tree()`:
```python
def my_backbone(nerve):
    return my_spanning_tree_algorithm(nerve)
```

### Custom Holonomy

Modify `compute_holonomy_boolean()` for different transport semantics.

### Custom Plan Generation

Subclass `TopologyCompiler` to modify plan assembly.

## Usage Examples

```python
from catbp.compiler.topology_compiler import TopologyCompiler
import numpy as np

var_domains = {"A": 2, "B": 2, "C": 2}
factors = {
    "f1": (("A", "B"), np.array([[0.9, 0.1], [0.2, 0.8]])),
    "f2": (("B", "C"), np.array([[0.3, 0.7], [0.5, 0.5]])),
}

compiler = TopologyCompiler()
plan, factor_store, selector_store, domains, mode_sizes = compiler.compile(
    var_domains_named=var_domains,
    factors_named=factors,
    support_predicate=lambda x: x != 0,
)

# Access plan components
print(f"Root: {plan.root}")
print(f"Postorder: {plan.postorder}")
print(f"Preorder: {plan.preorder}")
```
