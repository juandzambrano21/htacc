# solver/ and hatcc

## Intent

The top-level solver modules (`solver.py` and `hatcc.py`) provide the primary user-facing API for running belief propagation on factor graphs. They orchestrate the full pipeline from problem specification to result extraction.

## Public Surface

### From `hatcc.py`

| Export | Type | Description |
|--------|------|-------------|
| `hatcc_solve()` | function | Full solver with all options |
| `compute_partition_function()` | function | Compute Z only |
| `compute_marginals()` | function | Compute variable marginals |

### From `solver.py`

| Export | Type | Description |
|--------|------|-------------|
| `run_hatcc()` | function | Core solver implementation |
| `SolverResult` | dataclass | Result container (Z, slots, plan) |

## API Reference

### hatcc_solve

```python
def hatcc_solve(
    var_domains: Dict[str, int],
    factors: Dict[str, Tuple[Tuple[str, ...], np.ndarray]],
    *,
    semiring: str = "prob",
    root: Optional[str] = None,
    support_eps: float = 0.0,
) -> SolverResult:
    """
    Solve a factor graph using HATCC.
    
    Args:
        var_domains: Map from variable name to domain size
        factors: Map from factor name to (scope, tensor)
        semiring: "prob", "logprob", or "sat"
        root: Optional root factor name
        support_eps: Epsilon for support computation
        
    Returns:
        SolverResult with partition function and beliefs
    """
```

### compute_partition_function

```python
def compute_partition_function(
    var_domains: Dict[str, int],
    factors: Dict[str, Tuple[Tuple[str, ...], np.ndarray]],
    **kwargs
) -> float:
    """Compute the partition function Z."""
```

### compute_marginals

```python
def compute_marginals(
    var_domains: Dict[str, int],
    factors: Dict[str, Tuple[Tuple[str, ...], np.ndarray]],
    variables: Optional[list] = None,
    **kwargs
) -> Dict[str, np.ndarray]:
    """Compute marginal distributions for variables."""
```

### SolverResult

```python
@dataclass
class SolverResult:
    Z: float          # Partition function
    slots: Dict       # All computed tensors
    plan: BPPlan      # Execution plan
```

## Internal Details

### Solver Pipeline

1. **Semiring selection**: Choose runtime based on `semiring_name`
2. **Compilation**: `TopologyCompiler.compile()` produces plan and stores
3. **Provider creation**: Build factor/selector loaders
4. **Program emission**: Generate upward, downward, finalize programs
5. **Execution**: Run all programs on VM
6. **Result extraction**: Extract Z from slots

### Semiring Options

| Name | Semiring | Use Case |
|------|----------|----------|
| `"prob"` | Probability | Standard inference |
| `"logprob"` | Log-probability | Numerically stable |
| `"sat"` | Boolean | Satisfiability checking |

## Invariants

1. **Determinism**: Same inputs → same outputs
2. **Correctness**: Z matches brute-force on small instances
3. **Completeness**: All variables have marginals

## Extension Points

### Custom Semiring

Add to `run_hatcc()`:
```python
sr_map = {
    ...,
    "my_semiring": my_semiring_runtime(),
}
```

### Custom Compilation

Replace `TopologyCompiler` for modified behavior:
```python
def my_solver(var_domains, factors, ...):
    compiler = MyTopologyCompiler()
    plan, ... = compiler.compile(...)
    ...
```

## Usage Examples

### Basic Usage

```python
from catbp import hatcc_solve, compute_partition_function, compute_marginals
import numpy as np

# Define factor graph
var_domains = {"A": 2, "B": 2, "C": 2}
factors = {
    "f_A": (("A",), np.array([0.6, 0.4])),
    "f_AB": (("A", "B"), np.array([[0.9, 0.1], [0.2, 0.8]])),
    "f_BC": (("B", "C"), np.array([[0.3, 0.7], [0.5, 0.5]])),
}

# Full solve
result = hatcc_solve(var_domains, factors)
print(f"Z = {result.Z}")

# Just partition function
Z = compute_partition_function(var_domains, factors)

# Just marginals
marginals = compute_marginals(var_domains, factors)
for var, prob in marginals.items():
    print(f"P({var}) = {prob}")
```

### Log-Space for Numerical Stability

```python
# Use log-probability semiring
result = hatcc_solve(var_domains, factors, semiring="logprob")
print(f"log(Z) = {result.Z}")  # Result is log(Z)
```

### SAT Mode

```python
# Check satisfiability
factors = {
    "clause1": (("X", "Y"), np.array([[0, 1], [1, 1]])),  # X OR Y
    "clause2": (("Y", "Z"), np.array([[1, 0], [1, 1]])),  # NOT Y OR Z
}
result = hatcc_solve(var_domains, factors, semiring="sat")
print(f"Satisfiable: {result.Z > 0}")
```

### Accessing Beliefs

```python
result = hatcc_solve(var_domains, factors)

# Get belief at specific factor
bel_data, bel_spec = result.slots[("bel", factor_id)]

# Get message
msg_data, msg_spec = result.slots[("msg", from_node, to_node)]
```
