# Extensions Guide

This document provides guidance on extending CatBP safely and correctly.

## 1. How to Add a Semiring

### 1.1 Requirements

A new semiring must satisfy the algebraic axioms:

1. `(R, ⊕, 0)` is a commutative monoid
2. `(R, ⊗, 1)` is a commutative monoid
3. `a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)` (distributivity)
4. `a ⊗ 0 = 0` (annihilation)

### 1.2 Scalar Semiring Implementation

Create a dataclass implementing the Semiring protocol:

```python
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass(frozen=True)
class MySemiring:
    zero: Any = ...  # Additive identity
    one: Any = ...   # Multiplicative identity
    
    def add(self, a: Any, b: Any) -> Any:
        """Semiring addition (⊕)"""
        ...
    
    def mul(self, a: Any, b: Any) -> Any:
        """Semiring multiplication (⊗)"""
        ...
    
    def is_zero(self, a: Any) -> bool:
        """Check if value equals zero"""
        ...
    
    def add_reduce(self, x: np.ndarray, axis=None) -> np.ndarray:
        """Vectorized ⊕-reduction over axes"""
        ...
    
    def mul_reduce(self, xs: list[np.ndarray]) -> np.ndarray:
        """⊗-product of multiple arrays"""
        ...
    
    def normalize(self, x: np.ndarray, axis=None) -> np.ndarray:
        """Optional normalization (identity if not applicable)"""
        return x
```

### 1.3 VM Runtime Implementation

Create a factory function for the VM runtime:

```python
from catbp.vm.semiring import VMSemiringRuntime

def my_semiring_runtime(dtype=np.float64) -> VMSemiringRuntime:
    dt = np.dtype(dtype)
    
    def _add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Vectorized ⊕
        ...
    
    def _mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Vectorized ⊗
        ...
    
    def _add_reduce(x: np.ndarray, axis) -> np.ndarray:
        # ⊕-reduction over axes
        ...
    
    def _normalize(x: np.ndarray, axis=None) -> np.ndarray:
        # Optional normalization
        ...
    
    return VMSemiringRuntime(
        name="MY_SEMIRING",
        dtype=dt,
        mul=_mul,
        add_reduce=_add_reduce,
        one=lambda shape: np.full(shape, ONE_VALUE, dtype=dt),
        zero=lambda shape: np.full(shape, ZERO_VALUE, dtype=dt),
        normalize=_normalize,  # or None if not applicable
    )
```

### 1.4 Integration

Add to the solver's semiring selection:

```python
# In solver.py or hatcc.py
sr_map = {
    "prob": vm_prob_semiring(np.float64),
    "logprob": vm_logprob_semiring(np.float64),
    "sat": vm_sat_semiring(),
    "my_semiring": my_semiring_runtime(),  # Add here
}
```

### 1.5 Testing

Write tests verifying:

```python
def test_semiring_axioms():
    sr = MySemiring()
    
    # Additive identity
    assert sr.add(a, sr.zero) == a
    
    # Multiplicative identity
    assert sr.mul(a, sr.one) == a
    
    # Commutativity
    assert sr.add(a, b) == sr.add(b, a)
    assert sr.mul(a, b) == sr.mul(b, a)
    
    # Associativity
    assert sr.add(sr.add(a, b), c) == sr.add(a, sr.add(b, c))
    assert sr.mul(sr.mul(a, b), c) == sr.mul(a, sr.mul(b, c))
    
    # Distributivity
    assert sr.mul(a, sr.add(b, c)) == sr.add(sr.mul(a, b), sr.mul(a, c))
    
    # Annihilation
    assert sr.mul(a, sr.zero) == sr.zero
```

### 1.6 Example: Min-Plus (Tropical) Semiring

```python
@dataclass(frozen=True)
class MinPlusSemiring:
    zero: float = np.inf   # min identity
    one: float = 0.0       # plus identity
    
    def add(self, a, b) -> float:  # ⊕ = min
        return min(float(a), float(b))
    
    def mul(self, a, b) -> float:  # ⊗ = +
        return float(a) + float(b)
    
    def is_zero(self, a) -> bool:
        return np.isinf(float(a)) and float(a) > 0
    
    def add_reduce(self, x, axis=None):
        return np.min(x, axis=axis)
    
    def mul_reduce(self, xs):
        out = xs[0].copy()
        for t in xs[1:]:
            out = out + t
        return out
    
    def normalize(self, x, axis=None):
        return x  # No normalization for min-plus
```

## 2. How to Add Topology Extensions

### 2.1 Custom Nerve Construction

To modify how the nerve is built:

```python
from catbp.topology.nerve import Nerve

class CustomNerve(Nerve):
    def _build(self):
        # Custom edge weight computation
        for f in self.struct.factors:
            self.g.add_node(f)
        
        for v, fs in self.struct.var_to_factors.items():
            for i in range(len(fs)):
                for j in range(i + 1, len(fs)):
                    a, b = fs[i], fs[j]
                    I = self.struct.interface(a, b)
                    if I:
                        w = self.custom_weight(I)  # Custom weight
                        self.g.add_edge(a, b, interface=I, weight=w)
    
    def custom_weight(self, interface):
        # Your weight function
        ...
```

### 2.2 Custom Backbone Selection

To use different spanning tree algorithms:

```python
import networkx as nx

def custom_backbone(nerve: nx.Graph) -> nx.Graph:
    # Example: minimum spanning tree instead of maximum
    return nx.minimum_spanning_tree(nerve, weight="weight")
    
    # Example: random spanning tree
    # return nx.random_spanning_tree(nerve)
```

### 2.3 Custom Holonomy Computation

To modify holonomy analysis:

```python
from catbp.compiler.holonomy import compute_holonomy_boolean

def custom_holonomy(steps, factor_supports, domains):
    H = compute_holonomy_boolean(steps, factor_supports, domains)
    
    # Custom post-processing
    # Example: threshold small entries
    H.data[H.data < threshold] = 0
    H.eliminate_zeros()
    
    return H
```

## 3. How to Add Kernels

### 3.1 New VM Operation

1. Add opcode to `ir/ops.py`:

```python
class OpCode(Enum):
    ...
    MY_OP = 10  # New operation
```

2. Add instruction handling in `vm/vm.py`:

```python
def _execute(self, ins, ...):
    ...
    elif ins.op == OpCode.MY_OP:
        # Get inputs
        arr, spec = self.regs.get(ins.args['src'])
        
        # Perform operation
        result = my_kernel(arr, spec, self.sr, **ins.args)
        
        # Store result
        self.regs.put(ins.dst, result, out_spec)
```

3. Implement kernel in `vm/kernels.py`:

```python
def my_kernel(arr, spec, sr, **kwargs):
    # Implementation
    ...
    return result
```

### 3.2 Custom Contraction Strategy

To modify how tensors are contracted:

```python
def custom_contract(inputs, out_spec, sr):
    # Example: einsum-based contraction
    subscripts = build_einsum_subscripts(inputs, out_spec)
    arrays = [arr for arr, _ in inputs]
    return np.einsum(subscripts, *arrays)
```

### 3.3 Custom Elimination Strategy

To modify marginalization:

```python
def custom_eliminate(arr, in_spec, keep_keys, sr):
    # Example: sparse elimination for mostly-zero tensors
    if is_sparse(arr):
        return sparse_eliminate(arr, in_spec, keep_keys, sr)
    else:
        return vm_eliminate_keep(arr, in_spec, keep_keys, sr)
```

## 4. Invariants That Must Be Preserved

### 4.1 Axis Ordering

**Invariant**: Axes are always in canonical order within tensor specs.

```python
# CORRECT
spec = TensorSpec(axes=sorted(axes, key=lambda a: (a.kind.value, a.id)))

# WRONG
spec = TensorSpec(axes=axes)  # Unsorted
```

### 4.2 AxisKey Uniqueness

**Invariant**: AxisKeys must be unique within a TensorSpec.

```python
# CORRECT
axes = [
    Axis(0, GEOMETRIC, 2),
    Axis(1, GEOMETRIC, 3),
    Axis(0, TOPOLOGICAL, 4),  # Different type, OK
]

# WRONG
axes = [
    Axis(0, GEOMETRIC, 2),
    Axis(0, GEOMETRIC, 3),  # Duplicate key!
]
```

### 4.3 Support Computation

**Invariant**: Holonomy computation always uses SAT (Boolean) semiring.

```python
# CORRECT
support = factor_support(phi)  # Returns SATSemiring section
H = compute_holonomy_boolean(steps, supports, domains)

# WRONG
H = compute_holonomy(steps, prob_factors, domains)  # Using value semiring
```

### 4.4 Message Consistency

**Invariant**: Messages contain exactly the separator axes.

```python
# Separator for edge (a, b)
keep_keys = keep_axis_keys_on_edge[(a, b)]

# Message must have exactly these axes
msg = eliminate(belief, keep_keys)
assert set(msg.spec.keys) == set(keep_keys)
```

### 4.5 Mode Selector Structure

**Invariant**: Selectors have geometric axes for interface, then topological axis for mode.

```python
# CORRECT
selector_spec = TensorSpec(axes=(
    Axis(v1, GEOMETRIC, d1),
    Axis(v2, GEOMETRIC, d2),
    Axis(mode_id, TOPOLOGICAL, num_modes),
))

# WRONG
selector_spec = TensorSpec(axes=(
    Axis(mode_id, TOPOLOGICAL, num_modes),  # Mode first
    Axis(v1, GEOMETRIC, d1),
    Axis(v2, GEOMETRIC, d2),
))
```

## 5. Testing Extensions

### 5.1 Correctness Tests

Compare against brute-force:

```python
def test_extension_correctness():
    # Small problem
    var_domains = {"A": 2, "B": 2, "C": 2}
    factors = {...}
    
    # Your extension
    Z_ext = run_with_extension(var_domains, factors)
    
    # Brute force
    Z_brute = brute_force_enumerate(var_domains, factors)
    
    assert np.isclose(Z_ext, Z_brute, rtol=1e-10)
```

### 5.2 Invariant Tests

Check invariants are preserved:

```python
def test_invariants():
    # Compile with extension
    plan = compile_with_extension(...)
    
    # Check axis ordering
    for spec in plan.belief_spec.values():
        keys = spec.keys
        assert keys == tuple(sorted(keys, key=lambda k: (k[0].value, k[1])))
    
    # Check mode selectors
    for sel_id, payload in selector_store.items():
        arr, spec = load_selector(payload)
        geo_axes = [a for a in spec.axes if a.kind == GEOMETRIC]
        topo_axes = [a for a in spec.axes if a.kind == TOPOLOGICAL]
        assert len(topo_axes) == 1
        assert spec.axes[-1].kind == TOPOLOGICAL  # Mode last
```

### 5.3 Edge Case Tests

Test boundary conditions:

```python
def test_edge_cases():
    # Empty factor graph
    # Single variable
    # Disconnected components
    # All-zero factors (UNSAT)
    # Trivial holonomy (tree)
    # Maximum holonomy (all modes distinct)
```

## 6. Documentation Requirements

When adding extensions:

1. **Docstring**: Explain purpose and semantics
2. **Type hints**: Provide complete type annotations
3. **Invariants**: Document what must hold
4. **Examples**: Show usage
5. **Tests**: Provide comprehensive tests

```python
def my_extension(
    arg1: Type1,
    arg2: Type2,
    *,
    option: bool = False,
) -> ReturnType:
    """
    Brief description.
    
    Detailed explanation of what this does and why.
    
    Args:
        arg1: Description
        arg2: Description
        option: Description
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When arg1 is invalid
    
    Invariants:
        - Invariant 1
        - Invariant 2
    
    Example:
        >>> result = my_extension(x, y)
        >>> print(result)
    """
    ...
```
