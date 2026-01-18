# Testing Philosophy

This document describes the testing approach for CatBP, focusing on correctness verification rather than coverage metrics.

## 1. What Correctness Means

### 1.1 Formal Correctness

The system is correct iff for all valid inputs:

1. **Partition function**: `Z_computed = Z_brute`
2. **Marginals**: Computed marginals equal true marginals
3. **Consistency**: Beliefs are globally consistent
4. **Determinism**: Same inputs produce same outputs

### 1.2 Operational Correctness

The implementation is operationally correct iff:

1. No runtime errors for valid inputs
2. Graceful handling of edge cases
3. Predictable behavior for invalid inputs
4. Memory bounded proportional to problem size

## 2. Algebraic Tests

### 2.1 Semiring Axiom Tests

Every semiring must satisfy:

```python
def test_semiring_axioms(sr):
    a, b, c = sample_values(sr)
    
    # Additive identity
    assert_equal(sr.add(a, sr.zero), a)
    
    # Multiplicative identity
    assert_equal(sr.mul(a, sr.one), a)
    
    # Additive commutativity
    assert_equal(sr.add(a, b), sr.add(b, a))
    
    # Multiplicative commutativity
    assert_equal(sr.mul(a, b), sr.mul(b, a))
    
    # Additive associativity
    assert_equal(sr.add(sr.add(a, b), c), sr.add(a, sr.add(b, c)))
    
    # Multiplicative associativity
    assert_equal(sr.mul(sr.mul(a, b), c), sr.mul(a, sr.mul(b, c)))
    
    # Distributivity
    assert_equal(sr.mul(a, sr.add(b, c)), 
                 sr.add(sr.mul(a, b), sr.mul(a, c)))
    
    # Annihilation
    assert_equal(sr.mul(a, sr.zero), sr.zero)
```

### 2.2 Vectorized Operation Tests

Verify vectorized operations match scalar:

```python
def test_vectorized_consistency(sr):
    # add_reduce
    arr = np.array([[1, 2], [3, 4]])
    scalar_sum = sr.add(sr.add(arr[0,0], arr[0,1]), 
                        sr.add(arr[1,0], arr[1,1]))
    vector_sum = sr.add_reduce(arr, axis=None)
    assert_close(scalar_sum, vector_sum)
    
    # mul (elementwise)
    a = np.array([1, 2])
    b = np.array([3, 4])
    scalar_prod = np.array([sr.mul(a[0], b[0]), sr.mul(a[1], b[1])])
    vector_prod = sr.mul(a, b)
    assert_array_equal(scalar_prod, vector_prod)
```

### 2.3 Section Operation Tests

```python
def test_section_star():
    # f ⋆ g = g ⋆ f (commutativity)
    fg = f.star(g)
    gf = g.star(f)
    assert_array_equal(fg.data, gf.data)
    
    # (f ⋆ g) ⋆ h = f ⋆ (g ⋆ h) (associativity)
    fg_h = f.star(g).star(h)
    f_gh = f.star(g.star(h))
    assert_array_close(fg_h.data, f_gh.data)

def test_section_restrict():
    # ρ(f ⋆ g) = f ⋆ ρ(g) when f doesn't depend on eliminated
    joint = f.star(g)
    restrict_joint = joint.restrict(f.domain)
    restrict_g = g.restrict(tuple(set(g.domain) & set(f.domain)))
    f_star_restrict = f.star(restrict_g)
    assert_array_close(restrict_joint.data, f_star_restrict.data)
```

## 3. Topological Tests

### 3.1 Nerve Construction Tests

```python
def test_nerve_correctness():
    struct = create_test_structure()
    nerve = Nerve(struct)
    
    # Every edge has non-empty interface
    for u, v in nerve.edges():
        I = nerve.interface(u, v)
        assert len(I) > 0
    
    # Interface is intersection of scopes
    for u, v in nerve.edges():
        I = nerve.interface(u, v)
        Su = set(struct.get_scope(u))
        Sv = set(struct.get_scope(v))
        assert set(I) == Su & Sv
```

### 3.2 Backbone Tests

```python
def test_backbone_is_spanning_tree():
    nerve = create_test_nerve()
    T = choose_backbone_tree(nerve.g)
    
    # Spanning: same nodes
    assert set(T.nodes()) == set(nerve.g.nodes())
    
    # Tree: n-1 edges, connected, acyclic
    n = len(T.nodes())
    assert len(T.edges()) == n - 1
    assert nx.is_connected(T)
    assert nx.is_tree(T)
```

### 3.3 Holonomy Tests

```python
def test_trivial_holonomy_on_tree():
    # Tree factor graph has trivial holonomy
    fg = create_tree_factor_graph()
    result = run_hatcc(fg)
    
    # Should have no mode variables
    assert len(result.plan.mode_sizes) == 0

def test_holonomy_fixed_point():
    # Holonomy diagonal feasibility
    H = compute_holonomy(...)
    for u in range(H.shape[0]):
        if mapping[u] >= 0:  # Valid mode
            assert H[u, u] > 0  # Fixed point exists
```

### 3.4 Homology Tests

```python
def test_h1_chords_count():
    # Euler characteristic: χ = V - E + F
    # For nerve: dim(H₁) = |E| - |V| + 1 - |triangles_killing_cycles|
    nerve = create_test_nerve()
    tree = choose_backbone_tree(nerve.g)
    chords = nontrivial_h1_chords(tree=tree, nerve=nerve.g, triangles=sk2.triangles)
    
    # Bound check
    assert len(chords) <= len(nerve.g.edges()) - len(nerve.g.nodes()) + 1
```

## 4. Runtime Tests

### 4.1 Safety Tests

```python
def test_empty_factor_graph():
    result = run_hatcc({}, {})
    assert result.Z == 1.0  # Empty product

def test_single_variable():
    result = run_hatcc({"X": 2}, {"f": (("X",), np.array([0.3, 0.7]))})
    assert np.isclose(result.Z, 1.0)

def test_unsat():
    # All-zero factor
    result = run_hatcc({"X": 2}, {"f": (("X",), np.array([0.0, 0.0]))})
    assert result.Z == 0.0
```

### 4.2 Determinism Tests

```python
def test_determinism():
    args = (var_domains, factors)
    
    result1 = run_hatcc(*args)
    result2 = run_hatcc(*args)
    
    assert result1.Z == result2.Z
    for key in result1.slots:
        arr1, _ = result1.slots[key]
        arr2, _ = result2.slots[key]
        assert np.array_equal(arr1, arr2)
```

### 4.3 Scheduling Tests

```python
def test_postorder_correct():
    plan = compile(...)
    
    # Every child appears before parent
    seen = set()
    for node in plan.postorder:
        for child in plan.nodes[node].children:
            assert child in seen
        seen.add(node)

def test_preorder_correct():
    plan = compile(...)
    
    # Every parent appears before child
    seen = set()
    for node in plan.preorder:
        if plan.nodes[node].parent is not None:
            assert plan.nodes[node].parent in seen
        seen.add(node)
```

## 5. Known Gaps

### 5.1 Not Tested

1. **Very large graphs**: Memory/time limits prevent exhaustive testing
2. **Numerical precision**: Float comparison uses tolerances
3. **Platform differences**: Rounding may vary across systems

### 5.2 Assumptions

1. **numpy correctness**: We trust numpy operations
2. **scipy correctness**: We trust scipy.sparse and connected_components
3. **networkx correctness**: We trust graph algorithms

### 5.3 Edge Cases

1. **Disconnected graphs**: Each component computed separately
2. **Self-loops in nerve**: Not possible by construction
3. **Degenerate factors**: Single-element scopes handled

## 6. Test Organization

### 6.1 Test Files

| File | Coverage |
|------|----------|
| `test_semiring.py` | Semiring axioms, operations |
| `test_section.py` | Section star, restrict, support |
| `test_topology.py` | Nerve, backbone, cycles |
| `test_holonomy_modes.py` | Holonomy, mode quotient |
| `test_selectors.py` | Mode selector construction |
| `test_vm.py` | VM operations, kernels |
| `test_solver.py` | End-to-end correctness |

### 6.2 Test Fixtures

```python
@pytest.fixture
def simple_chain():
    """A -- B -- C chain"""
    return {
        "var_domains": {"A": 2, "B": 2, "C": 2},
        "factors": {
            "f_A": (("A",), np.array([0.6, 0.4])),
            "f_AB": (("A", "B"), np.array([[0.9, 0.1], [0.2, 0.8]])),
            "f_BC": (("B", "C"), np.array([[0.3, 0.7], [0.5, 0.5]])),
        }
    }

@pytest.fixture
def grid_2x2():
    """2x2 Ising grid"""
    ...

@pytest.fixture
def frustrated_cycle():
    """4-cycle with frustrated constraints"""
    ...
```

### 6.3 Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_solver.py -v

# With coverage
pytest tests/ --cov=catbp --cov-report=html
```

## 7. Verification Strategy

### 7.1 Brute Force Comparison

For small instances (≤ 10 variables), compare against brute force:

```python
def brute_force_Z(var_domains, factors):
    Z = 0.0
    for config in enumerate_all_configs(var_domains):
        weight = 1.0
        for name, (scope, tensor) in factors.items():
            idx = tuple(config[v] for v in scope)
            weight *= tensor[idx]
        Z += weight
    return Z
```

### 7.2 Junction Tree Comparison

For medium instances, compare against junction tree:

```python
def junction_tree_reference(var_domains, factors):
    # Use external library or manual implementation
    ...
```

### 7.3 Property-Based Testing

Use hypothesis for random instances:

```python
from hypothesis import given, strategies as st

@given(st.integers(2, 5), st.integers(2, 4))
def test_random_chain(n_vars, domain_size):
    fg = random_chain(n_vars, domain_size)
    Z_catbp = run_hatcc(fg)
    Z_brute = brute_force(fg)
    assert np.isclose(Z_catbp, Z_brute, rtol=1e-10)
```

## 8. Regression Testing

### 8.1 Known-Good Values

Store results for regression:

```python
GOLDEN_VALUES = {
    "simple_chain": {"Z": 1.234567},
    "grid_2x2": {"Z": 8.765432},
    ...
}

def test_regression_simple_chain(simple_chain):
    result = run_hatcc(**simple_chain)
    assert np.isclose(result.Z, GOLDEN_VALUES["simple_chain"]["Z"])
```

### 8.2 Example Outputs

Document expected outputs:

```python
def test_example_from_paper():
    """Test case from Section 5 of arXiv:2601.04456"""
    ...
    assert np.isclose(result.Z, expected_Z)
```
