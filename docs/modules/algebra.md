# algebra/

## Intent

The `algebra/` module provides the foundational algebraic structures for belief propagation: semirings and semiring-valued sections. These abstractions enable parametric inference algorithms that work uniformly across different computational domains (probability, log-probability, Boolean satisfiability, counting).

## Public Surface

### From `semiring.py`

| Export | Type | Description |
|--------|------|-------------|
| `SATSemiring` | class | Boolean semiring (OR, AND) |
| `ProbSemiring` | class | Probability semiring (+, ×) |
| `LogProbSemiring` | class | Log-probability semiring (logsumexp, +) |
| `CountingSemiring` | class | Integer counting semiring (+, ×) |
| `SemiringRuntime` | class | Vectorized numpy-backed semiring |
| `sat_semiring()` | function | Factory for SAT runtime |
| `prob_semiring(dtype)` | function | Factory for probability runtime |
| `logprob_semiring(dtype)` | function | Factory for log-probability runtime |
| `counting_semiring(dtype)` | function | Factory for counting runtime |

### From `section.py`

| Export | Type | Description |
|--------|------|-------------|
| `Section` | class | Domain-ordered semiring tensor |
| `Section.unit()` | staticmethod | Create unit (all-ones) section |
| `Section.zero_section()` | staticmethod | Create zero section |
| `Section.star()` | method | Join-product on union domain |
| `Section.restrict()` | method | Marginalization to subset |
| `Section.normalize()` | method | Semiring-specific normalization |
| `Section.as_bool_support()` | method | Convert to boolean support |

### From `support.py`

| Export | Type | Description |
|--------|------|-------------|
| `factor_support()` | function | Extract boolean support from section |
| `build_factor_supports()` | function | Batch support extraction |
| `mode_mapping_from_holonomy()` | function | Compute mode quotient from holonomy |
| `filter_fixed_points_for_modes()` | function | Filter by diagonal feasibility |
| `mapping_array_to_dict()` | function | Convert mapping array to dict |

## Internal Details

### Numerical Stability

`_logsumexp()` implements numerically stable log-sum-exp:
```python
def _logsumexp(x, axis=None):
    m = np.max(x, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(x - m), axis=axis))
```

### Axis Handling

`_axis_tuple()` normalizes axis specifications:
- `None` → `None`
- `int` → `(int,)`
- `tuple` → `tuple`

### Support Computation

`factor_support()` handles various input types:
- Boolean arrays: pass through
- Float arrays: compare against zero (with optional epsilon)
- Log-space arrays: check `isfinite()`
- NaN handling: treat as zero

## Invariants

1. **Section domain uniqueness**: No duplicate variables in domain
2. **Section domain-data consistency**: `len(domain) == data.ndim`
3. **Canonical domain ordering**: Operations may reorder to sorted
4. **Semiring consistency**: Star operands must have same semiring

## Extension Points

### Adding a Semiring

1. Create dataclass implementing the `Semiring` protocol
2. Implement `add`, `mul`, `is_zero`, `add_reduce`, `mul_reduce`, `normalize`
3. Create factory function for `SemiringRuntime`
4. Register in solver's semiring map

### Custom Section Operations

Subclass `Section` to add operations:
```python
class ExtendedSection(Section):
    def custom_op(self, other):
        ...
```

## Usage Examples

```python
from catbp.algebra.semiring import ProbSemiring, prob_semiring
from catbp.algebra.section import Section
from catbp.algebra.support import factor_support

# Create a section
sr = ProbSemiring()
domain = ("A", "B")
data = np.array([[0.9, 0.1], [0.2, 0.8]])
section = Section(domain, data, sr)

# Star product
other = Section(("B", "C"), np.array([[0.3, 0.7], [0.5, 0.5]]), sr)
joined = section.star(other)  # Domain: (A, B, C)

# Marginalize
marginal = joined.restrict(("A",))

# Extract support
support = factor_support(section)  # Boolean SATSemiring section
```
