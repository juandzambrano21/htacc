# Runtime Execution Model

This document specifies the runtime layer: scheduling, evidence handling, and execution orchestration.

## 1. Scheduler Semantics

### 1.1 Program Execution

The scheduler executes programs on the virtual machine:

```python
def run_program(
    program: Program,
    sr: VMSemiringRuntime,
    factor_provider: Callable[[int], Tuple[np.ndarray, TensorSpec]],
    selector_provider: Callable[[int], Tuple[np.ndarray, TensorSpec]],
    *,
    slots: Optional[Dict] = None,
) -> Dict[object, Tuple[np.ndarray, TensorSpec]]:
    vm = VirtualMachine(sr)
    if slots is not None:
        vm.slots.data.update(slots)
    vm.run(program, factor_provider=factor_provider, selector_provider=selector_provider)
    return vm.slots.data
```

### 1.2 Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    EXECUTION FLOW                            │
│                                                              │
│  1. Create VM with semiring runtime                         │
│  2. Initialize slots (if continuing from previous pass)     │
│  3. Execute program instructions sequentially               │
│  4. Return final slot contents                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Multi-Pass Execution

The solver runs three programs:

```python
# Pass 1: Upward (postorder)
slots = run_program(prog_up, sr, factor_provider, selector_provider)

# Pass 2: Downward (preorder)
slots = run_program(prog_down, sr, factor_provider, selector_provider, slots=slots)

# Pass 3: Finalize (compute Z)
slots = run_program(prog_fin, sr, factor_provider, selector_provider, slots=slots)
```

Slots persist across passes, allowing messages computed in the upward pass to be used in the downward pass.

## 2. Evidence Lifecycle

### 2.1 Evidence as Constraints

Evidence is incorporated as multiplicative constraint factors:

```python
def unary_evidence(
    var: int,
    allowed: List[int],
    domains: Dict[int, int],
    semiring
) -> NamedTensor:
    d = domains[var]
    data = np.full((d,), semiring.zero, dtype=np.float32)
    for a in allowed:
        data[a] = semiring.one
    return NamedTensor((var,), data, semiring)
```

**Semantics**: Evidence factor is 1 for allowed values, 0 for disallowed.

### 2.2 Observation Constraint

Hard observation (single value):

```python
def observe_variable(
    var: int,
    value: int,
    domains: Dict[int, int],
    semiring
) -> NamedTensor:
    return unary_evidence(var, [value], domains, semiring)
```

### 2.3 General Constraints

For k-ary constraints:

```python
def interface_mask(
    vars_: Tuple[int, ...],
    mask: np.ndarray,
    semiring
) -> NamedTensor:
    data = mask.astype(np.float32)
    return NamedTensor(vars_, data, semiring)
```

### 2.4 Evidence Integration

Evidence factors are added before compilation:

```python
# Add evidence as additional factors
factors["obs_X"] = ((X,), observe_variable(X, observed_value, domains, sr))

# Compile with evidence included
plan = compiler.compile(var_domains, factors, ...)
```

The compiler treats evidence factors identically to other factors.

## 3. Execution Order Constraints

### 3.1 Data Dependencies

Instructions must respect data dependencies:

```
LOAD_FACTOR(f)  →  CONTRACT(...)  →  ELIMINATE(...)  →  STORE_SLOT(...)
                              ↑                              ↓
                         LOAD_SLOT(child_msg)           LOAD_SLOT(next_use)
```

### 3.2 Upward Pass Order

Postorder traversal ensures children processed before parents:

```
For tree:        Process order:
    R                L1
   / \               L2
  A   B              A
 / \   \             L3
L1 L2   L3           B
                     R
```

### 3.3 Downward Pass Order

Preorder traversal ensures parents processed before children:

```
Process order:
    R
    A
    L1
    L2
    B
    L3
```

### 3.4 Invariants

1. **Upward**: Message `msg[c→p]` is stored before node `p` is processed
2. **Downward**: Belief `bel[p]` is computed before message `msg[p→c]` is needed
3. **Finalize**: Root belief exists before Z computation

## 4. Failure Handling

### 4.1 Execution Failures

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Missing slot | KeyError | None (programmer error) |
| Missing register | KeyError | None (programmer error) |
| Alignment mismatch | ValueError | None (compiler bug) |
| Memory exhaustion | MemoryError | None (problem too large) |

### 4.2 Numerical Failures

| Failure | Detection | Handling |
|---------|-----------|----------|
| Z = 0 | Check after finalize | Report UNSAT |
| NaN in computation | isnan check | Propagates to output |
| Overflow | isinf check | May indicate log-space needed |
| Underflow | Z → 0 | Use log-probability semiring |

### 4.3 Graceful Degradation

```python
def finalize_Z_from_slots(slots) -> float:
    z_arr, _ = slots[("Z",)]
    if np.ndim(z_arr) != 0:
        z_arr = np.asarray(z_arr).reshape(())
    z = float(z_arr)
    
    if z == 0:
        # UNSAT: no valid configurations
        pass
    elif np.isnan(z):
        # Numerical error
        pass
    elif np.isinf(z):
        # Overflow
        pass
    
    return z
```

## 5. Slot Key Schema

### 5.1 Message Slots

```python
("msg", source_node, target_node)
```

- `source_node`: Node sending the message
- `target_node`: Node receiving the message
- Stored after ELIMINATE in sender
- Loaded before CONTRACT in receiver

### 5.2 Belief Slots

```python
("bel", node)
```

- Stored after full contraction at node
- Used for marginal extraction
- Root belief used for Z computation

### 5.3 Partition Function Slot

```python
("Z",)
```

- Single-element tuple
- Scalar value (0-dimensional tensor)
- Final result of computation

### 5.4 Slot Lifecycle

```
Upward pass:
  msg[leaf→parent]     created
  msg[parent→grandp]   created
  bel[root]            created

Downward pass:
  bel[node]            updated for all nodes
  msg[parent→child]    created

Finalize:
  Z                    created
```

## 6. Provider Interface

### 6.1 Factor Provider

```python
def factor_provider(factor_id: int) -> Tuple[np.ndarray, TensorSpec]:
    arr, spec = factor_store[factor_id]
    return arr.astype(sr.dtype, copy=False), spec
```

**Requirements**:
- Return factor tensor in correct shape
- Axes in canonical (scope) order
- Type compatible with semiring

### 6.2 Selector Provider

```python
def selector_provider(selector_id: int) -> Tuple[np.ndarray, TensorSpec]:
    payload = selector_store[selector_id]
    return vm_load_selector(
        payload,
        domains,
        mode_sizes,
        dtype=sr.dtype,
        one_val=sr.one(()).item(),
        zero_val=sr.zero(()).item(),
    )
```

**Requirements**:
- Return selector tensor in correct shape
- Geometric axes for interface, topological axis for mode
- Values are semiring one/zero

## 7. Result Extraction

### 7.1 Partition Function

```python
Z = finalize_Z_from_slots(slots)
```

### 7.2 Marginals

```python
def marginal_from_factor_belief(
    belief_data: np.ndarray,
    belief_spec: TensorSpec,
    keep_geo_vars: Tuple[int, ...],
    sr: VMSemiringRuntime
) -> np.ndarray:
    keep_keys = tuple((AxisType.GEOMETRIC, v) for v in keep_geo_vars)
    out, out_spec = vm_eliminate_keep(belief_data, belief_spec, keep_keys, sr)
    
    if sr.supports_normalize() and out.ndim > 0:
        out = sr.normalize(out, axis=tuple(range(out.ndim)))
    return out
```

### 7.3 Full Beliefs

Beliefs are available in slots for any factor:

```python
bel_data, bel_spec = slots[("bel", factor_id)]
```

## 8. High-Level Solver Interface

### 8.1 run_hatcc

```python
def run_hatcc(
    var_domains_named: Dict[str, int],
    factors_named: Dict[str, Tuple[Tuple[str, ...], np.ndarray]],
    semiring_name: str = "prob",
    support_predicate=None,
    root_factor_name: Optional[str] = None
) -> SolverResult:
    # 1. Select semiring
    sr = {"prob": vm_prob_semiring, "logprob": vm_logprob_semiring, "sat": vm_sat_semiring}[semiring_name]()
    
    # 2. Compile
    plan, factor_store, selector_store, domains, mode_sizes = TopologyCompiler().compile(...)
    
    # 3. Build providers
    factor_provider = lambda fid: (factor_store[fid][0].astype(sr.dtype), factor_store[fid][1])
    selector_provider = lambda sid: vm_load_selector(selector_store[sid], ...)
    
    # 4. Emit programs
    prog_up = emit_upward_program(plan)
    prog_down = emit_downward_program_prefixsuffix(plan)
    prog_fin = emit_finalize_program(plan.root)
    
    # 5. Execute
    slots = run_program(prog_up, sr, factor_provider, selector_provider)
    slots = run_program(prog_down, sr, factor_provider, selector_provider, slots=slots)
    slots = run_program(prog_fin, sr, factor_provider, selector_provider, slots=slots)
    
    # 6. Extract result
    Z = finalize_Z_from_slots(slots)
    
    return SolverResult(Z=Z, slots=slots, plan=plan)
```

### 8.2 SolverResult

```python
@dataclass
class SolverResult:
    Z: float          # Partition function
    slots: Dict       # All computed tensors
    plan: BPPlan      # Execution plan
```

## 9. Concurrent Execution (Future)

### 9.1 Current Limitation

The current implementation is sequential. Each instruction executes completely before the next.

### 9.2 Parallelization Opportunities

1. **Independent loads**: Multiple LOAD_FACTOR/LOAD_SELECTOR can run in parallel
2. **Sibling contractions**: Child message computations are independent
3. **Prefix-suffix**: Prefix and suffix arrays can be computed in parallel

### 9.3 Thread Safety Requirements

For future parallel execution:
- RegisterBank would need locking
- SlotStore would need atomic operations
- Providers must be thread-safe
