# runtime/

## Intent

The `runtime/` module orchestrates program execution and handles evidence incorporation. It provides the interface between compiled programs and the virtual machine.

## Public Surface

### From `schedule.py`

| Export | Type | Description |
|--------|------|-------------|
| `run_program()` | function | Execute program on VM |
| `finalize_Z_from_slots()` | function | Extract partition function from slots |

### From `evidence.py`

| Export | Type | Description |
|--------|------|-------------|
| `unary_evidence()` | function | Create unary constraint factor |
| `interface_mask()` | function | Create k-ary constraint factor |
| `observe_variable()` | function | Create observation constraint |

## Internal Details

### Program Execution

```python
def run_program(program, sr, factor_provider, selector_provider, *, slots=None):
    vm = VirtualMachine(sr)
    if slots is not None:
        vm.slots.data.update(slots)  # Continue from previous pass
    vm.run(program, factor_provider, selector_provider)
    return vm.slots.data
```

### Evidence as Factors

Evidence is multiplicative, not domain-restricting:
```python
def unary_evidence(var, allowed, domains, semiring):
    data = np.full((domains[var],), semiring.zero)
    for a in allowed:
        data[a] = semiring.one
    return NamedTensor((var,), data, semiring)
```

## Invariants

1. **Slot persistence**: Slots persist across run_program calls
2. **Evidence multiplicativity**: Evidence ⊗ factor, not domain restriction
3. **Z scalar**: Partition function is 0-dimensional tensor

## Extension Points

### Custom Scheduling

Wrap `run_program()` for custom execution:
```python
def my_scheduler(programs, sr, providers):
    for prog in programs:
        slots = run_program(prog, sr, *providers, slots=slots)
    return slots
```

### Evidence Types

Add new evidence patterns:
```python
def pairwise_evidence(var1, var2, allowed_pairs, domains, semiring):
    # Create 2D constraint tensor
    ...
```

## Usage Examples

```python
from catbp.runtime.schedule import run_program, finalize_Z_from_slots
from catbp.runtime.evidence import observe_variable
from catbp.vm.semiring import vm_prob_semiring

sr = vm_prob_semiring()

# Execute programs
slots = run_program(prog_up, sr, factor_provider, selector_provider)
slots = run_program(prog_down, sr, factor_provider, selector_provider, slots=slots)
slots = run_program(prog_fin, sr, factor_provider, selector_provider, slots=slots)

# Get result
Z = finalize_Z_from_slots(slots)

# Create evidence
from catbp.algebra.semiring import ProbSemiring
evidence = observe_variable(var_id, observed_value, domains, ProbSemiring())
```
