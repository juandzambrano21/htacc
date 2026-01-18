# vm/

## Intent

The `vm/` module implements the virtual machine that executes compiled belief propagation programs. It provides tensor operations, memory management, and semiring-parameterized computation.

## Public Surface

### From `vm.py`

| Export | Type | Description |
|--------|------|-------------|
| `VirtualMachine` | class | Main VM for program execution |
| `SlotStore` | class | Message/belief storage |

### From `semiring.py`

| Export | Type | Description |
|--------|------|-------------|
| `VMSemiringRuntime` | dataclass | Vectorized semiring backend |
| `vm_sat_semiring()` | function | Boolean semiring runtime |
| `vm_prob_semiring(dtype)` | function | Probability semiring runtime |
| `vm_logprob_semiring(dtype)` | function | Log-probability semiring runtime |
| `vm_counting_semiring(dtype)` | function | Counting semiring runtime |

### From `kernels.py`

| Export | Type | Description |
|--------|------|-------------|
| `vm_unit()` | function | Create unit tensor |
| `vm_contract()` | function | Multiply tensors with alignment |
| `vm_eliminate_keep()` | function | Marginalize keeping specified axes |
| `vm_eliminate()` | function | Marginalize dropping specified axes |

### From `kernels_selector.py`

| Export | Type | Description |
|--------|------|-------------|
| `SelectorPayload` | dataclass | Mode selector specification |
| `vm_load_selector()` | function | Construct selector tensor |
| `vm_load_selector_dense()` | function | Dense selector (legacy) |

### From `align.py`

| Export | Type | Description |
|--------|------|-------------|
| `align_array_to()` | function | Align tensor to target spec |

### From `memory.py`

| Export | Type | Description |
|--------|------|-------------|
| `RegisterBank` | class | Tensor register storage |
| `RegisterInfo` | dataclass | Register metadata |

## Internal Details

### Instruction Dispatch

```python
def _execute(self, ins, factor_provider, selector_provider):
    if ins.op == OpCode.UNIT:
        arr = vm_unit(ins.spec, self.sr)
        self.regs.put(ins.dst, arr, ins.spec)
    elif ins.op == OpCode.CONTRACT:
        inputs = [self.regs.get(rid) for rid in ins.args['inputs']]
        result = vm_contract(inputs, ins.args['out_spec'], self.sr)
        self.regs.put(ins.dst, result, ins.args['out_spec'])
    # ... etc
```

### Alignment by AxisKey

```python
def align_array_to(arr, in_spec, out_spec):
    # Match by AxisKey = (AxisType, id)
    # Insert singleton dims for missing axes
    # Transpose to match output order
```

### Semiring Operations

All value operations use semiring runtime:
- `sr.mul(a, b)`: Elementwise multiplication
- `sr.add_reduce(x, axis)`: Sum reduction
- `sr.one(shape)`: Unit tensor
- `sr.zero(shape)`: Zero tensor

## Invariants

1. **AxisKey isolation**: VarID and ModeID cannot collide
2. **Alignment consistency**: Input/output shapes must be compatible
3. **Semiring consistency**: All operations use same runtime
4. **Register cleanup**: FREE releases memory

## Extension Points

### Custom Kernels

Add new kernels in `kernels.py`:
```python
def vm_my_kernel(arr, spec, sr, **kwargs):
    # Custom operation
    return result, out_spec
```

### Custom Semiring

Create new semiring runtime:
```python
def vm_my_semiring(dtype=np.float64):
    return VMSemiringRuntime(
        name="MY_SR",
        dtype=np.dtype(dtype),
        mul=my_mul,
        add_reduce=my_reduce,
        one=lambda shape: ...,
        zero=lambda shape: ...,
        normalize=my_normalize,
    )
```

## Usage Examples

```python
from catbp.vm.vm import VirtualMachine
from catbp.vm.semiring import vm_prob_semiring
from catbp.vm.kernels import vm_contract, vm_eliminate_keep
from catbp.ir.schema import TensorSpec, Axis, AxisType

# Create VM
sr = vm_prob_semiring()
vm = VirtualMachine(sr)

# Execute program
vm.run(program, factor_provider, selector_provider)

# Access results
z_arr, z_spec = vm.slots.get(("Z",))
print(f"Z = {float(z_arr)}")

# Direct kernel usage
spec = TensorSpec(axes=(Axis(0, AxisType.GEOMETRIC, 2),))
unit = vm_unit(spec, sr)  # [1.0, 1.0]
```
