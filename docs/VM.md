# Virtual Machine

This document specifies the virtual machine (VM) that executes compiled belief propagation programs.

## 1. VM Architecture

### 1.1 Components

```
┌─────────────────────────────────────────────────────────────┐
│                    VIRTUAL MACHINE                           │
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │ RegisterBank │   │  SlotStore   │   │VMSemiring    │    │
│  │              │   │              │   │  Runtime     │    │
│  │ data: Dict   │   │ data: Dict   │   │              │    │
│  │ spec: Dict   │   │              │   │ mul, add_red │    │
│  └──────────────┘   └──────────────┘   │ one, zero    │    │
│         │                  │           │ normalize    │    │
│         │                  │           └──────────────┘    │
│         └──────────────────┴─────────────────│             │
│                            │                  │             │
│                            ▼                  ▼             │
│                    ┌──────────────────────────────┐        │
│                    │      Instruction Dispatch     │        │
│                    └──────────────────────────────┘        │
│                                    │                        │
│                                    ▼                        │
│                    ┌──────────────────────────────┐        │
│                    │         VM Kernels            │        │
│                    │  vm_unit, vm_contract,        │        │
│                    │  vm_eliminate_keep            │        │
│                    └──────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 VirtualMachine Class

```python
class VirtualMachine:
    def __init__(self, sr: VMSemiringRuntime):
        self.sr = sr              # Semiring operations
        self.regs = RegisterBank()  # Tensor storage
        self.slots = SlotStore()    # Message/belief storage
    
    def run(self, program: Program, factor_provider, selector_provider):
        for ins in program.instructions:
            self._execute(ins, factor_provider, selector_provider)
```

## 2. Kernel Abstraction

### 2.1 vm_unit

Create a tensor filled with semiring multiplicative identity:

```python
def vm_unit(out_spec: TensorSpec, sr: VMSemiringRuntime) -> np.ndarray:
    return sr.one(out_spec.shape).astype(sr.dtype, copy=False)
```

**Semantics**: Returns tensor where every entry equals `1_R`.

### 2.2 vm_contract

Multiply tensors with axis alignment:

```python
def vm_contract(
    inputs: List[Tuple[np.ndarray, TensorSpec]],
    out_spec: TensorSpec,
    sr: VMSemiringRuntime
) -> np.ndarray:
    if not inputs:
        return vm_unit(out_spec, sr)
    
    acc = None
    for arr, spec in inputs:
        # Align input axes to output axes
        view = align_array_to(arr, spec, out_spec)
        # Broadcast to output shape
        aligned = np.broadcast_to(view, out_spec.shape)
        
        if acc is None:
            acc = aligned.copy()
        else:
            acc = sr.mul(acc, aligned)  # Semiring multiplication
    
    return acc
```

**Semantics**: Point-wise semiring product of all inputs, aligned by AxisKey.

### 2.3 vm_eliminate_keep

Marginalize (sum out) axes not in keep_keys:

```python
def vm_eliminate_keep(
    arr: np.ndarray,
    in_spec: TensorSpec,
    keep_keys: Tuple[AxisKey, ...],
    sr: VMSemiringRuntime
) -> Tuple[np.ndarray, TensorSpec]:
    if not keep_keys:
        # Eliminate all → scalar
        out = sr.add_reduce(arr, axis=tuple(range(arr.ndim)))
        return np.asarray(out), TensorSpec(axes=())
    
    # Transpose: kept axes first (in keep_keys order), dropped axes last
    pos = in_spec.axis_pos()
    present = [k for k in keep_keys if k in pos]
    keep_axes = [pos[k] for k in present]
    drop_axes = [i for i in range(arr.ndim) if i not in set(keep_axes)]
    
    perm = keep_axes + drop_axes
    x = np.transpose(arr, axes=perm)
    
    # Reduce over trailing dropped axes
    if drop_axes:
        red = tuple(range(len(keep_axes), x.ndim))
        x = sr.add_reduce(x, axis=red)
    
    # Build output spec
    out_axes = tuple(in_spec.axes[pos[k]] for k in present)
    return x, TensorSpec(axes=out_axes)
```

**Semantics**: Sum over eliminated axes using semiring addition.

## 3. Alignment Guarantees

### 3.1 align_array_to

Align input tensor to target specification:

```python
def align_array_to(arr: np.ndarray, in_spec: TensorSpec, out_spec: TensorSpec) -> np.ndarray:
    # 1. Match axes by AxisKey (not raw ID)
    # 2. Missing axes become singleton dimensions
    # 3. Extra axes must be size-1 (else compiler bug)
    # 4. Output has same axis ORDER as out_spec
```

**Rules**:
- Axes are matched by `AxisKey = (AxisType, id)`
- `(GEOMETRIC, 3)` and `(TOPOLOGICAL, 3)` are **different**
- Missing output axes are inserted as size-1
- Extra input axes that aren't size-1 cause ValueError

**Example**:
```python
in_spec = TensorSpec([Axis(0, GEOMETRIC, 2), Axis(1, GEOMETRIC, 3)])
out_spec = TensorSpec([Axis(1, GEOMETRIC, 3), Axis(2, GEOMETRIC, 4), Axis(0, GEOMETRIC, 2)])

# Input shape: (2, 3)
# Output shape: (3, 1, 2)  <- axis 2 is singleton, axes reordered
```

### 3.2 AxisKey Isolation

The use of AxisKey prevents collisions:

```python
# These are DIFFERENT axes:
var_3 = (AxisType.GEOMETRIC, 3)
mode_3 = (AxisType.TOPOLOGICAL, 3)

# align_array_to distinguishes them correctly
```

## 4. Memory Ownership

### 4.1 RegisterBank

```python
class RegisterBank:
    def __init__(self):
        self.data: Dict[RegID, np.ndarray] = {}
        self.spec: Dict[RegID, TensorSpec] = {}
        self.next_id: int = 0
    
    def put(self, rid: RegID, arr: np.ndarray, spec: TensorSpec):
        self.data[rid] = arr
        self.spec[rid] = spec
    
    def get(self, rid: RegID) -> Tuple[np.ndarray, TensorSpec]:
        return self.data[rid], self.spec[rid]
    
    def free(self, rid: RegID):
        self.data.pop(rid, None)
        self.spec.pop(rid, None)
```

**Ownership rules**:
- Arrays in registers are owned by the register bank
- `put` takes ownership of the array
- `get` returns the array (not a copy)
- `free` releases the reference

### 4.2 SlotStore

```python
class SlotStore:
    data: Dict[SlotKey, Tuple[np.ndarray, TensorSpec]]
    
    def set(self, key: SlotKey, arr: np.ndarray, spec: TensorSpec):
        self.data[key] = (arr, spec)
    
    def get(self, key: SlotKey) -> Tuple[np.ndarray, TensorSpec]:
        return self.data[key]
```

**Ownership rules**:
- Slots persist across instructions
- Arrays are stored by reference
- Slots are not freed during execution

## 5. Semiring Dispatch Logic

### 5.1 VMSemiringRuntime

```python
@dataclass(frozen=True)
class VMSemiringRuntime:
    name: str
    dtype: np.dtype
    mul: Callable[[np.ndarray, np.ndarray], np.ndarray]
    add_reduce: Callable[[np.ndarray, Tuple[int, ...]], np.ndarray]
    one: Callable[[Tuple[int, ...]], np.ndarray]
    zero: Callable[[Tuple[int, ...]], np.ndarray]
    normalize: Optional[Callable[...]]
```

### 5.2 Factory Functions

#### SAT Semiring
```python
def vm_sat_semiring() -> VMSemiringRuntime:
    return VMSemiringRuntime(
        name="SAT",
        dtype=np.dtype(np.bool_),
        mul=np.logical_and,
        add_reduce=lambda x, axis: np.any(x, axis=axis),
        one=lambda shape: np.ones(shape, dtype=np.bool_),
        zero=lambda shape: np.zeros(shape, dtype=np.bool_),
        normalize=None,
    )
```

#### Probability Semiring
```python
def vm_prob_semiring(dtype=np.float64) -> VMSemiringRuntime:
    return VMSemiringRuntime(
        name="PROB",
        dtype=np.dtype(dtype),
        mul=np.multiply,
        add_reduce=lambda x, axis: np.sum(x, axis=axis),
        one=lambda shape: np.ones(shape, dtype=dtype),
        zero=lambda shape: np.zeros(shape, dtype=dtype),
        normalize=_prob_normalize,
    )
```

#### Log-Probability Semiring
```python
def vm_logprob_semiring(dtype=np.float64) -> VMSemiringRuntime:
    return VMSemiringRuntime(
        name="LOGPROB",
        dtype=np.dtype(dtype),
        mul=np.add,  # log(a*b) = log(a) + log(b)
        add_reduce=logsumexp,
        one=lambda shape: np.zeros(shape, dtype=dtype),  # log(1) = 0
        zero=lambda shape: np.full(shape, -np.inf, dtype=dtype),  # log(0) = -inf
        normalize=_logprob_normalize,
    )
```

### 5.3 Dispatch in Kernels

Kernels use the semiring runtime consistently:

```python
# In vm_contract:
acc = sr.mul(acc, aligned)  # Calls np.multiply, np.logical_and, or np.add

# In vm_eliminate_keep:
x = sr.add_reduce(x, axis=red)  # Calls np.sum, np.any, or logsumexp
```

## 6. Selector Loading

### 6.1 SelectorPayload

```python
@dataclass(frozen=True)
class SelectorPayload:
    mode_axis_id: int
    interface_axis_ids: Tuple[int, ...]
    quotient_map: np.ndarray  # flat_state → mode_id
    num_modes: int
    diag_mask: Optional[np.ndarray]  # Fixed-point feasibility
```

### 6.2 vm_load_selector

```python
def vm_load_selector(
    payload: SelectorPayload,
    domains: Dict[int, int],
    mode_sizes: Dict[int, int],
    dtype=np.float32,
    one_val=None,
    zero_val=None,
) -> Tuple[np.ndarray, TensorSpec]:
    J = payload.interface_axis_ids
    geo_shape = tuple(domains[v] for v in J)
    flat_dim = np.prod(geo_shape)
    Q = payload.num_modes
    
    # Initialize with zeros
    data = np.full((flat_dim, Q), zero_val, dtype=dtype)
    
    # Set ones for valid (state, mode) pairs
    if payload.diag_mask is None:
        # All states valid
        for i, label in enumerate(payload.quotient_map):
            if label >= 0:
                data[i, label] = one_val
    else:
        # Only fixed-point feasible states
        for i, (label, ok) in enumerate(zip(payload.quotient_map, payload.diag_mask)):
            if ok and label >= 0:
                data[i, label] = one_val
    
    # Reshape to geometric + mode axes
    data = data.reshape(geo_shape + (Q,))
    
    # Build spec
    axes = [Axis(v, GEOMETRIC, domains[v]) for v in J]
    axes.append(Axis(payload.mode_axis_id, TOPOLOGICAL, mode_sizes[payload.mode_axis_id]))
    
    return data, TensorSpec(axes=tuple(axes))
```

**Semantics**: Selector `σ_e(j, c)` is 1 iff state `j` maps to mode `c` and is feasible.

## 7. Execution Loop

### 7.1 Main Dispatch

```python
def _execute(self, ins: Instruction, factor_provider, selector_provider):
    if ins.op == OpCode.UNIT:
        arr = vm_unit(ins.spec, self.sr)
        self.regs.put(ins.dst, arr, ins.spec)
        
    elif ins.op == OpCode.LOAD_FACTOR:
        arr, spec = factor_provider(ins.args['factor_id'])
        self.regs.put(ins.dst, arr.astype(self.sr.dtype), spec)
        
    elif ins.op == OpCode.LOAD_SELECTOR:
        arr, spec = selector_provider(ins.args['selector_id'])
        self.regs.put(ins.dst, arr.astype(self.sr.dtype), spec)
        
    elif ins.op == OpCode.LOAD_SLOT:
        arr, spec = self.slots.get(ins.args['slot'])
        self.regs.put(ins.dst, arr, spec)
        
    elif ins.op == OpCode.STORE_SLOT:
        arr, spec = self.regs.get(ins.args['src'])
        self.slots.set(ins.args['slot'], arr, spec)
        
    elif ins.op == OpCode.CONTRACT:
        inputs = [self.regs.get(rid) for rid in ins.args['inputs']]
        result = vm_contract(inputs, ins.args['out_spec'], self.sr)
        self.regs.put(ins.dst, result, ins.args['out_spec'])
        
    elif ins.op == OpCode.ELIMINATE:
        arr, in_spec = self.regs.get(ins.args['src'])
        result, out_spec = vm_eliminate_keep(arr, in_spec, ins.args['keep_keys'], self.sr)
        self.regs.put(ins.dst, result, out_spec)
        
    elif ins.op == OpCode.NORMALIZE:
        if self.sr.supports_normalize():
            arr, spec = self.regs.get(ins.args['src'])
            arr = self.sr.normalize(arr, None)
            self.regs.put(ins.args['src'], arr, spec)
            
    elif ins.op == OpCode.FREE:
        for rid in ins.args.get('srcs', ()):
            self.regs.free(rid)
```

## 8. Error Handling

### 8.1 Alignment Errors

```python
def align_array_to(...):
    # Check for extra non-singleton axes
    for i, ax in enumerate(in_spec.axes):
        if ax.key not in out_key_set and arr.shape[i] != 1:
            raise ValueError(f"axis {ax.key} not in output but size != 1")
    
    # Check size consistency
    if arr_t.shape[kept_cursor] != out_ax.size:
        raise ValueError(f"size mismatch for axis {out_ax.key}")
```

### 8.2 Missing Resources

```python
# SlotStore
def get(self, key):
    if key not in self.data:
        raise KeyError(f"Slot not found: {key}")
    return self.data[key]

# RegisterBank
def get(self, rid):
    # Returns self.data[rid] which raises KeyError if missing
```

### 8.3 Type Consistency

All arrays are cast to semiring dtype:

```python
arr.astype(self.sr.dtype, copy=False)
```

This ensures consistent numerical behavior across operations.

## 9. Performance Considerations

### 9.1 Memory Reuse

- FREE instructions allow garbage collection
- Registers are removed from dict, allowing Python GC

### 9.2 Copy Avoidance

- `copy=False` in `astype` when possible
- `np.broadcast_to` returns views, not copies
- Alignment creates minimal intermediate arrays

### 9.3 Vectorization

All semiring operations are numpy-vectorized:
- No Python loops over tensor elements
- Efficient C-level implementations
