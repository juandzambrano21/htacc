# Intermediate Representation

This document specifies the intermediate representation (IR) used by CatBP.

## 1. Purpose of the IR

The IR serves as the interface between the compiler and the virtual machine:

1. **Decoupling**: Compiler produces IR; VM executes IR
2. **Abstraction**: IR hides implementation details of tensor operations
3. **Optimization potential**: IR could be transformed before execution
4. **Debugging**: IR programs can be inspected for correctness

**Why direct execution is impossible**: Factor graphs have variable structure, but execution requires fixed operation sequences. The compiler analyzes topology once; the IR encodes the resulting execution plan.

## 2. IR Objects

### 2.1 Axis Types

```python
class AxisType(Enum):
    GEOMETRIC = 1      # Variable axes (VarID)
    TOPOLOGICAL = 2    # Mode axes (ModeID)
```

**Purpose**: Distinguish variable dimensions from mode dimensions to prevent ID collisions.

### 2.2 Axis Key

```python
AxisKey = Tuple[AxisType, int]
```

A unique identifier for a tensor axis:
- `(GEOMETRIC, 3)` = Variable with ID 3
- `(TOPOLOGICAL, 3)` = Mode with ID 3

**These are distinct** despite having the same integer.

### 2.3 Axis

```python
@dataclass(frozen=True)
class Axis:
    id: int           # VarID or ModeID
    kind: AxisType    # GEOMETRIC or TOPOLOGICAL
    size: int         # Domain/mode size
    
    @property
    def key(self) -> AxisKey:
        return (self.kind, self.id)
```

Complete specification of a single tensor dimension.

### 2.4 TensorSpec

```python
@dataclass(frozen=True)
class TensorSpec:
    axes: Tuple[Axis, ...]
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(ax.size for ax in self.axes)
    
    @property
    def keys(self) -> Tuple[AxisKey, ...]:
        return tuple(ax.key for ax in self.axes)
    
    @property
    def ndim(self) -> int:
        return len(self.axes)
```

Specification of a tensor's axes (shape with semantic labels).

**Key operations**:
- `axis_pos()`: Map from AxisKey to position
- `subset(keep_keys)`: Extract subset of axes
- `has_key(key)`: Check if axis present
- `get_axis(key)`: Get axis by key

### 2.5 OpCode

```python
class OpCode(Enum):
    UNIT = 1           # Create unit tensor
    LOAD_FACTOR = 2    # Load factor from store
    LOAD_SELECTOR = 3  # Load mode selector
    LOAD_SLOT = 4      # Load from slot (message/belief)
    STORE_SLOT = 5     # Store to slot
    CONTRACT = 6       # Multiply tensors
    ELIMINATE = 7      # Marginalize axes
    NORMALIZE = 8      # Normalize tensor
    FREE = 9           # Free registers
```

### 2.6 Instruction

```python
@dataclass(frozen=True)
class Instruction:
    op: OpCode
    dst: Optional[RegID]      # Destination register (None for STORE_SLOT, FREE)
    args: Dict[str, Any]      # Operation-specific arguments
    spec: Optional[TensorSpec]  # Output specification (for some ops)
```

### 2.7 Program

```python
@dataclass(frozen=True)
class Program:
    instructions: Sequence[Instruction]
```

A linear sequence of instructions to execute.

## 3. Well-Formedness Rules

### 3.1 Register Rules

1. **Defined before use**: A register must be written before being read
2. **Single assignment**: Each register is written at most once (SSA-like)
3. **Freed after use**: Registers should be freed when no longer needed

### 3.2 Slot Rules

1. **Write before read**: Slots must be stored before loading
2. **Unique keys**: Slot keys are globally unique identifiers
3. **Type consistency**: A slot's TensorSpec is fixed at first write

### 3.3 Instruction Rules

#### UNIT
- `dst` must be specified
- `spec` must be specified
- No args required

#### LOAD_FACTOR
- `dst` must be specified
- `args["factor_id"]` must be valid factor ID

#### LOAD_SELECTOR
- `dst` must be specified
- `args["selector_id"]` must be valid selector ID

#### LOAD_SLOT
- `dst` must be specified
- `args["slot"]` must be valid slot key

#### STORE_SLOT
- `args["slot"]` must be valid slot key
- `args["src"]` must be valid register

#### CONTRACT
- `dst` must be specified
- `args["inputs"]` must be tuple of valid registers
- `args["out_spec"]` or `spec` must specify output

#### ELIMINATE
- `dst` must be specified
- `args["src"]` must be valid register
- `args["keep_keys"]` must be tuple of AxisKeys

#### NORMALIZE
- `args["src"]` must be valid register

#### FREE
- `args["srcs"]` must be tuple of valid registers

### 3.4 Program Structure

1. **No cycles**: Instructions form a DAG of dependencies
2. **Single output per slot**: Each slot is written exactly once
3. **Complete cleanup**: All allocated registers are eventually freed

## 4. Transformations

### 4.1 Allowed Rewrites

The following transformations preserve semantics:

#### Dead Code Elimination
Remove instructions whose results are never used:
```
r1 = LOAD_FACTOR(f1)
r2 = LOAD_FACTOR(f2)   # Dead if r2 unused
r3 = CONTRACT(r1)
```

#### Common Subexpression Elimination
Reuse identical computations:
```
r1 = LOAD_FACTOR(f1)
r2 = LOAD_FACTOR(f1)   # Can reuse r1
```

#### Instruction Reordering
Move independent instructions (respecting data dependencies):
```
r1 = LOAD_FACTOR(f1)   # Independent
r2 = LOAD_FACTOR(f2)   # Independent
# Can swap order
```

### 4.2 Forbidden Rewrites

The following transformations are **not** valid:

#### Changing Contraction Order (Different Semirings)
Semiring operations may not be associative in practice (floating-point):
```
CONTRACT(CONTRACT(a, b), c) ≠ CONTRACT(a, CONTRACT(b, c))  # Due to rounding
```

#### Reordering Dependent Instructions
```
STORE_SLOT(s, r1)
LOAD_SLOT(s)      # Must come after store
```

#### Eliminating Before Required Axes Exist
```
r1 = CONTRACT(...)       # Creates axes
r2 = ELIMINATE(r1, ...)  # Must come after
```

### 4.3 Currently Unimplemented

The following optimizations are valid but not implemented:

1. **Fusion**: Combine CONTRACT + ELIMINATE into single operation
2. **Lazy evaluation**: Delay computation until result needed
3. **Parallelization**: Execute independent instructions concurrently
4. **Memory pooling**: Reuse register storage

## 5. IR Construction

### 5.1 Upward Program

```python
def emit_upward_program(plan) -> Program:
    instrs = []
    
    for a in plan.postorder:
        # 1. Load local factor
        r_phi = LOAD_FACTOR(a)
        
        # 2. Load anchored selectors
        for mid in node.anchored_modes:
            r_sel = LOAD_SELECTOR(mid)
        
        # 3. Load child messages
        for c in node.children:
            r_msg = LOAD_SLOT(("msg", c, a))
        
        # 4. Contract all inputs
        r_acc = CONTRACT(inputs, out_spec=belief_spec[a])
        
        # 5. Eliminate to parent message (or store root belief)
        if a == root:
            STORE_SLOT(("bel", root), r_acc)
        else:
            r_msg = ELIMINATE(r_acc, keep_keys=interface(a, parent))
            STORE_SLOT(("msg", a, parent), r_msg)
        
        # 6. Free registers
        FREE(all_used_registers)
    
    return Program(instrs)
```

### 5.2 Downward Program

```python
def emit_downward_program_prefixsuffix(plan) -> Program:
    instrs = []
    
    for a in plan.preorder:
        # 1. Base = factor(a) * selectors(a)
        base = CONTRACT(factor, selectors)
        
        # 2. Collect inbound messages
        inbound = [msg_from_parent] + [msg_from_children]
        
        # 3. Compute full belief
        belief = CONTRACT(base, inbound)
        STORE_SLOT(("bel", a), belief)
        
        # 4. Compute messages to children using prefix/suffix
        # prefix[i] = base * down * child[0] * ... * child[i-1]
        # suffix[i] = child[i+1] * ... * child[k]
        # msg_to_child[i] = prefix[i] * suffix[i+1]
        
        for i, child in enumerate(children):
            excl = CONTRACT(prefix[i], suffix[i+1])
            msg = ELIMINATE(excl, keep_keys=interface(a, child))
            STORE_SLOT(("msg", a, child), msg)
        
        FREE(registers)
    
    return Program(instrs)
```

### 5.3 Finalize Program

```python
def emit_finalize_program(root) -> Program:
    r_bel = LOAD_SLOT(("bel", root))
    r_z = ELIMINATE(r_bel, keep_keys=())  # Scalar
    STORE_SLOT(("Z",), r_z)
    FREE(r_bel, r_z)
    return Program(instrs)
```

## 6. Slot Key Conventions

Slot keys are tuples identifying stored tensors:

| Key Pattern | Meaning |
|-------------|---------|
| `("msg", u, v)` | Message from node u to node v |
| `("bel", a)` | Belief at node a |
| `("Z",)` | Partition function (scalar) |

## 7. Example IR

For a simple chain A → B:

```
UPWARD PROGRAM:
  [0] LOAD_FACTOR dst=1 args={'factor_id': 0}          # Load factor f_A
  [1] CONTRACT dst=2 args={'inputs': (1,), ...}        # Trivial contraction
  [2] ELIMINATE dst=3 args={'src': 2, 'keep_keys': ((GEOMETRIC, 0),)}
  [3] STORE_SLOT args={'slot': ('msg', 0, 1), 'src': 3}
  [4] FREE args={'srcs': (1, 2, 3)}
  
  [5] LOAD_FACTOR dst=4 args={'factor_id': 1}          # Load factor f_AB
  [6] LOAD_SLOT dst=5 args={'slot': ('msg', 0, 1)}     # Load A's message
  [7] CONTRACT dst=6 args={'inputs': (4, 5), ...}      # Combine
  [8] STORE_SLOT args={'slot': ('bel', 1), 'src': 6}   # Store root belief
  [9] FREE args={'srcs': (4, 5, 6)}

FINALIZE PROGRAM:
  [0] LOAD_SLOT dst=1 args={'slot': ('bel', 1)}
  [1] ELIMINATE dst=2 args={'src': 1, 'keep_keys': ()}
  [2] STORE_SLOT args={'slot': ('Z',), 'src': 2}
  [3] FREE args={'srcs': (1, 2)}
```

## 8. Future Extensions

### 8.1 Control Flow

Current IR is linear. Extensions could add:
- Conditional execution
- Loops for iterative algorithms
- Early termination

### 8.2 Parallel Execution

Mark independent instructions:
```python
@dataclass
class Instruction:
    ...
    parallel_group: Optional[int]  # Instructions in same group can run in parallel
```

### 8.3 Memory Hints

Optimization annotations:
```python
@dataclass
class Instruction:
    ...
    inplace: bool = False      # Can overwrite input
    reuse_from: Optional[RegID]  # Reuse memory from freed register
```
