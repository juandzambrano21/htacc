# ir/

## Intent

The `ir/` module defines the intermediate representation that bridges the compiler and virtual machine. It provides a typed schema for tensor specifications and an instruction set for belief propagation operations.

## Public Surface

### From `schema.py`

| Export | Type | Description |
|--------|------|-------------|
| `AxisType` | enum | GEOMETRIC (variable) or TOPOLOGICAL (mode) |
| `AxisKey` | type alias | `Tuple[AxisType, int]` for unique axis identification |
| `Axis` | dataclass | Full axis specification (id, kind, size) |
| `TensorSpec` | dataclass | Collection of axes defining tensor shape/semantics |
| `geometric_spec()` | function | Create spec for variable-only tensors |
| `sort_axes()` | function | Sort axes into canonical order |

### From `ops.py`

| Export | Type | Description |
|--------|------|-------------|
| `OpCode` | enum | VM operation codes |
| `Instruction` | dataclass | Single VM instruction |
| `Program` | dataclass | Sequence of instructions |
| `RegID` | type alias | Register identifier (int) |

## Internal Details

### AxisKey Design

AxisKey prevents ID collisions between variables and modes:
```python
# These are DIFFERENT despite same integer:
(AxisType.GEOMETRIC, 3)   # Variable 3
(AxisType.TOPOLOGICAL, 3)  # Mode 3
```

### TensorSpec Operations

```python
spec.shape      # Tuple of sizes
spec.keys       # Tuple of AxisKeys
spec.ndim       # Number of dimensions
spec.axis_pos() # Map from AxisKey to position
spec.subset(keep_keys)  # Extract subset of axes
spec.has_key(key)       # Check if axis present
spec.get_axis(key)      # Get axis by key
```

### OpCode Semantics

| OpCode | Inputs | Output | Description |
|--------|--------|--------|-------------|
| `UNIT` | spec | tensor | Create unit (all-ones) tensor |
| `LOAD_FACTOR` | factor_id | tensor | Load factor from provider |
| `LOAD_SELECTOR` | selector_id | tensor | Load mode selector |
| `LOAD_SLOT` | slot_key | tensor | Load from message/belief storage |
| `STORE_SLOT` | slot_key, src | - | Store to slot |
| `CONTRACT` | inputs[], out_spec | tensor | Multiply with alignment |
| `ELIMINATE` | src, keep_keys | tensor | Marginalize axes |
| `NORMALIZE` | src | tensor | Apply normalization |
| `FREE` | srcs[] | - | Release registers |

## Invariants

1. **AxisKey uniqueness**: No duplicate keys in TensorSpec
2. **Canonical ordering**: Axes sorted by (kind.value, id)
3. **Instruction completeness**: All required args present
4. **Program linearity**: No control flow, sequential execution

## Extension Points

### New Instructions

Add to `OpCode` enum and handle in `vm.py`:
```python
class OpCode(Enum):
    ...
    MY_OP = 10

# In VirtualMachine._execute():
elif ins.op == OpCode.MY_OP:
    ...
```

### Custom TensorSpec

Extend `TensorSpec` for additional metadata:
```python
@dataclass(frozen=True)
class ExtendedSpec(TensorSpec):
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## Usage Examples

```python
from catbp.ir.schema import AxisType, Axis, TensorSpec, geometric_spec
from catbp.ir.ops import OpCode, Instruction, Program

# Create tensor spec
spec = TensorSpec(axes=(
    Axis(id=0, kind=AxisType.GEOMETRIC, size=2),
    Axis(id=1, kind=AxisType.GEOMETRIC, size=3),
    Axis(id=0, kind=AxisType.TOPOLOGICAL, size=4),
))

print(spec.shape)  # (2, 3, 4)
print(spec.keys)   # ((GEOMETRIC, 0), (GEOMETRIC, 1), (TOPOLOGICAL, 0))

# Create instruction
ins = Instruction(
    op=OpCode.CONTRACT,
    dst=5,
    args={'inputs': (1, 2, 3), 'out_spec': spec},
    spec=spec
)

# Create program
program = Program(instructions=[ins1, ins2, ins3])
for ins in program:
    print(ins)
```
