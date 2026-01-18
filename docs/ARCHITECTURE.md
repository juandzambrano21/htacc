# Architecture Overview

This document describes the system architecture of CatBP as a layered compilation and execution pipeline.

## 1. High-Level Pipeline

```
                              USER INPUT
                                  │
                    ┌─────────────┴─────────────┐
                    │     Factor Graph Spec      │
                    │  - var_domains: Dict       │
                    │  - factors: Dict           │
                    │  - semiring: str           │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TOPOLOGY LAYER                                   │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐          │
│  │ FactorGraph    │   │     Nerve      │   │  2-Skeleton    │          │
│  │   Structure    │ → │   (G_N)        │ → │ (triangles)    │          │
│  │ (structure.py) │   │  (nerve.py)    │   │ (simplicial.py)│          │
│  └────────────────┘   └────────────────┘   └────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         COMPILER LAYER                                   │
│  ┌──────────────┐                                                       │
│  │   Backbone   │  Maximum spanning tree on nerve (by interface size)   │
│  │ (backbone.py)│                                                       │
│  └──────┬───────┘                                                       │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────┐                                                       │
│  │    Cycles    │  Fundamental cycles for non-tree edges (chords)       │
│  │  (cycles.py) │                                                       │
│  └──────┬───────┘                                                       │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────┐                                                       │
│  │   Homology   │  H₁ computation to identify essential chords          │
│  │ (homology.py)│                                                       │
│  └──────┬───────┘                                                       │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────┐                                                       │
│  │  Transport   │  Boolean transport kernels on factor supports         │
│  │(transport.py)│                                                       │
│  └──────┬───────┘                                                       │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────┐                                                       │
│  │  Holonomy    │  Holonomy matrix composition, SCC quotient            │
│  │ (holonomy.py)│                                                       │
│  └──────┬───────┘                                                       │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────┐                                                       │
│  │    Modes     │  Mode axis IDs, selector payloads                     │
│  │  (modes.py)  │                                                       │
│  └──────┬───────┘                                                       │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────┐                                                       │
│  │     Plan     │  BPPlan: tree structure, belief specs, message keys   │
│  │  (plan.py)   │                                                       │
│  └──────────────┘                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       EMISSION LAYER                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      emit.py                                     │    │
│  │  ┌────────────┐   ┌────────────┐   ┌────────────┐              │    │
│  │  │  Upward    │   │  Downward  │   │  Finalize  │              │    │
│  │  │  Program   │   │  Program   │   │  Program   │              │    │
│  │  │ (postorder)│   │ (preorder) │   │ (Z comp)   │              │    │
│  │  └────────────┘   └────────────┘   └────────────┘              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   INTERMEDIATE REPRESENTATION                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │   schema.py: AxisType, Axis, AxisKey, TensorSpec                │    │
│  │   ops.py: OpCode, Instruction, Program                          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Instructions: UNIT, LOAD_FACTOR, LOAD_SELECTOR, LOAD_SLOT,             │
│                STORE_SLOT, CONTRACT, ELIMINATE, NORMALIZE, FREE          │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       VIRTUAL MACHINE                                    │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐     │
│  │  RegisterBank  │  SlotStore │   │VMSemiring  │   │  Kernels   │     │
│  │  (memory.py)   │  (vm.py)   │   │(semiring.py)   │(kernels.py)│     │
│  └────────────┘   └────────────┘   └────────────┘   └────────────┘     │
│         │               │               │               │               │
│         └───────────────┴───────────────┴───────────────┘               │
│                                   │                                      │
│                                   ▼                                      │
│                          ┌────────────────┐                             │
│                          │ VirtualMachine │                             │
│                          │    (vm.py)     │                             │
│                          └────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           RUNTIME                                        │
│  ┌────────────────┐   ┌────────────────┐                               │
│  │   schedule.py  │   │  evidence.py   │                               │
│  │ Program exec   │   │ Constraint     │                               │
│  │ Z extraction   │   │ injection      │                               │
│  └────────────────┘   └────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                            ┌─────────────┐
                            │   OUTPUT    │
                            │  Z, beliefs,│
                            │  marginals  │
                            └─────────────┘
```

## 2. Layer Responsibilities

### 2.1 Topology Layer (`topology/`)

**Purpose**: Represent factor graph structure and compute topological invariants.

| Module | Responsibility |
|--------|---------------|
| `structure.py` | `FactorGraphStructure`: variable domains, factor scopes, incidence |
| `nerve.py` | `Nerve`: intersection graph of factors with interface labels |
| `simplicial.py` | `Nerve2Skeleton`: vertices, edges, triangles for homology |
| `homology.py` | GF(2) linear algebra to identify H₁-essential chords |

**Invariants**:
- Scopes are always canonically sorted (by VarID)
- Nerve edges carry interface variables
- Triangles have non-empty triple intersection

### 2.2 Algebra Layer (`algebra/`)

**Purpose**: Provide semiring abstractions and section operations.

| Module | Responsibility |
|--------|---------------|
| `semiring.py` | `Semiring` protocol, concrete implementations (SAT, Prob, LogProb, Counting) |
| `section.py` | `Section`: domain-ordered tensor with star/restrict operations |
| `support.py` | `factor_support()`: extract boolean support, mode mapping from holonomy |

**Invariants**:
- Section domain matches tensor axes 1-1
- No duplicate variables in domain
- Semiring operations are vectorized for performance

### 2.3 Compiler Layer (`compiler/`)

**Purpose**: Transform topology into an executable plan.

| Module | Responsibility |
|--------|---------------|
| `backbone.py` | Maximum spanning tree selection, rooting, Euler tour |
| `cycles.py` | `HolonomyCycle`: typed transport steps around fundamental cycles |
| `transport.py` | `build_transport_kernel_boolean()`: sparse CSR matrices |
| `holonomy.py` | `compute_holonomy_boolean()`: matrix composition, SCC quotient |
| `modes.py` | `ModeSpec`: mode axis ID, interface, quotient map |
| `plan.py` | `BPPlan`: complete execution specification |
| `topology_compiler.py` | `TopologyCompiler`: main compilation orchestrator |
| `emit.py` | Program generation for upward/downward passes |
| `factors.py` | `make_factor_provider()`: lazy factor loading |
| `selectors.py` | `make_selector_provider()`: mode selector construction |

**Invariants**:
- Backbone is a spanning tree (connected, acyclic)
- Every chord contributes at most one mode axis
- Mode selectors are anchored at chord endpoints

### 2.4 IR Layer (`ir/`)

**Purpose**: Define the intermediate representation for compiled programs.

| Module | Responsibility |
|--------|---------------|
| `schema.py` | `AxisType`, `Axis`, `AxisKey`, `TensorSpec` |
| `ops.py` | `OpCode`, `Instruction`, `Program` |

**Invariants**:
- `AxisKey = (AxisType, id)` prevents VarID/ModeID collisions
- Instructions have deterministic execution semantics
- Programs are sequences (no control flow)

### 2.5 VM Layer (`vm/`)

**Purpose**: Execute compiled programs on concrete tensors.

| Module | Responsibility |
|--------|---------------|
| `vm.py` | `VirtualMachine`: instruction dispatch, register/slot management |
| `semiring.py` | `VMSemiringRuntime`: vectorized numpy operations |
| `kernels.py` | `vm_contract()`, `vm_eliminate_keep()`, `vm_unit()` |
| `kernels_selector.py` | `vm_load_selector()`: mode selector tensor construction |
| `align.py` | `align_array_to()`: tensor axis alignment by AxisKey |
| `memory.py` | `RegisterBank`: tensor storage during execution |

**Invariants**:
- Alignment uses AxisKey, not raw integer IDs
- Contraction broadcasts singleton dimensions
- Elimination reduces over dropped axes with semiring ⊕

### 2.6 Runtime Layer (`runtime/`)

**Purpose**: Orchestrate program execution and result extraction.

| Module | Responsibility |
|--------|---------------|
| `schedule.py` | `run_program()`: execute program on VM, return slots |
| `evidence.py` | `unary_evidence()`, `observe_variable()`: constraint injection |

**Invariants**:
- Slot keys are tuples (e.g., `("msg", u, v)`, `("bel", node)`, `("Z",)`)
- Evidence is multiplicative (factors, not domain restriction)

### 2.7 API Layer (`api/`)

**Purpose**: User-facing interfaces for common operations.

| Module | Responsibility |
|--------|---------------|
| `marginals.py` | `marginal_from_factor_belief()`: extract single-variable marginals |

### 2.8 Tensor Layer (`tensor/`)

**Purpose**: Named tensor operations for symbolic manipulation.

| Module | Responsibility |
|--------|---------------|
| `named_tensor.py` | `NamedTensor`: variable-indexed arrays, alignment, reduction |

## 3. Dependency Direction

Dependencies flow **downward** through the layers:

```
         API
          │
          ▼
       Runtime
          │
          ▼
   ┌──────┴──────┐
   │             │
   ▼             ▼
Compiler       VM
   │             │
   ▼             │
Topology ◄──────┘
   │             │
   ▼             ▼
Algebra        IR
```

**Strict Rules**:
- Topology does not depend on Compiler
- IR does not depend on VM
- Algebra is foundational (no layer dependencies except numpy)
- VM may use IR schemas but not Compiler

## 4. Design Invariants

These properties must hold across all code changes:

### 4.1 Determinism

All operations produce identical outputs for identical inputs:
- Canonical variable ordering (sorted VarIDs)
- Deterministic graph algorithms (sorted iteration)
- No floating-point randomness

### 4.2 Axis Key Isolation

`AxisKey = (AxisType, id)` ensures:
- VarID 3 ≠ ModeID 3
- No accidental axis collisions during alignment
- Clean separation of geometric and topological dimensions

### 4.3 Semiring Parametrization

All value-level operations are semiring-generic:
- Contraction uses `sr.mul`
- Elimination uses `sr.add_reduce`
- Unit uses `sr.one`

### 4.4 Support/Value Separation

Compilation operates on **support** (Boolean SAT semiring):
- Holonomy computation uses boolean transport
- Mode quotient uses boolean SCC

Runtime operates on **values** (user-specified semiring):
- Factor tensors carry actual weights
- Selectors use semiring one/zero

## 5. Failure Boundaries

### 5.1 Topology Failures

| Failure | Detection | Consequence |
|---------|-----------|-------------|
| Disconnected graph | Empty nerve edges | Separate partition functions |
| Variable not in any factor | Missing in var_to_factors | Ignored (unit contribution) |

### 5.2 Compiler Failures

| Failure | Detection | Consequence |
|---------|-----------|-------------|
| Scope size mismatch | Section validation | ValueError at construction |
| Empty holonomy matrix | Zero diagonal | All modes infeasible; Z = 0 |
| Cycle typing error | Assertion in cycles.py | Internal error |

### 5.3 VM Failures

| Failure | Detection | Consequence |
|---------|-----------|-------------|
| Axis alignment mismatch | size check in align.py | ValueError |
| Missing slot | KeyError in SlotStore | Runtime error |
| Register not found | KeyError in RegisterBank | Runtime error |

### 5.4 Numerical Failures

| Failure | Detection | Consequence |
|---------|-----------|-------------|
| Underflow in prob semiring | Z ≈ 0 | Switch to logprob |
| NaN in computation | isnan checks | Propagation error |
| Overflow | isinf checks | May indicate problem structure |

## 6. Extension Points

See [EXTENSIONS.md](EXTENSIONS.md) for detailed guidance on:
- Adding new semirings
- Implementing custom kernels
- Extending the IR instruction set
- Supporting new evidence types
