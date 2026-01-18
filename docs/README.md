# CatBP — Categorical Belief Propagation with Topological Constraints

## 1. What This System Is

CatBP is an implementation of **Holonomy-Aware Tree-Compiled Computation (HATCC)**, a categorical framework for exact inference on factor graphs with loopy structure. The system compiles factor graphs into executable plans that exploit sheaf-theoretic structure, detecting and resolving descent obstructions via holonomy computation on the factor nerve.

## 2. What Problem It Solves

Standard belief propagation on loopy graphs produces approximations that may fail to converge or yield incorrect marginals. Junction tree algorithms provide exact inference but suffer exponential blowup on graphs with large treewidth. CatBP addresses this gap by:

- **Detecting descent obstructions** via holonomy computation on fundamental cycles of the factor nerve
- **Compiling non-trivial holonomy** into discrete mode variables that augment the factor graph
- **Reducing to tree BP** on the augmented graph, achieving exact inference with complexity polynomial in the number of modes

This approach unifies tree exactness, junction tree algorithms, and loopy BP failures under a single sheaf-theoretic framework.

## 3. Core Insight

The mathematical foundation rests on three pillars:

1. **Categorical Structure**: Factor graphs are morphisms in a free hypergraph category Syn_Σ. Compositional semantics arise from a unique functor to the matrix category Mat_R over a commutative semiring R.

2. **Semiring Abstraction**: All operations (message passing, marginalization, contraction) are parameterized by a semiring (S, ⊕, ⊗, 0, 1), enabling unified treatment of probability, log-probability, Boolean satisfiability, and counting problems.

3. **Topological Consistency**: Exact inference is characterized as effective descent—local beliefs form a valid global section when compatibility conditions hold on overlaps. Non-trivial first homology H₁ of the nerve indicates descent obstructions, which are resolved by computing holonomy (parallel transport around cycles) and introducing mode variables that quotient the obstruction space.

## 4. System Layers

The implementation is organized into the following subsystems:

| Directory | Purpose |
|-----------|---------|
| `algebra/` | Semiring abstractions, Section operations, support computation |
| `topology/` | Factor graph structure, nerve construction, simplicial complex, homology |
| `compiler/` | Backbone selection, cycle detection, holonomy computation, plan generation |
| `ir/` | Intermediate representation schema and instruction set |
| `vm/` | Virtual machine for plan execution, kernel operations, memory management |
| `runtime/` | Scheduling, evidence handling, program execution |
| `tensor/` | Named tensor operations with canonical ordering |
| `api/` | High-level marginal computation interface |
| `core/` | Registry and utilities |

## 5. Execution Model

```
Factor Graph Definition
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                  TOPOLOGY COMPILER                       │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐           │
│  │  Nerve   │ → │ Backbone │ → │  Chords   │           │
│  │ (G_N)    │   │  (MST)   │   │ (H₁ basis)│           │
│  └──────────┘   └──────────┘   └───────────┘           │
│                       │                                  │
│                       ▼                                  │
│              ┌────────────────┐                         │
│              │   Holonomy     │                         │
│              │ Computation    │                         │
│              └────────────────┘                         │
│                       │                                  │
│                       ▼                                  │
│              ┌────────────────┐                         │
│              │ Mode Variables │                         │
│              │  (Selectors)   │                         │
│              └────────────────┘                         │
│                       │                                  │
│                       ▼                                  │
│              ┌────────────────┐                         │
│              │ Execution Plan │                         │
│              │   (BPPlan)     │                         │
│              └────────────────┘                         │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                    CODE EMISSION                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│  │ Upward   │   │ Downward │   │ Finalize │            │
│  │ Program  │   │ Program  │   │ Program  │            │
│  └──────────┘   └──────────┘   └──────────┘            │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                  VIRTUAL MACHINE                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│  │ Register │   │ Semiring │   │  Slot    │            │
│  │  Bank    │   │ Runtime  │   │  Store   │            │
│  └──────────┘   └──────────┘   └──────────┘            │
│         │              │              │                  │
│         └──────────────┴──────────────┘                 │
│                        │                                 │
│                        ▼                                 │
│                 ┌─────────────┐                          │
│                 │   Beliefs   │                          │
│                 │  Marginals  │                          │
│                 │      Z      │                          │
│                 └─────────────┘                          │
└─────────────────────────────────────────────────────────┘
```

## 6. Who This Is For

This system is designed for:

- **Researchers** in probabilistic graphical models seeking exact inference on loopy graphs
- **Compiler engineers** interested in domain-specific compilation for algebraic structures
- **Category theorists** exploring applied sheaf theory and descent conditions
- **Systems developers** building inference engines for SAT, probabilistic reasoning, or counting

## 7. Document Index

| Document | Description |
|----------|-------------|
| [SPEC.md](SPEC.md) | Formal system specification |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture and design |
| [ALGEBRA.md](ALGEBRA.md) | Algebraic foundations (semirings, sections) |
| [TOPOLOGY.md](TOPOLOGY.md) | Topological foundations (nerve, homology, holonomy) |
| [SEMANTICS.md](SEMANTICS.md) | Compiler and runtime semantics |
| [IR.md](IR.md) | Intermediate representation specification |
| [VM.md](VM.md) | Virtual machine specification |
| [RUNTIME.md](RUNTIME.md) | Runtime execution model |
| [EXTENSIONS.md](EXTENSIONS.md) | Extension guide |
| [TESTING.md](TESTING.md) | Testing philosophy |
| [modules/](modules/) | Per-module documentation |

## 8. Reference

This implementation is based on the paper:

> **Categorical Belief Propagation: Sheaf-Theoretic Inference via Descent and Holonomy**  
> Enrique ter Horst, Sridhar Mahadevan, Juan Diego Zambrano  
> arXiv:2601.04456 [cs.AI]

## 9. License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](../LICENSE) for details.
