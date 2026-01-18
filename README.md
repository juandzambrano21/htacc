# CatBP — Categorical Belief Propagation

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2601.04456-b31b1b.svg)](https://arxiv.org/abs/2601.04456)

This is the reference implementation for [*Categorical Belief Propagation: Sheaf-Theoretic Inference via Descent and Holonomy*](https://arxiv.org/abs/2601.04456) (ter Horst, Mahadevan, Zambrano, 2026).

CatBP is an implementation of **Holonomy-Aware Tree-Compiled Computation (HATCC)**, a categorical framework for exact inference on factor graphs with loopy structure. The system compiles factor graphs into executable plans that exploit sheaf-theoretic structure, detecting and resolving descent obstructions via holonomy computation on the factor nerve.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/juandzambrano21/catbp.git
cd catbp

# Install dependencies
pip install numpy scipy networkx

# Install package
pip install -e .
```

### Requirements

- Python ≥ 3.9
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0
- NetworkX ≥ 2.6.0

## Quick Start

### Python API

```python
from catbp import hatcc_solve, compute_marginals
import numpy as np

# Define a factor graph
var_domains = {"A": 2, "B": 2, "C": 2}
factors = {
    "prior_A": (("A",), np.array([0.6, 0.4])),
    "factor_AB": (("A", "B"), np.array([[0.9, 0.1], [0.2, 0.8]])),
    "factor_BC": (("B", "C"), np.array([[0.3, 0.7], [0.5, 0.5]])),
}

# Solve for partition function
result = hatcc_solve(var_domains, factors)
print(f"Partition function Z = {result.Z}")

# Compute marginals
marginals = compute_marginals(var_domains, factors)
for var, prob in marginals.items():
    print(f"P({var}) = {prob}")
```

### Command Line

```bash
 # Solve from JSON file
    python main.py solve --input problem.json --output result.json
# Solve from command line
    python main.py solve --vars "A:2,B:2,C:2" --factors "f1:A,B:[[0.9,0.1],[0.2,0.8]]" --output result.json
    
# Run demos
    python main.py demo --example chain
    
# Run tests
    python main.py test
```

### JSON Input Format

```json
{
  "variables": {
    "A": 2,
    "B": 2,
    "C": 2
  },
  "factors": {
    "f1": {
      "scope": ["A", "B"],
      "values": [[0.9, 0.1], [0.2, 0.8]]
    },
    "f2": {
      "scope": ["B", "C"],
      "values": [[0.3, 0.7], [0.5, 0.5]]
    }
  }
}
```

## Semirings

CatBP supports multiple computational domains through semiring abstraction:

| Semiring | Addition (⊕) | Multiplication (⊗) | Use Case |
|----------|-------------|-------------------|----------|
| `prob` | + | × | Probability computations |
| `logprob` | logsumexp | + | Numerically stable inference |
| `sat` | OR | AND | Satisfiability checking |

```python
# Log-probability for numerical stability
result = hatcc_solve(var_domains, factors, semiring="logprob")

# Boolean satisfiability
result = hatcc_solve(var_domains, sat_factors, semiring="sat")
print(f"Satisfiable: {result.Z > 0}")
```

## How It Works

CatBP implements the HATCC algorithm from the paper:

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

```
Factor Graph → Nerve → Backbone → Cycles → Holonomy → Modes → Exact BP
```

### Complexity

```
O(n² · d_max + c · k_max · δ_max³ + n · δ_max²)
```

where:
- `n` = number of factors
- `d_max` = maximum factor scope size
- `c` = number of fundamental cycles
- `k_max` = maximum cycle length
- `δ_max` = maximum interface size

## Project Structure

```
catbp/
├── algebra/          # Semiring abstractions and sections
│   ├── semiring.py   # SAT, Prob, LogProb, Counting semirings
│   ├── section.py    # Domain-ordered tensors with star/restrict
│   └── support.py    # Boolean support extraction
├── topology/         # Factor graph structure
│   ├── structure.py  # FactorGraphStructure
│   ├── nerve.py      # Nerve graph construction
│   ├── simplicial.py # 2-skeleton for homology
│   └── homology.py   # H₁ computation over GF(2)
├── compiler/         # Topology to execution plan
│   ├── backbone.py   # Spanning tree selection
│   ├── cycles.py     # Fundamental cycle detection
│   ├── holonomy.py   # Transport kernel composition
│   ├── modes.py      # Mode variable creation
│   ├── plan.py       # BPPlan construction
│   └── emit.py       # IR program generation
├── ir/               # Intermediate representation
│   ├── schema.py     # AxisType, TensorSpec
│   └── ops.py        # OpCode, Instruction, Program
├── vm/               # Virtual machine
│   ├── vm.py         # VirtualMachine execution
│   ├── kernels.py    # contract, eliminate, unit
│   └── semiring.py   # VMSemiringRuntime
├── runtime/          # Execution orchestration
│   ├── schedule.py   # Program execution
│   └── evidence.py   # Constraint injection
├── hatcc.py          # High-level API
└── solver.py         # Core solver implementation
```

## Documentation

Documentation is available in the `docs/` directory:

- [README](docs/README.md) — Documentation entry point
- [SPEC](docs/SPEC.md) — Formal system specification
- [ARCHITECTURE](docs/ARCHITECTURE.md) — System architecture
- [ALGEBRA](docs/ALGEBRA.md) — Algebraic foundations
- [TOPOLOGY](docs/TOPOLOGY.md) — Topological foundations
- [SEMANTICS](docs/SEMANTICS.md) — Compiler/runtime semantics
- [IR](docs/IR.md) — Intermediate representation
- [VM](docs/VM.md) — Virtual machine specification
- [RUNTIME](docs/RUNTIME.md) — Execution model
- [EXTENSIONS](docs/EXTENSIONS.md) — Extension guide
- [TESTING](docs/TESTING.md) — Testing philosophy

## Examples

See the `examples/` directory for complete examples:

```bash
# Simple chain
python examples/simple_chain.py

# 2D grid Ising model
python examples/grid_model.py

# Frustrated cycle (demonstrates holonomy)
python examples/frustrated_z4_cycle.py
```

## Testing

```bash
# Run all tests
python main.py test

# With verbose output
python main.py test -v

# With coverage
python main.py test -c
```

## Citation

If you use CatBP in your research, please cite:

```bibtex
@article{terhorst2026categorical,
  title={Categorical Belief Propagation: Sheaf-Theoretic Inference via Descent and Holonomy},
  author={ter Horst, Enrique and Mahadevan, Sridhar and Zambrano, Juan Diego},
  journal={arXiv preprint arXiv:2601.04456},
  year={2026}
}
```

## Authors

- **Juan Diego Zambrano** — Implementation & Design & Algorithms
- **Enrique ter Horst** — Theory & Algorithms
- **Sridhar Mahadevan** — Theory & Algorithms

## License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read the documentation in `docs/EXTENSIONS.md` for guidelines on extending the system while preserving its invariants.

## Acknowledgments

This implementation is based on the theoretical framework developed in:

> **Categorical Belief Propagation: Sheaf-Theoretic Inference via Descent and Holonomy**  
> Enrique ter Horst, Sridhar Mahadevan, Juan Diego Zambrano  
> arXiv:2601.04456 [cs.AI]

The key insight is that exact inference on factor graphs corresponds to effective descent in a sheaf-theoretic sense, with obstructions characterized by holonomy around fundamental cycles of the factor nerve.
