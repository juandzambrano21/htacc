"""
CatBP: Categorical Belief Propagation

A production-ready implementation of holonomy-aware belief propagation
using category-theoretic foundations.

Key components:
- algebra: Semiring abstractions and section operations
- tensor: Named tensor operations with canonical ordering
- topology: Factor graph structure, nerve, and homology
- compiler: Topology compilation to execution plans
- ir: Intermediate representation schema and operations
- vm: Virtual machine for executing compiled plans
- runtime: Evidence handling and scheduling
- core: ID registry and utilities
- api: High-level marginal computation API
"""

__version__ = "1.0.0"
__author__ = "CatBP Team"

from catbp.algebra.semiring import (
    SATSemiring,
    ProbSemiring,
    LogProbSemiring,
    CountingSemiring,
    sat_semiring,
    prob_semiring,
    logprob_semiring,
    counting_semiring,
)
from catbp.algebra.section import Section
from catbp.algebra.support import factor_support, build_factor_supports
from catbp.topology.structure import FactorGraphStructure
from catbp.topology.nerve import Nerve
from catbp.hatcc import hatcc_solve, compute_partition_function, compute_marginals
from catbp.solver import run_hatcc, SolverResult

__all__ = [
    # Semirings
    "SATSemiring",
    "ProbSemiring", 
    "LogProbSemiring",
    "CountingSemiring",
    "sat_semiring",
    "prob_semiring",
    "logprob_semiring",
    "counting_semiring",
    # Sections
    "Section",
    "factor_support",
    "build_factor_supports",
    # Topology
    "FactorGraphStructure",
    "Nerve",
    # Solver
    "hatcc_solve",
    "compute_partition_function",
    "compute_marginals",
    "run_hatcc",
    "SolverResult",
]
