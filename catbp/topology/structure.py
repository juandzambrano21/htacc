"""
catbp/topology/structure.py

Factor graph structure with canonical scopes.

A factor graph consists of:
- Variables with domains (cardinalities)
- Factors with scopes (sets of variables they depend on)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class FactorDef:
    """Definition of a factor with its scope."""
    name: str
    scope: Tuple[str, ...]  # Canonical sorted variable names

    def __post_init__(self):
        # Ensure scope is sorted for canonical representation
        if self.scope != tuple(sorted(self.scope)):
            object.__setattr__(self, 'scope', tuple(sorted(self.scope)))


class FactorGraphStructure:
    """
    Structure of a factor graph (topology only, no values).
    
    Maintains:
    - Variable cardinalities
    - Factor definitions with scopes
    - Variable-to-factor incidence
    """
    
    def __init__(self):
        self.var_card: Dict[str, int] = {}
        self.factors: Dict[str, FactorDef] = {}
        self.var_to_factors: Dict[str, List[str]] = {}

    def add_variable(self, name: str, card: int) -> None:
        """Add a variable with given cardinality."""
        self.var_card[name] = card
        self.var_to_factors.setdefault(name, [])

    def add_factor(self, name: str, scope: List[str]) -> None:
        """Add a factor with given scope (will be canonicalized)."""
        scope_c = tuple(sorted(scope))
        self.factors[name] = FactorDef(name, scope_c)
        for v in scope_c:
            self.var_to_factors.setdefault(v, []).append(name)

    def interface(self, f1: str, f2: str) -> Tuple[str, ...]:
        """Get the interface (shared variables) between two factors."""
        s1 = set(self.factors[f1].scope)
        s2 = set(self.factors[f2].scope)
        return tuple(sorted(s1.intersection(s2)))

    def get_scope(self, factor_name: str) -> Tuple[str, ...]:
        """Get the scope of a factor."""
        return self.factors[factor_name].scope

    def get_domain(self, var_name: str) -> int:
        """Get the domain size of a variable."""
        return self.var_card[var_name]

    def all_variables(self) -> List[str]:
        """Get all variable names."""
        return list(self.var_card.keys())

    def all_factors(self) -> List[str]:
        """Get all factor names."""
        return list(self.factors.keys())

    def factors_containing(self, var_name: str) -> List[str]:
        """Get all factors containing a variable."""
        return self.var_to_factors.get(var_name, [])

    def __repr__(self) -> str:
        return f"FactorGraphStructure(vars={len(self.var_card)}, factors={len(self.factors)})"
