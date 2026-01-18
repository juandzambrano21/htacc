"""
catbp/core/registry.py

ID registry for variables, factors, and modes.

Maintains disjoint integer spaces to prevent ID collisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class IDRegistry:
    """
    Registry for mapping names to IDs.
    
    Attributes:
        var_name_to_id: Variable name -> ID
        fac_name_to_id: Factor name -> ID
        id_to_var_name: ID -> variable name
        id_to_fac_name: ID -> factor name
        next_mode_id: Next available mode ID
    """
    var_name_to_id: Dict[str, int]
    fac_name_to_id: Dict[str, int]
    id_to_var_name: List[str]
    id_to_fac_name: List[str]
    next_mode_id: int

    @staticmethod
    def build(var_domains: Dict[str, int], factors: Dict[str, Tuple[str, ...]]) -> "IDRegistry":
        """
        Build a registry from variable domains and factor scopes.
        
        Args:
            var_domains: Map from variable name to domain size
            factors: Map from factor name to scope (tuple of variable names)
            
        Returns:
            IDRegistry with assigned IDs
        """
        var_names = sorted(var_domains.keys())
        fac_names = sorted(factors.keys())

        var_name_to_id = {n: i for i, n in enumerate(var_names)}
        fac_name_to_id = {n: i for i, n in enumerate(fac_names)}

        # Allocate mode IDs after all vars to reduce accidental overlaps
        next_mode_id = len(var_names)

        return IDRegistry(
            var_name_to_id=var_name_to_id,
            fac_name_to_id=fac_name_to_id,
            id_to_var_name=var_names,
            id_to_fac_name=fac_names,
            next_mode_id=next_mode_id
        )

    def alloc_mode_id(self) -> int:
        """Allocate a new mode ID."""
        mid = self.next_mode_id
        self.next_mode_id += 1
        return mid

    def var_id(self, name: str) -> int:
        """Get variable ID by name."""
        return self.var_name_to_id[name]

    def fac_id(self, name: str) -> int:
        """Get factor ID by name."""
        return self.fac_name_to_id[name]

    def var_name(self, vid: int) -> str:
        """Get variable name by ID."""
        return self.id_to_var_name[vid]

    def fac_name(self, fid: int) -> str:
        """Get factor name by ID."""
        return self.id_to_fac_name[fid]
