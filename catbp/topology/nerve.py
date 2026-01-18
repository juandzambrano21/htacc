"""
catbp/topology/nerve.py

Nerve graph construction from factor graph structure.

The nerve graph G_N = (F, E_N) has:
- Nodes: factors
- Edges: pairs of factors with non-empty intersection
- Edge labels: interface variables I_{ab} = S_a ∩ S_b
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import networkx as nx

from catbp.topology.structure import FactorGraphStructure


class Nerve:
    """
    Nerve graph of a factor graph.
    
    Nodes are factors, edges connect factors sharing variables.
    Each edge is labeled with the interface (shared variables).
    """
    
    def __init__(self, struct: FactorGraphStructure):
        self.struct = struct
        self.g = nx.Graph()
        self._build()

    def _build(self) -> None:
        """Build the nerve graph from factor structure."""
        # Add all factors as nodes
        for f in self.struct.factors:
            self.g.add_node(f)

        # Add edges between factors sharing variables
        seen = set()
        for v, fs in self.struct.var_to_factors.items():
            for i in range(len(fs)):
                for j in range(i + 1, len(fs)):
                    a, b = fs[i], fs[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) in seen:
                        continue
                    seen.add((a, b))
                    I = self.struct.interface(a, b)
                    if I:
                        # Compute weight as log of interface size
                        w = sum(math.log(max(1, self.struct.var_card.get(v, 1))) for v in I)
                        self.g.add_edge(a, b, interface=I, weight=w)

    def interface(self, a: str, b: str) -> Tuple[str, ...]:
        """Get the interface between two factors."""
        data = self.g.get_edge_data(a, b)
        if data is None:
            return ()
        return data["interface"]

    def neighbors(self, f: str):
        """Get neighbors of a factor in the nerve."""
        return self.g.neighbors(f)

    def edges(self):
        """Get all edges in the nerve."""
        return self.g.edges()

    def nodes(self):
        """Get all nodes in the nerve."""
        return self.g.nodes()


def build_nerve_graph(
    factor_scopes: Dict[int, Tuple[int, ...]],
    var_domains: Dict[int, int]
) -> nx.Graph:
    """
    Build nerve graph from factor scopes (using integer IDs).
    
    Args:
        factor_scopes: Map from factor ID to tuple of variable IDs in scope
        var_domains: Map from variable ID to domain size
        
    Returns:
        NetworkX graph with:
        - Nodes: factor IDs
        - Edges: pairs with non-empty intersection
        - Edge attrs: interface (tuple of VarIDs), weight (log interface size)
    """
    g = nx.Graph()
    
    # Add all factors as nodes
    for f in factor_scopes.keys():
        g.add_node(f)

    # Add edges between factors with non-empty intersection
    fac_ids = sorted(factor_scopes.keys())
    for i in range(len(fac_ids)):
        a = fac_ids[i]
        Sa = set(factor_scopes[a])
        for j in range(i + 1, len(fac_ids)):
            b = fac_ids[j]
            inter = tuple(sorted(Sa.intersection(factor_scopes[b])))
            if not inter:
                continue
            # Weight = sum of log domain sizes (≈ log product)
            w = 0.0
            for v in inter:
                w += math.log(max(var_domains.get(v, 1), 1))
            g.add_edge(a, b, interface=inter, weight=w)

    return g
