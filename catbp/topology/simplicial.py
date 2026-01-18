"""
catbp/topology/simplicial.py

Kan-relevant 2-skeleton of the nerve for homology computation.

The 2-skeleton consists of:
- Vertices: factors
- Edges: pairwise nonempty intersections
- Triangles: triple nonempty intersections (2-simplex fillers)

Triangles are what kill 1-cycles in H1 via boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Set, Tuple

import networkx as nx

Node = Any
Var = Any


@dataclass(frozen=True)
class Nerve2Skeleton:
    """
    Kan-relevant 2-skeleton of the nerve.
    
    Attributes:
        vertices: Factor nodes
        edges: Pairs (u, v) with u < v having non-empty intersection
        triangles: Triples (a, b, c) with a < b < c having non-empty triple intersection
    """
    vertices: Tuple[Node, ...]
    edges: Tuple[Tuple[Node, Node], ...]
    triangles: Tuple[Tuple[Node, Node, Node], ...]


def build_2_skeleton_from_scopes(
    factor_scopes: Dict[Node, Tuple[Var, ...]],
) -> Nerve2Skeleton:
    """
    Build 2-skeleton directly from factor scopes.
    
    Args:
        factor_scopes: Map from factor to its scope (tuple of variables)
        
    Returns:
        Nerve2Skeleton with vertices, edges, and triangles
    """
    nodes = tuple(sorted(factor_scopes.keys()))

    # Edges by pairwise intersection
    edges: List[Tuple[Node, Node]] = []
    for u, v in combinations(nodes, 2):
        if set(factor_scopes[u]).intersection(factor_scopes[v]):
            edges.append((u, v))
    edges = sorted(edges)

    # Triangles by triple intersection (Kan 2-simplex fillers)
    triangles: List[Tuple[Node, Node, Node]] = []
    for a, b, c in combinations(nodes, 3):
        inter = set(factor_scopes[a]).intersection(factor_scopes[b]).intersection(factor_scopes[c])
        if inter:
            triangles.append(tuple(sorted((a, b, c))))
    triangles = sorted(set(triangles))

    return Nerve2Skeleton(vertices=nodes, edges=tuple(edges), triangles=tuple(triangles))


def build_2_skeleton_from_nerve_graph(
    nerve: nx.Graph,
    factor_scopes: Dict[Node, Tuple[Var, ...]],
) -> Nerve2Skeleton:
    """
    Build 2-skeleton using existing nerve 1-skeleton for edges.
    
    Args:
        nerve: NetworkX graph representing the nerve 1-skeleton
        factor_scopes: Map from factor to its scope
        
    Returns:
        Nerve2Skeleton with vertices, edges, and triangles
    """
    vertices = tuple(sorted(nerve.nodes()))

    # Edges from nerve graph
    edges = []
    for u, v in nerve.edges():
        a, b = (u, v) if u <= v else (v, u)
        edges.append((a, b))
    edges = tuple(sorted(set(edges)))

    # Triangles: candidate triangles from graph cliques of size 3, then verify triple intersection
    triangles: List[Tuple[Node, Node, Node]] = []
    nbrs: Dict[Node, Set[Node]] = {u: set(nerve.neighbors(u)) for u in vertices}
    
    for a in vertices:
        for b in nbrs[a]:
            if b <= a:
                continue
            common = nbrs[a].intersection(nbrs[b])
            for c in common:
                if c <= b:
                    continue
                inter = set(factor_scopes[a]).intersection(factor_scopes[b]).intersection(factor_scopes[c])
                if inter:
                    triangles.append((a, b, c))

    triangles = tuple(sorted(set(tuple(sorted(t)) for t in triangles)))
    return Nerve2Skeleton(vertices=vertices, edges=edges, triangles=triangles)
