"""
catbp/compiler/backbone.py

Backbone tree selection from nerve graph.

The backbone is a spanning tree of the nerve graph, chosen to minimize
message complexity (maximum spanning tree by interface size).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import networkx as nx

from catbp.topology.nerve import Nerve
from catbp.topology.structure import FactorGraphStructure


def edge_weight_log_interface_size(struct: FactorGraphStructure, interface_vars: Tuple[str, ...]) -> float:
    """
    Compute edge weight as log of interface size.
    
    Larger interfaces = larger weight = preferred for MST.
    """
    s = 0.0
    for v in interface_vars:
        s += math.log(max(1, struct.var_card.get(v, 1)))
    return s


def choose_backbone_tree_from_nerve(nerve: Nerve) -> nx.Graph:
    """
    Choose backbone tree from Nerve object.
    
    Uses maximum spanning tree on log interface size weights.
    """
    G = nerve.g.copy()
    for a, b, data in G.edges(data=True):
        data["w"] = edge_weight_log_interface_size(nerve.struct, data["interface"])
    T = nx.maximum_spanning_tree(G, weight="w")
    return T


def choose_backbone_tree(nerve: nx.Graph) -> nx.Graph:
    """
    Choose backbone tree from nerve graph.
    
    Uses maximum spanning tree by 'weight' attribute.
    """
    return nx.maximum_spanning_tree(nerve, weight="weight")


def root_tree(tree: nx.Graph, root: int) -> Tuple[Dict[int, Optional[int]], Dict[int, List[int]]]:
    """
    Root a tree at a given node.
    
    Args:
        tree: Undirected tree graph
        root: Root node
        
    Returns:
        (parent, children) where:
        - parent[node] = parent node (None for root)
        - children[node] = list of child nodes
    """
    parent: Dict[int, Optional[int]] = {root: None}
    children: Dict[int, List[int]] = {root: []}
    
    stack = [root]
    while stack:
        u = stack.pop()
        for v in tree.neighbors(u):
            if v in parent:
                continue
            parent[v] = u
            children.setdefault(u, []).append(v)
            children.setdefault(v, [])
            stack.append(v)
    
    return parent, children


def compute_euler_tour(
    parent: Dict[int, Optional[int]],
    children: Dict[int, List[int]],
    root: int
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Compute Euler tour timestamps for subtree queries.
    
    Args:
        parent: Parent mapping
        children: Children mapping
        root: Root node
        
    Returns:
        (tin, tout) where:
        - tin[node] = entry time
        - tout[node] = exit time
    """
    tin: Dict[int, int] = {}
    tout: Dict[int, int] = {}
    t = 0
    
    def dfs(u: int) -> None:
        nonlocal t
        tin[u] = t
        t += 1
        for c in children.get(u, []):
            dfs(c)
        tout[u] = t
    
    dfs(root)
    return tin, tout


def in_subtree(tin: Dict[int, int], tout: Dict[int, int], subroot: int, node: int) -> bool:
    """
    Check if node is in the subtree rooted at subroot.
    
    Uses Euler tour timestamps for O(1) query.
    """
    return tin[subroot] <= tin[node] < tout[subroot]


def path_in_tree(tree: nx.Graph, u: int, v: int) -> List[int]:
    """
    Get the unique simple path between two nodes in a tree.
    """
    return nx.shortest_path(tree, source=u, target=v)
