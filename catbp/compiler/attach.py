"""
catbp/compiler/attach.py

Anchor point computation for topological factors.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx


def _rooted_parent_depth(T: nx.Graph, root: str) -> Tuple[Dict[str, Optional[str]], Dict[str, int]]:
    """
    Compute parent and depth for each node in a rooted tree.
    """
    parent: Dict[str, Optional[str]] = {root: None}
    depth: Dict[str, int] = {root: 0}
    stack = [root]
    
    while stack:
        u = stack.pop()
        for v in T.neighbors(u):
            if v in parent:
                continue
            parent[v] = u
            depth[v] = depth[u] + 1
            stack.append(v)
    
    return parent, depth


def _lca(parent: Dict[str, Optional[str]], depth: Dict[str, int], a: str, b: str) -> str:
    """
    Compute lowest common ancestor of two nodes.
    """
    x, y = a, b
    while depth[x] > depth[y]:
        x = parent[x]
    while depth[y] > depth[x]:
        y = parent[y]
    while x != y:
        x = parent[x]
        y = parent[y]
    return x


def lca_anchor_for_nodes(T: nx.Graph, root: str, nodes: List[str]) -> str:
    """
    Compute deterministic anchor for a factor involving multiple backbone nodes.
    
    The anchor is the LCA of all listed nodes in the tree rooted at `root`.
    
    Args:
        T: Backbone tree
        root: Root node
        nodes: Nodes to find LCA of
        
    Returns:
        LCA node
    """
    if not nodes:
        return root
    
    parent, depth = _rooted_parent_depth(T, root)
    cur = nodes[0]
    for n in nodes[1:]:
        cur = _lca(parent, depth, cur, n)
    return cur


# Import Tuple for type hints
from typing import Tuple
