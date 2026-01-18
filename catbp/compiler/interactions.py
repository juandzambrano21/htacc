"""
catbp/compiler/interactions.py

Chord interaction detection for mode constraint generation.
"""

from __future__ import annotations

from typing import List, Set, Tuple

import networkx as nx

Chord = Tuple[str, str]


def chord_tree_path_nodes(T: nx.Graph, u: str, v: str) -> Set[str]:
    """
    Get the set of nodes on the tree path between u and v.
    """
    path = nx.shortest_path(T, u, v)
    return set(path)


def chord_interaction_pairs(T: nx.Graph, chords: List[Chord]) -> List[Tuple[Chord, Chord]]:
    """
    Find interacting chord pairs based on tree path overlap.
    
    Two chords interact if their tree paths share at least one node:
        e ~ f iff P_e ∩ P_f ≠ ∅
    
    Args:
        T: Backbone tree
        chords: List of chord edges
        
    Returns:
        List of interacting chord pairs
    """
    paths = []
    for (u, v) in chords:
        paths.append(((u, v), chord_tree_path_nodes(T, u, v)))
    
    out: List[Tuple[Chord, Chord]] = []
    for i in range(len(paths)):
        e, Pe = paths[i]
        for j in range(i + 1, len(paths)):
            f, Pf = paths[j]
            if Pe.intersection(Pf):
                out.append((e, f))
    return out
