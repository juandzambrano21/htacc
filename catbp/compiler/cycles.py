"""
catbp/compiler/cycles.py

Typed chord cycles for holonomy computation.

Each chord (edge not in backbone tree) defines a fundamental cycle.
The cycle is typed with transport steps specifying interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx

from catbp.topology.nerve import Nerve
from catbp.topology.simplicial import build_2_skeleton_from_nerve_graph
from catbp.topology.homology import nontrivial_h1_chords


@dataclass(frozen=True)
class TransportStep:
    """
    A single step in a holonomy cycle.
    
    Attributes:
        factor: Factor name/ID
        in_interface: Input interface variables (canonical order)
        out_interface: Output interface variables (canonical order)
    """
    factor: str
    in_interface: Tuple[str, ...]
    out_interface: Tuple[str, ...]


@dataclass(frozen=True)
class HolonomyCycle:
    """
    A complete holonomy cycle around a chord.
    
    Attributes:
        chord: (u, v) factor endpoints
        chord_interface: J_e = S_u ∩ S_v
        steps: Ordered transport steps around the loop
    """
    chord: Tuple[str, str]
    chord_interface: Tuple[str, ...]
    steps: Tuple[TransportStep, ...]


def build_fundamental_cycle(nerve: Nerve, T: nx.Graph, u: str, v: str) -> HolonomyCycle:
    """
    Build the fundamental cycle for a chord (u, v).
    
    The cycle traverses the tree path from u to v, then closes via the chord.
    
    Args:
        nerve: Nerve graph
        T: Backbone tree
        u, v: Chord endpoints
        
    Returns:
        HolonomyCycle with typed transport steps
    """
    J = nerve.interface(u, v)
    path = nx.shortest_path(T, source=u, target=v)  # u = p0 ... pk = v
    
    # Cycle order: u -> p1 -> ... -> v -> u (via chord)
    cycle_nodes = path + [u]
    
    steps: List[TransportStep] = []
    curr_I = J
    
    for i in range(len(cycle_nodes) - 1):
        a = cycle_nodes[i]
        b = cycle_nodes[i + 1]
        next_I = nerve.interface(a, b)
        steps.append(TransportStep(factor=a, in_interface=curr_I, out_interface=next_I))
        curr_I = next_I
    
    # Verify closure
    assert steps[0].in_interface == J
    assert steps[-1].out_interface == J
    
    return HolonomyCycle(chord=(u, v), chord_interface=J, steps=tuple(steps))


def select_chords_via_h1(nerve: Nerve, tree: nx.Graph) -> List[Tuple[str, str]]:
    """
    Select chords that are non-trivial in H1.
    
    Uses homology computation to filter out boundary cycles.
    
    Args:
        nerve: Nerve graph
        tree: Backbone tree
        
    Returns:
        List of chord edges (u, v) with u < v
    """
    factor_scopes = {f: tuple(nerve.struct.factors[f].scope) for f in nerve.struct.factors.keys()}
    sk2 = build_2_skeleton_from_nerve_graph(nerve.g, factor_scopes)
    kept = nontrivial_h1_chords(tree=tree, nerve=nerve.g, triangles=sk2.triangles)
    return kept


def build_h1_cycles(nerve: Nerve, T: nx.Graph) -> List[HolonomyCycle]:
    """
    Build holonomy cycles only for homology-essential chords.
    
    Args:
        nerve: Nerve graph
        T: Backbone tree
        
    Returns:
        List of HolonomyCycle objects
    """
    chords = select_chords_via_h1(nerve, T)
    out: List[HolonomyCycle] = []
    for (u, v) in chords:
        out.append(build_fundamental_cycle(nerve, T, u, v))
    return out


def chord_cycle_steps(
    tree: nx.Graph,
    nerve: nx.Graph,
    factor_scopes: Dict[int, Tuple[int, ...]],
    chord_u: int,
    chord_v: int
) -> Tuple[Tuple[int, ...], List[Tuple[int, Tuple[int, ...], Tuple[int, ...]]]]:
    """
    Build cycle steps for a chord using integer IDs.
    
    Args:
        tree: Backbone tree
        nerve: Nerve graph
        factor_scopes: Map from factor ID to scope (variable IDs)
        chord_u, chord_v: Chord endpoint factor IDs
        
    Returns:
        (J_e, steps) where:
        - J_e: chord interface (variable IDs)
        - steps: list of (factor_id, in_interface, out_interface)
    """
    # Chord interface is intersection of endpoint scopes
    J_e = tuple(sorted(set(factor_scopes[chord_u]).intersection(factor_scopes[chord_v])))
    
    # Path nodes including endpoints
    path = nx.shortest_path(tree, source=chord_u, target=chord_v)
    cycle_nodes = path[:]
    
    steps = []
    curr_in = J_e
    
    for i in range(len(cycle_nodes)):
        a = cycle_nodes[i]
        if i == len(cycle_nodes) - 1:
            # Last is v, out to chord interface to close loop
            next_out = J_e
        else:
            b = cycle_nodes[i + 1]
            # Interface between adjacent tree factors
            edge = nerve.get_edge_data(a, b)
            assert edge is not None
            next_out = tuple(edge["interface"])
        
        # Typing invariant: curr_in ⊆ scope(a), next_out ⊆ scope(a)
        Sa = set(factor_scopes[a])
        assert set(curr_in).issubset(Sa), f"curr_in {curr_in} not subset of scope {Sa}"
        assert set(next_out).issubset(Sa), f"next_out {next_out} not subset of scope {Sa}"
        
        steps.append((a, curr_in, next_out))
        curr_in = next_out
    
    return J_e, steps
