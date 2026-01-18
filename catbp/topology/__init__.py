"""
Topology module: Factor graph structure, nerve, and homology.
"""

from catbp.topology.structure import FactorDef, FactorGraphStructure
from catbp.topology.nerve import Nerve, build_nerve_graph
from catbp.topology.simplicial import Nerve2Skeleton, build_2_skeleton_from_scopes, build_2_skeleton_from_nerve_graph
from catbp.topology.homology import nontrivial_h1_chords

__all__ = [
    "FactorDef",
    "FactorGraphStructure",
    "Nerve",
    "build_nerve_graph",
    "Nerve2Skeleton",
    "build_2_skeleton_from_scopes",
    "build_2_skeleton_from_nerve_graph",
    "nontrivial_h1_chords",
]
