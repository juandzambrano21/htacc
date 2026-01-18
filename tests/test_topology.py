"""
Tests for topology module.
"""

import numpy as np
import pytest

from catbp.topology.structure import FactorGraphStructure, FactorDef
from catbp.topology.nerve import Nerve
from catbp.topology.simplicial import build_2_skeleton_from_scopes, Nerve2Skeleton
from catbp.topology.homology import nontrivial_h1_chords


class TestFactorGraphStructure:
    def test_add_variable(self):
        struct = FactorGraphStructure()
        struct.add_variable("X", 2)
        struct.add_variable("Y", 3)
        
        assert struct.var_card["X"] == 2
        assert struct.var_card["Y"] == 3
    
    def test_add_factor(self):
        struct = FactorGraphStructure()
        struct.add_variable("X", 2)
        struct.add_variable("Y", 3)
        struct.add_factor("f1", ["X", "Y"])
        
        assert "f1" in struct.factors
        assert struct.factors["f1"].scope == ("X", "Y")  # sorted
    
    def test_interface(self):
        struct = FactorGraphStructure()
        struct.add_variable("X", 2)
        struct.add_variable("Y", 3)
        struct.add_variable("Z", 2)
        struct.add_factor("f1", ["X", "Y"])
        struct.add_factor("f2", ["Y", "Z"])
        
        interface = struct.interface("f1", "f2")
        assert interface == ("Y",)


class TestNerve:
    def test_chain_nerve(self):
        struct = FactorGraphStructure()
        struct.add_variable("A", 2)
        struct.add_variable("B", 2)
        struct.add_variable("C", 2)
        struct.add_factor("f1", ["A", "B"])
        struct.add_factor("f2", ["B", "C"])
        
        nerve = Nerve(struct)
        
        assert nerve.g.has_node("f1")
        assert nerve.g.has_node("f2")
        assert nerve.g.has_edge("f1", "f2")
        assert nerve.interface("f1", "f2") == ("B",)
    
    def test_triangle_nerve(self):
        struct = FactorGraphStructure()
        struct.add_variable("X", 2)
        struct.add_variable("Y", 2)
        struct.add_variable("Z", 2)
        struct.add_factor("f1", ["X", "Y"])
        struct.add_factor("f2", ["Y", "Z"])
        struct.add_factor("f3", ["X", "Z"])
        
        nerve = Nerve(struct)
        
        # All pairs should be connected
        assert nerve.g.number_of_edges() == 3


class TestSimplicial:
    def test_2_skeleton_chain(self):
        factor_scopes = {
            "f1": ("A", "B"),
            "f2": ("B", "C"),
        }
        
        sk = build_2_skeleton_from_scopes(factor_scopes)
        
        assert len(sk.vertices) == 2
        assert len(sk.edges) == 1
        assert len(sk.triangles) == 0  # No triple intersection
    
    def test_2_skeleton_triangle(self):
        factor_scopes = {
            "f1": ("X", "Y", "Z"),
            "f2": ("X", "Y"),
            "f3": ("Y", "Z"),
        }
        
        sk = build_2_skeleton_from_scopes(factor_scopes)
        
        # f1-f2, f1-f3, f2-f3 all have nonempty intersection
        assert len(sk.edges) == 3
        # Triple intersection exists (Y is in all three)
        assert len(sk.triangles) == 1


class TestHomology:
    def test_tree_no_chords(self):
        import networkx as nx
        
        # Tree graph: no chords
        nerve = nx.Graph()
        nerve.add_edge("f1", "f2")
        nerve.add_edge("f2", "f3")
        
        tree = nerve.copy()  # Tree = nerve
        
        chords = nontrivial_h1_chords(tree=tree, nerve=nerve, triangles=[])
        assert len(chords) == 0
    
    def test_cycle_has_chord(self):
        import networkx as nx
        
        # Cycle: f1 - f2 - f3 - f1
        nerve = nx.Graph()
        nerve.add_edge("f1", "f2")
        nerve.add_edge("f2", "f3")
        nerve.add_edge("f3", "f1")
        
        # Tree: f1 - f2 - f3
        tree = nx.Graph()
        tree.add_edge("f1", "f2")
        tree.add_edge("f2", "f3")
        
        chords = nontrivial_h1_chords(tree=tree, nerve=nerve, triangles=[])
        assert len(chords) == 1
        assert ("f1", "f3") in chords or ("f3", "f1") in chords
