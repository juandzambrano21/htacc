"""
Tests for the HATCC solver.
"""

import numpy as np
import pytest

from catbp.hatcc import hatcc_solve, compute_partition_function, compute_marginals


class TestSimpleChain:
    """Test on a simple chain A--B--C."""
    
    @pytest.fixture
    def chain_model(self):
        var_domains = {"A": 2, "B": 2, "C": 2}
        
        phi_A = np.array([0.6, 0.4])
        phi_AB = np.array([[0.9, 0.1], [0.2, 0.8]])
        phi_BC = np.array([[0.3, 0.7], [0.5, 0.5]])
        
        factors = {
            "f_A": (("A",), phi_A),
            "f_AB": (("A", "B"), phi_AB),
            "f_BC": (("B", "C"), phi_BC),
        }
        
        return var_domains, factors
    
    def test_partition_function(self, chain_model):
        var_domains, factors = chain_model
        
        # Brute force
        phi_A = factors["f_A"][1]
        phi_AB = factors["f_AB"][1]
        phi_BC = factors["f_BC"][1]
        
        Z_brute = 0.0
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    Z_brute += phi_A[a] * phi_AB[a, b] * phi_BC[b, c]
        
        # HATCC
        Z_hatcc = compute_partition_function(var_domains, factors)
        
        assert np.isclose(Z_brute, Z_hatcc, rtol=1e-5)
    
    def test_marginals_sum_to_one(self, chain_model):
        var_domains, factors = chain_model
        
        marginals = compute_marginals(var_domains, factors)
        
        for var, marg in marginals.items():
            assert np.isclose(np.sum(marg), 1.0, rtol=1e-5)


class TestGridModel:
    """Test on a 2x2 grid with a cycle.
    
    Trivial-holonomy cycles fall back to carrying interface vars, so this
    should be exact.
    """
    
    @pytest.fixture
    def grid_model(self):
        var_domains = {"X00": 2, "X01": 2, "X10": 2, "X11": 2}
        
        J = 0.5
        psi = np.array([
            [np.exp(J), np.exp(-J)],
            [np.exp(-J), np.exp(J)]
        ])
        
        factors = {
            "f_00_01": (("X00", "X01"), psi),
            "f_00_10": (("X00", "X10"), psi),
            "f_01_11": (("X01", "X11"), psi),
            "f_10_11": (("X10", "X11"), psi),
        }
        
        return var_domains, factors, psi
    
    def test_partition_function(self, grid_model):
        var_domains, factors, psi = grid_model
        
        # Brute force
        Z_brute = 0.0
        for x00 in range(2):
            for x01 in range(2):
                for x10 in range(2):
                    for x11 in range(2):
                        w = (psi[x00, x01] * psi[x00, x10] * 
                             psi[x01, x11] * psi[x10, x11])
                        Z_brute += w
        
        # HATCC
        Z_hatcc = compute_partition_function(var_domains, factors)
        
        assert np.isclose(Z_brute, Z_hatcc, rtol=1e-5)


class TestSingleFactor:
    """Test with a single factor."""
    
    def test_unary_factor(self):
        var_domains = {"X": 3}
        factors = {
            "f": (("X",), np.array([0.2, 0.5, 0.3])),
        }
        
        Z = compute_partition_function(var_domains, factors)
        
        assert np.isclose(Z, 1.0, rtol=1e-5)
    
    def test_binary_factor(self):
        var_domains = {"X": 2, "Y": 2}
        phi = np.array([[0.1, 0.2], [0.3, 0.4]])
        factors = {
            "f": (("X", "Y"), phi),
        }
        
        Z = compute_partition_function(var_domains, factors)
        
        assert np.isclose(Z, 1.0, rtol=1e-5)


class TestLogProbSemiring:
    """Test with log-probability semiring."""
    
    def test_chain_logprob(self):
        var_domains = {"A": 2, "B": 2}
        
        # Use log-space values
        phi_A = np.array([0.6, 0.4])
        phi_AB = np.array([[0.9, 0.1], [0.2, 0.8]])
        
        factors = {
            "f_A": (("A",), phi_A),
            "f_AB": (("A", "B"), phi_AB),
        }
        
        # Brute force
        Z_brute = 0.0
        for a in range(2):
            for b in range(2):
                Z_brute += phi_A[a] * phi_AB[a, b]
        
        # HATCC with prob semiring
        result = hatcc_solve(var_domains, factors, semiring="prob")
        
        assert np.isclose(Z_brute, result.Z, rtol=1e-5)
