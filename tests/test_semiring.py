"""
Tests for semiring operations.
"""

import numpy as np
import pytest

from catbp.algebra.semiring import (
    SATSemiring,
    ProbSemiring,
    LogProbSemiring,
    CountingSemiring,
    sat_semiring,
    prob_semiring,
    logprob_semiring,
    counting_semiring,
)


class TestSATSemiring:
    def test_identities(self):
        sr = SATSemiring()
        assert sr.zero == False
        assert sr.one == True
    
    def test_add(self):
        sr = SATSemiring()
        assert sr.add(False, False) == False
        assert sr.add(False, True) == True
        assert sr.add(True, False) == True
        assert sr.add(True, True) == True
    
    def test_mul(self):
        sr = SATSemiring()
        assert sr.mul(False, False) == False
        assert sr.mul(False, True) == False
        assert sr.mul(True, False) == False
        assert sr.mul(True, True) == True
    
    def test_is_zero(self):
        sr = SATSemiring()
        assert sr.is_zero(False) == True
        assert sr.is_zero(True) == False
    
    def test_add_reduce(self):
        sr = SATSemiring()
        x = np.array([[True, False], [False, False]])
        assert sr.add_reduce(x, axis=0).tolist() == [True, False]
        assert sr.add_reduce(x, axis=1).tolist() == [True, False]


class TestProbSemiring:
    def test_identities(self):
        sr = ProbSemiring()
        assert sr.zero == 0.0
        assert sr.one == 1.0
    
    def test_add(self):
        sr = ProbSemiring()
        assert sr.add(0.3, 0.5) == pytest.approx(0.8)
    
    def test_mul(self):
        sr = ProbSemiring()
        assert sr.mul(0.3, 0.5) == pytest.approx(0.15)
    
    def test_normalize(self):
        sr = ProbSemiring()
        x = np.array([1.0, 2.0, 3.0])
        normalized = sr.normalize(x)
        assert np.sum(normalized) == pytest.approx(1.0)


class TestLogProbSemiring:
    def test_identities(self):
        sr = LogProbSemiring()
        assert sr.zero == -np.inf
        assert sr.one == 0.0
    
    def test_mul(self):
        sr = LogProbSemiring()
        # log(a*b) = log(a) + log(b)
        assert sr.mul(np.log(0.3), np.log(0.5)) == pytest.approx(np.log(0.15))
    
    def test_add(self):
        sr = LogProbSemiring()
        # log(a+b) = logsumexp(log(a), log(b))
        result = sr.add(np.log(0.3), np.log(0.5))
        assert np.exp(result) == pytest.approx(0.8)


class TestSemiringRuntime:
    def test_sat_runtime(self):
        sr = sat_semiring()
        assert sr.name == "SAT"
        assert sr.dtype == np.bool_
        
        x = sr.one((2, 2))
        assert x.shape == (2, 2)
        assert x.all()
    
    def test_prob_runtime(self):
        sr = prob_semiring()
        assert sr.name == "PROB"
        
        x = sr.one((3,))
        assert np.allclose(x, [1.0, 1.0, 1.0])
        
        y = sr.zero((3,))
        assert np.allclose(y, [0.0, 0.0, 0.0])
    
    def test_logprob_runtime(self):
        sr = logprob_semiring()
        assert sr.name == "LOGPROB"
        
        x = sr.one((2,))
        assert np.allclose(x, [0.0, 0.0])  # log(1) = 0
        
        y = sr.zero((2,))
        assert np.all(np.isneginf(y))  # log(0) = -inf
