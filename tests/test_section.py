"""
Tests for Section operations.
"""

import numpy as np
import pytest

from catbp.algebra.section import Section
from catbp.algebra.semiring import ProbSemiring, SATSemiring


class TestSection:
    def test_creation(self):
        sr = ProbSemiring()
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        sec = Section(domain=("A", "B"), data=data, semiring=sr)
        
        assert sec.domain == ("A", "B")
        assert sec.data.shape == (2, 2)
    
    def test_domain_mismatch_raises(self):
        sr = ProbSemiring()
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        with pytest.raises(ValueError):
            Section(domain=("A",), data=data, semiring=sr)
    
    def test_duplicate_domain_raises(self):
        sr = ProbSemiring()
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        with pytest.raises(ValueError):
            Section(domain=("A", "A"), data=data, semiring=sr)
    
    def test_unit(self):
        sr = ProbSemiring()
        sec = Section.unit(domain=("X", "Y"), shape=(2, 3), semiring=sr)
        
        assert sec.domain == ("X", "Y")
        assert sec.data.shape == (2, 3)
        assert np.all(sec.data == 1.0)
    
    def test_star_same_domain(self):
        sr = ProbSemiring()
        f = Section(domain=("A",), data=np.array([2.0, 3.0]), semiring=sr)
        g = Section(domain=("A",), data=np.array([4.0, 5.0]), semiring=sr)
        
        h = f.star(g)
        assert h.domain == ("A",)
        assert np.allclose(h.data, [8.0, 15.0])
    
    def test_star_disjoint_domains(self):
        sr = ProbSemiring()
        f = Section(domain=("A",), data=np.array([2.0, 3.0]), semiring=sr)
        g = Section(domain=("B",), data=np.array([4.0, 5.0]), semiring=sr)
        
        h = f.star(g)
        assert set(h.domain) == {"A", "B"}
        # Result should be outer product
        assert h.data.shape == (2, 2)
    
    def test_restrict(self):
        sr = ProbSemiring()
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        sec = Section(domain=("A", "B"), data=data, semiring=sr)
        
        # Marginalize out B
        marg = sec.restrict(("A",))
        assert marg.domain == ("A",)
        assert np.allclose(marg.data, [3.0, 7.0])  # sum over B
    
    def test_restrict_to_empty(self):
        sr = ProbSemiring()
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        sec = Section(domain=("A", "B"), data=data, semiring=sr)
        
        # Marginalize out everything
        marg = sec.restrict(())
        assert marg.domain == ()
        assert np.isclose(marg.data, 10.0)  # sum of all


class TestSectionBoolean:
    def test_sat_star(self):
        sr = SATSemiring()
        f = Section(domain=("A",), data=np.array([True, False]), semiring=sr)
        g = Section(domain=("A",), data=np.array([True, True]), semiring=sr)
        
        h = f.star(g)
        assert h.data.tolist() == [True, False]

    def test_sat_star_separate_instances(self):
        f = Section(domain=("A",), data=np.array([True, False]), semiring=SATSemiring())
        g = Section(domain=("A",), data=np.array([True, True]), semiring=SATSemiring())

        h = f.star(g)
        assert h.data.tolist() == [True, False]
    
    def test_sat_restrict(self):
        sr = SATSemiring()
        data = np.array([[True, False], [False, True]])
        sec = Section(domain=("A", "B"), data=data, semiring=sr)
        
        # OR over B
        marg = sec.restrict(("A",))
        assert marg.data.tolist() == [True, True]
