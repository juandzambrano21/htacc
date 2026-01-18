"""
Tests for VM operations.
"""

import numpy as np
import pytest

from catbp.ir.schema import Axis, AxisType, TensorSpec
from catbp.vm.semiring import vm_prob_semiring, vm_sat_semiring
from catbp.vm.kernels import vm_unit, vm_contract, vm_eliminate_keep
from catbp.vm.align import align_array_to


class TestAlign:
    def test_identity_alignment(self):
        spec = TensorSpec(axes=(
            Axis(id=0, kind=AxisType.GEOMETRIC, size=2),
            Axis(id=1, kind=AxisType.GEOMETRIC, size=3),
        ))
        
        arr = np.arange(6).reshape(2, 3)
        result = align_array_to(arr, spec, spec)
        
        assert result.shape == (2, 3)
        assert np.allclose(result, arr)
    
    def test_add_singleton_axis(self):
        in_spec = TensorSpec(axes=(
            Axis(id=0, kind=AxisType.GEOMETRIC, size=2),
        ))
        out_spec = TensorSpec(axes=(
            Axis(id=0, kind=AxisType.GEOMETRIC, size=2),
            Axis(id=1, kind=AxisType.GEOMETRIC, size=3),
        ))
        
        arr = np.array([1.0, 2.0])
        result = align_array_to(arr, in_spec, out_spec)
        
        assert result.shape == (2, 1)
    
    def test_reorder_axes(self):
        in_spec = TensorSpec(axes=(
            Axis(id=1, kind=AxisType.GEOMETRIC, size=3),
            Axis(id=0, kind=AxisType.GEOMETRIC, size=2),
        ))
        out_spec = TensorSpec(axes=(
            Axis(id=0, kind=AxisType.GEOMETRIC, size=2),
            Axis(id=1, kind=AxisType.GEOMETRIC, size=3),
        ))
        
        arr = np.arange(6).reshape(3, 2)
        result = align_array_to(arr, in_spec, out_spec)
        
        assert result.shape == (2, 3)


class TestKernels:
    def test_vm_unit(self):
        sr = vm_prob_semiring()
        spec = TensorSpec(axes=(
            Axis(id=0, kind=AxisType.GEOMETRIC, size=2),
            Axis(id=1, kind=AxisType.GEOMETRIC, size=3),
        ))
        
        result = vm_unit(spec, sr)
        
        assert result.shape == (2, 3)
        assert np.all(result == 1.0)
    
    def test_vm_contract_single(self):
        sr = vm_prob_semiring()
        spec = TensorSpec(axes=(
            Axis(id=0, kind=AxisType.GEOMETRIC, size=2),
        ))
        
        arr = np.array([2.0, 3.0])
        inputs = [(arr, spec)]
        
        result = vm_contract(inputs, spec, sr)
        
        assert np.allclose(result, [2.0, 3.0])
    
    def test_vm_contract_multiply(self):
        sr = vm_prob_semiring()
        spec = TensorSpec(axes=(
            Axis(id=0, kind=AxisType.GEOMETRIC, size=2),
        ))
        
        arr1 = np.array([2.0, 3.0])
        arr2 = np.array([4.0, 5.0])
        inputs = [(arr1, spec), (arr2, spec)]
        
        result = vm_contract(inputs, spec, sr)
        
        assert np.allclose(result, [8.0, 15.0])
    
    def test_vm_eliminate_keep(self):
        sr = vm_prob_semiring()
        in_spec = TensorSpec(axes=(
            Axis(id=0, kind=AxisType.GEOMETRIC, size=2),
            Axis(id=1, kind=AxisType.GEOMETRIC, size=3),
        ))
        
        arr = np.arange(6, dtype=np.float64).reshape(2, 3)
        keep_keys = ((AxisType.GEOMETRIC, 0),)
        
        result, out_spec = vm_eliminate_keep(arr, in_spec, keep_keys, sr)
        
        assert result.shape == (2,)
        assert np.allclose(result, [3.0, 12.0])  # sum over axis 1
    
    def test_vm_eliminate_all(self):
        sr = vm_prob_semiring()
        in_spec = TensorSpec(axes=(
            Axis(id=0, kind=AxisType.GEOMETRIC, size=2),
            Axis(id=1, kind=AxisType.GEOMETRIC, size=3),
        ))
        
        arr = np.arange(6, dtype=np.float64).reshape(2, 3)
        keep_keys = ()
        
        result, out_spec = vm_eliminate_keep(arr, in_spec, keep_keys, sr)
        
        assert result.shape == ()
        assert np.isclose(result, 15.0)  # sum of 0+1+2+3+4+5


class TestSATKernels:
    def test_sat_contract(self):
        sr = vm_sat_semiring()
        spec = TensorSpec(axes=(
            Axis(id=0, kind=AxisType.GEOMETRIC, size=2),
        ))
        
        arr1 = np.array([True, False])
        arr2 = np.array([True, True])
        inputs = [(arr1, spec), (arr2, spec)]
        
        result = vm_contract(inputs, spec, sr)
        
        assert result.tolist() == [True, False]
    
    def test_sat_eliminate(self):
        sr = vm_sat_semiring()
        in_spec = TensorSpec(axes=(
            Axis(id=0, kind=AxisType.GEOMETRIC, size=2),
            Axis(id=1, kind=AxisType.GEOMETRIC, size=2),
        ))
        
        arr = np.array([[True, False], [False, True]])
        keep_keys = ((AxisType.GEOMETRIC, 0),)
        
        result, out_spec = vm_eliminate_keep(arr, in_spec, keep_keys, sr)
        
        assert result.tolist() == [True, True]  # OR over axis 1
