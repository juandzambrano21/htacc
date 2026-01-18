"""
catbp/runtime/schedule.py

Program execution scheduling.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from catbp.ir.ops import Program
from catbp.ir.schema import TensorSpec
from catbp.vm.semiring import VMSemiringRuntime
from catbp.vm.vm import VirtualMachine


def run_program(
    program: Program,
    sr: VMSemiringRuntime,
    factor_provider: Callable[[int], Tuple[np.ndarray, TensorSpec]],
    selector_provider: Callable[[int], Tuple[np.ndarray, TensorSpec]],
    *,
    slots: Optional[Dict[object, Tuple[np.ndarray, TensorSpec]]] = None,
) -> Dict[object, Tuple[np.ndarray, TensorSpec]]:
    """
    Execute a program and return the slot store.
    
    Args:
        program: Program to execute
        sr: Semiring runtime
        factor_provider: Function to load factors
        selector_provider: Function to load selectors
        slots: Optional initial slot contents
        
    Returns:
        Slot store after execution
    """
    vm = VirtualMachine(sr)
    if slots is not None:
        vm.slots.data.update(slots)
    vm.run(program, factor_provider=factor_provider, selector_provider=selector_provider)
    return vm.slots.data


def finalize_Z_from_slots(slots: Dict[object, Tuple[np.ndarray, TensorSpec]]) -> float:
    """
    Extract partition function Z from slots.
    
    Args:
        slots: Slot store containing ("Z",) entry
        
    Returns:
        Partition function value
    """
    z_arr, _ = slots[("Z",)]
    if np.ndim(z_arr) != 0:
        z_arr = np.asarray(z_arr).reshape(())
    return float(z_arr)
