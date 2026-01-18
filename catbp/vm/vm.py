"""
catbp/vm/vm.py

Virtual machine for executing belief propagation programs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple

import numpy as np

from catbp.ir.ops import Instruction, OpCode, Program, RegID
from catbp.ir.schema import TensorSpec
from catbp.vm.kernels import vm_contract, vm_eliminate_keep, vm_unit
from catbp.vm.memory import RegisterBank
from catbp.vm.semiring import VMSemiringRuntime


SlotKey = Tuple[Any, ...]  # e.g. ("msg", u, v), ("bel", node), ("Z",)


@dataclass
class SlotStore:
    """Storage for messages and beliefs."""
    data: Dict[SlotKey, Tuple[np.ndarray, TensorSpec]] = field(default_factory=dict)
    
    def set(self, key: SlotKey, arr: np.ndarray, spec: TensorSpec) -> None:
        """Store a tensor in a slot."""
        self.data[key] = (arr, spec)
    
    def get(self, key: SlotKey) -> Tuple[np.ndarray, TensorSpec]:
        """Get a tensor from a slot."""
        if key not in self.data:
            raise KeyError(f"Slot not found: {key}")
        return self.data[key]
    
    def has(self, key: SlotKey) -> bool:
        """Check if a slot exists."""
        return key in self.data
    
    def __contains__(self, key: SlotKey) -> bool:
        return self.has(key)


class VirtualMachine:
    """
    Virtual machine for executing belief propagation programs.
    
    The VM maintains:
    - Register bank for intermediate tensors
    - Slot store for messages and beliefs
    - Semiring runtime for operations
    """
    
    def __init__(self, sr: VMSemiringRuntime):
        self.sr = sr
        self.regs = RegisterBank()
        self.slots = SlotStore()
    
    def run(
        self,
        program: Program,
        factor_provider: Callable[[int], Tuple[np.ndarray, TensorSpec]],
        selector_provider: Callable[[int], Tuple[np.ndarray, TensorSpec]]
    ) -> None:
        """
        Execute a program.
        
        Args:
            program: Program to execute
            factor_provider: Function to load factor tensors by ID
            selector_provider: Function to load selector tensors by ID
        """
        for ins in program.instructions:
            self._execute(ins, factor_provider, selector_provider)
    
    def _execute(
        self,
        ins: Instruction,
        factor_provider: Callable[[int], Tuple[np.ndarray, TensorSpec]],
        selector_provider: Callable[[int], Tuple[np.ndarray, TensorSpec]]
    ) -> None:
        """Execute a single instruction."""
        
        if ins.op == OpCode.UNIT:
            assert ins.dst is not None and ins.spec is not None
            arr = vm_unit(ins.spec, self.sr)
            self.regs.put(ins.dst, arr, ins.spec)
            
        elif ins.op == OpCode.LOAD_FACTOR:
            assert ins.dst is not None
            factor_id = int(ins.args['factor_id'])
            arr, spec = factor_provider(factor_id)
            self.regs.put(ins.dst, arr.astype(self.sr.dtype, copy=False), spec)
            
        elif ins.op == OpCode.LOAD_SELECTOR:
            assert ins.dst is not None
            selector_id = int(ins.args.get('selector_id', ins.args.get('mode_axis_id', 0)))
            arr, spec = selector_provider(selector_id)
            self.regs.put(ins.dst, arr.astype(self.sr.dtype, copy=False), spec)
            
        elif ins.op == OpCode.LOAD_SLOT:
            assert ins.dst is not None
            slot_key = ins.args['slot'] if 'slot' in ins.args else ins.args['slot_key']
            arr, spec = self.slots.get(slot_key)
            self.regs.put(ins.dst, arr, spec)
            
        elif ins.op == OpCode.STORE_SLOT:
            slot_key = ins.args['slot'] if 'slot' in ins.args else ins.args['slot_key']
            src = int(ins.args['src'])
            arr, spec = self.regs.get(src)
            self.slots.set(slot_key, arr, spec)
            
        elif ins.op == OpCode.CONTRACT:
            assert ins.dst is not None
            input_ids = ins.args['inputs']
            out_spec = ins.args.get('out_spec', ins.spec)
            
            if out_spec is None:
                raise ValueError("CONTRACT requires out_spec")
            
            inputs = []
            for rid in input_ids:
                arr, spec = self.regs.get(rid)
                inputs.append((arr, spec))
            
            result = vm_contract(inputs, out_spec, self.sr)
            self.regs.put(ins.dst, result, out_spec)
            
        elif ins.op == OpCode.ELIMINATE:
            assert ins.dst is not None
            src = int(ins.args['src'])
            arr, in_spec = self.regs.get(src)
            
            # Support both keep_keys and drop interfaces
            if 'keep_keys' in ins.args:
                keep_keys = tuple(ins.args['keep_keys'])
            elif 'keep_axis_keys' in ins.args:
                keep_keys = tuple(ins.args['keep_axis_keys'])
            else:
                # Legacy: drop specific axes
                drop = tuple(ins.args.get('drop', ()))
                drop_set = set(drop)
                keep_keys = tuple(ax.key for ax in in_spec.axes if ax.id not in drop_set)
            
            result, out_spec = vm_eliminate_keep(arr, in_spec, keep_keys, self.sr)
            self.regs.put(ins.dst, result, out_spec)
            
        elif ins.op == OpCode.NORMALIZE:
            src = int(ins.args['src'])
            if self.sr.supports_normalize():
                arr, spec = self.regs.get(src)
                arr = self.sr.normalize(arr, None)
                self.regs.put(src, arr, spec)
                
        elif ins.op == OpCode.FREE:
            srcs = ins.args.get('srcs', ins.args.get('src', ()))
            if isinstance(srcs, int):
                srcs = (srcs,)
            for rid in srcs:
                self.regs.free(rid)
                
        else:
            raise ValueError(f"Unknown opcode: {ins.op}")
