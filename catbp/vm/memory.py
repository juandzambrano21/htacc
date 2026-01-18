"""
catbp/vm/memory.py

Register bank for VM tensor storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from catbp.ir.schema import TensorSpec

RegID = int


@dataclass(frozen=True)
class RegisterInfo:
    """Metadata about a register's contents."""
    spec: TensorSpec
    shape: Tuple[int, ...]


class RegisterBank:
    """
    Register bank for storing tensors during VM execution.
    
    Each register holds a numpy array and its specification.
    """
    
    def __init__(self):
        self.data: Dict[RegID, np.ndarray] = {}
        self.spec: Dict[RegID, TensorSpec] = {}
        self.next_id: int = 0

    def alloc(self, arr: np.ndarray, spec: TensorSpec) -> RegID:
        """
        Allocate a new register with the given array and spec.
        
        Returns:
            The allocated register ID
        """
        rid = self.next_id
        self.next_id += 1
        self.data[rid] = arr
        self.spec[rid] = spec
        return rid

    def put(self, rid: RegID, arr: np.ndarray, spec: TensorSpec) -> None:
        """Store array and spec in an existing register."""
        self.data[rid] = arr
        self.spec[rid] = spec

    def get(self, rid: RegID) -> Tuple[np.ndarray, TensorSpec]:
        """Get array and spec from a register."""
        return self.data[rid], self.spec[rid]

    def get_array(self, rid: RegID) -> np.ndarray:
        """Get just the array from a register."""
        return self.data[rid]

    def get_spec(self, rid: RegID) -> TensorSpec:
        """Get just the spec from a register."""
        return self.spec[rid]

    def free(self, rid: RegID) -> None:
        """Free a register."""
        self.data.pop(rid, None)
        self.spec.pop(rid, None)

    def free_many(self, rids: Tuple[RegID, ...]) -> None:
        """Free multiple registers."""
        for rid in rids:
            self.free(rid)

    def has(self, rid: RegID) -> bool:
        """Check if a register exists."""
        return rid in self.data

    def __contains__(self, rid: RegID) -> bool:
        return self.has(rid)

    def __len__(self) -> int:
        return len(self.data)
