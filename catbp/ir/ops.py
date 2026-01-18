"""
catbp/ir/ops.py

Instruction set for the belief propagation virtual machine.

Operations:
- UNIT: Create unit tensor (all ones)
- LOAD_FACTOR: Load a factor tensor
- LOAD_SELECTOR: Load a mode selector tensor
- LOAD_SLOT: Load from message/belief slot
- STORE_SLOT: Store to message/belief slot
- CONTRACT: Multiply tensors with alignment
- ELIMINATE: Marginalize (sum out) axes
- NORMALIZE: Apply semiring normalization
- FREE: Release register memory
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple

from catbp.ir.schema import TensorSpec

# Type aliases
RegID = int
SlotID = int


class OpCode(Enum):
    """Virtual machine operation codes."""
    UNIT = 1           # Create unit tensor
    LOAD_FACTOR = 2    # Load factor from store
    LOAD_SELECTOR = 3  # Load mode selector
    LOAD_SLOT = 4      # Load from slot (message/belief)
    STORE_SLOT = 5     # Store to slot
    CONTRACT = 6       # Multiply tensors
    ELIMINATE = 7      # Marginalize axes
    NORMALIZE = 8      # Normalize tensor
    FREE = 9           # Free registers


@dataclass(frozen=True)
class Instruction:
    """
    A single VM instruction.
    
    Attributes:
        op: Operation code
        dst: Destination register (None for STORE_SLOT, FREE)
        args: Operation-specific arguments
        spec: Optional output tensor specification
    """
    op: OpCode
    dst: Optional[RegID]
    args: Dict[str, Any]
    spec: Optional[TensorSpec] = None

    def __repr__(self) -> str:
        return f"Instruction({self.op.name}, dst={self.dst}, args={self.args})"


@dataclass(frozen=True)
class Program:
    """
    A sequence of VM instructions.
    
    Attributes:
        instructions: Ordered sequence of instructions to execute
    """
    instructions: Sequence[Instruction]

    def __len__(self) -> int:
        return len(self.instructions)

    def __iter__(self):
        return iter(self.instructions)

    def __getitem__(self, idx: int) -> Instruction:
        return self.instructions[idx]
