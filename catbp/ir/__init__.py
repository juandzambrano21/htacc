"""
IR module: Intermediate representation schema and operations.
"""

from catbp.ir.schema import (
    AxisType,
    AxisKey,
    Axis,
    TensorSpec,
    geometric_spec,
    sort_axes,
)
from catbp.ir.ops import (
    OpCode,
    Instruction,
    Program,
    RegID,
    SlotID,
)

__all__ = [
    "AxisType",
    "AxisKey",
    "Axis",
    "TensorSpec",
    "geometric_spec",
    "sort_axes",
    "OpCode",
    "Instruction",
    "Program",
    "RegID",
    "SlotID",
]
