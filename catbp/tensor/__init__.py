"""
Tensor module: Named tensor operations with canonical ordering.
"""

from catbp.tensor.named_tensor import (
    NamedTensor,
    canonical,
    align_to,
    sum_out,
    mul_all,
    reorder_tensor,
)

__all__ = [
    "NamedTensor",
    "canonical",
    "align_to",
    "sum_out",
    "mul_all",
    "reorder_tensor",
]
