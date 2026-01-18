"""
VM module: Virtual machine for executing compiled belief propagation plans.
"""

from catbp.vm.memory import RegisterBank
from catbp.vm.align import align_array_to
from catbp.vm.semiring import (
    VMSemiringRuntime,
    vm_sat_semiring,
    vm_prob_semiring,
    vm_logprob_semiring,
    vm_counting_semiring,
)
from catbp.vm.kernels import vm_unit, vm_contract, vm_eliminate_keep
from catbp.vm.kernels_selector import SelectorPayload, vm_load_selector
from catbp.vm.vm import VirtualMachine, SlotStore

__all__ = [
    "RegisterBank",
    "align_array_to",
    "VMSemiringRuntime",
    "vm_sat_semiring",
    "vm_prob_semiring",
    "vm_logprob_semiring",
    "vm_counting_semiring",
    "vm_unit",
    "vm_contract",
    "vm_eliminate_keep",
    "SelectorPayload",
    "vm_load_selector",
    "VirtualMachine",
    "SlotStore",
]
