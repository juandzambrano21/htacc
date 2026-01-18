"""
catbp/compiler/modes.py

Mode specification and selector payload construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from catbp.vm.kernels_selector import SelectorPayload


@dataclass(frozen=True)
class ModeSpec:
    """
    Specification of a mode variable from holonomy analysis.
    
    Attributes:
        mode_axis_id: Unique ID for the mode axis
        chord_endpoints: (fac_u, fac_v) factor IDs
        chord_interface: J_e variable IDs
        num_modes: Number of modes
        labels: State -> mode label mapping
    """
    mode_axis_id: int
    chord_endpoints: Tuple[int, int]
    chord_interface: Tuple[int, ...]
    num_modes: int
    labels: np.ndarray

    def to_selector_payload(self, diag_mask: np.ndarray = None) -> SelectorPayload:
        """
        Convert to SelectorPayload for VM.
        
        Args:
            diag_mask: Optional fixed-point feasibility mask
            
        Returns:
            SelectorPayload for vm_load_selector
        """
        return SelectorPayload(
            mode_axis_id=self.mode_axis_id,
            interface_axis_ids=self.chord_interface,
            quotient_map=self.labels,
            num_modes=self.num_modes,
            diag_mask=diag_mask,
        )
