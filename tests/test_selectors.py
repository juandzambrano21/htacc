"""
Tests for selector tensor construction.
"""

import numpy as np

from catbp.vm.kernels_selector import SelectorPayload, vm_load_selector


def test_logprob_selector_values():
    payload = SelectorPayload(
        mode_axis_id=3,
        interface_axis_ids=(0,),
        quotient_map=np.array([0, 1], dtype=np.int64),
        num_modes=2,
        diag_mask=None,
    )
    domains = {0: 2}
    mode_sizes = {3: 2}

    arr, _ = vm_load_selector(
        payload,
        domains=domains,
        mode_sizes=mode_sizes,
        dtype=np.float64,
        one_val=0.0,
        zero_val=-np.inf,
    )

    assert arr.shape == (2, 2)
    assert arr[0, 0] == 0.0
    assert arr[1, 1] == 0.0
    assert np.isneginf(arr[0, 1])
    assert np.isneginf(arr[1, 0])
