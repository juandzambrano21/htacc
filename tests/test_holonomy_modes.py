"""
Tests for holonomy-aware mode handling in the compiler.
"""

import numpy as np

from catbp.compiler.topology_compiler import TopologyCompiler
from catbp.ir.schema import AxisType


def _triangle_model():
    var_domains = {"A": 2, "B": 3, "C": 3}
    factors = {
        "f1": (("A", "B"), np.ones((2, 3))),
        "f2": (("B", "C"), np.ones((3, 3))),
        "f3": (("A", "C"), np.ones((2, 3))),
    }
    return var_domains, factors


def test_mode_id_disjointness():
    var_domains, factors = _triangle_model()

    compiler = TopologyCompiler()
    _, _, _, domains, mode_sizes = compiler.compile(
        var_domains_named=var_domains,
        factors_named=factors,
        support_predicate=lambda x: x != 0,
    )

    assert mode_sizes
    max_var_id = max(domains.keys())
    min_mode_id = min(mode_sizes.keys())
    assert min_mode_id > max_var_id


def test_active_mode_propagation_on_tree_path():
    var_domains, factors = _triangle_model()

    compiler = TopologyCompiler()
    plan, _, _, domains, mode_sizes = compiler.compile(
        var_domains_named=var_domains,
        factors_named=factors,
        support_predicate=lambda x: x != 0,
    )

    mode_id = next(iter(mode_sizes.keys()))
    fac_names = sorted(factors.keys())
    fid_f1 = fac_names.index("f1")
    fid_f2 = fac_names.index("f2")
    fid_f3 = fac_names.index("f3")

    spec = plan.belief_spec[fid_f2]
    assert any(ax.kind == AxisType.TOPOLOGICAL and ax.id == mode_id for ax in spec.axes)

    for edge in ((fid_f1, fid_f2), (fid_f2, fid_f3)):
        assert (AxisType.TOPOLOGICAL, mode_id) in plan.keep_axis_keys_on_edge[edge]
