"""
Compiler module: Topology compilation to execution plans.
"""

from catbp.compiler.backbone import (
    choose_backbone_tree,
    choose_backbone_tree_from_nerve,
    root_tree,
    compute_euler_tour,
    in_subtree,
    path_in_tree,
)
from catbp.compiler.cycles import (
    TransportStep,
    HolonomyCycle,
    build_fundamental_cycle,
    select_chords_via_h1,
    build_h1_cycles,
    chord_cycle_steps,
)
from catbp.compiler.transport import (
    tensor_support,
    build_transport_kernel_boolean,
    build_transport_kernel_support,
)
from catbp.compiler.holonomy import (
    HolonomyModeSpec,
    compute_holonomy_boolean,
    quotient_scc_from_holonomy,
    analyze_cycle_support,
)
from catbp.compiler.plan import NodePlan, BPPlan, build_bp_plan
from catbp.compiler.modes import ModeSpec
from catbp.compiler.topology_compiler import TopologyCompiler
from catbp.compiler.emit import (
    emit_upward_program,
    emit_downward_program_prefixsuffix,
    emit_finalize_program,
    run_upward_sweep,
)
from catbp.compiler.factors import FactorDef, make_factor_provider
from catbp.compiler.selectors import SelectorDef, make_selector_provider

__all__ = [
    # backbone
    "choose_backbone_tree",
    "choose_backbone_tree_from_nerve",
    "root_tree",
    "compute_euler_tour",
    "in_subtree",
    "path_in_tree",
    # cycles
    "TransportStep",
    "HolonomyCycle",
    "build_fundamental_cycle",
    "select_chords_via_h1",
    "build_h1_cycles",
    "chord_cycle_steps",
    # transport
    "tensor_support",
    "build_transport_kernel_boolean",
    "build_transport_kernel_support",
    # holonomy
    "HolonomyModeSpec",
    "compute_holonomy_boolean",
    "quotient_scc_from_holonomy",
    "analyze_cycle_support",
    # plan
    "NodePlan",
    "BPPlan",
    "build_bp_plan",
    # modes
    "ModeSpec",
    # topology_compiler
    "TopologyCompiler",
    # emit
    "emit_upward_program",
    "emit_downward_program_prefixsuffix",
    "emit_finalize_program",
    "run_upward_sweep",
    # factors
    "FactorDef",
    "make_factor_provider",
    # selectors
    "SelectorDef",
    "make_selector_provider",
]
