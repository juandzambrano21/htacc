"""
catbp/compiler/plan.py

Execution plan for belief propagation.

The plan specifies:
- Tree structure (parent/children)
- Variable assignments per node
- Mode anchoring
- Message interfaces
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from catbp.ir.schema import Axis, AxisKey, AxisType, TensorSpec

NodeID = int
VarID = int
ModeID = int


@dataclass(frozen=True)
class NodePlan:
    """
    Plan for a single node in the backbone tree.
    
    Attributes:
        id: Node identifier
        parent: Parent node (None for root)
        children: Child nodes
        vars: Geometric variables assigned to this node
        anchored_modes: Mode axes anchored at this node
    """
    id: NodeID
    parent: NodeID | None
    children: Tuple[NodeID, ...]
    vars: Tuple[VarID, ...]
    anchored_modes: Tuple[ModeID, ...]


@dataclass(frozen=True)
class BPPlan:
    """
    Complete belief propagation execution plan.
    
    Attributes:
        root: Root node ID
        nodes: Map from node ID to NodePlan
        postorder: Postorder traversal (leaves to root)
        preorder: Preorder traversal (root to leaves)
        belief_spec: Tensor specification for belief at each node
        keep_axis_keys_on_edge: Axes to keep for messages on each directed edge
    """
    root: NodeID
    nodes: Dict[NodeID, NodePlan]
    postorder: Tuple[NodeID, ...]
    preorder: Tuple[NodeID, ...]
    belief_spec: Dict[NodeID, TensorSpec]
    keep_axis_keys_on_edge: Dict[Tuple[NodeID, NodeID], Tuple[AxisKey, ...]]


def _tree_orders(
    root: NodeID,
    children: Dict[NodeID, Tuple[NodeID, ...]]
) -> Tuple[Tuple[NodeID, ...], Tuple[NodeID, ...]]:
    """Compute postorder and preorder traversals."""
    pre: List[NodeID] = []
    post: List[NodeID] = []
    
    def dfs(u: NodeID) -> None:
        pre.append(u)
        for v in children.get(u, ()):
            dfs(v)
        post.append(u)
    
    dfs(root)
    return tuple(post), tuple(pre)


def build_bp_plan(
    *,
    root: NodeID,
    parent: Dict[NodeID, NodeID | None],
    children: Dict[NodeID, Tuple[NodeID, ...]],
    cluster_vars: Dict[NodeID, Tuple[VarID, ...]],
    anchored_modes: Dict[NodeID, Tuple[ModeID, ...]],
    domains: Dict[VarID, int],
    mode_sizes: Dict[ModeID, int],
) -> BPPlan:
    """
    Build a belief propagation execution plan.
    
    Args:
        root: Root node ID
        parent: Map from node to parent (None for root)
        children: Map from node to children
        cluster_vars: Variables assigned to each node
        anchored_modes: Modes anchored at each node
        domains: Variable domain sizes
        mode_sizes: Mode axis sizes
        
    Returns:
        BPPlan with complete execution specification
    """
    # Build node plans
    nodes: Dict[NodeID, NodePlan] = {}
    for nid, vs in cluster_vars.items():
        nodes[nid] = NodePlan(
            id=nid,
            parent=parent.get(nid),
            children=children.get(nid, ()),
            vars=tuple(vs),
            anchored_modes=tuple(anchored_modes.get(nid, ())),
        )
    
    postorder, preorder = _tree_orders(root, children)
    
    # Belief specs: (GEOMETRIC vars) + (TOPOLOGICAL modes)
    belief_spec: Dict[NodeID, TensorSpec] = {}
    for nid, n in nodes.items():
        axes: List[Axis] = []
        axes.extend(Axis(id=v, kind=AxisType.GEOMETRIC, size=domains[v]) for v in n.vars)
        axes.extend(Axis(id=m, kind=AxisType.TOPOLOGICAL, size=mode_sizes[m]) for m in n.anchored_modes)
        belief_spec[nid] = TensorSpec(axes=tuple(axes))
    
    # Separators: geometric intersections + straddling modes
    keep_axis_keys_on_edge: Dict[Tuple[NodeID, NodeID], Tuple[AxisKey, ...]] = {}
    for child, p in parent.items():
        if p is None:
            continue
        # Geometric separator
        sv = tuple(v for v in cluster_vars[child] if v in set(cluster_vars[p]))
        keep = tuple((AxisType.GEOMETRIC, v) for v in sv)
        keep_axis_keys_on_edge[(child, p)] = keep
        keep_axis_keys_on_edge[(p, child)] = keep
    
    return BPPlan(
        root=root,
        nodes=nodes,
        postorder=postorder,
        preorder=preorder,
        belief_spec=belief_spec,
        keep_axis_keys_on_edge=keep_axis_keys_on_edge,
    )
