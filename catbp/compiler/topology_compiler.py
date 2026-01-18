"""
catbp/compiler/topology_compiler.py

Main topology compiler: nerve → backbone → chords → holonomy → modes → execution plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import math
import numpy as np
import networkx as nx
import scipy.sparse as sp

from catbp.ir.schema import Axis, AxisType, TensorSpec, AxisKey
from catbp.compiler.backbone import choose_backbone_tree, root_tree, compute_euler_tour, in_subtree
from catbp.compiler.cycles import chord_cycle_steps
from catbp.compiler.holonomy import compute_holonomy_boolean, quotient_scc_from_holonomy
from catbp.compiler.modes import ModeSpec
from catbp.compiler.plan import NodePlan, BPPlan
from catbp.vm.kernels_selector import SelectorPayload
from catbp.topology.nerve import build_nerve_graph


def canon_tuple(xs: List[int]) -> Tuple[int, ...]:
    """Return canonical (sorted) tuple."""
    return tuple(sorted(xs))


def edge_key(a: int, b: int) -> Tuple[int, int]:
    """Return canonical edge key."""
    return (a, b) if a < b else (b, a)


def log_interface_size(interface_vars: Tuple[int, ...], domains: Dict[int, int]) -> float:
    """Compute log of interface size."""
    if not interface_vars:
        return 0.0
    s = 0.0
    for v in interface_vars:
        s += float(np.log(domains.get(v, 1)))
    return s


@dataclass(frozen=True)
class BoolSection:
    """
    Domain-ordered boolean tensor representing factor support.
    """
    domain: Tuple[int, ...]
    data: np.ndarray

    def restrict_exist(self, target_domain: Tuple[int, ...], domains: Dict[int, int]) -> "BoolSection":
        """
        Existentially quantify out variables not in target_domain (Boolean OR-reduce).
        """
        dom = self.domain
        pos = {v: i for i, v in enumerate(dom)}

        for v in target_domain:
            if v not in pos:
                raise ValueError("restrict_exist: target var not in domain")

        # Transpose to [target..., eliminated...]
        target_axes = [pos[v] for v in target_domain]
        elim_axes = [i for i, v in enumerate(dom) if v not in set(target_domain)]
        perm = target_axes + elim_axes
        x = np.transpose(self.data, axes=perm)

        # OR-reduce over eliminated axes
        if elim_axes:
            red_axes = tuple(range(len(target_domain), x.ndim))
            x = np.logical_or.reduce(x, axis=red_axes)
        return BoolSection(domain=target_domain, data=x)


class TopologyEngine:
    """Engine for computing transport kernels and holonomy."""
    
    def __init__(self, domains: Dict[int, int], factor_supports: Dict[int, BoolSection]):
        self.domains = domains
        self.supports = factor_supports

    def _flat_size(self, vars_: Tuple[int, ...]) -> int:
        s = 1
        for v in vars_:
            s *= self.domains[v]
        return int(s)

    def build_transport_matrix(self, factor_id: int, U: Tuple[int, ...], V: Tuple[int, ...]) -> sp.csr_matrix:
        """
        Build boolean transport kernel K^{U->V} induced by factor support.
        """
        sec = self.supports[factor_id]
        union = canon_tuple(list(set(U) | set(V)))
        proj = sec.restrict_exist(union, self.domains)

        rows = self._flat_size(U)
        cols = self._flat_size(V)

        if proj.data.size == 0:
            return sp.csr_matrix((rows, cols), dtype=np.int8)

        idx = np.argwhere(proj.data)
        if idx.shape[0] == 0:
            return sp.csr_matrix((rows, cols), dtype=np.int8)

        union_vars = proj.domain
        u_map = [union_vars.index(v) for v in U]
        v_map = [union_vars.index(v) for v in V]

        u_shape = tuple(self.domains[v] for v in U) if U else (1,)
        v_shape = tuple(self.domains[v] for v in V) if V else (1,)

        u_states = idx[:, u_map] if U else np.zeros((idx.shape[0], 1), dtype=np.int64)
        v_states = idx[:, v_map] if V else np.zeros((idx.shape[0], 1), dtype=np.int64)

        r = np.ravel_multi_index(u_states.T, u_shape)
        c = np.ravel_multi_index(v_states.T, v_shape)

        data = np.ones(len(r), dtype=np.int8)
        M = sp.coo_matrix((data, (r, c)), shape=(rows, cols)).tocsr()
        M.data = (M.data > 0).astype(np.int8)
        M.eliminate_zeros()
        return M

    def holonomy_and_quotient(
        self,
        cycle_factors: List[int],
        chord_interface: Tuple[int, ...],
        scopes: Dict[int, Tuple[int, ...]]
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Compute holonomy and mode quotient for a cycle.
        
        Returns:
            (labels, num_modes, diag_mask)
        """
        dimJ = self._flat_size(chord_interface)
        H = sp.identity(dimJ, format="csr", dtype=np.int8)

        curr = chord_interface
        for i in range(len(cycle_factors)):
            f_curr = cycle_factors[i]
            if i == len(cycle_factors) - 1:
                nxt = chord_interface
            else:
                f_next = cycle_factors[i + 1]
                nxt = canon_tuple(list(set(scopes[f_curr]).intersection(scopes[f_next])))

            K = self.build_transport_matrix(f_curr, curr, nxt)
            H = H.dot(K)
            H.data = (H.data > 0).astype(np.int8)
            H.eliminate_zeros()
            curr = nxt

        # SCC quotient
        from scipy.sparse.csgraph import connected_components
        ncomp, labels = connected_components(H, directed=True, connection="strong")

        # Diagonal admissibility (Patch E)
        diag = H.diagonal()
        diag_mask = (diag > 0)

        return labels.astype(np.int64), int(ncomp), diag_mask


class TopologyCompiler:
    """
    Main compiler: factor graph → execution plan.
    """
    
    def compile(
        self,
        var_domains_named: Dict[str, int],
        factors_named: Dict[str, Tuple[Tuple[str, ...], np.ndarray]],
        support_predicate,
        root_factor_name: Optional[str] = None
    ):
        """
        Compile a factor graph to an execution plan.
        
        Args:
            var_domains_named: Map from variable name to domain size
            factors_named: Map from factor name to (scope_names, data)
            support_predicate: Function to compute support from data
            root_factor_name: Optional root factor name
            
        Returns:
            (plan, factor_store, selector_store, domains, mode_sizes)
        """
        # 1) Assign IDs
        var_names = sorted(var_domains_named.keys())
        var_id = {n: i for i, n in enumerate(var_names)}
        domains = {var_id[n]: int(var_domains_named[n]) for n in var_names}

        fac_names = sorted(factors_named.keys())
        fac_id = {n: i for i, n in enumerate(fac_names)}
        id_fac = {i: n for n, i in fac_id.items()}

        # 2) Canonical scopes and factor_store
        scopes: Dict[int, Tuple[int, ...]] = {}
        factor_store: Dict[int, Tuple[np.ndarray, TensorSpec]] = {}
        factor_supports: Dict[int, BoolSection] = {}

        for fname, (scope_names, data) in factors_named.items():
            fid = fac_id[fname]
            scope = tuple(sorted(var_id[v] for v in scope_names))
            scopes[fid] = scope

            # Factor tensor spec
            axes = tuple(Axis(id=v, kind=AxisType.GEOMETRIC, size=domains[v]) for v in scope)
            spec = TensorSpec(axes=axes)
            factor_store[fid] = (np.asarray(data), spec)

            # Support section
            supp = support_predicate(np.asarray(data))
            factor_supports[fid] = BoolSection(domain=scope, data=np.asarray(supp, dtype=bool))

        # 3) Build nerve graph
        G = nx.Graph()
        for fid in scopes:
            G.add_node(fid)

        for a in scopes:
            Sa = set(scopes[a])
            for b in scopes:
                if b <= a:
                    continue
                Sb = set(scopes[b])
                inter = tuple(sorted(Sa.intersection(Sb)))
                if inter:
                    w = log_interface_size(inter, domains)
                    G.add_edge(a, b, interface=inter, weight=w)

        if root_factor_name is None:
            root = fac_id[fac_names[0]]
        else:
            root = fac_id[root_factor_name]

        # 4) Backbone: maximum spanning tree
        T = nx.maximum_spanning_tree(G, weight="weight")
        tree_edges = set(edge_key(u, v) for u, v in T.edges())
        all_edges = set(edge_key(u, v) for u, v in G.edges())
        chords = sorted(list(all_edges - tree_edges))

        # 5) Root tree
        parent: Dict[int, Optional[int]] = {root: None}
        children: Dict[int, List[int]] = {fid: [] for fid in scopes}
        order = [root]
        for u in order:
            for v in T.neighbors(u):
                if v in parent:
                    continue
                parent[v] = u
                children[u].append(v)
                order.append(v)

        preorder = tuple(order)
        postorder = tuple(reversed(order))

        # 6) Compute subtree sets
        subtree: Dict[int, Set[int]] = {fid: set() for fid in scopes}
        for a in postorder:
            s = {a}
            for c in children[a]:
                s |= subtree[c]
            subtree[a] = s

        # 7) Holonomy → mode selectors
        topo = TopologyEngine(domains, factor_supports)

        selector_store: Dict[int, SelectorPayload] = {}
        mode_sizes: Dict[int, int] = {}
        anchored_modes: Dict[int, List[int]] = {fid: [] for fid in scopes}
        next_mode_id = max(domains.keys(), default=-1) + 1

        chord_meta = []

        for chord_idx, (u, v) in enumerate(chords):
            J = G.get_edge_data(u, v)["interface"]
            path = nx.shortest_path(T, u, v)
            cycle_factors = list(path)

            labels, ncomp, diag_mask = topo.holonomy_and_quotient(cycle_factors, J, scopes)

            # Always create mode axis, even for trivial holonomy (ncomp=1)
            # The mode axis ensures proper handling of cycles
            mode_axis_id = next_mode_id
            next_mode_id += 1
            
            payload = SelectorPayload(
                mode_axis_id=mode_axis_id,
                interface_axis_ids=J,
                quotient_map=labels,
                num_modes=ncomp,
                diag_mask=diag_mask.astype(bool),
            )

            selector_store[mode_axis_id] = payload
            mode_sizes[mode_axis_id] = ncomp

            # Anchor the selector at BOTH endpoints of the chord
            # This ensures consistency of the mode variable across the cycle
            anchored_modes[u].append(mode_axis_id)
            anchored_modes[v].append(mode_axis_id)

            chord_meta.append((mode_axis_id, u, v, J, ncomp))

        # 8) Modes and (for trivial holonomy) interface vars that straddle each tree edge
        edge_modes: Dict[Tuple[int, int], Tuple[int, ...]] = {}
        edge_straddle_vars: Dict[Tuple[int, int], Tuple[int, ...]] = {}
        for (a, b) in tree_edges:
            if parent[a] == b:
                child = a
            elif parent[b] == a:
                child = b
            else:
                child = a
            S = subtree[child]
            ms = []
            straddle_vars = set()
            for mid, mu, mv, J, ncomp in chord_meta:
                if (mu in S) ^ (mv in S):
                    ms.append(mid)
                    if ncomp == 1:
                        straddle_vars.update(J)
            edge_modes[(a, b)] = tuple(sorted(ms))
            edge_straddle_vars[(a, b)] = tuple(sorted(straddle_vars))

        # 9) Keep keys for directed messages
        # Include tree separator, trivial-holonomy interface vars, and straddling mode axes
        keep_axis_keys_on_edge: Dict[Tuple[int, int], Tuple[AxisKey, ...]] = {}
        for (a, b) in tree_edges:
            I = canon_tuple(list(set(scopes[a]).intersection(scopes[b])))
            keep_set = {(AxisType.GEOMETRIC, v) for v in I}
            keep_set.update((AxisType.GEOMETRIC, v) for v in edge_straddle_vars[(a, b)])
            keep_set.update((AxisType.TOPOLOGICAL, mid) for mid in edge_modes[(a, b)])
            keep = tuple(sorted(keep_set, key=lambda x: (x[0].value, x[1])))
            keep_axis_keys_on_edge[(a, b)] = keep
            keep_axis_keys_on_edge[(b, a)] = keep

        # 10) Belief specs
        # Include factor scope variables, trivial-holonomy interface vars, and incident mode axes
        belief_spec: Dict[int, TensorSpec] = {}
        node_modes: Dict[int, Set[int]] = {fid: set(anchored_modes[fid]) for fid in scopes}
        for (a, b), ms in edge_modes.items():
            node_modes[a].update(ms)
            node_modes[b].update(ms)
        for a in scopes:
            var_set = set(scopes[a])
            for b in T.neighbors(a):
                var_set.update(edge_straddle_vars[edge_key(a, b)])
            axes = [Axis(id=v, kind=AxisType.GEOMETRIC, size=domains[v]) for v in sorted(var_set)]
            axes.extend(
                Axis(id=m, kind=AxisType.TOPOLOGICAL, size=mode_sizes[m])
                for m in sorted(node_modes[a])
            )
            belief_spec[a] = TensorSpec(axes=tuple(axes))

        # 11) Build nodes
        nodes: Dict[int, NodePlan] = {}
        for a in scopes:
            nodes[a] = NodePlan(
                id=a,
                parent=parent[a],
                children=tuple(children[a]),
                vars=scopes[a],
                anchored_modes=tuple(sorted(anchored_modes[a])),
            )

        plan = BPPlan(
            root=root,
            nodes=nodes,
            postorder=postorder,
            preorder=preorder,
            belief_spec=belief_spec,
            keep_axis_keys_on_edge=keep_axis_keys_on_edge,
        )

        return plan, factor_store, selector_store, domains, mode_sizes
