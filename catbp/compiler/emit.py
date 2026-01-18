"""
catbp/compiler/emit.py

Program emission for belief propagation.

Generates VM programs for:
- Upward pass (leaves to root)
- Downward pass (root to leaves)
- Finalization (compute Z)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from catbp.ir.ops import Instruction, OpCode, Program, RegID
from catbp.ir.schema import AxisKey, TensorSpec
from catbp.algebra.section import Section


def _new_reg(counter: List[int]) -> int:
    """Allocate a new register ID."""
    counter[0] += 1
    return counter[0]


def emit_upward_program(plan) -> Program:
    """
    Emit upward pass program.
    
    Requires:
        - plan.nodes[f].children
        - plan.nodes[f].parent
        - plan.nodes[f].anchored_modes
        - plan.belief_spec[f]
        - plan.keep_axis_keys_on_edge[(child, parent)]
    
    Effects (slots):
        - ("msg", child, parent) for each directed tree edge
        - ("bel", root) at the end
    """
    instrs: List[Instruction] = []
    r = [0]
    root = plan.root
    
    for a in plan.postorder:
        node = plan.nodes[a]
        regs: List[RegID] = []
        
        # Local factor
        r_phi = _new_reg(r)
        instrs.append(Instruction(OpCode.LOAD_FACTOR, dst=r_phi, args={"factor_id": a}, spec=None))
        regs.append(r_phi)
        
        # Anchored selectors at node
        for mid in node.anchored_modes:
            r_sel = _new_reg(r)
            instrs.append(Instruction(OpCode.LOAD_SELECTOR, dst=r_sel, args={"selector_id": mid}, spec=None))
            regs.append(r_sel)
        
        # Child messages
        for c in node.children:
            r_msg = _new_reg(r)
            instrs.append(Instruction(OpCode.LOAD_SLOT, dst=r_msg, args={"slot": ("msg", c, a)}, spec=None))
            regs.append(r_msg)
        
        # Contract
        out_spec: TensorSpec = plan.belief_spec[a]
        r_acc = _new_reg(r)
        instrs.append(Instruction(OpCode.CONTRACT, dst=r_acc, args={"inputs": tuple(regs), "out_spec": out_spec}, spec=out_spec))
        
        if a == root:
            instrs.append(Instruction(OpCode.STORE_SLOT, dst=None, args={"slot": ("bel", root), "src": r_acc}, spec=None))
            instrs.append(Instruction(OpCode.FREE, dst=None, args={"srcs": tuple(regs + [r_acc])}, spec=None))
            continue
        
        # Eliminate to parent message
        p = node.parent
        keep: Tuple[AxisKey, ...] = plan.keep_axis_keys_on_edge[(a, p)]
        r_out = _new_reg(r)
        instrs.append(Instruction(OpCode.ELIMINATE, dst=r_out, args={"src": r_acc, "keep_keys": keep}, spec=None))
        instrs.append(Instruction(OpCode.STORE_SLOT, dst=None, args={"slot": ("msg", a, p), "src": r_out}, spec=None))
        instrs.append(Instruction(OpCode.FREE, dst=None, args={"srcs": tuple(regs + [r_acc, r_out])}, spec=None))
    
    return Program(instructions=tuple(instrs))


def emit_downward_program_prefixsuffix(plan) -> Program:
    """
    Emit downward pass program using prefix/suffix optimization.
    
    Requires upward slots ("msg", child, parent) already exist.
    
    Effects:
        - stores beliefs ("bel", a) for all nodes
        - stores downward messages ("msg", parent, child) for all directed edges
    """
    instrs: List[Instruction] = []
    r = [0]
    root = plan.root
    
    for a in plan.preorder:
        node = plan.nodes[a]
        out_spec: TensorSpec = plan.belief_spec[a]
        
        # Base = factor(a) * selectors(a)
        regs_base: List[RegID] = []
        
        r_phi = _new_reg(r)
        instrs.append(Instruction(OpCode.LOAD_FACTOR, dst=r_phi, args={"factor_id": a}, spec=None))
        regs_base.append(r_phi)
        
        for mid in node.anchored_modes:
            r_sel = _new_reg(r)
            instrs.append(Instruction(OpCode.LOAD_SELECTOR, dst=r_sel, args={"selector_id": mid}, spec=None))
            regs_base.append(r_sel)
        
        r_base = _new_reg(r)
        instrs.append(Instruction(OpCode.CONTRACT, dst=r_base, args={"inputs": tuple(regs_base), "out_spec": out_spec}, spec=out_spec))
        
        # Inbound: optional down from parent + ups from children
        inbound_down: RegID | None = None
        if node.parent is not None:
            inbound_down = _new_reg(r)
            instrs.append(Instruction(OpCode.LOAD_SLOT, dst=inbound_down, args={"slot": ("msg", node.parent, a)}, spec=None))
        
        child_regs: List[RegID] = []
        child_ids: List[Any] = []
        for c in node.children:
            rc = _new_reg(r)
            instrs.append(Instruction(OpCode.LOAD_SLOT, dst=rc, args={"slot": ("msg", c, a)}, spec=None))
            child_regs.append(rc)
            child_ids.append(c)
        
        # Belief = base * down * Π up(children)
        regs_total = [r_base]
        if inbound_down is not None:
            regs_total.append(inbound_down)
        regs_total.extend(child_regs)
        
        r_bel = _new_reg(r)
        instrs.append(Instruction(OpCode.CONTRACT, dst=r_bel, args={"inputs": tuple(regs_total), "out_spec": out_spec}, spec=out_spec))
        instrs.append(Instruction(OpCode.STORE_SLOT, dst=None, args={"slot": ("bel", a), "src": r_bel}, spec=None))
        
        # Prefix/suffix over child ups
        # pref[0] = base*down(if any), pref[i+1] = pref[i]*child[i]
        r_pref0_inputs = [r_base] + ([inbound_down] if inbound_down is not None else [])
        r_pref0 = _new_reg(r)
        instrs.append(Instruction(OpCode.CONTRACT, dst=r_pref0, args={"inputs": tuple(r_pref0_inputs), "out_spec": out_spec}, spec=out_spec))
        
        pref: List[RegID] = [r_pref0]
        for rc in child_regs:
            rp = _new_reg(r)
            instrs.append(Instruction(OpCode.CONTRACT, dst=rp, args={"inputs": (pref[-1], rc), "out_spec": out_spec}, spec=out_spec))
            pref.append(rp)
        
        # suf[k] = UNIT; suf[i] = child[i] * suf[i+1]
        k = len(child_regs)
        r_unit = _new_reg(r)
        instrs.append(Instruction(OpCode.UNIT, dst=r_unit, args={}, spec=out_spec))
        
        suf: List[RegID] = [0] * (k + 1)
        suf[k] = r_unit
        for i in range(k - 1, -1, -1):
            rs = _new_reg(r)
            instrs.append(Instruction(OpCode.CONTRACT, dst=rs, args={"inputs": (child_regs[i], suf[i + 1]), "out_spec": out_spec}, spec=out_spec))
            suf[i] = rs
        
        # Message to each child: excl_i = pref[i] * suf[i+1] then eliminate
        for i, c in enumerate(child_ids):
            r_excl = _new_reg(r)
            instrs.append(Instruction(OpCode.CONTRACT, dst=r_excl, args={"inputs": (pref[i], suf[i + 1]), "out_spec": out_spec}, spec=out_spec))
            
            keep = plan.keep_axis_keys_on_edge[(a, c)]
            r_msg = _new_reg(r)
            instrs.append(Instruction(OpCode.ELIMINATE, dst=r_msg, args={"src": r_excl, "keep_keys": keep}, spec=None))
            instrs.append(Instruction(OpCode.STORE_SLOT, dst=None, args={"slot": ("msg", a, c), "src": r_msg}, spec=None))
        
        # Frees
        to_free = regs_base + [r_base, r_bel, r_pref0] + pref[1:] + [r_unit] + [s for s in suf[:k] if s] + child_regs
        if inbound_down is not None:
            to_free.append(inbound_down)
        instrs.append(Instruction(OpCode.FREE, dst=None, args={"srcs": tuple(to_free)}, spec=None))
    
    return Program(instructions=tuple(instrs))


def emit_finalize_program(root: int) -> Program:
    """
    Emit finalization program to compute Z.
    
    Reads ("bel", root), eliminates all axes, stores ("Z",).
    """
    instrs: List[Instruction] = []
    r = [0]
    
    r_bel = _new_reg(r)
    instrs.append(Instruction(OpCode.LOAD_SLOT, dst=r_bel, args={"slot": ("bel", root)}, spec=None))
    
    r_z = _new_reg(r)
    instrs.append(Instruction(OpCode.ELIMINATE, dst=r_z, args={"src": r_bel, "keep_keys": tuple()}, spec=None))
    instrs.append(Instruction(OpCode.STORE_SLOT, dst=None, args={"slot": ("Z",), "src": r_z}, spec=None))
    instrs.append(Instruction(OpCode.FREE, dst=None, args={"srcs": (r_bel, r_z)}, spec=None))
    
    return Program(instructions=tuple(instrs))


def run_upward_sweep(
    plan,
    *,
    factor_provider: Callable[[str], Section],
    selector_provider: Callable[[int], Section],
    message_store: Dict[Tuple[str, str], Section],
) -> Section:
    """
    Execute exact deterministic postorder sweep using Sections.
    
    This version multiplies:
        - local factor φ_a
        - all chord selectors anchored at a
        - all child->a messages
    then eliminates to I_{a,p} ∪ ActiveModes(a->p).
    
    Args:
        plan: Execution plan
        factor_provider: Function to get factor Section by name
        selector_provider: Function to get selector Section by ID
        message_store: Dict to store messages
        
    Returns:
        Root belief Section
    """
    for a in plan.postorder:
        node = plan.nodes[a]
        
        # Gather inputs
        parts = [factor_provider(a)]
        
        # Selectors for chord modes
        for mid in node.anchored_modes:
            parts.append(selector_provider(mid))
        
        # Child messages
        for c in node.children:
            parts.append(message_store[(c, a)])
        
        # Contract
        acc = parts[0]
        for t in parts[1:]:
            acc = acc.star(t)
        
        # Eliminate to parent message (or return root belief)
        if node.parent is None:
            return acc
        
        keep_keys = plan.keep_axis_keys_on_edge[(a, node.parent)]
        keep_vars = tuple(k[1] for k in keep_keys)  # Extract var IDs from AxisKeys
        msg = acc.restrict(keep_vars)
        message_store[(a, node.parent)] = msg
    
    raise RuntimeError("unreachable: postorder did not include root")
