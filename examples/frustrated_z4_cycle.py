"""
Example: Frustrated Z4 3-cycle comparing loopy BP and holonomy-compiled propagation.

Model:
  x0, x1, x2 in Z4 with pairwise factors on (0,1), (1,2), (2,0).
  Two edges prefer equality, one edge prefers a +2 shift (mod 4).
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np

from catbp.hatcc import hatcc_solve
from catbp.api.marginals import marginal_variable
from catbp.vm.semiring import vm_prob_semiring


def _psi_eq(beta: float) -> np.ndarray:
    hi = np.exp(beta)
    lo = np.exp(-beta)
    data = np.full((4, 4), lo, dtype=np.float64)
    for a in range(4):
        data[a, a] = hi
    return data


def _psi_shift2(beta: float) -> np.ndarray:
    hi = np.exp(beta)
    lo = np.exp(-beta)
    data = np.full((4, 4), lo, dtype=np.float64)
    for a in range(4):
        b = (a + 2) % 4
        data[a, b] = hi
    return data


def _build_model(beta: float):
    var_domains = {"x0": 4, "x1": 4, "x2": 4}
    psi_eq = _psi_eq(beta)
    psi_sh2 = _psi_shift2(beta)

    factors = {
        "psi_01": (("x0", "x1"), psi_eq.copy()),
        "psi_12": (("x1", "x2"), psi_eq.copy()),
        "psi_20": (("x2", "x0"), psi_sh2.copy()),
    }
    return var_domains, factors, psi_eq, psi_sh2


def _bruteforce_marginals(
    var_order: List[str],
    var_domains: Dict[str, int],
    factors: Dict[str, Tuple[Tuple[str, ...], np.ndarray]],
) -> Tuple[float, Dict[str, np.ndarray]]:
    factor_list = [factors[k] for k in sorted(factors.keys())]
    Z = 0.0
    marg = {v: np.zeros((var_domains[v],), dtype=np.float64) for v in var_order}

    for assignment in itertools.product(*[range(var_domains[v]) for v in var_order]):
        assign = dict(zip(var_order, assignment))
        w = 1.0
        for scope, data in factor_list:
            idx = tuple(assign[v] for v in scope)
            w *= data[idx]
        Z += w
        for v in var_order:
            marg[v][assign[v]] += w

    for v in var_order:
        marg[v] /= Z
    return Z, marg


def _loopy_bp(
    psi: Dict[Tuple[int, int], np.ndarray],
    neighbors: Dict[int, List[int]],
    max_iters: int = 200,
    tol: float = 1e-6,
    seed: int = 7,
    noise: float = 0.02,
):
    rng = np.random.default_rng(seed)
    messages = {}
    for i in neighbors:
        for j in neighbors[i]:
            msg = np.full((4,), 1.0 / 4.0, dtype=np.float64)
            msg += noise * rng.standard_normal(4)
            msg = np.clip(msg, 1e-12, None)
            msg /= msg.sum()
            messages[(i, j)] = msg

    residuals = []
    for t in range(max_iters):
        new_messages = {}
        max_delta = 0.0
        for i in neighbors:
            for j in neighbors[i]:
                incoming = [messages[(k, i)] for k in neighbors[i] if k != j]
                prod = incoming[0].copy() if incoming else np.ones((4,), dtype=np.float64)
                if len(incoming) > 1:
                    for m in incoming[1:]:
                        prod *= m
                m_out = prod @ psi[(i, j)]
                m_out /= m_out.sum()
                new_messages[(i, j)] = m_out
                max_delta = max(max_delta, float(np.max(np.abs(m_out - messages[(i, j)]))))
        residuals.append(max_delta)
        messages = new_messages
        if max_delta < tol:
            break

    return messages, residuals


def _beliefs_from_messages(
    messages: Dict[Tuple[int, int], np.ndarray],
    neighbors: Dict[int, List[int]],
) -> Dict[int, np.ndarray]:
    beliefs = {}
    for i in neighbors:
        b = np.ones((4,), dtype=np.float64)
        for k in neighbors[i]:
            b *= messages[(k, i)]
        b /= b.sum()
        beliefs[i] = b
    return beliefs


def _holonomy_kernel(psi_01: np.ndarray, psi_12: np.ndarray, psi_20: np.ndarray) -> np.ndarray:
    K01 = psi_01.T
    K12 = psi_12.T
    K20 = psi_20.T
    return K20 @ K12 @ K01


def _quotient_classes(H: np.ndarray, threshold_ratio: float = 1e-6) -> List[List[int]]:
    rel = H / H.max()
    adj = rel >= threshold_ratio
    adj = np.logical_or(adj, adj.T)

    seen = set()
    comps = []
    for i in range(H.shape[0]):
        if i in seen:
            continue
        stack = [i]
        comp = []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            nbrs = np.where(adj[u])[0].tolist()
            for v in nbrs:
                if v not in seen:
                    stack.append(v)
        comps.append(sorted(comp))
    return comps


def _fmt_vec(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:.3f}" for x in v.tolist()) + "]"


def main() -> bool:
    beta = 10.0
    var_domains, factors, psi_eq, psi_sh2 = _build_model(beta)

    print("Frustrated Z4 Cycle Demo")
    print("- Nodes: x0, x1, x2 in Z4")
    print("- Potentials: eq on (0,1), (1,2); +2 shift on (2,0)")
    print(f"- beta = {beta}")

    var_order = ["x0", "x1", "x2"]
    Z_exact, marg_exact = _bruteforce_marginals(var_order, var_domains, factors)

    psi = {
        (0, 1): psi_eq,
        (1, 0): psi_eq.T,
        (1, 2): psi_eq,
        (2, 1): psi_eq.T,
        (2, 0): psi_sh2,
        (0, 2): psi_sh2.T,
    }
    neighbors = {0: [1, 2], 1: [0, 2], 2: [0, 1]}

    messages, residuals = _loopy_bp(psi, neighbors)
    final_res = residuals[-1] if residuals else 0.0
    print("\nLoopy BP (parallel, no damping)")
    print(f"  iterations: {len(residuals)}")
    print(f"  final residual: {final_res:.6e}")
    print(f"  converged: {final_res < 1e-6}")

    beliefs = _beliefs_from_messages(messages, neighbors)
    uniform = np.full((4,), 0.25, dtype=np.float64)
    max_dev = max(float(np.max(np.abs(beliefs[i] - uniform))) for i in beliefs)
    print("  beliefs:")
    for i in range(3):
        print(f"    x{i}: {_fmt_vec(beliefs[i])}")
    print(f"  max deviation from uniform: {max_dev:.3e}")

    H = _holonomy_kernel(psi_eq, psi_eq, psi_sh2)
    classes = _quotient_classes(H, threshold_ratio=1e-6)
    H_norm = H / H.max()
    print("\nHolonomy kernel H (normalized by max)")
    print(np.array2string(H_norm, precision=3, suppress_small=True))
    print(f"Holonomy quotient classes (threshold 1e-6): {classes}")

    result = hatcc_solve(var_domains, factors, semiring="prob")
    sr = vm_prob_semiring(np.float64)
    var_id = {name: i for i, name in enumerate(sorted(var_domains.keys()))}
    hatcc_marg = {
        v: marginal_variable(result.slots, result.plan, var_id[v], sr)
        for v in var_order
    }

    print(f"\nHATCC Z: {result.Z:.6f}")
    print("\nHATCC marginals:")
    max_err = 0.0
    for v in var_order:
        err = float(np.max(np.abs(hatcc_marg[v] - marg_exact[v])))
        max_err = max(max_err, err)
        print(f"  {v}: {_fmt_vec(hatcc_marg[v])} (max err {err:.3e})")

    print(f"HATCC max marginal error: {max_err:.3e}")
    return max_err < 1e-6


if __name__ == "__main__":
    main()
