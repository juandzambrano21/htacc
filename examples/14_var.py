"""
14 binary variables with a single holonomy cycle:
  - 4 high-arity cycle factors (each enforces multiple constraints).
  - Pairwise parity constraints on overlapping pairs.
  - Unary clues on unique variables.

This keeps holonomy nontrivial while staying under numpy's axis limit.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np

from catbp.hatcc import hatcc_solve


VARS = [f"X{i}" for i in range(14)]
PATTERN = np.array(
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
    dtype=int,
)

FACTOR_SCOPES = [
    ("X0", "X1", "X2", "X3", "X4", "X5"),
    ("X4", "X5", "X6", "X7", "X8", "X9"),
    ("X8", "X9", "X10", "X11", "X12", "X13"),
    ("X12", "X13", "X0", "X1", "X2", "X3"),
]


def _make_factor(
    scope: Tuple[str, ...],
    predicate,
    domains: Dict[str, int],
) -> np.ndarray:
    shape = tuple(domains[v] for v in scope)
    grid = np.indices(shape).reshape(len(scope), -1).T
    data = np.zeros((grid.shape[0],), dtype=bool)
    for i, vals in enumerate(grid):
        data[i] = predicate(*vals.tolist())
    return data.reshape(shape)


def _pattern_map() -> Dict[str, int]:
    return {v: int(PATTERN[i]) for i, v in enumerate(VARS)}


def _factor_predicate(scope: Tuple[str, ...], pattern_map: Dict[str, int]):
    values = [pattern_map[v] for v in scope]
    target_sum = sum(values)
    target_parity = values[0] ^ values[1] ^ values[2] ^ values[3]
    target_mod = sum((i + 1) * v for i, v in enumerate(values)) % 3

    def predicate(*vs, ts=target_sum, tp=target_parity, tm=target_mod) -> bool:
        if sum(vs) != ts:
            return False
        if (vs[0] ^ vs[1] ^ vs[2] ^ vs[3]) != tp:
            return False
        checksum = sum((i + 1) * v for i, v in enumerate(vs)) % 3
        return checksum == tm

    return predicate


def build_problem() -> Tuple[Dict[str, int], Dict[str, Tuple[Tuple[str, ...], np.ndarray]], List[str]]:
    var_domains = {v: 2 for v in VARS}
    pattern_map = _pattern_map()
    factors: Dict[str, Tuple[Tuple[str, ...], np.ndarray]] = {}

    for i, scope in enumerate(FACTOR_SCOPES):
        predicate = _factor_predicate(scope, pattern_map)
        factors[f"cycle_{i}"] = (scope, _make_factor(scope, predicate, var_domains))

    pair_scopes = [
        ("X0", "X1"),
        ("X4", "X5"),
        ("X8", "X9"),
        ("X12", "X13"),
        ("X6", "X7"),
        ("X10", "X11"),
    ]
    for a, b in pair_scopes:
        parity = pattern_map[a] ^ pattern_map[b]
        scope = (a, b)
        factors[f"pair_{a}_{b}"] = (
            scope,
            _make_factor(scope, lambda x, y, p=parity: (x ^ y) == p, var_domains),
        )

    clues = ["X6", "X7", "X10", "X11"]
    for v in clues:
        arr = np.zeros((2,), dtype=bool)
        arr[pattern_map[v]] = True
        factors[f"clue_{v}"] = ((v,), arr)

    var_order = list(VARS)
    return var_domains, factors, var_order


def _count_solutions(
    var_order: List[str],
    factors: Dict[str, Tuple[Tuple[str, ...], np.ndarray]],
) -> int:
    factor_list = [factors[k] for k in sorted(factors.keys())]
    count = 0
    for bits in itertools.product([0, 1], repeat=len(var_order)):
        assignment = dict(zip(var_order, bits))
        ok = True
        for scope, data in factor_list:
            idx = tuple(assignment[v] for v in scope)
            if not data[idx]:
                ok = False
                break
        if ok:
            count += 1
    return count


def main() -> bool:
    var_domains, factors, var_order = build_problem()

    print("- 14 binary variables, 4-cycle with high-arity constraints")
    print(f"- Factors: {len(factors)}")

    result = hatcc_solve(var_domains, factors, semiring="prob")
    print(result)
    sat = bool(result.Z)
    print(f"Satisfiable (HATCC/SAT): {sat}")

    brute = _count_solutions(var_order, factors)
    print(f"Satisfiable assignments (brute force): {brute}")

    return sat == (brute > 0)


if __name__ == "__main__":
    main()
