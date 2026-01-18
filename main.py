#!/usr/bin/env python3
"""
CatBP: Categorical Belief Propagation

A implementation of holonomy-aware belief propagation
using category-theoretic foundations.

Usage:
    # Solve from JSON file
    python main.py solve --input problem.json --output result.json
    
    # Solve from command line
    python main.py solve --vars "A:2,B:2,C:2" --factors "f1:A,B:[[0.9,0.1],[0.2,0.8]]"
    
    # Run demos
    python main.py demo --example chain
    
    # Run tests
    python main.py test

Authors: Juan Diego Zambrano, Enrique ter Horst, Sridhar Mahadevan
License: GPL-3.0
Reference: arXiv:2601.04456
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Handle imports whether running as package or directly
try:
    from catbp import (
        compute_partition_function,
        compute_marginals,
        hatcc_solve,
        __version__,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from catbp import (
        compute_partition_function,
        compute_marginals,
        hatcc_solve,
        __version__,
    )


def load_problem_from_json(filepath: str) -> Tuple[Dict[str, int], Dict[str, Tuple[Tuple[str, ...], np.ndarray]]]:
    """
    Load a factor graph problem from JSON file.
    
    Expected format:
    {
        "variables": {"A": 2, "B": 3},
        "factors": {
            "f1": {"scope": ["A", "B"], "values": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}
        }
    }
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    var_domains = data["variables"]
    
    factors = {}
    for name, fdata in data["factors"].items():
        scope = tuple(fdata["scope"])
        values = np.array(fdata["values"], dtype=np.float64)
        factors[name] = (scope, values)
    
    return var_domains, factors


def save_result_to_json(filepath: str, result: Any, marginals: Dict[str, np.ndarray]) -> None:
    """Save solver result to JSON file."""
    output = {
        "partition_function": float(result.Z),
        "marginals": {var: prob.tolist() for var, prob in marginals.items()},
        "status": "success" if result.Z > 0 else "unsat"
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)


def parse_vars_string(vars_str: str) -> Dict[str, int]:
    """Parse variable specification: 'A:2,B:3,C:2'"""
    var_domains = {}
    for part in vars_str.split(','):
        part = part.strip()
        if ':' in part:
            name, size = part.split(':')
            var_domains[name.strip()] = int(size.strip())
    return var_domains


def parse_factors_string(factors_str: str) -> Dict[str, Tuple[Tuple[str, ...], np.ndarray]]:
    """Parse factor specification: 'f1:A,B:[[0.9,0.1],[0.2,0.8]];f2:B,C:[[0.3,0.7],[0.5,0.5]]'"""
    factors = {}
    for factor_spec in factors_str.split(';'):
        factor_spec = factor_spec.strip()
        if not factor_spec:
            continue
        
        parts = factor_spec.split(':')
        if len(parts) >= 3:
            name = parts[0].strip()
            scope = tuple(v.strip() for v in parts[1].split(','))
            values_str = ':'.join(parts[2:])
            values = np.array(json.loads(values_str), dtype=np.float64)
            factors[name] = (scope, values)
    
    return factors



def cmd_solve(args):
    """Execute the solve command."""
    
    if args.input:
        print(f"Loading problem from: {args.input}")
        var_domains, factors = load_problem_from_json(args.input)
    elif args.vars and args.factors:
        var_domains = parse_vars_string(args.vars)
        factors = parse_factors_string(args.factors)
    else:
        print("Error: Must specify either --input FILE or both --vars and --factors")
        return 1
    
    print(f"\nProblem specification:")
    print(f"  Variables: {len(var_domains)}")
    for var, size in sorted(var_domains.items()):
        print(f"    {var}: domain size {size}")
    print(f"  Factors: {len(factors)}")
    for name, (scope, values) in sorted(factors.items()):
        print(f"    {name}: scope {scope}, shape {values.shape}")
    
    semiring = args.semiring
    print(f"\nSemiring: {semiring}")
    
    print("\nRunning HATCC solver...")
    try:
        result = hatcc_solve(var_domains, factors, semiring=semiring)
    except Exception as e:
        print(f"Error during solving: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\nResults:")
    if semiring == "logprob":
        print(f"  log(Z) = {result.Z:.10f}")
        print(f"  Z = {np.exp(result.Z):.10e}")
    else:
        print(f"  Z = {result.Z:.10f}")
    
    if result.Z == 0:
        print("  Status: UNSAT (no valid configurations)")
    else:
        print("  Status: SAT")
    
    marginals = {}
    if args.marginals and result.Z > 0:
        print("\nMarginal distributions:")
        marginals = compute_marginals(var_domains, factors, semiring=semiring)
        for var, prob in sorted(marginals.items()):
            prob_str = ', '.join(f'{p:.6f}' for p in prob)
            print(f"  P({var}) = [{prob_str}]")
    
    if args.output:
        if not marginals and result.Z > 0:
            marginals = compute_marginals(var_domains, factors, semiring=semiring)
        save_result_to_json(args.output, result, marginals)
        print(f"\nResults saved to: {args.output}")
    
    return 0



def demo_simple_chain():
    """Demo: Simple chain A -- B -- C"""
    print("=" * 60)
    print("Demo: Simple Chain A -- B -- C")
    print("=" * 60)
    
    var_domains = {"A": 2, "B": 2, "C": 2}
    
    phi_A = np.array([0.6, 0.4])
    phi_AB = np.array([[0.9, 0.1], [0.2, 0.8]])
    phi_BC = np.array([[0.3, 0.7], [0.5, 0.5]])
    
    factors = {
        "f_A": (("A",), phi_A),
        "f_AB": (("A", "B"), phi_AB),
        "f_BC": (("B", "C"), phi_BC),
    }
    
    print("\nFactor Graph: A -- B -- C")
    print("  Variables: A, B, C (each binary)")
    
    Z = compute_partition_function(var_domains, factors)
    print(f"\nPartition function Z = {Z:.6f}")
    
    marginals = compute_marginals(var_domains, factors)
    print("\nMarginal distributions:")
    for var, prob in sorted(marginals.items()):
        print(f"  P({var}) = [{prob[0]:.4f}, {prob[1]:.4f}]")
    
    Z_brute = sum(
        phi_A[a] * phi_AB[a, b] * phi_BC[b, c]
        for a in range(2) for b in range(2) for c in range(2)
    )
    print(f"\nVerification (brute force): Z = {Z_brute:.6f}")
    match = np.isclose(Z, Z_brute)
    print(f"Match: {match}")
    
    return match


def demo_grid_2x2():
    """Demo: 2x2 Grid Ising Model"""
    print("=" * 60)
    print("Demo: 2x2 Grid Ising Model")
    print("=" * 60)
    
    var_domains = {"X00": 2, "X01": 2, "X10": 2, "X11": 2}
    
    J = 0.5
    psi = np.array([[np.exp(J), np.exp(-J)], [np.exp(-J), np.exp(J)]])
    
    factors = {
        "f_00_01": (("X00", "X01"), psi),
        "f_00_10": (("X00", "X10"), psi),
        "f_01_11": (("X01", "X11"), psi),
        "f_10_11": (("X10", "X11"), psi),
    }
    
    print("\nFactor Graph:")
    print("  X00 -- X01")
    print("   |      |")
    print("  X10 -- X11")
    print(f"  Coupling J = {J}")
    
    Z = compute_partition_function(var_domains, factors)
    print(f"\nPartition function Z = {Z:.6f}")
    
    marginals = compute_marginals(var_domains, factors)
    print("\nMarginal distributions:")
    for var, prob in sorted(marginals.items()):
        print(f"  P({var}) = [{prob[0]:.4f}, {prob[1]:.4f}]")
    
    Z_brute = sum(
        psi[x00, x01] * psi[x00, x10] * psi[x01, x11] * psi[x10, x11]
        for x00 in range(2) for x01 in range(2) for x10 in range(2) for x11 in range(2)
    )
    print(f"\nVerification (brute force): Z = {Z_brute:.6f}")
    match = np.isclose(Z, Z_brute)
    print(f"Match: {match}")
    
    return match


def demo_grid_3x3():
    """Demo: 3x3 Grid Ising Model"""
    print("=" * 60)
    print("Demo: 3x3 Grid Ising Model")
    print("=" * 60)
    
    var_domains = {f'X{i}{j}': 2 for i in range(3) for j in range(3)}
    
    J = 0.3
    psi = np.array([[np.exp(J), np.exp(-J)], [np.exp(-J), np.exp(J)]])
    
    factors = {}
    for i in range(3):
        for j in range(2):
            factors[f'h_{i}{j}'] = ((f'X{i}{j}', f'X{i}{j+1}'), psi)
    for i in range(2):
        for j in range(3):
            factors[f'v_{i}{j}'] = ((f'X{i}{j}', f'X{i+1}{j}'), psi)
    
    print("\nFactor Graph:")
    print("  X00 -- X01 -- X02")
    print("   |      |      |")
    print("  X10 -- X11 -- X12")
    print("   |      |      |")
    print("  X20 -- X21 -- X22")
    print(f"  Coupling J = {J}, {len(factors)} factors")
    
    Z = compute_partition_function(var_domains, factors)
    print(f"\nPartition function Z = {Z:.6f}")
    
    marginals = compute_marginals(var_domains, factors)
    print("\nMarginal distributions (corners):")
    for var in ["X00", "X02", "X20", "X22"]:
        prob = marginals[var]
        print(f"  P({var}) = [{prob[0]:.4f}, {prob[1]:.4f}]")
    
    Z_brute = 0.0
    for config in range(512):
        bits = [(config >> i) & 1 for i in range(9)]
        x = {f'X{i}{j}': bits[i*3+j] for i in range(3) for j in range(3)}
        w = 1.0
        for name, (scope, _) in factors.items():
            w *= psi[x[scope[0]], x[scope[1]]]
        Z_brute += w
    
    print(f"\nVerification (brute force): Z = {Z_brute:.6f}")
    match = np.isclose(Z, Z_brute)
    print(f"Match: {match}")
    
    return match


def demo_sat():
    """Demo: SAT problem (satisfiability checking)"""
    print("=" * 60)
    print("Demo: SAT Problem")
    print("=" * 60)
    
    var_domains = {"X": 2, "Y": 2, "Z": 2}
    
    # CNF: (X OR Y) AND (NOT X OR Z) AND (NOT Y OR NOT Z)
    c1 = np.array([[0, 1], [1, 1]], dtype=np.float64)
    c2 = np.array([[1, 1], [0, 1]], dtype=np.float64)
    c3 = np.array([[1, 0], [1, 1]], dtype=np.float64)
    
    factors = {
        "c1": (("X", "Y"), c1),
        "c2": (("X", "Z"), c2),
        "c3": (("Y", "Z"), c3),
    }
    
    print("\nCNF Formula:")
    print("  (X OR Y) AND (NOT X OR Z) AND (NOT Y OR NOT Z)")
    
    result = hatcc_solve(var_domains, factors, semiring="sat")
    is_sat = result.Z > 0
    print(f"\nSatisfiable: {is_sat}")
    
    result_count = hatcc_solve(var_domains, factors, semiring="prob")
    print(f"Number of solutions: {int(result_count.Z)}")
    
    print("\nSolutions:")
    for x in range(2):
        for y in range(2):
            for z in range(2):
                if c1[x,y] and c2[x,z] and c3[y,z]:
                    print(f"  X={x}, Y={y}, Z={z}")
    
    return is_sat


def cmd_demo(args):
    """Execute the demo command."""
    demos = {
        "chain": demo_simple_chain,
        "grid": demo_grid_2x2,
        "grid3": demo_grid_3x3,
        "sat": demo_sat,
    }
    
    if args.example == "all":
        results = []
        for name, func in demos.items():
            try:
                passed = func()
                results.append((name, passed))
            except Exception as e:
                print(f"Error in {name}: {e}")
                import traceback
                traceback.print_exc()
                results.append((name, False))
            print()
        
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        all_passed = True
        for name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {name}: {status}")
            if not passed:
                all_passed = False
        
        return 0 if all_passed else 1
    
    elif args.example in demos:
        try:
            passed = demos[args.example]()
            return 0 if passed else 1
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print(f"Unknown example: {args.example}")
        print(f"Available: {', '.join(demos.keys())}, all")
        return 1


def cmd_test(args):
    """Execute the test command."""
    import subprocess
    
    test_dir = Path(__file__).parent / "tests"
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        return 1
    
    cmd = [sys.executable, "-m", "pytest", str(test_dir)]
    if args.verbose:
        cmd.append("-v")
    if args.coverage:
        cmd.extend(["--cov=catbp", "--cov-report=term-missing"])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def cmd_info(args):
    """Display system information."""
    print(f"CatBP v{__version__}")
    print(f"Categorical Belief Propagation with Topological Constraints")
    print()
    print("Authors:")
    print("  Juan Diego Zambrano")
    print("  Enrique ter Horst")
    print("  Sridhar Mahadevan")
    print()
    print("Reference:")
    print("  arXiv:2601.04456")
    print("  'Categorical Belief Propagation: Sheaf-Theoretic Inference")
    print("   via Descent and Holonomy'")
    print()
    print("License: GPL-3.0")
    print()
    print("Available semirings:")
    print("  prob    - Probability semiring (+, ×)")
    print("  logprob - Log-probability semiring (logsumexp, +)")
    print("  sat     - Boolean semiring (OR, AND)")
    print()
    print("Python:", sys.version.split()[0])
    print("NumPy:", np.__version__)
    
    try:
        import scipy
        print("SciPy:", scipy.__version__)
    except ImportError:
        print("SciPy: not installed")
    
    try:
        import networkx
        print("NetworkX:", networkx.__version__)
    except ImportError:
        print("NetworkX: not installed")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="catbp",
        description="CatBP: Categorical Belief Propagation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Solve from JSON file
  catbp solve --input problem.json --output result.json
  
  # Solve with command-line specification
  catbp solve --vars "A:2,B:2" --factors "f1:A,B:[[0.9,0.1],[0.2,0.8]]"
  
  # Run demos
  catbp demo --example chain
  catbp demo --example all
  
  # Run tests
  catbp test -v
  
  # Show info
  catbp info

For more information, see: https://arxiv.org/abs/2601.04456
"""
    )
    
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"CatBP {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Solve command
    solve_parser = subparsers.add_parser("solve", help="Solve a factor graph problem")
    solve_parser.add_argument("--input", "-i", type=str, help="Input JSON file")
    solve_parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    solve_parser.add_argument("--vars", type=str, help="Variables spec: 'A:2,B:3'")
    solve_parser.add_argument("--factors", type=str, help="Factors spec: 'f1:A,B:[[...]]'")
    solve_parser.add_argument(
        "--semiring", "-s",
        choices=["prob", "logprob", "sat"],
        default="prob",
        help="Semiring to use (default: prob)"
    )
    solve_parser.add_argument(
        "--marginals", "-m",
        action="store_true",
        help="Compute and display marginals"
    )
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demonstration examples")
    demo_parser.add_argument(
        "--example", "-e",
        choices=["chain", "grid", "grid3", "sat", "all"],
        default="all",
        help="Which example to run (default: all)"
    )
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run test suite")
    test_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    test_parser.add_argument("--coverage", "-c", action="store_true", help="With coverage")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    if args.command == "solve":
        return cmd_solve(args)
    elif args.command == "demo":
        return cmd_demo(args)
    elif args.command == "test":
        return cmd_test(args)
    elif args.command == "info":
        return cmd_info(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
