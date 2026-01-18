"""
Example: Simple chain factor graph.

A--B--C with pairwise factors.
"""

import numpy as np
from catbp.hatcc import hatcc_solve, compute_marginals


def main():
    # Define variable domains
    var_domains = {
        "A": 2,
        "B": 2,
        "C": 2,
    }
    
    # Define factors
    # Unary on A
    phi_A = np.array([0.6, 0.4])
    
    # Pairwise on (A, B)
    phi_AB = np.array([
        [0.9, 0.1],
        [0.2, 0.8]
    ])
    
    # Pairwise on (B, C)
    phi_BC = np.array([
        [0.3, 0.7],
        [0.5, 0.5]
    ])
    
    factors = {
        "f_A": (("A",), phi_A),
        "f_AB": (("A", "B"), phi_AB),
        "f_BC": (("B", "C"), phi_BC),
    }
    
    # Solve
    print("Running HATCC solver on simple chain A--B--C...")
    result = hatcc_solve(var_domains, factors)
    
    print(f"\nPartition function Z = {result.Z:.6f}")
    
    # Compute marginals
    marginals = compute_marginals(var_domains, factors)
    
    print("\nMarginal distributions:")
    for var, marg in marginals.items():
        print(f"  P({var}) = {marg}")
    
    # Verify by brute force
    print("\n--- Verification by brute force ---")
    Z_brute = 0.0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                w = phi_A[a] * phi_AB[a, b] * phi_BC[b, c]
                Z_brute += w
    
    print(f"Z (brute force) = {Z_brute:.6f}")
    print(f"Z (HATCC)       = {result.Z:.6f}")
    print(f"Match: {np.isclose(Z_brute, result.Z)}")


if __name__ == "__main__":
    main()
