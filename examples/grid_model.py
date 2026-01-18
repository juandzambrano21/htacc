"""
Example: 2x2 Grid Ising-like model.

  X00 -- X01
   |      |
  X10 -- X11

This creates a cycle in the factor graph, demonstrating holonomy handling.
"""

import numpy as np
from catbp.hatcc import hatcc_solve, compute_marginals


def ising_potential(J: float = 1.0) -> np.ndarray:
    """Create Ising pairwise potential."""
    return np.array([
        [np.exp(J), np.exp(-J)],
        [np.exp(-J), np.exp(J)]
    ])


def main():
    # Define variable domains (binary)
    var_domains = {
        "X00": 2,
        "X01": 2,
        "X10": 2,
        "X11": 2,
    }
    
    # Coupling strength
    J = 0.5
    psi = ising_potential(J)
    
    # Define pairwise factors (edges of the grid)
    factors = {
        "f_00_01": (("X00", "X01"), psi),  # Top edge
        "f_00_10": (("X00", "X10"), psi),  # Left edge
        "f_01_11": (("X01", "X11"), psi),  # Right edge
        "f_10_11": (("X10", "X11"), psi),  # Bottom edge
    }
    
    print("Running HATCC solver on 2x2 grid Ising model...")
    print(f"Coupling J = {J}")
    
    result = hatcc_solve(var_domains, factors)
    
    print(f"\nPartition function Z = {result.Z:.6f}")
    
    # Compute marginals
    marginals = compute_marginals(var_domains, factors)
    
    print("\nMarginal distributions:")
    for var in sorted(marginals.keys()):
        marg = marginals[var]
        print(f"  P({var}) = [{marg[0]:.4f}, {marg[1]:.4f}]")
    
    # Verify by brute force
    print("\n--- Verification by brute force ---")
    Z_brute = 0.0
    for x00 in range(2):
        for x01 in range(2):
            for x10 in range(2):
                for x11 in range(2):
                    w = (psi[x00, x01] * psi[x00, x10] * 
                         psi[x01, x11] * psi[x10, x11])
                    Z_brute += w
    
    print(f"Z (brute force) = {Z_brute:.6f}")
    print(f"Z (HATCC)       = {result.Z:.6f}")
    print(f"Match: {np.isclose(Z_brute, result.Z)}")


if __name__ == "__main__":
    main()
