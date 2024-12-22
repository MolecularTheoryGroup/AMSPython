#!/usr/local/bin/python3

import numpy as np
from math import atan, sqrt

def calculate_directionality(xx, xy, xz, yy, yz, zz, cp_type="bond"):
    """
    Calculate theta and phi values for a critical point.
    
    Args:
        xx, xy, xz, yy, yz, zz: Elements of the 3x3 symmetric Hessian matrix
        cp_type: String indicating the critical point type ("bond", "ring", or "cage")
    
    Returns:
        tuple: (theta, phi) in radians
    """
    # Construct the 3x3 symmetric matrix
    hessian = [
        [xx, xy, xz],
        [xy, yy, yz],
        [xz, yz, zz]
    ]
    
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigh(np.array(hessian))[0]
    
    if cp_type == "bond":
        theta = atan(sqrt(abs(eigenvalues[0] / eigenvalues[2])))
        phi = atan(sqrt(abs(eigenvalues[1] / eigenvalues[2])))
    else:  # ring or cage
        theta = atan(sqrt(abs(eigenvalues[2] / eigenvalues[0])))
        phi = atan(sqrt(abs(eigenvalues[1] / eigenvalues[0])))
    
    return theta, phi

# Example usage
def main():
    # Example Hessian values (from a bond point in FCC Pd)
    # Replace these values with your actual Hessian elements
    xx = -0.039287665
    xy = -0.000313931
    xz = 6.87E-05
    yy = 0.15495616
    yz = -2.52E-05
    zz = -0.033867396
    
    # Calculate for bond critical point
    theta_bond, phi_bond = calculate_directionality(xx, xy, xz, yy, yz, zz, "bond")
    print(f"Bond critical point:")
    print(f"theta = {theta_bond:.4f} radians")
    print(f"phi = {phi_bond:.4f} radians")
    
    # Calculate for ring/cage critical point
    theta_ring, phi_ring = calculate_directionality(xx, xy, xz, yy, yz, zz, "ring")
    print(f"\nRing/cage critical point:")
    print(f"theta = {theta_ring:.4f} radians")
    print(f"phi = {phi_ring:.4f} radians")

if __name__ == "__main__":
    main()