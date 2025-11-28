#!/usr/bin/env python3
"""
ADF Dimer Bond Critical Point Analysis

This script analyzes dimers (two-atom systems) simulated with AMS.
It processes adf.rkf files, extracting geometry, energy, and BCP information,
then writes results to a CSV file and generates plots.
"""

import os
import subprocess
import csv
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import matplotlib.pyplot as plt


class DimerBCPAnalyzer:
    """Analyzes bond critical point data for dimers from ADF calculations."""
    
    def __init__(self, output_dir: str = "."):
        """
        Initialize the analyzer.
        
        Args:
            output_dir: Directory for output CSV and plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def run_amsreport(self, rkf_path: str, query: str) -> str:
        """
        Run amsreport command to extract data from RKF file.
        
        Args:
            rkf_path: Path to the adf.rkf file
            query: amsreport query string
            
        Returns:
            Output from amsreport command
        """
        try:
            result = subprocess.run(
                ["amsreport", rkf_path, "-r", query],
                cwd=os.path.dirname(rkf_path),
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error running amsreport on {rkf_path}: {e.stderr}")
            return ""
    
    def parse_coordinates(self, coord_str: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Parse coordinate string from amsreport.
        
        Args:
            coord_str: String like "0.0 0.0 0.0 0.0 0.0 1.683257577324652"
            
        Returns:
            Tuple of (separation_distance, separation_distance)
        """
        try:
            coords = [float(x) for x in coord_str.split()]
            if len(coords) == 6:
                x1, y1, z1, x2, y2, z2 = coords
                separation = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
                return separation, separation
            return None, None
        except (ValueError, IndexError):
            return None, None
    
    def parse_atom_types(self, atomtype_str: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse atom type string from amsreport.
        
        Args:
            atomtype_str: String like "O                                                       C"
            
        Returns:
            Tuple of (atom1_type, atom2_type)
        """
        try:
            # Split by whitespace and filter empty strings
            parts = [x.strip() for x in atomtype_str.split()]
            if len(parts) >= 2:
                return parts[0], parts[1]
            elif len(parts) == 1:
                return parts[0], None
            return None, None
        except Exception:
            return None, None
    
    def parse_energy(self, energy_str: str) -> Optional[float]:
        """
        Parse energy string from amsreport.
        
        Args:
            energy_str: String like "-0.2988220262757486"
            
        Returns:
            Energy value in hartree
        """
        try:
            return float(energy_str)
        except ValueError:
            return None
    
    def get_bcp_index(self, cp_codes_str: str) -> Optional[int]:
        """
        Find the index of the bond critical point (BCP).
        
        BCP is identified by signature 3.0
        
        Args:
            cp_codes_str: String like "1.0 1.0 3.0"
            
        Returns:
            Index of BCP (0-based), or None if not found
        """
        try:
            codes = [float(x) for x in cp_codes_str.split()]
            for i, code in enumerate(codes):
                if code == 3.0:
                    return i
            return None
        except ValueError:
            return None
    
    def parse_bcp_density(self, density_str: str, bcp_index: int) -> Optional[float]:
        """
        Parse charge density at BCP.
        
        Args:
            density_str: String of space-separated density values
            bcp_index: Index of the BCP
            
        Returns:
            Charge density at the BCP
        """
        try:
            densities = [float(x) for x in density_str.split()]
            if bcp_index < len(densities):
                return densities[bcp_index]
            return None
        except ValueError:
            return None
    
    def parse_bcp_hessian(self, hessian_str: str, bcp_index: int) -> Optional[np.ndarray]:
        """
        Parse the charge density Hessian matrix at BCP.
        
        The Hessian is stored as an upper triangular matrix (6 values per CP):
        XX, XY, XZ, YY, YZ, ZZ
        
        Data is stored with outer loop over matrix elements and inner loop over CPs:
        XX1 XX2 XX3 XY1 XY2 XY3 XZ1 XZ2 XZ3 YY1 YY2 YY3 YZ1 YZ2 YZ3 ZZ1 ZZ2 ZZ3
        
        Args:
            hessian_str: String of space-separated Hessian values
            bcp_index: Index of the BCP
            
        Returns:
            3x3 Hessian matrix, or None if parsing fails
        """
        try:
            hessian_values = [float(x) for x in hessian_str.split()]
            num_cps = len(hessian_values) // 6
            
            if bcp_index >= num_cps:
                return None
            
            # Extract the 6 upper triangular values for this BCP
            # They are organized as: XX, XY, XZ, YY, YZ, ZZ all interleaved by CP
            hessian_bcp = []
            for element_idx in range(6):
                position = element_idx * num_cps + bcp_index
                if position < len(hessian_values):
                    hessian_bcp.append(hessian_values[position])
            
            if len(hessian_bcp) != 6:
                return None
            
            # Reconstruct full symmetric matrix from upper triangular
            xx, xy, xz, yy, yz, zz = hessian_bcp
            hessian_matrix = np.array([
                [xx, xy, xz],
                [xy, yy, yz],
                [xz, yz, zz]
            ])
            
            return hessian_matrix
        except (ValueError, IndexError):
            return None
    
    def calculate_eigenvalues(self, hessian: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate eigenvalues of the Hessian matrix.
        
        For a BCP, we expect one negative (concentration) and two positive (depletion)
        eigenvalues. We return the negative eigenvalue and the largest positive eigenvalue.
        
        Args:
            hessian: 3x3 Hessian matrix
            
        Returns:
            Tuple of (negative_eigenvalue, positive_eigenvalue)
        """
        try:
            eigenvalues = np.linalg.eigvalsh(hessian)
            eigenvalues = np.sort(eigenvalues)  # Sort in ascending order
            
            # For BCP: one negative, two positive eigenvalues
            # eigenvalues[0] should be negative
            # eigenvalues[1] and eigenvalues[2] should be positive
            
            if eigenvalues[0] < 0 and eigenvalues[1] > 0:
                return eigenvalues[0], eigenvalues[2]  # Return largest positive
            else:
                # Fallback: just return smallest (most negative) and largest
                return eigenvalues[0], eigenvalues[2]
        except Exception:
            return None, None
    
    def analyze_rkf_file(self, rkf_path: str) -> Optional[Dict]:
        """
        Analyze a single adf.rkf file and extract all relevant data.
        
        Args:
            rkf_path: Path to the adf.rkf file
            
        Returns:
            Dictionary with analysis results, or None if analysis fails
        """
        rkf_path = str(rkf_path)
        
        # Get job name from directory name (remove .results suffix if present)
        job_name = os.path.basename(os.path.dirname(rkf_path))
        
        print(f"Analyzing {job_name}...", end=" ")
        
        # Extract coordinates
        coord_output = self.run_amsreport(rkf_path, "Geometry%xyz")
        if not coord_output:
            print("FAILED (coordinates)")
            return None
        
        separation, _ = self.parse_coordinates(coord_output)
        if separation is None:
            print("FAILED (coordinate parsing)")
            return None
        
        # Extract atom types
        atomtype_output = self.run_amsreport(rkf_path, "Geometry%atomtype")
        atom1, atom2 = self.parse_atom_types(atomtype_output)
        
        # Extract energy
        energy_output = self.run_amsreport(rkf_path, "AMSResults%Energy")
        if not energy_output:
            print("FAILED (energy)")
            return None
        
        energy = self.parse_energy(energy_output)
        if energy is None:
            print("FAILED (energy parsing)")
            return None
        
        # Extract CP information
        cp_codes_output = self.run_amsreport(rkf_path, "Properties%CP code number for (Rank,Signatu")
        bcp_index = self.get_bcp_index(cp_codes_output)
        
        if bcp_index is None:
            print("FAILED (BCP not found)")
            return None
        
        # Extract charge density at BCP
        density_output = self.run_amsreport(rkf_path, "Properties%CP density at")
        rho_bcp = self.parse_bcp_density(density_output, bcp_index)
        
        if rho_bcp is None:
            print("FAILED (BCP density)")
            return None
        
        # Extract Hessian and calculate eigenvalues
        hessian_output = self.run_amsreport(rkf_path, "Properties%CP density Hessian at")
        hessian = self.parse_bcp_hessian(hessian_output, bcp_index)
        
        if hessian is None:
            print("FAILED (Hessian parsing)")
            return None
        
        lambda_neg, lambda_pos = self.calculate_eigenvalues(hessian)
        
        if lambda_neg is None or lambda_pos is None:
            print("FAILED (eigenvalue calculation)")
            return None
        
        # Calculate rhoTan
        if lambda_neg < 0 and lambda_pos > 0:
            rho_tan = rho_bcp * np.sqrt(lambda_pos / (-lambda_neg))
        else:
            rho_tan = None
        
        print("OK")
        
        return {
            'job_name': job_name,
            'atom1': atom1,
            'atom2': atom2,
            'separation': separation,
            'lambda_neg': lambda_neg,
            'lambda_pos': lambda_pos,
            'energy': energy,
            'rho_bcp': rho_bcp,
            'rho_tan': rho_tan
        }
    
    def walk_directory(self, root_dir: str) -> None:
        """
        Walk through directory structure and analyze all adf.rkf files.
        
        Args:
            root_dir: Root directory to start walking from
        """
        root_path = Path(root_dir)
        
        for rkf_file in root_path.rglob("adf.rkf"):
            result = self.analyze_rkf_file(str(rkf_file))
            if result:
                self.results.append(result)
    
    def write_csv(self, output_file: Optional[str] = None) -> str:
        """
        Write analysis results to CSV file.
        
        Args:
            output_file: Output CSV filename. If None, uses default.
            
        Returns:
            Path to the output CSV file
        """
        if output_file is None:
            output_file = str(self.output_dir / "dimer_bcp_analysis.csv")
        else:
            output_file = str(Path(output_file))
        
        if not self.results:
            print("No results to write!")
            return output_file
        
        # Sort by separation distance
        self.results.sort(key=lambda x: x['separation'])
        
        with open(output_file, 'w', newline='') as f:
            fieldnames = [
                'job_name',
                'atom1',
                'atom2',
                'separation',
                'lambda_neg',
                'lambda_pos',
                'energy',
                'rho_bcp',
                'rho_tan'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"\nResults written to {output_file}")
        return output_file
    
    def create_plots(self, output_prefix: Optional[str] = None) -> Tuple[str, str]:
        """
        Create plots for the analysis results.
        
        Creates:
        1. A plot with dual y-axes showing bond energy and rhoTan vs separation distance
        2. A plot showing eigenvalues vs separation distance
        
        Args:
            output_prefix: Prefix for output plot files. If None, uses default.
            
        Returns:
            Tuple of (plot1_path, plot2_path)
        """
        if not self.results:
            print("No results to plot!")
            return "", ""
        
        if output_prefix is None:
            output_prefix = str(self.output_dir / "dimer_bcp_analysis")
        
        # Sort results by separation for plotting
        sorted_results = sorted(self.results, key=lambda x: x['separation'])
        
        separations = [r['separation'] for r in sorted_results]
        energies = [r['energy'] for r in sorted_results]
        rho_tans = [r['rho_tan'] if r['rho_tan'] is not None else 0 for r in sorted_results]
        lambda_negs = [r['lambda_neg'] for r in sorted_results]
        lambda_poss = [r['lambda_pos'] for r in sorted_results]
        
        # Plot 1: Energy and rhoTan vs Separation (dual y-axes)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Separation Distance (bohr)')
        ax1.set_ylabel('Bond Energy (Hartree)', color=color)
        line1 = ax1.plot(separations, energies, 'o-', color=color, label='Bond Energy')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('ρTan (rho_bcp * sqrt(λ+/|λ-|))', color=color)
        line2 = ax2.plot(separations, rho_tans, 's--', color=color, label='ρTan')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Combine legends
        lines = line1 + line2
        labels = [str(l.get_label()) for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        fig.tight_layout()
        plot1_path = f"{output_prefix}_energy_vs_separation.png"
        fig.savefig(plot1_path, dpi=150, bbox_inches='tight')
        print(f"Created plot: {plot1_path}")
        plt.close(fig)
        
        # Plot 2: Eigenvalues vs Separation
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(separations, lambda_negs, 'o-', color='tab:blue', label='λ- (negative eigenvalue)')
        ax.plot(separations, lambda_poss, 's-', color='tab:red', label='λ+ (positive eigenvalue)')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Separation Distance (bohr)')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('BCP Hessian Eigenvalues vs Separation Distance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        plot2_path = f"{output_prefix}_eigenvalues_vs_separation.png"
        fig.savefig(plot2_path, dpi=150, bbox_inches='tight')
        print(f"Created plot: {plot2_path}")
        plt.close(fig)
        
        return plot1_path, plot2_path


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze dimers from AMS/ADF calculations"
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing ADF calculation results"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output CSV file (default: ./dimer_bcp_analysis.csv)"
    )
    parser.add_argument(
        "-d", "--output-dir",
        default=".",
        help="Output directory (default: current directory)"
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate plots (default: False)"
    )
    
    args = parser.parse_args()
    
    analyzer = DimerBCPAnalyzer(output_dir=args.output_dir)
    analyzer.walk_directory(args.input_dir)
    analyzer.write_csv(args.output)
    
    if args.plots:
        analyzer.create_plots()


if __name__ == "__main__":
    main()
