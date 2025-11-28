# ADF Dimer Bond Critical Point Analysis

## Overview

This tool analyzes dimers (two-atom systems) from AMS/ADF calculations, extracting bond critical point (BCP) properties and generating visualizations. It processes `adf.rkf` files to compute BCP density, Hessian eigenvalues, and derives the `ρTan` metric for characterizing bond interactions.

## Features

- **Automated RKF File Processing**: Walks directory trees to find and analyze all `adf.rkf` files
- **BCP Property Extraction**: Extracts charge density, Hessian matrix, and eigenvalues at the bond critical point
- **Advanced Metrics**: Calculates `ρTan = ρ(r_BCP) × √(λ+ / |λ-|)` for bond characterization
- **Data Export**: Results saved to CSV for further analysis
- **Visualization**: Generates publication-quality plots showing energy, density, and eigenvalue trends

## Installation

Ensure you have the required dependencies:

```bash
pip install numpy matplotlib
```

The tool requires access to the AMS/ADF `amsreport` command-line utility.

## Usage

### Basic Usage

```bash
python ADF_dimer_bcp_analysis.py <input_directory> [options]
```

### Command-Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `input_dir` | — | Directory containing ADF calculation results | Required |
| `--output` | `-o` | Output CSV file path | `./dimer_bcp_analysis.csv` |
| `--output-dir` | `-d` | Directory for output files | Current directory |
| `--plots` | — | Generate visualization plots | False |

### Examples

**Basic analysis:**
```bash
python ADF_dimer_bcp_analysis.py /path/to/adf/results
```

**With custom output location and plots:**
```bash
python ADF_dimer_bcp_analysis.py /path/to/adf/results -o results.csv -d ./output --plots
```

## Output

### CSV File

Contains the following columns:

| Column | Unit | Description |
|--------|------|-------------|
| `job_name` | — | Name of the calculation job |
| `atom1` | — | First atom type |
| `atom2` | — | Second atom type |
| `separation` | bohr | Atomic separation distance |
| `lambda_neg` | a.u. | Negative eigenvalue (concentration) |
| `lambda_pos` | a.u. | Positive eigenvalue (depletion) |
| `energy` | Hartree | Total bond energy |
| `rho_bcp` | a.u. | Charge density at BCP |
| `rho_tan` | — | Tangential density metric |

Rows are sorted by separation distance (ascending).

### Plots

When `--plots` is specified, two PNG files are generated:

1. **`dimer_bcp_analysis_energy_vs_separation.png`**
   - Dual-axis plot showing bond energy and ρTan vs separation distance
   - Helps visualize bond stability and electron density trends

2. **`dimer_bcp_analysis_eigenvalues_vs_separation.png`**
   - Shows negative and positive eigenvalues vs separation
   - Illustrates Hessian curvature changes along the reaction coordinate

## Physical Interpretation

### Bond Critical Point (BCP)

The BCP is located at a saddle point of the electron density where the density gradient is zero. It lies between bonded atoms and provides insight into bond character.

### Hessian Eigenvalues

For a proper BCP (rank-3, signature 3):
- **λ- (negative eigenvalue)**: Concentration of density along the bond axis
- **λ+ (positive eigenvalue)**: Density depletion perpendicular to the bond axis
- The ratio λ+/|λ-| indicates the degree of bond directionality

### ρTan Metric

The `ρTan = ρ(r_BCP) × √(λ+ / |λ-|)` metric combines:
- Bond density strength (ρ)
- Geometric directionality (λ+/|λ-|)

Higher values indicate stronger, more directional bonds.

## Technical Details

### File Structure

The tool expects ADF calculations with the following output:
```
input_directory/
├── job_name_1/
│   └── adf.rkf
├── job_name_2/
│   └── adf.rkf
└── ...
```

### Hessian Matrix Parsing

The charge density Hessian is stored as an upper triangular matrix (6 values per CP):
```
Stored order: XX, XY, XZ, YY, YZ, ZZ
Reconstructed as:
[XX  XY  XZ]
[XY  YY  YZ]
[XZ  YZ  ZZ]
```

### BCP Identification

The tool automatically identifies the BCP using its critical point signature code:
- Signature 3.0 = Bond critical point (rank 3, signature -1)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Error running amsreport" | Ensure `amsreport` is in your PATH and AMS is properly installed |
| "BCP not found" | Some calculations may not have critical points detected; check ADF settings |
| "FAILED (coordinate parsing)" | Verify RKF file integrity; re-run ADF calculation if needed |
| Missing plots | Use `--plots` flag; ensure matplotlib is installed |

## Example Workflow

```bash
# Run analysis on NEB pathway
python ADF_dimer_bcp_analysis.py ./plams_workdir/Cys_propane_NEB_p01 \
    -o neb_analysis.csv \
    -d ./results \
    --plots

# View the CSV
cat results/neb_analysis.csv

# Results contain structure-property relationships for the entire NEB path
```

## References

- **Critical Point Analysis**: Bader, R. F. W. (1990). Atoms in Molecules: A Quantum Theory.
- **AMS/ADF Documentation**: [SCM Documentation](https://www.scm.com/doc/)
- **Bond Critical Point Theory**: Popelier, P. L. A. (2000). Computational Medicinal Chemistry for Drug Discovery.

## Author

Part of the AMSPython analysis toolkit for AMS/ADF quantum chemistry calculations.

## License

See repository LICENSE file.
