# Bond Critical Point Eigenvector Rotation Analysis

A Python utility for analyzing the rotation of bond critical point (BCP) eigenvectors throughout chemical reactions using QTAIM (Quantum Theory of Atoms in Molecules) data from NEB (Nudged Elastic Band) calculations.

## Overview

This tool analyzes how the eigenvectors of the Hessian matrix at bond critical points rotate during chemical reactions. It handles eigenvector continuity, including eigenvalue crossings, sign ambiguity, and eigenvector swapping that occur during quantum chemical calculations.

## Features

### Core Capabilities
- **Eigenvector Rotation Analysis**: Calculate rotation angles of BCP eigenvectors throughout reaction pathways
- **Multi-Density Support**: Analyze Total Density, Density A (spin-up), and Density B (spin-down) consistently
- **Multiple Calculation Types**: Handle different electronic field configurations (EEF) simultaneously
- **Automated Continuity Correction**: Detect and fix eigenvector discontinuities using comprehensive 4-way checking

### Advanced Algorithms
- **Physical Eigenvector Tracking**: Follow the same physical eigenvectors through eigenvalue crossings
- **Comprehensive Continuity Detection**: 4-way algorithm checks for:
  1. No change (original assignment)
  2. Sign flip (180° eigenvector rotation)
  3. Eigenvector swap (EV1 ↔ EV2)
  4. Combined swap and sign flip
- **Cross-Density Consistency**: Maintains physical relationships between density components
- **Real-time Correction**: Applies corrections before visualization and analysis

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup
```bash
# Clone or navigate to the project directory
cd /path/to/AMSPython

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- `pandas >= 2.0.0` - Data manipulation and analysis
- `numpy >= 1.24.0` - Numerical computations
- `matplotlib >= 3.7.0` - Plotting and visualization
- `seaborn >= 0.12.0` - Statistical visualization

## Usage

### Input Data Format
The tool expects CSV data from QTAIM analysis with the following columns:
- `ATOMS`: Bond critical point identifier (e.g., "Fe1-N44")
- `JOB_NAME`: Calculation identifier with image numbers
- `EIGENVALUES OF HESSIAN MATRIX_X/Y/Z`: Eigenvalues for Total Density
- `EIGENVALUES_A_X/Y/Z`: Eigenvalues for Density A
- `EIGENVALUES_B_X/Y/Z`: Eigenvalues for Density B
- `EIGENVECTORS (ORTHONORMAL) OF HESSIAN MATRIX (COLUMNS)_EV1/2/3_X/Y/Z`: Eigenvectors for Total Density
- Similar columns for Density A and B components

### Running the Analysis
```bash
# Activate virtual environment
source venv/bin/activate

# Run the main analysis
python scripts/bcp_eigenvector_rotation_analysis.py
```

### Configuration
Edit the configuration constants in `bcp_eigenvector_rotation_analysis.py`:
```python
# Input data file
CSV_FILE = "His_propane_NEB_NF_cp_info_full.csv"

# Bond critical points to analyze
BOND_CRITICAL_POINTS = ['Fe1-N44']  # Add more BCPs as needed

# Output directory
OUTPUT_DIR = Path("bcp_rotation_plots")
```

## Output

### Visualizations
The tool generates several types of plots:
- **Individual EV1/EV2 plots** for each calculation type
- **EV3 comparison plots** across all calculation types
- **Separate density type visualizations** for detailed analysis

### Data Files
- **CSV angle data**: Complete angle measurements throughout the reaction
- **Rotation summary**: Start-to-end rotation values for all eigenvectors
- **Corrected datasets**: Data after comprehensive continuity correction

### Console Output
```
================================================================================
BOND CRITICAL POINT EIGENVECTOR ROTATION ANALYSIS
================================================================================

Loading data from His_propane_NEB_NF_cp_info_full.csv...
Loaded 600 rows

============================================================
Processing BCP: Fe1-N44
============================================================

Applying comprehensive eigenvector swap and sign flip correction...
  Applied 14 corrections to Fe1-N44 His_propane_NEB_NF_origEEF 
  Applied 2 corrections to Fe1-N44 His_propane_NEB_NF_origEEF A

================================================================================
START-TO-END EIGENVECTOR ROTATION SUMMARY
================================================================================

Fe1-N44:
--------------------------------------------------------------------------------
Calculation     Eigenvector     Density              Rotation (deg) 
--------------------------------------------------------------------------------
origEEF         EV1             Total Density                  3.20°
origEEF         EV1             Density A                      2.27°
origEEF         EV1             Density B                     35.17°
...
```

## Algorithm Details

### Eigenvector Continuity Problem
During chemical reactions, several issues can cause artificial discontinuities in eigenvector trajectories:

1. **Eigenvalue Crossings**: When eigenvalues become nearly equal, eigenvectors can swap labels
2. **Sign Ambiguity**: Eigenvectors v and -v represent the same physical state
3. **Numerical Instabilities**: Small changes can cause large reorientations near crossings
4. **Cross-Density Inconsistencies**: Different density types may show different swapping patterns

### Solution: 4-Way Continuity Checking
At each NEB image i+1, the algorithm tests four possibilities:

```python
candidates = [
    (curr_ev1, curr_ev2, "no_change"),           # Keep original assignment
    (180-curr_ev1, 180-curr_ev2, "sign_flip"),  # Flip both eigenvector signs
    (curr_ev2, curr_ev1, "swap"),               # Swap EV1 ↔ EV2
    (180-curr_ev2, 180-curr_ev1, "swap_flip")  # Swap and flip signs
]
```

The algorithm selects the option that minimizes the total angular change from the previous image:
```python
total_diff = abs(new_ev1 - prev_ev1) + abs(new_ev2 - prev_ev2)
```

### Cross-Density Consistency
The tool ensures that Total Density, Density A, and Density B maintain consistent physical eigenvector assignments throughout the reaction, respecting the fundamental relationship:
```
Total Density ≈ (Density A + Density B) / 2
```

## Scientific Applications

### Transition Metal Complexes
This tool is particularly valuable for analyzing:
- **Bond formation/breaking** in catalytic cycles
- **Electronic rearrangements** during oxidation state changes
- **Ligand binding dynamics** in coordination complexes
- **Spin state transitions** in paramagnetic systems

### External Electric Field Effects
Compare eigenvector rotations under different conditions:
- **origEEF**: Original electric field configuration
- **revEEF**: Reversed electric field
- **noEEF**: No external electric field

### Example Results
For the Fe1-N44 bond in a heme-propane reaction system:
- **origEEF**: Minimal rotation (~3°) in Total/A densities, significant rotation (~35°) in B density
- **revEEF/noEEF**: Consistent small rotations (~10-13°) across all densities
- **Physical interpretation**: External electric fields primarily affect spin-density components

## Technical Implementation

### Key Functions
- `calculate_angles_for_bcp()`: Extract and process eigenvector data
- `apply_comprehensive_eigenvector_correction()`: 4-way continuity checking
- `track_physical_eigenvectors_across_densities()`: Cross-density consistency
- `plot_eigenvector_rotation()`: Generate visualizations

### Performance Considerations
- **Memory efficient**: Processes data in chunks by BCP and calculation type
- **Scalable**: Handles multiple BCPs and calculation types simultaneously
- **Robust**: Comprehensive error handling for missing or invalid data

## Troubleshooting

### Common Issues
1. **Missing Data**: Ensure all required columns are present in input CSV
2. **Eigenvalue Degeneracies**: Algorithm handles near-degenerate cases automatically
3. **Large Discontinuities**: Check for unexpected eigenvalue crossings in raw data
4. **Visualization Issues**: Ensure matplotlib backend is properly configured

### Validation
The tool includes extensive validation:
- **Continuity checks**: Warns about large angle changes (>15°)
- **Cross-density verification**: Checks density sum relationships
- **Physical constraints**: Ensures eigenvector orthonormality

## Citation

If you use this tool in scientific publications, please cite:
```
Bond Critical Point Eigenvector Rotation Analysis Tool
Developed for QTAIM analysis of chemical reaction pathways
https://github.com/MolecularTheoryGroup/AMSPython
```

## Development History

This tool was developed to address the complex challenges of eigenvector continuity in quantum chemical calculations. Key development phases:

1. **Initial Implementation**: Basic eigenvector extraction and angle calculation
2. **Continuity Detection**: Sign consistency and simple swap detection
3. **Cross-Density Integration**: Ensuring consistency across spin densities
4. **Comprehensive Correction**: 4-way algorithm for all discontinuity types
5. **Pipeline Integration**: Real-time correction before visualization

## Contributing

Contributions are welcome! Please focus on:
- **Algorithm improvements**: Enhanced continuity detection methods
- **Performance optimization**: Faster processing for large datasets
- **Visualization enhancements**: Better plotting and analysis tools
- **Documentation**: Examples and use cases

## License

This project is part of the Molecular Theory Group's computational chemistry tools.
For usage permissions and licensing, contact the development team.

---

**Author**: Molecular Theory Group  
**Last Updated**: October 2025  
**Version**: 1.0 - Comprehensive Eigenvector Continuity Analysis