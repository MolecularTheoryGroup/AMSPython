# Bond Critical Point Eigenvector Rotation Analysis

A Python utility for analyzing the rotation of bond critical point (BCP) eigenvectors throughout chemical reactions using QTAIM (Quantum Theory of Atoms in Molecules) data from NEB (Nudged Elastic Band) calculations.

## Overview

This tool analyzes how the eigenvectors of the Hessian matrix at bond critical points rotate during chemical reactions. It handles eigenvector continuity, including eigenvalue crossings, sign ambiguity, and eigenvector swapping that occur during quantum chemical calculations.

## Features

### Comprehensive Eigenvector Tracking
- **4-way continuity checking**: Detects and corrects sign flips, swaps, and combinations
- **Cross-density consistency**: Maintains physical continuity across Total, A, and B densities
- **Robust correction algorithm**: Handles complex eigenvalue crossings and discontinuities

### Multiple Analysis Types
- **Standard rotation plots**: Eigenvector angles vs. reaction coordinate  
- **Enhanced rotation rate analysis**: Velocity of rotation between consecutive images
- **Maximum rotation range calculations**: Full extent of eigenvector motion
- **Cross-calculation comparisons**: Compare different electric field conditions
- **Intermediate motion analysis**: Quantifies dynamics beyond net rotation

### Visualization Suite
- **Individual calculation plots**: Separate analysis for each field condition
- **Cross-density visualization**: Compare Total, A, and B density behaviors
- **Rotation rate plots**: Show dynamic changes throughout reaction
- **Publication-ready figures**: High-quality PNG output with clear labeling

### Complete Data Output
- **Detailed CSV files**: Full time-series data with relative angle changes
- **Summary CSV files**: Statistical analysis with start-to-end and maximum rotations
- **Timestamped text reports**: Comprehensive analysis with scientific interpretation
- **Multiple output formats**: Ready for further analysis and publication

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
- **Relative rotation rate plots** showing angular velocity between consecutive images (one per density per calculation type)
- **Separate density type visualizations** for detailed analysis

### Data Files

#### Detailed Time-Series Data
- **`{BCP}_angles.csv`**: Complete trajectory analysis
  - `Angle_to_Reference_deg`: Absolute rotation from initial reference
  - `Relative_Angle_deg`: Frame-to-frame rotation changes
  - Full time-series for all 25 NEB images
  - All eigenvectors, densities, and calculation types

#### Statistical Summary Data  
- **`{BCP}_summary.csv`**: Condensed quantitative analysis
  - `Start_to_End_Rotation_deg`: Net rotation from start to finish
  - `Maximum_Range_deg`: Largest angle span during reaction
  - `Intermediate_Motion_deg`: Extra motion beyond net change
  - One row per eigenvector/density/calculation combination

#### Analysis Reports
- **`bcp_analysis_output_{timestamp}.txt`**: Complete interpretation
  - Detailed rotation summaries with scientific insights
  - Maximum range analysis and interpretations
  - Key findings and mechanistic implications

#### Complete Output Structure
```
bcp_rotation_plots/
├── Fe1_N44_angles.csv                        # Detailed time-series data
├── Fe1_N44_summary.csv                       # Statistical summaries  
├── bcp_analysis_output_20251009_111202.txt   # Comprehensive reports
├── Fe1_N44_noEEF_EV1_EV2.png                 # Standard rotation plots
├── Fe1_N44_origEEF_EV1_EV2.png               # (one per calculation)
├── Fe1_N44_revEEF_EV1_EV2.png                # 
├── Fe1_N44_EV3_all_calcs.png                 # EV3 comparison across methods
├── Fe1_N44_noEEF_Total_Density_rotation_rates.png    # Rotation rate analysis
├── Fe1_N44_noEEF_Density_A_rotation_rates.png        # (9 plots total:
├── Fe1_N44_noEEF_Density_B_rotation_rates.png        #  3 calc × 3 density)
├── Fe1_N44_origEEF_Total_Density_rotation_rates.png  # 
├── ... (additional rate plots)                        #
└── Fe1_N44_revEEF_Density_B_rotation_rates.png      # 
```

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

================================================================================
MAXIMUM EIGENVECTOR ROTATION RANGES
================================================================================

Fe1-N44:
--------------------------------------------------------------------------------
Calculation  Eigenvector  Density              Max Range (deg)
--------------------------------------------------------------------------------
origEEF      EV1          Total Density                21.19°
origEEF      EV1          Density A                     9.19°
origEEF      EV1          Density B                    39.80°
origEEF      EV2          Total Density                21.09°
origEEF      EV2          Density A                     8.77°
origEEF      EV2          Density B                    19.66°
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

#### Fe1-N44 Bond Critical Point Analysis

**Net Rotations (Start-to-End):**
- **origEEF**: Minimal rotation (~3°) in Total/A densities, significant rotation (~35°) in B density
- **revEEF**: Substantial rotation (~13°) across all densities  
- **noEEF**: Moderate rotation (~12°) with consistent behavior across densities

**Maximum Rotation Ranges:**
- **origEEF Total Density**: 21.2° range despite only 3.2° net rotation → significant intermediate motion
- **origEEF B Density**: 39.8° range with 35.2° net rotation → complex dynamics with large final displacement
- **All configurations**: EV3 shows minimal rotation (bond axis direction preserved)

**Key Scientific Insights:**
- **Hidden dynamics**: Large intermediate motion even when net rotation is small
- **Field sensitivity**: External electric fields dramatically alter eigenvector behavior  
- **Density dependence**: A and B densities can show very different rotation patterns

#### Intermediate Motion Discovery
The analysis revealed a crucial distinction between **net rotation** and **maximum rotation range**:

| Analysis Type | origEEF Example | Scientific Interpretation |
|---------------|----------------|---------------------------|
| **Net Rotation** | 3.2° (Total Density) | Final orientation change |
| **Max Range** | 21.2° (Total Density) | Full motion during reaction |
| **Intermediate Motion** | 18.0° | Hidden dynamics that return to near-start |

**Scientific Significance**: This discovery shows that traditional start-to-end QTAIM analysis misses substantial intermediate eigenvector dynamics, revealing complex field-induced reorganization invisible in conventional approaches.

## Technical Implementation

### Key Functions
- `calculate_angles_for_bcp()`: Extract and process eigenvector data
- `apply_comprehensive_eigenvector_correction()`: 4-way continuity checking
- `track_physical_eigenvectors_across_densities()`: Cross-density consistency
- `plot_eigenvector_rotation()`: Generate standard rotation visualizations
- `plot_relative_rotation_rates()`: Create rotation velocity analysis plots
- `calculate_maximum_rotations()`: Determine full rotation ranges
- `save_rotation_data_to_csv()`: Export detailed time-series data with relative angles  
- `save_summary_data_to_csv()`: Export statistical summaries with intermediate motion analysis
- `save_analysis_output_to_file()`: Generate comprehensive text reports

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

## Development History

This tool was developed to address the complex challenges of eigenvector continuity in quantum chemical calculations. Key development phases:

1. **Initial Implementation**: Basic eigenvector extraction and angle calculation
2. **Continuity Detection**: Sign consistency and simple swap detection
3. **Cross-Density Integration**: Ensuring consistency across spin densities
4. **Comprehensive Correction**: 4-way algorithm for all discontinuity types
5. **Pipeline Integration**: Real-time correction before visualization

## License

This project is part of the Molecular Theory Group's computational chemistry tools.
For usage permissions and licensing, contact the development team.

---

**Author**: Molecular Theory Group  
**Last Updated**: October 2025  
## Version History

### Version 2.0 - Complete Analysis Suite (October 2025)

**Major Enhancements:**
- **Dual CSV Output System**: Detailed time-series data + statistical summaries
- **Relative Angle Analysis**: Frame-to-frame rotation dynamics 
- **Intermediate Motion Quantification**: Hidden dynamics beyond net rotation
- **Comprehensive Text Reports**: Timestamped scientific documentation
- **Enhanced Visualizations**: Rotation rate plots for dynamic analysis

**Technical Improvements:**
- 4-way eigenvector continuity checking (swap + sign flip detection)
- Cross-density consistency algorithms
- Real-time correction pipeline integration
- Publication-ready output formats

### Version 1.0 - Core Implementation
- Basic eigenvector rotation tracking
- Standard visualization plots
- CSV data export
- Initial continuity correction algorithms