# AMSPython - Quantum Chemistry Analysis Tools

A collection of Python utilities for analyzing quantum chemistry calculations, with a focus on QTAIM (Quantum Theory of Atoms in Molecules) analysis and bond critical point eigenvector dynamics.

## Featured Tool: Bond Critical Point Eigenvector Rotation Analysis

This repository's main tool analyzes the rotation of bond critical point eigenvectors throughout chemical reactions using NEB (Nudged Elastic Band) calculations.

### Key Features

- **Comprehensive Eigenvector Tracking**: Follow physical eigenvectors through eigenvalue crossings
- **4-Way Continuity Correction**: Automatically detect and fix swaps, sign flips, and combinations
- **Multi-Density Analysis**: Consistent tracking across Total, A, and B electron densities
- **Enhanced Data Output**: Dual CSV system with detailed trajectories and statistical summaries
- **Rotation Dynamics Analysis**: Frame-to-frame velocity and intermediate motion quantification
- **Complete Documentation**: Timestamped reports with scientific interpretation

### Quick Start

```bash
# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run analysis
python scripts/bcp_eigenvector_rotation_analysis.py
```

### Example Results

For Fe1-N44 bond analysis in heme-propane reaction:

| Calculation | Net Rotation | Max Range | Intermediate Motion | Interpretation |
|-------------|--------------|-----------|--------------------|-----------------------|
| origEEF     | 3.2°         | 21.2°     | 18.0°              | Significant hidden dynamics |
| revEEF      | 12.8°        | 14.2°     | 1.4°               | Steady progression |
| noEEF       | 12.0°        | 13.5°     | 1.5°               | Consistent rotation |

**Key Insight**: The origEEF calculation shows dramatic intermediate eigenvector motion (21.2° range) that largely returns to near the starting orientation (3.2° net), revealing complex field-induced dynamics invisible in traditional analysis.

## Documentation

See **[BCP_EIGENVECTOR_ANALYSIS.md](BCP_EIGENVECTOR_ANALYSIS.md)** for complete documentation including:

- Detailed installation and usage instructions
- Algorithm descriptions and technical details
- Scientific applications and interpretation
- Troubleshooting and development notes

## Repository Structure

```
AMSPython/
├── scripts/
│   ├── bcp_eigenvector_rotation_analysis.py  # Main analysis tool
│   ├── ADF_NEB_bcp_analysis.py              # ADF-specific utilities
│   └── ...                                   # Other quantum chemistry tools
├── bcp_rotation_plots/                       # Analysis output directory
│   ├── Fe1_N44_angles.csv                   # Detailed time-series data
│   ├── Fe1_N44_summary.csv                  # Statistical summary data
│   ├── bcp_analysis_output_*.txt            # Comprehensive reports
│   └── *.png                                # Visualization plots
├── BCP_EIGENVECTOR_ANALYSIS.md             # Comprehensive documentation
├── requirements.txt                          # Python dependencies
└── His_propane_NEB_NF_cp_info_full.csv     # Sample QTAIM data
```

## Other Analysis Tools

The repository also contains utilities for:

- **ADF Output Processing**: Extract critical point information from ADF calculations
- **BAND RKF Analysis**: Process BAND calculation results
- **Density Analysis**: Various electron density analysis tools

## Contributing

This project is part of the Molecular Theory Group's computational chemistry toolkit. Contributions welcome for:

- Algorithm improvements
- New analysis methods
- Documentation and examples
- Performance optimizations

## Citation

If you use these tools in scientific work, please cite appropriately and reference:
```
AMSPython - Quantum Chemistry Analysis Tools
Molecular Theory Group
https://github.com/MolecularTheoryGroup/AMSPython
```

---

**Developed by**: Molecular Theory Group  
**License**: See project license for terms of use