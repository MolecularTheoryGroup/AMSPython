# AMSPython - Quantum Chemistry Analysis Tools

A collection of Python utilities for analyzing quantum chemistry calculations, with a focus on QTAIM (Quantum Theory of Atoms in Molecules) analysis and bond critical point eigenvector dynamics.

## Featured Tool: Bond Critical Point Eigenvector Rotation Analysis

This repository's main tool analyzes the rotation of bond critical point eigenvectors throughout chemical reactions using NEB (Nudged Elastic Band) calculations.

### Key Features

- **Comprehensive Eigenvector Tracking**: Follow physical eigenvectors through eigenvalue crossings
- **4-Way Continuity Correction**: Automatically detect and fix swaps, sign flips, and combinations
- **Multi-Density Analysis**: Consistent tracking across Total, A, and B electron densities
- **Real-Time Correction**: Apply fixes before visualization and analysis

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

| Calculation | EV1 Rotation | EV2 Rotation | Physical Interpretation |
|-------------|--------------|--------------|------------------------|
| origEEF     | 3.2°         | 3.3°         | Minimal field effects on primary eigenvectors |
| revEEF      | 12.8°        | 12.5°        | Moderate rotation with reversed field |
| noEEF       | 12.0°        | 11.7°        | Baseline rotation without field |

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