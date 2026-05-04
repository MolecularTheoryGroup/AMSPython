"""
ADF_densf_line_scan.py

Run densf with an inline grid of N equally spaced points along the straight line
between two atoms in an adf.rkf file, computing one or more requested properties.
Extracts the resulting values from the TAPE41 output and saves them to a CSV file.

Usage (as a script):
    Edit the USER SETTINGS section below and run:
        python ADF_densf_line_scan.py

Usage (as a module):
    from ADF_densf_line_scan import run_line_scan
    run_line_scan("path/to/adf.rkf", ("H21", "O7"), n_points=100,
                  variables=["density scf", "density frag"])

Atom references can be:
  - Integer input-order indices (1-based), e.g. 21
  - Label strings like "H21" or "O7" — the numeric suffix is used as the index
    and the element prefix is used for human-readable naming only.

Job naming:
  If the rkf file lives inside a directory whose name ends with ".results"
  (the standard AMS layout), the job name is derived from the parent directory
  name by stripping ".results".  Output files (t41, csv) are placed in the
  directory that contains the ".results" folder.  Otherwise the rkf file's
  stem is used as the job name and files are placed alongside the rkf.
"""

import csv
import os
import re
import subprocess
from pathlib import Path

BOHR_TO_ANGSTROM = 0.529177210903

# Mapping from densf variable keyword prefixes to TAPE41 amsreport section%key names.
# Each entry is (densf_keyword_prefix, t41_amsreport_key, csv_column_name).
# Add entries here as additional densf keywords are needed.
_VARIABLE_KEY_MAP: list[tuple[str, str, str]] = [
    ("density scf",   "SCF%Density",        "density_scf"),
    ("density frag",  "SumFrag%Density",    "density_frag"),
    ("density ortho", "Ortho%Density",      "density_ortho"),
    ("kindens scf",   "SCF%KinDens",        "kindens_scf"),
    ("kindens frag",  "SumFrag%KinDens",    "kindens_frag"),
    ("laplacian",     "SCF%DensityLap",     "laplacian_scf"),
    # DenGrad is handled as a special case in extract_line_scan_to_csv:
    # the three component keys below are read and the magnitude is computed.
    ("dengrad",       "SCF%DensityGradX",   "dengrad_mag"),  # sentinel; see _DENGRAD_KEYS
]

# T41 keys for the three DenGrad components (used to compute gradient magnitude).
_DENGRAD_KEYS = ("SCF%DensityGradX", "SCF%DensityGradY", "SCF%DensityGradZ")


def _variable_to_t41_key(variable: str) -> tuple[str, str]:
    """
    Map a densf variable string to a (t41_amsreport_key, csv_column_name) pair.
    Matching is case-insensitive on the variable string prefix.
    Raises ValueError if no match is found.
    """
    v_lower = variable.strip().lower()
    for prefix, t41_key, col_name in _VARIABLE_KEY_MAP:
        if v_lower.startswith(prefix):
            return t41_key, col_name
    raise ValueError(
        f"No TAPE41 key mapping for densf variable {variable!r}. "
        "Add an entry to _VARIABLE_KEY_MAP."
    )


# ---------------------------------------------------------------------------
# Internal helpers (shared pattern from ADF_dir_walk_densf.py)
# ---------------------------------------------------------------------------

def _run_amsreport(rkf_path: str, key: str) -> list[str]:
    """Run amsreport on a rkf/t41 file and return the whitespace-split tokens of stdout."""
    cmd = f'$AMSBIN/amsreport \'{rkf_path}\' -r "{key}"'
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True, check=True)
    return result.stdout.split()


def _derive_job_name(rkf_path: str) -> tuple[str, Path]:
    """
    Derive the job name and preferred output directory from an rkf path.

    If the immediate parent directory ends with ".results" (AMS convention),
    the job name is that directory name without the ".results" suffix, and
    output files go in the grandparent directory.

    Otherwise, the job name is the rkf file stem and output files go next to
    the rkf.

    Returns:
        (job_name, output_dir)
    """
    rkf_p = Path(rkf_path).resolve()
    parent = rkf_p.parent
    if parent.name.endswith(".results"):
        job_name = parent.name[: -len(".results")]
        output_dir = parent.parent
    else:
        job_name = rkf_p.stem
        output_dir = parent
    return job_name, output_dir


def _parse_atom_ref(ref) -> tuple[int, str]:
    """
    Parse an atom reference to a (1-based input index, label string) pair.

    Accepts:
      - int  → (int, str(int))
      - str like "H21", "O7", "21" → (21, "H21") / (7, "O7") / (21, "21")
    """
    if isinstance(ref, int):
        return ref, str(ref)
    s = str(ref).strip()
    m = re.search(r"\d+", s)
    if not m:
        raise ValueError(f"Cannot parse atom reference: {ref!r}")
    return int(m.group()), s


def _get_coords_by_input_index(rkf_path: str, atom_input_indices: list[int]) -> list[tuple[float, float, float]]:
    """
    Return Bohr coordinates for 1-based input-order atom indices.

    Args:
        rkf_path: Path to the .rkf file.
        atom_input_indices: 1-based input-order atom indices.

    Returns:
        List of (x, y, z) tuples in Bohr, one per requested index.
    """
    # Fetch coordinates in internal order (Bohr)
    xyz_tokens = _run_amsreport(rkf_path, "Geometry%xyz")
    n_atoms = len(xyz_tokens) // 3
    coords_internal = [
        (float(xyz_tokens[3 * i]),
         float(xyz_tokens[3 * i + 1]),
         float(xyz_tokens[3 * i + 2]))
        for i in range(n_atoms)
    ]

    # Build input-order -> internal-order mapping.
    # Geometry%atom order index stores 2*N values; the first N values encode the
    # permutation directly: order_ints[i] is the 1-based INTERNAL index for
    # input atom (i+1).  The second N values are the inverse permutation.
    order_tokens = _run_amsreport(rkf_path, "Geometry%atom order index")
    order_ints = [int(t) for t in order_tokens]
    input_to_internal = {i + 1: order_ints[i] for i in range(n_atoms)}

    result = []
    for idx in atom_input_indices:
        if idx not in input_to_internal:
            raise ValueError(
                f"Atom input index {idx} not found in {rkf_path}. "
                f"Valid indices: 1–{n_atoms}."
            )
        internal_idx = input_to_internal[idx]  # 1-based
        result.append(coords_internal[internal_idx - 1])
    return result


# ---------------------------------------------------------------------------
# Atom XYZ export
# ---------------------------------------------------------------------------

def write_atom_xyz(rkf_path: str, xyz_path: str) -> str:
    """
    Write a plain-text XYZ file containing index, element symbol, and
    Cartesian coordinates (Angstrom) for every atom in the system.

    Atoms are listed in 1-based input order (matching the original AMS input).

    File format (space-separated)::

        <N atoms>
        index  symbol  x(Ang)  y(Ang)  z(Ang)  source=<rkf_path>
        1  Fe    1.23456789   2.34567890   3.45678901
        ...

    Args:
        rkf_path: Path to the adf.rkf file.
        xyz_path: Destination .xyz file path.

    Returns:
        The path that was written.
    """
    rkf_path = str(Path(rkf_path).resolve())

    # --- coordinates (Bohr → Angstrom, internal order) ---
    xyz_tokens = _run_amsreport(rkf_path, "Geometry%xyz")
    n_atoms = len(xyz_tokens) // 3
    coords_internal_ang = [
        (
            float(xyz_tokens[3 * i])     * BOHR_TO_ANGSTROM,
            float(xyz_tokens[3 * i + 1]) * BOHR_TO_ANGSTROM,
            float(xyz_tokens[3 * i + 2]) * BOHR_TO_ANGSTROM,
        )
        for i in range(n_atoms)
    ]

    # --- element symbols (internal order) ---
    atomtype_tokens = _run_amsreport(rkf_path, "Geometry%atomtype")
    unique_elements = list(atomtype_tokens)
    frag_tokens = _run_amsreport(rkf_path, "Geometry%fragment and atomtype index")
    frag_ints = [int(t) for t in frag_tokens]  # length 2*N
    atomtype_indices = frag_ints[n_atoms:]       # 1-based into unique_elements
    symbols_internal = [
        unique_elements[atomtype_indices[i] - 1] for i in range(n_atoms)
    ]

    # --- input-order → internal-order mapping ---
    order_tokens = _run_amsreport(rkf_path, "Geometry%atom order index")
    order_ints = [int(t) for t in order_tokens]
    input_to_internal = {i + 1: order_ints[i] for i in range(n_atoms)}

    # --- write file ---
    with open(xyz_path, "w") as fh:
        fh.write(f"{n_atoms}\n")
        fh.write(f"{rkf_path}\n")
        for inp in range(1, n_atoms + 1):
            intn = input_to_internal[inp]
            sym = symbols_internal[intn - 1]
            x, y, z = coords_internal_ang[intn - 1]
            fh.write(
                f"{sym:<4s}  "
                f"{x:14.8f}  {y:14.8f}  {z:14.8f}\n"
            )

    print(f"XYZ written: {xyz_path}")
    return xyz_path


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _line_points(coord_a: tuple[float, float, float],
                 coord_b: tuple[float, float, float],
                 n_points: int) -> list[tuple[float, float, float]]:
    """Return N equally-spaced points (Bohr) between coord_a and coord_b (inclusive)."""
    if n_points < 2:
        raise ValueError("n_points must be at least 2 (the two endpoints).")
    ax, ay, az = coord_a
    bx, by, bz = coord_b
    pts = []
    for i in range(n_points):
        t = i / (n_points - 1)
        pts.append((ax + t * (bx - ax), ay + t * (by - ay), az + t * (bz - az)))
    return pts


def build_inline_grid(coord_a: tuple[float, float, float],
                      coord_b: tuple[float, float, float],
                      n_points: int) -> str:
    """
    Build a densf GRID Inline block for N equally-spaced points between two coordinates.

    Coordinates must be in Bohr (matching the UNITS block added to the densf input).

    Returns:
        Multi-line GRID Inline block string.
    """
    lines = ["Grid Inline"]
    for x, y, z in _line_points(coord_a, coord_b, n_points):
        lines.append(f"  {x:.8f}  {y:.8f}  {z:.8f}")
    lines.append("End")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV extraction
# ---------------------------------------------------------------------------

def extract_line_scan_to_csv(t41_path: str,
                              coord_a: tuple[float, float, float],
                              coord_b: tuple[float, float, float],
                              n_points: int,
                              variables: list[str],
                              csv_path: str,
                              label_a: str = "A",
                              label_b: str = "B") -> str:
    """
    Read densf TAPE41 results for a line scan and write them to a CSV file.

    Args:
        t41_path: Path to the densf TAPE41 output file.
        coord_a: (x, y, z) Angstrom coordinates of the first endpoint atom.
        coord_b: (x, y, z) Angstrom coordinates of the second endpoint atom.
        n_points: Number of grid points used in the line scan.
        variables: List of densf variable strings that were requested (same as
                   passed to run_line_scan), used to look up TAPE41 keys.
        csv_path: Destination CSV file path.
        label_a: Human-readable label for atom A (e.g. "H21").
        label_b: Human-readable label for atom B (e.g. "O7").

    Returns:
        Path to the written CSV file.
    """
    # Compute grid point coordinates
    pts = _line_points(coord_a, coord_b, n_points)

    # Extract each variable from the t41
    extracted: dict[str, list[float]] = {}
    col_names: list[str] = []

    for var in variables:
        t41_key, col_name = _variable_to_t41_key(var)

        if var.strip().lower().startswith("dengrad"):
            # Gradient magnitude: extract x, y, z components and compute sqrt(x²+y²+z²)
            components: list[list[float]] = []
            for key in _DENGRAD_KEYS:
                tokens = _run_amsreport(t41_path, key)
                if not tokens:
                    print(f"  Warning: no data returned for key '{key}' ({var!r})")
                    components.append([float("nan")] * n_points)
                else:
                    components.append([float(v) for v in tokens])
            mag = [
                (components[0][i] ** 2 + components[1][i] ** 2 + components[2][i] ** 2) ** 0.5
                for i in range(n_points)
            ]
            extracted[col_name] = mag
        else:
            tokens = _run_amsreport(t41_path, t41_key)
            if not tokens:
                print(f"  Warning: no data returned for key '{t41_key}' ({var!r})")
                extracted[col_name] = [float("nan")] * n_points
            else:
                values = [float(v) for v in tokens]
                if len(values) != n_points:
                    print(
                        f"  Warning: expected {n_points} values for '{t41_key}', "
                        f"got {len(values)}"
                    )
                extracted[col_name] = values

        col_names.append(col_name)

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        header = ["point", "x_bohr", "y_bohr", "z_bohr",
                  "arclength_bohr"] + col_names
        writer.writerow(header)
        for i, (x, y, z) in enumerate(pts):
            arclength = (
                sum((c - a) ** 2 for c, a in zip((x, y, z), coord_a)) ** 0.5
            )
            row = [i, f"{x:.8f}", f"{y:.8f}", f"{z:.8f}", f"{arclength:.8f}"]
            for col in col_names:
                vals = extracted[col]
                row.append(f"{vals[i]:.10e}" if i < len(vals) else "")
            writer.writerow(row)

    print(f"CSV written: {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_line_scan(rkf_path: str,
                  atom_pair: tuple,
                  n_points: int,
                  variables: list[str],
                  output_path: str | None = None,
                  csv_path: str | None = None,
                  xyz_path: str | None = None,
                  dry_run: bool = False) -> tuple[str, str | None, str | None]:
    """
    Run densf along a straight line between two atoms using an inline grid,
    then extract the computed values and save them to a CSV file.

    Args:
        rkf_path: Path to the adf.rkf file.
        atom_pair: Two atom references, each either a 1-based integer input-order
                   index or a label string like "H21" / "O7".
        n_points: Number of equally-spaced grid points (includes both endpoints).
        variables: List of densf property keyword strings, e.g.
                   ["density scf", "density frag"].
        output_path: Path for the output .t41 file.  Defaults to
                     <output_dir>/<job>_line_<labelA>_<labelB>_N<n>.t41.
        csv_path: Path for the output CSV file.  Defaults to the same stem as
                  the t41 with a .csv extension.
        xyz_path: Path for the output .xyz atom file.  Defaults to
                  <output_dir>/<job>_atoms.xyz.
        dry_run: If True, print the densf input but do not execute densf or
                 write the CSV or XYZ.

    Returns:
        (t41_path, csv_path, xyz_path) — csv_path and xyz_path are None when
        dry_run=True.
    """
    rkf_path = str(Path(rkf_path).resolve())

    if len(atom_pair) != 2:
        raise ValueError("atom_pair must contain exactly two atom references.")

    idx_a, label_a = _parse_atom_ref(atom_pair[0])
    idx_b, label_b = _parse_atom_ref(atom_pair[1])

    # Derive job name and output directory
    job_name, output_dir = _derive_job_name(rkf_path)
    stem = f"{job_name}_line_{label_a}_{label_b}_N{n_points}"

    if output_path is None:
        output_path = str(output_dir / f"{stem}.t41")
    if csv_path is None:
        csv_path = str(output_dir / f"{stem}.csv")
    if xyz_path is None:
        xyz_path = str(output_dir / f"{job_name}_atoms.xyz")

    # Get atom coordinates (Angstrom)
    coords = _get_coords_by_input_index(rkf_path, [idx_a, idx_b])
    coord_a, coord_b = coords[0], coords[1]

    print(f"Atom {label_a} (input index {idx_a}): "
          f"({coord_a[0]:.6f}, {coord_a[1]:.6f}, {coord_a[2]:.6f}) Bohr")
    print(f"Atom {label_b} (input index {idx_b}): "
          f"({coord_b[0]:.6f}, {coord_b[1]:.6f}, {coord_b[2]:.6f}) Bohr")

    dist = sum((b - a) ** 2 for a, b in zip(coord_a, coord_b)) ** 0.5
    print(f"Interatomic distance: {dist:.6f} Bohr  ({n_points} points, "
          f"step {dist / (n_points - 1):.6f} Bohr)")

    # Build densf input
    grid_block = build_inline_grid(coord_a, coord_b, n_points)
    variable_block = "\n".join(variables)

    densf_input = (
        f"ADFFILE {rkf_path}\n"
        f"OUTPUTFILE {output_path}\n"
        f"UNITS\n Length Bohr\nEND\n"
        f"{grid_block}\n"
        f"{variable_block}"
    )

    print("\n--- densf input ---")
    print(densf_input)
    print("-------------------\n")

    if dry_run:
        print("Dry run — densf not executed.")
        return output_path, None, None

    if os.path.exists(output_path):
        print(f"Output t41 already exists, overwriting: {output_path}")

    try:
        subprocess.run(
            ["$AMSBIN/densf"],
            input=densf_input,
            text=True,
            shell=True,
            check=True,
        )
        print(f"densf finished successfully.\nOutput: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"densf failed (exit code {e.returncode}):\n{e.stderr}")
        raise

    # Extract values to CSV
    print("\nExtracting values from TAPE41...")
    extract_line_scan_to_csv(
        t41_path=output_path,
        coord_a=coord_a,
        coord_b=coord_b,
        n_points=n_points,
        variables=variables,
        csv_path=csv_path,
        label_a=label_a,
        label_b=label_b,
    )

    # Write atom XYZ file
    print("\nWriting atom XYZ file...")
    write_atom_xyz(rkf_path=rkf_path, xyz_path=xyz_path)

    return output_path, csv_path, xyz_path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Human-readable axis labels for known column names.
_COLUMN_LABELS: dict[str, str] = {
    "density_scf":  r"$\rho_\mathrm{SCF}$ (a.u.)",
    "density_frag": r"$\rho_\mathrm{frag}$ (a.u.)",
    "density_ortho":r"$\rho_\mathrm{ortho}$ (a.u.)",
    "kindens_scf":  r"$\tau_\mathrm{SCF}$ (a.u.)",
    "kindens_frag": r"$\tau_\mathrm{frag}$ (a.u.)",
    "laplacian_scf":r"$|\nabla^2\rho_\mathrm{SCF}|$ (a.u.)",
    "dengrad_mag":  r"$|\nabla\rho|$ (a.u.)",
}

# Columns to plot as absolute value (always positive, use log scale).
_ABS_COLS: frozenset[str] = frozenset({"laplacian_scf"})

# Columns that can take negative values — use symlog instead of log scale.
# (laplacian is excluded because it is plotted as |laplacian|.)
_SIGNED_COLS: frozenset[str] = frozenset()


def plot_line_scan_csv(csv_path: str,
                       pdf_path: str | None = None,
                       label_a: str = "A",
                       label_b: str = "B") -> str:
    """
    Read a line-scan CSV produced by extract_line_scan_to_csv and save a grid
    of subplots (one per variable column) as a PDF file.

    Y-axes use log scale for positive-definite quantities and symlog for
    quantities that can be negative (e.g. the Laplacian).

    Args:
        csv_path: Path to the CSV file.
        pdf_path: Destination PDF path.  Defaults to the same stem as the CSV
                  with a .pdf extension.
        label_a: Label for the first atom endpoint (x-axis origin label).
        label_b: Label for the second atom endpoint.

    Returns:
        Path to the written PDF file.
    """
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if pdf_path is None:
        pdf_path = str(Path(csv_path).with_suffix(".pdf"))

    # Read CSV
    arclength: list[float] = []
    data: dict[str, list[float]] = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames is not None
        arc_col = "arclength_bohr"
        var_cols = [c for c in reader.fieldnames
                    if c not in ("point", "x_bohr", "y_bohr", "z_bohr", arc_col)]
        for col in var_cols:
            data[col] = []
        for row in reader:
            arclength.append(float(row[arc_col]))
            for col in var_cols:
                data[col].append(float(row[col]))

    n_vars = len(var_cols)
    n_cols = min(n_vars, 2)
    n_rows = math.ceil(n_vars / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 3.5 * n_rows),
                             squeeze=False)
    fig.suptitle(
        f"Line scan: {label_a} → {label_b}",
        fontsize=12, fontweight="bold", y=1.01
    )

    for idx, col in enumerate(var_cols):
        ax = axes[idx // n_cols][idx % n_cols]
        vals = data[col]

        # Take absolute value for designated columns.
        if col in _ABS_COLS:
            vals = [abs(v) for v in vals]

        # Choose scale: symlog for signed columns, log for positive-definite.
        # Also fall back to symlog if the data itself contains negative values.
        is_signed = col in _SIGNED_COLS or any(v < 0 for v in vals)
        if is_signed:
            # Symmetric log: linear region around zero, log elsewhere.
            # linthresh = smallest |nonzero| value in the data.
            nonzero = [abs(v) for v in vals if v != 0]
            linthresh = min(nonzero) if nonzero else 1e-10
            ax.set_yscale("symlog", linthresh=linthresh)
            ax.axhline(0, color="grey", lw=0.5, ls="--")
        else:
            ax.set_yscale("log")

        ax.plot(arclength, vals, lw=1.5, color=f"C{idx}")
        ax.set_xlabel(f"Arclength ({label_a}→{label_b}) / Bohr", fontsize=9)
        ax.set_ylabel(_COLUMN_LABELS.get(col, col), fontsize=9)
        ax.set_title(_COLUMN_LABELS.get(col, col), fontsize=9)
        ax.tick_params(labelsize=8)

    # Hide any unused axes
    for idx in range(n_vars, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.tight_layout()
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    print(f"PDF written: {pdf_path}")
    return pdf_path


# ---------------------------------------------------------------------------
# USER SETTINGS — edit these when running as a script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Path to the adf.rkf file
    RKF_PATH = "/Users/haiiro/scratch/phenol-dimer_B3LYP_TZ2P_GO.results/adf.rkf"

    # Two atom references: 1-based input-order integers or label strings like "H21"
    ATOM_PAIR = ("O7", "H21")

    # Number of grid points along the line (including both endpoints)
    N_POINTS = 1000

    # densf property keywords to compute — one string per property
    VARIABLES = [
        "density scf",
        "density frag",
        "DenGrad",
        "Laplacian",
    ]

    # Output paths — set to None for automatic naming
    OUTPUT_PATH = None  # .t41 file
    CSV_PATH = None     # .csv file

    # Set to True to preview the densf input without running the calculation
    DRY_RUN = False

    _label_a = str(ATOM_PAIR[0])
    _label_b = str(ATOM_PAIR[1])

    _, csv_out, _xyz_out = run_line_scan(
        rkf_path=RKF_PATH,
        atom_pair=ATOM_PAIR,
        n_points=N_POINTS,
        variables=VARIABLES,
        output_path=OUTPUT_PATH,
        csv_path=CSV_PATH,
        dry_run=DRY_RUN,
    )

    if csv_out:
        plot_line_scan_csv(csv_out, label_a=_label_a, label_b=_label_b)
