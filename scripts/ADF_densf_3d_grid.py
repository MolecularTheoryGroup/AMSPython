"""
ADF_densf_3d_grid.py

Run densf with an orthogonal 3-D grid over an adf.rkf file, extract the
results into a NumPy array, save everything to a pickle file, and optionally
render ρ isosurfaces using Plotly.

Usage (as a script):
    Edit the USER SETTINGS section below and run:
        python ADF_densf_3d_grid.py

Usage (as a module):
    from ADF_densf_3d_grid import run_3d_grid, plot_isosurfaces

Job naming:
    Follows the same convention as ADF_densf_line_scan.py: if the rkf lives
    inside a *.results/ directory, the job name is the parent stem and output
    files go in the grandparent directory.

Output pickle schema:
    {
        "job_name":  str,
        "rkf_path":  str,
        "spacing":   float,          # Bohr
        "extend":    float,          # Bohr
        "origin":    (ox, oy, oz),   # Bohr, corner of the bounding box
        "shape":     (nx, ny, nz),
        "x":         np.ndarray,     # 1-D, length nx, Bohr
        "y":         np.ndarray,     # 1-D, length ny, Bohr
        "z":         np.ndarray,     # 1-D, length nz, Bohr
        "variables": {
            "density_scf": np.ndarray shape (nx, ny, nz),
            ...
        },
    }
"""

import math
import os
import pickle
import subprocess
from pathlib import Path

import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903

# ---------------------------------------------------------------------------
# Variable → t41 key map  (same logic as line-scan script)
# ---------------------------------------------------------------------------

_VARIABLE_KEY_MAP: list[tuple[str, str, str]] = [
    ("density scf",   "SCF%Density",        "density_scf"),
    ("density frag",  "SumFrag%Density",    "density_frag"),
    ("density ortho", "Ortho%Density",      "density_ortho"),
    ("kindens scf",   "SCF%KinDens",        "kindens_scf"),
    ("kindens frag",  "SumFrag%KinDens",    "kindens_frag"),
    ("laplacian",     "SCF%DensityLap",     "laplacian_scf"),
    ("dengrad",       "SCF%DensityGradX",   "dengrad_mag"),  # special-cased below
]

_DENGRAD_KEYS = ("SCF%DensityGradX", "SCF%DensityGradY", "SCF%DensityGradZ")


def _variable_to_t41_key(variable: str) -> tuple[str, str]:
    v_lower = variable.strip().lower()
    for prefix, t41_key, col_name in _VARIABLE_KEY_MAP:
        if v_lower.startswith(prefix):
            return t41_key, col_name
    raise ValueError(
        f"No TAPE41 key mapping for densf variable {variable!r}. "
        "Add an entry to _VARIABLE_KEY_MAP."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_amsreport(path: str, key: str) -> list[str]:
    cmd = f'$AMSBIN/amsreport \'{path}\' -r "{key}"'
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True, check=True)
    return result.stdout.split()


def _derive_job_name(rkf_path: str) -> tuple[str, Path]:
    rkf_p = Path(rkf_path).resolve()
    parent = rkf_p.parent
    if parent.name.endswith(".results"):
        return parent.name[: -len(".results")], parent.parent
    return rkf_p.stem, parent


def _get_atom_coords_bohr(rkf_path: str,
                           atom_inds: list[int] | None = None) -> list[tuple[float, float, float]]:
    """
    Return Bohr coordinates for a subset (or all) atoms, using 1-based
    input-order indices.  atom_inds=None or [] means all atoms.
    """
    xyz_tokens = _run_amsreport(rkf_path, "Geometry%xyz")
    n_atoms = len(xyz_tokens) // 3
    coords_internal = [
        (float(xyz_tokens[3 * i]),
         float(xyz_tokens[3 * i + 1]),
         float(xyz_tokens[3 * i + 2]))
        for i in range(n_atoms)
    ]

    if not atom_inds:
        return coords_internal

    order_tokens = _run_amsreport(rkf_path, "Geometry%atom order index")
    order_ints = [int(t) for t in order_tokens]
    # First N values: order_ints[i] = 1-based internal index for input atom (i+1)
    input_to_internal = {i + 1: order_ints[i] for i in range(n_atoms)}

    selected = []
    for inp in atom_inds:
        if inp not in input_to_internal:
            raise ValueError(
                f"Atom input index {inp} not found (valid: 1–{n_atoms})."
            )
        selected.append(coords_internal[input_to_internal[inp] - 1])
    return selected


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def build_grid_block(rkf_path: str,
                     spacing: float,
                     extend: float,
                     atom_inds: list[int] | None = None,
                     grid_save: bool = False) -> tuple[str, tuple, tuple, tuple]:
    """
    Build the GRID block string and return grid metadata.

    Args:
        rkf_path:   Path to the adf.rkf file.
        spacing:    Grid point spacing in Bohr.
        extend:     Padding around the bounding box in Bohr.
        atom_inds:  1-based input-order atom indices to use for the bounding
                    box.  None / [] means all atoms.
        grid_save:  Include the 'save' keyword so densf writes xyz point
                    coordinates into the t41.

    Returns:
        (grid_block_str, origin, lengths, n_points)
        origin    = (ox, oy, oz) in Bohr
        lengths   = (lx, ly, lz) in Bohr
        n_points  = (nx, ny, nz) integers
    """
    coords = _get_atom_coords_bohr(rkf_path, atom_inds)

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]

    origin = (min(xs) - extend, min(ys) - extend, min(zs) - extend)
    lengths = (
        (max(xs) - min(xs)) + 2 * extend,
        (max(ys) - min(ys)) + 2 * extend,
        (max(zs) - min(zs)) + 2 * extend,
    )
    n_points = tuple(math.ceil(l / spacing) + 1 for l in lengths)

    save_str = "save " if grid_save else ""
    grid_block = (
        f"GRID {save_str}\n"
        f" {origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}\n"
        f" {n_points[0]} {n_points[1]} {n_points[2]}\n"
        f" 1 0 0 {lengths[0]:.6f}\n"
        f" 0 1 0 {lengths[1]:.6f}\n"
        f" 0 0 1 {lengths[2]:.6f}\n"
        f"END"
    )
    return grid_block, origin, lengths, n_points


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_3d_grid_to_pickle(
        t41_path: str,
        origin: tuple,
        lengths: tuple,
        n_points: tuple,
        variables: list[str],
        pickle_path: str,
        job_name: str = "",
        rkf_path: str = "",
        spacing: float = 0.0,
        extend: float = 0.0,
) -> str:
    """
    Read a densf TAPE41 result for a 3-D grid scan and save to a pickle file.

    The flat arrays from amsreport are stored in Fortran (column-major) order
    by densf: the first index (x) runs fastest.  We reshape accordingly and
    expose the data with shape (nx, ny, nz) so that data[ix, iy, iz]
    corresponds to grid point (x[ix], y[iy], z[iz]).

    Returns the path to the written pickle file.
    """
    nx, ny, nz = n_points
    n_total = nx * ny * nz
    ox, oy, oz = origin
    lx, ly, lz = lengths

    # 1-D coordinate axes
    x_ax = np.linspace(ox, ox + lx, nx)
    y_ax = np.linspace(oy, oy + ly, ny)
    z_ax = np.linspace(oz, oz + lz, nz)

    result: dict = {
        "job_name":  job_name,
        "rkf_path":  rkf_path,
        "spacing":   spacing,
        "extend":    extend,
        "origin":    origin,
        "lengths":   lengths,
        "shape":     n_points,
        "x":         x_ax,
        "y":         y_ax,
        "z":         z_ax,
        "variables": {},
    }

    for var in variables:
        t41_key, col_name = _variable_to_t41_key(var)

        if var.strip().lower().startswith("dengrad"):
            components = []
            for key in _DENGRAD_KEYS:
                tokens = _run_amsreport(t41_path, key)
                if not tokens:
                    print(f"  Warning: no data for '{key}'")
                    components.append(np.full(n_total, float("nan")))
                else:
                    components.append(np.array([float(v) for v in tokens]))
            flat = np.sqrt(components[0] ** 2 + components[1] ** 2 + components[2] ** 2)
            col_name = "dengrad_mag"
        else:
            tokens = _run_amsreport(t41_path, t41_key)
            if not tokens:
                print(f"  Warning: no data for '{t41_key}' ({var!r})")
                flat = np.full(n_total, float("nan"))
            else:
                flat = np.array([float(v) for v in tokens])

        if len(flat) != n_total:
            print(
                f"  Warning: expected {n_total} values for '{col_name}', "
                f"got {len(flat)}"
            )
            flat = np.resize(flat, n_total)

        # densf writes points with x varying fastest → reshape (nx, ny, nz)
        result["variables"][col_name] = flat.reshape((nx, ny, nz), order="F")
        print(f"  Extracted '{col_name}': shape {result['variables'][col_name].shape}, "
              f"min={float(flat.min()):.4g}, max={float(flat.max()):.4g}")

    with open(pickle_path, "wb") as fh:
        pickle.dump(result, fh)
    print(f"Pickle written: {pickle_path}")
    return pickle_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_3d_grid(
        rkf_path: str,
        variables: list[str],
        spacing: float = 0.2,
        extend: float = 3.0,
        atom_inds: list[int] | None = None,
        grid_save: bool = False,
        output_path: str | None = None,
        pickle_path: str | None = None,
        dry_run: bool = False,
) -> tuple[str, str | None]:
    """
    Run densf with an orthogonal 3-D grid, then extract results to a pickle.

    Args:
        rkf_path:    Path to the adf.rkf file.
        variables:   List of densf property keyword strings.
        spacing:     Grid point spacing in Bohr.  Default 0.2.
        extend:      Padding around the atom bounding box in Bohr.  Default 3.0.
        atom_inds:   1-based input-order indices used to define the bounding
                     box.  None / [] means all atoms.
        grid_save:   Include 'save' in the GRID block (stores xyz in t41).
        output_path: Path for the .t41 output.  Auto-named if None.
        pickle_path: Path for the .pkl output.  Auto-named if None.
        dry_run:     Print the densf input without running anything.

    Returns:
        (t41_path, pickle_path)  — pickle_path is None for dry_run.
    """
    rkf_path = str(Path(rkf_path).resolve())
    job_name, output_dir = _derive_job_name(rkf_path)

    atom_tag = ("_atoms" + "-".join(str(i) for i in atom_inds)) if atom_inds else ""
    stem = f"{job_name}_3dgrid_sp{spacing:.3g}_ext{extend:.3g}{atom_tag}"

    if output_path is None:
        output_path = str(output_dir / f"{stem}.t41")
    if pickle_path is None:
        pickle_path = str(output_dir / f"{stem}.pkl")

    grid_block, origin, lengths, n_points = build_grid_block(
        rkf_path, spacing=spacing, extend=extend,
        atom_inds=atom_inds, grid_save=grid_save,
    )

    nx, ny, nz = n_points
    n_total = nx * ny * nz
    print(f"Grid: {nx} × {ny} × {nz} = {n_total:,} points  "
          f"(spacing={spacing} Bohr, extend={extend} Bohr)")
    print(f"Origin: ({origin[0]:.4f}, {origin[1]:.4f}, {origin[2]:.4f}) Bohr")

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
        return output_path, None

    if os.path.exists(output_path):
        print(f"t41 already exists, overwriting: {output_path}")

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
        print(f"densf failed (exit {e.returncode}):\n{e.stderr}")
        raise

    print("\nExtracting values from TAPE41...")
    extract_3d_grid_to_pickle(
        t41_path=output_path,
        origin=origin,
        lengths=lengths,
        n_points=n_points,
        variables=variables,
        pickle_path=pickle_path,
        job_name=job_name,
        rkf_path=rkf_path,
        spacing=spacing,
        extend=extend,
    )

    return output_path, pickle_path


# ---------------------------------------------------------------------------
# 3-D isosurface plot
# ---------------------------------------------------------------------------

def plot_isosurfaces(
        pickle_path: str,
        variable: str = "density_scf",
        isomin: float = 0.001,
        isomax: float = 0.5,
        n_surfaces: int = 5,
        output_path: str | None = None,
        opacity: float = 0.3,
) -> str:
    """
    Load a pickle produced by run_3d_grid and plot isosurfaces of *variable*
    using Plotly, saving the result as an interactive HTML file.

    Args:
        pickle_path: Path to the pickle file.
        variable:    Key in data["variables"] to plot.  Default "density_scf".
        isomin:      Minimum isosurface value.  Default 0.001.
        isomax:      Maximum isosurface value.  Default 0.5.
        n_surfaces:  Number of evenly-spaced isosurface levels.  Default 5.
        output_path: Output file path (.html).  Defaults to pickle stem + .html.
        opacity:     Surface opacity.  Default 0.3.

    Returns:
        Path to the written output file.
    """
    import plotly.graph_objects as go

    with open(pickle_path, "rb") as fh:
        data = pickle.load(fh)

    if variable not in data["variables"]:
        available = list(data["variables"].keys())
        raise ValueError(
            f"Variable '{variable}' not found in pickle. Available: {available}"
        )

    if output_path is None:
        output_path = str(Path(pickle_path).with_suffix(".html"))

    vol = data["variables"][variable]   # shape (nx, ny, nz)
    x_ax = data["x"]
    y_ax = data["y"]
    z_ax = data["z"]

    # Plotly Volume needs flat coordinate meshes
    X, Y, Z = np.meshgrid(x_ax, y_ax, z_ax, indexing="ij")

    isovals = np.linspace(isomin, isomax, n_surfaces)

    fig = go.Figure()

    colorscale = "Viridis"
    for level in isovals:
        fig.add_trace(go.Isosurface(
            x=X.ravel(),
            y=Y.ravel(),
            z=Z.ravel(),
            value=vol.ravel(),
            isomin=level,
            isomax=level,
            surface_count=1,
            opacity=opacity,
            colorscale=colorscale,
            showscale=False,
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))

    job_name = data.get("job_name", Path(pickle_path).stem)
    fig.update_layout(
        title=dict(
            text=f"{job_name} — {variable} isosurfaces<br>"
                 f"levels: {', '.join(f'{v:.3g}' for v in isovals)} a.u.",
            font_size=12,
        ),
        scene=dict(
            xaxis_title="x / Bohr",
            yaxis_title="y / Bohr",
            zaxis_title="z / Bohr",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"HTML written: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# USER SETTINGS — edit these when running as a script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Path to the adf.rkf file
    RKF_PATH = "/Users/haiiro/scratch/phenol-dimer_B3LYP_TZ2P_GO.results/adf.rkf"

    # densf property keywords to compute
    VARIABLES = [
        "density scf",
        # "density frag",
        # "Laplacian",
        # "DenGrad",
    ]

    # Grid spacing in Bohr (0.125$\AA$) = 0.125 / 0.529177210903 ≈ 0.236 Bohr
    SPACING = 0.236

    # Bounding-box padding in Bohr
    EXTEND = 3.0

    # 1-based input-order atom indices to use for the bounding box.
    # Set to None or [] to use all atoms.
    ATOM_INDS = None

    # Include 'save' keyword in GRID block (writes xyz coords to t41)
    GRID_SAVE = False

    # Output paths — None for automatic naming
    OUTPUT_PATH = None   # .t41
    PICKLE_PATH = None   # .pkl

    # Set True to preview densf input without running
    DRY_RUN = False

    _, pkl = run_3d_grid(
        rkf_path=RKF_PATH,
        variables=VARIABLES,
        spacing=SPACING,
        extend=EXTEND,
        atom_inds=ATOM_INDS,
        grid_save=GRID_SAVE,
        output_path=OUTPUT_PATH,
        pickle_path=PICKLE_PATH,
        dry_run=DRY_RUN,
    )

    if pkl:
        plot_isosurfaces(
            pickle_path=pkl,
            variable="density_scf",
            isomin=0.001,
            isomax=0.5,
            n_surfaces=5,
        )

