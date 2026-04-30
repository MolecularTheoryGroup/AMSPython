import math
import os
from pathlib import Path
import subprocess

BOHR_TO_ANGSTROM = 0.529177210903

def _run_amsreport(rkf_path, key):
    """Run amsreport on an rkf file and return the whitespace-split tokens of stdout."""
    cmd = f'$AMSBIN/amsreport \'{rkf_path}\' -r "{key}"'
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True, check=True)
    return result.stdout.split()


def get_custom_grid_string(rkf_path, spacing=0.05, extend=4.2,
                           atom_inds=None, atom_types=None, length_unit="bohr", grid_save=False):
    """
    Build a densf GRID block string for a custom grid derived from atom positions
    in the given rkf file.

    Args:
        rkf_path (str): Path to the .rkf file.
        spacing (float): Grid point spacing in Angstrom. Default 0.05.
        extend (float): Bounding-box extension in Angstrom beyond the atom extents. Default 4.2.
        atom_inds (list[int] | None): 1-based input-order atom indices to include.
            Mutually exclusive with atom_types. None or [] means all atoms.
        atom_types (list[str] | None): Element symbols to include.
            Mutually exclusive with atom_inds. None or [] means all atoms.
        length_unit (str): Unit of atomic coordinates returned by amsreport.
            'bohr' (default) or 'angstrom'; checked by first character.
        grid_save (bool): Whether to include the 'save' keyword in the GRID block. Default False.

    Returns:
        str: Multi-line GRID block suitable for inclusion in densf input.
    """
    atom_inds = atom_inds or []
    atom_types = atom_types or []

    if atom_inds and atom_types:
        raise ValueError("atom_inds and atom_types are mutually exclusive.")

    # --- fetch xyz coordinates (in internal order) ---
    xyz_tokens = _run_amsreport(rkf_path, "Geometry%xyz")
    n_atoms = len(xyz_tokens) // 3
    coords_internal = [
        (float(xyz_tokens[3 * i]),
         float(xyz_tokens[3 * i + 1]),
         float(xyz_tokens[3 * i + 2]))
        for i in range(n_atoms)
    ]

    # Convert from Bohr to Angstrom if needed
    if length_unit[0].lower() == 'b':
        coords_internal = [
            (x * BOHR_TO_ANGSTROM, y * BOHR_TO_ANGSTROM, z * BOHR_TO_ANGSTROM)
            for x, y, z in coords_internal
        ]

    # --- fetch element symbols for each atom (internal order) ---
    atomtype_tokens = _run_amsreport(rkf_path, "Geometry%atomtype")
    unique_elements = list(atomtype_tokens)  # ordered list of unique element symbols

    frag_tokens = _run_amsreport(rkf_path, "Geometry%fragment and atomtype index")
    frag_ints = [int(t) for t in frag_tokens]  # length 2*N
    # second half: 1-based index into unique_elements for each atom in internal order
    atomtype_indices = frag_ints[n_atoms:]
    # internal_to_element: 1-based internal atom index -> element symbol
    internal_to_element = {
        i + 1: unique_elements[atomtype_indices[i] - 1]
        for i in range(n_atoms)
    }

    # --- filter by atom_inds (1-based input order) ---
    if atom_inds:
        order_tokens = _run_amsreport(rkf_path, "Geometry%atom order index")
        order_ints = [int(t) for t in order_tokens]
        # First N values: order_ints[i] = 1-based internal index for input atom (i+1)
        input_to_internal = {i + 1: order_ints[i] for i in range(n_atoms)}
        selected_coords = [
            coords_internal[input_to_internal[inp] - 1]
            for inp in atom_inds
            if inp in input_to_internal
        ]
    # --- filter by atom_types ---
    elif atom_types:
        atom_types_set = set(atom_types)
        selected_coords = [
            coords_internal[i]
            for i in range(n_atoms)
            if internal_to_element[i + 1] in atom_types_set
        ]
    else:
        selected_coords = coords_internal

    if not selected_coords:
        raise ValueError(
            "No atoms selected for custom grid. Check atom_inds / atom_types."
        )

    # --- compute bounding box and grid parameters ---
    xs = [c[0] for c in selected_coords]
    ys = [c[1] for c in selected_coords]
    zs = [c[2] for c in selected_coords]

    origin = (min(xs) - extend, min(ys) - extend, min(zs) - extend)
    lengths = (
        (max(xs) - min(xs)) + 2 * extend,
        (max(ys) - min(ys)) + 2 * extend,
        (max(zs) - min(zs)) + 2 * extend,
    )
    n_points = tuple(math.ceil(l / spacing) + 1 for l in lengths)

    save_str = "save" if grid_save else ""
    return (
        f"GRID {save_str}\n"
        f" {origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}\n"
        f" {n_points[0]} {n_points[1]} {n_points[2]}\n"
        f" 1 0 0 {lengths[0]:.6f}\n"
        f" 0 1 0 {lengths[1]:.6f}\n"
        f" 0 0 1 {lengths[2]:.6f}\n"
        f"END"
    )


def walk_and_rename_rkf_files(root_dir):
    """
    Walk through directory structure and generate new filenames for .rkf files.
    
    Args:
        root_dir (str): The root directory to start walking from
    
    Returns:
        list: List of tuples containing (original_path, new_filename)
    """
    root_path = Path(root_dir)
    results = []
    
    # Walk through all subdirectories
    for current_dir, subdirs, files in os.walk(root_dir):
        current_path = Path(current_dir)
        
        # Check if current directory contains any .rkf files
        rkf_files = [f for f in files if f.endswith('.rkf')]
        
        if rkf_files:
            # Get relative path from root to current directory
            relative_path = current_path.relative_to(root_path)
            
            # Create the new filename by joining path parts with underscores
            if relative_path == Path('.'):  # If we're in the root directory
                path_parts = []
            else:
                path_parts = relative_path.parts
            
            for rkf_file in rkf_files:
                # Create new filename: path_parts joined with underscores + .t41
                if path_parts:
                    new_filename = '_'.join(path_parts) + '.t41'
                else:
                    # If in root directory, use just the original filename with .t41
                    new_filename = Path(rkf_file).stem + '.t41'
                
                original_path = current_path / rkf_file
                # Create absolute path for new file in the same directory as original
                new_absolute_path = current_path / new_filename
                results.append((str(original_path), str(new_absolute_path)))
    
    return sorted(results, key=lambda x: x[0])  # Sort by original path

def execute_densf_commands(file_mappings, grid="fine", spacing=0.05, extend=4.2,
                           atom_inds=None, atom_types=None, length_unit="bohr", grid_save=False):
    """
    Execute densf commands for each file mapping.

    Args:
        file_mappings (list): List of tuples containing (original_path, new_path).
        grid (str): Grid type — 'coarse', 'medium', 'fine' (default), or 'custom'.
        spacing (float): Grid spacing in Angstrom, used when grid='custom'. Default 0.05.
        extend (float): Bounding-box extension in Angstrom, used when grid='custom'. Default 4.2.
        atom_inds (list[int] | None): 1-based input-order atom indices for the custom grid
            bounding box. Mutually exclusive with atom_types. None or [] means all atoms.
        atom_types (list[str] | None): Element symbols for the custom grid bounding box.
            Mutually exclusive with atom_inds. None or [] means all atoms.
        length_unit (str): Coordinate unit in the rkf file — 'bohr' (default) or 'angstrom'.
        grid_save (bool): Whether to include the 'save' keyword in the GRID block for custom grids. Default False.
    """
    for original_path, new_path in file_mappings:
        # Skip if new_path already exists
        if os.path.exists(new_path):
            print(f"Skipping {new_path}, already exists.")
            continue

        # Build the GRID block
        if grid in ("coarse", "medium", "fine"):
            save_str = "save " if grid_save else ""
            grid_block = f"GRID {save_str}{grid}\nEND"
        elif grid == "custom":
            grid_block = get_custom_grid_string(
                original_path, spacing=spacing, extend=extend,
                atom_inds=atom_inds, atom_types=atom_types, length_unit=length_unit, grid_save=grid_save
            )
        else:
            raise ValueError(f"Unknown grid type '{grid}'. Use 'coarse', 'medium', 'fine', or 'custom'.")
        
        # Build the densf input
        densf_input = f"""ADFFILE {original_path}
OUTPUTFILE {new_path}
{grid_block}
density scf
KinDens scf"""
        
        # Create the shell command
        command = ["$AMSBIN/densf"]
        
        print(f"Processing: {original_path}")
        print(f"Output: {new_path}")
        print(f"Using densf input:\n{densf_input}")
        
        try:
            # Execute the command with the input, showing output in real-time
            process = subprocess.run(
                command,
                input=densf_input,
                text=True,
                shell=True,
                check=True
            )
            print(f"Success: {new_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {original_path}: {e}")
        except Exception as e:
            print(f"Unexpected error processing {original_path}: {e}")
        
        print("-" * 50)

# Example usage
if __name__ == "__main__":
    # Define your starting root directory here
    root_directory = "/Users/haiiro/scratch/C4H8Na2.results/"
    
    # Get the results
    file_mappings = walk_and_rename_rkf_files(root_directory)
    
    # Print the results
    for original, new_path in file_mappings:
        print(f"Original: {original}")
        print(f"New path: {new_path}")
        print("-" * 50)
    
    # Execute densf commands
    print("\nExecuting densf commands...")
    execute_densf_commands(file_mappings, grid="custom", spacing=0.05, extend=4.2)

    # --- Custom grid examples (uncomment to use) ---
    # All atoms, custom spacing/extend:
    # execute_densf_commands(file_mappings, grid="custom", spacing=0.05, extend=4.2)
    #
    # Restrict grid to specific atoms by 1-based input-order index:
    # execute_densf_commands(file_mappings, grid="custom", atom_inds=[1, 5, 6])
    #
    # Restrict grid to specific element types:
    # execute_densf_commands(file_mappings, grid="custom", atom_types=["Fe", "S"])