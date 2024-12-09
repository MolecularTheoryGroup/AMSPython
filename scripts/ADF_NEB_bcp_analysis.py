# from scm.plams import *
import os
import csv
import subprocess
from math import sqrt, atan, floor, ceil
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# This script is intended to be run on the results of a NEB calculation in AMS.
# The script will create a series of single-point calculations for each image in the NEB calculation,
# with the molecule from each image as the input geometry.
# The script will also create three jobs for each image if the NEB calculation includes an electric field,
# with the electric field set to the original field, opposite (reverse) field, and no field.
# Then the script will extract from the output of the single point calculations information for bond
# critical points for specified atom pairs.

# Use KFBrowser in File->Expert Mode to see the contents of the .rkf file(s)

# Documentation for PLAMS:
# https://www.scm.com/doc.2018/plams/components/components.html

# Run with the command:
# plams /path/to/NEB_bcp_analysis.py

########################################################################################
# USER SETTINGS
########################################################################################

# Define the path to the AMS job file
ams_job_paths = ["/Users/haiiro/Dropbox/AMSPythonData/Cys_NEB_B3LYP/ams.rkf"]
# To rerun on a previously processed file, set the restart_dill_path to the path of the dill file in the
# working directory of the previous run. Otherwise, set to None, False, or ''.
restart_dill_paths = [
    "/Users/haiiro/Dropbox/AMSPythonData/workspaces/plams_workdir.022/ams/ams.dill",
    # "/Users/haiiro/Dropbox/AMSPythonData/workspaces/plams_workdir.004/Cys_propane_near_TS/Cys_propane_near_TS.dill",
    # "/Users/haiiro/Dropbox/AMSPythonData/workspaces/plams_workdir_tyr/Tyr_propane_NEB_NF/Tyr_propane_NEB_NF.dill",
    # "/Users/haiiro/Dropbox/AMSPythonData/workspaces/plams_workdir.003/His_propane_near_TS/His_propane_near_TS.dill",
]

# Define atom pairs (pairs of atom numbers) for which to extract bond critical point information.
# One list for each input file defined above
atom_pairs_list = (  # one-based indices, same as shown in AMSView
    (
        (51, 47),  # CH
        (47, 39),  # OH
        (1, 39),  # FeO
        (1, 40),  # Fe-amino acid Cys
        (1, 2),  # FeN
        (1, 3),  # FeN
        (1, 4),  # FeN
        (1, 5),  # FeN
    ),
)

##### densf grid settings #####

densf_bb_atom_numbers = [47, 40, 39, 1]
densf_bb_padding = 5.0  # Angstroms
densf_bb_spacing = (
    0.05  # Angstroms (densf "fine" is 0.05, "medium" is 0.1, "coarse" is 0.2)
)

##### end densf full grid settings #####

# Index of atom pair for which to print the bond distance of each image to be run.
atom_pair_for_bond_distance_printout = 1

# If the input NEB file includes an applied electric field, then that field will determine the
# magnitude and direction of the electric fields applied to the images in the NEB calculation,
# which will include the no-field and opposite-field cases, in addition to the original field.
# However, you may want to specify your own set of electric fields.
# You may define a single electric field as XYZ components in atomic units (Ha/(e bohr)),
# and the opposite and no-field case will be included.
# THIS OVERRIDES THE EEF USED IN THE NEB CALCULATION.
# Uncomment the following line to specify your own electric field, or leave it as None to use the NEB eef if present.
# user_eef = (0.0, 0.0, 0.01)
user_eef = None  # (0.0, 0.0, 0.01)

# To get better resultion around the transition state, we'll identify the TS image (highest energy image)
# and create additional images between it and the adjacent images, using a linear interpolation of the
# coordinates of the adjacent images. Here, you specify how many extra images to add on *each* side of the TS image.
# Set to 0 to disable this feature.
num_extra_images = 0

# Now define the x and y properties for generated plots:

# in addition to any of the properties that appear as column headings in the output CSV file,
# you may specify the following properties to plot as x or y axis values:
#
# "Reaction coordinate"         (i.e. the image number) which will likely be changed by the nubmer of extra images added)
# "<atom1>-<atom2> distance"    for each atom pair defined above, including its symbol, e.g. "C40-H47 distance" for the CH bond
# "Molecular bond energy"       the total energy of the molecule

plot_y_prop_list = [
    "Rho",
    "Theta",
    "Phi",
    "Molecular bond energy",
]

# This specifies properties to include in combined plots with rows for each y property and columns for each EEF type.
# Additionally, the suffix " d/dx" causes the dy/dx derivative to be computed and plotted.
combined_plots_y_prop_lists = {
    "Molecular bond energy": ["Molecular bond energy"],
    "Rho": ["Molecular bond energy", "Rho"],
    "Rho d/dx": ["Molecular bond energy", "Rho d/dx"],
    "Angles": ["Molecular bond energy", "Theta", "Phi"],
    "Angles d/dx": ["Molecular bond energy", "Theta d/dx", "Phi d/dx"],
}

plot_x_prop_list = [
    "H47-O39 distance",
    "Reaction coordinate",
]

# BCP locations in A/B spin densities are not the same as in the total density, so we do a basic
# search for the minimum gradient magnitude point in a 3D grid around the total density CP location
# for each BCP in each A/B spin density.
# These parameters control that process.
# Number of check points in each dimension. Increasing this value will result in a better approximation of
# The spin A/B CP locations, but will increase densf runtime.
num_check_points = 9
# Fraction of the distance between the two atoms. Increasing this value will result in a search being
# done over a larger region around the total-density CP location, and will increase the spacing between
# the check points. If too small, the search grid may not include the true spin A/B CP locations.
check_point_grid_extent_fraction = 0.02

# Need to convert electric field magnitude units. In the NEB rkf file, they're Ha/(e bohr), but in the
# new jobs they need to be V/Angstrom. The conversion factor is 51.4220861908324.
eef_conversion_factor = 51.4220861908324

# Define the EEF pairs
# eef_pairs = (("origEEF", eef_conversion_factor), ("revEEF", -eef_conversion_factor), ("noEEF", 0))
eef_pairs = (("origEEF", eef_conversion_factor), ("revEEF", -eef_conversion_factor))

########################################################################################
# END OF USER SETTINGS
########################################################################################

num_check_points_total = num_check_points**3

# Get the number of CPU cores
num_cores = ceil(os.cpu_count() / 2)

########################################################################################
# Step 0: define helper functions
########################################################################################


def compute_derivative(x, y, order=1):
    """
    Calculate higher-order derivatives using repeated application of np.diff().

    Parameters:
    x (array): x-coordinates
    y (array): y-coordinates
    order (int): The order of the derivative to calculate (default is 1)

    Returns:
    tuple: (x_values, derivative_values)
    """
    if order < 1:
        raise ValueError("Order must be at least 1")

    x_values = np.array(x)
    y_values = np.array(y)

    for _ in range(order):
        # Calculate the differences
        dy = np.diff(y_values)
        dx = np.diff(x_values)

        # Calculate the derivative
        derivative = dy / dx

        # Update x_values and y_values for the next iteration
        x_values = (x_values[1:] + x_values[:-1]) / 2
        y_values = derivative

    return x_values, y_values


def log_print(*args, **kwargs):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(timestamp, *args, **kwargs)


def parse_cp_info(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    cp_blocks = content.split(
        " --------------------------------------------------------"
    )[1:]
    cp_data = []

    for block in cp_blocks:
        if not "CP #" in block:
            continue
        lines = [s.strip() for s in block.strip().split("\n")]
        cp_info = {}

        i = 0
        while i < len(lines):
            line = lines[i]
            if not line:
                i += 1
                continue
            if line.startswith("CP #"):
                cp_info["CP #"] = int(line.split("#")[1].strip())
            elif line.startswith("(RANK,SIGNATURE):"):
                rank_signature = line.split(":")[1].strip().split(",")
                cp_info["RANK"] = int(rank_signature[0].strip()[1:])
                cp_info["SIGNATURE"] = int(rank_signature[1].strip()[:-1])
            elif line.startswith("CP COORDINATES:"):
                cp_info["CP COORDINATES"] = [
                    float(coord) for coord in line.split(":")[1].strip().split()
                ]
            elif line.startswith("EIGENVALUES OF HESSIAN MATRIX:"):
                eigenvalues = []
                i += 2
                while i < len(lines) and lines[i]:
                    eigenvalues.extend([float(val) for val in lines[i].split()])
                    i += 1
                cp_info["EIGENVALUES OF HESSIAN MATRIX"] = eigenvalues
                continue
            elif line.startswith(
                "EIGENVECTORS (ORTHONORMAL) OF HESSIAN MATRIX (COLUMNS):"
            ):
                eigenvectors = []
                i += 2
                while i < len(lines) and lines[i]:
                    eigenvectors.append([float(val) for val in lines[i].split()])
                    i += 1
                # transpose eigenvectors
                eigenvectors = [
                    [eigenvectors[j][i] for j in range(len(eigenvectors))]
                    for i in range(len(eigenvectors[0]))
                ]
                cp_info["EIGENVECTORS (ORTHONORMAL) OF HESSIAN MATRIX (COLUMNS)"] = (
                    eigenvectors
                )
                continue
            elif line.startswith("HESSIAN MATRIX:"):
                hessian_matrix = []
                i += 2
                while i < len(lines) and lines[i]:
                    hessian_matrix.append([float(val) for val in lines[i].split()])
                    i += 1
                # convert from upper-triangular to full matrix
                for j in range(len(hessian_matrix)):
                    for k in range(j + 1, len(hessian_matrix)):
                        hessian_matrix[k].insert(j, hessian_matrix[j][k])
                cp_info["HESSIAN MATRIX"] = hessian_matrix
                continue
            elif line.startswith("VALUES OF SOME FUNCTIONS AT CPs (a.u.):"):
                i += 2
                while i < len(lines) and lines[i]:
                    key, value = lines[i].split("=")
                    cp_info[key.strip()] = float(value.split()[0])
                    i += 1
                # Compute directionality of bond/ring CPs using the eigenvalues (ev).
                # For bond CPs (3,-1), Theta = arctan(sqrt(ev[0]/ev[2])) and Phi = arctan(sqrt(ev[1]/ev[2])).
                # For ring CPs (3,1), Theta = arctan(sqrt(ev[2]/ev[0])) and Phi = arctan(sqrt(ev[1]/ev[0])).
                if abs(cp_info["SIGNATURE"]) == 1:
                    ev = cp_info["EIGENVALUES OF HESSIAN MATRIX"]
                    if cp_info["SIGNATURE"] == -1:
                        cp_info["Theta"] = atan(sqrt(abs(ev[0] / ev[2])))
                        cp_info["Phi"] = atan(sqrt(abs(ev[1] / ev[2])))
                    else:
                        cp_info["Theta"] = atan(sqrt(abs(ev[2] / ev[0])))
                        cp_info["Phi"] = atan(sqrt(abs(ev[1] / ev[0])))

            i += 1

        cp_data.append(cp_info)

    return cp_data


def get_bcp_properties(job, atom_pairs, unrestricted=False):
    # first get kf file from finished job
    kf = KFFile(job.results["adf.rkf"])
    cp_type_codes = {"nuclear": 1, "bond": 3, "ring": 4, "cage": 2}
    num_cps = kf[("Properties", "CP number of")]
    cp_coords = kf[("Properties", "CP coordinates")]
    cp_codes = kf[("Properties", "CP code number for (Rank,Signatu")]
    # cp_coords is all the x, then all the y, then all the z. So we need to reshape it to a 2D list
    cp_coords = [
        (cp_coords[i], cp_coords[i + num_cps], cp_coords[i + 2 * num_cps])
        for i in range(num_cps)
    ]
    # Now loop over each atom pair, find the bond cp closest to the atom pair midpoint, and save its
    # index to a list.
    cp_indices = []
    bcp_coords = []
    bcp_atom_indices = []
    bcp_check_points = []
    out_mol = job.results.get_main_molecule()

    # get image number from job name, which is present as, for example, 'im001' in the job name
    image_number = int(job.name.split("im")[1])

    for pair in atom_pairs:
        a1 = [getattr(out_mol.atoms[pair[0] - 1], d) for d in ["x", "y", "z"]]
        a2 = [getattr(out_mol.atoms[pair[1] - 1], d) for d in ["x", "y", "z"]]
        midpoint = [(a1[i] + a2[i]) / 2 for i in range(3)]
        midpoint = [
            Units.convert(v, "angstrom", "bohr") for v in midpoint
        ]  # convert to bohr for comparison to cp_coords
        min_dist = 1e6
        min_index = -1
        for i, cp in enumerate(cp_coords):
            if cp_codes[i] != cp_type_codes["bond"]:
                continue
            dist = sum((midpoint[j] - cp[j]) ** 2 for j in range(3)) ** 0.5
            if dist < min_dist:
                min_dist = dist
                min_index = i
        cp_indices.append(min_index + 1)
        bcp_coords.append(cp_coords[min_index])
        bcp_atom_indices.append(
            "-".join([f"{out_mol.atoms[pair[i]-1].symbol}{pair[i]}" for i in range(2)])
        )
        if unrestricted:
            # the CP locations in the unrestricted spin-a and spin-b densities are not the
            # same as in the total density (those are the ones we have now). So we need to
            # include a bunch of points around the known total-density CP location at which
            # to compute the spin-a and spin-b densities, and then we'll take the point with
            # lowest spin-a/b density gradient magnitude as the CP location in the spin-a/b.
            # The check points will be on a regular 3d grid around the total-density CP location,
            # with point spacing based on the distance between the two atoms using the defined
            # fraction of that distance.
            bond_length = out_mol.atoms[pair[0] - 1].distance_to(
                out_mol.atoms[pair[1] - 1]
            )
            check_point_spacing = (
                bond_length * check_point_grid_extent_fraction / num_check_points
            )
            origin = [
                bcp_coords[-1][i] - check_point_spacing * (num_check_points - 1) / 2
                for i in range(3)
            ]
            check_points = []
            for i in range(num_check_points):
                for j in range(num_check_points):
                    for k in range(num_check_points):
                        points = [
                            origin[0] + i * check_point_spacing,
                            origin[1] + j * check_point_spacing,
                            origin[2] + k * check_point_spacing,
                        ]
                        points_str = " ".join([f"{p:.6f}" for p in points])
                        check_points.append(points_str)
            bcp_check_points.append("\n".join(check_points))

    # Now extract the properties of the bond critical points from the output file
    output_file = [
        f for f in job.results.files if f.endswith(".out") and "CreateAtoms" not in f
    ][0]
    # log_print(f"Extracting CP data from {output_file}")
    out_cp_data = parse_cp_info(job.results[output_file])
    # only keep the bond critical points we want
    out_cp_data = [cp for cp in out_cp_data if cp["CP #"] in cp_indices]

    # Loop over atom pairs, adding thier distances to the out_cp_data with "<bond name> distance" as the key.
    # Each cp will need to store the distances for each atom pair.
    for i, pair in enumerate(atom_pairs):
        bond_length = out_mol.atoms[pair[0] - 1].distance_to(out_mol.atoms[pair[1] - 1])
        bond_length_str = f"{out_mol.atoms[pair[0]-1].symbol}{pair[0]}-{out_mol.atoms[pair[1]-1].symbol}{pair[1]} distance"
        for cp in out_cp_data:
            cp[bond_length_str] = bond_length

    # Add the job name as a key to each element in out_cp_data
    job_energy = job.results.get_energy(engine="adf")
    for i in range(len(out_cp_data)):
        out_cp_data[i]["JOB_NAME"] = job.name
        out_cp_data[i]["Reaction coordinate"] = image_number
        out_cp_data[i]["Molecular bond energy"] = job_energy

    # match cp_indices to the right element in out_cp_data using the [CP #] key
    out_cp_data_cp_inds = {}
    for i in range(len(cp_indices)):
        for j, cp in enumerate(out_cp_data):
            if cp["CP #"] == cp_indices[i]:
                out_cp_data_cp_inds[i] = j
                break
        out_cp_data[out_cp_data_cp_inds[i]]["ATOMS"] = bcp_atom_indices[i]
        # log_print(f"Found CP {cp_indices[i]} at index {out_cp_data_cp_inds[i]} with atoms {bcp_atom_indices[i]}")

    # We've only gotten cp data for the bond critical points we want, so we can use those coordinates
    # to create and run a densf job to get the A and B density values and geometry at the points.
    # But only if it's an unrestricted calculation.
    if unrestricted:
        # We'll create a densf input file that will compute the density at all the collected
        # bond critical points.
        # https://www.scm.com/doc/ADF/Input/Densf.html
        in_file = job.results["adf.rkf"]
        out_file = job.results["adf.rkf"].replace(".rkf", ".t41")
        densf_run_file = job.results["adf.rkf"].replace(".rkf", "_densf.run")
        # if out_file exists, check if the nubmer of points is correct. If so,
        # skip the densf run. If not, delete it and make a new one.
        densf_kf = None
        if os.path.exists(out_file):
            densf_kf = KFFile(out_file)
            vals = densf_kf[("x values", f"x values")]
            if len(vals) == num_check_points_total * len(cp_indices):
                log_print(f"Skipping densf run for {job.name} CPs")
            else:
                os.remove(out_file)
                densf_kf = None
        if densf_kf is None:
            # grid_coords = '\n'.join([f'{cp[0]} {cp[1]} {cp[2]}' for cp in bcp_coords])
            grid_coords = "\n".join(bcp_check_points)
            grid_block = f"Grid Inline\n{grid_coords}\nEnd\n"
            with open(densf_run_file, "w") as file:
                densf_content = f"""ADFFILE {in_file}
OUTPUTFILE {out_file}
{grid_block}
UNITS
    Length bohr
END
Density scf
DenGrad
DenHess"""
                file.write(densf_content)

            # Step 5: Construct the command to run densf
            densf_command = f"$AMSBIN/densf < {densf_run_file}"

            # Step 6: Run the densf job
            log_print(
                f"Running densf for {job.name} CPs with run file {densf_run_file}"
            )
            densf_out = subprocess.run(
                densf_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # log_print(densf_out.stdout.decode())
            if densf_out.returncode != 0:
                log_print(f"Error running densf: {densf_out.stderr}")
            # remove densf run file
            os.remove(densf_run_file)
            densf_kf = KFFile(out_file)
        for i, cpi in enumerate(cp_indices):

            def get_saddle_t41_properties(cp_data, cp_ind, out_cp_ind, field):
                # First need to find the minumum gradient magnitude in each block of `num_check_points_total` points
                # in the densf output file.
                total_rho_cp_ind = floor(num_check_points_total / 2)
                grad_x = densf_kf[("SCF", f"DensityGradX_{field}")][
                    cp_ind
                    * num_check_points_total : (cp_ind + 1)
                    * num_check_points_total
                ]
                grad_y = densf_kf[("SCF", f"DensityGradY_{field}")][
                    cp_ind
                    * num_check_points_total : (cp_ind + 1)
                    * num_check_points_total
                ]
                grad_z = densf_kf[("SCF", f"DensityGradZ_{field}")][
                    cp_ind
                    * num_check_points_total : (cp_ind + 1)
                    * num_check_points_total
                ]
                grad_mags = [
                    sqrt(grad_x[i] ** 2 + grad_y[i] ** 2 + grad_z[i] ** 2)
                    for i in range(num_check_points_total)
                ]
                min_grad_ind = grad_mags.index(min(grad_mags))
                min_grad_ind += cp_ind * num_check_points_total
                total_rho_cp_ind += cp_ind * num_check_points_total

                coords = [
                    densf_kf[(f"{ax} values", f"{ax} values")][min_grad_ind]
                    for ax in ["x", "y", "z"]
                ]

                cp_data[out_cp_ind][f"CP COORDINATES_{field}"] = coords
                cp_data[out_cp_ind][f"Rho_{field}"] = densf_kf[
                    ("SCF", f"Density_{field}")
                ][min_grad_ind]
                grad = [
                    densf_kf[("SCF", f"DensityGrad{ax}_{field}")][min_grad_ind]
                    for ax in ["X", "Y", "Z"]
                ]
                grad_mag = sum(g**2 for g in grad) ** 0.5
                cp_data[out_cp_ind][f"|GRAD(Rho)|_{field}"] = grad_mag
                for ax in ["x", "y", "z"]:
                    cp_data[out_cp_ind][f"GRAD(Rho){ax}_{field}"] = grad[
                        "xyz".index(ax)
                    ]
                hess = [
                    densf_kf[("SCF", f"DensityHess{ax}_{field}")][min_grad_ind]
                    for ax in ["XX", "XY", "XZ", "YY", "YZ", "ZZ"]
                ]
                hess = [
                    [hess[0], hess[1], hess[2]],
                    [hess[1], hess[3], hess[4]],
                    [hess[2], hess[4], hess[5]],
                ]
                hess = np.array(hess)
                ev, evec = np.linalg.eigh(hess)
                if cp_codes[cpi - 1] == cp_type_codes["bond"]:
                    cp_data[out_cp_ind][f"Theta_{field}"] = atan(
                        sqrt(abs(ev[0] / ev[2]))
                    )
                    cp_data[out_cp_ind][f"Phi_{field}"] = atan(sqrt(abs(ev[1] / ev[2])))
                elif cp_codes[cpi - 1] == cp_type_codes["ring"]:
                    cp_data[out_cp_ind][f"Theta_{field}"] = atan(
                        sqrt(abs(ev[2] / ev[0]))
                    )
                    cp_data[out_cp_ind][f"Phi_{field}"] = atan(sqrt(abs(ev[1] / ev[0])))
                cp_data[out_cp_ind][f"HESSIAN MATRIX_{field}"] = hess.tolist()
                cp_data[out_cp_ind][
                    f"EIGENVECTORS (ORTHONORMAL) OF HESSIAN MATRIX (COLUMNS)_{field}"
                ] = evec.T.tolist()
                cp_data[out_cp_ind][f"EIGENVALUES_{field}"] = ev.tolist()

            for field in ["A", "B"]:
                get_saddle_t41_properties(out_cp_data, i, out_cp_data_cp_inds[i], field)

    return out_cp_data


def generate_full_t41(job, output_dir):
    # Create bounding box for densf grid.
    # Start by getting min/max coordinates of the specified atoms.
    out_mol = job.results.get_main_molecule()
    min_xyz = [
        min([getattr(out_mol.atoms[i - 1], d) for i in densf_bb_atom_numbers])
        for d in ["x", "y", "z"]
    ]
    max_xyz = [
        max([getattr(out_mol.atoms[i - 1], d) for i in densf_bb_atom_numbers])
        for d in ["x", "y", "z"]
    ]
    # Add padding to the bounding box
    min_xyz = [min_xyz[i] - densf_bb_padding for i in range(3)]
    max_xyz = [max_xyz[i] + densf_bb_padding for i in range(3)]
    # compute number of points in each dimension based on the spacing
    num_points = [int((max_xyz[i] - min_xyz[i]) / densf_bb_spacing) for i in range(3)]
    total_num_points = num_points[0] * num_points[1] * num_points[2]
    outfile = os.path.join(output_dir, f"{job.name}_densf_full.t41")

    # If outfile is present and it contains the correct number of points, skip the densf run
    densf_kf = None
    if os.path.exists(outfile):
        densf_kf = KFFile(outfile)
        vals = densf_kf[("x values", f"x values")]
        if len(vals) == total_num_points:
            log_print(f"Skipping densf run for {job.name} full density")
        else:
            log_print(
                f"Deleting {outfile} and rerunning densf for {job.name} full density"
            )
            os.remove(outfile)
            densf_kf = None

    if densf_kf is None:
        # asjust spacing to make sure we start and end at the min and max coordinates
        spacing = (max_xyz[0] - min_xyz[0]) / num_points[0]
        # create grid specification for padded, bounding box
        # Grid save
        #     x0 y0 z0
        #     n1 [n2 [n3]]
        #     v1x v1y v1z length1
        #     [v2x v2y v2z length2]
        #     [v3x v3y v3z length3]
        # END
        grid_str = f"""Grid Save
    {min_xyz[0]:.6f} {min_xyz[1]:.6f} {min_xyz[2]:.6f}
    {num_points[0]} {num_points[1]} {num_points[2]}
    {densf_bb_spacing:.6f} 0.0 0.0 {spacing * num_points[0]:.6f}
    0.0 {densf_bb_spacing:.6f} 0.0 {spacing * num_points[1]:.6f}
    0.0 0.0 {densf_bb_spacing:.6f} {spacing * num_points[2]:.6f}
END"""
        densf_str = f"""ADFFILE {job.results['adf.rkf']}
OUTPUTFILE {outfile}
{grid_str}
UNITS
    Length angstrom
END
Density scf
KinDens scf"""
        densf_run_file = os.path.join(output_dir, f"{job.name}_densf_full.run")
        densf_log_file = os.path.join(output_dir, f"{job.name}_densf.log")
        with open(densf_run_file, "w") as file:
            file.write(densf_str)
        densf_command = f"$AMSBIN/densf < {densf_run_file}"
        log_print(f"Running densf for {job.name} full density")
        with open(densf_log_file, "a") as log_file:
            densf_out = subprocess.run(
                densf_command,
                shell=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        if densf_out.returncode != 0:
            log_print(f"Error running densf: {densf_out.stderr}")
        else:
            log_print(f"Finished densf for {job.name} full density")
            os.remove(densf_run_file)


def generate_plots(cp_data, prop_list, x_prop_list, out_dir, combined_y_prop_lists):
    # Ensure the output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    unique_bcp_atoms = sorted(list(set([cp["ATOMS"] for cp in cp_data])))
    image_names = sorted(list(set([cp["JOB_NAME"] for cp in cp_data])))
    job_name = os.path.commonprefix(image_names)
    has_eef = any([eef in str(image_names) for eef in ["origEEF", "revEEF", "noEEF"]])

    all_props = []
    for prop in prop_list + x_prop_list:
        for cp_prop in cp_data[0].keys():
            if cp_prop in [prop, f"{prop}_A", f"{prop}_B"]:
                all_props.append(cp_prop)

    bcp_prop_dict = {}
    eef_types = ([f"_{eef[0]}" for eef in eef_pairs] + [""]) if has_eef else [""]
    for bcp in unique_bcp_atoms:
        bcp_prop_dict[bcp] = {}
        bcp_data = sorted(
            [cp for cp in cp_data if cp["ATOMS"] == bcp], key=lambda x: x["JOB_NAME"]
        )
        for eef in eef_types:
            bcp_eef_data = [cp for cp in bcp_data if eef in cp["JOB_NAME"]]
            for prop in all_props:
                bcp_prop_dict[bcp][f"{prop}{eef}"] = []
                for cp in bcp_eef_data:
                    bcp_prop_dict[bcp][f"{prop}{eef}"].append(cp[prop])

    all_props = [
        p for p in bcp_prop_dict[unique_bcp_atoms[0]].keys() if p not in x_prop_list
    ]

    # plots for combinations of x_prop_list vs prop_list
    eef_types = [f"_{eef[0]}" for eef in eef_pairs] if has_eef else [""]

    num_eef = len(eef_types)

    line_styles = {
        "_origEEF": "-",  # solid
        "_revEEF": "--",  # dashed
        "_noEEF": ":",  # dotted
        "": "-",  # solid (for non-EEF plots)
    }

    for x_prop in x_prop_list:
        for plot_name, y_prop_list in combined_plots_y_prop_lists.items():
            log_print(
                f"Plotting combined plots for {plot_name} vs {x_prop} for bond CPs"
            )
            # Expand y_prop_list to include A, B, and total variants
            expanded_y_prop_list = []
            for y_prop in y_prop_list:
                if " d/dx" in y_prop:
                    base_prop = y_prop.replace(" d/dx", "")
                    if f"{base_prop}_A" in all_props:
                        expanded_y_prop_list.extend(
                            [
                                f"{base_prop}_A d/dx",
                                f"{base_prop}_B d/dx",
                                f"{base_prop} d/dx",
                            ]
                        )
                    else:
                        expanded_y_prop_list.append(y_prop)
                elif f"{y_prop}_A" in all_props:
                    expanded_y_prop_list.extend([f"{y_prop}_A", f"{y_prop}_B", y_prop])
                else:
                    expanded_y_prop_list.append(y_prop)

            # Create a large figure for the combined plot
            fig, axs = plt.subplots(
                len(expanded_y_prop_list),
                num_eef + 1,
                figsize=(20, (num_eef + 2) * len(expanded_y_prop_list)),
            )
            fig.suptitle(
                f"{job_name}: Combined plots for {x_prop} ({plot_name})",
                fontsize=16,
                y=1.02,
            )

            for i, y_prop in enumerate(expanded_y_prop_list):
                if x_prop in y_prop or y_prop in x_prop:
                    continue

                is_derivative = " d/dx" in y_prop
                base_prop = y_prop.replace(" d/dx", "") if is_derivative else y_prop

                for j, eef in enumerate(eef_types):
                    ax = axs[i, j] if len(expanded_y_prop_list) > 1 else axs[j]
                    ax.set_title(f"{y_prop}{eef} vs {x_prop}", fontsize=9)
                    ax.set_xlabel(x_prop)
                    ax.set_ylabel(y_prop)

                    for bcp, bcp_props in bcp_prop_dict.items():
                        x_values = bcp_props.get(f"{x_prop}", None)
                        y_values = bcp_props.get(f"{base_prop}{eef}", None)
                        if x_values and y_values:
                            if len(x_values) != len(y_values):
                                if len(x_values) == num_eef * len(y_values):
                                    x_values = x_values[: len(y_values)]
                                else:
                                    print(
                                        f"Warning: Unexpected length mismatch for {bcp}. Skipping this plot."
                                    )
                                    continue
                            x_values, y_values = zip(*sorted(zip(x_values, y_values)))

                            if is_derivative:
                                x_vals, y_vals = compute_derivative(x_values, y_values)
                                ax.plot(
                                    x_vals[1:-1],
                                    y_vals[1:-1],
                                    "-o",
                                    label=bcp,
                                    markersize=2,
                                )
                            else:
                                ax.plot(
                                    x_values, y_values, "-o", label=bcp, markersize=2
                                )

                    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=7)
                    ax.grid(True)

                # Add the "All EEF" plot in the fourth column
                if has_eef:
                    ax = (
                        axs[i, num_eef]
                        if len(expanded_y_prop_list) > 1
                        else axs[num_eef]
                    )
                    ax.set_title(f"{y_prop} vs {x_prop} (All EEF)", fontsize=9)
                    ax.set_xlabel(x_prop)
                    ax.set_ylabel(y_prop)

                    for bcp, bcp_props in bcp_prop_dict.items():
                        for eef in ["_origEEF", "_noEEF", "_revEEF"]:
                            x_values = bcp_props.get(f"{x_prop}{eef}", None)
                            y_values = bcp_props.get(f"{base_prop}{eef}", None)
                            if x_values and y_values:
                                if len(x_values) != len(y_values):
                                    if len(x_values) == num_eef * len(y_values):
                                        x_values = x_values[: len(y_values)]
                                    else:
                                        print(
                                            f"Warning: Unexpected length mismatch for {bcp}. Skipping this plot."
                                        )
                                        continue
                                x_values, y_values = zip(
                                    *sorted(zip(x_values, y_values))
                                )

                                if is_derivative:
                                    x_vals, y_vals = compute_derivative(
                                        x_values, y_values
                                    )
                                    ax.plot(
                                        x_vals[1:-1],
                                        y_vals[1:-1],
                                        f"{line_styles[eef]}o",
                                        label=f"{bcp}{eef}",
                                        markersize=0,
                                    )
                                else:
                                    ax.plot(
                                        x_values,
                                        y_values,
                                        f"{line_styles[eef]}o",
                                        label=f"{bcp}{eef}",
                                        markersize=0,
                                    )

                    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=7)
                    ax.grid(True)

            plt.tight_layout()
            # Determine top padding based on number of rows; the more rows, the less padding
            top = min(0.995, 0.9 + 0.02 * len(expanded_y_prop_list))
            plt.subplots_adjust(top=top)  # Add more padding at the top
            f_base = f"{job_name}combined_{x_prop}_{plot_name}.png".replace("/", "-")
            plt.savefig(os.path.join(out_dir, f_base), dpi=300, bbox_inches="tight")
            plt.close()

    return

    # Generate individual plots (as in the original function)
    for y_prop in all_props:
        for x_prop in x_prop_list:
            if x_prop in y_prop or y_prop in x_prop:
                continue
            for eef in eef_types:
                log_print(f"Plotting {y_prop}{eef} vs {x_prop} for bond CPs")
                fig, ax = plt.subplots()
                ax.set_title(
                    f"{job_name}: {y_prop}{eef} vs {x_prop} for bond CPs", fontsize=9
                )
                ax.set_xlabel(x_prop)
                ax.set_ylabel(y_prop)
                for bcp, bcp_props in bcp_prop_dict.items():
                    x_values = bcp_props.get(f"{x_prop}", None)
                    y_values = bcp_props.get(f"{y_prop}{eef}", None)
                    if x_values and y_values:
                        if len(x_values) != len(y_values):
                            if len(x_values) == num_eef * len(y_values):
                                x_values = x_values[: len(y_values)]
                            else:
                                print(
                                    f"Warning: Unexpected length mismatch for {bcp}. Skipping this plot."
                                )
                                continue
                        x_values, y_values = zip(*sorted(zip(x_values, y_values)))
                        ax.plot(x_values, y_values, "-o", label=bcp, markersize=2)
                    else:
                        ax = None
                        plt.close()
                        break
                if ax:
                    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=7)
                    plt.subplots_adjust(right=0.75)
                    ax.grid(True)
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            out_dir, f"{job_name}{y_prop}{eef}_vs_{x_prop}.png"
                        )
                    )
                    plt.close()

            # Combined EEF plot (as in the original function)
            if has_eef:
                log_print(f"Plotting {y_prop} vs {x_prop} for bond CPs (All EEF)")
                fig, ax = plt.subplots()
                ax.set_title(
                    f"{job_name}: {y_prop} vs {x_prop} for bond CPs (All EEF)",
                    fontsize=9,
                )
                ax.set_xlabel(x_prop)
                ax.set_ylabel(y_prop)
                for bcp, bcp_props in bcp_prop_dict.items():
                    for eef in ["_origEEF", "_revEEF", "_noEEF"]:
                        x_values = bcp_props.get(f"{x_prop}{eef}", None)
                        y_values = bcp_props.get(f"{y_prop}{eef}", None)
                        if x_values and y_values:
                            if len(x_values) != len(y_values):
                                if len(x_values) == num_eef * len(y_values):
                                    x_values = x_values[: len(y_values)]
                                else:
                                    print(
                                        f"Warning: Unexpected length mismatch for {bcp}. Skipping this plot."
                                    )
                                    continue
                            x_values, y_values = zip(*sorted(zip(x_values, y_values)))
                            ax.plot(
                                x_values,
                                y_values,
                                f"{line_styles[eef]}o",
                                label=f"{bcp}{eef}",
                                markersize=0,
                            )
                        else:
                            ax = None
                            plt.close()
                            break
                if ax:
                    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=7)
                    plt.subplots_adjust(right=0.75)
                    ax.grid(True)
                    plt.savefig(
                        os.path.join(
                            out_dir, f"{job_name}{y_prop}_vs_{x_prop}_all_eef.png"
                        )
                    )
                    plt.close()


def write_csv(cp_data, input_file_path):

    # Determine all keys present in the data
    all_keys = set()
    for cp in cp_data:
        for key in cp.keys():
            if isinstance(cp[key], list):
                if isinstance(cp[key][0], list):  # 2D list
                    if "EIGENVECTORS (ORTHONORMAL) OF HESSIAN MATRIX (COLUMNS)" in key:
                        all_keys.update(
                            [
                                f"{key}_EV{i+1}_{axis}"
                                for i in range(3)
                                for axis in ["X", "Y", "Z"]
                            ]
                        )
                    elif "HESSIAN MATRIX" in key:
                        all_keys.update(
                            [
                                f"{key}_{axis1}{axis2}"
                                for axis1 in ["X", "Y", "Z"]
                                for axis2 in ["X", "Y", "Z"]
                            ]
                        )
                else:  # 1D list
                    all_keys.update([f"{key}_{axis}" for axis in ["X", "Y", "Z"]])
            else:
                all_keys.add(key)
    # Sort the keys
    all_keys = sorted(all_keys)

    # Define the preferred order of columns
    preferred_order = [
        "JOB_NAME",
        "CP #",
        "ATOMS",
        "RANK",
        "SIGNATURE",
        "CP COORDINATES_X",
        "CP COORDINATES_Y",
        "CP COORDINATES_Z",
        "Rho",
        "|GRAD(Rho)|",
        "GRAD(Rho)x",
        "GRAD(Rho)y",
        "GRAD(Rho)z",
        "Laplacian",
        "(-1/4)Del**2(Rho))",
        "Diamond",
        "Metallicity",
        "Ellipticity",
        "Theta",
        "Phi",
    ]

    # Now add Eigenvalues and Eigenvectors
    for h in [s for s in all_keys if "EIGENVALUES" in s]:
        preferred_order.append(h)
    for h in [s for s in all_keys if "EIGENVECTORS" in s]:
        preferred_order.append(h)

    # Then Hessian
    for h in [s for s in all_keys if "HESSIAN" in s]:
        preferred_order.append(h)

    # Now add the rest if not already in the preferred order
    for h in all_keys:
        if h not in preferred_order:
            preferred_order.append(h)

    headers = preferred_order

    # Define the output CSV file path
    base_name = os.path.splitext(input_file_path)[0]
    csv_file_path = f"{base_name}_cp_info.csv"

    # Write the CSV file
    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for cp in cp_data:
            row = {}
            for key, value in cp.items():
                if isinstance(value, list):
                    if isinstance(value[0], list):  # 2D list
                        if (
                            "EIGENVECTORS (ORTHONORMAL) OF HESSIAN MATRIX (COLUMNS)"
                            in key
                        ):
                            for i in range(3):
                                for j, axis in enumerate(["X", "Y", "Z"]):
                                    row[f"{key}_EV{i+1}_{axis}"] = value[i][j]
                        elif "HESSIAN MATRIX" in key:
                            for i, axis1 in enumerate(["X", "Y", "Z"]):
                                for j, axis2 in enumerate(["X", "Y", "Z"]):
                                    row[f"{key}_{axis1}{axis2}"] = value[i][j]
                    else:  # 1D list
                        for i, axis in enumerate(["X", "Y", "Z"]):
                            row[f"{key}_{axis}"] = value[i]
                else:
                    row[key] = value
            writer.writerow(row)

    log_print(f"CSV file written to {csv_file_path}")


def interpolate_molecules(mol1, mol2, num_images):
    # Interpolate between two molecules
    interpolated_molecules = []
    for i in range(num_images):
        mol = Molecule()
        for atom1, atom2 in zip(mol1, mol2):
            x = atom1.x + (atom2.x - atom1.x) * (i + 1) / (num_images + 1)
            y = atom1.y + (atom2.y - atom1.y) * (i + 1) / (num_images + 1)
            z = atom1.z + (atom2.z - atom1.z) * (i + 1) / (num_images + 1)
            mol.add_atom(Atom(symbol=atom1.symbol, coords=(x, y, z)))
        mol.guess_bonds()
        interpolated_molecules.append(mol)
    return interpolated_molecules


def main(ams_job_path, atom_pairs):
    ########################################################################################
    # Step 1: getting basic input file information
    ########################################################################################

    # Get the results directory from the ams job file path (same as job path but with extension replaced with .results)
    job_ext = os.path.splitext(ams_job_path)[1]
    results_dir = ams_job_path.replace(job_ext, ".results")
    if ams_job_path.endswith(".ams"):
        kf_path = os.path.join(results_dir, "ams.rkf")
        input_job_name = os.path.basename(ams_job_path).replace(job_ext, "")
    elif ams_job_path.endswith(".rkf"):
        kf_path = ams_job_path
        input_job_name = os.path.basename(ams_job_path).replace(".rkf", "")
    else:
        log_print(f"Invalid job file path: {ams_job_path}")
        return

    # job name is the job base name

    # Load the results kf file
    kf = KFFile(kf_path)

    term_status = kf[("General", "termination status")]

    log_print(f"Processing job {input_job_name} with termination status {term_status}")

    is_unrestricted = "Unrestricted Yes" in kf[("General", "user input")]

    # Check if EEF is present
    has_eef = "im0.eeEField" in kf.read_section("NEB").keys() or user_eef is not None

    # Get exiting job object from run file
    run_file_path = ams_job_path.replace(job_ext, ".run")
    # if run file does not exist, create it using the contents of General: user input
    if not os.path.exists(run_file_path):
        with open(run_file_path, "w") as run_file:
            run_file.write(kf[("General", "user input")])

    input_job = AMSJob.from_inputfile(run_file_path)

    # replace NEB task with singlepoint
    base_settings = input_job.settings
    base_settings.input.adf.QTAIM.Enabled = "yes"
    ams_settings = Settings()
    ams_settings.task = "SinglePoint"
    base_settings.input.ams = ams_settings

    # log_print(base_settings)

    # remove occupations from input if present
    # if base_settings.input.adf.get("occupations"):
    #     occ = base_settings.input.adf.find_case("occupations")  # gets key if case differs
    #     del base_settings.input.adf[occ]

    num_images = kf[("NEB", "nebImages")] + 2

    highest_index_image = kf[("NEB", "highestIndex")]

    ########################################################################################
    # Step 2: create jobs for each image and then run them
    ########################################################################################

    jobs = MultiJob(name=input_job_name)

    im_num = 0

    for i in range(num_images):
        num_atoms = kf[("NEB", f"im{i}.nAtoms")]
        num_species = kf[("NEB", f"im{i}.nSpecies")]
        atom_species_indices = kf[("NEB", f"im{i}.SpIndices")]
        species_symbols = [
            kf[("NEB", f"im{i}.sp{j+1}symbol")] for j in range(num_species)
        ]
        species_Z = [kf[("NEB", f"im{i}.sp{j+1}Z")] for j in range(num_species)]
        atom_species = [
            species_symbols[atom_species_indices[j] - 1] for j in range(num_atoms)
        ]
        atom_numbers = [
            species_Z[atom_species_indices[j] - 1] for j in range(num_atoms)
        ]
        coords = kf[("NEB", f"im{i}.xyzAtoms")]
        coords = [
            Units.convert(v, "bohr", "angstrom") for v in coords
        ]  # to convert to angstrom for comparison to coords shown in ADFMovie

        # coords is a 1D list of coordinates, so we need to reshape it to a 2D list
        atom_coords = [tuple(coords[i : i + 3]) for i in range(0, len(coords), 3)]

        # create molecule for image
        im_mol = Molecule(positions=atom_coords, numbers=atom_numbers)
        im_mol.guess_bonds()
        moles = [im_mol]

        if i in [highest_index_image - 1, highest_index_image]:
            ip1 = i + 1
            ip1_coords = kf[("NEB", f"im{ip1}.xyzAtoms")]
            ip1_coords = [
                Units.convert(v, "bohr", "angstrom") for v in ip1_coords
            ]  # to convert to angstrom for comparison to coords shown in ADFMovie

            # coords is a 1D list of coordinates, so we need to reshape it to a 2D list
            ip1_atom_coords = [
                tuple(ip1_coords[i : i + 3]) for i in range(0, len(ip1_coords), 3)
            ]
            ip1_mol = Molecule(positions=ip1_atom_coords, numbers=atom_numbers)
            moles.extend(interpolate_molecules(im_mol, ip1_mol, num_extra_images))

        for mol in moles:
            if has_eef:
                # EEF is present, so need to run three jobs with pos/neg/no EEF
                eef = (
                    None
                    if not has_eef
                    else (
                        kf[("NEB", f"im{i}.eeEField")] if user_eef is None else user_eef
                    )
                )
                job_settings = base_settings.copy()
                for eef_pair in eef_pairs:
                    eef_name, eef_val = eef_pair
                    eef_vals = [v * eef_val if v != 0.0 else v for v in eef]
                    eef_str = " ".join([f"{v}" for v in eef_vals])
                    s = Settings()
                    s.ElectrostaticEmbedding.ElectricField = eef_str
                    job_settings.input.ams.system = s
                    job_name = f"{input_job_name}_{eef_name}_im{im_num:03d}"
                    job = AMSJob(molecule=mol, settings=job_settings, name=job_name)
                    jobs.children.append(job)
            else:
                # no EEF, so only need to run one job for the image
                job_name = f"{input_job_name}_im{im_num:03d}"
                job = AMSJob(molecule=mol, settings=base_settings, name=job_name)
                jobs.children.append(job)
            im_num += 1

    # print each job's name and coordinates of first atom (for debugging only; remember to check that the correct bond distance is being printed)
    job_num = 0
    atom_nums = atom_pairs[atom_pair_for_bond_distance_printout]
    bond_name = f"{atom_species[atom_nums[0]-1]}{atom_nums[0]}-{atom_species[atom_nums[1]-1]}{atom_nums[1]}"
    log_print(f"Printing bond distances for {bond_name} in each job:")
    for job in jobs.children:
        if "origEEF" in job.name:
            if num_extra_images > 0:
                if job_num == highest_index_image:
                    log_print("start extra images")
                elif job_num == highest_index_image + num_extra_images:
                    log_print("highest image vvvv")
                elif job_num == highest_index_image + num_extra_images + 1:
                    log_print("highest image ^^^^")
                elif job_num == highest_index_image + 2 * num_extra_images + 1:
                    log_print("end extra images")
            log_print(
                f"{job.name}: {bond_name} distance = {job.molecule.atoms[atom_nums[0]-1].distance_to(job.molecule.atoms[atom_nums[1]-1])}"
            )
            job_num += 1

    log_print(f"Running {len(jobs.children)} jobs...")

    jobs.run()

    process_results(
        jobs,
        atom_pairs,
        ams_job_path,
        plot_y_prop_list,
        plot_x_prop_list,
        unrestricted=is_unrestricted,
    )


def bcp_func_wrapper(args):
    job, atom_pairs, unrestricted = args
    return get_bcp_properties(job, atom_pairs, unrestricted=unrestricted)


def densf_func_wrapper(args):
    job, path = args
    return generate_full_t41(job, path)


def process_results(jobs, atom_pairs, path, prop_list, x_prop_list, unrestricted=False):
    ########################################################################################
    # Step 3: extract bond critical point information from each job's output
    ########################################################################################

    # We'll save results to a single CSV file in the results_dir
    total_cp_data = []
    for job in jobs.children:
        cp_data = get_bcp_properties(job, atom_pairs, unrestricted=unrestricted)
        total_cp_data.extend(cp_data)

    write_csv(total_cp_data, path)

    generate_plots(
        total_cp_data,
        prop_list,
        x_prop_list,
        os.path.dirname(path),
        combined_plots_y_prop_lists,
    )

    if len(densf_bb_atom_numbers) > 0:
        # with ThreadPoolExecutor(max_workers=num_cores) as executor:
        #     args = [(job, os.path.dirname(path)) for job in jobs.children]
        #     list(executor.map(densf_func_wrapper, args))
        for job in jobs.children:
            generate_full_t41(job, os.path.dirname(path))


def test_post_processing_single_job(job_path, atom_pairs):
    input_job = AMSJob.load_external(job_path)
    cp_data = get_bcp_properties(input_job, atom_pairs, unrestricted=True)

    write_csv(cp_data, job_path)


def test_post_processing_multiple_jobs(dill_path, atom_pairs, unrestricted=False):
    # every directory in jobs_path can be loaded using AMSJob.load_external
    jobs = load(dill_path)
    process_results(
        jobs,
        atom_pairs,
        dill_path,
        plot_y_prop_list,
        plot_x_prop_list,
        unrestricted=unrestricted,
    )
    return


if __name__ == "__main__":
    if restart_dill_paths and len(restart_dill_paths) > 0:
        for restart_dill_path, atom_pairs in zip(restart_dill_paths, atom_pairs_list):
            test_post_processing_multiple_jobs(
                restart_dill_path, atom_pairs, unrestricted=True
            )
    else:
        for job_path, atom_pairs in zip(ams_job_paths, atom_pairs_list):
            main(job_path, atom_pairs)
