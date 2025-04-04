# from scm.plams import *
import os
import csv
import subprocess
from math import sqrt, atan, floor, ceil
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# from concurrent.futures import ThreadPoolExecutor

# This script is intended to be run on the results of a NEB calculation in AMS.
# The script will create a series of single-point calculations for each image in the NEB calculation, with the molecule from each image as the input geometry.
# The script will also create three jobs for each image if the NEB calculation includes an electric field, with the electric field set to the original field, opposite (reverse) field, and no field.
# Then the script will extract from the output of the single point calculations information for bond critical points for specified atom pairs.

# Run with the command:
# plams /path/to/ADF_NEB_bcp_analysis.py

# The working directory will be placed in the directory from which you run the command.
# Generated CSV and plot files will be placed adjacent to the specified input file (next to the AMS job file, or the dill file if restarting).

# Documentation for PLAMS:
# https://www.scm.com/doc.2018/plams/components/components.html

# Use KFBrowser in File->Expert Mode to see the contents of the .rkf file(s)

########################################################################################
# USER SETTINGS: change the following parameters as needed
########################################################################################

# There are three ways to run this script:
# 1. Initial run on NEB output by pointing to the your-job.ams file directly (or if you don't have it, you can point to the output ams.rkf file, which is less preferred)
# 2. You can restart using the "dill" file that is created inside the working directory of a previous run, which will use the previously run single-point calculations. This is useful for generating new sets of plots quickly
# 3. You can run the script on previously processed CSV files to do statistical analysis or generate new plots.

#### Run on one job at a time. It was initially intended to be able to run on multiple jobs, but in the course of development that has been temporarily broken.

# Define the path to the AMS job file (`path/to/job.ams`), or if you don't have an ams file, use
# the path to the ams.rkf result file. Set the dill and csv paths to be empty in order to have the script use the AMS job file input.
ams_job_paths = ["/Users/haiiro/NoSync/2025_AMSPythonData/CM_Ashley/no_field_NEB.ams"]

# To rerun on a previously processed file, set the restart_dill_path to the path of the dill file in the working directory of the previous run. Otherwise, set to None, False, or ''. Set the csv paths to be an empty list if you want the script to use the dill file input.
restart_dill_paths = []

# Define paths to previously created cp data CSV files in order to do statistical analysis.
csv_file_paths = []

# You can control the starting and ending NEB image number to include in the analysis here.
start_image = 2 # 0 means the first image in the NEB, 1 means the second image, etc.
end_image = 20  # -1 means the last image in the NEB

# Define atom pairs (pairs of atom numbers) for which to extract bond critical point information.
# One list for each input file defined above
atom_pairs_list = (  # one-based indices, same as shown in AMSView
    (
        (9, 12),  # breaking C-O bond
        (12, 13), # breaking C-O O's remaining (single -> double) C-O bond
        (7, 9), # breaking C-O C's C-C bond towards OH
        (9, 11), # breaking C-O C's C-C bond towards CO2
        (1, 14), # forming C-C bond
        (13, 14), # forming C-C other (double -> single) C-C bond
        (1, 15), # forming C-C ring C-C bond with CO2 C
        (1, 3), # forming C-C ring C-C bond with (aromatic -> single) ring C
        (1, 11), # forming C-C ring C-C bond with (aromatic -> single) other ring C
    ),
)

# Index of atom pair for which to print the bond distance of each image to be run. (or set to -1 to not print any distances)
atom_pair_for_bond_distance_printout = 4

##### densf full grid settings #####

# This script can also be used to create full 3d grids for each step along the NEB.
# This is useful if Bondalyzer is to be used.
# Configure the full grid settings below.
# If no full grid is desired, leave densf_bb_atom_numbers as an empty list.

densf_bb_atom_numbers = []
densf_bb_padding = 5.0  # Angstroms
densf_bb_spacing = (
    0.05  # Angstroms (densf "fine" is 0.05, "medium" is 0.1, "coarse" is 0.2)
)

##### end densf full grid settings #####

##### EEF Settings #####
# If the input NEB file includes an applied electric field, then that field will determine the
# magnitude and direction of the electric fields applied to the images in the NEB calculation,
# which will include the no-field and opposite-field cases, in addition to the original field.
# However, you may want to specify your own set of electric fields.
# You may define a single electric field as XYZ components in atomic units (Ha/(e bohr)),
# and the opposite and no-field case will be included.
# THIS OVERRIDES THE EEF USED IN THE NEB CALCULATION.
# Uncomment the following line to specify your own electric field, or leave it as None to use the NEB eef if present.
user_eef = None  # (0.0, 0.0, 0.01)
# Need to convert electric field magnitude units. In the NEB rkf file, they're Ha/(e bohr), but in the
# new jobs they need to be V/Angstrom. The conversion factor is 51.4220861908324.
eef_conversion_factor = 51.4220861908324
# Define the EEF pairs
# eef_pairs = (("origEEF", eef_conversion_factor), ("revEEF", -eef_conversion_factor), ("noEEF", 0))
eef_pairs = (("origEEF", eef_conversion_factor))
##### end EEF Settings #####

##### Extra interpolated single point settings #####
# To get better resultion around the transition state, we'll identify the TS image (highest energy image)
# and create additional images between it and the adjacent images, using a linear interpolation of the
# coordinates of the adjacent images. Here, you specify how many extra images to add on *each* side of the TS image.
num_extra_images = 0
# This then determines how many images to the left/right of the TS image to create. `num_extra_images` images will be created between each adjacent pair of images.
# So "1" will result in `num_extra_images` images being added only between the TS image and its adjacent images,
# while "3" will add `num_extra_images` between each image pair starting 3 images before the TS, etc.
num_adjacent_extra_images = 2
# Set `num_extra_images` to 0 to disable this feature.
##### end Extra interpolated single point settings #####

##### Plot settings #####
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
    "Rho": ["Molecular bond energy", "Rho"],
    "Rho d/dx": ["Molecular bond energy", "Rho d/dx"],
    "Angles": ["Molecular bond energy", "Theta", "Phi"],
    "Angles d/dx": ["Molecular bond energy", "Theta d/dx", "Phi d/dx"],
}

plot_x_prop_list = [
    "C1-C14 distance",
    "Reaction coordinate",
]
##### end Plot settings #####

##### Spin density CP search settings #####
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
##### end Spin density CP search settings #####

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
    atom_pair_distances = []
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
        bond_length = out_mol.atoms[pair[0] - 1].distance_to(out_mol.atoms[pair[1] - 1])
        atom_pair_distances.append(bond_length)
        if unrestricted:
            # the CP locations in the unrestricted spin-a and spin-b densities are not the
            # same as in the total density (those are the ones we have now). So we need to
            # include a bunch of points around the known total-density CP location at which
            # to compute the spin-a and spin-b densities, and then we'll take the point with
            # lowest spin-a/b density gradient magnitude as the CP location in the spin-a/b.
            # The check points will be on a regular 3d grid around the total-density CP location,
            # with point spacing based on the distance between the two atoms using the defined
            # fraction of that distance.
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
        bond_length = atom_pair_distances[i]
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
    eef_strs = ["origEEF", "revEEF", "noEEF"]
    has_eef = any([eef in str(image_names) for eef in eef_strs])

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

    num_eef = len(eef_types) if has_eef else 0

    line_styles = {
        "_origEEF": "-",  # solid
        "_revEEF": "--",  # dashed
        "_noEEF": ":",  # dotted
        "": "-",  # solid (for non-EEF plots)
    }

    for x_prop in x_prop_list:
        for plot_name, y_prop_list in combined_y_prop_lists.items():
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
                figsize=(1+5*(num_eef+1), 3 * len(expanded_y_prop_list)),
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
                    if num_eef > 0:
                        ax = axs[i, j] if len(expanded_y_prop_list) > 1 else axs[j]
                    else:
                        ax = axs[i] if len(expanded_y_prop_list) > 1 else axs
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
        # Now add atom properties from input job molecule
        for i in range(len(mol1.atoms)):
            mol.atoms[i].properties = mol1.atoms[i].properties.copy()
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

    is_unrestricted = "Unrestricted Yes".lower() in kf[("General", "user input")].lower()

    # Check if EEF is present
    has_eef = ("im0.eeEField" in kf.read_section("NEB").keys() and any(float(i) != 0.0 for i in kf[("NEB", "im0.eeEField")])) or user_eef is not None
    
    if has_eef:
        log_print(f"Using user-defined EEF: {user_eef}" if user_eef is not None else "EEF is present in the NEB job.")
    else:
        log_print("No EEF present in the NEB job. Proceeding without EEF.")
    
    user_input = kf[("General", "user input")]

    # Get exiting job object from run file
    run_file_path = ams_job_path.replace(job_ext, ".run")
    # if run file does not exist, create it using the contents of General: user input
    if not os.path.exists(run_file_path):
        with open(run_file_path, "w") as run_file:
            run_file.write(user_input)

    input_job = AMSJob.from_inputfile(run_file_path)
    input_job_mol = input_job.molecule['']
    
    # log_print(input_job_mol.properties)
    # log_print(input_job_mol[1].properties)
    # return

    # replace NEB task with singlepoint
    base_settings = input_job.settings
    base_settings.input.adf.QTAIM.Enabled = "yes"
    ams_settings = Settings()
    ams_settings.task = "SinglePoint"
    base_settings.input.ams = ams_settings
    
    log_print("Base settings for Single points (minus any applied EEFs):")
    log_print(base_settings)

    # remove occupations from input if present
    # if base_settings.input.adf.get("occupations"):
    #     occ = base_settings.input.adf.find_case("occupations")  # gets key if case differs
    #     del base_settings.input.adf[occ]

    num_images = kf[("NEB", "nebImages")] + 2

    highest_index_image = kf[("NEB", "highestIndex")]
    
    log_print( f"Total number of images in NEB: {num_images}, highest index image: {highest_index_image}")

    ########################################################################################
    # Step 2: create jobs for each image and then run them
    ########################################################################################

    jobs = MultiJob(name=input_job_name)

    im_num = 0
    
    extra_image_nums = list(range(
        highest_index_image - num_adjacent_extra_images,
        highest_index_image + num_adjacent_extra_images
    )) if num_adjacent_extra_images > 0 and num_extra_images > 0 else []
    extra_image_job_names = []

    for i in range(num_images):
        if i < start_image or (i > end_image and end_image > 0):
            log_print(f"Skipping image {i} as it is outside the specified range of NEB images to include.")
            im_num += 1
            continue
        num_atoms = kf[("NEB", f"im{i}.nAtoms")]
        num_species = kf[("NEB", f"im{i}.nSpecies")]
        atom_species_indices = kf[("NEB", f"im{i}.SpIndices")]
        species_symbols = [
            kf[("NEB", f"im{i}.sp{j+1}symbol")] for j in range(num_species)
        ]
        # if any of the species symbols contain a digit, then we need to redo them using `inputSymbol` instead of `symbol` (handles a change in the recent AMS versions)
        for j in range(num_species):
            if any(char.isdigit() for char in species_symbols[j]):
                species_symbols[j] = kf[("NEB", f"im{i}.sp{j+1}inputSymbol")]
        
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
        im_mol.properties = input_job.molecule[''].properties.copy()  # copy properties from the input job
        im_mol.guess_bonds()
        # Now add atom properties from input job molecule
        for i in range(len(input_job_mol.atoms)):
            im_mol.atoms[i].properties = input_job_mol.atoms[i].properties.copy()
        
        moles = [im_mol]

        if i in extra_image_nums:
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
            is_in_extra_image_range = True
        else:
            is_in_extra_image_range = False

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
                    if is_in_extra_image_range and eef_pair == eef_pairs[0] and mol != moles[0]:
                        extra_image_job_names.append(job_name)
            else:
                # no EEF, so only need to run one job for the image
                job_name = f"{input_job_name}_im{im_num:03d}"
                job = AMSJob(molecule=mol, settings=base_settings, name=job_name)
                jobs.children.append(job)
                if is_in_extra_image_range and mol != moles[0]:
                    extra_image_job_names.append(job_name)
            im_num += 1

    # print each job's name and coordinates of first atom (for debugging only; remember to check that the correct bond distance is being printed)
    if atom_pair_for_bond_distance_printout >= 0:
        atom_nums = atom_pairs[atom_pair_for_bond_distance_printout]
        bond_name = f"{atom_species[atom_nums[0]-1]}{atom_nums[0]}-{atom_species[atom_nums[1]-1]}{atom_nums[1]}"
        log_print(f"Printing bond distances for {bond_name} in each job:")
        for job in jobs.children:
            suffix = f" (Extra Image)" if job.name in extra_image_job_names else ""
            log_print(
                f"{job.name}: {bond_name} distance = {job.molecule.atoms[atom_nums[0]-1].distance_to(job.molecule.atoms[atom_nums[1]-1])}{suffix}"
            )
    
    # return

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


def convert_string_to_number(s, h, no_convert_headers=["SIGNATURE"]):
    """
    Convert string to int or float if possible, otherwise return original string.

    Args:
        s (str): String to convert

    Returns:
        Union[int, float, str]: Converted value
    """
    if h in no_convert_headers:
        return s
    try:
        # First try converting to int
        value = int(s)
        return value
    except ValueError:
        try:
            # Then try converting to float
            value = float(s)
            return value
        except ValueError:
            # If both fail, return original string
            return s


def read_simple_csv(csv_path):
    """
    Read a CSV file written by write_csv() into a list of dictionaries.

    Args:
        csv_path (str): Path to the CSV file

    Returns:
        list: List of dictionaries, same structure as all_cp_data
    """
    cp_data = []

    with open(csv_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Remove empty fields, convert string'None' to None, and convert numbers
            cp_dict = {
                k: (None if v == "None" else convert_string_to_number(v, k))
                for k, v in row.items()
                if v != ""
            }
            cp_data.append(cp_dict)

    return cp_data


def prepare_bcp_crossing_df(
    df_full,
    rxn_coord_name,
    use_theta_phi_crossings=False,
    use_spin_crossings=False,
    use_spin_delta_e_r=False,
    sys_whitelist=[],
    eef_whitelist=[],
):
    from scipy.interpolate import interp1d

    ### BEGIN HELPER FUNCTIONS ###

    # Step 2: Create nested dict of BCP interpolations
    def prepare_bcp_interpolations(df_full, rxn_coord_name, system_eef_pairs):
        # First, identify float columns that aren't the reaction coordinate
        float_cols = df_full.select_dtypes(include=[np.float64, np.float32]).columns
        prop_cols = [col for col in float_cols]

        # Initialize the outer dictionary
        bcp_interps = {}

        # For each system-EEF pair
        for system_eef in system_eef_pairs:
            system, eef = system_eef.split("_")
            # Get data for this system
            system_mask = (df_full["SYSTEM"] == system) & (
                df_full["EEF"].astype(str) == eef
            )
            system_data = df_full[system_mask]

            # Initialize dict for this system
            bcp_interps[system_eef] = {}

            # For each unique BCP in this system
            for atoms in system_data["ATOMS"].unique():
                # Get data for this BCP
                bcp_mask = system_data["ATOMS"] == atoms
                bcp_data = system_data[bcp_mask]

                # Initialize dict for this BCP's properties
                bcp_interps[system_eef][atoms] = {}

                # Create interpolation for each property
                for prop in prop_cols:
                    x = bcp_data[rxn_coord_name].values
                    y = bcp_data[prop].values
                    # Sort x and y by x values to ensure proper interpolation
                    sort_idx = np.argsort(x)
                    x = x[sort_idx]
                    y = y[sort_idx]
                    # Create interpolation
                    interp = interp1d(x, y, kind="linear")
                    bcp_interps[system_eef][atoms][prop] = interp

        return bcp_interps

    # Step 3: Find BCP crossing points for each system
    def identify_all_system_crossings(
        bcp_interps,
        system_eef_pairs,
        use_theta_phi_crossings=False,
        use_spin_crossings=False,
    ):
        def identify_bcp_crossings(
            bcp_interps, system_eef_pairs, crossing_var_name="Rho"
        ):
            system_crossings = {}

            for system_eef in system_eef_pairs:
                system_crossings[system_eef] = {}

                # Get all BCPs for this system
                bcps = list(bcp_interps[system_eef].keys())

                # For each pair of BCPs
                for i, bcp1 in enumerate(bcps):
                    for bcp2 in bcps[i + 1 :]:
                        # Get Rho interpolations for both BCPs
                        rho1 = bcp_interps[system_eef][bcp1][crossing_var_name]
                        rho2 = bcp_interps[system_eef][bcp2][crossing_var_name]

                        # Get the domain where both interpolations are valid
                        x_min = max(rho1.x[0], rho2.x[0])
                        x_max = min(rho1.x[-1], rho2.x[-1])

                        # Create dense sampling of points in this domain
                        x_vals = np.linspace(x_min, x_max, 1000)
                        y1 = rho1(x_vals)
                        y2 = rho2(x_vals)

                        # Find where the difference changes sign (crossing points)
                        diff = y1 - y2
                        cross_indices = np.where((diff[:-1] * diff[1:]) < 0)[0]

                        # If there's a crossing
                        if len(cross_indices) > 0:
                            # For each crossing (there might be multiple)
                            for idx in cross_indices:
                                # Linear interpolation to get more precise crossing point
                                x0 = x_vals[idx]
                                x1 = x_vals[idx + 1]
                                y0 = diff[idx]
                                y1 = diff[idx + 1]
                                x_cross = x0 - y0 * (x1 - x0) / (y1 - y0)

                                # Create BCP_CROSSING key (sorted ATOMS values with underscore)
                                bcp_crossing = "_".join(sorted([bcp1, bcp2]))

                                # Store the crossing point
                                system_crossings[system_eef][bcp_crossing] = x_cross

            return system_crossings

        all_system_crossings = {}
        vars = ["Rho"] + (["Theta", "Phi"] if use_theta_phi_crossings else [])
        for var in vars:
            for spin in [""] + (["_A", "_B"] if use_spin_crossings else []):
                if any(c.endswith(spin) for c in df_full.columns):
                    var_Str = var + spin
                    all_system_crossings[var_Str] = identify_bcp_crossings(
                        bcp_interps, system_eef_pairs, crossing_var_name=var_Str
                    )

        return all_system_crossings

    # Step 4: Find common crossings and create ordered dicts
    def get_common_crossing_bcp_order(all_system_crossings, system_eef_pairs):
        # Find crossings that occur in all systems
        first_var_system_key = list(all_system_crossings.keys())[0]
        all_crossings = set(
            all_system_crossings[first_var_system_key][system_eef_pairs[0]].keys()
        )
        for var_systems in all_system_crossings.values():
            for system_eef in system_eef_pairs:
                all_crossings &= set(var_systems[system_eef].keys())

        # Order crossings by their R value in the first system
        first_system = system_eef_pairs[0]
        ordered_crossings = sorted(
            all_crossings,
            key=lambda x: all_system_crossings[first_var_system_key][first_system][x],
        )

        # Create crossing_order dict
        crossing_order = {
            crossing: idx for idx, crossing in enumerate(ordered_crossings)
        }

        # Get unique BCPs involved in these crossings
        unique_bcps = set()
        for crossing in all_crossings:
            bcp1, bcp2 = crossing.split("_")
            unique_bcps.add(bcp1)
            unique_bcps.add(bcp2)

        # Create ordered list of BCPs
        ordered_bcps = sorted(unique_bcps)

        # Create bcp_order dict
        bcp_order = {bcp: idx for idx, bcp in enumerate(ordered_bcps)}

        return ordered_crossings, crossing_order, ordered_bcps, bcp_order

    # Step 5: Create output DataFrame with all properties
    def generate_dataframe(
        system_eef_pairs,
        bcp_interps,
        all_system_crossings,
        ordered_crossings,
        ordered_bcps,
        use_theta_phi_crossings=False,
        use_spin_crossings=False,
        use_spin_delta_e_r=use_spin_delta_e_r
    ):
        rows = []

        for system_eef in system_eef_pairs:
            system, eef = system_eef.split("_")

            # Get first BCP to find E_TS and R_TS (all BCPs share same energy values)
            first_bcp = ordered_bcps[0]
            energy_interp = bcp_interps[system_eef][first_bcp]["Molecular bond energy"]
            x_vals = np.linspace(energy_interp.x[0], energy_interp.x[-1], 1000)
            y_vals = energy_interp(x_vals)
            max_idx = np.argmax(y_vals)
            R_TS = x_vals[max_idx]
            E_TS = y_vals[max_idx]

            # For each crossing in the ordered list
            for bcp_crossing in ordered_crossings:
                row = {
                    "SYSTEM": system,
                    "EEF": np.float64(eef),
                    "BCP_CROSSING": bcp_crossing,
                }

                # Get R and E at crossing
                R = all_system_crossings["Rho"][system_eef][bcp_crossing]
                E = energy_interp(R)

                # Calculate ∆E_TS and ∆R_TS
                row["$\\Delta E_{{\\rm{{{TS}}}}}$"] = E_TS - E
                row["$\\Delta R_{{\\rm{{{TS}}}}}$"] = R_TS - R

                # Get rho value at crossing
                bcp1, bcp2 = bcp_crossing.split("_")
                rho_interp = bcp_interps[system_eef][bcp1]["Rho"]
                row["$\\rho$"] = np.float64(rho_interp(R))

                # Calculate ∆rho_k for all BCPs
                for k, bcp_k in enumerate(ordered_bcps):
                    key = f"$\\Delta \\rho_{{\\rm{{{bcp_k}}}}}$"
                    if bcp_k in bcp_crossing:
                        row[key] = np.float64(0.0)
                    else:
                        rho_k_interp = bcp_interps[system_eef][bcp_k]["Rho"]
                        rho_k = rho_k_interp(R)
                        row[key] = row["$\\rho$"] - rho_k

                # Get all bond distances
                distance_cols = [
                    col
                    for col in bcp_interps[system_eef][first_bcp].keys()
                    if col.endswith(" distance")
                ]
                for dist_col in distance_cols:
                    bcp_name = dist_col.replace(" distance", "")
                    dist_interp = bcp_interps[system_eef][first_bcp][dist_col]
                    row[f"$d_{{\\rm{{{bcp_name}}}}}$"] = np.float64(dist_interp(R))

                # Add theta/phi crossing information if requested
                if use_theta_phi_crossings:
                    if "Theta" in all_system_crossings:
                        R_theta = all_system_crossings["Theta"][system_eef][
                            bcp_crossing
                        ]
                        row["$\\Delta R_{TS_{\\theta}}$"] = R_TS - R_theta
                    if "Phi" in all_system_crossings:
                        R_phi = all_system_crossings["Phi"][system_eef][bcp_crossing]
                        row["$\\Delta R_{TS_{\\phi}}$"] = R_TS - R_phi

                # Add spin-resolved properties if present
                if use_spin_crossings:
                    for spin in ["_A", "_B"]:
                        spin_str = spin.replace("_", "")
                        if f"Rho{spin}" in all_system_crossings:
                            R_spin = all_system_crossings[f"Rho{spin}"][system_eef][
                                bcp_crossing
                            ]
                            if use_spin_delta_e_r:
                                E_spin = energy_interp(R_spin)
                                row[f"$\\Delta E_{{\\rm{{{spin_str},TS}}}}$"] = (
                                    E_TS - E_spin
                                )
                                row[f"$\\Delta R_{{\\rm{{{spin_str},TS}}}}$"] = (
                                    R_TS - R_spin
                                )
                            rho_spin_interp = bcp_interps[system_eef][bcp1][
                                f"Rho{spin}"
                            ]
                            row[f"$\\rho_{{\rm{{{spin_str}}}}}$"] = rho_spin_interp(
                                R_spin
                            )

                            # Calculate ∆rho_k for spin component
                            for k, bcp_k in enumerate(ordered_bcps):
                                rho_k_spin_interp = bcp_interps[system_eef][bcp_k][
                                    f"Rho{spin}"
                                ]
                                rho_k_spin = rho_k_spin_interp(R_spin)
                                row[
                                    f"$\\Delta \\rho_{{\\rm{{{spin_str},{bcp_k}}}}}$"
                                ] = (row[f"$\\rho_{{\rm{{{spin_str}}}}}$"] - rho_k_spin)

                rows.append(row)

        # Create DataFrame from rows
        return pd.DataFrame(rows), rows

    ### END HELPER FUNCTIONS ###

    # Step 1: Create sorted list of unique SYSTEM-EEF pairs
    # Create system_eef identifiers and get unique sorted list
    system_eef_pairs = sorted(
        set(f"{system}_{eef}" for system, eef in zip(df_full["SYSTEM"], df_full["EEF"]))
    )

    # Apply whitelists if provided
    if sys_whitelist:
        system_eef_pairs = [
            s for s in system_eef_pairs if s.split("_")[0] in sys_whitelist
        ]
    if eef_whitelist:
        system_eef_pairs = [
            s for s in system_eef_pairs if s.split("_")[1] in eef_whitelist
        ]

    # Step 2: Prepare BCP interpolations
    bcp_interps = prepare_bcp_interpolations(df_full, rxn_coord_name, system_eef_pairs)

    # Step 3: Find BCP rho crossings for each system (and each spin if present) (and theta and phi crossings if requested)
    all_system_crossings = identify_all_system_crossings(
        bcp_interps,
        system_eef_pairs,
        use_theta_phi_crossings=use_theta_phi_crossings,
        use_spin_crossings=use_spin_crossings,
    )

    # Step 4: Find common crossings and create ordered dicts
    ordered_crossings, crossing_order, ordered_bcps, bcp_order = (
        get_common_crossing_bcp_order(all_system_crossings, system_eef_pairs)
    )

    # Step 5: Create output DataFrame with all properties
    df, rows = generate_dataframe(
        system_eef_pairs,
        bcp_interps,
        all_system_crossings,
        ordered_crossings,
        ordered_bcps,
        use_theta_phi_crossings=use_theta_phi_crossings,
        use_spin_crossings=use_spin_crossings,
    )

    return df, rows


def statistical_analysis(
    cp_data,
    output_dir
):
    """
    Here, we'll perform some statistical analysis on the data.
    Early analysis from the plots revealed that the bond critical points' (BCPs) rho, theta, and phi values will cross as the reaction progresses.
    We want to better understand how these crossings correlate with the reaction coordinate and energy.
    """

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from boruta import BorutaPy
    from sklearn.ensemble import RandomForestRegressor
    
    use_spin_delta_e_r = False
    
    ### BEGIN HELPER FUNCTIONS ###

    # Generate correlations matrices from a provided DataFrame
    def create_correlation_matrix(df: pd.DataFrame, output_dir: str) -> None:
        """
        Creates and saves correlation matrix plots along with system and BCP crossing information.
        LaTeX rendering is enabled for column names. Annotations show r-squared * 100,
        rounded to the tens place (-10 to 10).

        Args:
            df: Input DataFrame containing 'SYSTEM', 'EEF', and 'BCP_CROSSING' columns
            output_dir: Directory to save outputs
        """
        # Enable LaTeX rendering
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            }
        )

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate system information
        systems = (
            df[["SYSTEM", "EEF"]]
            .apply(lambda x: f"{x['SYSTEM']}_{x['EEF']}", axis=1)
            .unique()
        )
        bcp_crossings = df["BCP_CROSSING"].unique()
        
        unique_sys = "+".join(df["SYSTEM"].unique().tolist())
        if any("{A" in c or "{B" in c for c in df.columns):
            unique_sys += "_spin"
        if any("theta" in c or "phi" in c for c in df.columns):
            unique_sys += "_directionality"

        # Save system and BCP crossing information
        with open(os.path.join(output_dir, f"system_info_{unique_sys}.txt"), "w") as f:
            f.write("Systems:\n")
            for system in sorted(systems):
                f.write(f"{system}\n")
            f.write("\nBCP Crossings:\n")
            for bcp in sorted(bcp_crossings):
                f.write(f"{bcp}\n")

        # Create correlation matrix
        # Remove non-numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()

        # Create rounded r-squared values for annotations
        annot_matrix = np.round(corr_matrix * 10)  # round to nearest 10

        # Create correlation matrix plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            cmap="coolwarm",
            center=0,
            annot=annot_matrix,
            fmt=".0f",  # use integers for annotation
            square=True,
            cbar_kws={"label": "Correlation Coefficient"},
        )
        plt.title(f"Correlation Matrix (systems: {unique_sys})")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=60, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"correlation_matrix_{unique_sys}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Save correlation matrix as CSV
        corr_matrix.to_csv(os.path.join(output_dir, f"correlation_matrix_{unique_sys}.csv"))

        # Reset matplotlib parameters to default
        plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif"})

    def perform_pca(
        df: pd.DataFrame,
        target_cols: list[str],
        output_dir: str,
        n_components: int = None,
    ) -> tuple:
        """
        Performs PCA on features and creates visualization plots.

        Args:
            df: Input DataFrame containing 'SYSTEM', 'EEF', and 'BCP_CROSSING' columns
            target_cols: List of numeric column indices to exclude from PCA (0-based)
            output_dir: Directory to save outputs
            n_components: Number of components for PCA (default=None for all)

        Returns:
            tuple: (PCA object, transformed features DataFrame)
        """
        # Enable LaTeX rendering
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            }
        )

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get system information
        systems = (
            df[["SYSTEM", "EEF"]]
            .apply(lambda x: f"{x['SYSTEM']}_{x['EEF']}", axis=1)
            .unique()
        )
        bcp_crossings = df["BCP_CROSSING"].unique()
        
        unique_sys = "+".join(df["SYSTEM"].unique().tolist())
        if any("{A" in c or "{B" in c for c in df.columns):
            unique_sys += "_spin"
        if any("theta" in c or "phi" in c for c in df.columns):
            unique_sys += "_directionality"

        # Save system and BCP crossing information
        with open(os.path.join(output_dir, f"system_info_{unique_sys}.txt"), "w") as f:
            f.write("Systems:\n")
            for system in sorted(systems):
                f.write(f"{system}\n")
            f.write("\nBCP Crossings:\n")
            for bcp in sorted(bcp_crossings):
                f.write(f"{bcp}\n")

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Identify target and feature columns
        feature_cols = [
            col for col in numeric_cols if col not in target_cols
        ]
        X = df[feature_cols]

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Get variance ratios
        var_ratio = pca.explained_variance_ratio_
        cum_var_ratio = np.cumsum(var_ratio)

        # Print variance explained
        with open(os.path.join(output_dir, f"variance_explained_{unique_sys}.txt"), "w") as f:
            f.write("Variance explained by each component:\n")
            for i, var in enumerate(var_ratio):
                f.write(f"PC{i+1}: {var:.4f} ({cum_var_ratio[i]:.4f} cumulative)\n")

        # 1. Scree plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(var_ratio) + 1), var_ratio, "bo-")
        plt.plot(range(1, len(var_ratio) + 1), cum_var_ratio, "ro-")
        plt.xlabel("Principal Component")
        plt.ylabel("Proportion of Variance Explained")
        plt.title(f"Scree Plot (systems: {unique_sys})")
        plt.legend(["Individual", "Cumulative"])
        plt.grid(True)
        plt.savefig(
            os.path.join(output_dir, f"scree_plot_{unique_sys}.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 2. Loading plot with circle
        plt.figure(figsize=(12, 8))
        loadings = pca.components_
        loading_matrix = loadings[:2].T  # Take first two PCs

        # Plot arrows
        for i, (feature, loading) in enumerate(zip(feature_cols, loading_matrix)):
            plt.arrow(
                0,
                0,
                loading[0],
                loading[1],
                head_width=0.02,
                head_length=0.02,
                fc="blue",
                ec="blue",
            )
            plt.text(
                loading[0] * 1.1, loading[1] * 1.1, feature, ha="center", va="center"
            )

        # Add circle
        circle = plt.Circle((0, 0), 1, fill=False, linestyle="--", color="gray")
        plt.gca().add_patch(circle)
        plt.axis("equal")
        plt.xlabel(f"PC1 ({var_ratio[0]:.2%} variance explained)")
        plt.ylabel(f"PC2 ({var_ratio[1]:.2%} variance explained)")
        plt.title(f"PCA Loading Plot (systems: {unique_sys})")
        plt.grid(True)
        plt.savefig(
            os.path.join(output_dir, f"loading_plot_{unique_sys}.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 3. Score plot
        plt.figure(figsize=(10, 8))

        # Get unique systems, BCPs, and EEFs
        unique_systems = df["SYSTEM"].unique()
        unique_bcps = df["BCP_CROSSING"].unique()
        unique_eefs = sorted(df["EEF"].unique())  # sort EEF values

        # Create marker and color mappings
        markers = ["o", "s", "^", "v", "D", "p", "h8"]  # add more if needed
        system_markers = dict(zip(unique_systems, markers[: len(unique_systems)]))

        # Use a colormap suitable for the number of BCP crossings (fill colors)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_bcps)))
        bcp_colors = dict(zip(unique_bcps, colors))

        # Create edge color mapping for EEF values
        edge_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_eefs)))
        eef_colors = dict(zip(unique_eefs, edge_colors))

        # Plot each point with appropriate marker, fill color, and edge color
        for system in unique_systems:
            for bcp in unique_bcps:
                for eef in unique_eefs:
                    mask = (
                        (df["SYSTEM"] == system)
                        & (df["BCP_CROSSING"] == bcp)
                        & (df["EEF"] == eef)
                    )
                    if mask.any():  # only plot if this combination exists
                        plt.scatter(
                            X_pca[mask, 0],
                            X_pca[mask, 1],
                            marker=system_markers[system],
                            c=[bcp_colors[bcp]],
                            s=100,  # adjust point size as needed
                            linewidth=1.5,  # fixed edge width
                            edgecolor=eef_colors[eef],
                        )

        # Format the variance explained percentages properly for LaTeX
        pc1_var = f"{var_ratio[0]*100:.1f}\\%"
        pc2_var = f"{var_ratio[1]*100:.1f}\\%"

        plt.xlabel(f"PC1 ({pc1_var} variance explained)")
        plt.ylabel(f"PC2 ({pc2_var} variance explained)")
        plt.title(f"PCA Score Plot (systems: {unique_sys})")
        plt.grid(True)

        # Create custom legends
        # Legend for BCP crossings (fill colors)
        bcp_legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                label=bcp,
                markersize=10,
            )
            for bcp, color in bcp_colors.items()
        ]

        # Legend for systems (markers)
        system_legend_elements = [
            plt.Line2D(
                [0], [0], marker=marker, color="gray", label=system, markersize=10
            )
            for system, marker in system_markers.items()
        ]

        # Legend for EEF values (edge colors)
        eef_legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="w",
                label=f"{int(eef)}",
                markersize=10,
                markeredgewidth=2,
                markeredgecolor=color,
            )
            for eef, color in eef_colors.items()
        ]

        # Add three separate legends
        first_legend = plt.legend(
            handles=bcp_legend_elements,
            title="BCP Crossings",
            bbox_to_anchor=(1.2, 1),
            loc="upper right",
        )
        plt.gca().add_artist(first_legend)

        second_legend = plt.legend(
            handles=system_legend_elements,
            title="Systems",
            bbox_to_anchor=(1.2, 0.15),
            loc="lower right",
        )
        plt.gca().add_artist(second_legend)

        plt.legend(
            handles=eef_legend_elements,
            title="EEF Values",
            bbox_to_anchor=(1.2, 0.0),
            loc="lower right",
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"score_plot_{unique_sys}.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 4. Loadings heatmap
        plt.figure(figsize=(12, 8))
        num_pcs = min(5, len(loadings))  # Show first 5 PCs or all if less than 5
        loadings_df = pd.DataFrame(
            loadings[:num_pcs].T,
            columns=[f"PC{i+1}" for i in range(num_pcs)],
            index=feature_cols,
        )
        sns.heatmap(loadings_df, cmap="RdBu_r", center=0, annot=True, fmt=".2f")
        plt.title(f"PCA Loadings Heatmap (systems: {unique_sys})")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"loadings_heatmap_{unique_sys}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Save loadings to CSV
        loadings_df.to_csv(os.path.join(output_dir, f"pca_loadings_{unique_sys}.csv"))

        # Create DataFrame with transformed features
        pca_df = pd.DataFrame(
            X_pca, columns=[f"PC{i+1}" for i in range(pca.n_components_)], index=X.index
        )

        # Reset matplotlib parameters to default
        plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif"})

        return pca, pca_df

    def perform_boruta(df: pd.DataFrame,
                    target_col: int,
                    output_dir: str,
                    max_iter: int = 100,
                    perc: int = 100,
                    alpha: float = 0.05,
                    random_state: int = 42) -> pd.DataFrame:
        """
        Performs Boruta feature selection with adjustable sensitivity parameters.
        
        Args:
            df Input DataFrame
            target_col: Index of target column among numeric columns
            max_iter: Maximum number of iterations for Boruta algorithm
            perc: Percentile of shadow features maximal importance for comparison
                Higher values = more strict feature selection (default 100)
            alpha: P-value threshold for feature importance (default 0.05)
                Lower values = more strict feature selection
            random_state: Random state for reproducibility
        
        Returns:
            DataFrame with selected features and target column
        """
        # Get numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        feature_cols = [col for i, col in enumerate(numeric_df.columns) if i != target_col]
        target_name = numeric_df.columns[target_col]
        
        unique_sys = "+".join(df["SYSTEM"].unique().tolist())
        if any("{A" in c or "{B" in c for c in df.columns):
            unique_sys += "_spin"
        if any("theta" in c or "phi" in c for c in df.columns):
            unique_sys += "_directionality"
        
        print(f"Performing Boruta feature selection for target column: {target_name}")
        
        # Prepare X and y
        X = numeric_df[feature_cols]
        y = numeric_df[target_name]
        
        # Initialize Random Forest classifier
        rf = RandomForestRegressor(n_jobs=-1, random_state=random_state)
        
        # Initialize and run Boruta
        boruta = BorutaPy(rf, n_estimators='auto',
                        max_iter=max_iter,
                        perc=perc,
                        alpha=alpha,
                        random_state=random_state)
        
        # Fit Boruta
        boruta.fit(X.values, y.values)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # Create detailed results file
        with open(os.path.join(output_dir, f'boruta_results_{unique_sys}.txt'), 'w') as f:
            # Write parameters
            f.write("Boruta Feature Selection Analysis\n")
            f.write("================================\n\n")
            f.write("Parameters:\n")
            f.write(f"- Maximum iterations: {max_iter}\n")
            f.write(f"- Percentile threshold: {perc}\n")
            f.write(f"- Significance level (alpha): {alpha}\n")
            f.write(f"- Target variable: {target_name}\n\n")
            
            # Write feature selection results
            selected_features = X.columns[boruta.support_].tolist()
            tentative_features = X.columns[boruta.support_weak_].tolist()
            rejected_features = [feat for feat in X.columns 
                            if feat not in selected_features + tentative_features]
            
            f.write("Selected Features:\n")
            for feat in selected_features:
                f.write(f"- {feat}\n")
            f.write(f"\nTotal selected features: {len(selected_features)}\n\n")
            
            f.write("Tentative Features:\n")
            for feat in tentative_features:
                f.write(f"- {feat}\n")
            f.write(f"\nTotal tentative features: {len(tentative_features)}\n\n")
            
            f.write("Rejected Features:\n")
            for feat in rejected_features:
                f.write(f"- {feat}\n")
            f.write(f"\nTotal rejected features: {len(rejected_features)}\n\n")
            
            # Write feature rankings with importance scores
            f.write("Feature Rankings and Importance Scores:\n")
            f.write("(Rankings: lower = more important, 1 is best)\n")
            f.write("(Decision: 'Confirmed', 'Tentative', or 'Rejected')\n\n")
            
            # Get feature importance scores
            ranks = pd.Series(boruta.ranking_, index=X.columns)
            
            # Create decision mapping
            decision_map = {}
            for feat in X.columns:
                if feat in selected_features:
                    decision_map[feat] = 'Confirmed'
                elif feat in tentative_features:
                    decision_map[feat] = 'Tentative'
                else:
                    decision_map[feat] = 'Rejected'
            
            # Sort by ranking and write
            for feat, rank in ranks.sort_values().items():
                f.write(f"Feature: {feat}\n")
                f.write(f"- Ranking: {rank}\n")
                f.write(f"- Decision: {decision_map[feat]}\n\n")
        
        # First create list of columns we want to keep
        columns_to_keep = ['SYSTEM', 'BCP_CROSSING', target_name]
        if 'EEF' not in selected_features + tentative_features:
            columns_to_keep.append('EEF')
        
        # Add the features and target
        columns_to_keep.extend(selected_features +
                            tentative_features)
        
        # Create the new DataFrame in one operation
        selected_tentative_df = df[columns_to_keep].copy()
        
        # First create list of columns we want to keep
        columns_to_keep = ['SYSTEM', 'BCP_CROSSING', target_name]
        if 'EEF' not in selected_features:
            columns_to_keep.append('EEF')
        
        # Add the features and target
        columns_to_keep.extend(selected_features)
        
        # Create the new DataFrame in one operation
        selected_df = df[columns_to_keep].copy()
        
        # Save feature lists to separate files
        pd.Series(selected_features).to_csv(
            os.path.join(output_dir, f'selected_features_{unique_sys}.csv'), index=False)
        pd.Series(tentative_features).to_csv(
            os.path.join(output_dir, f'tentative_features_{unique_sys}.csv'), index=False)
        pd.Series(rejected_features).to_csv(
            os.path.join(output_dir, f'rejected_features_{unique_sys}.csv'), index=False)
        
        # save selected and selected_tentative DataFrames to CSV
        selected_df.to_csv(os.path.join(output_dir, f'selected_df_{unique_sys}.csv'), index=False)
        selected_tentative_df.to_csv(os.path.join(output_dir, f'selected_tentative_df_{unique_sys}.csv'), index=False)
        
        return selected_df, selected_tentative_df, boruta
    
    ### END HELPER FUNCTIONS ###

    # First, convert the data to a pandas DataFrame
    # Get keys common to all dictionaries
    all_keys = set(list(cp_data[0].keys()))
    for d in cp_data[1:]:
        all_keys &= set(list(d.keys()))

    cp_data_by_col = {k: [d[k] for d in cp_data] for k in all_keys}

    df_full = pd.DataFrame(cp_data_by_col)

    # First, we'll add some extra columns to the data:
    # - "EEF" which is -1, 0, 1 if the "JOB_NAME" contains "revEEF", "noEEF", "origEEF" respectively
    # - "sys" which is the system number extracted from the "JOB_NAME", the first underscore-separated value

    df_full["EEF"] = df_full["JOB_NAME"].apply(
        lambda x: -1 if "revEEF" in x else (0 if "noEEF" in x else 1)
    )
    df_full["SYSTEM"] = df_full["JOB_NAME"].apply(lambda x: x.split("_")[0])

    # get unique SYSTEM values
    sys_list = df_full["SYSTEM"].unique().tolist()
    sys_whitelists = [[sys] for sys in sys_list] + [sys_list]
    directionality_spin = [False, True]
    alpha_list = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    perc_list = [100, 95, 90, 85, 80]
    
    alpha_list = [0.01, 0.025, 0.05, 0.1]
    perc_list = [100, 95, 90]
    
    target_col = '$\\Delta E_{{\\rm{{{TS}}}}}$'
    
    for sys_whitelist in sys_whitelists:
        for use_theta_phi_crossings in directionality_spin:
            for use_spin_crossings in directionality_spin:
                print(f"\nsys_whitelist: {sys_whitelist}, use_theta_phi_crossings: {use_theta_phi_crossings}, use_spin_crossings: {use_spin_crossings}")
                
                df, rows = prepare_bcp_crossing_df(
                    df_full,
                    "H47-O39 distance",
                    use_theta_phi_crossings=use_theta_phi_crossings,
                    use_spin_crossings=use_spin_crossings,
                    sys_whitelist=sys_whitelist
                )
                
                
                # Get list of unique SYSTEM values
                systems = "_".join(df["SYSTEM"].unique().tolist())
                eefs = df["EEF"].unique().tolist()
                spin_string = "__spin" if use_spin_crossings else ""
                directionality_string = "__directionality" if use_theta_phi_crossings else ""

                system_string = f"{systems}{spin_string}{directionality_string}"
                
                # save the DataFrame to CSV
                # make dirs
                os.makedirs(f"{output_dir}/{system_string}", exist_ok=True)
                df.to_csv(f"{output_dir}/{system_string}/{system_string}_bcp_crossing_data.csv", index=False)
                
                print(f"system_string: {system_string}\neefs: {eefs}")

                # First with all features
                create_correlation_matrix(
                    df, f"{output_dir}/{system_string}/all_features/correlation_analysis"
                )
                pca, pca_df = perform_pca(
                    df, [target_col], f"{output_dir}/{system_string}/all_features/pca_analysis"
                )
                
                do_break = False
                check_num_features = 7
                if use_spin_crossings and use_spin_delta_e_r:
                    check_num_features += 4
                # Perform Boruta feature selection
                for pi, perc in enumerate(perc_list):
                    for ai, alpha in enumerate(alpha_list):
                        print(f"System: {system_string}, perc: {perc}, alpha: {alpha}")
                        
                        boruta_string = f"perc_{perc}_alpha_{alpha}"
                        
                        selected_df, selected_tentative_df, boruta = perform_boruta(
                            df, 1, f"{output_dir}/{system_string}/boruta_{boruta_string}", perc=perc, alpha=alpha
                        )
                        # continue if all independent variable features are rejected
                        if len(selected_df.columns) > 4:
                            create_correlation_matrix(
                                selected_df, f"{output_dir}/{system_string}/boruta_{boruta_string}/selected_correlation_analysis"
                            )
                            pca, pca_df = perform_pca(
                                selected_df, [target_col], f"{output_dir}/{system_string}/boruta_{boruta_string}/selected_pca_analysis"
                            )
                            
                        if len(selected_tentative_df.columns) > len(selected_df.columns):
                            create_correlation_matrix(
                                selected_tentative_df, f"{output_dir}/{system_string}/boruta_{boruta_string}/selected_tentative_correlation_analysis"
                            )
                            pca, pca_df = perform_pca(
                                selected_tentative_df, [target_col], f"{output_dir}/{system_string}/boruta_{boruta_string}/selected_tentative_pca_analysis"
                            )
                        
                        if len(selected_df.columns) > check_num_features:
                            do_break = True
                            break
                    if do_break:
                        break

    return


if __name__ == "__main__":
    if csv_file_paths:
        log_print("Performing statistical analysis on provided CSV files...")
        cp_data = []
        for csv_file_path in csv_file_paths:
            cp_data.extend(read_simple_csv(csv_file_path))
        output_dir = os.path.dirname(csv_file_paths[0]) + "/analysis"
        statistical_analysis(
            cp_data, output_dir
        )
    elif restart_dill_paths and len(restart_dill_paths) > 0:
        log_print("Performing post-processing on provided restart dill files...")
        for restart_dill_path, atom_pairs in zip(restart_dill_paths, atom_pairs_list):
            test_post_processing_multiple_jobs(
                restart_dill_path, atom_pairs, unrestricted=False
            )
    else:
        log_print("Running ADF NEB BCP analysis on provided job paths...")
        for job_path, atom_pairs in zip(ams_job_paths, atom_pairs_list):
            main(job_path, atom_pairs)
