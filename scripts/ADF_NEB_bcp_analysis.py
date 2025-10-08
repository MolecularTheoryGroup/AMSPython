# from scm.plams import *
import os
import csv
import subprocess
from math import sqrt, atan, floor, ceil
import numpy as np
from scipy.interpolate import interp1d, make_interp_spline
import matplotlib.pyplot as plt
import matplotlib.cm as mcm # Added for colormap access
from datetime import datetime
import re
import warnings  # To warn about potential issues
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

# Run on one job at a time. It was initially intended to be able to run on multiple jobs, but in the course of development that has been temporarily broken.

# Define the path to the AMS job file (`path/to/job.ams`), or if you don't have an ams file, use
# the path to the ams.rkf result file. Set the dill and csv paths to be empty in order to have the script use the AMS job file input.
# ams_job_paths = [
#     '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/his/His_propane_near_TS.ams']
# ams_job_paths = ['/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/cys/Field_n01/Cys_near_TS_NEB_n01/Cys_propane_near_TS_n01.ams',
#                 '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/cys/Field_p01/Cys_near_TS_NEB_p01/Cys_propane_near_TS_p01.ams']
# ams_job_paths = ['/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/cys/No_Field/Full_reaction/Cys_propane_NEB_NF.ams',
#                 '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/cys/Field_p01/Full_reaction/Cys_propane_NEB_p01.ams',
#                 '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/cys/Field_n01/full_reaction/Cys_propane_NEB_n01.ams']

# ams_job_paths = ['/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/cys/No_Field/Full_reaction/Cys_propane_NEB_NF.ams',
#                 '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/cys/Field_p01/Full_reaction/Cys_propane_NEB_p01.ams',
#                 '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/cys/Field_n01/full_reaction/Cys_propane_NEB_n01.ams']

ams_job_paths = [
    '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/v2/cys/n01/Cys_propane_NEB_n01.ams',
    '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/v2/cys/p01/Cys_propane_NEB_p01.ams'
]

# To rerun on a previously processed file, set the restart_dill_path to the path of the dill file in the working directory of the previous run. Otherwise, set to None, False, or ''. Set the csv paths to be an empty list if you want the script to use the dill file input.
# restart_dill_paths = [
#     '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/cys/Near_TS_plams_workdir.003/Cys_propane_near_TS/Cys_propane_near_TS.dill',
#     '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/cys/Near_TS_plams_workdir.003/Cys_propane_near_TS_n01/Cys_propane_near_TS_n01.dill',
#     '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/cys/Near_TS_plams_workdir.003/Cys_propane_near_TS_p01/Cys_propane_near_TS_p01.dill'
# ]
# restart_dill_paths = [
#     '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/cys/Full_reaction_plams_workdir.004/Cys_propane_NEB_NF/Cys_propane_NEB_NF.dill',
#     '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/cys/Full_reaction_plams_workdir.004/Cys_propane_NEB_p01/Cys_propane_NEB_p01.dill',
#     '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/cys/Full_reaction_plams_workdir.004/Cys_propane_NEB_n01/Cys_propane_NEB_n01.dill'
# ]
# restart_dill_paths = [
#     '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/his/plams_workdir.002/His_propane_near_TS/His_propane_near_TS.dill',
# ]
restart_dill_paths = [
    '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/v2/cys_his_tyr_nf_workdir/Cys_propane_NEB_NF/Cys_propane_NEB_NF.dill',
    '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/v2/cys_his_tyr_nf_workdir/His_propane_NEB_NF/His_propane_NEB_NF.dill',
    '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/v2/cys_his_tyr_nf_workdir/Tyr_propane_NEB_NF/Tyr_propane_NEB_NF.dill',
    '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/v2/cys_plus_minus_workdir.002/Cys_propane_NEB_n01/Cys_propane_NEB_n01.dill',
    '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/v2/cys_plus_minus_workdir.002/Cys_propane_NEB_p01/Cys_propane_NEB_p01.dill',
]
unrestricted_calculation = True

# Define paths to previously created cp data CSV files in order to do statistical analysis.
csv_file_paths = [
    # '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/cys/Full_reaction_plams_workdir.004/Cys_propane_NEB_n01/Cys_propane_NEB_n01_cp_info.csv',
    # '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/cys/Full_reaction_plams_workdir.004/Cys_propane_NEB_p01/Cys_propane_NEB_p01_cp_info.csv',
    # '/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/cys/Full_reaction_plams_workdir.004/Cys_propane_NEB_NF/Cys_propane_NEB_NF_cp_info.csv'
]

# You can control the starting and ending NEB image number to include in the analysis here.
# 0 means the first image in the NEB, 1 means the second image, etc.
start_image = 0
end_image = -1  # -1 means the last image in the NEB

# Define atom pairs (pairs of atom numbers with associated descriptions) for which to extract bond critical point information.
# One list for each input file defined above
atom_pairs_list = (  # one-based indices, same as shown in AMSView
    # { # Chorismate Mutase
    #     (9, 12): "Breaking bond",  # breaking C-O bond
    #     (12, 13): "- → =", # breaking C-O O's remaining (single -> double) C-O bond
    #     (7, 9): "Ring to OH - → ≃", # breaking C-O C's C-C bond towards OH
    #     (9, 11): "Ring - → ≃", # breaking C-O C's C-C bond towards CO2
    #     (1, 14): "Forming bond", # forming C-C bond
    #     (13, 14): "C3O3 = → -", # forming C-C other (double -> single) C-C bond
    #     (1, 15): "Ring CO2 C", # forming C-C ring C-C bond with CO2 C
    #     (1, 3): "Ring ≃ → -", # forming C-C ring C-C bond with (aromatic -> single) ring C
    #     (1, 11): "Ring to OH ≃ → -", # forming C-C ring C-C bond with (aromatic -> single) other ring C
    # },
    # { # CP450 Heme His
    #     (40, 47): "Breaking bond",  # CH
    #     (47, 39): "Forming bond",  # OH
    #     (1, 39): "",  # FeO
    #     (1, 55): "Fe-Ligand",  # Fe-amino acid His
    #     (1, 2): "",  # FeN
    #     (1, 3): "",  # FeN
    #     (1, 4): "",  # FeN
    #     (1, 5): "",  # FeN
    # },
    # { # CP450 Heme Cys
    #     (40, 47): "Breaking bond",  # CH
    #     (47, 39): "Forming bond",  # OH
    #     (1, 39): "",  # FeO
    #     (1, 53): "Fe-Ligand",  # Fe-amino acid Cys
    #     (1, 2): "",  # FeN
    #     (1, 3): "",  # FeN
    #     (1, 4): "",  # FeN
    #     (1, 5): "",  # FeN
    # },
    { # CP450 Heme Cys NF v2 
        (44, 46): "Breaking bond",  # CH
        (46, 38): "Forming bond",  # OH
        (1, 38): "",  # FeO
        (1, 43): "Fe-Ligand",  # Fe-amino acid Cys
        (1, 2): "",  # FeN
        (1, 3): "",  # FeN
        (1, 4): "",  # FeN
        (1, 5): "",  # FeN
    },
    { # CP450 Heme His NF v2 
        (78, 77): "Breaking bond",  # CH
        (77, 38): "Forming bond",  # OH
        (1, 38): "",  # FeO
        (1, 44): "Fe-Ligand",  # Fe-amino acid Cys
        (1, 2): "",  # FeN
        (1, 3): "",  # FeN
        (1, 4): "",  # FeN
        (1, 5): "",  # FeN
    },
    { # CP450 Heme Tyr NF v2 
        (55, 54): "Breaking bond",  # CH
        (54, 38): "Forming bond",  # OH
        (1, 38): "",  # FeO
        (1, 40): "Fe-Ligand",  # Fe-amino acid Cys
        (1, 2): "",  # FeN
        (1, 3): "",  # FeN
        (1, 4): "",  # FeN
        (1, 5): "",  # FeN
    },
    { # CP450 Heme Cys n01 v2 
        (45, 44): "Breaking bond",  # CH
        (44, 38): "Forming bond",  # OH
        (1, 38): "",  # FeO
        (1, 43): "Fe-Ligand",  # Fe-amino acid Cys
        (1, 2): "",  # FeN
        (1, 3): "",  # FeN
        (1, 4): "",  # FeN
        (1, 5): "",  # FeN
    },
    { # CP450 Heme Cys p01 v2 
        (45, 44): "Breaking bond",  # CH
        (44, 38): "Forming bond",  # OH
        (1, 38): "",  # FeO
        (1, 43): "Fe-Ligand",  # Fe-amino acid Cys
        (1, 2): "",  # FeN
        (1, 3): "",  # FeN
        (1, 4): "",  # FeN
        (1, 5): "",  # FeN
    },
    # { # CP450 Heme Tyr
    #     (40, 47): "Breaking bond",  # CH
    #     (47, 39): "Forming bond",  # OH
    #     (1, 39): "",  # FeO
    #     (1, 50): "Fe-Ligand",  # Fe-amino acid Tyr
    #     (1, 2): "",  # FeN
    #     (1, 3): "",  # FeN
    #     (1, 4): "",  # FeN
    #     (1, 5): "",  # FeN
    # },
)

# Index of atom pair for which to print the bond distance of each image to be run. (or set to None to not print any distances)
atom_pair_for_bond_distance_printout = None #(46, 38)

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
user_eef = (0.0, 0.0, 0.01)
# Need to convert electric field magnitude units. In the NEB rkf file, they're Ha/(e bohr), but in the
# new jobs they need to be V/Angstrom. The conversion factor is 51.4220861908324.
eef_conversion_factor = 51.4220861908324
# Define the EEF pairs
# eef_pairs = [("origEEF", eef_conversion_factor), ("revEEF", -eef_conversion_factor), ("noEEF", 0)]
eef_pairs = [("origEEF", eef_conversion_factor)]
##### end EEF Settings #####

##### Extra interpolated single point settings #####
# To get better resultion around the transition state, we'll identify the TS image (highest energy image)
# and create additional images between it and the adjacent images, using a linear interpolation of the
# coordinates of the adjacent images. Here, you specify how many extra images to add on *each* side of the TS image.
num_extra_images = 3
# This then determines how many images to the left/right of the TS image to create. `num_extra_images` images will be created between each adjacent pair of images.
# So "1" will result in `num_extra_images` images being added only between the TS image and its adjacent images,
# while "3" will add `num_extra_images` between each image pair starting 3 images before the TS, etc.
extra_images_num_adjacent_images = 2
# Set `num_extra_images` to 0 to disable this feature.
##### end Extra interpolated single point settings #####

##### Plot settings #####
smooth_derivatives = True
# Now define the x and y properties for generated plots:

# in addition to any of the properties that appear as column headings in the output CSV file,
# you may specify the following properties to plot as x or y axis values:
#
# "NEB image"         (i.e. the image number) which will likely be changed by the nubmer of extra images added)
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
# Each value in the below dictionary is a list of properties to plot on the y axis, with one plot
# made per y-property.
# If the list contains a list (a sublist) of properties, then those properties will be
# plotted on the same plot.
# (For d/dx calculations, you can use the smoothed (polynomial fit) by includeing (smoothed) after the "d/dx" in the property name).
combined_plots_y_prop_lists = {
    "Rho": ["Molecular bond energy", "Rho"],
    "Rho d/dx": ["Molecular bond energy", "Rho d/dx"],
    "Rho d/dx (smoothed)": ["Molecular bond energy", "Rho d/dx (smoothed)"],
    "Angles": ["Molecular bond energy", "Theta", "Phi"],
    "Angles overlay": ["Molecular bond energy", ["Theta", "Phi"]],
    "Angles d/dx": ["Molecular bond energy", "Theta d/dx", "Phi d/dx"],
    "Angles overlay d/dx": ["Molecular bond energy", ["Theta d/dx", "Phi d/dx"]],
    "Angles d/dx (smoothed)": ["Molecular bond energy", "Theta d/dx (smoothed)", "Phi d/dx (smoothed)"],
    "Angles overlay d/dx (smoothed)": ["Molecular bond energy", ["Theta d/dx (smoothed)", "Phi d/dx (smoothed)"]],
}

# Specify properties to be used as the x axis (independent variable) for the plots.
# For each x-axis property specified, a full set of plots will be generated for each y property.
# (Add " (reverse)" to reverse the x-axis direction)
plot_x_prop_list = [
    "H46-O38 distance (reverse)",
    "NEB image",
]

plot_x_prop_lists = [
    [
        "H46-O38 distance (reverse)",
        "NEB image",
    ],
    [
        "H77-O38 distance (reverse)",
        "NEB image",
    ],
    [
        "H54-O38 distance (reverse)",
        "NEB image",
    ],
    [
        "H44-O38 distance (reverse)",
        "NEB image",
    ],
    [
        "H44-O38 distance (reverse)",
        "NEB image",
    ],
]
##### end Plot settings #####

##### Spin density CP search settings #####
# BCP locations in A/B spin densities are not the same as in the total density, so we do a basic
# search for the minimum gradient magnitude point in a 3D grid around the total density CP location
# for each BCP in each A/B spin density.
# These parameters control that process.
# Number of check points in each dimension. Increasing this value will result in a better approximation of
# The spin A/B CP locations, but will increase densf runtime.
num_check_points = 15
# Fraction of the distance between the two atoms. Increasing this value will result in a search being
# done over a larger region around the total-density CP location, and will increase the spacing between
# the check points. If too small, the search grid may not include the true spin A/B CP locations.
check_point_grid_extent_fraction = 0.02
##### end Spin density CP search settings #####

# List of properties to include in reduced output CSV file
reduced_csv_keys = [
    r"ATOMS",
    r"NEB image",
    r".*distance.*",
    r"JOB_NAME",
    r"Molecular bond energy",
    r"CP #",
    r"RANK",
    r"SIGNATURE",
    r"^Rho(_[AB])?",
    # "Rho_A",
    # "Rho_B",
    r"Theta.*",
    # "Theta_A",
    # "Theta_B",
    r"Phi.*",
    # "Phi_A",
    # "Phi_B",
    r"CP COORDINATES.*",
    # "CP COORDINATES_A",
    # "CP COORDINATES_B",
    r"EIGENVALUES.*",
    # "EIGENVALUES_A",
    # "EIGENVALUES_B",
    # "EIGENVALUES OF HESSIAN MATRIX",
    # "EIGENVALUES OF HESSIAN MATRIX_A",
    # "EIGENVALUES OF HESSIAN MATRIX_B",
    r"EIGENVECTORS.*",
    # "EIGENVECTORS_A",
    # "EIGENVECTORS_B",
    # "EIGENVECTORS (ORTHONORMAL) OF HESSIAN MATRIX (COLUMNS)",
    # "EIGENVECTORS (ORTHONORMAL) OF HESSIAN MATRIX (COLUMNS)_A",
    # "EIGENVECTORS (ORTHONORMAL) OF HESSIAN MATRIX (COLUMNS)_B"
]

reduced_csv_rename_key_map = {
    "NEB image": "Reaction coordinate"
}

########################################################################################
# END OF USER SETTINGS
########################################################################################

num_check_points_total = num_check_points**3

# Get the number of CPU cores
num_cores = ceil(os.cpu_count() / 2)

########################################################################################
# Step 0: define helper functions
########################################################################################


def compute_bspline_derivative(x, y, num_points=200, k=21, original_x_points=False):
    """
    Computes a smoothed first-order derivative (dy/dx) using resampling, 
    linear interpolation, and B-spline differentiation, evaluated at the 
    original unique x-coordinates.

    Parameters:
    x (array-like): Original x-coordinates.
    y (array-like): Original y-coordinates, corresponding to x.
    num_points (int): Number of points for intermediate resampling. 
                      Used for fitting the spline.
    k (int): Degree of the B-spline (e.g., 3 for cubic). Must be 
               less than the number of resampled points and less than 
               the number of unique original x points.

    Returns:
    tuple: (x_original_unique, dy_dx_at_original_x) - Unique original x values 
           (sorted) and corresponding smoothed derivative values.
           Returns (None, None) if input data is insufficient.
    """
    x = np.array(x)
    y = np.array(y)

    if len(x) < 2 or len(y) < 2:
        print("Warning: Input arrays must have at least two points.")
        return None, None
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if num_points <= k:
        raise ValueError(
            f"Number of resampled points ({num_points}) must be greater than spline degree k ({k}).")

    # Ensure x is sorted for interpolation/splines
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Handle duplicate x values by averaging corresponding y values
    # Keep track of the unique x values, as these are our target evaluation points
    x_original_unique, unique_indices = np.unique(x_sorted, return_index=True)
    if len(x_original_unique) < len(x_sorted):
        print("Warning: Duplicate x values found. Averaging corresponding y values.")
        y_original_unique = np.array(
            [np.mean(y_sorted[x_sorted == ux]) for ux in x_original_unique])
    else:
        y_original_unique = y_sorted  # No duplicates, use sorted y directly

    # Need at least k+1 unique points for spline of degree k
    # if len(x_original_unique) <= k:
    #      print(f"Warning: Not enough unique points ({len(x_original_unique)}) for spline degree {k}.")
    #      return None, None

    # --- Intermediate steps for smoothing ---
    # 1. Define evenly spaced x-values for resampling (for fitting the spline)
    x_resample = np.linspace(np.min(x_original_unique),
                             np.max(x_original_unique), num=num_points)

    # 2. Resample y using linear interpolation based on the unique original points
    #    Using interp1d requires at least 2 points. Handled by initial checks.
    f_interp = interp1d(x_original_unique, y_original_unique,
                        kind='linear', fill_value="extrapolate")
    y_resample = f_interp(x_resample)

    # 3. B-spline fit of the *resampled* data
    #    make_interp_spline requires k+1 points. Handled by initial checks.
    spl = make_interp_spline(x_resample, y_resample, k=k)

    # 4. Compute the derivative of the B-spline
    spl_deriv = spl.derivative(nu=1)  # nu=1 for first derivative

    # --- Final Evaluation ---
    # 5. Evaluate the derivative spline *at the original unique x-values*
    if original_x_points:
        dy_dx_at_original_x = spl_deriv(x_original_unique)
        return x_original_unique, dy_dx_at_original_x
    else:
        dy_dx_at_resampled_x = spl_deriv(x_resample)
        return x_resample, dy_dx_at_resampled_x


def compute_polynomial_derivative(
    x,
    y,
    n_fine=100,
    n_validation_factor=5,
    min_order=3,
    max_order=9,
    output_at='original',
    # Percentage of data range (centered) to use for validation error
    interior_percent=96,
    overfitting_tolerance=1e-4
):
    """
    Computes a smoothed derivative using polynomial fitting with validation based
    on maximum absolute error over the interior range.

    Fits polynomials of increasing order to finely resampled data. Uses the
    maximum absolute error on an even finer resampling (excluding boundary 
    points) to select the best order, avoiding overfitting. The derivative 
    of the best-fit polynomial is returned.

    Parameters:
    x (array-like): Original x-coordinates.
    y (array-like): Original y-coordinates, corresponding to x.
    n_fine (int): Number of points for the primary fitting dataset (resampled).
    n_validation_factor (int): Multiplier for n_fine to determine the number
                               of points in the validation dataset 
                               (n_validation = n_fine * n_validation_factor).
    min_order (int): Minimum polynomial order to try.
    max_order (int): Maximum polynomial order to try. Must be less than n_fine.
    output_at (str): Specifies the x-coordinates for the output derivative.
                       'original': Output derivative at the unique original x values.
                       'fine': Output derivative at the n_fine resampled x values.
    interior_percent (float): The percentage (0-100) of the central data range 
                              of the validation set to use for calculating the 
                              maximum absolute error metric. E.g., 96 means exclude 
                              2% from each end.
    overfitting_tolerance (float): Tolerance for early stopping based on the max error. 
                                   If the max interior error increases by more than 
                                   this fraction relative to the minimum error found 
                                   so far, fitting stops. Set <= 0 to disable.

    Returns:
    tuple: (x_output, derivative_values) 
           - x_output: x-coordinates where the derivative was evaluated.
           - derivative_values: Corresponding smoothed derivative values.
           Returns (None, None) if fitting fails or input is invalid.
    """
    x = np.array(x)
    y = np.array(y)
    n_validation = n_fine * n_validation_factor

    # --- Input Validation and Preprocessing ---
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) < 2:
        warnings.warn(
            "Input arrays must have at least two points. Cannot compute derivative.")
        return None, None
    if max_order >= n_fine:
        raise ValueError(
            f"max_order ({max_order}) must be less than n_fine ({n_fine}).")
    if min_order < 0:
        raise ValueError("min_order cannot be negative.")
    if not (0 < interior_percent <= 100):
        raise ValueError(
            "interior_percent must be between 0 (exclusive) and 100 (inclusive).")
    if output_at not in ['original', 'fine']:
        raise ValueError("output_at must be 'original' or 'fine'.")

    # Sort and handle duplicates
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    x_unique, unique_indices = np.unique(x_sorted, return_index=True)
    if len(x_unique) < len(x_sorted):
        warnings.warn(
            "Duplicate x values found. Averaging corresponding y values.")
        y_unique = np.array([np.mean(y_sorted[x_sorted == ux])
                            for ux in x_unique])
    else:
        y_unique = y_sorted[unique_indices]  # Use sorted unique y directly

    if len(x_unique) < 2:
        warnings.warn(
            "Need at least 2 unique points after handling duplicates. Cannot compute derivative.")
        return None, None
    if len(x_unique) <= min_order:
        warnings.warn(
            f"Need at least {min_order + 1} unique points for minimum polynomial order {min_order}, found {len(x_unique)}. Cannot proceed.")
        return None, None

    # --- Generate Interpolation Function ---
    try:
        interp_func = interp1d(
            x_unique, y_unique, kind='linear', fill_value="extrapolate")
    except ValueError as e:
        warnings.warn(f"Could not create interpolation function: {e}")
        return None, None

    # --- Generate Fitting and Validation Data ---
    x_min, x_max = np.min(x_unique), np.max(x_unique)
    x_fine = np.linspace(x_min, x_max, n_fine)
    y_fine = interp_func(x_fine)

    x_validation = np.linspace(x_min, x_max, n_validation)
    y_validation = interp_func(x_validation)

    # --- Determine Interior Indices for Validation ---
    if interior_percent == 100:
        idx_start = 0
        idx_end = n_validation - 1
    else:
        margin = (100.0 - interior_percent) / 2.0 / 100.0
        idx_start = int(np.floor(n_validation * margin))
        idx_end = int(np.ceil(n_validation * (1.0 - margin))) - 1
        # Ensure indices are valid and range is sensible
        idx_start = max(0, idx_start)
        idx_end = min(n_validation - 1, idx_end)
        if idx_start >= idx_end:
            warnings.warn(
                f"Interior range calculation resulted in non-positive length ({idx_start} to {idx_end}). Using full range.")
            idx_start = 0
            idx_end = n_validation - 1

    # print(f"Using validation points from index {idx_start} to {idx_end} (inclusive) for max error calculation.")

    # --- Iterative Polynomial Fitting and Validation ---
    best_order = -1
    min_validation_max_error = np.inf  # Now tracking minimum of the maximum errors
    best_coeffs = None

    # print(f"Fitting polynomial orders {min_order} to {max_order}...")

    for order in range(min_order, max_order + 1):
        if n_fine <= order:
            warnings.warn(
                f"Skipping order {order}: n_fine ({n_fine}) is not greater than order.")
            continue

        # Fit polynomial to the 'fine' dataset
        try:
            coeffs = np.polyfit(x_fine, y_fine, deg=order)
        except (np.linalg.LinAlgError, ValueError) as e:
            warnings.warn(
                f"Polyfit failed for order {order}: {e}. Stopping search.")
            break

        # Evaluate on the 'validation' dataset
        poly = np.poly1d(coeffs)
        y_predicted_validation = poly(x_validation)

        # Calculate Maximum Absolute Error over the INTERIOR range
        abs_errors = np.abs(y_predicted_validation - y_validation)
        # Use slicing for interior
        max_interior_error = np.max(abs_errors[idx_start:idx_end+1])

        # print(f"  Order {order}: Max Interior Abs Error = {max_interior_error:.4g}")

        # Check if this is the best model so far based on max interior error
        if max_interior_error < min_validation_max_error * 0.95:
            min_validation_max_error = max_interior_error
            best_order = order
            best_coeffs = coeffs
            # print(f"    New best order: {best_order}")
        # Check for overfitting (early stopping based on max interior error)
        elif overfitting_tolerance > 0 and max_interior_error > min_validation_max_error * (1 + overfitting_tolerance):
            # print(f"    Max interior error increased significantly. Stopping early at order {order}.")
            break

    # --- Calculate Derivative of Best Polynomial ---
    if best_coeffs is None:
        warnings.warn(
            "No suitable polynomial fit found within the specified orders.")
        return None, None

    # print(f"Selected best polynomial order: {best_order} (based on min max interior error)")
    best_poly = np.poly1d(best_coeffs)
    derivative_poly = best_poly.deriv()
    
    # get min and max x value (not index) based on `interior_percent`
    x_min = np.min(x_fine)
    x_max = np.max(x_fine)
    interior_range = (x_min + (x_max - x_min) * (100 - interior_percent) / 2.0 / 100,
                      x_max - (x_max - x_min) * (100 - interior_percent) / 2.0 / 100)

    # --- Evaluate Derivative at Chosen Points ---
    if output_at == 'original':
        x_output = np.array([v for v in x_unique if interior_range[0] <= v <= interior_range[1]])
        derivative_values = derivative_poly(x_output)
        # print(f"Evaluated derivative at {len(x_output)} original unique x points.")
    elif output_at == 'fine':
        x_output = np.array([v for v in x_fine if interior_range[0] <= v <= interior_range[1]])
        derivative_values = derivative_poly(x_output)
        # print(f"Evaluated derivative at {len(x_output)} fine resampled x points.")
    else:
        raise ValueError("Internal error: Invalid output_at value.")

    # Store best_order alongside output if needed for plotting label
    # (We can't directly return it without changing signature, but it's printed)
    # If needed, could return a dictionary or tuple: (x_output, derivative_values, best_order)

    return x_output, derivative_values, best_order


def compute_derivative(x, y, order=-1, method="basic", num_points=100, k=3, output_at="fine"):
    """
    Calculate higher-order derivatives using repeated application of np.diff().

    Parameters:
    x (array): x-coordinates
    y (array): y-coordinates
    order (int): The order of the derivative to calculate (default is 1)

    Returns:
    tuple: (x_values, derivative_values)
    """

    if method == "bspline":
        out = compute_bspline_derivative(x, y, num_points, k)
        return out[0], out[1], None
    elif method == "polynomial":
        if order > 0:
            # Use polynomial fitting for the derivative
            return compute_polynomial_derivative(
                x, y, n_validation_factor=5, min_order=order, max_order=order, output_at=output_at)
        else:
            return compute_polynomial_derivative(x, y, output_at=output_at)
    
    order = max(1, order)  # Ensure order is at least 1

    # The following block implements the "basic" centered difference method
    x_current = np.array(x[2:-2])  # Exclude the first and last points for centered difference
    y_current = np.array(y[2:-2])

    # Check if there are enough points for the requested order
    # For a k-th order centered derivative, at least 2k+1 points are needed to produce one value.
    if len(x_current) < 2 * order + 1:
        warnings.warn(
            f"For 'basic' method with order {order}, need at least {2 * order + 1} points. "
            f"Got {len(x_current)}. Returning empty arrays."
        )
        return np.array([]), np.array([]), None

    for _ in range(order):
        # Centered difference formula: (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
        # This is applied iteratively for higher-order derivatives.
        dy = y_current[2:] - y_current[:-2]
        dx = x_current[2:] - x_current[:-2]

        # numpy.divide will issue a RuntimeWarning and return inf/nan if dx contains zeros.
        derivative = dy / dx
        
        # The x-coordinates for the derivative values are the central points of the intervals used.
        x_current = x_current[1:-1] 
        y_current = derivative

    return x_current, y_current, None


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
                    eigenvalues.extend([float(val)
                                       for val in lines[i].split()])
                    i += 1
                cp_info["EIGENVALUES OF HESSIAN MATRIX"] = eigenvalues
                continue
            elif line.startswith(
                "EIGENVECTORS (ORTHONORMAL) OF HESSIAN MATRIX (COLUMNS):"
            ):
                eigenvectors = []
                i += 2
                while i < len(lines) and lines[i]:
                    eigenvectors.append([float(val)
                                        for val in lines[i].split()])
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
                    hessian_matrix.append([float(val)
                                          for val in lines[i].split()])
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


def get_bcp_properties(job, atom_pairs_dict, unrestricted=False):
    # first get kf file from finished job
    kf = KFFile(job.results["adf.rkf"])
    cp_type_codes = {"nuclear": 1, "bond": 3, "ring": 4, "cage": 2}
    num_cps = kf[("Properties", "CP number of")]
    cp_coords = kf[("Properties", "CP coordinates")]
    cp_codes = kf[("Properties", "CP code number for (Rank,Signatu")]
    eef = kf[("Molecule", "eeEField")]
    eef_mag = sum([v**2 for v in eef])**0.5 * float(np.sign(sum(eef)))
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

    atom_pairs = list(atom_pairs_dict.keys())
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
        bond_length = out_mol.atoms[pair[0] -
                                    1].distance_to(out_mol.atoms[pair[1] - 1])
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
                bcp_coords[-1][i] - check_point_spacing *
                (num_check_points - 1) / 2
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
        out_cp_data[i]["NEB image"] = image_number
        out_cp_data[i]["Molecular bond energy"] = job_energy
        out_cp_data[i]["EEF"] = eef_mag

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
                    * num_check_points_total: (cp_ind + 1)
                    * num_check_points_total
                ]
                grad_y = densf_kf[("SCF", f"DensityGradY_{field}")][
                    cp_ind
                    * num_check_points_total: (cp_ind + 1)
                    * num_check_points_total
                ]
                grad_z = densf_kf[("SCF", f"DensityGradZ_{field}")][
                    cp_ind
                    * num_check_points_total: (cp_ind + 1)
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
                    cp_data[out_cp_ind][f"Phi_{field}"] = atan(
                        sqrt(abs(ev[1] / ev[2])))
                elif cp_codes[cpi - 1] == cp_type_codes["ring"]:
                    cp_data[out_cp_ind][f"Theta_{field}"] = atan(
                        sqrt(abs(ev[2] / ev[0]))
                    )
                    cp_data[out_cp_ind][f"Phi_{field}"] = atan(
                        sqrt(abs(ev[1] / ev[0])))
                cp_data[out_cp_ind][f"HESSIAN MATRIX_{field}"] = hess.tolist()
                cp_data[out_cp_ind][
                    f"EIGENVECTORS (ORTHONORMAL) OF HESSIAN MATRIX (COLUMNS)_{field}"
                ] = evec.T.tolist()
                cp_data[out_cp_ind][f"EIGENVALUES_{field}"] = ev.tolist()

            for field in ["A", "B"]:
                get_saddle_t41_properties(
                    out_cp_data, i, out_cp_data_cp_inds[i], field)

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
    num_points = [int((max_xyz[i] - min_xyz[i]) / densf_bb_spacing)
                  for i in range(3)]
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


def generate_plots(cp_data, prop_list, x_prop_list, out_dir, combined_y_prop_lists, atom_pairs_dict):
    # Ensure the output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    unique_bcp_atoms = sorted(list(set([cp["ATOMS"] for cp in cp_data])))
    image_names = sorted(list(set([cp["JOB_NAME"] for cp in cp_data])))
    job_name = os.path.commonprefix(image_names)
    eef_strs = ["origEEF", "revEEF", "noEEF"]
    has_eef = [eef in str(image_names) for eef in eef_strs].count(True) > 1

    reverse_x_axis = [" (reverse)" in x_prop for x_prop in x_prop_list]
    x_prop_list = [
        x_prop.replace(" (reverse)", "") for x_prop in x_prop_list
    ]

    all_props = []
    for prop in prop_list + x_prop_list:
        for cp_prop in cp_data[0].keys():
            if cp_prop in [prop, f"{prop}_A", f"{prop}_B"]:
                all_props.append(cp_prop)

    bcp_prop_dict = {}
    eef_types = ([f"_{eef[0]}" for eef in eef_pairs] +
                 [""]) if has_eef else [""]
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

    for xi, x_prop in enumerate(x_prop_list):
        x_prop_tuple = None
        if "distance" in x_prop.lower():
            x_prop_tuple = tuple(map(int, re.findall(
                r"[a-zA-Z]{1,2}(\d+)-[a-zA-Z]{1,2}(\d+)", x_prop.lower().replace(" distance", ""))[0]))
            if x_prop_tuple not in atom_pairs_dict:
                # reverse tuple order
                x_prop_tuple = (x_prop_tuple[1], x_prop_tuple[0])
                if x_prop_tuple not in atom_pairs_dict:
                    log_print(
                        f"Warning: Atom pair {x_prop_tuple} not found in atom_pairs_dict.")
                    x_prop_tuple = None

        if x_prop_tuple and len(atom_pairs_dict[x_prop_tuple]) > 0:
            x_prop_label = x_prop + ": (" + atom_pairs_dict[x_prop_tuple] + ")"
        else:
            x_prop_label = x_prop

        for plot_name, y_prop_list_from_dict_val in combined_y_prop_lists.items():
            log_print(
                f"Plotting combined plots for {plot_name} vs {x_prop_label} for bond CPs"
            )

            expanded_y_prop_list_final = []
            smooth_derivatives_final = []

            for y_prop_entry in y_prop_list_from_dict_val:
                if isinstance(y_prop_entry, list):
                    processed_group = []
                    group_contains_smoothed_derivative = False
                    for sub_prop_name_full in y_prop_entry:
                        is_smooth_flag_for_sub_prop = " (smoothed)" in sub_prop_name_full
                        sub_prop_name_cleaned = sub_prop_name_full.replace(" (smoothed)", "")
                        processed_group.append(sub_prop_name_cleaned)
                        if is_smooth_flag_for_sub_prop and " d/dx" in sub_prop_name_cleaned:
                            group_contains_smoothed_derivative = True
                    
                    # Check if all base properties in the group have _A variants
                    can_expand_group = True
                    if not processed_group: # Handle empty group case
                        can_expand_group = False
                        
                    for sub_prop_cleaned in processed_group:
                        base_sub_prop = sub_prop_cleaned.replace(" d/dx", "")
                        if f"{base_sub_prop}_A" not in all_props:
                            can_expand_group = False
                            break
                    
                    if can_expand_group:
                        group_A = []
                        group_B = []
                        for sub_prop_cleaned in processed_group:
                            is_deriv = " d/dx" in sub_prop_cleaned
                            base_name = sub_prop_cleaned.replace(" d/dx", "")
                            group_A.append(f"{base_name}_A{' d/dx' if is_deriv else ''}")
                            group_B.append(f"{base_name}_B{' d/dx' if is_deriv else ''}")
                        
                        expanded_y_prop_list_final.extend([group_A, group_B, processed_group])
                        smooth_derivatives_final.extend([group_contains_smoothed_derivative] * 3)
                    else:
                        expanded_y_prop_list_final.append(processed_group)
                        smooth_derivatives_final.append(group_contains_smoothed_derivative)
                else: # y_prop_entry is a string
                    is_smooth_for_this_entry = " (smoothed)" in y_prop_entry
                    y_prop_cleaned = y_prop_entry.replace(" (smoothed)", "")

                    base_prop_for_ab_check = y_prop_cleaned.replace(" d/dx", "")
                    can_expand_single_prop = f"{base_prop_for_ab_check}_A" in all_props

                    if can_expand_single_prop:
                        is_single_deriv = " d/dx" in y_prop_cleaned
                        expanded_y_prop_list_final.extend([
                            f"{base_prop_for_ab_check}_A{' d/dx' if is_single_deriv else ''}",
                            f"{base_prop_for_ab_check}_B{' d/dx' if is_single_deriv else ''}",
                            y_prop_cleaned, # The original (total) or cleaned name
                        ])
                        smooth_derivatives_final.extend([is_smooth_for_this_entry] * 3)
                    else:
                        expanded_y_prop_list_final.append(y_prop_cleaned)
                        smooth_derivatives_final.append(is_smooth_for_this_entry)
            
            current_expanded_y_prop_list = expanded_y_prop_list_final
            current_smooth_derivatives = smooth_derivatives_final

            if not current_expanded_y_prop_list:
                log_print(f"No properties to plot for {plot_name} vs {x_prop_label}. Skipping.")
                continue

            fig, axs = plt.subplots(
                len(current_expanded_y_prop_list),
                (num_eef if num_eef > 0 else 0) + 1, # num_eef can be 0 if has_eef is false or eef_types is [""]
                figsize=(1 + 7 * ((num_eef if num_eef > 0 else 0) + 1), 5 * len(current_expanded_y_prop_list)),
                squeeze=False # Always return 2D array for axs
            )
            fig.suptitle(
                f"{job_name}: Combined plots for {x_prop_label} ({plot_name})",
                fontsize=16,
                y=1.00, # Adjusted y to prevent overlap with tight_layout
            )
            
            # Define colors for BCPs and markers for properties within a group
            if unique_bcp_atoms: # Ensure there are BCPs to map colors to
                if len(unique_bcp_atoms) <= 10:
                    bcp_color_map = {bcp: mcm.get_cmap('tab10')(i) for i, bcp in enumerate(unique_bcp_atoms)}
                elif len(unique_bcp_atoms) <= 20:
                    bcp_color_map = {bcp: mcm.get_cmap('tab20')(i) for i, bcp in enumerate(unique_bcp_atoms)}
                else:
                    bcp_color_map = {bcp: mcm.get_cmap('viridis')(i / (len(unique_bcp_atoms)-1) if len(unique_bcp_atoms)>1 else 0.5) for i, bcp in enumerate(unique_bcp_atoms)}
            else:
                bcp_color_map = {}


            property_markers = ['o', 's', '^', 'P', 'X', 'D', '*', 'v', '<', '>']
            property_line_styles = ['-', '--', '-.', ':']

            for i, y_prop_group in enumerate(current_expanded_y_prop_list):
                # Skip plotting if x_prop is part of y_prop_group (avoid plotting prop vs itself)
                if isinstance(y_prop_group, str):
                    if x_prop in y_prop_group or y_prop_group in x_prop:
                        for col_ax in range(axs.shape[1]): axs[i, col_ax].set_visible(False)
                        continue
                elif isinstance(y_prop_group, list):
                    if any(x_prop in item or item in x_prop for item in y_prop_group):
                        for col_ax in range(axs.shape[1]): axs[i, col_ax].set_visible(False)
                        continue
                
                row_y_label = ""
                if isinstance(y_prop_group, str):
                    row_y_label = y_prop_group
                elif isinstance(y_prop_group, list):
                    common_suffix = ""
                    is_all_deriv = all(" d/dx" in prop for prop in y_prop_group)
                    temp_props_for_label = []
                    for prop in y_prop_group:
                        name_part = prop.replace(" d/dx", "") if is_all_deriv else prop
                        # Further simplify for legend e.g. Rho_A -> Rho (A)
                        if "_A" in name_part: name_part = name_part.replace("_A", " (A)")
                        if "_B" in name_part: name_part = name_part.replace("_B", " (B)")
                        temp_props_for_label.append(name_part)
                    
                    row_y_label = ", ".join(temp_props_for_label)
                    if is_all_deriv: row_y_label += " d/dx"


                for j_col, eef_suffix_for_plot in enumerate(eef_types if num_eef > 0 else [""]): # Loop for EEF columns
                    current_ax = axs[i, j_col]
                    
                    title_prop_str = row_y_label
                    current_ax.set_title(f"{title_prop_str}{eef_suffix_for_plot} vs {x_prop_label}", fontsize=9)
                    current_ax.set_xlabel(x_prop_label)
                    current_ax.set_ylabel(row_y_label)

                    if reverse_x_axis[xi]:
                        current_ax.invert_xaxis()

                    props_for_this_ax = y_prop_group if isinstance(y_prop_group, list) else [y_prop_group]
                    
                    for bcp_i, (bcp, bcp_data_for_bcp) in enumerate(bcp_prop_dict.items()):
                        
                        # First need to pre-compute any len > 1 property groups that are derivatives.
                        # This is to find the minimum order of derivative for the group so that
                        # we can use the same order for all properties in the group.
                        deriv_orders = []
                        for prop_idx, actual_y_prop in enumerate(props_for_this_ax):
                            deriv_orders.append(None)
                            is_derivative = " d/dx" in actual_y_prop
                            if not all([current_smooth_derivatives[i], is_derivative]):
                                continue
                            base_actual_prop = actual_y_prop.replace(" d/dx", "") if is_derivative else actual_y_prop

                            # X values: try EEF-specific first, then universal for that x_prop
                            x_values = bcp_data_for_bcp.get(f"{x_prop}{eef_suffix_for_plot}", bcp_data_for_bcp.get(x_prop))
                            y_values = bcp_data_for_bcp.get(f"{base_actual_prop}{eef_suffix_for_plot}")

                            if x_values and y_values:
                                if len(x_values) != len(y_values):
                                    print(f"Warning: Length mismatch for BCP {bcp}, Y-prop {actual_y_prop}{eef_suffix_for_plot} (len {len(y_values)}) vs X-prop {x_prop} (len {len(x_values)}). Skipping.")
                                    continue
                                if not x_values:
                                    continue
                                x_values_sorted, y_values_sorted = zip(*sorted(zip(x_values, y_values)))
                                # Compute the derivative order for this property
                                _, _, order = compute_derivative(x_values_sorted, y_values_sorted, method="polynomial")
                                deriv_orders[-1] = order
                        # Now find the minimum order for the group
                        if len(deriv_orders) > 0 and any(deriv_orders):
                            # Only compute min if there are any non-None orders
                            # This is to avoid errors if all orders are None
                            # or if the group is empty
                            poly_order = min([order for order in deriv_orders if order is not None])
                        else:
                            poly_order = 1
                        
                        
                        for prop_idx, actual_y_prop in enumerate(props_for_this_ax):
                            is_derivative = " d/dx" in actual_y_prop
                            base_actual_prop = actual_y_prop.replace(" d/dx", "") if is_derivative else actual_y_prop

                            # X values: try EEF-specific first, then universal for that x_prop
                            x_values = bcp_data_for_bcp.get(f"{x_prop}{eef_suffix_for_plot}", bcp_data_for_bcp.get(x_prop))
                            y_values = bcp_data_for_bcp.get(f"{base_actual_prop}{eef_suffix_for_plot}")

                            if x_values and y_values:
                                if len(x_values) != len(y_values):
                                    print(f"Warning: Length mismatch for BCP {bcp}, Y-prop {actual_y_prop}{eef_suffix_for_plot} (len {len(y_values)}) vs X-prop {x_prop} (len {len(x_values)}). Skipping.")
                                    continue
                                if not x_values: # Skip if empty after filtering
                                    continue

                                x_values_sorted, y_values_sorted = zip(*sorted(zip(x_values, y_values)))

                                atom_pair_match = re.search(r"([a-zA-Z]{1,2}\d+)-([a-zA-Z]{1,2}\d+)", bcp)
                                bcp_display_name = bcp
                                if atom_pair_match:
                                    try:
                                        atom_nums_in_bcp = tuple(map(int, re.findall(r'\d+', atom_pair_match.group(0))))
                                        # Ensure correct order for dict lookup if necessary, or just use bcp string
                                        # For atom_pairs_dict, we need the numbers.
                                        # The keys in atom_pairs_dict are tuples of integers.
                                        # We need to parse the numbers from the bcp string like "C1-O2"
                                        bcp_atom_num_tuple = []
                                        for atom_label_part in bcp.split('-'):
                                            num_match_bcp = re.search(r'\d+', atom_label_part)
                                            if num_match_bcp: bcp_atom_num_tuple.append(int(num_match_bcp.group(0)))
                                        
                                        bcp_atom_num_tuple = tuple(bcp_atom_num_tuple)
                                        desc = atom_pairs_dict.get(bcp_atom_num_tuple, atom_pairs_dict.get(tuple(reversed(bcp_atom_num_tuple)), ""))
                                        if desc: bcp_display_name = f"{bcp} ({desc})"
                                    except: # Parsing failed
                                        pass


                                legend_label = bcp_display_name
                                current_marker = 'o' # Default marker'
                                current_line_style = "-"
                                is_prop_group = False
                                if isinstance(y_prop_group, list) and len(y_prop_group) > 1:
                                    is_prop_group = True
                                    prop_legend_name = actual_y_prop.replace(" d/dx", " deriv")
                                    if "_A" in prop_legend_name: prop_legend_name = prop_legend_name.replace("_A", " (A)")
                                    if "_B" in prop_legend_name: prop_legend_name = prop_legend_name.replace("_B", " (B)")
                                    legend_label += f" ({prop_legend_name})"
                                    current_marker = property_markers[prop_idx % len(property_markers)]
                                    current_line_style = property_line_styles[prop_idx % len(property_line_styles)]
                                else:
                                    if is_derivative:
                                        current_line_style = property_line_styles[bcp_i % len(property_line_styles)]
                                    else:
                                        current_marker = property_markers[bcp_i % len(property_markers)]

                                current_bcp_color = bcp_color_map.get(bcp, 'k') # Default to black if bcp not in map
                                

                                if is_derivative:
                                    x_deriv, y_deriv, order = compute_derivative(x_values_sorted, y_values_sorted, order = poly_order, method=("polynomial" if current_smooth_derivatives[i] else "basic"))
                                    if x_deriv is None or y_deriv is None or not list(x_deriv): continue
                                    
                                    last_order = order

                                    if reverse_x_axis[xi]: y_deriv = [-y_val for y_val in y_deriv]
                                    if order is not None: legend_label += f" (ord {order})"
                                    
                                    current_ax.plot(
                                        x_deriv[1:-1], y_deriv[1:-1],
                                        linestyle=current_line_style,
                                        marker=current_marker if not current_smooth_derivatives[i] and not is_prop_group else None,
                                        color=current_bcp_color, label=legend_label, markersize=3
                                    )
                                else:
                                    current_ax.plot(
                                        x_values_sorted, y_values_sorted,
                                        linestyle=current_line_style, marker=current_marker if not is_prop_group else None,
                                        color=current_bcp_color, label=legend_label, markersize=3
                                    )
                    current_ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7)
                    current_ax.grid(True)

                # "All EEF" plot column
                if has_eef and num_eef > 0 : # Only if there are actual EEF types
                    ax_all_eef = axs[i, num_eef if num_eef > 0 else 0] # num_eef is count, so index is num_eef
                    
                    title_prop_str_all_eef = row_y_label
                    ax_all_eef.set_title(f"{title_prop_str_all_eef} vs {x_prop_label} (All EEF)", fontsize=9)
                    ax_all_eef.set_xlabel(x_prop_label)
                    ax_all_eef.set_ylabel(title_prop_str_all_eef)

                    if reverse_x_axis[xi]: ax_all_eef.invert_xaxis()

                    props_for_this_ax_all_eef = y_prop_group if isinstance(y_prop_group, list) else [y_prop_group]
                    
                    for bcp, bcp_data_for_bcp_all_eef in bcp_prop_dict.items():
                        
                        # First need to pre-compute any len > 1 property groups that are derivatives.
                        # This is to find the minimum order of derivative for the group so that
                        # we can use the same order for all properties in the group.
                        deriv_orders_all_eef = []
                        for prop_idx_all_eef, actual_y_prop_all_eef in enumerate(props_for_this_ax_all_eef):
                            deriv_orders_all_eef.append(None)
                            is_derivative_all_eef = " d/dx" in actual_y_prop_all_eef
                            if not all([current_smooth_derivatives[i], is_derivative_all_eef]):
                                continue
                            base_actual_prop_all_eef = actual_y_prop_all_eef.replace(" d/dx", "") if is_derivative_all_eef else actual_y_prop_all_eef

                            # X values: try EEF-specific first, then universal for that x_prop
                            x_values_all_eef = bcp_data_for_bcp_all_eef.get(f"{x_prop}{eef_suffix_for_plot}", bcp_data_for_bcp_all_eef.get(x_prop))
                            y_values_all_eef = bcp_data_for_bcp_all_eef.get(f"{base_actual_prop_all_eef}{eef_suffix_for_plot}")

                            if x_values_all_eef and y_values_all_eef:
                                if len(x_values_all_eef) != len(y_values_all_eef):
                                    print(f"Warning: Length mismatch for BCP {bcp}, Y-prop {actual_y_prop_all_eef}{eef_suffix_for_plot} (len {len(y_values_all_eef)}) vs X-prop {x_prop} (len {len(x_values_all_eef)}). Skipping.")
                                    continue
                                if not x_values_all_eef: continue
                                
                                x_v_sorted_all_eef, y_v_sorted_all_eef = zip(*sorted(zip(x_values_all_eef, y_values_all_eef)))
                                # Compute the derivative order for this property
                                _, _, order = compute_derivative(x_v_sorted_all_eef, y_v_sorted_all_eef, method="polynomial")
                                deriv_orders_all_eef[-1] = order
                        # Now find the minimum order for the group
                        if len(deriv_orders_all_eef) > 0 and any(deriv_orders_all_eef):
                            # Only compute min if there are any non-None orders
                            # This is to avoid errors if all orders are None
                            # or if the group is empty
                            poly_order_all_eef = min([order for order in deriv_orders_all_eef if order is not None])
                        else:
                            poly_order_all_eef = 1

                        for prop_idx_all_eef, actual_y_prop_all_eef in enumerate(props_for_this_ax_all_eef):
                            is_derivative_all_eef = " d/dx" in actual_y_prop_all_eef
                            base_actual_prop_all_eef = actual_y_prop_all_eef.replace(" d/dx", "") if is_derivative_all_eef else actual_y_prop_all_eef

                            current_bcp_color = bcp_color_map.get(bcp, 'k')
                            current_prop_marker_all_eef = 'o'
                            if isinstance(y_prop_group, list) and len(y_prop_group) > 1:
                                current_prop_marker_all_eef = property_markers[prop_idx_all_eef % len(property_markers)]

                            for eef_plot_suffix in eef_types: # Iterate through EEF types for lines
                                x_values_all_eef = bcp_data_for_bcp_all_eef.get(f"{x_prop}{eef_plot_suffix}", bcp_data_for_bcp_all_eef.get(x_prop))
                                y_values_all_eef = bcp_data_for_bcp_all_eef.get(f"{base_actual_prop_all_eef}{eef_plot_suffix}")

                                if x_values_all_eef and y_values_all_eef:
                                    if len(x_values_all_eef) != len(y_values_all_eef):
                                        print(f"Warning: Length mismatch for BCP {bcp}{eef_plot_suffix} in All EEF plot. Y-prop {actual_y_prop_all_eef} (len {len(y_values_all_eef)}) vs X-prop {x_prop} (len {len(x_values_all_eef)}). Skipping.")
                                        continue
                                    if not x_values_all_eef: continue
                                    
                                    x_v_sorted_all_eef, y_v_sorted_all_eef = zip(*sorted(zip(x_values_all_eef, y_values_all_eef)))
                                    
                                    bcp_display_name_all_eef = bcp
                                    # Simplified BCP name for All EEF plot to reduce legend clutter
                                    # Or use the full name if preferred.
                                    # For consistency, using the same logic as individual EEF plots:
                                    atom_pair_match_all_eef = re.search(r"([a-zA-Z]{1,2}\d+)-([a-zA-Z]{1,2}\d+)", bcp)
                                    if atom_pair_match_all_eef:
                                        try:
                                            bcp_atom_num_tuple_all_eef = []
                                            for atom_label_part in bcp.split('-'):
                                                num_match_bcp_all_eef = re.search(r'\d+', atom_label_part)
                                                if num_match_bcp_all_eef: bcp_atom_num_tuple_all_eef.append(int(num_match_bcp_all_eef.group(0)))
                                            bcp_atom_num_tuple_all_eef = tuple(bcp_atom_num_tuple_all_eef)
                                            desc_all_eef = atom_pairs_dict.get(bcp_atom_num_tuple_all_eef, atom_pairs_dict.get(tuple(reversed(bcp_atom_num_tuple_all_eef)), ""))
                                            if desc_all_eef: bcp_display_name_all_eef = f"{bcp} ({desc_all_eef})"
                                        except: pass

                                    legend_label_all_eef = f"{bcp_display_name_all_eef}{eef_plot_suffix}"
                                    if isinstance(y_prop_group, list) and len(y_prop_group) > 1:
                                        prop_legend_name_all_eef = actual_y_prop_all_eef.replace(" d/dx", " deriv")
                                        if "_A" in prop_legend_name_all_eef: prop_legend_name_all_eef = prop_legend_name_all_eef.replace("_A", " (A)")
                                        if "_B" in prop_legend_name_all_eef: prop_legend_name_all_eef = prop_legend_name_all_eef.replace("_B", " (B)")
                                        legend_label_all_eef += f" ({prop_legend_name_all_eef})"
                                        
                                    current_prop_marker_all_eef = property_markers[prop_idx_all_eef % len(property_markers)]
                                    current_line_style = line_styles.get(eef_plot_suffix, "-")

                                    if is_derivative_all_eef:
                                        x_d_all_eef, y_d_all_eef, order_all_eef = compute_derivative(x_v_sorted_all_eef, y_v_sorted_all_eef, order = poly_order_all_eef, method=("polynomial" if current_smooth_derivatives[i] else "basic"))
                                        if x_d_all_eef is None or y_d_all_eef is None or not list(x_d_all_eef): continue
                                        if reverse_x_axis[xi]: y_d_all_eef = [-y_val for y_val in y_d_all_eef]
                                        if order_all_eef is not None: legend_label_all_eef += f" (ord {order_all_eef})"
                                        
                                        ax_all_eef.plot(
                                            x_d_all_eef[1:-1], y_d_all_eef[1:-1],
                                            linestyle=current_line_style,
                                            marker=current_prop_marker_all_eef if not current_smooth_derivatives[i] else None,
                                            color=current_bcp_color, label=legend_label_all_eef, markersize=2
                                        )
                                    else:
                                        ax_all_eef.plot(
                                            x_v_sorted_all_eef, y_v_sorted_all_eef,
                                            linestyle=current_line_style, marker=current_prop_marker_all_eef,
                                            color=current_bcp_color, label=legend_label_all_eef, markersize=2
                                        )
                    ax_all_eef.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7)
                    ax_all_eef.grid(True)

            plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust rect to make space for suptitle if y was lowered
            # Determine top padding based on number of rows; the more rows, the less padding
            # top_padding = min(0.995, 0.9 + 0.02 * len(current_expanded_y_prop_list)) # Original
            # plt.subplots_adjust(top=top_padding, hspace=0.4) # Added hspace
            # tight_layout often handles this better. If suptitle overlaps, adjust its y or use fig.subplots_adjust.
            
            f_base = f"{job_name}combined_{x_prop.replace(' ', '_')}_{plot_name.replace(' ', '_')}.png".replace("/", "-")
            plt.savefig(os.path.join(out_dir, f_base), dpi=300, bbox_inches="tight")
            plt.close(fig) # Close the figure explicitly

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
                        x_values, y_values = zip(
                            *sorted(zip(x_values, y_values)))
                        ax.plot(x_values, y_values, "-o",
                                label=bcp, markersize=2)
                    else:
                        ax = None
                        plt.close()
                        break
                if ax:
                    ax.legend(loc="upper left",
                              bbox_to_anchor=(1, 1), fontsize=7)
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
                log_print(
                    f"Plotting {y_prop} vs {x_prop} for bond CPs (All EEF)")
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
                            x_values, y_values = zip(
                                *sorted(zip(x_values, y_values)))
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
                    ax.legend(loc="upper left",
                              bbox_to_anchor=(1, 1), fontsize=7)
                    plt.subplots_adjust(right=0.75)
                    ax.grid(True)
                    plt.savefig(
                        os.path.join(
                            out_dir, f"{job_name}{y_prop}_vs_{x_prop}_all_eef.png"
                        )
                    )
                    plt.close()


def write_csv(cp_data, input_file_path, include_keys=None, rename_key_map=None):

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
                    all_keys.update(
                        [f"{key}_{axis}" for axis in ["X", "Y", "Z"]])
            else:
                all_keys.add(key)
    # Sort the keys
    all_keys = sorted(all_keys)

    # Define the preferred order of columns
    preferred_order = [
        "ATOMS",
        "NEB image",
        "JOB_NAME",
        "CP #",
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
    for h in [s for s in all_keys if "EIGENVALUES" in s and s not in preferred_order]:
        preferred_order.append(h)
    for h in [s for s in all_keys if "EIGENVECTORS (ORTHONORMAL) OF HESSIAN MATRIX (COLUMNS)" in s and s not in preferred_order]:
        preferred_order.append(h)

    # Then Hessian
    for h in [s for s in all_keys if "HESSIAN" in s and s not in preferred_order]:
        preferred_order.append(h)

    # Now add the rest if not already in the preferred order
    for h in all_keys:
        if h not in preferred_order:
            preferred_order.append(h)
            

    headers = preferred_order
    if include_keys is not None:
        headers = [h for h in headers if any(re.fullmatch(pattern, h) for pattern in include_keys)]
    
    # headers = sorted(list(set(headers)))
    


    # Define the output CSV file path
    base_name = os.path.splitext(input_file_path)[0]
    suffix = "_reduced" if include_keys is not None else ""
    csv_file_path = f"{base_name}_cp_info{suffix}.csv"

    # Write the CSV file
    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for cp in cp_data:
            row = {}
            for key, value in cp.items():
                if isinstance(value, list):
                    if isinstance(value[0], list):  # 2D list
                        if "EIGENVECTORS (ORTHONORMAL) OF HESSIAN MATRIX (COLUMNS)" in key:
                            for i in range(3):
                                for j, axis in enumerate(["X", "Y", "Z"]):
                                    new_key = f"{key}_EV{i+1}_{axis}"
                                    if include_keys is not None and not any(re.fullmatch(pattern, new_key) for pattern in include_keys):
                                        continue
                                    row[new_key] = value[i][j]
                        elif "HESSIAN MATRIX" in key:
                            for i, axis1 in enumerate(["X", "Y", "Z"]):
                                for j, axis2 in enumerate(["X", "Y", "Z"]):
                                    new_key = f"{key}_{axis1}{axis2}"
                                    if include_keys is not None and not any(re.fullmatch(pattern, new_key) for pattern in include_keys):
                                        continue
                                    row[new_key] = value[i][j]
                    else:  # 1D list
                        for i, axis in enumerate(["X", "Y", "Z"]):
                            new_key = f"{key}_{axis}"
                            if include_keys is not None and not any(re.fullmatch(pattern, new_key) for pattern in include_keys):
                                continue
                            row[new_key] = value[i]
                else:
                    if include_keys is not None and not any(re.fullmatch(pattern, key) for pattern in include_keys):
                        continue
                    row[key] = value
            writer.writerow(row)

    # Open the file and apply any header renaming if specified
    if rename_key_map is not None and rename_key_map:
        with open(csv_file_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)

        # Rename headers in the first row
        headers = rows[0]
        renamed_headers = [rename_key_map.get(h, h) for h in headers]
        rows[0] = renamed_headers

        # Write back the modified rows to the CSV file
        with open(csv_file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)
    
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
        mol.properties = mol1.properties.copy()
        interpolated_molecules.append(mol)
    return interpolated_molecules


def main(ams_job_path, atom_pairs, x_prop_list=plot_x_prop_list):
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

    log_print(
        f"Processing job {input_job_name} with termination status {term_status}")

    is_unrestricted = "Unrestricted Yes".lower(
    ) in kf[("General", "user input")].lower()

    # Check if EEF is present
    has_eef = ("im0.eeEField" in kf.read_section("NEB").keys() and any(
        float(i) != 0.0 for i in kf[("NEB", "im0.eeEField")])) or user_eef is not None

    if has_eef:
        log_print(
            f"Using user-defined EEF: {user_eef}" if user_eef is not None else "EEF is present in the NEB job.")
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

    log_print(
        f"Total number of images in NEB: {num_images}, highest index image: {highest_index_image}")

    ########################################################################################
    # Step 2: create jobs for each image and then run them
    ########################################################################################

    jobs = MultiJob(name=input_job_name)

    im_num = 0

    extra_image_nums = list(range(
        highest_index_image - extra_images_num_adjacent_images,
        highest_index_image + extra_images_num_adjacent_images
    )) if extra_images_num_adjacent_images > 0 and num_extra_images > 0 else []
    extra_image_job_names = []

    for i in range(num_images):
        if i < start_image or (i > end_image and end_image > 0):
            log_print(
                f"Skipping image {i} as it is outside the specified range of NEB images to include.")
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

        species_Z = [kf[("NEB", f"im{i}.sp{j+1}Z")]
                     for j in range(num_species)]
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
        atom_coords = [tuple(coords[i: i + 3])
                       for i in range(0, len(coords), 3)]

        # create molecule for image
        im_mol = Molecule(positions=atom_coords, numbers=atom_numbers)
        # copy properties from the input job
        im_mol.properties = input_job.molecule[''].properties.copy()
        im_mol.guess_bonds()
        # Now add atom properties from input job molecule
        for j in range(len(input_job_mol.atoms)):
            im_mol.atoms[j].properties = input_job_mol.atoms[j].properties.copy()

        moles = [im_mol]

        if i in extra_image_nums:
            ip1 = i + 1
            ip1_coords = kf[("NEB", f"im{ip1}.xyzAtoms")]
            ip1_coords = [
                Units.convert(v, "bohr", "angstrom") for v in ip1_coords
            ]  # to convert to angstrom for comparison to coords shown in ADFMovie

            # coords is a 1D list of coordinates, so we need to reshape it to a 2D list
            ip1_atom_coords = [
                tuple(ip1_coords[i: i + 3]) for i in range(0, len(ip1_coords), 3)
            ]
            ip1_mol = Molecule(positions=ip1_atom_coords, numbers=atom_numbers)
            # copy properties from the input job
            ip1_mol.properties = im_mol.properties.copy()
            moles.extend(interpolate_molecules(
                im_mol, ip1_mol, num_extra_images))
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
                        kf[("NEB", f"im{i}.eeEField")
                           ] if user_eef is None else user_eef
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
                    job = AMSJob(
                        molecule=mol, settings=job_settings, name=job_name)
                    jobs.children.append(job)
                    if is_in_extra_image_range and mol != moles[0]:
                        extra_image_job_names.append(job_name)

                    # save job runscript for inspection
                    # run_file_path = f"{job_name}.run"
                    # with open(run_file_path, "w") as run_file:
                    #     run_file.write(str(job.molecule)+"\n")
                    #     run_file.write(str(job.settings)+"\n")
                    #     run_file.write(str(job.get_runscript()))
            else:
                # no EEF, so only need to run one job for the image
                job_name = f"{input_job_name}_im{im_num:03d}"
                job = AMSJob(
                    molecule=mol, settings=base_settings, name=job_name)
                jobs.children.append(job)
                if is_in_extra_image_range and mol != moles[0]:
                    extra_image_job_names.append(job_name)

                # save job runscript for inspection
                # run_file_path = f"{job_name}.run"
                # with open(run_file_path, "w") as run_file:
                #     run_file.write(str(job.molecule)+"\n")
                #     run_file.write(str(job.settings)+"\n")
                #     run_file.write(str(job.get_runscript()))
            im_num += 1

    # print each job's name and coordinates of first atom (for debugging only; remember to check that the correct bond distance is being printed)
    if atom_pair_for_bond_distance_printout is not None:
        atom_nums = atom_pair_for_bond_distance_printout
        bond_name = f"{atom_species[atom_nums[0]-1]}{atom_nums[0]}-{atom_species[atom_nums[1]-1]}{atom_nums[1]}"
        log_print(f"Printing bond distances for {bond_name} in each job:")
        for job in jobs.children:
            if not has_eef or eef_pairs[0][0] in job.name:
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
        x_prop_list,
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
        cp_data = get_bcp_properties(
            job, atom_pairs, unrestricted=unrestricted)
        total_cp_data.extend(cp_data)

    write_csv(total_cp_data, path)
    extra_keys = [s.replace(" (reverse)", "") for s in x_prop_list]
    write_csv(total_cp_data, path, include_keys=reduced_csv_keys + extra_keys, rename_key_map=reduced_csv_rename_key_map)

    generate_plots(
        total_cp_data,
        prop_list,
        x_prop_list,
        os.path.dirname(path),
        combined_plots_y_prop_lists,
        atom_pairs
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
    extra_keys = [s.replace(" (reverse)", "") for s in x_prop_list]
    write_csv(cp_data, job_path, include_keys=reduced_csv_keys + extra_keys, rename_key_map=reduced_csv_rename_key_map)


def test_post_processing_multiple_jobs(dill_path, atom_pairs, unrestricted=False, x_prop_list=plot_x_prop_list):
    # every directory in jobs_path can be loaded using AMSJob.load_external
    jobs = load(dill_path)
    process_results(
        jobs,
        atom_pairs,
        dill_path,
        plot_y_prop_list,
        x_prop_list,
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
            float_cols = df_full.select_dtypes(
                include=[np.float64, np.float32]).columns
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
                        for bcp2 in bcps[i + 1:]:
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
                all_system_crossings[first_var_system_key][system_eef_pairs[0]].keys(
                )
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
                
                # Get original x-values from the interpolation object for start and end points
                original_x_for_energy = energy_interp.x
                
                # Calculate E_start and E_end
                E_start = energy_interp(original_x_for_energy[0])
                E_end = energy_interp(original_x_for_energy[-1])

                # Find E_TS (max energy) using dense sampling
                x_vals_dense_sampling = np.linspace(original_x_for_energy[0], original_x_for_energy[-1], 1000)
                y_vals_dense_sampling = energy_interp(x_vals_dense_sampling)
                max_idx = np.argmax(y_vals_dense_sampling)
                R_TS = x_vals_dense_sampling[max_idx]
                E_TS = y_vals_dense_sampling[max_idx]

                # Calculate reaction barrier energy
                reaction_barrier_energy = E_TS - max(E_start, E_end)

                # For each crossing in the ordered list
                for bcp_crossing in ordered_crossings:
                    row = {
                        "SYSTEM": system,
                        "Reaction Barrier Energy": reaction_barrier_energy,
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
                        row[f"$d_{{\\rm{{{bcp_name}}}}}$"] = np.float64(
                            dist_interp(R))

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
            set(f"{system}_{eef}" for system, eef in zip(
                df_full["SYSTEM"], df_full["EEF"]))
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
        bcp_interps = prepare_bcp_interpolations(
            df_full, rxn_coord_name, system_eef_pairs)

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
        corr_matrix.to_csv(os.path.join(
            output_dir, f"correlation_matrix_{unique_sys}.csv"))

        # Reset matplotlib parameters to default
        plt.rcParams.update(
            {"text.usetex": False, "font.family": "sans-serif"})

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
                f.write(
                    f"PC{i+1}: {var:.4f} ({cum_var_ratio[i]:.4f} cumulative)\n")

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
        circle = plt.Circle((0, 0), 1, fill=False,
                            linestyle="--", color="gray")
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
        system_markers = dict(
            zip(unique_systems, markers[: len(unique_systems)]))

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
        # Show first 5 PCs or all if less than 5
        num_pcs = min(5, len(loadings))
        loadings_df = pd.DataFrame(
            loadings[:num_pcs].T,
            columns=[f"PC{i+1}" for i in range(num_pcs)],
            index=feature_cols,
        )
        sns.heatmap(loadings_df, cmap="RdBu_r",
                    center=0, annot=True, fmt=".2f")
        plt.title(f"PCA Loadings Heatmap (systems: {unique_sys})")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"loadings_heatmap_{unique_sys}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Save loadings to CSV
        loadings_df.to_csv(os.path.join(
            output_dir, f"pca_loadings_{unique_sys}.csv"))

        # Create DataFrame with transformed features
        pca_df = pd.DataFrame(
            X_pca, columns=[f"PC{i+1}" for i in range(pca.n_components_)], index=X.index
        )

        # Reset matplotlib parameters to default
        plt.rcParams.update(
            {"text.usetex": False, "font.family": "sans-serif"})

        return pca, pca_df

    def perform_boruta(df: pd.DataFrame,
                       target_name: str,
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
        # Identify target column
        target_col = numeric_df.columns.get_loc(target_name)
        feature_cols = [col for i, col in enumerate(
            numeric_df.columns) if i != target_col]

        unique_sys = "+".join(df["SYSTEM"].unique().tolist())
        if any("{A" in c or "{B" in c for c in df.columns):
            unique_sys += "_spin"
        if any("theta" in c or "phi" in c for c in df.columns):
            unique_sys += "_directionality"

        print(
            f"Performing Boruta feature selection for target column: {target_name}, {target_col = }")

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
            f.write(
                f"\nTotal tentative features: {len(tentative_features)}\n\n")

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
        columns_to_keep = ['SYSTEM', 'BCP_CROSSING', target_col]
        if 'EEF' not in selected_features + tentative_features:
            columns_to_keep.append('EEF')

        # Add the features and target
        columns_to_keep.extend(selected_features +
                               tentative_features)

        # Create the new DataFrame in one operation
        selected_tentative_df = df[columns_to_keep].copy()

        # First create list of columns we want to keep
        columns_to_keep = ['SYSTEM', 'BCP_CROSSING', target_col]
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
        selected_df.to_csv(os.path.join(
            output_dir, f'selected_df_{unique_sys}.csv'), index=False)
        selected_tentative_df.to_csv(os.path.join(
            output_dir, f'selected_tentative_df_{unique_sys}.csv'), index=False)

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

    # df_full["EEF"] = df_full["JOB_NAME"].apply(
    #     lambda x: -1 if "revEEF" in x else (0 if "noEEF" in x else 1)
    # )
    df_full["SYSTEM"] = df_full["JOB_NAME"].apply(lambda x: x.split("_")[0])

    # get unique SYSTEM values
    sys_list = df_full["SYSTEM"].unique().tolist()
    sys_whitelists = [[sys] for sys in sys_list] + [sys_list]
    directionality_spin = [False, True]
    alpha_list = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    perc_list = [100, 95, 90, 85, 80]

    alpha_list = [0.01, 0.025, 0.05, 0.1]
    perc_list = [100, 95, 90]

    target_col = 'Reaction Barrier Energy'

    for sys_whitelist in sys_whitelists:
        for use_theta_phi_crossings in directionality_spin:
            for use_spin_crossings in directionality_spin:
                print(
                    f"\nsys_whitelist: {sys_whitelist}, use_theta_phi_crossings: {use_theta_phi_crossings}, use_spin_crossings: {use_spin_crossings}")

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
                df.to_csv(
                    f"{output_dir}/{system_string}/{system_string}_bcp_crossing_data.csv", index=False)

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
                        print(
                            f"System: {system_string}, perc: {perc}, alpha: {alpha}")

                        boruta_string = f"perc_{perc}_alpha_{alpha}"

                        selected_df, selected_tentative_df, boruta = perform_boruta(
                            df, target_col, f"{output_dir}/{system_string}/boruta_{boruta_string}", perc=perc, alpha=alpha
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
        if len(plot_x_prop_lists) == 0:
            for restart_dill_path, atom_pairs in zip(restart_dill_paths, atom_pairs_list):
                test_post_processing_multiple_jobs(
                    restart_dill_path, atom_pairs, unrestricted=unrestricted_calculation
                )
        else:
            for restart_dill_path, atom_pairs, x_prop_list in zip(restart_dill_paths, atom_pairs_list, plot_x_prop_lists):
                test_post_processing_multiple_jobs(
                    restart_dill_path, atom_pairs, unrestricted=unrestricted_calculation, x_prop_list=x_prop_list
                )
    else:
        log_print("Running ADF NEB BCP analysis on provided job paths...")
        if len(plot_x_prop_lists) == 0:    
            for job_path, atom_pairs in zip(ams_job_paths, atom_pairs_list):
                main(job_path, atom_pairs)
        else:
            for job_path, atom_pairs, x_prop_list in zip(ams_job_paths, atom_pairs_list, plot_x_prop_lists):
                main(job_path, atom_pairs, x_prop_list)
            
