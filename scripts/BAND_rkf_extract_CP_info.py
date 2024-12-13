#!/usr/local/bin/python3

import csv
import json
import os
import argparse
from math import sqrt, atan
import subprocess # for running amsreport
import re # for parsing amsreport output

# Initial CP plots
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy import stats

# Correlation matrices
import seaborn as sns
import pandas as pd

# Feature rejection
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor

# PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BOHR_TO_ANG = 0.529177210903

# https://imechanica.org/comment/7760 http://imechanica.org/files/images/ElasticConstantsCubicCrystal.jpg
# Atomic and Electronic Structure of Solids
# By Efthimios Kaxiras
# Rhodium: using PBEsol data from Table II in https://pubs.aip.org/aip/adv/article/14/4/045229/3282885/Thermodynamic-properties-of-rhodium-A-first
MAT_PROPS = {
    "Ir": {
        "structure": "FCC",
        "C11": 5.8,
        "C12": 2.42,
        "C44": 2.56
    },
    "W": {
        "structure": "BCC",
        "C11": 5.224,
        "C12": 2.044,
        "C44": 1.608
    },
    "Rh": {
        "structure": "FCC",
        "C11": 4.76,
        "C12": 2.09,
        "C44": 2.15
    },
    "Mo": {
        "structure": "BCC",
        "C11": 4.637,
        "C12": 1.578,
        "C44": 1.092
    },
    "Pt": {
        "structure": "BCC",
        "C11": 3.467,
        "C12": 2.507,
        "C44": 0.765
    },
    "Cr": {
        "structure": "BCC",
        "C11": 3.398,
        "C12": 0.586,
        "C44": 0.990
    },
    "Ta": {
        "structure": "BCC",
        "C11": 2.602,
        "C12": 1.545,
        "C44": 0.826
    },
    "Ni": {
        "structure": "FCC",
        "C11": 2.481,
        "C12": 1.549,
        "C44": 1.242
    },
    "Nb": {
        "structure": "BCC",
        "C11": 2.465,
        "C12": 1.345,
        "C44": 0.287
    },
    "Fe": {
        "structure": "BCC",
        "C11": 2.26,
        "C12": 1.4,
        "C44": 1.16
    },
    "V": {
        "structure": "BCC",
        "C11": 2.287,
        "C12": 1.19,
        "C44": 0.432
    },
    "Pd": {
        "structure": "FCC",
        "C11": 2.271,
        "C12": 1.76,
        "C44": 0.717
    },
    "Cu": {
        "structure": "FCC",
        "C11": 1.683,
        "C12": 1.221,
        "C44": 0.757
    },
    "Ag": {
        "structure": "FCC",
        "C11": 1.24,
        "C12": 0.937,
        "C44": 0.461
    },
    "Au": {
        "structure": "FCC",
        "C11": 1.924,
        "C12": 1.630,
        "C44": 0.420
    },
    "Al": {
        "structure": "FCC",
        "C11": 1.067,
        "C12": 0.604,
        "C44": 0.283
    },
    "Pb": {
        "structure": "FCC",
        "C11": 0.497,
        "C12": 0.423,
        "C44": 0.150
    },
    "Li": {
        "structure": "BCC",
        "C11": 0.135,
        "C12": 0.114,
        "C44": 0.088
    },
    "Na": {
        "structure": "BCC",
        "C11": 0.074,
        "C12": 0.062,
        "C44": 0.042
    },
    "K": {
        "structure": "BCC",
        "C11": 0.037,
        "C12": 0.031,
        "C44": 0.019
    },
    "Rb": {
        "structure": "BCC",
        "C11": 0.030,
        "C12": 0.025,
        "C44": 0.017
    },
    "Cs": {
        "structure": "FCC",
        "C11": 0.025,
        "C12": 0.021,
        "C44": 0.015
    }
}

# update MAT_PROPS to compute C-prime ((C11-C12)/2) for each material
for mat, props in MAT_PROPS.items():
    props["Cprime"] = (props["C11"] - props["C12"]) / 2


def amsreport(f, sec, var):
    """
    Execute amsreport command and return its output directly from stdout.

    Args:
        f (str): RKF filename
        sec (str): Section name
        var (str): Variable name

    Returns:
        str: Output from amsreport command
    """
    try:
        cmd = f'amsreport {f} -r "{sec}%{var}"'
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"amsreport command failed: {stderr.decode()}")

        out = stdout.decode().strip()
        if "not found" in out or len(out) == 0:
            raise RuntimeError(
                f"Variable {var} not found in section {sec} for file {f}"
            )

        return out

    except Exception as e:
        raise RuntimeError(f"Error running amsreport: {str(e)}")


def angle_between(v1, v2):
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if not np.isnan(denom) and denom > 0:
        arg = abs(np.dot(v1, v2)) / denom
        # Clamp arg between 0 and 1 to avoid numerical issues
        arg = np.clip(arg, 0.0, 1.0)
        return np.arccos(arg)
    else:
        return 0.0


def extract_bond_paths(file_path, cp_coords):
    num_bps = int(amsreport(file_path, "Properties", "BP number of"))
    num_bp_steps = [
        int(i) for i in amsreport(file_path, "Properties", "BP step number").split()
    ]
    max_num_bp_steps = max(num_bp_steps)
    group_len = max_num_bp_steps * num_bps
    bp_props = amsreport(file_path, "Properties", "BPs and their properties").split()

    bond_paths = {}

    # Properties per point (x,y,z, density, grad[x,y,z], hess[xx,xy,xz,yy,yz,zz])
    num_properties = 13

    # Calculate total points across all paths
    total_points = sum(num_bp_steps)
    values_per_path = num_properties * num_bp_steps[0]

    for path_idx in range(num_bps):
        bond_path = {"coords": [], "density": [], "grad": [], "hessian": []}
        points = []
        for step_num in range(num_bp_steps[path_idx]):
            bond_path["coords"].append(
                [
                    float(bp_props[path_idx + i * group_len + step_num * num_bps])
                    * BOHR_TO_ANG
                    for i in range(3)
                ]
            )
            bond_path["density"].append(
                float(bp_props[path_idx + 3 * group_len + step_num * num_bps])
            )
            bond_path["grad"].append(
                [
                    float(bp_props[path_idx + (4 + i) * group_len + step_num * num_bps])
                    for i in range(3)
                ]
            )
            bond_path["hessian"].append(
                [
                    float(bp_props[path_idx + (7 + i) * group_len + step_num * num_bps])
                    for i in range(6)
                ]
            )

            points.append(
                {
                    "coords": bond_path["coords"][-1],
                    "density": bond_path["density"][-1],
                    "grad": bond_path["grad"][-1],
                    "hessian": bond_path["hessian"][-1],
                }
            )
        bond_path["points"] = points

        # now find closest CP to the bond path
        closest_cp = -1
        closest_dist = 1e100
        for cp_idx, cp_coord in enumerate(cp_coords):
            for point in bond_path["coords"]:
                dist = sqrt(sum((cp_coord[i] - point[i]) ** 2 for i in range(3)))
                if dist < closest_dist:
                    closest_dist = dist
                    closest_cp = cp_idx

        # Now compute length of the bond path
        length = 0
        for i in range(1, len(bond_path["coords"])):
            length += sqrt(
                sum(
                    (bond_path["coords"][i][j] - bond_path["coords"][i - 1][j]) ** 2
                    for j in range(3)
                )
            )
        bond_path["length"] = length

        # Now compute curvature of the bond path (start-end point distance / length)
        curvature = (
            sqrt(
                sum(
                    (bond_path["coords"][-1][i] - bond_path["coords"][0][i]) ** 2
                    for i in range(3)
                )
            )
            / length
        )
        bond_path["curvature (length ratio)"] = curvature

        # Now compute angle difference between the first and last segments of the bond path.
        # This is the angle between the vectors from the first to second point and the second to last point.
        curvature_angle = angle_between(
            np.array(bond_path["coords"][1]) - np.array(bond_path["coords"][0]),
            np.array(bond_path["coords"][-1]) - np.array(bond_path["coords"][-2]),
        )
        bond_path["curvature (end angle)"] = curvature_angle

        bond_paths[closest_cp + 1] = bond_path

    return bond_paths


def parse_cp_list(cp_list_str):
    cp_list = []
    for part in cp_list_str.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            cp_list.extend(range(start, end + 1))
        else:
            cp_list.append(int(part))
    return cp_list


cp_type_code_map = {1: "nuclear", 2: "cage", 3: "bond", 4: "ring"}
cp_sig_code_map = {1: "-3", 2: "+3", 3: "-1", 4: "+1"}
cp_sig_typename_map = {-3: "nuclear", 3: "cage", -1: "bond", 1: "ring"}

def parse_cp_info(
    file_path,
    cp_list_str=None,
    combine_degenerate_cps=True,
    degenerate_cp_cutoff_diff=1e-6,
):
    cp_data = []

    # Get CP properties
    num_cps = int(amsreport(file_path, "Properties", "CP number of"))
    cp_type_codes = [
        int(float(i))
        for i in amsreport(
            file_path, "Properties", "CP code number for (Rank,Signatu"
        ).split()
    ]
    cp_types = [cp_type_code_map[cp] for cp in cp_type_codes]
    cp_coords = [
        float(i) * BOHR_TO_ANG
        for i in amsreport(file_path, "Properties", "CP coordinates").split()
    ]
    # cp_coords are a list of strings, first all the x positions, then all the y positions, then all the z positions.
    # Want to group them into a list of lists, where each sublist is [x, y, z]
    cp_coords = [cp_coords[i : i + 2 * num_cps + 1 : num_cps] for i in range(num_cps)]
    cp_density = amsreport(file_path, "Properties", "CP density at").split()
    cp_grad = amsreport(file_path, "Properties", "CP density gradient at").split()
    cp_grad = [cp_grad[i : i + 2 * num_cps + 1 : num_cps] for i in range(num_cps)]
    cp_hessian = [
        float(i)
        for i in amsreport(file_path, "Properties", "CP density Hessian at").split()
    ]
    cp_hessian = [
        cp_hessian[i : i + 5 * num_cps + 1 : num_cps] for i in range(num_cps)
    ]  # 6 values per CP: XX, XY, XZ, YY, YZ, ZZ

    # Get BP (bond path) properties
    try:
        bp_data = extract_bond_paths(file_path, cp_coords)
    except Exception as e:
        print(f"Error extracting bond paths: {str(e)}")
        bp_data = {}

    # Combine degenerate CPs if requested, showing the number of degenerate CPs in a column "Number of CPs"
    degenerate_cp_reverse_map = {}
    if combine_degenerate_cps:
        # We'll determine degenerate CPs based on the absolute difference between their Rho values and type.
        # If the difference is less than `degenerate_cp_cutoff_diff`, we'll consider them degenerate.
        degenerate_cp_count = {}
        for cp_idx in range(num_cps):
            if cp_idx in degenerate_cp_reverse_map:
                continue
            degenerate_cp_count[cp_idx] = 1
            for cp_idx2 in range(cp_idx + 1, num_cps):
                if cp_idx2 in degenerate_cp_reverse_map:
                    continue
                if (
                    abs(float(cp_density[cp_idx]) - float(cp_density[cp_idx2]))
                    < degenerate_cp_cutoff_diff
                    and cp_types[cp_idx] == cp_types[cp_idx2]
                ):
                    degenerate_cp_count[cp_idx] += 1
                    degenerate_cp_reverse_map[cp_idx2] = cp_idx

    # Parse the CP list if provided
    if cp_list_str:
        cp_list = parse_cp_list(cp_list_str)
    else:
        cp_list = [i + 1 for i in range(num_cps)]

    if combine_degenerate_cps:
        for cp_i in range(len(cp_list)):
            cp_idx = cp_list[cp_i] - 1
            if cp_idx in degenerate_cp_reverse_map:
                cp_list[cp_i] = degenerate_cp_reverse_map[cp_idx] + 1

    # Now prepare cp_data
    # Also, for bond and ring CPs, compute the directionality using the eigenvalues.
    for cp_idx in range(num_cps):
        cp_idx1 = cp_idx + 1
        if cp_idx1 not in cp_list or (cp_idx in degenerate_cp_reverse_map):
            continue
        cp_dict = {
            "CP #": cp_idx1,
            "RANK": 3,
            "SIGNATURE": cp_sig_code_map[cp_type_codes[cp_idx]],
            "CP COORDINATES_X": cp_coords[cp_idx][0],
            "CP COORDINATES_Y": cp_coords[cp_idx][1],
            "CP COORDINATES_Z": cp_coords[cp_idx][2],
        }
        if combine_degenerate_cps:
            cp_dict["Number of CPs"] = degenerate_cp_count[cp_idx]
        if cp_types[cp_idx] != "nuclear":
            cp_dict.update(
                {
                    "Rho": float(cp_density[cp_idx]),
                    "|GRAD(Rho)|": sqrt(sum(float(i) ** 2 for i in cp_grad[cp_idx])),
                    "GRAD(Rho)x": cp_grad[cp_idx][0],
                    "GRAD(Rho)y": cp_grad[cp_idx][1],
                    "GRAD(Rho)z": cp_grad[cp_idx][2],
                }
            )
            hessian6 = cp_hessian[cp_idx]
            hessian = [
                [hessian6[0], hessian6[1], hessian6[2]],
                [hessian6[1], hessian6[3], hessian6[4]],
                [hessian6[2], hessian6[4], hessian6[5]],
            ]
            eig = np.linalg.eigh(np.array(hessian))
            ev = eig[0]
            if cp_types[cp_idx] == "bond":
                theta = atan(sqrt(abs(ev[0] / ev[2])))
                phi = atan(sqrt(abs(ev[1] / ev[2])))
            else: # ring or cage
                theta = atan(sqrt(abs(ev[2] / ev[0])))
                phi = atan(sqrt(abs(ev[1] / ev[0])))
            laplacian = sum(ev)
            cp_dict.update(
                {
                    "HESSIAN MATRIX": hessian,
                    "EIGENVALUES": ev.tolist(),
                    "EIGENVECTORS": eig[
                        1
                    ].tolist(),
                    "Laplacian": laplacian,
                    "Theta": theta,
                    "Phi": phi,
                }
            )
            if cp_types[cp_idx] == "bond" and cp_idx1 in bp_data:
                cp_dict.update(
                    {
                        "Bond Path Length": bp_data[cp_idx1]["length"],
                        "Bond Path Curvature (length ratio)": bp_data[cp_idx1][
                            "curvature (length ratio)"
                        ],
                        "Bond Path Curvature (end angle)": bp_data[cp_idx1][
                            "curvature (end angle)"
                        ],
                    }
                )
        cp_data.append(cp_dict)

    return cp_data


def write_csv(cp_data, input_file_path, out_file_prefix=""):
    # Determine all keys present in the data
    all_keys = set()
    for cp in cp_data:
        for key in cp.keys():
            if isinstance(cp[key], list):
                if isinstance(cp[key][0], list):  # 2D list
                    if key == "EIGENVECTORS":
                        all_keys.update(
                            [
                                f"{key}_EV{i+1}_{axis}"
                                for i in range(3)
                                for axis in ["X", "Y", "Z"]
                            ]
                        )
                    elif key == "HESSIAN MATRIX":
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
        "CP #",
        "RANK",
        "SIGNATURE",
        "CP COORDINATES_X",
        "CP COORDINATES_Y",
        "CP COORDINATES_Z",
        "Rho",
        "Laplacian",
        "Theta",
        "Phi",
    ]

    if "Number of CPs" in all_keys:
        preferred_order.insert(3, "Number of CPs")
    if "Job name" in all_keys:
        preferred_order.insert(0, "Job name")

    # Now add Eigenvalues and Eigenvectors
    for h in [s for s in all_keys if "EIGENVALUES" in s]:
        preferred_order.append(h)
    for h in [s for s in all_keys if "EIGENVECTORS" in s]:
        preferred_order.append(h)

    # Then Hessian
    for h in [s for s in all_keys if "HESSIAN MATRIX" in s and "EIGEN" not in s]:
        preferred_order.append(h)

    # Now add the rest if not already in the preferred order
    for h in all_keys:
        if h not in preferred_order:
            preferred_order.append(h)

    headers = preferred_order

    # Define the output CSV file path
    if os.path.isdir(input_file_path):
        base_name = input_file_path + "/"
        csv_file_path = f"{base_name}{out_file_prefix}cp_info.csv"
    else:
        base_name = os.path.splitext(input_file_path)[0]
        csv_file_path = f"{base_name}{out_file_prefix}_cp_info.csv"

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
                            "EIGENVECTORS"
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

    print(f"CSV file written to {csv_file_path}")
    return csv_file_path

#### END OF CP DATA EXTRACTION CODE; BEGIN CP ANALYSIS CODE ####

PROP_LIST = ['Rho', 'Laplacian', 'Theta', 'Phi']
DISPLACEMENT_STR = 'Displacement'
USE_ALL_DELTA_PROP = True

def group_systems(all_cp_data):
    """
    Group critical point data by system based on Job name patterns.
    
    Args:
        all_cp_data (list): List of dictionaries containing critical point data
            where each dictionary must have a 'Job name' key
    
    Returns:
        dict: Dictionary where keys are system names and values are lists of 
            critical point data dictionaries belonging to that system
    """
    all_systems = {}
    
    for cp_data in all_cp_data:
        job_name = cp_data["Job name"]
        parts = job_name.split('_')
        
        # Handle different possible formats of job names
        if len(parts) >= 2:
            # Base case: at least distortion_type and system (e.g., "C11_Al")
            base_system = f"{parts[0]}_{parts[1]}"
            
            # Check if there's a trailing letter identifier
            if len(parts) >= 3 and not parts[-1].isdigit():
                # Case like "C11_Al_a" or "C11_Al_02_a"
                system_name = f"{base_system}_{parts[-1]}"
            else:
                # Case like "C11_Al" or "C11_Al_02"
                system_name = base_system
                
            # Add to dictionary, creating new list if system not seen before
            if system_name not in all_systems:
                all_systems[system_name] = []
            all_systems[system_name].append(cp_data)
        else:
            # Handle invalid job names
            print(f"Warning: Invalid job name format: {job_name}")
            
    return all_systems

def add_job_columns(all_cp_data):
    """
    Add distortion type, system, and percentage to each item in all_cp_data based on Job name.
    
    Args:
        all_cp_data (list): List of dictionaries containing critical point data
            where each dictionary must have a 'Job name' key
    
    Returns:
        list: The same list with'distortion_type', 'system', and 'distortion_percent' 
        added to each dictionary
    """
    for cp_data in all_cp_data:
        job_name = cp_data["Job name"]
        parts = job_name.split('_')
        
        # Get distortion type (first part)
        cp_data["distortion_type"] = parts[0]
        
        # Find system (part that matches a key in MAT_PROPS)
        system = next((part for part in parts if part in MAT_PROPS), None)
        if system is None:
            print(f"Warning: No recognized system found in job name: {job_name}")
        if not parts[-1].isdecimal() and parts[-1] != system:
            system_part_index = parts.index(system)
            system += f"_{'_'.join(parts[system_part_index + 1:])}"
        cp_data["system"] = system
        
        # Initialize distortion percent to 0 (undistorted case)
        distortion_percent = 0
        
        # Look for numerical value in parts
        for part in parts[2:]:  # Skip distortion_type and system name
            try:
                # Convert to integer, keeping as percentage
                distortion_percent = int(part)
                break  # Stop at first number found
            except ValueError:
                continue
                
        cp_data["distortion_percent"] = distortion_percent
    
    return all_cp_data

def map_critical_points(cp_group):
    """
    Map critical points across different distortion levels within a group.
    
    Args:
        cp_group (list): List of dictionaries containing critical point data
            for a single system at different distortion levels
    
    Returns:
        dict: Keys are CP numbers from the relaxed system, values are lists of
            corresponding CP dictionaries ordered by increasing distortion
    """
    import numpy as np
    
    # First, sort the group by distortion percentage
    sorted_group = sorted(cp_group, key=lambda x: x["distortion_percent"])
    
    # Separate CPs by distortion level
    distortion_levels = {}
    for cp_dict in sorted_group:
        dist_percent = cp_dict["distortion_percent"]
        if dist_percent not in distortion_levels:
            distortion_levels[dist_percent] = []
        distortion_levels[dist_percent].append(cp_dict)
    
    # Initialize the mapping with relaxed system CPs
    cp_mapping = {}
    if 0 in distortion_levels:
        for cp_dict in distortion_levels[0]:
            cp_dict["mapped_distance"] = 0.0  # Add 0.0 distance for relaxed system
            cp_mapping[cp_dict["CP #"]] = [cp_dict]
    else:
        return {}  # Return empty dict if no relaxed system found
    
    # Get ordered list of distortion percentages (excluding 0)
    dist_percentages = sorted(p for p in distortion_levels.keys() if p > 0)
    
    # For each CP in the relaxed system, find corresponding CPs at each distortion level
    for cp_num, cp_list in cp_mapping.items():
        last_cp = cp_list[0]  # Start with relaxed system CP
        
        for dist_percent in dist_percentages:
            # Get coordinates of the last matched CP
            last_coords = np.array([
                last_cp["CP COORDINATES_X"],
                last_cp["CP COORDINATES_Y"],
                last_cp["CP COORDINATES_Z"]
            ])
            
            # Find closest CP of same type in current distortion level
            min_dist = float('inf')
            closest_cp = None
            
            for cp_dict in distortion_levels[dist_percent]:
                # Check if same signature (CP type)
                if cp_dict["SIGNATURE"] != last_cp["SIGNATURE"]:
                    continue
                curr_coords = np.array([
                    cp_dict["CP COORDINATES_X"],
                    cp_dict["CP COORDINATES_Y"],
                    cp_dict["CP COORDINATES_Z"]
                ])
                
                dist = np.linalg.norm(curr_coords - last_coords)
                if dist < min_dist:
                    min_dist = dist
                    closest_cp = cp_dict
            
            if closest_cp is not None:
                closest_cp["mapped_distance"] = float(min_dist)  # Store the mapping distance
                cp_list.append(closest_cp)
                last_cp = closest_cp
    
    return cp_mapping

def map_all_systems(all_systems):
    """
    Apply CP mapping to all system groups.
    
    Args:
        all_systems (dict): Dictionary of CP groups as returned by group_systems()
    
    Returns:
        dict: Dictionary with same keys as all_systems, but values are the mapped
            CP dictionaries as returned by map_critical_points()
    """
    mapped_systems = {}
    
    for system_name, cp_group in all_systems.items():
        mapped_systems[system_name] = map_critical_points(cp_group)
    
    return mapped_systems

def remove_degenerate_cps(mapped_systems, rho_tolerance=2e-3):
    """
    Remove degenerate critical points from mapped_systems and add degeneracy count.
    
    Args:
        mapped_systems (dict): Dictionary of mapped CP systems as returned by map_all_systems()
        rho_tolerance (float): Tolerance for considering two Rho values equal
    
    Returns:
        dict: Same structure as mapped_systems but with degenerate CPs removed and
            'Number of CPs' added to indicate degeneracy
    """
    unique_systems = {}
    
    for system_name, cp_mapping in mapped_systems.items():
        unique_mapping = {}
        processed_cps = set()  # Track which CPs have been processed
        
        # Get all CP numbers in this system
        cp_numbers = list(cp_mapping.keys())
        
        for cp_num in cp_numbers:
            if cp_num in processed_cps:
                continue
                
            cp_list = cp_mapping[cp_num]
            signature = cp_list[0]["SIGNATURE"]
            
            # Find all CPs with same signature that haven't been processed
            candidate_cp_nums = [
                n for n in cp_numbers 
                if n not in processed_cps 
                and cp_mapping[n][0]["SIGNATURE"] == signature
            ]
            
            # Group degenerate CPs
            degenerate_groups = {cp_num: []}  # Initialize with current CP
            
            for candidate_num in candidate_cp_nums:
                if candidate_num == cp_num:
                    continue
                candidate_list = cp_mapping[candidate_num]
                
                # Check if candidate is degenerate with current CP across all distortions
                is_degenerate = True
                if len(cp_list) != len(candidate_list):
                    is_degenerate = False
                else:
                    for cp1, cp2 in zip(cp_list, candidate_list):
                        if abs(cp1["Rho"] - cp2["Rho"]) > rho_tolerance:
                            is_degenerate = False
                            break
                if is_degenerate:
                    degenerate_groups[cp_num].append(candidate_num)
                    processed_cps.add(candidate_num)
            
            # Add Number of CPs to all CPs in the list
            num_cps = len(degenerate_groups[cp_num]) + 1
            for cp in cp_list:  # Fixed typo here
                cp["Number of CPs"] = num_cps
            
            # Add to unique mapping
            unique_mapping[cp_num] = cp_list
            processed_cps.add(cp_num)
        
        unique_systems[system_name] = unique_mapping
    
    return unique_systems

def add_rho_rankings(mapped_systems):
    """
    Add rho_rank to each CP group based on the Rho value ordering of most distorted CPs
    within each signature type.
    
    We do this because all the systems being considered are FCC, and their symmetry
    is broken in the same way the distortions, so the "rho ranking" of the CPs should
    be the same across all systems and can thus be used to compare CPs across systems.
    
    Args:
        mapped_systems (dict): Dictionary of unique CP systems as returned by 
            remove_degenerate_cps() or map_all_systems()
    
    Returns:
        dict: Same structure as input but with 'rho_rank' added to each CP in
            each CP list
    """
    ranked_systems = {}
    
    for system_name, cp_mapping in mapped_systems.items():        
        ranked_mapping = {}
        
        # Group CPs by signature
        signature_groups = {}
        
        # First, group CPs by signature and get their most distorted Rho values
        for cp_num, cp_list in cp_mapping.items():
            # Skip CPs whose `SIGNATURE` is -3 (nuclear)
            if cp_list[0]['SIGNATURE'] == '-3':
                ranked_mapping[cp_num] = cp_list
                continue
            
            signature = cp_list[0]['SIGNATURE']
            if signature not in signature_groups:
                signature_groups[signature] = []
            
            # Get the most distorted CP's Rho value
            comp_cp = cp_list[-1]  # Last CP in list is most distorted
            signature_groups[signature].append({
                'cp_num': cp_num,
                'rho': comp_cp['Rho'],
                'cp_list': cp_list
            })
        
        # Sort each signature group by Rho value and assign ranks
        for signature in signature_groups:
            # Sort CPs within this signature group by Rho value
            sorted_cps = sorted(signature_groups[signature],
                              key=lambda x: x['rho'])
            
            # Add rank to each CP in the group
            for rank, cp_info in enumerate(sorted_cps):
                cp_num = cp_info['cp_num']
                cp_list = cp_info['cp_list']
                
                # Create new list with rank added to each CP
                ranked_cp_list = []
                for cp in cp_list:
                    cp_with_rank = cp.copy()
                    cp_with_rank['rho_rank'] = rank
                    ranked_cp_list.append(cp_with_rank)
                
                ranked_mapping[cp_num] = ranked_cp_list
        
        ranked_systems[system_name] = ranked_mapping
    
    return ranked_systems

def plot_cp_properties(mapped_systems, output_dir, output_format='jpg'):
    """
    Create and save 2D grids of plots showing CP properties vs distortion.
    For each distortion type and system, creates a grid with properties as columns
    and CPs as rows.
    
    Args:
        mapped_systems (dict): Dictionary of mapped CP systems as returned by map_all_systems()
        output_dir (str): Directory where plot files should be saved
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Group systems by distortion type
    distortion_types = {}
    for system_name, cp_mapping in mapped_systems.items():
        # Get first CP's distortion type and system
        first_cp = next(iter(cp_mapping.values()))[0]
        dist_type = first_cp['distortion_type']
        system = first_cp['system']
        
        if dist_type not in distortion_types:
            distortion_types[dist_type] = {}
        if system not in distortion_types[dist_type]:
            distortion_types[dist_type][system] = cp_mapping
    
    # Layout parameters
    left_margin = 0.1
    right_margin = 0.9
    top_margin = 0.94
    bottom_margin = 0.05
    
    # Additional spacing for labels
    label_padding = 0.02  # Space between grid and labels
    title_height = 0.98   # Y-position of the suptitle
    
    # Derived positions
    left_labels_x = left_margin - label_padding * 2
    right_labels_x = right_margin + label_padding
    top_labels_y = top_margin + label_padding
    bottom_labels_y = bottom_margin - label_padding * 1.8
    
    # Grid width and height in figure coordinates
    grid_width = right_margin - left_margin
    grid_height = top_margin - bottom_margin
    
    # Create plots for each distortion type and system
    for dist_type, systems in distortion_types.items():
        print(f"Creating plots for {dist_type} distortion - ", end='', flush=True)
        for system, cp_mapping in systems.items():
            print(f"{system}, ", end='', flush=True)
            do_skip = False
            
            # Filter out nuclear CPs
            non_nuclear_cps = {
                cp_num: cp_list for cp_num, cp_list in cp_mapping.items()
                if cp_list[0]['SIGNATURE'] != '-3'
            }
            
            # Get sorted order (descending order) of CP numbers based on mean `Rho` value of CP list
            sorted_cp_nums = sorted(
                non_nuclear_cps.keys(),
                key=lambda cp_num: np.mean([cp['Rho'] for cp in non_nuclear_cps[cp_num]]),
                reverse=True
            )
            
            # Set up the plot grid
            n_rows = len(non_nuclear_cps)
            n_cols = len(PROP_LIST)
            fig = plt.figure(figsize=(5*n_cols, 3*n_rows))
            
            # Create GridSpec with space for labels
            gs = GridSpec(n_rows, n_cols,
                         left=left_margin, right=right_margin,
                         top=top_margin, bottom=bottom_margin)
            
            # Add super title with padding
            fig.suptitle(f'{dist_type} Distortion - {system}',
                        fontsize=16, y=title_height)
            
            # Add column headers (PROP_LIST) at top and bottom
            for col, prop in enumerate(PROP_LIST):
                # Calculate x position for column headers
                col_x = left_margin + (col + 0.5) * grid_width / n_cols
                
                # Top labels
                fig.text(col_x, top_labels_y,
                        prop,
                        ha='center', va='bottom',
                        fontsize=14)
                # Bottom labels
                fig.text(col_x, bottom_labels_y,
                        prop,
                        ha='center', va='top',
                        fontsize=14)
            
            # Create plots
            for row, cp_num in enumerate(sorted_cp_nums):
                cp_list = non_nuclear_cps[cp_num]
                # Calculate y position for row labels
                row_y = top_margin - (row + 0.5) * grid_height / n_rows
                
                signature = cp_list[0]['SIGNATURE']
                cp_label = f'CP {cp_num} ({cp_sig_typename_map[int(signature)]})'
                
                # Left labels
                fig.text(left_labels_x, row_y,
                        cp_label,
                        ha='right', va='center',
                        fontsize=12)
                
                # Right labels
                fig.text(right_labels_x, row_y,
                        cp_label,
                        ha='left', va='center',
                        fontsize=12)
                
                for col, prop in enumerate(PROP_LIST):
                    ax = fig.add_subplot(gs[row, col])
                    
                    # Extract x and y data
                    x = [cp['distortion_percent'] for cp in cp_list]
                    y = [cp[prop] for cp in cp_list]
                    
                    if len(x) < 2:
                        # Skip plotting if only one CP
                        do_skip = True
                        break
                    
                    # Perform linear regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    r_squared = r_value**2
                    
                    # Plot data points and fit line
                    ax.scatter(x, y)
                    x_fit = np.array([min(x), max(x)])
                    y_fit = slope * x_fit + intercept
                    ax.plot(x_fit, y_fit, '--', alpha=0.5)
                    
                    # Add equation and R² to title
                    eq = f'y = {slope:.2e}x + {intercept:.2e}, R² = {r_squared:.4f}'
                    ax.set_title(eq, fontsize=10)
                    
                    if row == n_rows - 1:
                        ax.set_xlabel('Distortion %')
                    ax.grid(True, alpha=0.3)
                if do_skip:
                    break
            if do_skip:
                continue
            
            # Save the figure
            plt.savefig(
                os.path.join(output_dir, f'{dist_type} - {system}.{output_format}'),
                bbox_inches='tight',
                dpi=300
            )
            plt.close()
        print("")
            
def write_ranked_csv(ranked_systems, output_path):
    """
    Write ranked CP data to CSV, including Parent CP # column.
    
    Args:
        ranked_systems (dict): Dictionary of ranked CP systems
        output_path (str): Path for output CSV file
    """
    # First collect all possible keys from all CPs
    all_keys = set()
    for system_name, cp_mapping in ranked_systems.items():
        for parent_cp_num, cp_list in cp_mapping.items():
            for cp in cp_list:
                for key in cp.keys():
                    if isinstance(cp[key], list):
                        if isinstance(cp[key][0], list):  # 2D list
                            if key == "EIGENVECTORS":
                                all_keys.update([
                                    f"{key}_EV{i+1}_{axis}"
                                    for i in range(3)
                                    for axis in ["X", "Y", "Z"]
                                ])
                            elif key == "HESSIAN MATRIX":
                                all_keys.update([
                                    f"{key}_{axis1}{axis2}"
                                    for axis1 in ["X", "Y", "Z"]
                                    for axis2 in ["X", "Y", "Z"]
                                ])
                        else:  # 1D list
                            all_keys.update([f"{key}_{axis}" for axis in ["X", "Y", "Z"]])
                    else:
                        all_keys.add(key)
    
    # Define preferred order of columns
    preferred_order = [
        "Job name",
        "distortion_type",
        "system",
        "distortion_percent",
        "Parent CP #",
        "CP #",
        "Number of CPs",
        "rho_rank",
        "RANK",
        "SIGNATURE",
        "CP COORDINATES_X",
        "CP COORDINATES_Y",
        "CP COORDINATES_Z",
        "Rho",
        "Laplacian",
        "Theta",
        "Phi",
    ]
    
    # Add remaining columns in specific order
    for h in [s for s in all_keys if "EIGENVALUES" in s]:
        preferred_order.append(h)
    for h in [s for s in all_keys if "EIGENVECTORS" in s]:
        preferred_order.append(h)
    for h in [s for s in all_keys if "HESSIAN MATRIX" in s and "EIGEN" not in s]:
        preferred_order.append(h)
    
    # Add any remaining keys not in preferred order
    headers = preferred_order + [h for h in sorted(all_keys) if h not in preferred_order]
    
    # Write the CSV file
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        for system_name, cp_mapping in ranked_systems.items():
            for parent_cp_num, cp_list in cp_mapping.items():
                for cp in cp_list:
                    row = {}
                    # Add parent CP number
                    row["Parent CP #"] = parent_cp_num
                    
                    # Process all other fields
                    for key, value in cp.items():
                        if isinstance(value, list):
                            if isinstance(value[0], list):  # 2D list
                                if "EIGENVECTORS" in key:
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
    
    print(f"CSV file written to {output_path}")
    return output_path

def add_delta_properties(mapped_systems, most_distorted_only=True):
    """
    Add delta-property columns to relaxed CPs showing change upon distortion.
    
    Args:
        mapped_systems (dict): Dictionary of mapped CP systems
        most_distorted_only (bool): If True, only compute deltas for most distorted
            system. If False, compute deltas for all non-zero distortions.
    
    Returns:
        dict: Same structure as input but with delta columns added to relaxed CPs
    """
    modified_systems = {}
    
    for system_name, cp_mapping in mapped_systems.items():
        modified_mapping = {}
        
        for parent_cp_num, cp_list in cp_mapping.items():
            modified_cp_list = []
            
            # Get relaxed CP (first in list)
            relaxed_cp = cp_list[0]
            
            cur_coords = np.array([
                relaxed_cp["CP COORDINATES_X"],
                relaxed_cp["CP COORDINATES_Y"],
                relaxed_cp["CP COORDINATES_Z"]
            ])
            
            if most_distorted_only:
                # Get most distorted CP (last in list)
                distorted_cps = [cp_list[-1]]
            else:
                # Get all distorted CPs (all but first)
                distorted_cps = cp_list[1:]
            
            # Create modified relaxed CP with delta properties
            modified_relaxed = relaxed_cp.copy()
            
            # For each distorted CP, compute and add delta properties
            for distorted_cp in distorted_cps:
                dist_percent = distorted_cp['distortion_percent']
                
                # Calculate deltas for each property in PROP_LIST
                for prop in PROP_LIST:
                    if prop in relaxed_cp:  # Skip if property doesn't exist
                        delta = distorted_cp[prop] - relaxed_cp[prop]
                        delta_key = f"{prop}_{dist_percent}"
                        modified_relaxed[delta_key] = delta
                
                # Always include displacement due to distortion
                dist_coords = np.array([
                    distorted_cp["CP COORDINATES_X"],
                    distorted_cp["CP COORDINATES_Y"],
                    distorted_cp["CP COORDINATES_Z"]
                ])
                modified_relaxed[f"{DISPLACEMENT_STR}_{dist_percent}"] = float(np.linalg.norm(dist_coords - cur_coords))
            
            # Add modified relaxed CP as first in list
            modified_cp_list.append(modified_relaxed)
            # Add remaining CPs unchanged
            modified_cp_list.extend(cp_list[1:])
            
            modified_mapping[parent_cp_num] = modified_cp_list
        
        modified_systems[system_name] = modified_mapping
    
    return modified_systems

def prepare_analysis_arrays(ranked_systems, cp_selection='relaxed', use_highest_delta=True):
    """
    Prepare 2D arrays for statistical analysis, organized by distortion type.
    
    Args:
        ranked_systems (dict): Dictionary of ranked CP systems
        cp_selection (str): Which CPs to analyze:
            'relaxed' - only relaxed CPs (distortion_percent = 0)
            'distorted' - only most distorted CPs
            'all' - all CPs across all distortions
    
    Returns:
        dict: Dictionary with distortion types as keys, values are dictionaries containing:
            'data': 2D numpy array of values
            'rows': List of system names
            'columns': List of variable names
    """
    analysis_data = {}
    
    # Group systems by distortion type
    distortion_groups = {}
    for system_name, cp_mapping in ranked_systems.items():
        # Get distortion type and system from first CP
        first_cp = next(iter(cp_mapping.values()))[0]
        dist_type = first_cp['distortion_type']
        system = first_cp['system']
        
        if dist_type not in distortion_groups:
            distortion_groups[dist_type] = {}
        distortion_groups[dist_type][system] = cp_mapping
    
    # Process each distortion type
    for dist_type, systems in distortion_groups.items():
        # Verify consistent CP structure across systems and identify outliers
        structures = {}  # Dictionary to store different structures found
        system_structures = {}  # Map systems to their structure
        
        # Get list of properties starting with DISPLACEMENT_STR
        delta_props = set()
        
        # First collect all structures
        for system, cp_mapping in systems.items():
            system_structure = {}
            for cp_list in cp_mapping.values():
                if cp_list[0]['SIGNATURE'] != '-3':
                    signature = cp_list[0]['SIGNATURE']
                    rho_rank = cp_list[0]['rho_rank']
                    key = (signature, rho_rank)
                    system_structure[key] = system_structure.get(key, 0) + 1
                    for key in cp_list[0].keys():
                        if key.startswith(DISPLACEMENT_STR):
                            delta_props.add(key)
                        else:
                            for prop in PROP_LIST:
                                if key.startswith(f"{prop}_"):
                                    delta_props.add(key)
            
            # Convert dict to tuple of sorted items for hashable key
            structure_key = tuple(sorted(system_structure.items()))
            if structure_key not in structures:
                structures[structure_key] = []
            structures[structure_key].append(system)
            system_structures[system] = structure_key
        
        # Find dominant structure (the one with most systems)
        dominant_structure = max(structures.items(), key=lambda x: len(x[1]))
        excluded_systems = []
        for structure, systems_list in structures.items():
            if structure != dominant_structure[0]:
                excluded_systems.extend(systems_list)
                print(f"Warning: For {dist_type}, excluding systems {systems_list} "
                      f"due to different CP structure")
        
        # Convert dominant structure back to dictionary
        cp_structure = dict(dominant_structure[0])
        
        # Filter out excluded systems
        systems = {sys: cp_map for sys, cp_map in systems.items() 
                  if sys not in excluded_systems}

        # Remove any displacement properties that are not present in all systems
        to_remove = set()
        for system, cp_mapping in systems.items():
            for cp_list in cp_mapping.values():
                if cp_list[0]['SIGNATURE'] != '-3':
                    for key in list(delta_props):
                        if key not in to_remove and key not in cp_list[0]:
                            to_remove.add(key)
        delta_props -= to_remove
        
        # Also only keep the highest value delta property (highest trailing number in key when everything but the number is the same)
        if use_highest_delta:
            to_remove = set()
            for prop in delta_props:
                split_key = prop.split('_')
                if split_key[-1].isnumeric():
                    base_key = '_'.join(split_key[:-1])
                    like_keys = [k for k in delta_props if k.startswith(base_key)]
                    highest_delta = max(like_keys, key=lambda x: int(x.split('_')[-1]))
                    for key in like_keys:
                        if key != highest_delta:
                            to_remove.add(key)
            delta_props -= to_remove
        
        delta_props = sorted(delta_props)
        
        delta_props_sorted = []
        for prop in PROP_LIST:
            for key in delta_props:
                if key.startswith(f"{prop}_"):
                    delta_props_sorted.append(key)
        for key in delta_props:
            if key not in delta_props_sorted:
                delta_props_sorted.append(key)

        prop_list = PROP_LIST + delta_props_sorted
        
        # Prepare data arrays
        rows = []
        columns = ['Elastic Constant']
        data_rows = []
        
        # Build column headers first
        for (signature, rho_rank), count in sorted(cp_structure.items()):
            cp_type = cp_sig_typename_map[int(signature)]
            if count > 1:
                for i in range(count):
                    cp_base = f"{cp_type}({rho_rank+1},{i+1})"
                    for prop in prop_list:
                        columns.append(f"{cp_base}_{prop}")
            else:
                cp_base = f"{cp_type}({rho_rank+1})"
                for prop in prop_list:
                    columns.append(f"{cp_base}_{prop}")
        
        # Build data rows
        for system, cp_mapping in sorted(systems.items()):
            rows.append(system)
            system_element = system.split('_')[0]
            row_data = [MAT_PROPS[system_element][dist_type]]  # Start with elastic constant
            
            # Group CPs by signature and rho_rank
            grouped_cps = {}
            for cp_list in cp_mapping.values():
                if cp_list[0]['SIGNATURE'] != '-3':
                    key = (cp_list[0]['SIGNATURE'], cp_list[0]['rho_rank'])
                    if key not in grouped_cps:
                        grouped_cps[key] = []
                    
                    # Select appropriate CP based on cp_selection
                    if cp_selection == 'relaxed':
                        cp = cp_list[0]
                    elif cp_selection == 'distorted':
                        cp = cp_list[-1]
                    else:  # 'all'
                        # Handle this case if needed
                        raise NotImplementedError("'all' cp_selection not yet implemented")
                    
                    grouped_cps[key].append(cp)
            
            # Add CP properties to row in consistent order
            for (signature, rho_rank), count in sorted(cp_structure.items()):
                cps = sorted(grouped_cps[(signature, rho_rank)], 
                           key=lambda x: x['Rho'])  # Sort by Rho for consistent ordering
                for cp in cps:
                    for prop in prop_list:
                        row_data.append(cp[prop])
            
            data_rows.append(row_data)
        
        # Convert to numpy array and store
        analysis_data[dist_type] = {
            'data': np.array(data_rows, dtype=float),
            'rows': rows,
            'columns': columns
        }
    
    return analysis_data

def create_correlation_matrices(analysis_data, output_dir, output_format='jpg'):
    """
    Create and save correlation matrix plots for each distortion type.
    
    Args:
        analysis_data (dict): Dictionary containing analysis arrays and labels
        output_dir (str): Directory where plots should be saved
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # For each distortion type
    for dist_type, data_dict in analysis_data.items():
        print(f"Creating correlation matrices for {dist_type} distortion")
        
        # Create DataFrame with proper column labels
        df = pd.DataFrame(data_dict['data'], columns=data_dict['columns'])
        
        # Create correlation matrices
        pearson_corr = df.corr(method='pearson')
        spearman_corr = df.corr(method='spearman')
        
        # Create figures for both correlation types
        for corr_type, corr_matrix in [('Pearson', pearson_corr), 
                                      ('Spearman', spearman_corr)]:
            plt.figure(figsize=(12, 10))
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix), k=1)
            
            # Create heatmap
            sns.heatmap(corr_matrix, 
                        mask=mask,
                        cmap='RdBu_r',  # Red-Blue diverging colormap
                        vmin=-1, vmax=1,  # Fix scale from -1 to 1
                        center=0,  # Center colormap at 0
                        square=True,  # Make cells square
                        annot=False,  # Show correlation values
                        fmt='.2f',  # Format correlation values to 2 decimal places
                        xticklabels=data_dict['columns'],  # Force all x-labels
                        yticklabels=data_dict['columns'],  # Force all y-labels
                        cbar_kws={'label': 'Correlation Coefficient'})
            
            # Rotate x-axis labels for better readability, with small font size
            plt.xticks(rotation=60, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            
            # Add title
            plt.title(f'{dist_type} Distortion - {corr_type} Correlation Matrix')
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Save figure
            plt.savefig(
                os.path.join(output_dir, f'{dist_type}_{corr_type}_correlation.{output_format}'),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()

def perform_feature_selection(analysis_data, max_iter=100, perc=85, verbose=False):
    """
    Perform feature selection using Boruta algorithm.
    
    Args:
        analysis_data (dict): Dictionary containing analysis arrays and labels
        max_iter (int): Maximum number of iterations for Boruta
        verbose (bool): Whether to print detailed information
    
    Returns:
        dict: Reduced version of analysis_data with unimportant features removed
    """
    reduced_data = {}
    
    for dist_type, data_dict in analysis_data.items():
        print(f"\nAnalyzing {dist_type} distortion:")
            
        # Separate features and target
        X = data_dict['data'][:, 1:]  # All columns except elastic constant
        y = data_dict['data'][:, 0]   # Elastic constant
        feature_names = data_dict['columns'][1:]  # All columns except elastic constant
        
        # Initialize Random Forest and Boruta
        rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
        feat_selector = BorutaPy(
            rf,
            perc=perc,
            n_estimators='auto',
            max_iter=max_iter,
            verbose=2 if verbose else 0
        )
        
        # Perform Boruta feature selection
        feat_selector.fit(X, y)
        
        if verbose:
            # Print feature ranking information
            print("\nFeature Selection Results:")
            for feat, supported, rank in zip(feature_names, 
                                           feat_selector.support_, 
                                           feat_selector.ranking_):
                status = "Selected" if supported else "Rejected"
                print(f"  {feat}: {status} (rank: {rank})")
            
            print(f"\nNumber of selected features: {sum(feat_selector.support_)}")
        
        # Filter features using transform
        X_filtered = feat_selector.transform(X)
        
        # Include elastic constant column (index 0) and selected features
        selected_cols = [0] + [i + 1 for i, supported in enumerate(feat_selector.support_) 
                             if supported]
        reduced_data[dist_type] = {
            'data': data_dict['data'][:, selected_cols],
            'rows': data_dict['rows'],
            'columns': [data_dict['columns'][i] for i in selected_cols]
        }
        
        if verbose:
            print(f"Reduced from {len(feature_names)} to {X_filtered.shape[1]} features")
    
    return reduced_data

def perform_pca_analysis(analysis_data, output_dir):
    """
    Perform PCA analysis and create visualizations.
    
    Args:
        analysis_data (dict): Dictionary containing analysis arrays and labels
        output_dir (str): Directory where plots should be saved
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for dist_type, data_dict in analysis_data.items():
        print(f"\nPerforming PCA analysis for {dist_type} distortion:")
        
        # Separate elastic constant and features
        X = data_dict['data'][:, 1:]  # All columns except elastic constant
        feature_names = data_dict['columns'][1:]  # All columns except elastic constant
        system_names = data_dict['rows']
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate explained variance and cumulative variance
        var_ratio = pca.explained_variance_ratio_
        cum_var_ratio = np.cumsum(var_ratio)
        
        # Print variance explained
        print("\nVariance explained by each component:")
        for i, var in enumerate(var_ratio):
            print(f"PC{i+1}: {var:.4f} ({cum_var_ratio[i]:.4f} cumulative)")
        
        # 1. Scree plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(var_ratio) + 1), var_ratio, 'bo-')
        plt.plot(range(1, len(var_ratio) + 1), cum_var_ratio, 'ro-')
        plt.xlabel('Principal Component')
        plt.ylabel('Proportion of Variance Explained')
        plt.title(f'{dist_type} - Scree Plot')
        plt.legend(['Individual', 'Cumulative'])
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{dist_type}_scree_plot.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Loading plot (first two PCs)
        plt.figure(figsize=(12, 8))
        loadings = pca.components_
        
        # Create loading plot
        plt.figure(figsize=(12, 8))
        loading_matrix = loadings[:2].T  # Take first two PCs
        
        # Plot arrows
        for i, (feature, loading) in enumerate(zip(feature_names, loading_matrix)):
            plt.arrow(0, 0, loading[0], loading[1],
                     head_width=0.02, head_length=0.02, fc='blue', ec='blue')
            plt.text(loading[0]*1.1, loading[1]*1.1, feature, ha='center', va='center')
        
        # Add circle
        circle = plt.Circle((0,0), 1, fill=False, linestyle='--', color='gray')
        plt.gca().add_patch(circle)
        
        plt.axis('equal')
        plt.xlabel(f'PC1 ({var_ratio[0]:.2%} variance explained)')
        plt.ylabel(f'PC2 ({var_ratio[1]:.2%} variance explained)')
        plt.title(f'{dist_type} - PCA Loading Plot')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{dist_type}_loading_plot.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Score plot (first two PCs)
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1])
        
        # Add labels for each point
        for i, txt in enumerate(system_names):
            plt.annotate(txt, (X_pca[i, 0], X_pca[i, 1]),
                        xytext=(5, 5), textcoords='offset points')
            
        plt.xlabel(f'PC1 ({var_ratio[0]:.2%} variance explained)')
        plt.ylabel(f'PC2 ({var_ratio[1]:.2%} variance explained)')
        plt.title(f'{dist_type} - PCA Score Plot')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{dist_type}_score_plot.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Heatmap of loadings
        plt.figure(figsize=(12, 8))
        num_pcs = min(5, len(loadings))  # Show first 5 PCs or all if less than 5
        loadings_df = pd.DataFrame(
            loadings[:num_pcs].T,
            columns=[f'PC{i+1}' for i in range(num_pcs)],
            index=feature_names
        )
        sns.heatmap(loadings_df, cmap='RdBu_r', center=0, annot=True, fmt='.2f')
        plt.title(f'{dist_type} - PCA Loadings Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dist_type}_loadings_heatmap.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

def cp_analysis(in_data, out_dir):
    # If in_data is a list, run the full anaysis
    if isinstance(in_data, list):
        print("Processing CP data...")
        in_data = add_job_columns(in_data)
        # Group CPs by system (grouping distorted and relaxed jobs for a single system)
        grouped_systems = group_systems(in_data)
        
        # Map critical points across different distortion levels within each system
        mapped_systems = map_all_systems(grouped_systems)
        
        # Remove degenerate CPs and add degeneracy count
        mapped_systems = remove_degenerate_cps(mapped_systems)
        
        # Compute delta values for each CP property in PROP_LIST
        mapped_systems = add_delta_properties(mapped_systems, most_distorted_only=not USE_ALL_DELTA_PROP)
        
        # Add rho_rank to each CP group based on the Rho value ordering of relaxed CPs
        ranked_systems = add_rho_rankings(mapped_systems)
        
        # Write ranked CP data to CSV
        csv_path = write_ranked_csv(ranked_systems, os.path.join(out_dir, "ranked_cp_info.csv"))
    elif isinstance(in_data, dict):
        print("Processing ranked CP data...")
        ranked_systems = in_data
    else:
        raise ValueError("Invalid input data format")
    
    # Prepare 2D arrays for statistical analysis
    analysis_data = prepare_analysis_arrays(ranked_systems)
    
    # Perform feature selection using Boruta algorithm
    reduced_data = perform_feature_selection(analysis_data, verbose=True)
    
    # Create correlation matrices
    # create_correlation_matrices(analysis_data, os.path.join(out_dir, "correlation_matrices"))
    # New correlation matrices with reduced features
    create_correlation_matrices(reduced_data, os.path.join(out_dir, "correlation_matrices_reduced"))
    
    # Perform PCA analysis
    perform_pca_analysis(analysis_data, os.path.join(out_dir, "pca_analysis"))
    perform_pca_analysis(reduced_data, os.path.join(out_dir, "pca_analysis_reduced"))
    
    return
    
    # Plot CP properties vs distortion for each system
    plot_cp_properties(ranked_systems, os.path.join(out_dir, "cp_plots"))

def convert_string_to_number(s, h, no_convert_headers=['SIGNATURE']):
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
    
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Remove empty fields, convert string'None' to None, and convert numbers
            cp_dict = {
                k: (None if v == 'None' else convert_string_to_number(v, k))
                for k, v in row.items() 
                if v != ''
            }
            cp_data.append(cp_dict)
    
    return cp_data

def read_ranked_csv(csv_path):
    """
    Read a CSV file written by write_ranked_csv() into a nested dictionary structure.
    
    Args:
        csv_path (str): Path to the CSV file
    
    Returns:
        dict: Dictionary with same structure as mapped_systems
    """
    ranked_systems = {}
    
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Remove empty fields, convert string 'None' to None, and convert numbers
            cp_dict = {
                k: (None if v == 'None' else convert_string_to_number(v, k))
                for k, v in row.items() 
                if v != ''
            }
            
            # Get system name from distortion_type and system
            system_name = f"{cp_dict['distortion_type']}_{cp_dict['system']}"
            parent_cp = cp_dict['Parent CP #']
            
            # Initialize nested dictionaries if needed
            if system_name not in ranked_systems:
                ranked_systems[system_name] = {}
            if parent_cp not in ranked_systems[system_name]:
                ranked_systems[system_name][parent_cp] = []
            
            # Remove the Parent CP # from the dictionary as it's now in the structure
            del cp_dict['Parent CP #']
            
            # Add the CP to its group
            ranked_systems[system_name][parent_cp].append(cp_dict)
    
    # Sort each CP group by distortion_percent to maintain expected ordering
    for system_name in ranked_systems:
        for parent_cp in ranked_systems[system_name]:
            ranked_systems[system_name][parent_cp].sort(
                key=lambda x: float(x['distortion_percent'])
            )
    
    return ranked_systems

def read_csv(csv_path):
    """
    Read a CSV file, automatically determining whether it's a simple or ranked format.
    
    Args:
        csv_path (str): Path to the CSV file
    
    Returns:
        Union[list, dict]: List of dictionaries (simple format) or nested dictionary
            structure (ranked format)
    """
    with open(csv_path, 'r') as csvfile:
        header = csvfile.readline()
        
    if 'Parent CP #' in header:
        return read_ranked_csv(csv_path)
    else:
        return read_simple_csv(csv_path)

def main(args=None):
    parser = argparse.ArgumentParser(
        description="Parse CP info from an AMS output file and write to a CSV file."
    )
    parser.add_argument(
        "input_file_or_folder",
        type=str,
        help='Path to the input band .rkf file, or a folder that will "walked" through, processing all band.rkf files in any of the subfolders.',
    )
    parser.add_argument(
        "--cps",
        type=str,
        default=None,
        help='Comma-delimited list of CPs to include (e.g., "1,3,5-8,11"; See AMSview or output file for CP numbering). If not specified, all CPs will be included.',
    )
    parser.add_argument(
        "--keep-degenerate-cps",
        action="store_true",
        help="Keep degenerate CPs as separate lines in the output CSV file. Default is to combine them, showing the number of degenerate CPs in a column. This option overrides --cps, so if the CP list includes degenerate CPs, those CPs will be replaced with the lowest index non-degenerate CP.",
    )

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    if os.path.isfile(args.input_file_or_folder) and not args.input_file_or_folder.endswith(".csv"):
        if args.input_file_or_folder.endswith("band.rkf"):
            cp_data = parse_cp_info(
                args.input_file_or_folder,
                args.cps,
                not args.keep_degenerate_cps,
                degenerate_cp_cutoff_diff=1e-6,
            )
            write_csv(cp_data, args.input_file_or_folder)
        else:
            print("Input file is not a band.rkf file")
    elif os.path.isfile(args.input_file_or_folder) and args.input_file_or_folder.endswith(".csv"):
        cp_data = read_csv(args.input_file_or_folder)
        cp_analysis(cp_data, os.path.dirname(args.input_file_or_folder))
    elif os.path.isdir(args.input_file_or_folder):
        # first walk to get all path names in order to find common part of full path
        paths = []
        for root, dirs, files in os.walk(args.input_file_or_folder):
            for file in files:
                if "band" in file and file.endswith(".rkf"):
                    paths.append(os.path.join(root, file))
        common_path = os.path.commonpath(paths)

        path_prefixes = set()
        all_cp_data = []
        num_paths = len(paths)
        for i, path in enumerate(paths):
            # The path prefix for the output file will be the common path of all files, minus the file name and extension
            path_minus_filename = os.path.split(path)[0]
            path_prefix = path_minus_filename.replace(common_path, "").replace(
                "/", "_"
            )[1:]
            if ".results" in path_prefix:
                path_prefix = path_prefix.split(".results")[0]
            elif "_results" in path_prefix:
                path_prefix = path_prefix.split("_results")[0]
            # Last, use regex to remove repeated strings with underscores in between (i.e. '(\w+)_\1' -> '\1')
            path_prefix = re.sub(r"(\w+)_\1", r"\1", path_prefix)
            # Then, if path_prefix is in path_prefixes, add a suffix i.e. '_a', '_b', etc. until it's unique
            suffix = "a"
            while path_prefix in path_prefixes:
                path_prefix = f"{path_prefix}_{suffix}"
                suffix = chr(ord(suffix) + 1)
            path_prefixes.add(path_prefix)
            # print(path_prefix)
            try:
                cp_data = parse_cp_info(
                    path,
                    args.cps,
                    False,
                    degenerate_cp_cutoff_diff=1e-6,
                )
                # add path prefix as a key:value pair to each item in cp_data
                for cp in cp_data:
                    cp["Job name"] = path_prefix
                all_cp_data.extend(cp_data)
                print(f"Processed file {i+1} of {num_paths}: {path}")
                # if i >= 3:
                #     break
            except Exception as e:
                print(f"Error processing file {i+1} of {num_paths}: {path}: {str(e)}")
                path_prefixes.remove(path_prefix)
        if len(all_cp_data) > 0:
            print(f"Processed {len(all_cp_data)} CPs from {num_paths} files")
            # Add job data (system, distortion, distortion percent) to each item in all_cp_data
            all_cp_data = add_job_columns(all_cp_data)
            # print(f"Writing CSV file with common path {common_path}")
            csv_path = write_csv(all_cp_data, common_path)
            
            # Perform CP analysis
            cp_analysis(all_cp_data, args.input_file_or_folder)

if __name__ == "__main__":
    # run test with input file "/Users/haiiro/NoSync/AMSPython/data/Pd.results/band.rkf"
    # main(["/Users/haiiro/NoSync/AMSPython/data/ALL_ELEMENTS_ALL_DISTORTIONS"])
    main(["/Users/haiiro/NoSync/AMSPython/data/ALL_ELEMENTS_ALL_DISTORTIONS/cp_info_all.csv"])
    # main(["/Users/haiiro/NoSync/AMSPython/data/ALL_ELEMENTS_ALL_DISTORTIONS/ranked_cp_info.csv"])
    # main()
