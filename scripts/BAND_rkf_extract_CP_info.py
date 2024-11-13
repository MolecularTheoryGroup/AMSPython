import csv
import os
import argparse
from math import sqrt, atan

# for computing eignevalues and eigenvectors
import numpy as np
import subprocess
import re

BOHR_TO_ANG = 0.529177210903


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

    if os.path.isfile(args.input_file_or_folder):
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
                    not args.keep_degenerate_cps,
                    degenerate_cp_cutoff_diff=1e-6,
                )
                # add path prefix as a key:value pair to each item in cp_data
                for cp in cp_data:
                    cp["Job name"] = path_prefix
                all_cp_data.extend(cp_data)
                print(f"Processed file {i+1} of {num_paths}: {path}")
            except Exception as e:
                print(f"Error processing file {i+1} of {num_paths}: {path}: {str(e)}")
                path_prefixes.remove(path_prefix)
        if len(all_cp_data) > 0:
            print(f"Processed {len(all_cp_data)} CPs from {num_paths} files")
            print(f"Writing CSV file with common path {common_path}")
            write_csv(all_cp_data, common_path)


if __name__ == "__main__":
    # run test with input file "/Users/haiiro/NoSync/AMSPython/data/Pd.results/band.rkf"
    # main(["/Users/haiiro/NoSync/AMSPython/data/ALL_ELEMENTS_ALL_DISTORTIONS"])
    main()
