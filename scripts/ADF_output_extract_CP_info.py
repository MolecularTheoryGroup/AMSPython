import csv
import os
import argparse
from math import sqrt, atan

def parse_cp_info(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    cp_blocks = content.split(' --------------------------------------------------------')[1:]
    cp_data = []

    for block in cp_blocks:
        if not 'CP #' in block:
            continue
        lines = [s.strip() for s in block.strip().split('\n')]
        cp_info = {}

        i = 0
        while i < len(lines):
            line = lines[i]
            if not line:
                i += 1
                continue
            if line.startswith('CP #'):
                cp_info['CP #'] = int(line.split('#')[1].strip())
            elif line.startswith('(RANK,SIGNATURE):'):
                rank_signature = line.split(':')[1].strip().split(',')
                cp_info['RANK'] = int(rank_signature[0].strip()[1:])
                cp_info['SIGNATURE'] = int(rank_signature[1].strip()[:-1])
            elif line.startswith('CP COORDINATES:'):
                cp_info['CP COORDINATES'] = [float(coord) for coord in line.split(':')[1].strip().split()]
            elif line.startswith('EIGENVALUES OF HESSIAN MATRIX:'):
                eigenvalues = []
                i += 2
                while i < len(lines) and lines[i]:
                    eigenvalues.extend([float(val) for val in lines[i].split()])
                    i += 1
                cp_info['EIGENVALUES OF HESSIAN MATRIX'] = eigenvalues
                continue
            elif line.startswith('EIGENVECTORS (ORTHONORMAL) OF HESSIAN MATRIX (COLUMNS):'):
                eigenvectors = []
                i += 2
                while i < len(lines) and lines[i]:
                    eigenvectors.append([float(val) for val in lines[i].split()])
                    i += 1
                # transpose eigenvectors
                eigenvectors = [ [ eigenvectors[j][i] for j in range(len(eigenvectors)) ] for i in range(len(eigenvectors[0])) ]
                cp_info['EIGENVECTORS (ORTHONORMAL) OF HESSIAN MATRIX (COLUMNS)'] = eigenvectors
                continue
            elif line.startswith('HESSIAN MATRIX:'):
                hessian_matrix = []
                i += 2
                while i < len(lines) and lines[i]:
                    hessian_matrix.append([float(val) for val in lines[i].split()])
                    i += 1
                # convert from upper-triangular to full matrix
                for j in range(len(hessian_matrix)):
                    for k in range(j+1, len(hessian_matrix)):
                        hessian_matrix[k].insert(j, hessian_matrix[j][k])
                cp_info['HESSIAN MATRIX'] = hessian_matrix
                continue
            elif line.startswith('VALUES OF SOME FUNCTIONS AT CPs (a.u.):'):
                i += 2
                while i < len(lines) and lines[i]:
                    key, value = lines[i].split('=')
                    cp_info[key.strip()] = float(value.split()[0])
                    i += 1
                # Compute directionality of bond/ring CPs using the eigenvalues (ev).
                # For bond CPs (3,-1), Theta = arctan(sqrt(ev[0]/ev[2])) and Phi = arctan(sqrt(ev[1]/ev[2])).
                # For ring CPs (3,1), Theta = arctan(sqrt(ev[2]/ev[0])) and Phi = arctan(sqrt(ev[1]/ev[0])).
                if abs(cp_info['SIGNATURE']) == 1:
                    ev = cp_info['EIGENVALUES OF HESSIAN MATRIX']
                    if cp_info['SIGNATURE'] == -1:
                        cp_info['Theta'] = atan(sqrt(abs(ev[0]/ev[2])))
                        cp_info['Phi'] = atan(sqrt(abs(ev[1]/ev[2])))
                    else:
                        cp_info['Theta'] = atan(sqrt(abs(ev[2]/ev[0])))
                        cp_info['Phi'] = atan(sqrt(abs(ev[1]/ev[0])))

            i += 1

        cp_data.append(cp_info)

    return cp_data

def parse_cp_list(cp_list_str):
    cp_list = []
    for part in cp_list_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            cp_list.extend(range(start, end + 1))
        else:
            cp_list.append(int(part))
    return cp_list

def write_csv(cp_data, input_file_path, cp_list_str=None):
    # Parse the CP list if provided
    if cp_list_str:
        cp_list = parse_cp_list(cp_list_str)
    else:
        cp_list = [cp['CP #'] for cp in cp_data]

    # Filter cp_data to include only the specified CPs
    cp_data = [cp_data[i-1] for i in cp_list if i <= len(cp_data)]

    # Determine all keys present in the data
    all_keys = set()
    for cp in cp_data:
        for key in cp.keys():
            if isinstance(cp[key], list):
                if isinstance(cp[key][0], list):  # 2D list
                    if key == 'EIGENVECTORS (ORTHONORMAL) OF HESSIAN MATRIX (COLUMNS)':
                        all_keys.update([ f'{key}_EV{i+1}_{axis}' for i in range(3) for axis in ['X', 'Y', 'Z'] ])
                    elif key == 'HESSIAN MATRIX':
                        all_keys.update([ f'{key}_{axis1}{axis2}' for axis1 in ['X', 'Y', 'Z'] for axis2 in ['X', 'Y', 'Z'] ])
                else:  # 1D list
                    all_keys.update([ f'{key}_{axis}' for axis in ['X', 'Y', 'Z'] ])
            else:
                all_keys.add(key)
    # Sort the keys
    all_keys = sorted(all_keys)
  
    # Define the preferred order of columns
    preferred_order = [
        'CP #', 'RANK', 'SIGNATURE',
        'CP COORDINATES_X', 'CP COORDINATES_Y', 'CP COORDINATES_Z',
        'Rho', '|GRAD(Rho)|', 'GRAD(Rho)x', 'GRAD(Rho)y', 'GRAD(Rho)z',
        'Laplacian', '(-1/4)Del**2(Rho))', 'Diamond', 'Metallicity', 'Ellipticity',
        'Theta', 'Phi']
    
    # Now add Eigenvalues and Eigenvectors
    for h in [s for s in all_keys if 'EIGENVALUES' in s]:
        preferred_order.append(h)
    for h in [s for s in all_keys if 'EIGENVECTORS' in s]:
        preferred_order.append(h)
        
    # Then Hessian
    for h in [s for s in all_keys if 'HESSIAN' in s]:
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
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for cp in cp_data:
            row = {}
            for key, value in cp.items():
                if isinstance(value, list):
                    if isinstance(value[0], list):  # 2D list
                        if 'EIGENVECTORS (ORTHONORMAL) OF HESSIAN MATRIX (COLUMNS)' in key:
                            for i in range(3):
                                for j, axis in enumerate(['X', 'Y', 'Z']):
                                    row[f'{key}_EV{i+1}_{axis}'] = value[i][j]
                        elif 'HESSIAN MATRIX' in key:
                            for i, axis1 in enumerate(['X', 'Y', 'Z']):
                                for j, axis2 in enumerate(['X', 'Y', 'Z']):
                                    row[f'{key}_{axis1}{axis2}'] = value[i][j]
                    else:  # 1D list
                        for i, axis in enumerate(['X', 'Y', 'Z']):
                            row[f'{key}_{axis}'] = value[i]
                else:
                    row[key] = value
            writer.writerow(row)
    
    print(f"CSV file written to {csv_file_path}")

def main():
    parser = argparse.ArgumentParser(description='Parse CP info from an AMS output file and write to a CSV file.')
    parser.add_argument('input_file', type=str, help='Path to the input text file.')
    parser.add_argument('--cps', type=str, default=None, help='Comma-delimited list of CPs to include (e.g., "1,3,5-8,11"). If not specified, all CPs will be included.')

    args = parser.parse_args()

    cp_data = parse_cp_info(args.input_file)
    write_csv(cp_data, args.input_file, args.cps)

if __name__ == '__main__':
    main()