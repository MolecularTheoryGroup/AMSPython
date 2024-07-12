#!/usr/bin/env plams
# H2O_opti.py
# Geometry optimization of a water molecule using ADF
import os
import subprocess
from math import sqrt, atan
import numpy as np

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

def get_bcp_properties(job, atom_pairs, unrestricted=False):
    # first get kf file from finished job
    kf = KFFile(job.results['adf.rkf'])
    cp_type_codes = {'nuclear': 1, 'bond':3, 'ring':4, 'cage':2}
    num_cps = kf[('Properties','CP number of')]
    cp_coords = kf[('Properties','CP coordinates')]
    cp_codes = kf[('Properties','CP code number for (Rank,Signatu')]
    # cp_coords is all the x, then all the y, then all the z. So we need to reshape it to a 2D list
    cp_coords = [(cp_coords[i], cp_coords[i+num_cps], cp_coords[i+2*num_cps]) for i in range(num_cps)]
    # Now loop over each atom pair, find the bond cp closest to the atom pair midpoint, and save its
    # index to a list.
    cp_indices = []
    bcp_coords = []
    bcp_atom_indices = []
    out_mol = job.results.get_main_molecule()
    for pair in atom_pairs:
        a1 = [getattr(out_mol.atoms[pair[0]-1], d) for d in ['x', 'y', 'z']]
        a2 = [getattr(out_mol.atoms[pair[1]-1], d) for d in ['x', 'y', 'z']]
        midpoint = [(a1[i] + a2[i])/2 for i in range(3)]
        min_dist = 1e6
        min_index = -1
        for i, cp in enumerate(cp_coords):
            if cp_codes[i] != cp_type_codes['bond']:
                continue
            dist = sum((midpoint[j] - cp[j])**2 for j in range(3))**0.5
            if dist < min_dist:
                min_dist = dist
                min_index = i
        cp_indices.append(min_index+1)
        bcp_coords.append(cp_coords[min_index])
        bcp_atom_indices.append([f'{out_mol.atoms[pair[i]-1].symbol}{pair[i]}' for i in range(2)])

    # Now extract the properties of the bond critical points from the output file
    output_file = [f for f in job.results.files if f.endswith('.out') and 'CreateAtoms' not in f][0]
    out_cp_data = parse_cp_info(job.results[output_file])
    # only keep the bond critical points we want
    out_cp_data = [cp for cp in out_cp_data if cp['CP #'] in cp_indices]
    # match cp_indices to the right element in out_cp_data using the [CP #] key
    out_cp_data_cp_inds = {}
    for i in range(len(cp_indices)):
        for j, cp in enumerate(out_cp_data):
            if cp['CP #'] == cp_indices[i]:
                out_cp_data_cp_inds[i] = j
                print(f"Found CP {cp_indices[i]} at index {j}")
                break
        out_cp_data[out_cp_data_cp_inds[i]]['ATOMS'] = bcp_atom_indices[i]
    # We've only gotten cp data for the bond critical points we want, so we can use those coordinates
    # to create and run a densf job to get the A and B density values and geometry at the points.
    # But only if it's an unrestricted calculation.
    if unrestricted:
        # We'll create a densf input file that will compute the density at all the collected
        # bond critical points.
        # https://www.scm.com/doc/ADF/Input/Densf.html
        in_file = job.results['adf.rkf']
        out_file = job.results['adf.rkf'].replace('.rkf', '.t41')
        grid_coords = '\n'.join([f'{cp[0]} {cp[1]} {cp[2]}' for cp in bcp_coords])
        grid_block = f'Grid Inline\n{grid_coords}\nEnd\n'
        densf_str = f"""$AMSBIN/densf << eor
    ADFFILE {in_file}
    OUTPUTFILE {out_file}
    {grid_block}
    Density scf
    DenGrad
    DenHess
eor"""
        # Now run the densf job
        densf_out = subprocess.run(densf_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if densf_out.returncode != 0:
            print(f"Error running densf: {densf_out.stderr}")
        densf_kf = KFFile(out_file)
        for i, cpi in enumerate(cp_indices):            
            def get_saddle_t41_properties(cp_data, cp_ind, out_cp_ind, field):
                cp_data[out_cp_ind][f'Rho_{field}'] = densf_kf[('SCF',f'Density_{field}')][cp_ind]
                grad = [densf_kf[('SCF',f'DensityGrad{ax}_{field}')][cp_ind] for ax in ['X', 'Y', 'Z']]
                grad_mag = sum([g**2 for g in grad])**0.5
                cp_data[out_cp_ind][f'|GRAD(Rho)|_{field}'] = grad_mag
                for ax in ['x', 'y', 'z']:
                    cp_data[out_cp_ind][f'GRAD(Rho){ax}_{field}'] = grad['xyz'.index(ax)]
                hess = [densf_kf[('SCF',f'DensityHess{ax}_{field}')][cp_ind] for ax in ['XX', 'XY', 'XZ', 'YY', 'YZ', 'ZZ']]
                hess = [[hess[0], hess[1], hess[2]], [hess[1], hess[3], hess[4]], [hess[2], hess[4], hess[5]]]
                hess = np.array(hess)
                ev, evec = np.linalg.eigh(hess)
                if cp_codes[cpi-1] == cp_type_codes['bond']:
                    cp_data[out_cp_ind][f'Theta_{field}'] = atan(sqrt(abs(ev[0]/ev[2])))
                    cp_data[out_cp_ind][f'Phi_{field}'] = atan(sqrt(abs(ev[1]/ev[2])))
                elif cp_codes[cpi-1] == cp_type_codes['ring']:
                    cp_data[out_cp_ind][f'Theta_{field}'] = atan(sqrt(abs(ev[2]/ev[0])))
                    cp_data[out_cp_ind][f'Phi_{field}'] = atan(sqrt(abs(ev[1]/ev[0])))
                cp_data[out_cp_ind][f'HESSIAN MATRIX_{field}'] = hess.tolist()
                cp_data[out_cp_ind][f'EIGENVECTORS (ORTHONORMAL) OF HESSIAN MATRIX (COLUMNS)_{field}'] = evec.T.tolist()
                cp_data[out_cp_ind][f'EIGENVALUES_{field}'] = ev.tolist()
            for field in ['A', 'B']:
                get_saddle_t41_properties(out_cp_data, i, out_cp_data_cp_inds[i], field)
    return out_cp_data

# Create the molecule:
mol = Molecule()
mol.add_atom(Atom(symbol="O", coords=(0, 0, 0)))
mol.add_atom(Atom(symbol="H", coords=(1, 0, 0)))
mol.add_atom(Atom(symbol="H", coords=(0, 1, 0)))
mol.guess_bonds()

# Initialize the settings for the ADF calculation:
sett = Settings()
sett.input.ams.task = "SinglePoint"
sett.input.adf.basis.type = "SZ"
sett.input.adf.xc.gga = "PBE"
sett.input.adf.qtaim.enabled = "yes"
sett.input.adf.unrestricted = "yes"
sett.input.adf.spinpolarization = 2.0

# Create and run the job:
job = AMSJob(molecule=mol, settings=sett, name="water_GO")
job.run()

# Fetch and print some results:
energy = job.results.get_energy()
opt_mol = job.results.get_main_molecule()
opt_mol.guess_bonds()
bond_angle = opt_mol.atoms[0].angle(opt_mol.atoms[1], opt_mol.atoms[2])

print("== Water optimization Results ==")
print("Bonding energy: {:8.2f} kcal/mol".format(Units.convert(energy, "au", "kcal/mol")))
print("Bond angle:     {:8.2f} degree".format(Units.convert(bond_angle, "rad", "degree")))
print("Optimized coordinates:")
print(opt_mol)

r = job.results

print(r.files)

a_pairs = ((1,2),(1,3)) # one-based indices

cp_data = get_bcp_properties(job, a_pairs, unrestricted=True)

print(cp_data)