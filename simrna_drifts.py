import os
import numpy as np
import glob
import sys
# for saving huge files
from sklearn.externals import joblib

atom_list = ["P","C4'","C2-GA_Pur","C6","N9","C2-CU_Pyr","C4","N1"]
atid = dict(zip(atom_list, np.arange(len(atom_list)))) # atom to index dict
NUM_ATOMS = len(atom_list)
print(NUM_ATOMS)

GA_accepted = ["P","C4'","C2","C6","N9"] # ,"C2-GA_Pur","C4","C6","N1",
CU_accepted = ["P","C4'","C2","C4","N1"] # ,"C2-GA_Pur","C4","C6","N1",


def map_atom_name(resname, atom_name):
    
    try:
        if resname in ['G','A']:
            if atom_name in GA_accepted:
                if atom_name == 'C2':
                    return 'C2-GA_Pur'
                return atom_name
            else:
                return None

        elif resname in ['C','U']:
            if atom_name in CU_accepted:
                if atom_name == 'C2':
                    return 'C2-CU_Pyr'
                return atom_name
            else:
                return None        

        else:
            raise
    except:
        print(resname, atom_name)
        raise
 

def split_function(line):
    mylist = [line[0:6], 
              line[6:11],
              line[13:16],
              line[16],
              line[17:20],
              line[21],
              line[22:26],
              line[26],
              line[30:38],
              line[38:46],
              line[46:54],
              line[54:60],
              line[60:66],
              line[76:78],
              line[78:80]]
    #mylist = [x.replace(' ','') for k, x in enumerate(mylist) if k!=6]
    mylist = [x.replace(' ','') for x in mylist]
    return mylist


def fix_resSeq(x):
    return '%04d' % int(x)


def load_structure_from_pdb_file(fullpath):

    structure = dict()
    chain_is_terminated = set()
        
    with open(fullpath, 'r') as fileobj:

        for line in fileobj:
            first_six = line[:6]
            if ('HETATM' in first_six) or ('ATOM' in first_six):

                line_split = split_function(line)

                atom_name = line_split[2]
                resName   = line_split[4]
                chain     = line_split[5]
                resSeq    = line_split[6]
                iCode     = line_split[7]
                
                resSeq    = fix_resSeq(resSeq) # since resSeq is a string now ...
                
                if iCode != '':
                    resSeq = resSeq + iCode
                    # mark inserted residues ...

                if chain not in chain_is_terminated:

                    if chain not in structure:
                        structure[chain] = dict()

                    if resSeq not in structure[chain]:
                        structure[chain][resSeq] = (dict(), resName)

                    coords = np.array([float(x) for x in line_split[8:11]])
                    structure[chain][resSeq][0][atom_name] = coords


            if line[:3] == 'TER':
                chain_is_terminated.update(chain)
    
    return structure
                

def fix_chain_breaks(structure, pdb_seq):
    
    structure_fixed = dict()
    for chain in structure:
        chain_fixed = chain

        k = 0
        c = 0
        resSeq = sorted(list(structure[chain].keys()))

        for letter in pdb_seq:
            
            if letter=='&':
                chain_fixed = chain + '_a'
                c = 0 # reset counter
                print('FIXED CHAIN BREAK')
                continue

            if chain_fixed not in structure_fixed:
                structure_fixed[chain_fixed] = dict()

            structure_fixed[chain_fixed][c] = structure[chain][resSeq[k]]

            c += 1
            k += 1

    return structure_fixed


atoms_dict = dict()
atoms_dict['A'] = set(['C2', 'C4', 'C5', 'C6', 'C8', 'N1', 'N3', 'N6', 'N7', 'N9'])
atoms_dict['C'] = set(['C2', 'C4', 'C5', 'C6', 'N1', 'N3', 'N4', 'O2'])
atoms_dict['G'] = set(['C2', 'C4', 'C5', 'C6', 'C8', 'N1', 'N2', 'N3', 'N7', 'N9', 'O6'])
atoms_dict['U'] = set(['C2', 'C4', 'C5', 'C6', 'N1', 'N3', 'O2', 'O4'])


def infer_residue(atoms):
    for key in list(atoms_dict):
        if atoms_dict[key].issubset(atoms):
            return key
    return None


def get_coords_for_residue(residue):
    
    resname = residue[1]
    residue = residue[0] # dictionary with atom names as keys and coordinates as values

    atoms = set(list(residue.keys()))
    inferred_resname = infer_residue(atoms)

    coords = np.zeros([NUM_ATOMS,3])
    mask   = np.zeros([NUM_ATOMS])

    ### check: is inferred residue name okay?
    if inferred_resname not in list('ACGU'):
        inferred_resname = resname # try PDB resname ...
    
    ### if the PDB resname also is not okay: exit without result.
    if inferred_resname not in list('ACGU'):
        return None

    for atom_name in residue:
        atom_name_MAPPED = map_atom_name(inferred_resname, atom_name) # especially C2 -> C2-GA_Pur or C2-CU_Pyr
        if atom_name_MAPPED:
            if atom_name_MAPPED in atid:
                idx = atid[atom_name_MAPPED]
                coords[idx] = residue[atom_name] # use atom_name, nor atom_name_MAPPED !!!
                mask[idx]   = 1     

    return coords, mask, inferred_resname


def get_coords_for_structure(structure):
    
    all_coords = []
    all_masks  = []
    seq        = []
    for chain_idx in structure:
        for residue_idx in sorted(structure[chain_idx]):
            out = get_coords_for_residue(structure[chain_idx][residue_idx])
            if out is not None:
                coords, mask, resname = out
            else:
                return None

            all_coords += [coords]
            all_masks  += [mask]
            seq.append(resname)

    coords = np.concatenate(all_coords,axis=0)    
    mask   = np.concatenate(all_masks, axis=0)   
    
    return coords, mask, seq


def get_distances(midpoints):

    try:
        a = midpoints.reshape([-1,1,3])
        b = np.transpose(a, [1,0,2])

        d = np.sqrt(np.sum((a-b)**2, axis=2)) # distance matrix
        
    except:
        d = None
    
    return d


def get_mask(mask):
    mask = np.reshape(mask,[-1,1])
    return mask*mask.T
    
    
    
def flatten_big_matrix(big_matrix):

    L_big = big_matrix.shape[0] # number of nucleotides*23
    L = L_big//NUM_ATOMS # number of nucleotides

    flat_matrix = np.zeros([L,L,NUM_ATOMS*NUM_ATOMS])
    idx = np.arange(0, L_big, NUM_ATOMS)
    ctr = 0
    for i in range(0,NUM_ATOMS):
        for j in range(0,NUM_ATOMS):
            idx_i = idx + i
            idx_j = idx + j
            flat_matrix[:,:,ctr] = big_matrix[idx_i,:][:,idx_j]
            ctr += 1
    return flat_matrix


def main_pkl_load(fullpath):
    
    structure = load_structure_from_pdb_file(fullpath)

    out = get_coords_for_structure(structure)
    if out is not None:
        coords, mask, seq = out
    else:
        print('skipping: ', k, pdb_id)
        raise Exception('Could not extract PDB structure.')
        #return None

    big_matrix = get_distances(coords)
    big_mask   = get_mask(mask)

    big_matrix_flat = flatten_big_matrix(big_matrix)
    big_mask_flat   = flatten_big_matrix(big_mask) 
    
    return big_matrix_flat, big_mask_flat