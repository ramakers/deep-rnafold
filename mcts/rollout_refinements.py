import os
import subprocess
import tempfile
import pickle
import pandas as pd
import warnings
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
import sonnet as snt
import tensorflow as tf
import tarfile
from six.moves import cPickle
from six.moves import urllib
from six.moves import xrange
from simrna_drifts import *
from tensorflow.python.client import device_lib
import rna_enc_pdb
from mpl_toolkits.mplot3d import Axes3D
import glob

"""

  this script looks for all rollouts in the subfolder ./rollouts = embeddings
  and produces the corresponding pdb files after decoding into realspace coors

"""


rollouts   = glob.glob('./rollouts/*')
output_dir = './rollouts_pdb_files/'

ckpt_dir = '../vqvae/ckpt/vqvae_60000.ckpt'
data_dir = '../data/substructures_drifts.pkl'





enc_seq_pdb   = rna_enc_pdb.RNAEncoder(L=100, args="seq")
atom_list = ["P","C4'","C2-GA_Pur","C6","N9","C2-CU_Pyr","C4","N1"]
atid = dict(zip(atom_list, np.arange(len(atom_list)))) # atom to index dict
NUM_ATOMS = len(atom_list)


def map_idx(i):
    """
    index mapper for ATOMS
    """
    within_index  = i % NUM_ATOMS
    between_index = i//NUM_ATOMS
    return within_index*NUM_ATOMS + between_index

symmetrization_indices = map_idx(np.arange(NUM_ATOMS**2))


def symmetrize_LxLx64(D):
    """
    input:  D [L,L,64]
    output: D [L,L,64]
    
    If this function is applied to an unsymmetrized tensor, it will be symmetrized.
    If this function is applied to a symmetrized tensor, it will be un-symmetrized!
    
    """    
    upper = np.transpose(np.triu(np.transpose(D,[2,0,1]), 0),[1,2,0])  # 0 includes diagonal
    lower = np.transpose(np.tril(np.transpose(D,[2,0,1]), -1),[1,2,0]) # 0 includes diagonal
    return upper + lower[:,:,symmetrization_indices]


def Lx3_coords_to_LxLx64_tensor(array_a, *args):

    """

    input(s):
        either 
            coords [Ra*8, 3] and optionally [Rb*8, 3]
        or
            mask [Ra*8] and optionally [Rb*8]
        
    output:
        either
            D [Ra, Ra, 64]
        or
            D [Ra, Rb, 64]
    """
    

    if len(args)==1:
        array_b = args[0]
    else:
        array_b = array_a
    
    array_a_shape = array_a.shape
    array_b_shape = array_b.shape
    if array_a_shape[0]%NUM_ATOMS!=0:
        raise Exception('Shape must be [R*8, 3] or [R*8], but is', array_a_shape)
    if array_b_shape[0]%NUM_ATOMS!=0:
        raise Exception('Shape must be [R*8, 3] or [R*8], but is', array_b_shape)        
    
    if len(array_a_shape)!=len(array_b_shape):
        raise Exception('Inputs have different dimensions:', array_a_shape, array_b_shape)  
    
    ### assume that input was "coords"
    if len(array_a_shape)==2:
        Ra = len(array_a)//NUM_ATOMS
        Rb = len(array_b)//NUM_ATOMS
        A = array_a.reshape([Ra, 1, NUM_ATOMS, 1, 3]) # [R*8, 3] -> [R, 1, 8, 1, 3]
        B = array_b.reshape([1, Rb, 1, NUM_ATOMS, 3]) # [R*8, 3] -> [1, R, 1, 8, 3]
        
        D = np.sqrt(np.sum((A-B)**2, axis=4, keepdims=True)) # [R, R, 8, 8, 1]
        D = np.reshape(D, [Ra,Rb,NUM_ATOMS**2]) # [R, R, 8, 8, 1] -> [R, R, 64]
        
        
    ### assume that input was "mask"
    elif len(array_a_shape)==1:
        Ra = len(array_a)//NUM_ATOMS
        Rb = len(array_b)//NUM_ATOMS
        A = array_a.reshape([Ra, 1, NUM_ATOMS, 1]) # [R*8] -> [R, 1, 8, 1]
        B = array_b.reshape([1, Rb, 1, NUM_ATOMS]) # [R*8] -> [1, R, 1, 8] 
        D = A*B
        D = np.reshape(D, [Ra,Rb,NUM_ATOMS**2])
        
        
    else:
        raise Exception('Shape must be [R*8, 3] or [R*8], but is', array_shape)
    
    
    ### symmetrize
    if Ra==Rb:
        return symmetrize_LxLx64(D)
    return D

def LxLx64_tensor_to_L8xL8_matrix(tensor):

    """
    Convert flat (L x L x 64) matrix to big (L*8 x L*8) matrix
    input:  tensor [L,L,64]
    output: matrix [L*8, L*8]
    """

    L8 = tensor.shape[0] * NUM_ATOMS

    matrix = np.zeros([L8,L8])
    idx = np.arange(0, L8, NUM_ATOMS)
    ctr = 0
    for i in range(0,NUM_ATOMS):
        for j in range(0,NUM_ATOMS):
            idx_i = idx + i
            idx_j = idx + j
            putmask1 = np.zeros([L8,L8])
            putmask1[idx_i]=1
            putmask2 = np.zeros([L8,L8])
            putmask2[:,idx_j]=1
            
            matrix[np.logical_and(putmask1, putmask2)] = tensor[:,:,ctr].reshape(-1)
            ctr += 1
    return matrix


def matrix_to_coords(D):
    """
    
    input:  D (distance matrix)
    output: x (reconstructed coordinates)
    """
    
    D_sq = D**2
    M = 0.5 * (D_sq[[0],:] + D_sq[:,[0]] - D_sq)
    q,v = np.linalg.eigh(M) # numpy.linalg.eig often produces complex number with very small imaginary part.
    x = np.sqrt(np.reshape(q[-3:],[1,3])) * v[:,-3:]
    x = x[:,::-1] # numpy.linalg.eigh has largest eigenvalues last, so we need to flip the vector here...
    
    return x


def L8xL8_matrix_to_Lx3_coords(matrix, mask):

    matrix_reduced = matrix[mask==1,:][:,mask==1]
    coords_reduced = matrix_to_coords(matrix_reduced)
    
    L8 = matrix.shape[0]
    coords = np.zeros([L8,3])    
    coords[mask==1] = coords_reduced
    return coords


def superimposer(x,y, mask=None):
    
    if mask is not None:
        x = x[mask==1]
        y = y[mask==1]
        
    xn = x - np.mean(x,axis=0)
    yn = y - np.mean(y,axis=0)    
    
    C = np.matmul(xn.T,yn)

    U, s, V = np.linalg.svd(C, full_matrices=1, compute_uv=1)
    R = np.matmul(U,V)
    
    yn_proj = np.matmul(yn, R.T)
    
    diff = xn - yn_proj
    rms = np.sqrt(sum(sum(diff * diff)) / x.shape[0])
    
    
    return rms, xn, yn_proj


def get_mask_for_nucleotide(nuc):
    
    """
    input: nucleotide (string; either 'A', 'C', 'G', 'U')
    output: mask (numpy array of shape [8])
    
    Atoms are in the following order:
    0   P
    1   C4'a
    2   C2-GA_Pur
    3   C6
    4   N9
    5   C2-CU_Pyr
    6   C4
    7   N1
    """
    
    if nuc not in list('ACGU'):
        raise Exception('Nucleotide must be either A, C, G or U but is:', nuc)
        
    mask = np.zeros([8])
    mask[:2] = 1
    if nuc in list('GA'):
        mask[2:5]=1
        return mask
    elif nuc in list('CU'):
        mask[5:]=1
        return mask
    


# In[38]:


def get_mask_for_sequence(seq):
    return np.concatenate([get_mask_for_nucleotide(nuc) for nuc in seq], axis=0)




def get_3D_tensor(array_a, *args):
    
    """
    input(s):
        either 
            coords [Ra*8, 3] and optionally [Rb*8, 3]
        or
            mask [Ra*8] and optionally [Rb*8]
        
    output:
        either
            D [Ra, Ra, 64]
        or
            D [Ra, Rb, 64]
    """
    

    if len(args)==1:
        array_b = args[0]
    else:
        array_b = array_a
    
    array_a_shape = array_a.shape
    array_b_shape = array_b.shape
    if array_a_shape[0]%NUM_ATOMS!=0:
        raise Exception('Shape must be [R*8, 3] or [R*8], but is', array_a_shape)
    if array_b_shape[0]%NUM_ATOMS!=0:
        raise Exception('Shape must be [R*8, 3] or [R*8], but is', array_b_shape)        
    
    if len(array_a_shape)!=len(array_b_shape):
        raise Exception('Inputs have different dimensions:', array_a_shape, array_b_shape)  
    
    ### assume that input was "coords"
    if len(array_a_shape)==2:
        Ra = len(array_a)//NUM_ATOMS
        Rb = len(array_b)//NUM_ATOMS
        A = array_a.reshape([Ra, 1, NUM_ATOMS, 1, 3]) # [R*8, 3] -> [R, 1, 8, 1, 3]
        B = array_b.reshape([1, Rb, 1, NUM_ATOMS, 3]) # [R*8, 3] -> [1, R, 1, 8, 3]
        
        D = np.sqrt(np.sum((A-B)**2, axis=4, keepdims=True)) # [R, R, 8, 8, 1]
        D = np.reshape(D, [Ra,Rb,NUM_ATOMS**2]) # [R, R, 8, 8, 1] -> [R, R, 64]
        
        
    ### assume that input was "mask"
    elif len(array_a_shape)==1:
        Ra = len(array_a)//NUM_ATOMS
        Rb = len(array_b)//NUM_ATOMS
        A = array_a.reshape([Ra, 1, NUM_ATOMS, 1]) # [R*8] -> [R, 1, 8, 1]
        B = array_b.reshape([1, Rb, 1, NUM_ATOMS]) # [R*8] -> [1, R, 1, 8] 
        D = A*B
        D = np.reshape(D, [Ra,Rb,NUM_ATOMS**2])
        
        
    else:
        raise Exception('Shape must be [R*8, 3] or [R*8], but is', array_shape)
    
    
    ### symmetrize
    if Ra==Rb:
        return symmetrize(D)
    return D

def symmetrize(D):
    """
    input:  D [L,L,64]
    output: D [L,L,64]
    
    If this function is applied to an unsymmetrized tensor, it will be symmetrized.
    If this function is applied to a symmetrized tensor, it will be un-symmetrized!
    
    """    
    upper = np.transpose(np.triu(np.transpose(D,[2,0,1]), 0),[1,2,0])  # 0 includes diagonal
    lower = np.transpose(np.tril(np.transpose(D,[2,0,1]), -1),[1,2,0]) # 0 includes diagonal
    return upper + lower[:,:,symmetrization_indices]


def get_3D_tensor(array_a, *args):
    
    """
    input(s):
        either 
            coords [Ra*8, 3] and optionally [Rb*8, 3]
        or
            mask [Ra*8] and optionally [Rb*8]
        
    output:
        either
            D [Ra, Ra, 64]
        or
            D [Ra, Rb, 64]
    """
    

    if len(args)==1:
        array_b = args[0]
    else:
        array_b = array_a
    
    array_a_shape = array_a.shape
    array_b_shape = array_b.shape
    if array_a_shape[0]%NUM_ATOMS!=0:
        raise Exception('Shape must be [R*8, 3] or [R*8], but is', array_a_shape)
    if array_b_shape[0]%NUM_ATOMS!=0:
        raise Exception('Shape must be [R*8, 3] or [R*8], but is', array_b_shape)        
    
    if len(array_a_shape)!=len(array_b_shape):
        raise Exception('Inputs have different dimensions:', array_a_shape, array_b_shape)  
    
    ### assume that input was "coords"
    if len(array_a_shape)==2:
        Ra = len(array_a)//NUM_ATOMS
        Rb = len(array_b)//NUM_ATOMS
        A = array_a.reshape([Ra, 1, NUM_ATOMS, 1, 3]) # [R*8, 3] -> [R, 1, 8, 1, 3]
        B = array_b.reshape([1, Rb, 1, NUM_ATOMS, 3]) # [R*8, 3] -> [1, R, 1, 8, 3]
        
        D = np.sqrt(np.sum((A-B)**2, axis=4, keepdims=True)) # [R, R, 8, 8, 1]
        D = np.reshape(D, [Ra,Rb,NUM_ATOMS**2]) # [R, R, 8, 8, 1] -> [R, R, 64]
        
        
    ### assume that input was "mask"
    elif len(array_a_shape)==1:
        Ra = len(array_a)//NUM_ATOMS
        Rb = len(array_b)//NUM_ATOMS
        A = array_a.reshape([Ra, 1, NUM_ATOMS, 1]) # [R*8] -> [R, 1, 8, 1]
        B = array_b.reshape([1, Rb, 1, NUM_ATOMS]) # [R*8] -> [1, R, 1, 8] 
        D = A*B
        D = np.reshape(D, [Ra,Rb,NUM_ATOMS**2])
        
        
    else:
        raise Exception('Shape must be [R*8, 3] or [R*8], but is', array_shape)
    
    
    ### symmetrize
    if Ra==Rb:
        triu           = np.reshape(np.triu(np.ones([Ra,Ra]), 1), [Ra,Ra,1])
        triu_plus_diag = np.reshape(np.triu(np.ones([Ra,Ra]), 0), [Ra,Ra,1])
        D              = D*triu_plus_diag + np.transpose(D*triu, axes=[1,0,2])        

    return D

def get_substructure_position(coords, substructure_length):
    
    """
    inputs: 
        (R is the number of residues)
        coords:              [R*8, 3]
        substructure_length: requested number of residues in substructure, must be smaller than R
    
    output:
        tuple consisting of:
            i: integer indicating the starting position of the substructure in the structure
            substructure_length: just the input.
    """
    
    R = len(coords)//NUM_ATOMS
    if R<=substructure_length:    
        return (None, substructure_length)
    else:
        i = np.random.randint(0,R-substructure_length+1) # random index
        return (i, substructure_length)
    
    
    
def get_substructure(coords, mask, i, return_contact_mask=True, cutoff=5):

    """
    Developed 2023-02-17 in:
    
    inputs:
        (R is the number of residues)
        coords: [R*8, 3]
        mask:   [R*8]
        i:      tuple of (starting position, substructure_length)
        
    outputs:
        (Rs is the number of residues in the substructure)
        coords: [Rs*8, 3]
        mask:   [Rs*8]
        contact_mask: [Rs] (only if return_contact_mask=True)
    """    
    
    (i, Rs) = i
    
    if len(np.shape(coords))!=2 or len(np.shape(mask))!=1:
        raise Exception('Input arrays shoudl have shapes [R*8, 3] and [R*8] but have:', np.shape(coords), np.shape(mask))
    
    lengths = [len(x) for x in [coords, mask]]
    u = np.unique(lengths)
    if len(u)>1:
        raise Exception('Input arrays have different size along dimension 0:', lengths)    
    if u%NUM_ATOMS!=0:
        raise Exception('Input dimension must be divisible by %d but is %d' %(NUM_ATOMS, u))
        
    if i==None:
        return coords, mask
    
    

    else:
        ### randomly select substructure
        sel_substr = np.arange(i*NUM_ATOMS, (i+Rs)*NUM_ATOMS)
        
        if len(sel_substr)%NUM_ATOMS!=0:
            raise Exception(len(sel_substr))

        ### selector for matrix (remaining residues)
        sel_matrix = np.arange(len(coords))
        sel_matrix = np.delete(sel_matrix, sel_substr)

        ### select coords and mask
        coords_substr = coords[sel_substr]
        coords_matrix = coords[sel_matrix]
        mask_substr   = mask[sel_substr]
        mask_matrix   = mask[sel_matrix]

        outputs = [coords_substr, mask_substr]
        
        if return_contact_mask:
            D = get_3D_tensor(coords_substr, coords_matrix) # [R1, R2, 64]
            M = get_3D_tensor(mask_substr,   mask_matrix)   # [R1, R2, 64]

            # Crucial!! This sets masked values to really large numbers!
            # This is because we take the minimum later, but "min" does not know which values to ignore.
            D[M==0] = 10**6 
            D_min   = np.min(D, axis=2) # [R1, R2, 64] -> [R1, R2], use a single atom as a proxy for the whole residue

            contact_mask = np.any(D_min<cutoff, axis=1)*1 # [R1]
            outputs.append(contact_mask)

        return outputs
    
def enc2str(valid_sq):
    vseq = ''.join(enc_seq_pdb.decode(valid_sq[0]))
    s = ''
    for ss in vseq:
        if ss=='<':
            break
        else:
            s+=ss
    return s


def transposeD(tJ):
    tjlist = []
    for m in range(64):
        tmp = (tJ[:,:,m]    + tJ[:,:,m].T) * 0.5
        tjlist.append(tmp)
    return np.stack(tjlist,-1)


def np_pad(matrix, pad_max=100, pad_val=0, max_depth=64):
    M_pad = np.full([pad_max, pad_max, max_depth], pad_val, dtype=float)
    M_pad[:matrix.shape[0], :matrix.shape[1], :] = matrix
    return M_pad


atom_list = ["P","C4'","C2-GA_Pur","C6","N9","C2-CU_Pyr","C4","N1"]
atid = dict(zip(atom_list, np.arange(len(atom_list)))) # atom to index dict
NUM_ATOMS = len(atom_list)
print(NUM_ATOMS)


map_idx_to_atom = dict(zip(np.arange(NUM_ATOMS), ["P","C4'","C2","C6","N9","C2","C4","N1"])) # atom to index dict
atom_types = list(map_idx_to_atom.values())
atom_type_dict = dict(zip(atom_types, [x[0] for x in atom_types]))




def check_check_dict(check_dict):
    """
    Version: 2023-02-17
    """    
    
    allowed = [set(["P","C4'","C2","C6","N9"]),
               set(["P","C4'","C2","C4","N1"])]
    
    for key, atoms in check_dict.items():
        if not any([set(atoms) == x for x in allowed]):
            print('Residue does not have all necessary atoms', key, atoms)
            return False
    
    return True





def write_8_atom_structure_to_pdb_file(filename, coords, mask, seq, occupancy, temperature_factor):
    """
    function to write Lx3 coords into .pdb files
    """
    
    if len(coords)//NUM_ATOMS>max_len:
        print('Did not write (structure too > max_len residues)', filename)
        return None
    
    
    lines_list = []
    #lines_list.append('MODEL        1                                                                  \n')

    seq_list = list(seq)
    atom_number = 0
    check_dict = dict()
    
    for k, (c, m, occ, temp) in enumerate(zip(coords, mask, occupancy, temperature_factor)):

        if temp==0:
            temp=10
        
        within_residue_idx  = k % NUM_ATOMS  # 0,1,2,3,4,5,6, 0,1,2,3,4,5,6, 0,1,2,3,4,5,6, ...
        within_sequence_idx = k // NUM_ATOMS # 0,0,0,0,0,0,0, 1,1,1,1,1,1,1, 2,2,2,2,2,2,2, ... 


        ### if no data for first phosphate: append "fake" phosphate
        if k==0 and not m:
            atom_number += 1
            residue_name = seq_list[within_sequence_idx]
            atom_name    = 'P'
            atom_type    = 'P'

            res1_C4prime = coords[1]
            res2_P       = coords[8]
            c            = res1_C4prime + (res1_C4prime-res2_P)

            string_input = (atom_number, atom_name, residue_name, within_sequence_idx+1, 
                            c[0], c[1], c[2], 1.0, 10.0, '', atom_type)
            line = 'ATOM  %5d%5s%4s A%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n' % string_input
            lines_list.append(line)  
            
            if within_sequence_idx not in check_dict:
                check_dict[within_sequence_idx] = []
            check_dict[within_sequence_idx].append(atom_name)                

        if m:
            atom_number += 1
            residue_name = seq_list[within_sequence_idx]
            atom_name    = map_idx_to_atom[within_residue_idx]
            atom_type    = atom_type_dict[atom_name]

            string_input = (atom_number, atom_name, residue_name, within_sequence_idx+1, 
                            c[0], c[1], c[2], occ, temp, '', atom_type)
            line = 'ATOM  %5d%5s%4s A%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n' % string_input
            lines_list.append(line)

            ### also create last line in case this really is the last line
            string_input = (atom_number+1, '', residue_name, within_sequence_idx+1)                
            last_line = 'TER   %5d%5s%4s A%4d                                                      \n' % string_input
            
            if within_sequence_idx not in check_dict:
                check_dict[within_sequence_idx] = []
            check_dict[within_sequence_idx].append(atom_name)                

    lines_list.append(last_line)
    #lines_list.append('ENDMDL                                                                          \n')
    lines_list.append('END                                                                             \n')
    
    
    if check_check_dict(check_dict):
        if filename is not None:
            with open(filename, 'w') as fileobj:
                for line in lines_list:
                    fileobj.write(line)
        return lines_list

    else:
        print('Did not write', filename)
        return None
    


########################################################
##### train , valid , test datasets as pandas dataframes
##### VQVAE :  compress raw LxLx64 dist matrices into LxLx3
########################################################
with open( data_dir, 'rb') as f:
    traindata_df, valdata_df, testdata_df = pickle.load( f)

##### SET NEW MAXIMAL SEQ LENGTH  #####
PARAMS               = {}
PARAMS['batch_size'] = 1
PARAMS['RNA_len']    = max_len = 100
PARAMS['max_depth']  = 64 # 8*8 atom level
PARAMS['N_GPUS']     = 1 # inference just 1 GPU or even cpu

# Set hyper-parameters.
batch_size = PARAMS['batch_size']
image_size = PARAMS['RNA_len'] 

# 100k steps should take < 30 minutes on a modern (>= 2017) GPU.
num_training_updates = 50 
num_hiddens          = 16
num_residual_hiddens = 4
num_residual_layers  = 4
embedding_dim = 8   
# The higher this value, the higher the capacity in the information bottleneck.
num_embeddings = 3

sub_str = '_L100_lre5_emb'+str(num_embeddings)+'_dim'+str(embedding_dim)

# commitment_cost should be set appropriately. It's often useful to try a couple
# of values. It mostly depends on the scale of the reconstruction cost
# (log p(x|z)). So if the reconstruction cost is 100x higher, the
# commitment_cost should also be multiplied with the same amount.
commitment_cost = 0.25
# Use EMA updates for the codebook (instead of the Adam optimizer).
# This typically converges faster, and makes the model less dependent on choice
# of the optimizer. In the VQ-VAE paper EMA updates were not used (but was
# developed afterwards). See Appendix of the paper for more details.
vq_use_ema = True
# This is only used for EMA updates.
decay = 0.99
learning_rate = 3e-5
_BATCH_NORM_DECAY = 0.99
_BATCH_NORM_EPSILON = 1e-3


def conv2d_layer(pad_in, n_filers=None, conv_name=None, k=None, args=None, raw_sq_input=None):
    
    #if k is None:
    #    k = params["resnet_k"]

    pad_val = (k-1)//2
    paddings = tf.constant([[0,0], [pad_val, pad_val], [pad_val, pad_val], [0,0]])
    padded_inputs = tf.pad(tensor=pad_in,
                           paddings=paddings,
                           mode="CONSTANT",
                           constant_values=0)
    
    conv = tf.layers.conv2d(inputs=padded_inputs,
                            filters=n_filers,
                            kernel_size=[k,k],
                            strides=[1,1],
                            padding="VALID",
                            use_bias=False, 
                            kernel_initializer=tf.initializers.variance_scaling(),
                            name=conv_name)
    return conv


def batch_norm(inputs, training, data_format, batch_norm_name):
    return tf.layers.batch_normalization(inputs=inputs, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
                                         scale=True, training=training, fused=True, name=batch_norm_name)


def block_layer(pos, inputs, num_hiddens, training):
    shortcut = inputs
    inputs = batch_norm(inputs, training, 'channels_last', 'batch_norm_' + str(pos) + '_0')
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_layer(inputs, num_hiddens, 'conv_' + str(pos) + '_0', 5)
    inputs = batch_norm(inputs, training, 'channels_last', 'batch_norm_' + str(pos) + '_1')
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_layer(inputs, num_hiddens, 'conv_' + str(pos) + '_1', 3)
    return inputs + shortcut


def dec_block_layer(pos, inputs, num_hiddens):
    shortcut = inputs
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_layer(inputs, num_hiddens, 'conv_' + str(pos) + '_0', 5)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_layer(inputs, num_hiddens, 'conv_' + str(pos) + '_1', 3)
    return inputs + shortcut


class Encoder(snt.AbstractModule):
  def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
               name='encoder'):
    super(Encoder, self).__init__(name=name)
    self._num_hiddens = num_hiddens
    self._num_residual_layers = num_residual_layers
    self._num_residual_hiddens = num_residual_hiddens
    
  def _build(self, x, train_ph):
    h = conv2d_layer(x, self._num_hiddens*2, "enc_1", 7)
    h = tf.nn.relu(h)
    h = conv2d_layer(h, self._num_hiddens, "enc_2", 5)
    h = tf.nn.relu(h) 
    h = conv2d_layer(h, self._num_residual_hiddens, "enc_3", 3)
    h = block_layer(0, h, self._num_residual_hiddens, train_ph)
    return h


class Decoder(snt.AbstractModule):
  def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
               name='decoder'):
    super(Decoder, self).__init__(name=name)
    self._num_hiddens = num_hiddens
    self._num_residual_layers = num_residual_layers
    self._num_residual_hiddens = num_residual_hiddens
  
  def _build(self, x, features_mask_placeholder):
    d = conv2d_layer(x, self._num_hiddens / 2, "dec_1", 7)
    d = dec_block_layer(3, d, self._num_hiddens / 2)
    d = conv2d_layer(d, self._num_hiddens , "dec_2", 5)
    d = dec_block_layer(4, d, self._num_hiddens )
    d = conv2d_layer(d, self._num_hiddens*2 , "dec_3", 5)
    d = conv2d_layer(d, self._num_hiddens*4 , "dec_3b", 5)
    d = dec_block_layer(5, d, self._num_hiddens*4)
    x_recon = dec_block_layer(6, d, self._num_hiddens * 4 ) * features_mask_placeholder
    return tf.nn.relu(x_recon) 



##############################
### define VQVAE graph
### : encoder - decoder 
##############################

g = tf.Graph()
with g.as_default():
    with tf.variable_scope('resnet', reuse=tf.AUTO_REUSE):
        tconfig=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

        c1 = []
        device_list = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
        print("device_list: %s \n" % device_list)
        features_placeholder_full      = tf.placeholder(tf.float32, (None ,PARAMS['RNA_len'],PARAMS['RNA_len'] ,PARAMS['max_depth'] ))
        features_mask_placeholder_full = tf.placeholder(tf.float32, (None, PARAMS['RNA_len'],PARAMS['RNA_len'] ,PARAMS['max_depth'] ))
        train_ph                  = tf.placeholder(tf.bool, ())
        seq_ph_full = tf.placeholder(tf.int32, [None])
        step_ph  = tf.placeholder(tf.float32, ())
        start_idx = 0

        features_placeholder      = features_placeholder_full[start_idx:start_idx + PARAMS['batch_size']]
        features_mask_placeholder = features_mask_placeholder_full[start_idx:start_idx + PARAMS['batch_size']]
        seq_ph                    = seq_ph_full[start_idx:start_idx + PARAMS['batch_size']]
        one_list = []
        for m in range(0,PARAMS['batch_size']) :
            newone_mask = tf.ones([seq_ph[m], seq_ph[m]])
            paddings = [[0, PARAMS['RNA_len']-tf.shape(newone_mask)[0]], [0, PARAMS['RNA_len']-tf.shape(newone_mask)[0]]]
            newone_mask = tf.pad(newone_mask, paddings, "CONSTANT")
            one_list.append(newone_mask)
        one_mask = tf.stack(one_list)
        one_exp_list = []
        for m in range(0,embedding_dim) :
            one_exp_list.append(one_mask)
        one_mask_tile = tf.stack(one_exp_list,-1)   
        
        # Build modules.
        encoder = Encoder(num_hiddens, embedding_dim, embedding_dim)
        decoder = Decoder(num_hiddens, num_residual_layers, PARAMS['max_depth'])
        # Process inputs with conv stack, finishing with 1x1 to get to correct size.
        x = features_placeholder * features_mask_placeholder
        enc_x = encoder(x, train_ph) 

        if vq_use_ema:
            vq_vae = snt.nets.VectorQuantizerEMA(
                    embedding_dim=embedding_dim,
                    num_embeddings=num_embeddings,
                    commitment_cost=commitment_cost,
                    decay=decay)
        else:
            vq_vae = snt.nets.VectorQuantizer(
                    embedding_dim=embedding_dim,
                    num_embeddings=num_embeddings,
                    commitment_cost=commitment_cost)
        zp = conv2d_layer(enc_x, embedding_dim, "to_vq", 3) 

        # embeddings should be symmetric
        z_T = tf.concat( [tf.stack([tf.reshape(tf.transpose(zp[i,:,:,ilast]),[1,PARAMS['RNA_len'],PARAMS['RNA_len']]) for ilast in range(embedding_dim)], axis=-1) for i in range(batch_size)] , axis=0) 
        z  = 0.5 * (zp + z_T) * one_mask_tile 

        # For training
        vq_output_train = vq_vae(z, is_training=True)
        dec_in = vq_output_train["quantize"] * one_mask_tile
        x_recon = decoder(dec_in, features_mask_placeholder) * features_mask_placeholder
        # For evaluation, make sure is_training=False!
        vq_output_eval = vq_vae(z, is_training=False)
        x_recon_eval = decoder(vq_output_eval["quantize"], features_mask_placeholder)
        recon_error = tf.reduce_mean((x_recon - x)**2 * features_mask_placeholder) 
        loss = recon_error + vq_output_train["loss"] 
        c1.append(loss)
        start_idx = start_idx + PARAMS["batch_size"]
        tot_loss = tf.add_n(c1) / PARAMS['N_GPUS']

        # The following is a useful value to track during training.
        # It indicates how many codes are 'active' on average.
        perplexity = vq_output_train["perplexity"] 

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.train.get_global_step()
        tvars = tf.trainable_variables()

        gradients = optimizer.compute_gradients(tot_loss, tvars, colocate_gradients_with_ops=True)
        minimize_op = optimizer.apply_gradients(gradients, global_step=global_step, name="train")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)
        
        saver = tf.train.Saver( max_to_keep=0)
        init_op = tf.global_variables_initializer()




###############################
### Dataset lodaer
### : batch functionality
###############################

### pdb ids train
pdb_ids = np.array(list(traindata_df.index))

### pdb ids val
pdb_ids_val = np.array(list(valdata_df.index))


### pdb ids test
pdb_ids_test = np.array(list(testdata_df.index))

### calc sampling probability to sample based on clustering and pdb_id counts
traindata_df['pdb_id'] = [s[:4] for s in traindata_df.index]
value_counts = traindata_df['pdb_id'].value_counts()
count_list = []
for pdb_id in traindata_df['pdb_id']:
    count_list.append(value_counts.loc[pdb_id]/len(traindata_df))
    
traindata_df['probability'] = count_list

traindata_df['pdb_id'] = [s[:4] for s in traindata_df.index]
value_counts = traindata_df['pdb_id'].value_counts()
count_list = []
for pdb_id in traindata_df['pdb_id']:
    count_list.append(1/value_counts.loc[pdb_id])
traindata_df['probability'] = count_list 
traindata_df['probability'] = traindata_df['probability'] / np.sum(count_list) 




def batch_data_train( batch_size = PARAMS['batch_size']*PARAMS['N_GPUS'], trainflag='train'):
    if trainflag == 'train':
        sampled_pdb_ids = np.random.choice(pdb_ids, batch_size, replace= False,p=traindata_df['probability'])
        return traindata_df.loc[sampled_pdb_ids]['len'].values,             np.stack([np_pad(get_3D_tensor(tcoord)) for tcoord in traindata_df.loc[sampled_pdb_ids]['coords'].values]),             np.stack([np_pad_mask(get_3D_tensor(tcoord)) for tcoord in traindata_df.loc[sampled_pdb_ids]['mask'].values])
    
    if trainflag == 'valid':
        sampled_pdb_ids_val = np.random.choice(pdb_ids_val, batch_size, replace= False)
        return valdata_df.loc[sampled_pdb_ids_val]['len'].values,             np.stack([np_pad(get_3D_tensor(tcoord)) for tcoord in valdata_df.loc[sampled_pdb_ids_val]['coords'].values]),             np.stack([np_pad_mask(get_3D_tensor(tcoord)) for tcoord in valdata_df.loc[sampled_pdb_ids_val]['mask'].values])       
                                           
    if trainflag == 'test':
        sampled_pdb_ids_test = np.random.choice(pdb_ids_test, batch_size, replace= False)
        return testdata_df.loc[sampled_pdb_ids_test]['len'].values,             np.stack([np_pad(get_3D_tensor(tcoord)) for tcoord in testdata_df.loc[sampled_pdb_ids_test]['coords'].values]),             np.stack([np_pad_mask(get_3D_tensor(tcoord)) for tcoord in testdata_df.loc[sampled_pdb_ids_test]['mask'].values])        




def iter_data_train( i, batch_size = PARAMS['batch_size']*PARAMS['N_GPUS'], trainflag='train'):
    if trainflag == 'train':
        sampled_pdb_ids = pdb_ids[i*batch_size:(i+1)*batch_size]#np.random.choice(pdb_ids, batch_size, replace= False)
        if len(sampled_pdb_ids)<batch_size:
            sampled_pdb_ids = pdb_ids[-batch_size:]
        return sampled_pdb_ids,traindata_df.loc[sampled_pdb_ids]['len'].values,             np.stack([np_pad(get_3D_tensor(tcoord)) for tcoord in traindata_df.loc[sampled_pdb_ids]['coords'].values]),             np.stack([np_pad_mask(get_3D_tensor(tcoord)) for tcoord in traindata_df.loc[sampled_pdb_ids]['mask'].values])
    
    if trainflag == 'valid':
        sampled_pdb_ids_val = pdb_ids_val[i*batch_size:(i+1)*batch_size]#np.random.choice(pdb_ids_val, batch_size, replace= False)
        if len(sampled_pdb_ids_val)<batch_size:
            sampled_pdb_ids_val = pdb_ids_val[-batch_size:]
        return sampled_pdb_ids_val,valdata_df.loc[sampled_pdb_ids_val]['len'].values,             np.stack([np_pad(get_3D_tensor(tcoord)) for tcoord in valdata_df.loc[sampled_pdb_ids_val]['coords'].values]),             np.stack([np_pad_mask(get_3D_tensor(tcoord)) for tcoord in valdata_df.loc[sampled_pdb_ids_val]['mask'].values])       
                                           
    if trainflag == 'test':
        sampled_pdb_ids_test = pdb_ids_test[i*batch_size:(i+1)*batch_size]#np.random.choice(pdb_ids_test, batch_size, replace= False)
        if len(sampled_pdb_ids_test)<batch_size:
            sampled_pdb_ids_test = pdb_ids_test[-batch_size:]
        return sampled_pdb_ids_test,testdata_df.loc[sampled_pdb_ids_test]['len'].values,            
    np.stack([np_pad(get_3D_tensor(tcoord)) for tcoord in testdata_df.loc[sampled_pdb_ids_test]['coords'].values]),
    np.stack([np_pad_mask(get_3D_tensor(tcoord)) for tcoord in testdata_df.loc[sampled_pdb_ids_test]['mask'].values])        




with tf.Session(graph=g, config=tconfig) as sess:
    saver.restore(sess, ckpt_dir)

    for file in rollouts:
        with open(file, 'rb') as f:
            seq, embedding = pickle.load(f)
            
        lseq_in   = np.array([len(seq) for i in range(PARAMS['batch_size']) ])
        x_in      = np.ones((PARAMS['batch_size'], 100, 100, 64))
        mask_in   = np.ones((PARAMS['batch_size'], 100, 100, 64))
        feed_dict = {vq_output_eval['encoding_indices']: embedding,
             features_placeholder_full : x_in,  features_mask_placeholder_full : mask_in , 
             train_ph: False, seq_ph_full:lseq_in }
        
        LxLx64_distance_tensor = sess.run(x_recon_eval, feed_dict=feed_dict)
        
        seq_len = len(seq)
        LxLx64_distance_tensor = LxLx64_distance_tensor[0,:seq_len,:seq_len]
        L8xL8_distance_matrix = LxLx64_tensor_to_L8xL8_matrix(symmetrize_LxLx64(LxLx64_distance_tensor))
        L_mask = get_mask_for_sequence(seq)
        
        ##### get final Lx3 coords
        Lx3_coords_reconstructed = L8xL8_matrix_to_Lx3_coords(L8xL8_distance_matrix, L_mask)
        
        ##### write to .pdb file for further  refinement and clustering
        
        occupancy = np.ones(len(Lx3_coords_reconstructed))
        temperature_factor = np.zeros(len(Lx3_coords_reconstructed)) 

        for k in range(1000):
            DO_WRITE = write_8_atom_structure_to_pdb_file(output_dir + '{0}_{1}.pdb'.format(seq, k), Lx3_coords_reconstructed, L_mask, seq, occupancy, temperature_factor)            
