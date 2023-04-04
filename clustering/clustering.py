import os
import glob
import numpy as np


pdb_list = [
            '3owz_A', '3owz_B',
 '4l81_A', '4lck_C', '4lck_F',
 '4p9r_A', '4qlm_A', '4r4v_A',
 '4xw7_A', '5di4_A', '5di4_B',
 '5k7c_A', '5k7c_B', '5kpy_A',
 '5lyu_A', '5lyu_B', '5nwq_A',
 '5nwq_B', '5swd_A', '5swd_B',
 '5swe_X', '5tpy_A', '5y85_A',
 '5y85_B', '5y85_C', '5y85_D',
 '6jq5_A', '6jq5_B', '6p2h_A',
 '6pom_A', '6pom_B', '6ufm_A', '6ufm_B','5tpy_1_a_0', '6pom_0', '6jq6_0', '5ddo_0', '5lys_0',
               '6pmo_0', '6pmo_1_a_0', '4lck_1_b_0', '4lck_1_f_0', '3mei_0',
               '4gxy_0', '5y85_0', '4qlm_1_a_0', '6jq5_0', '6ufm_0', '5t5a_0', 
               '4p9r_0', '3owz_0', '6e8u_1_b_0', '5kpy_0', '6p2h_0', '5di4_0', '5nwq_1_a_0', 
               '6ol3_0', '6p2h_1_a_0', '4xw7_0', '3v7e_1_c_0', '3mei_1_a-b_0', '4r4v_0',
               '4gxy_1_a_0', '5nz6_0', 
               '5t5a_1_a_0', '4l81_0', '5nwq_0', '4r4v_1_a_0', '5iem_0', '5lys_1_b_0']


directory = './rollouts_replica/'

for pdb_id in pdb_list:
    # concat all trafl files into big .trafl
    cmd = 'cat {1}{0}*trafl > /gpfs/project/ramakers/SimRNA0/{0}.trafl'.format(pdb_id, directory)
    os.system(cmd)

    # produce five cluster centers based on concatinated .trafl file
    cmd = '/gpfs/project/ramakers/SimRNA0/clustering /gpfs/project/ramakers/SimRNA0/{0}.trafl 0.25 5.0 10.0 15.0 20.0 25.0'.format(pdb_id)
    os.system(cmd)

    cmd = 'cp {1}{0}0.pdb-000011_AA.pdb /gpfs/project/ramakers/SimRNA0/'.format(pdb_id, directory)
    os.system(cmd)
    cmd = 'mv /gpfs/project/ramakers/SimRNA0/{0}0X95.pdb-000011_AA.pdb /gpfs/project/ramakers/SimRNA0/{0}.pdb'.format(pdb_id)
    os.system(cmd)

    for rms in [2, 4, 6, 8, 10, 12, 16, 20, 25, 30]:

        cmd = './SimRNA_trafl2pdbs ./trafl_files/{0}.pdb ./trafl_files/{0}_thrs{1}.00A_clust01.trafl 1 AA'.format(pdb_id,rms)
        os.system(cmd)