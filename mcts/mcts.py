import tensorflow as tf
import numpy as np
import pickle
from tensorflow.python.client import device_lib
import time
import sys, os
import resnet_emb_funcs as funcs
import rna_enc_pdb
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from collections import defaultdict
import operator
from generator import *

#### main parameters, nr_rollouts = for mcts search, N_simrna = how many pdb files as output
puzzle_nr        = 115 #=4xw7 in our test set
nr_rollouts      = 50000
N_simrna         = 1000


with open('../data/npy_vqvae_embeddings_28022023.pkl', 'rb') as f:
    traindata, valdata, testdata = pickle.load( f)       


column_seq      = 0
column_z        = 1
column_cm       = 2
column_shape    = 3
column_homologs = 4


# network parameters
params = {'seq_vocabsize': 4, 'N_value': 4, 'prob_N': 0.25,
          'optimizer_adam_beta1': 0.9, 'optimizer_adam_beta2': 0.997, 
          'optimizer_adam_epsilon': 1e-08, 'weight_decay': 0.01, 
          'loss_filter_fn': False, 'data_format': 'channels_last',
          'pre_activation': True, 'resnet_k': 3, 'RNA_length': 100,
          'n_classes': 3, 'padding': 0, 'struc_vocabsize': 700, 
          'use_outline_padding': True, 'use_shape_data': False, 
          'data_augmentation_shift': True, 
           'N_GPUS': 10, 'n_iters': 10000001, 'default_batch_size': 20, 
          'learning_rate': 0.005, 'lr_warmup_steps': 50000, 
          'lr_alpha': 0.001, 'use_dilation': True, 'block_sizes': 16, 
          'resnet_filter': 26, 'dilation_stdev': 0.01, 'dilation_rate': 4, 
          'use_regularization': True, 
          'keep_ckpt': 2500, 'nr_homologs' : 50, 'units_att' : 64}


    
    
# perform inference for a single puzzle
params["N_GPUS"] = 1
params["default_batch_size"] = 30
batch_size = params["N_GPUS"] * params["default_batch_size"]

enc_seq_pdb   = rna_enc_pdb.RNAEncoder(L=params["RNA_length"], args="seq")
enc_struc_pdb = rna_enc_pdb.RNAEncoder(L=params["RNA_length"], args="struc")

def get_puzzle(puzzle_idx):
    batch_rnd = testdata[[puzzle_idx]]
    homologs_from_batch = np.stack([homologs[np.random.choice(homologs.shape[0], 50, replace=False)] for homologs in batch_rnd[:,column_homologs]])
    pdb_ids = batch_rnd[:, -1]
    return np.concatenate(batch_rnd[:,column_seq]), np.stack(batch_rnd[:,column_z]), np.stack(batch_rnd[:,column_cm]),\
            np.concatenate(batch_rnd[:,column_shape]), homologs_from_batch, pdb_ids 

def get_puzzle_old(puzzle_idx):
    batch_rnd = testdata_old[[puzzle_idx]]
    homologs_from_batch = np.stack([homologs[np.random.choice(homologs.shape[0], 50, replace=False)] for homologs in batch_rnd[:,column_homologs]])
    pdb_ids = batch_rnd[:, -1]
    return np.concatenate(batch_rnd[:,column_seq]), np.stack(batch_rnd[:,column_z]), np.stack(batch_rnd[:,column_cm]),\
            np.concatenate(batch_rnd[:,column_shape]), homologs_from_batch, pdb_ids 

def get_puzzle_batch(puzzle_idx, batch_size):
    batch_rnd = testdata[[puzzle_idx]]
    homologs_from_batch = np.stack([homologs[np.random.choice(homologs.shape[0], 50, replace=False)] for homologs in batch_rnd[:,column_homologs]])
    pdb_ids     = batch_rnd[:, -1]
    seq_batch   = np.concatenate([np.concatenate(batch_rnd[:,column_seq]) for i in range(batch_size)])
    z_batch     = np.concatenate([np.stack(batch_rnd[:,column_z]) for i in range(batch_size)])
    cm_batch    = np.concatenate([np.stack(batch_rnd[:,column_cm]) for i in range(batch_size)])
    shape_batch = np.concatenate([np.concatenate(batch_rnd[:,column_shape]) for i in range(batch_size)])
    homologs_from_batch = np.concatenate([ np.stack([homologs[np.random.choice(homologs.shape[0], 50, replace=False)] for homologs in batch_rnd[:,column_homologs]]) for i in range(batch_size)])
    return seq_batch, z_batch, cm_batch, shape_batch, homologs_from_batch, pdb_ids

def transform_argmax(tmp):
    tmp2 = np.zeros((1,100,100))
    for i in range(len(tmp)):
        for j in range(len(tmp)):
            idx = np.argmax(tmp[i,j])
            tmp2[0,i,j] = idx
    return tmp2


def emb_to_inputactions(targets_emb):
    input_actions_set = np.zeros((1,3,100,100))
    for i in range(100):
        for j in range(100):
            idx = targets_emb[0,i,j]
            input_actions_set[0,:,i,j]   = -1        
            input_actions_set[0,idx,i,j] = +1
    return input_actions_set


def get_zero_actions(batch_size):
    #zero_idx = 1
    eye   = np.eye(100,100)
    zeros = np.zeros((100,100))
    diag_set = np.concatenate([np.stack([np.stack([ zeros-eye, eye, zeros-eye])]) for i in range(batch_size)])
    for p in range(4):
        diag_set[:,:,0,p] = [1,-1,-1]
        diag_set[:,:,1,p] = [1,-1,-1]
        diag_set[:,:,2,p] = [1,-1,-1]
        diag_set[:,:,3,p] = [1,-1,-1]
        diag_set[:,:,4,p] = [1,-1,-1]
    for p in range(96,100):
        diag_set[:,:,97,p] = [1,-1,-1]
        diag_set[:,:,98,p] = [1,-1,-1]
        diag_set[:,:,95,p] = [1,-1,-1]
        diag_set[:,:,96,p] = [1,-1,-1]
        diag_set[:,:,99,p] = [1,-1,-1]
    return diag_set


def set_target_next_action(input_actions, idx1, idx2):
    emb0_action = input_actions.copy()
    
    emb0_action[0,0,idx1,idx2] = +1
    emb0_action[0,1,idx1,idx2] = -1
    emb0_action[0,2,idx1,idx2] = -1
    emb0_action[0,0,idx2,idx1] = +1
    emb0_action[0,1,idx2,idx1] = -1
    emb0_action[0,2,idx2,idx1] = -1
    
    emb1_action = input_actions.copy()
    
    emb1_action[0,0,idx1,idx2] = -1
    emb1_action[0,1,idx1,idx2] = +1
    emb1_action[0,2,idx1,idx2] = -1
    emb1_action[0,0,idx2,idx1] = -1
    emb1_action[0,1,idx2,idx1] = +1
    emb1_action[0,2,idx2,idx1] = -1
    
    emb2_action = input_actions.copy()
    
    emb2_action[0,0,idx1,idx2] = -1
    emb2_action[0,1,idx1,idx2] = -1
    emb2_action[0,2,idx1,idx2] = +1
    emb2_action[0,0,idx2,idx1] = -1
    emb2_action[0,1,idx2,idx1] = -1
    emb2_action[0,2,idx2,idx1] = +1
    return emb0_action, emb1_action, emb2_action


def expand_S_R(action_matrix):
    S_R = np.random.randint(0, seqlen, size = (10 * params["N_GPUS"], 2))
    pixels_per_gpu = 10 #*3 actions=embedding choices = 30 per GPU
    for n, pixel in enumerate(S_R):
        nm = n * 3
        x, y = pixel

        # set symetric action across all embedding dimensions
        action_matrix[nm,:,x,y] = [+1,-1,-1]    
        action_matrix[nm,:,y,x] = [+1,-1,-1]

        action_matrix[nm+1,:,x,y] = [-1,+1,-1]    
        action_matrix[nm+1,:,y,x] = [-1,+1,-1]

        action_matrix[nm+2,:,x,y] = [-1,-1,+1]    
        action_matrix[nm+2,:,y,x] = [-1,-1,+1]
    return action_matrix, S_R


def calc_entropy(softmax_probs, seqlen):
    ##### zero out null probabilites via log(1)=0 since log(0)=-inf
    softmax_probs[softmax_probs==0] = 1
    Hentr = np.log(softmax_probs[:,:seqlen,:seqlen]) * softmax_probs[:,:seqlen,:seqlen]
    return - np.sum(np.sum(np.sum(Hentr,-1),-1),-1)


 eps = 1.0

class Node():
    
    def __init__(self):
        self.Q = 0  # total reward of each node
        self.N = 0  # total visit count for each node
        self.children = dict()  # children of each node
        self.terminal_state = []
        self.terminal_state_q = []
        self.terminal_state_v = []
    
    
    def not_leaf_node(self):
        return len(self.children)!=0
    
    
    def expand_leaf(self, action_matrix):
        
        S_R = np.random.randint(0, seqlen, size = (10 * params["N_GPUS"], 2))
        ##### sort that we only sample upper triangle since embedding is symmetric
        sr = []
        for s in S_R:
            if s[0] < s[1]:
                sr.append(list(s))
            else:
                sr.append([s[1],s[0]])
        S_R = np.array(sr)
        
        pixels_per_gpu = 10 #*3 actions=embedding choices = 30 per GPU
        for n, pixel in enumerate(S_R):
            nm = n * 3
            x, y = pixel

            # set symetric action across all embedding dimensions
            action_matrix[nm,:,x,y] = [+1,-1,-1]    
            action_matrix[nm,:,y,x] = [+1,-1,-1]

            action_matrix[nm+1,:,x,y] = [-1,+1,-1]    
            action_matrix[nm+1,:,y,x] = [-1,+1,-1]

            action_matrix[nm+2,:,x,y] = [-1,-1,+1]    
            action_matrix[nm+2,:,y,x] = [-1,-1,+1]
            
        s_r_a = []
        # add embedding dimensions
        for s in S_R:
            for i in range(3):
                s_r_a.append([s[0],s[1],i])       
        return action_matrix, s_r_a
    
    
    def uct_select(self):
        #### uct select (with eps noise for exploration) untill we hit leaf node
        action_trajectory, path = [], []
        path = [self]
        node = path[-1]
        if len(self.children)==0:
            #print('init select')
            return [],[node]
        
        #cnt = 0
        while True:
            qsu_vals = []
            max_val  = -1000
            #print(cnt, self, self.children)
            nodesum = 0
            for child in set(self.children):
                nodesum += self.children[child].N
            for child in set(self.children):
                exploration_term = np.random.rand() * eps
                #print(child, tree.children[child].Q , np.sqrt(nsum))
                tmp = self.children[child].Q + np.sqrt(nodesum) / (1 + self.children[child].N) + exploration_term
                if tmp > max_val:
                    max_child  = self.children[child]
                    max_action = child
                    max_val    = tmp
                    node       = max_child
            path.append(node)
            self = path[-1]
            action_trajectory.append(max_action)
            print(max_action, max_val, exploration_term)
            
            ##### continue recursive
            #print('while loop',node)
            #cnt = cnt+1
            if not self.not_leaf_node():
                print('last q value: ', max_val)
                return action_trajectory, path
    
    def argmax_select(self):
        #### uct select (with eps noise for exploration) untill we hit leaf node
        action_trajectory, path = [], []
        path = [self]
        node = path[-1]
        if len(self.children)==0:
            #print('init select')
            return ['[0, 0, 0]'],[node]
        
        #cnt = 0
        while True:
            qsu_vals = []
            max_val  = -1000
            #print(cnt, self, self.children)
            nodesum = 0
            for child in set(self.children):
                nodesum += self.children[child].N
            for child in set(self.children):
                #exploration_term = np.random.rand() * eps
                #print(child, tree.children[child].Q , np.sqrt(nsum))
                tmp = self.children[child].Q + np.sqrt(nodesum) / (1 + self.children[child].N)# + exploration_term
                if tmp > max_val:
                    max_child  = self.children[child]
                    max_action = child
                    max_val    = tmp
                    node       = max_child
                    nval    = self.children[child].N
            path.append(node)
            self = path[-1]
            action_trajectory.append(max_action)
            print(max_action, max_val, nval)
            
            ##### continue recursive
            #print('while loop',node)
            #cnt = cnt+1
            if not self.not_leaf_node():
                return action_trajectory, path



# we can set the diagonal and the next diagonal elements sicne in vqvae embedding they will always be closest distance class
def get_zero_actions(batch_size):
    
    
    a = np.ones((1, 100))[0]
    b = np.ones((1, 99))[0]
    c = np.ones((2, 98))[0]
    d = np.ones((3, 97))[0]
    eye = np.diag(a, 0) + np.diag(b, -1) + np.diag(b, 1)+ np.diag(c, -2) + np.diag(c, 2)+ np.diag(d, -3) + np.diag(d, 3)
    
    zeros = np.zeros((100,100))
    # first dimension on the ckpt networks is closest distance
    diag_set = np.concatenate([np.stack([np.stack([eye, zeros-eye, zeros-eye])]) for i in range(batch_size)])
    return diag_set 



##########################################################################
##########################################################################
###### MCTS rollout loop #################################################
##########################################################################
##########################################################################

selectors        = params["N_GPUS"]
zero_action_init = get_zero_actions(batch_size)
Nexp             = 10
lambdaH          = 1.0
logK             = np.log(3)
debug            = False
ckpt_idx         = 5100000
qsa              = {}
usa              = {}
nsa              = {}
zero_shift       = np.array([[0] for _ in range(batch_size)])

##### take all sequence information from puzzle to init mcts
tbatch           = get_puzzle_batch(puzzle_nr, batch_size) 
seq_encoded      = tbatch[0][0]
seq              = decode_encseq(seq_encoded)
seqlen           = len(seq)
pdb_id           = tbatch[-1][0]
print(pdb_id)    
targets_emb      = tbatch[column_z]





zero_action      = get_zero_actions(batch_size)
feed_dict={
          xsequence:tbatch[column_seq], 
          xcontact_mask: xcontact_mask: tbatch[column_cm],
          xshape: tbatch[column_shape],
          xhomologs: tbatch[column_homologs],
          xshift_vals: zero_shift, train_ph: False, step_ph: ckpt_idx,
          ################################
          ###### only show diagonal ######
          ################################
            actions : zero_action
         }




with tf.Session(config=tconfig) as sess:
    saver.restore(sess,'../generator/ckpt/generator_{0}.ckpt'.format(ckpt_idx))
    tree = Node()
    
    ##### calc H0
    feed_dict.update({actions : zero_action_init.copy()})
    softmax_probs_t0 = sess.run( soft_maps, feed_dict)
    H0 = calc_entropy(softmax_probs_t0, seqlen)[0]    
    
    trajectory = []
    
    print('add new leafs to tree to have higher sampling quality at the beginning. ')
    ########################################################
    ##### MCTS EXPANSION: sample new leaf nodes ############
    ########################################################

    print('-------------------------') 
    for k in range(nr_rollouts):
        print(tree)
        
        print(k, '+++++++++++++++++++++++++++++++', trajectory)
        
        ########################################################
        ##### MCTS: SELECTION ##################################
        ##### at t=0 first rollout need to directly select #####
        ##### else take last selected node                 #####
        ########################################################

        leaf = tree
        
                            
        trajectory, path = leaf.uct_select( )
        
        ##### update visit count after selection
        for move,childnode in zip(trajectory,path):
            childnode.children[move].N += 1
        
        ##### save rollouts where 5% of pixels are set since then usually the  
        ##### model can already generate good embeddings with argmax fillup   
        
        if len(trajectory) == int(seqlen**2/2*0.05):
            tree.terminal_state.append(trajectory)
            tree.terminal_state.append(path[-1].Q)
            
            ##### enforce extrapolation for terminal node states
            for move,childnode in zip(trajectory,path):
                childnode.children[move].N += Nexp
            
        if len(trajectory) <= int(seqlen**2/2*0.05):
            
            leaf = path[-1]
            #print('added new leaf node', leaf)
        
            new_action_init = zero_action_init.copy()
            for s in trajectory:
                s = s.split(',')
                xval, yval, zval = int(s[0][1:]), int(s[1]), int(s[2][:-1])
                ##### set trajectory
                new_action_init[:,:,xval,yval] = [-1,-1,-1]
                new_action_init[:,:,yval,xval]  = [-1,-1,-1]
                new_action_init[:,zval,xval,yval] = +1
                new_action_init[:,zval,yval,xval]  = +1

            feed_dict.update({actions : new_action_init})
            softmax_probs_t = sess.run( soft_maps, feed_dict)    

            ########################################################
            ##### MCTS EXPANSION: sample new leaf nodes ############
            ########################################################
            
            new_actions, s_r = leaf.expand_leaf(new_action_init.copy())
            feed_dict.update({actions : new_actions})
            softmax_probs_tplus1 = sess.run( soft_maps, feed_dict)
            
            deltaHentr = - calc_entropy(softmax_probs_tplus1, seqlen) + calc_entropy(softmax_probs_t, seqlen)
            print(deltaHentr.mean(), deltaHentr.min(), deltaHentr.max())
            str_move_list = [str(s) for s in s_r]
            
            Scount = 0
            clean_trajectory = []
            for move, child, delta in zip(s_r, str_move_list, deltaHentr):
                # ignore diagonal (they are set) and what has been already in path trajectory
                if (move[0] != move[1]):
                    if child not in trajectory:
                        if delta > lambdaH * logK:
                            Scount += 1
                            clean_trajectory.append(child)
                            
                            
            ########################################################
            ##### Value function computation #######################
            ########################################################
            T = len(trajectory)+1    

            trace_rollout = zero_action_init.copy()
            for r, trace in enumerate(clean_trajectory):
                ### set trace of trajectory
                x,y,e = int(trace.split(',')[0][1:]), int(trace.split(',')[1]), int(trace.split(',')[2][:-1])
                trace_rollout[:r,:,x,y] = [-1,-1,-1]
                trace_rollout[:r,:,y,x] = [-1,-1,-1]
                trace_rollout[:r,e,x,y] = +1
                trace_rollout[:r,e,y,x] = +1
                r = r+1

            feed_dict.update({actions : trace_rollout})
            softmax_probs_deltaH = sess.run( soft_maps, feed_dict)
            H_0_Tplus1 = calc_entropy(softmax_probs_deltaH, seqlen)[:T]
            vcumulative = (H0-H_0_Tplus1)/H0
            ########################################################
                            
            t=1
            for move, child, delta in zip(s_r, str_move_list, deltaHentr):
                if debug:print(child, delta)
                # ignore diagonal (they are set) and what has been already in path trajectory
                if (move[0] != move[1]):
                    if child not in trajectory:
                        if delta > lambdaH * logK:
                            if debug:print('NEW CHILD NODE: ',child, delta)
                            child_node = Node()
                            child_node.N += 1
                            child_node.Q = sum(vcumulative[:t])/t #+ (H0 - delta)/H0
                            leaf.children.update({child : child_node})
                            t = t+1
                            
            
            ##### winner takes it all q updates
            for move,childnode in zip(trajectory,path):
                childnode.children[move].N += Scount
                childnode.children[move].Q += Scount/childnode.children[move].N * (deltaHentr.max()-childnode.children[move].Q)
         
        ##### NEW CONDITION #####
        else:
            
            ########################################################
            ##### Value function computation #######################
            ########################################################
            T = len(trajectory)+1    

            trace_rollout = zero_action_init.copy()
            for r, trace in enumerate(clean_trajectory):
                ### set trace of trajectory
                x,y,e = int(trace.split(',')[0][1:]), int(trace.split(',')[1]), int(trace.split(',')[2][:-1])
                trace_rollout[:r,:,x,y] = [-1,-1,-1]
                trace_rollout[:r,:,y,x] = [-1,-1,-1]
                trace_rollout[:r,e,x,y] = +1
                trace_rollout[:r,e,y,x] = +1
                r = r+1

            feed_dict.update({actions : trace_rollout})
            softmax_probs_deltaH = sess.run( soft_maps, feed_dict)
            H_0_Tplus1 = calc_entropy(softmax_probs_deltaH, seqlen)[:T]
            vcumulative = sum((H0-H_0_Tplus1)/H0)/T
            #vcumulative = (H0-H_0_Tplus1)/H0#/T
            ########################################################
            
            
            print('FOUND TERMINAL LEAF:::::::::::::::::::::', trajectory)
            tree.terminal_state.append(trajectory)
            tree.terminal_state_q.append(path[-1].Q)
            tree.terminal_state_v.append(vcumulative)
            
            ##### enforce extrapolation for terminal node states
            for move,childnode in zip(trajectory,path):
                childnode.children[move].N += Nexp


#########################################################################
########## after sampling the MCTS search tree, uct sample rollouts #####
########## as candidate embeddings for further SimRNA refinement    #####
#########################################################################
with tf.Session(config=tconfig) as sess:
    saver.restore(sess, '../generator/ckpt/generator_{0}.ckpt'.format(ckpt_idx))
    for k in range(N_simrna):
        trajectory, path = tree.uct_select( )
        trace_rollout = zero_action_init.copy()
        for r, trace in enumerate(trajectory):
            ### set trace of trajectory
            x,y,e = int(trace.split(',')[0][1:]), int(trace.split(',')[1]), int(trace.split(',')[2][:-1])
            trace_rollout[0,:,x,y] = [-1,-1,-1]
            trace_rollout[0,:,y,x] = [-1,-1,-1]
            trace_rollout[0,e,x,y] = +1    
            trace_rollout[0,e,y,x] = +1

        #### fill out remaining pixels with generator and take argmax
        feed_dict.update({actions : trace_rollout})
        softmax_probs_argmax = sess.run( soft_maps, feed_dict)
        rollout = transform_argmax(softmax_probs_argmax[0])
        with open('./rollouts/{0}_{1}.pkl'.format(pdb_id,k), 'wb') as f:
            pickle.dump(rollout, f)


#########################################################################
########## continue with rollout_refinements.py ...                 #####
#########################################################################