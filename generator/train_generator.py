import tensorflow as tf
import numpy as np
import pickle
from tensorflow.python.client import device_lib
import time
import sys,  os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# resent and sequence encoding functions
import resnet_emb_funcs as funcs
import rna_enc_pdb



ckpt_path = './ckpt'


# use compact .npy file for faster training
with open('../data/npy_vqvae_embeddings_28022023.pkl', 'rb') as f:
    traindata, valdata, testdata = pickle.load( f)   

column_seq      = 0
column_z        = 1
column_cm       = 2
column_shape    = 3
column_homologs = 4

    
     


print(len(traindata))    
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

# may need to change depending on available resources
params["N_GPUS"]             = 10
params["default_batch_size"] = 20

enc_seq_pdb   = rna_enc_pdb.RNAEncoder(L=params["RNA_length"], args="seq")
enc_struc_pdb = rna_enc_pdb.RNAEncoder(L=params["RNA_length"], args="struc")


def np_pad_emb(matrix, shift=0):
    M_pad = np.full([params["RNA_length"], params["RNA_length"], params["n_classes"]], params["padding"], dtype=np.float32)
    M_pad[shift:matrix.shape[0]+shift, shift:matrix.shape[1]+shift, :] = matrix
    return M_pad

def np_pad_cm(matrix, shift=0):
    M_pad = np.full([1, params["RNA_length"], params["RNA_length"]], params["padding"], dtype=np.float32)
    M_pad[:, shift:matrix.shape[0]+shift, shift:matrix.shape[1]+shift] = matrix
    return M_pad

def get_shape_for_substructure(shape, i):
    """ UPDATE 2020-04-06: now always does substructure data augmentation """
    (i, Rs) = i
    if len(np.shape(shape))!=1:
        raise Exception('Input array should have shape [R] but has:', np.shape(shape))
    return shape[i:i+Rs]

def transform_argmax(tmp2):
    tmp = tmp2.copy()
    for i in range(len(tmp)):
        for j in range(len(tmp)):
            idx = np.argmax(tmp[i,j])
            tmp[i,j][:] = 0
            tmp[i,j][idx] = 1
    return tmp

batch_size = params["N_GPUS"] * params["default_batch_size"]


with open('./ckpt/params.txt', 'w') as fileobj:
    fileobj.write(str(params))
    
    
    
def batch():
    batch_rnd = traindata[np.random.choice(traindata.shape[0], batch_size, replace=False)]
    homologs_from_batch = np.stack([homologs[np.random.choice(homologs.shape[0], 50, replace=False)] for homologs in batch_rnd[:,column_homologs]])
    pdb_ids = batch_rnd[:, -1]
    return np.concatenate(batch_rnd[:,column_seq]), np.stack(batch_rnd[:,column_z]), np.stack(batch_rnd[:,column_cm]),\
            np.concatenate(batch_rnd[:,column_shape]), homologs_from_batch, pdb_ids 

def valbatch():
    batch_rnd = valdata[np.random.choice(valdata.shape[0], batch_size, replace=False)]
    homologs_from_batch = np.stack([homologs[np.random.choice(homologs.shape[0], 50, replace=False)] for homologs in batch_rnd[:,column_homologs]])
    pdb_ids = batch_rnd[:, -1]
    return np.concatenate(batch_rnd[:,column_seq]), np.stack(batch_rnd[:,column_z]), np.stack(batch_rnd[:,column_cm]),\
            np.concatenate(batch_rnd[:,column_shape]), homologs_from_batch, pdb_ids 

def testbatch():
    batch_rnd = testdata[np.random.choice(testdata.shape[0], batch_size, replace=True)]
    homologs_from_batch = np.stack([homologs[np.random.choice(homologs.shape[0], 50, replace=False)] for homologs in batch_rnd[:,column_homologs]])
    pdb_ids = batch_rnd[:, -1]
    return np.concatenate(batch_rnd[:,column_seq]), np.stack(batch_rnd[:,column_z]), np.stack(batch_rnd[:,column_cm]),\
            np.concatenate(batch_rnd[:,column_shape]), homologs_from_batch, pdb_ids 

def get_input_and_pad_mask(inputs_raw, params):
    
    # onehot.shape == [b,L,4]
    # check if "N" (== value of 4) is in inputs and mask every nucleotide in onehot with 0.25
    onehot = tf.one_hot(indices=inputs_raw, depth=params["seq_vocabsize"])
    equal_N = tf.to_float(tf.equal(inputs_raw, params["N_value"])) * params["prob_N"]
    tile_eq_N = tf.tile(tf.expand_dims(equal_N, axis=-1), [1,1,params["seq_vocabsize"]])
    onehot = onehot + tile_eq_N

    # {reshape_onehot, ones}.shape ==[b,4,L,1]
    # {reshape_onehot_T, ones_T}.shape == [b,4,1,L]
    # {s, sT}.shape == [b,4,L,L]
    # M.shape == [b,8,L,L]
    reshape_onehot = tf.expand_dims(tf.transpose(onehot, perm=[0,2,1]), axis=3)
    reshape_onehot_T = tf.transpose(reshape_onehot, perm=[0,1,3,2])
    ones_T = tf.ones_like(reshape_onehot_T)
    ones = tf.ones_like(reshape_onehot)
    s = reshape_onehot * ones_T
    sT = reshape_onehot_T * ones
    M = tf.concat([s,sT], axis=1)
    
    # generate mask for padding
    equal = tf.to_float(tf.not_equal(inputs_raw, -1))
    tile = tf.tile(tf.reshape(equal, [-1, 1, params["RNA_length"], 1]), [1, 1, 1, params["RNA_length"]])
    pad_mask = tile * tf.transpose(tile, perm=[0,1,3,2])
    
    return M, pad_mask

def exclude_batch_norm(name):
    return 'batch_normalization' not in name



tf.reset_default_graph()
with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    tconfig=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    
    # full x network inputs
    xsequence         = tf.placeholder(tf.int32,   [None, params["RNA_length"]])
    xtargets          = tf.placeholder(tf.int32,   [None, params["RNA_length"], params["RNA_length"]])
    xcontact_mask     = tf.placeholder(tf.float32, [None, 1, params["RNA_length"], params["RNA_length"]])
    xshape            = tf.placeholder(tf.float32, [None, 1, params["RNA_length"], 1])
    xhomologs         = tf.placeholder(tf.int32,   [None, params["nr_homologs"], params["RNA_length"]])
    xshift_vals       = tf.placeholder(tf.int32,   [None,1])
    train_ph          = tf.placeholder(tf.bool, ())
    step_ph           = tf.placeholder(tf.float32, ())
    device_list       = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
    loss_values       = []
    start_idx         = 0
    for d in device_list:
        with tf.device(d):

            
            sequence          = xsequence[start_idx:start_idx + params["default_batch_size"]]

            # targets will get maskes below, so that network only sees partial information
            targets           = xtargets[start_idx:start_idx + params["default_batch_size"]]
            contact_mask      = xcontact_mask[start_idx:start_idx + params["default_batch_size"]]
            shape             = xshape[start_idx:start_idx + params["default_batch_size"]]
            homologs          = xhomologs[start_idx:start_idx + params["default_batch_size"]]
            shift_vals        = xshift_vals[start_idx:start_idx + params["default_batch_size"]]
            
            onehot_tar       = tf.one_hot(targets, 3)
            M, pad_mask      = get_input_and_pad_mask(sequence, params)

            ################################
            ##### prepare homolgs ##########
            ################################
            onehot_homologs  = tf.one_hot(homologs, params["seq_vocabsize"], axis=-1)
            equal_N          = tf.to_float(tf.equal(homologs, params["N_value"])) * params["prob_N"]
            equal_N          = tf.tile(tf.expand_dims(equal_N, axis=-1), [1,1,1,params["seq_vocabsize"]])
            enc_homologs     = onehot_homologs + equal_N
            enc_dense        = tf.layers.dense(tf.layers.dense(enc_homologs, 8),1)
            shape_dense      = tf.layers.dense(tf.layers.dense(shape, 8),1)

            aligned_shape_homologs = tf.concat([enc_dense, shape_dense], axis = 1)
            aligned_shape_homologs = tf.transpose(aligned_shape_homologs, [0,2,1,3])[:,:,:,0]

            ################################
            ##### START self attention #####
            ################################
            
            Q = tf.layers.dense(aligned_shape_homologs, params['units_att'])
            K = tf.layers.dense(aligned_shape_homologs, params['units_att'])
            V = tf.layers.dense(aligned_shape_homologs, 1) 
            attention = tf.matmul(Q, K, transpose_b=True)
            
            d_k       = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
            attention = tf.divide(attention, tf.sqrt(d_k))
            attention = tf.nn.softmax(attention, dim=-1)

            attention = tf.matmul(attention, V)
            attention = (attention + tf.transpose(attention, perm=[0, 2, 1]) ) *0.5
            attention = tf.expand_dims(attention,axis=1) * contact_mask
            ################################
            ##### END self attention #######
            ################################


            ################################
            ##### Stacking part ############
            ################################

            # compute symmetric around diagonal masking
            sym_mask   = funcs.get_target_mask_distributed_v2(sequence, shift_vals, params)
            M_pad = (M * pad_mask) + (pad_mask - 1)
            onehot_tar       = tf.transpose(onehot_tar, perm=[0,3,1,2])
            tar_constraints = (onehot_tar + (onehot_tar - 1)) * (1-contact_mask)
            
            # compute masked targets
            masked_tar = ((onehot_tar + (onehot_tar - 1)) * sym_mask) + (pad_mask - 1)
            masked_tar = (masked_tar * contact_mask) + tar_constraints

            # stack sequence block, masked targets and support coordinate frames into one big block 
            diag_frame = tf.matrix_band_part(tf.ones([tf.shape(M_pad)[0], 1, params["RNA_length"], params["RNA_length"]]), 0, 0)
            diag_frame = (diag_frame + (diag_frame - 1)) * pad_mask + (pad_mask - 1)
            C_layer = ((1-contact_mask) * pad_mask) + (pad_mask - 1)
            STACK_pairing = tf.concat([M_pad,  masked_tar, attention,  C_layer, diag_frame], axis=1)
            STACK_pairing = tf.transpose(STACK_pairing, [0,2,3,1])



            ###################################
            ##### RESIDUAL NETWORK ############
            ###################################
            inputs = funcs.conv2d_layer(STACK_pairing, params, 'conv_prev', args='first_conv',\
                                        raw_sq_input=sequence, pad_mask=pad_mask, shift_vals=shift_vals)


            for i, num_blocks in enumerate(range(params["block_sizes"])):
                inputs = funcs.block_layer(i, inputs, train_ph, params)

            if params["pre_activation"]:
                inputs_bn  = funcs.batch_norm(inputs, train_ph, params["data_format"], 'batch_norm_pre_activation')
                inputs_elu = tf.nn.elu(inputs_bn)
            else:
                inputs_elu = tf.nn.elu(inputs)


            dense_dist  = tf.layers.conv2d(inputs=inputs_elu, filters=params["n_classes"],\
                                kernel_size=[1,1], strides=[1,1], padding='VALID', activation=None, name='distances_dense')
            z_T = tf.concat( [tf.stack([tf.reshape(tf.transpose(dense_dist[i,:,:,ilast]),[1,params['RNA_length'],params['RNA_length']]) for ilast in range(params["n_classes"])], axis=-1) for i in range(params["default_batch_size"])] , axis=0) 
            dense_dist  = 0.5 * (dense_dist + z_T)
            logits_softmax = tf.nn.softmax(dense_dist, axis=-1)
            logits_softmax = logits_softmax * tf.transpose(pad_mask, perm=[0,2,3,1])
            # mask out sequence N parts in logits
            eq_pos_N = tf.to_float(tf.not_equal(sequence, params["N_value"]))
            tile1 = tf.tile(tf.reshape(eq_pos_N, [-1,1,params["RNA_length"],1]), [1,params["RNA_length"],1,1])
            tile2 = tf.tile(tf.reshape(eq_pos_N, [-1,params["RNA_length"],1,1]), [1,1,params["RNA_length"],1])
            logits_masked = logits_softmax * (tile1 * tile2)
            cm_onehot = tf.one_hot(targets, 3) *  tf.transpose(contact_mask, [0,2,3,1])


            # compute loss
            loss = tf.losses.softmax_cross_entropy(onehot_labels=cm_onehot,
                                                   logits=logits_masked, reduction=tf.losses.Reduction.SUM)
            loss_filter_fn = params["loss_filter_fn"] or exclude_batch_norm
            l2_loss = params["weight_decay"] * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables() if loss_filter_fn(v.name)])
            loss = loss + l2_loss
            loss = loss / params["default_batch_size"]
            loss_values.append(loss)
            start_idx = start_idx + params["default_batch_size"]
    
    with tf.device('/cpu:' + str([x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU'][0][-1])):
        # accumulate loss
        loss = tf.add_n(loss_values) / params['N_GPUS']
        learning_rate = funcs.get_learning_rate_v3(params["learning_rate"], params["lr_warmup_steps"], step_ph, params["n_iters"], params["lr_alpha"])
        optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate, beta1=params["optimizer_adam_beta1"],
                                                     beta2=params["optimizer_adam_beta2"], 
                                                     epsilon=params["optimizer_adam_epsilon"])
        global_step = tf.train.get_global_step()
        tvars = tf.trainable_variables()
        gradients = optimizer.compute_gradients(loss, tvars, colocate_gradients_with_ops=True)
        minimize_op = optimizer.apply_gradients(gradients, global_step=global_step, name="train")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=params["keep_ckpt"])
        
        
zero_shift = np.array([[0] for _ in range(batch_size)])
loss_lst = []
lr_lst   = []

# start training loop
with tf.Session(config=tconfig) as sess:
    sess.run(init_op)
    time1 = time.time()
    for i in range(params["n_iters"]):
        tbatch = batch()
        
        _, loss_np, lr_np = sess.run([train_op, loss, learning_rate],\
                                   feed_dict={xsequence: tbatch[column_seq], 
                                              xtargets: tbatch[column_z],
                                              xcontact_mask: tbatch[column_cm].reshape(batch_size,1,100,100),
                                              xshape: tbatch[column_shape],
                                              xhomologs: tbatch[column_homologs],
                                              xshift_vals: zero_shift, train_ph: True, step_ph: i
                                             })

        loss_lst.append(loss_np)
        lr_lst.append(lr_np)
        if (i % 5000)==0:
            time2 = time.time()
            
            # just track time since training can take couple of days on 10 GPUs (GTX 1080i)
            time_diff = time2-time1
            save_path = saver.save(sess, ckpt_path + '/generator_{0}.ckpt'.format(i))
            time1 = time.time()
            tbatch = valbatch()
            valid_loss = sess.run(loss,\
                                   feed_dict={xsequence: tbatch[column_seq], 
                                              xtargets: tbatch[column_z],
                                              xcontact_mask: tbatch[column_cm].reshape(batch_size,1,100,100),
                                              xshape: tbatch[column_shape],
                                              xhomologs: tbatch[column_homologs],
                                              xshift_vals: zero_shift, train_ph: False, step_ph: i
                                             })
            
            tbatch = testbatch()
            test_loss = sess.run(loss,\
                                   feed_dict={xsequence: tbatch[column_seq], 
                                              xtargets: tbatch[column_z],
                                              xcontact_mask: tbatch[column_cm].reshape(batch_size,1,100,100),
                                              xshape: tbatch[column_shape],
                                              xhomologs: tbatch[column_homologs],
                                              xshift_vals: zero_shift, train_ph: False, step_ph: i
                                             })            

            funcs.write_file('./' + 'loss_values.txt', "\niter: %s, trainloss: %.2f, valloss: %.2f, testloss: %.2f, lr: %.6f, time_diff: %.6f" % (i, loss_np, valid_loss, test_loss, lr_np, time_diff))
            
            
