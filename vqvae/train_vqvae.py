import numpy as np
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sonnet as snt
import tensorflow as tf
from six.moves import cPickle
from six.moves import urllib
from six.moves import xrange
from utils import *
from tensorflow.python.client import device_lib


########################################################
##### train , valid , test datasets as pandas dataframes
##### VQVAE :  compress raw LxLx64 dist matrices into LxLx3
########################################################
with open( '../data/substructures_drifts.pkl', 'rb') as f:
    traindata_df, valdata_df, testdata_df = pickle.load( f)

##### SET NEW MAXIMAL SEQ LENGTH  #####
PARAMS               = {}
PARAMS['batch_size'] = 20
PARAMS['RNA_len']    = max_len = 100
PARAMS['max_depth']  = 64 # 8*8 atom level
PARAMS['N_GPUS'] = 5 # we use 5 GPUs to have a effective batch size of 100

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
        with tf.device('/cpu:' + str([x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU'][0][-1])):

            c1 = []
            device_list = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
            print("device_list: %s \n" % device_list)
            features_placeholder_full      = tf.placeholder(tf.float32, (None ,PARAMS['RNA_len'],PARAMS['RNA_len'] ,PARAMS['max_depth'] ))
            features_mask_placeholder_full = tf.placeholder(tf.float32, (None, PARAMS['RNA_len'],PARAMS['RNA_len'] ,PARAMS['max_depth'] ))
            train_ph                  = tf.placeholder(tf.bool, ())
            seq_ph_full = tf.placeholder(tf.int32, [None])
            step_ph  = tf.placeholder(tf.float32, ())
        start_idx = 0
        for d in device_list:
            with tf.device(d):
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
        with tf.device('/cpu:' + str([x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU'][0][-1])):
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




train_error, train_res_recon_error, val_res_recon_error, train_res_perplexity = [],[],[],[]

num_training_updates = 150000
# variable to save every ckpt and plot loss curve
PRINTER = 500
with tf.Session(graph=g, config=tconfig) as sess:
    sess.run(init_op)
    # Train.
    print('Start training')
    for i in range(num_training_updates):

            # pass
            lseq_in, x_in, mask_in = batch_data_train( PARAMS['batch_size']*PARAMS['N_GPUS'], 'train')

            feed_dict = {features_placeholder_full : x_in,  features_mask_placeholder_full : mask_in , train_ph: True, seq_ph_full:lseq_in }
            results = sess.run([train_op, tot_loss, perplexity],
                             feed_dict=feed_dict)

            # may want to study loss and perplexity, usually this setup needs ~60k iterations to train
            train_res_recon_error.append(results[1])
            train_res_perplexity.append(results[2])

            if (i % PRINTER == 0) | (i==1):
                lseq_in, x_in, mask_in = lseq_in, x_in, mask_in = batch_data_train( PARAMS['batch_size']*PARAMS['N_GPUS'], 'test')
                val_feed_dict = {features_placeholder_full : x_in,  features_mask_placeholder_full : mask_in , train_ph: True, seq_ph_full:lseq_in }
                val_results = sess.run( tot_loss, feed_dict=val_feed_dict)  
                val_res_recon_error.append(val_results)
                fig, axs = plt.subplots(1,2, figsize=(10, 2), sharey=False)
                axs[0].plot( range(len(train_res_recon_error)),train_res_recon_error)
                axs[1].plot( range(len(val_res_recon_error)),val_res_recon_error)
                plt.savefig('./ckpt/vqvae_loss.png')
                feed_dict = {features_placeholder_full : x_in,  features_mask_placeholder_full : mask_in , train_ph: False, seq_ph_full:lseq_in }
                print('%d iterations' % (i+1))
                print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
                print('perplexity:  %.3f' % np.mean( train_res_perplexity[-100:]))

                ### save model
                save_path = saver.save(sess, head_ckpt_dir+'./ckpt/vqvae_{0}.ckpt'.format(i))



print('Finished training')



