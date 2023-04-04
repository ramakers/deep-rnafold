### Version from 2023-03-04 ###

import tensorflow as tf
import numpy as np

_BATCH_NORM_DECAY = 0.99
_BATCH_NORM_EPSILON = 1e-5


# ------------------------------------------------
# MODEL PARAMS
# ------------------------------------------------

def params_dict():
    
    params = {}
    
    params["seq_vocabsize"] = 4
    params["N_value"] = 4
    params["prob_N"] = 0.25
    
    params["resnet_k"] = 3
    params["data_format"] = 'channels_last'
    params["pre_activation"] = True
    
    params["loss_filter_fn"] = False
    params["weight_decay"] = 1e-4

    params["optimizer_adam_beta1"] = 0.9
    params["optimizer_adam_beta2"] = 0.997
    params["optimizer_adam_epsilon"] = 1e-09

    return params


# ------------------------------------------------------------
# PREPROCESSING INPUT
# ------------------------------------------------------------

def get_coordinate_frame(sq_input, shift_vals, params):
    
    seq_len = tf.reshape(tf.reduce_sum(tf.to_int64(tf.not_equal(sq_input, -1)), axis=-1), [-1,1])
    split_seq_len = tf.split(seq_len, params["default_batch_size"], axis=0)
    split_shift_vals = tf.split(shift_vals, params["default_batch_size"], axis=0)

    scatter_lst1, scatter_lst2 = [], []
    for s1, s2 in zip(split_seq_len, split_shift_vals):
        indices1 = tf.concat((s2, s2+s1+1), axis=0)
        concat1 = tf.concat((tf.zeros(s2[0]), tf.ones(s1[0]+2), tf.zeros(params["RNA_length"] - s2[0] - s1[0])), axis=0)
        updates1 = tf.tile(tf.reshape(concat1, [1,-1]), [2,1])
        scatter1 = tf.scatter_nd(indices1, updates1, tf.constant([params["RNA_length"]+2, params["RNA_length"]+2], dtype=tf.int64))
        scatter_lst1.append(scatter1)

        indices2 = indices1
        concat2 = tf.concat((tf.zeros(s2[0]+1), tf.ones(s1[0]), tf.zeros(params["RNA_length"]+2 - s2[0]-1 - s1[0])), axis=0)
        updates2 = tf.tile(tf.reshape(concat2, [1,-1]), [2,1])
        scatter2 = tf.scatter_nd(indices2, updates2, tf.constant([params["RNA_length"]+2, params["RNA_length"]+2], dtype=tf.int64))
        scatter_lst2.append(scatter2)

    scatter_stack1 = tf.stack(scatter_lst1, axis=0)
    scatter_stack2 = tf.transpose(tf.stack(scatter_lst2, axis=0), perm=[0,2,1])

    frame = scatter_stack1 + scatter_stack2
    
    return frame


# ------------------------------------------------------------
# RESNET FUNCTIONS (batch_norm, conv2d, block_layer)
# ------------------------------------------------------------

def batch_norm(inputs, training, data_format, batch_norm_name):
    return tf.layers.batch_normalization(inputs=inputs, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
                                         scale=True, training=training, fused=True, name=batch_norm_name)


def conv2d_layer(pad_in, params, conv_name, k=None, args=None, raw_sq_input=None, pad_mask=None, shift_vals=None):
    
    if k is None:
        k = params["resnet_k"]

    pad_val = (k-1)//2
    paddings = tf.constant([[0,0], [pad_val, pad_val], [pad_val, pad_val], [0,0]])
    padded_inputs = tf.pad(tensor=pad_in,
                           paddings=paddings,
                           mode="CONSTANT",
                           constant_values=0)

    if args == 'first_conv':
        coordinate_frame = tf.expand_dims(get_coordinate_frame(raw_sq_input, shift_vals, params), axis=-1)
        padded_pad_mask = tf.pad(tf.transpose(1 - pad_mask, perm=[0,2,3,1]), paddings, mode='CONSTANT', constant_values=1)
        coordinate_frame = coordinate_frame * padded_pad_mask
        padded_inputs = tf.concat([padded_inputs, coordinate_frame], axis=-1)
        #return padded_inputs
        
    conv = tf.layers.conv2d(inputs=padded_inputs,
                            filters=params["resnet_filter"],
                            kernel_size=[k,k],
                            strides=[1,1],
                            padding="VALID",
                            use_bias=False, 
                            kernel_initializer=tf.initializers.variance_scaling(),
                            activation=None,
                            name=conv_name)
    return conv



def atrous_conv2d(pad_in, params, conv_name, weight_name, atrous_rate, k=None):

    if k is None:
        k = params["resnet_k"]

    pad_val = ((k-1)//2) * atrous_rate
    paddings = tf.constant([[0,0], [pad_val, pad_val], [pad_val, pad_val], [0,0]])
    padded_inputs = tf.pad(tensor=pad_in,
                           paddings=paddings,
                           mode="CONSTANT",
                           constant_values=0)
    
    weights_var = tf.get_variable(weight_name, [k, k, params["resnet_filter"], params["resnet_filter"]], 
                                  initializer=tf.random_normal_initializer(stddev=params["dilation_stdev"]))
    
    conv = tf.nn.atrous_conv2d(value=padded_inputs, 
                               filters=weights_var,
                               rate=atrous_rate,
                               padding='VALID', 
                               name=conv_name)
    return conv


def block_layer(pos, inputs, training, params):

    if pos % 2:
        atrous_rate = params["atrous_rate"]
    else:
        atrous_rate = 1
    
    shortcut = inputs
    
    inputs = batch_norm(inputs, training, params["data_format"], 'batch_norm_' + str(pos) + '_0')
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_layer(inputs, params, 'conv_' + str(pos) + '_0', k=3)

    inputs = batch_norm(inputs, training, params["data_format"], 'batch_norm_' + str(pos) + '_1')
    inputs = tf.nn.relu(inputs)
    
    if params["use_dilation"]:
        inputs = atrous_conv2d(inputs, params, 'atrous_conv_' + str(pos), 'w_dilation_' + str(pos), atrous_rate, k=3)
    else:
        inputs = conv2d_layer(inputs, params, 'conv_' + str(pos) + '_1', k=3)
    
    return inputs + shortcut


# ------------------------------------------------
# LEARNING RATE
# ------------------------------------------------

def get_lr(learning_rate, step, lr_decay):
    learning_rate *= lr_decay ** step
    return learning_rate


# ------------------------------------------------
# N_PARAMS
# ------------------------------------------------
# modified from https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model (2019-02-05)

def get_num_of_params():
    total_params = 0
    for var in tf.trainable_variables():
        shape = var.get_shape()
        var_param = 1
        for dim in shape:
            var_param *= dim.value
        total_params += var_param   
    return total_params


# ------------------------------------------------
# WRITE FILE
# ------------------------------------------------

def write_file(filename, text):
    f = open(filename, "a")
    f.write(text)
    f.close()
