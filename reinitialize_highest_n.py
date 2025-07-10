import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Concatenate , Add, Dot, Activation, Lambda
from tensorflow.keras.models import Model

from tensorflow.image import flip_up_down, flip_left_right, rot90
from tensorflow.linalg import normalize

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy import ndimage, fft

import math

import matplotlib.pyplot as plt
import sys 

from tensorflow.keras import activations


def reinitialize_model(model, n_params=3):

	layers = [l for l in model.layers]

	x = layers[0].output

	# Track layers dynamically
	layer_outputs = {model.layers[0].name: x}


	indicies = [[(0,0)], [(0,1), (1,0)], [(1,1)], [(2,0), (0,2)], [(1,2), (2,1)], [(2,2)] ]




	d_train = []
	for  params in indicies:
		for i,j in params:
			t = np.zeros((3,3))
			t[i,j] =1
			d_train.append(tf.reshape(fft.idctn(t, norm='ortho'), (3,3,1,1)))
	d_train = tf.cast(tf.stack(d_train), dtype=tf.float32)

	n_params = len(sum(indicies[-n_params:], []))

	for i in range(1, len(layers)):
		if 'conv2d' in str(type(layers[i])).lower() and (layers[i].kernel_size == (3,3) or layers[i].kernel_size == 3):
			filters, biases = layers[i].get_weights()

			n_channels = filters.shape[-2]
			n_filters = filters.shape[-1]

			w_random = tf.keras.initializers.HeNormal()(shape=[3, 3,  n_channels, n_filters])
			w_dct = tf.reshape(tf.reduce_sum(w_random*d_train, axis=(1,2)), (d_train.shape[0],1,1, n_channels, n_filters))

			print("DCT SHAPE : ", w_dct.shape )

			w_dct_high =  w_dct[-n_params:, ...] # tf.reshape(tf.reduce_sum(filters*d_train[:n_params], axis=(1,2)), (n_params,1,1, w_random.shape[-2], w_random.shape[-1]))
			print("DCT HIGH: ", w_dct_high.shape )
			print("filters shape: ", filters.shape )
			print("2 shape: ", tf.reduce_sum(filters*d_train[:-n_params], axis=(1,2)).shape )

			#w_dct_high[:n_params, ...] = 0
			w_dct_low = tf.reshape(tf.reduce_sum(filters*d_train[:-n_params], axis=(1,2)), (d_train.shape[0]-n_params,1,1, w_random.shape[-2], w_random.shape[-1]))
			w_dct = tf.concat([w_dct_low, w_dct_high], axis=0)
			print("DCT : ", w_dct.shape )

			w = tf.math.reduce_sum(w_dct*d_train, axis=0) 

			print("W : ", w.shape )

			layers[i].set_weights([w, biases])
			#layers[i].set_bias(biases)
			#layer_outputs[layers[i].name] = x



	return model