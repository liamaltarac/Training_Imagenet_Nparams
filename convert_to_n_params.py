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



class NConv2D(tf.keras.layers.Layer):
	def __init__(self, filters, n=3, padding = 'VALID', strides = (1, 1), activation=None, use_bias = True, kernel_regularizer = None, **kwargs):
		super(NConv2D, self).__init__(**kwargs)
		self.filters = filters
		self.kernel_size = (3,3)
		self.activation = activations.get(activation)
		self.padding = padding
		self.anti_kernel_initializer = None 
		self.sym_kernel_initializer = None
		self.kernel_regularizer = kernel_regularizer

		self.bias_initializer = tf.initializers.Zeros()
		self.strides = strides
		self.use_bias = use_bias

		self.indicies = [[(0,0)], [(0,1), (1,0)], [(1,1)], [(2,0), (0,2)], [(1,2), (2,1)], [(2,2)] ]

		self.w_train = None
		self.w_fixed = None
		self.bias = None



		self.n_params = n

		d_train = []
		for  params in self.indicies[:self.n_params]:
			for i,j in params:
				t = np.zeros((3,3))
				t[i,j] =1
				d_train.append(tf.reshape(fft.idctn(t, norm='ortho'), (3,3,1,1)))
		self.d_train = tf.cast(tf.stack(d_train), dtype=tf.float32)




	def get_config(self):
		config = super().get_config()
		config.update({
			"filters": self.filters,
			"padding": self.padding,
			"strides": self.strides,
			"activation": activations.serialize(self.activation),
			"use_bias": self.use_bias,
		})
		return config

	def get_weights(self):
		
		w = tf.math.reduce_sum(self.w_train*self.d_train, axis=0)
		return w, self.bias
	
	

	def build(self, input_shape):
		*_, self.n_channels = input_shape
		print(self.n_channels, self.filters)
		w_example = tf.keras.initializers.HeNormal()(shape=[3, 3,  self.n_channels, self.filters])
		w_train = tf.reshape(tf.reduce_sum(w_example*self.d_train, axis=(1,2)), (self.d_train.shape[0],1,1, self.n_channels, self.filters))
		
		#Empirically scale the weights in order to ensure Glorot Condition
		'''test_in = tf.keras.activations.relu(tf.random.normal(stddev=1., shape=[100, 30,30, self.n_channels],))
		test_output = tf.keras.activations.relu(tf.nn.conv2d(test_in, tf.math.reduce_sum(w*self.d, axis=0) , 1, 'VALID'))
		input_var = tfp.stats.variance(
			test_in, sample_axis=None)      
		output_var= tfp.stats.variance(
			test_output, sample_axis=None)
		scale = tf.math.sqrt(input_var/output_var)
		w *= scale
		print("SCALED BY :", scale)'''


		self.w_train = self.add_weight(shape=[self.d_train.shape[0], 1, 1,  self.n_channels, self.filters], regularizer=None, trainable=True, name="w_train" )
		self.w_train.assign(w_train)


		if self.use_bias:
			self.bias = self.add_weight(shape=(self.filters,), initializer = self.bias_initializer, trainable=True, name="bias" )


	def call(self, inputs, training=None):



		w = tf.math.reduce_sum(self.w_train*self.d_train, axis=0) 
		#tf.print('1', tf.convert_to_tensor(w[:,:,0, 0]), output_stream=sys.stderr)
		y =   tf.nn.conv2d(inputs, w , strides=self.strides, 
						  padding=self.padding)
		
		if self.use_bias:
			y += self.bias
		if self.activation  is not None :
			y= self.activation(y)

		return y
	
	def set_weights(self, w_example):
		w_train = tf.reshape(tf.reduce_sum(w_example*self.d_train, axis=(1,2)), (self.d_train.shape[0],1,1, w_example.shape[-2], w_example.shape[-1]))
		self.w_train.assign(w_train)

	def set_bias(self, b):
		if self.use_bias:
			self.bias.assign(b)



	

def convert_model(model, n=3):

	layers = [l for l in model.layers]

	x = layers[0].output

	# Track layers dynamically
	layer_outputs = {model.layers[0].name: x}

	for i in range(1, len(layers)):
		if 'conv2d' in str(type(layers[i])).lower() and (layers[i].kernel_size == (3,3) or layers[i].kernel_size == 3):
			filters, biases = layers[i].get_weights()

			l = NConv2D(layers[i].filters, padding='SAME', activation=layers[i].activation, name=layers[i].name+f"_{n}p", n=n)
			x = l(x)
			
			l.set_weights(filters)
			l.set_bias(biases)
			layer_outputs[layers[i].name] = x

		else:
			#if isinstance(layers[i], (keras.layers.Add)):  # Handle `Add` layers separately
			inputs = layers[i].input
			if type(inputs) is list:
				inputs = [layer_outputs[inp.name.split('/')[0]] for inp in inputs]
			else:
				inputs = layer_outputs[inputs.name.split('/')[0]]

			l = layers[i]
			x = l(inputs)  
			layer_outputs[layers[i].name] = x

	new_model = Model(inputs=layers[0].input, outputs=x)
	return new_model