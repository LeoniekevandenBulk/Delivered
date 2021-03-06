# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:14:02 2017

@author: WoutervanderWeel
"""

__author__ = 'Guido Zuidhof'
import numpy as np
import theano
#import theano.tensor as T
from theano import pp, tensor as T

import lasagne
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, ConcatLayer, Upscale2DLayer
from lasagne.init import HeNormal
from lasagne import nonlinearities
from lasagne.regularization import l2, regularize_network_params


'''
Unet implementation, based on the architecture proposed in
   U-Net: Convolutional Networks for Biomedical Image Segmentation
   Olaf Ronneberger, Philipp Fischer, Thomas Brox
   Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9351: 234--241, 2015,
   available at arXiv:1505.04597 [cs.CV]
(see http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

The output of the network is a segmentation of the input image. This is a crop of the
original image as no padding is applied (because this may introduce a border effect).
To determine the size of this output image, you can use the function output_size_for_input (default is 388x388 pixels).

The input is an image, the segmentation truth image, and a weight map. This can be used to make certain areas
in the image more important than others when determining the loss.
'''

class UNetClass:
	def __init__(self, input_var, input_size=(256,256),
                        depth=5,
                        branching_factor=6, #2^6 filters for first level, 2^7 for second, etc.
                        num_input_channels=1,
                        num_classes=2,
                        pad='valid'):
		self.input_var = input_var
		self.input_size = input_size
		self.depth = depth
		self.branching_factor = branching_factor
		self.num_input_channels =  num_input_channels
		self.num_classes = num_classes
		self.pad = pad
		
		self.batch_size = None
		nonlinearity = nonlinearities.rectify
		
		self.net = {}
		#Moeten num_input_channels en input size omgedraaid? (zoals in de overview)
		self.net['input'] = InputLayer(shape=(self.batch_size, self.num_input_channels, self.input_size[0],self.input_size[1]), input_var=self.input_var)
		
		def contraction(depth, deepest):
			n_filters = self._num_filters_for_depth(depth, branching_factor)
			incoming = self.net['input'] if depth == 0 else self.net['pool{}'.format(depth-1)]
			
			self.net['conv{}_1'.format(depth)] = Conv2DLayer(incoming, name = 'conv{}_1'.format(depth),
									num_filters=n_filters, filter_size=3, pad=pad,
									W=HeNormal(gain='relu'),
									nonlinearity=nonlinearity)
	
			self.net['conv{}_2'.format(depth)] = Conv2DLayer(self.net['conv{}_1'.format(depth)], name = 'conv{}_2'.format(depth),
									num_filters=n_filters, filter_size=3, pad=pad,
									W=HeNormal(gain='relu'),
									nonlinearity=nonlinearity)
	
			if not deepest:
				self.net['pool{}'.format(depth)] = MaxPool2DLayer(self.net['conv{}_2'.format(depth)], name = 'pool{}'.format(depth), pool_size=2, stride=2)
			
		def expansion(depth, deepest):
			n_filters = self._num_filters_for_depth(depth, branching_factor)

			incoming = self.net['conv{}_2'.format(depth+1)] if deepest else self.net['_conv{}_2'.format(depth+1)]

			upscaling = Upscale2DLayer(incoming, 4)
			self.net['upconv{}'.format(depth)] = Conv2DLayer(upscaling, name = 'upconv{}_1'.format(depth),
									num_filters=n_filters, filter_size=2, stride=2,
	                                        W=HeNormal(gain='relu'),
	                                        nonlinearity=nonlinearity)

			self.net['bridge{}'.format(depth)] = ConcatLayer([
	                                        self.net['upconv{}'.format(depth)],
									self.net['conv{}_2'.format(depth)]], name = 'bridge{}'.format(depth),
									axis=1, cropping=[None, None, 'center', 'center'])

			self.net['_conv{}_1'.format(depth)] = Conv2DLayer(self.net['bridge{}'.format(depth)], name = '_conv{}_1'.format(depth),
									num_filters=n_filters, filter_size=3, pad=pad,
	                                        W=HeNormal(gain='relu'),
	                                        nonlinearity=nonlinearity)

			self.net['_conv{}_2'.format(depth)] = Conv2DLayer(self.net['_conv{}_1'.format(depth)], name = '_conv{}_2'.format(depth),
									num_filters=n_filters, filter_size=3, pad=pad,
	                                        W=HeNormal(gain='relu'),
	                                        nonlinearity=nonlinearity)
		
		# Contraction
		for d in range(self.depth):
			#There is no pooling at the last layer
			deepest = d == self.depth-1
			contraction(d, deepest)

		# Expansion
		for d in reversed(range(self.depth-1)):
			deepest = d == self.depth-2
			expansion(d, deepest)

		# Output layer

		# Germonda: The following line gives an error in Lasagne, changed net['out'] to net, based on
		# http://stackoverflow.com/questions/36185639/load-a-pretrained-caffe-model-to-lasagne
		# net['out'] = Conv2DLayer(net['_conv0_2'], num_filters=num_classes, filter_size=(1,1), pad='valid',
		#                                nonlinearity=None)
		self.net = Conv2DLayer(self.net['_conv0_2'], num_filters=num_classes, filter_size=(1,1), pad=pad,
                                    nonlinearity=None)

		print ('Network output shape {}'.format(str(lasagne.layers.get_output_shape(self.net))))
		
		
	def _num_filters_for_depth(self,depth, branching_factor):
		return 2**(branching_factor+depth)
		
	def output_size_for_input(self,in_size, depth):
	    in_size = np.array(in_size)
	    in_size -= 4
	    for _ in range(depth-1):
	        in_size = in_size//2
	        in_size -= 4
	    for _ in range(depth-1):
	        in_size = in_size*2
	        in_size -= 4
	    return in_size
		
	def define_updates(self, input_var, target_var, weight_var, learning_rate=0.01, momentum=0.9, l2_lambda=1e-5):
	    params = lasagne.layers.get_all_params(self.net, trainable=True)
	
	    out = lasagne.layers.get_output(self.net)
	    test_out = lasagne.layers.get_output(self.net, deterministic=True)
	
	    l2_loss = l2_lambda * regularize_network_params(self.net, l2)
	
	    train_metrics = self._score_metrics(out, target_var, weight_var, l2_loss)
	    loss, acc, target_prediction, prediction = train_metrics
	
	    val_metrics = self._score_metrics(test_out, target_var, weight_var, l2_loss)
	    t_loss, t_acc, t_target_prediction, t_prediction = val_metrics
	
	    updates = lasagne.updates.nesterov_momentum(
	            loss, params, learning_rate=learning_rate, momentum=momentum)
	
	    train_fn = theano.function([input_var, target_var, weight_var], [
	        loss, l2_loss, acc, target_prediction, prediction],
	                               updates=updates)
	    val_fn = theano.function([input_var, target_var, weight_var], [
	        t_loss, l2_loss, t_acc, t_target_prediction, t_prediction])
	
	    return train_fn, val_fn
					
	def define_predict(self, input_var):
	    #params = lasagne.layers.get_all_params(network, trainable=True)
	    out = lasagne.layers.get_output(self.net, deterministic=True)
	    out_flat = out.dimshuffle(1,0,2,3).flatten(ndim=2).dimshuffle(1,0)
	    prediction = lasagne.nonlinearities.softmax(out_flat)
	
	    predict_fn = theano.function([input_var],[prediction])
	    print("Defining predict")
	
	    return predict_fn
					
	def _score_metrics(self, out, target_var, weight_map, l2_loss=0):
	    _EPSILON=1e-8
	
	    out_flat = out.dimshuffle(1,0,2,3).flatten(ndim=2).dimshuffle(1,0)
	    target_flat = target_var.dimshuffle(1,0,2,3).flatten(ndim=1)
	    weight_flat = weight_map.dimshuffle(1,0,2,3).flatten(ndim=1)
	
	    # Softmax output, original paper may have used a sigmoid output_size_for_input
	    # but here we opt for softmax, as this also works for multiclass segmentation.
	    prediction = lasagne.nonlinearities.softmax(out_flat)
	
	    loss = lasagne.objectives.categorical_crossentropy(T.clip(prediction,_EPSILON,1-_EPSILON), target_flat)
	    loss = loss * weight_flat
	    loss = loss.mean()
	    loss += l2_loss
	
	    # Pixelwise accuracy
	    accuracy = T.mean(T.eq(T.argmax(prediction, axis=1), target_flat), dtype=theano.config.floatX)
	
	    return loss, accuracy, target_flat, prediction