

import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L

'''
 Build the fully convolutional CNN

We start with defining the network architecture in Lasagne.
The network we are using is a socalled **fully convolutional network**, a network that does not contain any fully connected layers. This type of network is well suited for image segmentation as a very efficient implementation can be used.

A direct, but naive approach to pixel classification using a CNN would be to extract a patch around every pixel and assign the output back to the pixel location. This naive approach is computationally inefficient, because when we move our 31x31 patch by
one pixel to classify the next pixel, 30x31 pixels of the new patch are identical to the previous patch, and we would apply the same convolutions many times over.


As the network we use here only contains convolutional filters, i.e. the spatial structure is kept intact, the whole network can be thought of as a single large convolutional filter that can be applied to the whole image at once. This speeds up the classification greatly and allows full image segmentation in seconds!

What is important for fully convolutional networks is that the size of the feature maps within the network goes down to exactly 1 for the final feature map. This requires some computation, but luckily it is pretty straightforward.

Assume:

    - i = Input featuremap size
    - f = Convolution filter size (uneven)
    - max pooling size = 2


Then:
Output size of a feature map  after convolution with a convolution filter of size f:

     output_size = input_size -(f-1)

Output size of a feature map  after max pooling by a factor 2:

    output_size = floor(input_size/2)


Let's define a baseline model:
* input layer (given)
* 32 filters of 3x3
* 32 filters of 3x3
* pooling
* 64 filters of 3x3
* 64 filters of 3x3
* pooling
* 128 filters of ?x?
* 64 filters of 1x1
* 2 filters of 1x1 (given)
'''

# Define your network builder function.

def build_network(input_tensor):
    # define the inputlayer
    inputlayer = L.InputLayer(shape=(None, 1, None, None), input_var=input_tensor)

    # Hidden layers and the output layer.
    #
    # 31 -> 29
    conv1 = lasagne.layers.Conv2DLayer(inputlayer,
                                       num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify,
                                       W=lasagne.init.GlorotUniform())
    # 29 -> 27
    conv2 = lasagne.layers.Conv2DLayer(conv1,
                                       num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify,
                                       W=lasagne.init.GlorotUniform())
    # 27 -> 13
    maxpool1 = lasagne.layers.MaxPool2DLayer(conv2, pool_size=(2, 2))
    # 13 -> 11
    conv3 = lasagne.layers.Conv2DLayer(maxpool1,
                                       num_filters=64, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    # 11 -> 9
    conv4 = lasagne.layers.Conv2DLayer(conv3,
                                       num_filters=64, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    # 9 -> 4
    maxpool2 = lasagne.layers.MaxPool2DLayer(conv4, pool_size=(2, 2))
    # 4 -> 1
    conv5 = lasagne.layers.Conv2DLayer(maxpool2,
                                       num_filters=128, filter_size=(4, 4), nonlinearity=lasagne.nonlinearities.rectify)
    # maxpool3 = lasagne.layers.MaxPool2DLayer(conv5, pool_size=(2, 2))
    conv6 = lasagne.layers.Conv2DLayer(conv5,
                                       num_filters=64, filter_size=(1, 1), nonlinearity=lasagne.nonlinearities.rectify)

    # 6-> 6 layers
    # define the final layer.
    # because we are now creating a fully convolutional layer, the last layer does NOT contain a softmax function.
    # the softmax function will be implemented later in a different way allowing a pixelwise loss
    final_layer = L.Conv2DLayer(conv6, 2, (1, 1),
                                nonlinearity=lasagne.nonlinearities.identity,
                                W=lasagne.init.GlorotUniform())
    return final_layer


def softmax(network):
    output = lasagne.layers.get_output(network)
    exp = T.exp(output - output.max(axis=1, keepdims=True))  # subtract max for numeric stability (overflow)
    return exp / exp.sum(axis=1, keepdims=True)


def softmax_deterministic(network):
    output = lasagne.layers.get_output(network, deterministic=True)
    exp = T.exp(output - output.max(axis=1, keepdims=True))  # subtract max for numeric stability (overflow)
    return exp / exp.sum(axis=1, keepdims=True)


# ## Define the training, validation and evaluation function

def training_function(network, input_tensor, target_tensor, learning_rate, use_l2_regularization=False,
                      l2_lambda=0.000001):
    # Get the network output and calculate metrics.
    network_output = softmax(network)

    if use_l2_regularization:
        l2_loss = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
        loss = lasagne.objectives.categorical_crossentropy(network_output, target_tensor).mean() + l2_lambda * l2_loss
    else:
        loss = lasagne.objectives.categorical_crossentropy(network_output, target_tensor).mean()
    accuracy = T.mean(T.eq(T.argmax(network_output, axis=1), T.argmax(target_tensor, axis=1)),
                      dtype=theano.config.floatX)
    # Get the network parameters and the update function.
    network_params = L.get_all_params(network, trainable=True)
    weight_updates = lasagne.updates.sgd(loss, network_params, learning_rate=learning_rate)
    # Construct the training function.
    return theano.function([input_tensor, target_tensor], [loss, accuracy], updates=weight_updates)

def validate_function(network, input_tensor, target_tensor):
    # Get the network output and calculate metrics.
    network_output = softmax_deterministic(network)
    loss = lasagne.objectives.categorical_crossentropy(network_output, target_tensor).mean()
    accuracy = T.mean(T.eq(T.argmax(network_output, axis=1), T.argmax(target_tensor, axis=1)),
                      dtype=theano.config.floatX)

    # Construct the validation function.
    return theano.function([input_tensor, target_tensor], [loss, accuracy])

def evaluate_function(network, input_tensor):
    # Get the network output and calculate metrics.
    network_output = softmax_deterministic(network)

    # Construct the evaluation function.
    return theano.function([input_tensor], network_output)
