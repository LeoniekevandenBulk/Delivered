
import os
import numpy as np
np.set_printoptions(precision=2, suppress=True)
import nibabel as nib

import os
import numpy as np
from PIL import Image
from math import cos, sin, radians, floor, ceil

import theano.tensor as T
import lasagne
import lasagne.layers as L

import random

SURFsara = False

import matplotlib
if SURFsara:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = (40, 24)
matplotlib.rcParams['xtick.labelsize'] = 30

from UNetClass import UNetClass
from Trainer import Trainer
from BatchGenerator import BatchGenerator
from BatchAugmenter import BatchAugmenter
from Evaluator import Evaluator
from tools import get_file_list
from tools import output_size_for_input
from tools import show_slices
from tools import show_volumes
from tools import show_preprocessing
from tools import show_segmentation_prediction

import random

random.seed(0)

'''
Data loading
'''

train_batch_dir='../data/Training_Batch'

vol_batch = sorted(get_file_list(train_batch_dir, 'volume')[0])
seg_batch = sorted(get_file_list(train_batch_dir, 'segmentation')[0])

show_vol = False
if show_vol:
        show_volumes(vol_batch, seg_batch)

'''
Split the 3D volumes in a training and validation set.
'''

nr_volumes = len(vol_batch)
vol_list = range(nr_volumes)
random.shuffle(vol_list)

validation_percentage = 0.3
nr_val_volumes = int(ceil(len(vol_batch)*validation_percentage))
nr_tra_volumes = nr_volumes - nr_val_volumes

# use the first images as validation
tra_list = vol_list[0:nr_tra_volumes]
val_list = vol_list[nr_tra_volumes:]

print("nr of training 3D volumes: " + str(len(tra_list)) + "\nnr of validation 3D volumes: " + str(len(val_list)))

'''
Set parameters
Set the parameters to suitable values.
Certain paramters have a large influence on the performance of your network, try to find the optimal parameters for
this problem!
'''

# UNet architecture
depth=5
branching_factor = 6

# Image dimensions
patch_size = (650,650)
out_size = output_size_for_input(patch_size, depth)
img_center = [256, 256]

# Training
learning_rate = 0.1
nr_epochs = 1
nr_train_batches = 3
nr_val_batches = 3
batch_size = 5
max_rotation = 10
gaussian_blur = False
elastic_deformation = True
class_balancing = True
nr_validation_samples_per_3D_volume = 1
validation_batch_size = 1
liver_network_name = 'liver_network_LiTS'
lesion_network_name = 'lesion_network_LiTS'

'''
Creating the batch generators and testing the preprocessing
'''

# Implement the get_random_sample_from_class() function in the class above. You can use the functions below to test if the output makes sense.

# Testing the preprocessing of the input image (patch_size) and output labels (out_size)
show_preprocess = True
if show_preprocess:
    nr_tests = 5 # 5 slices per test
    for i in range(nr_tests):
        show_preprocessing(nr_tra_volumes, train_batch_dir, trainBatchGenerator, patch_size, out_size, img_center)

'''
Create UNet networks
'''

ps_x = patch_size[0]
ps_y = patch_size[1]

# ## Compile the theano functions
inputs = T.ftensor4('X') #dtensor
targets = T.itensor4('Y') 
weights = T.ftensor4('W') #dtensor

# Unet defined with pad='same' so input_size=output_size
# The original Unet is defined without padding to not create border effects in case of tiling a large image.
# Our images are not that large, and taken as a whole, so border effect are no concern here.

print("Liver network:")
liverNetwork = UNetClass(inputs, 
					    input_size=patch_size,
					    depth=depth,
					    branching_factor=branching_factor, # 2^6 filters for first level, 2^7 for second, etc.
					    num_input_channels=1,
					    num_classes=2,
					    pad='valid')

#outputs van liverNetwork worden inputs van lesionNetwork
print("Lesion network:")
lesionNetwork = UNetClass(inputs,
					    input_size=patch_size,
                        depth=depth,
                        branching_factor=branching_factor,  # 2^6 filters for first level, 2^7 for second, etc.
                        num_input_channels=1,
                        num_classes=2,
                        pad='valid')

print("Creating Theano training and validation functions for liver network ...")
liverNetwork.train_fn, liverNetwork.val_fn = liverNetwork.define_updates(inputs, targets, weights)
liverNetwork.predict_fn = liverNetwork.define_predict(inputs)
print("Creating Theano training and validation functions for lesion network ...")
lesionNetwork.train_fn, lesionNetwork.val_fn = lesionNetwork.define_updates(inputs, targets, weights)
lesionNetwork.predict_fn = lesionNetwork.define_predict(inputs)

'''
Training
'''

# plot learning curves
fig = plt.figure(figsize=(30, 15))
plt.xlabel('epoch', size=40)
plt.ylabel('loss', size=40)
fig.labelsize=40

liverTrainer = Trainer(liverNetwork, "liverNet", tra_list, val_list, learning_rate=learning_rate, patch_size=patch_size)
lesionTrainer = Trainer(lesionNetwork, "lesionNet", tra_list, val_list, learning_rate=learning_rate, patch_size=patch_size)

# Creation of generators
trainBatchGenerator = BatchGenerator(augment=True) # (with augmentation)
valBatchGenerator = BatchGenerator(augment=False) # (without augmentation)
evaluator = Evaluator()

# Main training loop
tra_dice_lst = []
val_dice_lst = []
tra_ce_lst = []
val_ce_lst = []
best_val_dice = 0
best_val_threshold = 0
for epoch in range(nr_epochs):
    print('Epoch {}/{}'.format(epoch + 1,nr_epochs))

    tra_dices = []
    tra_ces = []
    val_dices= []
    val_ces= []
    val_thres = []
    for batch in range(nr_train_batches):
        print('Batch {}/{}'.format(batch + 1, nr_batches))
        #Training batch generation
        X_tra, Y_tra = trainBatchGenerator.get_batch(tra_list, train_batch_dir, batch_size,
                                     patch_size, out_size, img_center, group_labels="lesion", group_percentages=(0.5,0.5))

        #Data augmentation
        #myAugmenter = BatchAugmenter(X_tra, Y_tra, [[0.1,0.8,0.9],[0.1,0.8,0.6]])
        #X_tra, Y_tra = myAugmenter.getAugmentation()

        #Training
        prediction = lesionTrainer.train_batch(X_tra, Y_tra, verbose=True)
        tra_error_report = evaluator.get_evaluation(prediction, Y_tra)
        dice = tra_error_report[0][1]
        cross_entropy = tra_error_report[1][1]
 
        print 'training dice, cross entropy', dice, cross_entropy
        tra_dices.append(dice)
        tra_ces.append(cross_entropy)

    for batch in range(nr_val_batches):
        # Validation
        X_val, Y_val = valBatchGenerator.get_batch(val_list, train_batch_dir, batch_size,
                                     patch_size, out_size, img_center, group_labels="lesion", group_percentages=(0.5,0.5))

        prediction = lesionTrainer.predict_batch(X_val, Y_val, verbose=True)
        val_error_report = evaluator.get_evaluation(prediction, Y_val)
        dice = val_error_report[0][1]
        threshold = val_error_report[0][2]
        cross_entropy = val_error_report[1][1]
        
        print 'validation dice, cross entropy', val_error_report[0][1], val_error_report[1][1]
        val_dices.append(dice)
        val_ces.append(cross_entropy)
        val_thres.append(threshold)

    tra_dice_lst.append(np.mean(tra_dices))
    tra_ce_lst.append(np.mean(tra_ces))
    val_dice_lst.append(np.mean(val_dices))
    val_ce_lst.append(np.mean(val_ces))

    if np.mean(val_dices) > best_val_dice:
        best_val_dice = np.mean(val_dices)
        best_val_threshold = np.mean(val_thres)
        # save networks
        params = L.get_all_param_values(liverNetwork.net)
        np.savez(os.path.join('./', liver_network_name + '.npz'), params=params)
        params = L.get_all_param_values(lesionNetwork.net)
        np.savez(os.path.join('./', lesion_network_name + '.npz'), params=params)

    # plot learning curves
    fig = plt.figure(figsize=(10, 5))
    tra_dice_plt, = plt.plot(range(len(tra_dice_lst)), tra_dice_lst, 'b')
    val_dice_plt, = plt.plot(range(len(val_dice_lst)), val_dice_lst, 'g')
    tra_ce_plt, = plt.plot(range(len(tra_ce_lst)), tra_ce_lst, 'm')
    val_ce_plt, = plt.plot(range(len(val_ce_lst)), val_ce_lst, 'r')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend([tra_dice_plt, val_dice_plt, tra_ce_plt, val_ce_plt],
               ['training dice', 'validation dice', 'training cross_entropy', 'validation cross_entropy'],
               loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Best validation dice = {:.2f}% with threshold {:d}'.format(best_val_dice, best_val_threshold), size=40)
    plt.show(block=False)
    plt.pause(.5)

plt.legend([tra_dice_plt, val_dice_plt, tra_ce_plt, val_ce_plt],
               ['training dice', 'validation dice', 'training cross_entropy', 'validation cross_entropy'],
               loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('Dice_and_cross_entropy.png')

'''
Visualize some validation segmentations
'''

read_network_parameters = False
if read_network_parameters:
    liver_network_name = 'liver_network_test_LiTS'
    lesion_network_name = 'lesion_network_test_LiTS'
    # initialize the networks used in this experiment (this may change)
    liverNetwork = UNetClass(inputs, input_size=patch_size,
                             depth=5,
                             branching_factor=3,  # 2^6 filters for first level, 2^7 for second, etc.
                             num_input_channels=1,
                             num_classes=2,
                             pad='valid')
    npz = np.load('./' + liver_network_name + '.npz')  # load stored parameters
    lasagne.layers.set_all_param_values(liverNetwork.net, npz['params'])  # set parameters
    lesionNetwork = UNetClass(inputs, input_size=patch_size,
                              depth=5,
                              branching_factor=3,  # 2^6 filters for first level, 2^7 for second, etc.
                              num_input_channels=1,
                              num_classes=2,
                              pad='valid')
    npz = np.load('./' + lesion_network_name + '.npz')  # load stored parameters
    lasagne.layers.set_all_param_values(lesionNetwork.net, npz['params'])  # set parameters

    print("Creating Theano training and validation functions for liver network ...")
    liverNetwork.train_fn, liverNetwork.val_fn = liverNetwork.define_updates(inputs, targets, weights)
    print("Creating Theano training and validation functions for lesion network ...")
    lesionNetwork.train_fn, lesionNetwork.val_fn = lesionNetwork.define_updates(inputs, targets, weights)

show_segmentations = True
if show_segmentations:
    show_segmentation_prediction(valBatchGenerator, lesionNetwork, val_list, train_batch_dir,
                                 patch_size, out_size, img_center)

'''
Evaluate the performance on the test set and submit to Challenger

We can now apply the trained fully convolutional network on our independent test set.
'''

# Load the images:

# test images
test_dir='../data/Test-Data'

vol_test = sorted(get_file_list(test_dir, 'test-volume')[0])

result_output_folder = os.path.join(test_dir, 'results')
if not (os.path.exists(result_output_folder)):
    os.mkdir(result_output_folder)

# ## Apply the trained network to the images

read_network_parameters = False
if read_network_parameters:
    liver_network_name = 'liver_network_LiTS'
    lesion_network_name = 'lesion_network_LiTS'
    # initialize the networks used in this experiment (this may change)
    liverNetwork = UNetClass(inputs, input_size=patch_size,
                             depth=5,
                             branching_factor=branching_factor,  # 2^6 filters for first level, 2^7 for second, etc.
                             num_input_channels=1,
                             num_classes=2,
                             pad='same')
    npz = np.load('./' + network_name + '.npz')  # load stored parameters
    lasagne.layers.set_all_param_values(liverNetwork.net, npz['params'])  # set parameters
    lesionNetwork = UNetClass(inputs, input_size=patch_size,
                              depth=5,
                              branching_factor=branching_factor,  # 2^6 filters for first level, 2^7 for second, etc.
                              num_input_channels=1,
                              num_classes=2,
                              pad='same')
    npz = np.load('./' + network_name + '.npz')  # load stored parameters
    lasagne.layers.set_all_param_values(lesionNetwork.net, npz['params'])  # set parameters

print("Creating Theano predict function ...")
lesionNetwork.predict_fn = lesionNetwork.define_predict(inputs)

# one forward pass on the test set
threshold = 0.5
print ('predicting...')
for i, vol in enumerate(vol_test):
    # load 3D volumes
    vol_array = nib.load(vol).get_data()
    vol_array=vol_array[:,:,vol_array.shape[2]/2:vol_array.shape[2]/2+1] # testing middle slice
    # organize test slices one slice per batch
    X = np.zeros((1, 1, vol_array.shape[0], vol_array.shape[1]))

    # get every slice
    thresholded_image = np.zeros(vol_array.shape)
    for slice in range(vol_array.shape[2]):
        print ('predicting slice '+str(slice)+'/'+str(vol_array.shape[2]-1)+', test volume '+vol)
        X[0, 0, :, :] = vol_array[:ps_y, :ps_x, slice]

        # predict softmax proabbilities
        probability = lesionNetwork.predict_fn(X.astype(np.float32))

        #probability[0] is for no label, probability[1] is for label
        probability_labeled = np.asarray(probability)[:,:,1].reshape(patch_size)

        # binary version by thresholding
        thresholded_image[:, :, slice] = probability_labeled > threshold

    # save output to file
    np.save(vol.replace('volume', 'lesions'), thresholded_image)







